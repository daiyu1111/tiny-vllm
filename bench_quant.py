import argparse
import gc
import os
from random import randint, seed

import torch

from nanovllm import LLM, SamplingParams


DEFAULT_MODEL = "/mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B"
DEFAULT_INT8_MODEL = "/mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B-int8"
DEFAULT_W8A8_MODEL = "/mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B-w8a8"


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Nano-vLLM bf16/fp16 vs INT8/W8A8 quantized models.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model directory used for config/tokenizer.")
    parser.add_argument("--int8-model", default=DEFAULT_INT8_MODEL, help="INT8 quantized model directory.")
    parser.add_argument("--w8a8-model", default=DEFAULT_W8A8_MODEL, help="W8A8 quantized model directory.")
    parser.add_argument("--mode", choices=["bf16", "int8", "w8a8", "both", "all"], default="all")
    parser.add_argument("--num-seqs", type=int, default=256)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--int8-backend",
        choices=["native", "cutlass", "fallback", "auto"],
        default="native",
        help="INT8 backend for the primary INT8 benchmark run.",
    )
    parser.add_argument(
        "--compare-cutlass",
        action="store_true",
        help="Also benchmark the CUTLASS backend when the primary INT8 backend is not cutlass.",
    )
    parser.add_argument(
        "--compare-native",
        action="store_true",
        help="Also benchmark the native backend when the primary INT8 backend is not native.",
    )
    return parser.parse_args()


def bytes_to_gib(value: int) -> float:
    return value / 1024 ** 3


def print_memory(label: str, snapshot: dict):
    allocated = bytes_to_gib(snapshot["memory_allocated"])
    reserved = bytes_to_gib(snapshot["memory_reserved"])
    max_allocated = bytes_to_gib(snapshot["max_memory_allocated"])
    print(
        f"{label} memory after load: "
        f"allocated={allocated:.2f}GiB, "
        f"reserved={reserved:.2f}GiB, "
        f"max_allocated={max_allocated:.2f}GiB"
    )


def print_run_memory(label: str):
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    allocated = bytes_to_gib(torch.cuda.memory_allocated())
    reserved = bytes_to_gib(torch.cuda.memory_reserved())
    peak_allocated = bytes_to_gib(torch.cuda.max_memory_allocated())
    print(
        f"{label} runtime memory: "
        f"allocated={allocated:.2f}GiB, "
        f"reserved={reserved:.2f}GiB, "
        f"peak_allocated={peak_allocated:.2f}GiB"
    )


def format_ttft(value: float | None) -> str:
    return f"{value:.2f}s" if value is not None else "n/a"


def make_workload(args):
    seed(args.seed)
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, args.max_input_len))]
        for _ in range(args.num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, args.max_output_len))
        for _ in range(args.num_seqs)
    ]
    return prompt_token_ids, sampling_params


def set_int8_env(backend: str) -> dict[str, str | None]:
    previous = {
        "NANOVLLM_INT8_BACKEND": os.environ.get("NANOVLLM_INT8_BACKEND"),
    }
    os.environ["NANOVLLM_INT8_BACKEND"] = backend
    return previous


def restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def run_one(label: str, args, prompt_token_ids, sampling_params, backend: str | None = None):
    kwargs = dict(
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    env_snapshot = None
    llm = None
    if label == "int8":
        kwargs.update(
            quantization="int8",
            quantized_model_path=os.path.expanduser(args.int8_model),
        )
        backend = backend or args.int8_backend
        env_snapshot = set_int8_env(backend)
    elif label == "w8a8":
        kwargs.update(
            quantization="w8a8",
            quantized_model_path=os.path.expanduser(args.w8a8_model),
        )
        backend = "native"
        env_snapshot = {
            "NANOVLLM_W8A8_BACKEND": os.environ.get("NANOVLLM_W8A8_BACKEND"),
        }
        os.environ["NANOVLLM_W8A8_BACKEND"] = backend

    try:
        llm = LLM(os.path.expanduser(args.model), **kwargs)
        run_label = label if label not in ("int8", "w8a8") else f"{label}[backend={backend}]"
        if label in ("int8", "w8a8"):
            print(f"{run_label} config")
        print_memory(run_label, llm.model_runner.memory_snapshots["after_model_load"])

        llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=False)
        torch.cuda.reset_peak_memory_stats()
        outputs, stats = llm.generate_with_stats(prompt_token_ids, sampling_params, use_tqdm=False)
        total_tokens = sum(len(output["token_ids"]) for output in outputs)
        throughput = total_tokens / stats["wall_time_seconds"] if stats["wall_time_seconds"] > 0 else 0.0
        print(
            f"{run_label} total: {total_tokens}tok, "
            f"time={stats['wall_time_seconds']:.2f}s, "
            f"throughput={throughput:.2f}tok/s"
        )
        print(
            f"{run_label} split: "
            f"prefill={stats['prefill_tokens_per_second']:.2f}tok/s "
            f"(tokens={stats['prefill_tokens']}, time={stats['prefill_time_seconds']:.2f}s, steps={stats['prefill_steps']}), "
            f"decode={stats['decode_tokens_per_second']:.2f}tok/s "
            f"(tokens={stats['decode_tokens']}, time={stats['decode_time_seconds']:.2f}s, steps={stats['decode_steps']}), "
            f"ttft~={format_ttft(stats['ttft_seconds_approx'])}"
        )
        print_run_memory(run_label)

        return {
            "throughput": throughput,
            "prefill_tps": stats["prefill_tokens_per_second"],
            "decode_tps": stats["decode_tokens_per_second"],
            "backend": backend,
            "run_label": run_label,
        }
    finally:
        if llm is not None:
            llm.exit()
            del llm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if env_snapshot is not None:
            restore_env(env_snapshot)


def print_ratio(name: str, numerator: dict, denominator: dict):
    total_ratio = numerator["throughput"] / denominator["throughput"] if denominator["throughput"] > 0 else 0.0
    prefill_ratio = numerator["prefill_tps"] / denominator["prefill_tps"] if denominator["prefill_tps"] > 0 else 0.0
    decode_ratio = numerator["decode_tps"] / denominator["decode_tps"] if denominator["decode_tps"] > 0 else 0.0
    print(
        f"{name}: "
        f"total={total_ratio:.3f}x "
        f"prefill={prefill_ratio:.3f}x "
        f"decode={decode_ratio:.3f}x"
    )


def main():
    args = parse_args()
    prompt_token_ids, sampling_params = make_workload(args)

    results = {}
    if args.mode in ("bf16", "both", "all"):
        results["bf16"] = run_one("bf16", args, prompt_token_ids, sampling_params)
    if args.mode in ("int8", "both", "all"):
        results["int8"] = run_one("int8", args, prompt_token_ids, sampling_params)
        if args.compare_cutlass and args.int8_backend != "cutlass":
            results["int8_cutlass"] = run_one("int8", args, prompt_token_ids, sampling_params, backend="cutlass")
        if args.compare_native and args.int8_backend != "native":
            results["int8_native"] = run_one("int8", args, prompt_token_ids, sampling_params, backend="native")
    if args.mode in ("w8a8", "all"):
        results["w8a8"] = run_one("w8a8", args, prompt_token_ids, sampling_params)

    if "bf16" in results and "int8" in results:
        print_ratio("int8/bf16 ratios", results["int8"], results["bf16"])
    if "bf16" in results and "w8a8" in results:
        print_ratio("w8a8/bf16 ratios", results["w8a8"], results["bf16"])
    if "int8" in results and "w8a8" in results:
        print_ratio("w8a8/int8 ratios", results["w8a8"], results["int8"])
    if "bf16" in results and "int8_cutlass" in results:
        print_ratio("int8_cutlass/bf16 ratios", results["int8_cutlass"], results["bf16"])
    if "bf16" in results and "int8_native" in results:
        print_ratio("int8_native/bf16 ratios", results["int8_native"], results["bf16"])


if __name__ == "__main__":
    main()
