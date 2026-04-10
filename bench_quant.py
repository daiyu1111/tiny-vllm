import argparse
import gc
import os
import time
from random import randint, seed

import torch

from nanovllm import LLM, SamplingParams


DEFAULT_MODEL = "/mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B"
DEFAULT_INT8_MODEL = "/mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B-int8"


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Nano-vLLM bf16/fp16 vs INT8 weight-only models.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model directory used for config/tokenizer.")
    parser.add_argument("--int8-model", default=DEFAULT_INT8_MODEL, help="INT8 quantized model directory.")
    parser.add_argument("--mode", choices=["bf16", "int8", "both"], default="both")
    parser.add_argument("--num-seqs", type=int, default=256)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
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


def run_one(label: str, args, prompt_token_ids, sampling_params):
    kwargs = dict(
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    if label == "int8":
        kwargs.update(
            quantization="int8",
            quantized_model_path=os.path.expanduser(args.int8_model),
        )

    llm = LLM(os.path.expanduser(args.model), **kwargs)
    print_memory(label, llm.model_runner.memory_snapshots["after_model_load"])

    llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=False)
    start = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    elapsed = time.time() - start
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / elapsed
    print(f"{label} total: {total_tokens}tok, time={elapsed:.2f}s, throughput={throughput:.2f}tok/s")

    llm.exit()
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return throughput


def main():
    args = parse_args()
    prompt_token_ids, sampling_params = make_workload(args)

    results = {}
    if args.mode in ("bf16", "both"):
        results["bf16"] = run_one("bf16", args, prompt_token_ids, sampling_params)
    if args.mode in ("int8", "both"):
        results["int8"] = run_one("int8", args, prompt_token_ids, sampling_params)

    if "bf16" in results and "int8" in results:
        ratio = results["int8"] / results["bf16"] if results["bf16"] > 0 else 0.0
        print(f"int8/bf16 throughput ratio: {ratio:.3f}x")


if __name__ == "__main__":
    main()
