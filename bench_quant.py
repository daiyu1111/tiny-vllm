import argparse
import gc
import json
import os
from datetime import datetime
from random import randint, seed

DEFAULT_MODEL = "/mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B"
DEFAULT_INT8_MODEL = "/mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B-int8"
DEFAULT_W8A8_MODEL = "/mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B-w8a8"
DEFAULT_OUTPUT = "bench_quant_results.json"
PROFILE_DEFAULTS = {
    "prefill_heavy": {"input_len": 2048, "output_len": 64},
    "decode_heavy": {"input_len": 64, "output_len": 512},
}


def parse_int_list(value: str | None) -> list[int]:
    if value is None:
        return []
    items = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        parsed = int(part)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("Sweep values must be positive integers.")
        items.append(parsed)
    if not items:
        raise argparse.ArgumentTypeError("Sweep list cannot be empty.")
    return items


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Nano-vLLM bf16/fp16 vs INT8/W8A8 quantized models.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model directory used for config/tokenizer.")
    parser.add_argument("--int8-model", default=DEFAULT_INT8_MODEL, help="INT8 quantized model directory.")
    parser.add_argument("--w8a8-model", default=DEFAULT_W8A8_MODEL, help="W8A8 quantized model directory.")
    parser.add_argument("--mode", choices=["bf16", "int8", "w8a8", "both", "all"], default="all")
    parser.add_argument("--num-seqs", type=int, default=256)
    parser.add_argument(
        "--profile",
        choices=["mixed", "prefill_heavy", "decode_heavy"],
        default="mixed",
        help="Workload profile used to generate benchmark requests.",
    )
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("--fixed-input-len", type=int, help="Fixed prompt length for non-mixed profiles.")
    parser.add_argument("--fixed-output-len", type=int, help="Fixed output length for non-mixed profiles.")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=512, help="Scheduler max concurrent sequences.")
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=16384,
        help="Scheduler max total tokens per prefill batch.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of total GPU memory reserved for model + KV cache.",
    )
    parser.add_argument(
        "--sweep-max-num-seqs",
        type=parse_int_list,
        help="Comma-separated max_num_seqs sweep values, e.g. 64,128,256,512.",
    )
    parser.add_argument(
        "--sweep-max-num-batched-tokens",
        type=parse_int_list,
        help="Comma-separated max_num_batched_tokens sweep values, e.g. 4096,8192,16384.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="JSON report path.")
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
    args = parser.parse_args()
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in the range (0, 1].")
    if args.max_num_batched_tokens < args.max_model_len:
        parser.error("--max-num-batched-tokens must be >= --max-model-len.")
    if args.fixed_input_len is not None and args.fixed_input_len <= 0:
        parser.error("--fixed-input-len must be positive.")
    if args.fixed_output_len is not None and args.fixed_output_len <= 0:
        parser.error("--fixed-output-len must be positive.")
    return args


def bytes_to_gib(value: int | None) -> float | None:
    if value is None:
        return None
    return value / 1024 ** 3


def snapshot_to_gib(snapshot: dict | None) -> dict | None:
    if snapshot is None:
        return None
    result = {}
    for key, value in snapshot.items():
        result[key] = bytes_to_gib(value) if "memory" in key else value
    return result


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


def capture_run_memory() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {
            "memory_allocated": 0,
            "memory_reserved": 0,
            "max_memory_allocated": 0,
        }
    torch.cuda.synchronize()
    return {
        "memory_allocated": int(torch.cuda.memory_allocated()),
        "memory_reserved": int(torch.cuda.memory_reserved()),
        "max_memory_allocated": int(torch.cuda.max_memory_allocated()),
    }


def print_run_memory(label: str, snapshot: dict):
    allocated = bytes_to_gib(snapshot["memory_allocated"])
    reserved = bytes_to_gib(snapshot["memory_reserved"])
    peak_allocated = bytes_to_gib(snapshot["max_memory_allocated"])
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
    if args.profile == "mixed":
        prompt_token_ids = [
            [randint(0, 10000) for _ in range(randint(100, args.max_input_len))]
            for _ in range(args.num_seqs)
        ]
        sampling_params = [
            SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, args.max_output_len))
            for _ in range(args.num_seqs)
        ]
        return prompt_token_ids, sampling_params

    defaults = PROFILE_DEFAULTS[args.profile]
    input_len = args.fixed_input_len or defaults["input_len"]
    output_len = args.fixed_output_len or defaults["output_len"]
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(input_len)]
        for _ in range(args.num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len)
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


def build_run_kwargs(args, config_override: dict | None = None) -> dict:
    kwargs = dict(
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    if config_override:
        kwargs.update(config_override)
    return kwargs


def build_run_config(args, config_override: dict | None = None) -> dict:
    config = {
        "num_seqs": args.num_seqs,
        "profile": args.profile,
        "max_input_len": args.max_input_len,
        "max_output_len": args.max_output_len,
        "fixed_input_len": args.fixed_input_len,
        "fixed_output_len": args.fixed_output_len,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "configured_max_num_seqs": args.max_num_seqs,
        "configured_max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "seed": args.seed,
    }
    if config_override:
        config.update(config_override)
    return config


def build_workload_summary(prompt_token_ids, sampling_params) -> dict:
    input_lengths = [len(tokens) for tokens in prompt_token_ids]
    output_lengths = [sp.max_tokens for sp in sampling_params]
    return {
        "num_seqs": len(prompt_token_ids),
        "input_tokens_total": sum(input_lengths),
        "output_tokens_requested_total": sum(output_lengths),
        "input_length_min": min(input_lengths) if input_lengths else 0,
        "input_length_max": max(input_lengths) if input_lengths else 0,
        "input_length_mean": sum(input_lengths) / len(input_lengths) if input_lengths else 0,
        "output_length_min": min(output_lengths) if output_lengths else 0,
        "output_length_max": max(output_lengths) if output_lengths else 0,
        "output_length_mean": sum(output_lengths) / len(output_lengths) if output_lengths else 0,
    }


def mode_specs(args) -> list[dict]:
    specs = []
    if args.mode in ("bf16", "both", "all"):
        specs.append({"label": "bf16", "backend": None})
    if args.mode in ("int8", "both", "all"):
        specs.append({"label": "int8", "backend": args.int8_backend})
        if args.compare_cutlass and args.int8_backend != "cutlass":
            specs.append({"label": "int8", "backend": "cutlass"})
        if args.compare_native and args.int8_backend != "native":
            specs.append({"label": "int8", "backend": "native"})
    if args.mode in ("w8a8", "all"):
        specs.append({"label": "w8a8", "backend": "native"})
    return specs


def ratio(value: float, base: float) -> float | None:
    if base <= 0:
        return None
    return value / base


def build_success_result(
    *,
    label: str,
    backend: str | None,
    run_label: str,
    run_config: dict,
    outputs: list[dict],
    stats: dict,
    load_snapshot: dict,
    runtime_snapshot: dict,
):
    total_tokens = sum(len(output["token_ids"]) for output in outputs)
    throughput = total_tokens / stats["wall_time_seconds"] if stats["wall_time_seconds"] > 0 else 0.0
    return {
        "status": "ok",
        "mode": label,
        "backend": backend,
        "run_label": run_label,
        "config": run_config,
        "total_tokens": total_tokens,
        "throughput": throughput,
        "prefill_tps": stats["prefill_tokens_per_second"],
        "decode_tps": stats["decode_tokens_per_second"],
        "ttft_seconds": stats["ttft_seconds_approx"],
        "stats": stats,
        "memory_after_load_bytes": load_snapshot,
        "memory_after_load_gib": snapshot_to_gib(load_snapshot),
        "runtime_memory_bytes": runtime_snapshot,
        "runtime_memory_gib": snapshot_to_gib(runtime_snapshot),
        "error": None,
    }


def build_error_result(
    *,
    label: str,
    backend: str | None,
    run_label: str,
    run_config: dict,
    error: Exception,
):
    return {
        "status": "error",
        "mode": label,
        "backend": backend,
        "run_label": run_label,
        "config": run_config,
        "total_tokens": 0,
        "throughput": 0.0,
        "prefill_tps": 0.0,
        "decode_tps": 0.0,
        "ttft_seconds": None,
        "stats": None,
        "memory_after_load_bytes": None,
        "memory_after_load_gib": None,
        "runtime_memory_bytes": None,
        "runtime_memory_gib": None,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }


def print_result_line(result: dict, context: str):
    config = result["config"]
    prefix = (
        f"{context} {result['run_label']} "
        f"[max_num_seqs={config['max_num_seqs']}, "
        f"max_num_batched_tokens={config['max_num_batched_tokens']}, "
        f"gpu_mem_util={config['gpu_memory_utilization']:.2f}]"
    )
    if result["status"] != "ok":
        error = result["error"]
        print(f"{prefix} FAILED: {error['type']}: {error['message']}")
        return
    print(
        f"{prefix} "
        f"total={result['throughput']:.2f}tok/s "
        f"prefill={result['prefill_tps']:.2f}tok/s "
        f"decode={result['decode_tps']:.2f}tok/s "
        f"ttft~={format_ttft(result['ttft_seconds'])}"
    )


def run_one(
    label: str,
    args,
    prompt_token_ids,
    sampling_params,
    backend: str | None = None,
    config_override: dict | None = None,
):
    kwargs = build_run_kwargs(args, config_override)
    run_config = build_run_config(args, config_override)
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

    run_label = label if label not in ("int8", "w8a8") else f"{label}[backend={backend}]"
    try:
        llm = LLM(os.path.expanduser(args.model), **kwargs)
        if label in ("int8", "w8a8"):
            print(f"{run_label} config")
        load_snapshot = llm.model_runner.memory_snapshots["after_model_load"]
        print_memory(run_label, load_snapshot)

        llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=False)
        torch.cuda.reset_peak_memory_stats()
        outputs, stats = llm.generate_with_stats(prompt_token_ids, sampling_params, use_tqdm=False)
        runtime_snapshot = capture_run_memory()
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
        print_run_memory(run_label, runtime_snapshot)

        return build_success_result(
            label=label,
            backend=backend,
            run_label=run_label,
            run_config=run_config,
            outputs=outputs,
            stats=stats,
            load_snapshot=load_snapshot,
            runtime_snapshot=runtime_snapshot,
        )
    except Exception as exc:
        return build_error_result(
            label=label,
            backend=backend,
            run_label=run_label,
            run_config=run_config,
            error=exc,
        )
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
    if numerator["status"] != "ok" or denominator["status"] != "ok":
        print(f"{name}: skipped due to failed run")
        return
    total_ratio = ratio(numerator["throughput"], denominator["throughput"]) or 0.0
    prefill_ratio = ratio(numerator["prefill_tps"], denominator["prefill_tps"]) or 0.0
    decode_ratio = ratio(numerator["decode_tps"], denominator["decode_tps"]) or 0.0
    print(
        f"{name}: "
        f"total={total_ratio:.3f}x "
        f"prefill={prefill_ratio:.3f}x "
        f"decode={decode_ratio:.3f}x"
    )


def sweep_values(current: int, values: list[int]) -> list[int]:
    return values if values else [current]


def build_sweep_points(args) -> list[dict]:
    max_num_seqs_values = sweep_values(args.max_num_seqs, args.sweep_max_num_seqs or [])
    max_num_batched_tokens_values = sweep_values(
        args.max_num_batched_tokens,
        args.sweep_max_num_batched_tokens or [],
    )
    points = []
    for max_num_seqs in max_num_seqs_values:
        for max_num_batched_tokens in max_num_batched_tokens_values:
            if max_num_batched_tokens < args.max_model_len:
                raise ValueError("All sweep max_num_batched_tokens values must be >= max_model_len.")
            points.append(
                {
                    "max_num_seqs": max_num_seqs,
                    "max_num_batched_tokens": max_num_batched_tokens,
                }
            )
    return points


def is_sweep_mode(args) -> bool:
    return bool(args.sweep_max_num_seqs or args.sweep_max_num_batched_tokens)


def print_sweep_summary(results: list[dict]):
    print("\nSweep summary by run")
    print("run_label                     status   max_num_seqs  max_batched_tok  total_tps   prefill_tps  decode_tps  ttft")
    for result in results:
        config = result["config"]
        if result["status"] == "ok":
            line = (
                f"{result['run_label']:<28} "
                f"{result['status']:<7} "
                f"{config['max_num_seqs']:<13} "
                f"{config['max_num_batched_tokens']:<16} "
                f"{result['throughput']:<10.2f} "
                f"{result['prefill_tps']:<12.2f} "
                f"{result['decode_tps']:<11.2f} "
                f"{format_ttft(result['ttft_seconds'])}"
            )
        else:
            error = result["error"]["type"]
            line = (
                f"{result['run_label']:<28} "
                f"{result['status']:<7} "
                f"{config['max_num_seqs']:<13} "
                f"{config['max_num_batched_tokens']:<16} "
                f"{'-':<10} {'-':<12} {'-':<11} {error}"
            )
        print(line)


def print_best_by_mode(results: list[dict]):
    print("\nBest successful sweep point by mode")
    grouped = {}
    for result in results:
        if result["status"] != "ok":
            continue
        grouped.setdefault(result["run_label"], []).append(result)
    if not grouped:
        print("No successful sweep runs.")
        return
    print("run_label                     metric        value       max_num_seqs  max_batched_tok")
    for run_label in sorted(grouped):
        candidates = grouped[run_label]
        best_total = max(candidates, key=lambda item: item["throughput"])
        best_prefill = max(candidates, key=lambda item: item["prefill_tps"])
        best_decode = max(candidates, key=lambda item: item["decode_tps"])
        for metric, best in (
            ("total_tps", best_total),
            ("prefill_tps", best_prefill),
            ("decode_tps", best_decode),
        ):
            value = best["throughput"] if metric == "total_tps" else best[metric]
            config = best["config"]
            print(
                f"{run_label:<28} "
                f"{metric:<12} "
                f"{value:<11.2f} "
                f"{config['max_num_seqs']:<13} "
                f"{config['max_num_batched_tokens']}"
            )


def write_json_report(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nWrote JSON results to {path}")


def run_single_mode(args, prompt_token_ids, sampling_params) -> dict:
    results = {}
    for spec in mode_specs(args):
        result = run_one(
            spec["label"],
            args,
            prompt_token_ids,
            sampling_params,
            backend=spec["backend"],
        )
        key = result["run_label"].replace("[backend=", "_").replace("]", "")
        results[key] = result

    if "bf16" in results and "int8_backend=native" in results and "int8_backend=cutlass" not in results:
        print_ratio("int8/bf16 ratios", results["int8_backend=native"], results["bf16"])
    if "bf16" in results and "w8a8_backend=native" in results:
        print_ratio("w8a8/bf16 ratios", results["w8a8_backend=native"], results["bf16"])
    if "int8_backend=native" in results and "w8a8_backend=native" in results:
        print_ratio("w8a8/int8 ratios", results["w8a8_backend=native"], results["int8_backend=native"])
    if "bf16" in results and "int8_backend=cutlass" in results:
        print_ratio("int8_cutlass/bf16 ratios", results["int8_backend=cutlass"], results["bf16"])
    if "bf16" in results and "int8_backend=native" in results:
        print_ratio("int8_native/bf16 ratios", results["int8_backend=native"], results["bf16"])

    return {
        "kind": "single",
        "results": list(results.values()),
    }


def run_sweep_mode(args, prompt_token_ids, sampling_params) -> dict:
    sweep_results = []
    points = build_sweep_points(args)
    specs = mode_specs(args)
    print(
        f"Running sweep with {len(points)} config point(s) "
        f"and {len(specs)} mode/backend combination(s)."
    )
    for point in points:
        for spec in specs:
            result = run_one(
                spec["label"],
                args,
                prompt_token_ids,
                sampling_params,
                backend=spec["backend"],
                config_override=point,
            )
            print_result_line(result, context="sweep")
            sweep_results.append(result)
    print_sweep_summary(sweep_results)
    print_best_by_mode(sweep_results)
    return {
        "kind": "sweep",
        "sweep_points": points,
        "results": sweep_results,
    }


def build_report(args, workload_summary: dict, run_payload: dict) -> dict:
    return {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "report_type": run_payload["kind"],
        "comparison_modes": {
            "fixed_config": "Control-variable comparison under identical workload and scheduler limits.",
            "capacity_sweep": "Scaling comparison across max_num_seqs/max_num_batched_tokens operating points.",
        },
        "cli_args": vars(args),
        "workload": {
            "profile": args.profile,
            "fixed_input_len": args.fixed_input_len,
            "fixed_output_len": args.fixed_output_len,
            **workload_summary,
        },
        "results": run_payload["results"],
        "sweep_points": run_payload.get("sweep_points"),
    }


def main():
    args = parse_args()
    global torch, LLM, SamplingParams
    import torch
    from nanovllm import LLM, SamplingParams

    prompt_token_ids, sampling_params = make_workload(args)
    workload_summary = build_workload_summary(prompt_token_ids, sampling_params)

    if is_sweep_mode(args):
        run_payload = run_sweep_mode(args, prompt_token_ids, sampling_params)
    else:
        run_payload = run_single_mode(args, prompt_token_ids, sampling_params)

    report = build_report(args, workload_summary, run_payload)
    write_json_report(args.output, report)


if __name__ == "__main__":
    main()
