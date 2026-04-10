from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanovllm import LLM
from scripts.benchmark_cases import BenchmarkCase, get_default_cases


def parse_args():
    parser = argparse.ArgumentParser(description="Run Nano-vLLM baseline benchmark and save JSON results.")
    parser.add_argument("--model", required=True, help="Local model path.")
    parser.add_argument("--output-dir", default="artifacts/baseline", help="Directory for JSON outputs.")
    parser.add_argument("--mode", choices=["eager", "graph", "both"], default="both", help="Benchmark runtime mode.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Runtime max model length.")
    parser.add_argument("--max-num-seqs", type=int, default=512, help="Runtime max concurrent sequences.")
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384, help="Runtime max batched tokens.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization.")
    parser.add_argument("--case", action="append", dest="cases", help="Case names to run. Repeatable.")
    return parser.parse_args()


def sanitize_model_name(model_path: str) -> str:
    return Path(model_path).name.replace(".", "-").replace(" ", "-").lower()


def build_runtime_info() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for baseline benchmark.")
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info()
    return {
        "gpu_name": props.name,
        "gpu_total_memory": int(props.total_memory),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "device_index": int(device),
        "initial_free_memory": int(free),
        "initial_total_memory": int(total),
    }


def snapshot_memory() -> dict:
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {
        "memory_allocated": int(torch.cuda.memory_allocated()),
        "max_memory_allocated": int(torch.cuda.max_memory_allocated()),
        "memory_reserved": int(torch.cuda.memory_reserved()),
        "free_memory": int(free),
        "total_memory": int(total),
    }


def summarize_case(case: BenchmarkCase, outputs: list[dict], stats: dict, elapsed_seconds: float, memory: dict) -> dict:
    output_token_counts = [len(output["token_ids"]) for output in outputs]
    total_output_tokens = sum(output_token_counts)
    sample_outputs = [
        {
            "index": i,
            "text": output["text"],
            "output_token_count": len(output["token_ids"]),
        }
        for i, output in enumerate(outputs[:3])
    ]
    return {
        "name": case.name,
        "kind": case.kind,
        "description": case.description,
        "metadata": case.metadata,
        "sampling_params": [asdict(sp) for sp in case.sampling_params],
        "request_count": len(case.prompts),
        "wall_time_seconds": elapsed_seconds,
        "total_output_tokens": int(total_output_tokens),
        "tokens_per_second": total_output_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0,
        "average_request_latency_seconds": elapsed_seconds / len(case.prompts) if case.prompts else 0.0,
        "prefill_tokens_per_second": stats["prefill_tokens_per_second"],
        "decode_tokens_per_second": stats["decode_tokens_per_second"],
        "ttft_seconds_approx": stats["ttft_seconds_approx"],
        "engine_stats": stats,
        "memory": memory,
        "samples": sample_outputs,
    }


def run_case(llm: LLM, case: BenchmarkCase) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = snapshot_memory()
    start = perf_counter()
    outputs, stats = llm.generate_with_stats(case.prompts, case.sampling_params, use_tqdm=False)
    elapsed = perf_counter() - start
    after = snapshot_memory()
    memory = {
        "before_case": before,
        "after_case": after,
        "peak_memory_allocated": after["max_memory_allocated"],
    }
    return summarize_case(case, outputs, stats, elapsed, memory)


def run_mode(args, mode_name: str, cases: list[BenchmarkCase]) -> dict:
    enforce_eager = mode_name == "eager"
    llm = LLM(
        args.model,
        enforce_eager=enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    runtime_info = build_runtime_info()
    init_memory = {
        "post_init": snapshot_memory(),
        "runner_snapshots": llm.model_runner.memory_snapshots,
        "num_kvcache_blocks": int(llm.model_runner.config.num_kvcache_blocks),
    }
    cases_result = [run_case(llm, case) for case in cases]
    total_wall_time = sum(case_result["wall_time_seconds"] for case_result in cases_result)
    total_output_tokens = sum(case_result["total_output_tokens"] for case_result in cases_result)
    summary = {
        "total_cases": len(cases_result),
        "total_wall_time_seconds": total_wall_time,
        "total_output_tokens": int(total_output_tokens),
        "aggregate_tokens_per_second": total_output_tokens / total_wall_time if total_wall_time > 0 else 0.0,
        "mean_prefill_tokens_per_second": (
            sum(case_result["prefill_tokens_per_second"] for case_result in cases_result) / len(cases_result)
            if cases_result else 0.0
        ),
        "mean_decode_tokens_per_second": (
            sum(case_result["decode_tokens_per_second"] for case_result in cases_result) / len(cases_result)
            if cases_result else 0.0
        ),
    }
    llm.exit()
    return {
        "mode": mode_name,
        "runtime": runtime_info,
        "initialization": init_memory,
        "cases": cases_result,
        "summary": summary,
    }


def print_summary(mode_result: dict):
    summary = mode_result["summary"]
    print(
        f"[{mode_result['mode']}] "
        f"cases={summary['total_cases']} "
        f"tokens={summary['total_output_tokens']} "
        f"time={summary['total_wall_time_seconds']:.2f}s "
        f"throughput={summary['aggregate_tokens_per_second']:.2f} tok/s "
        f"prefill={summary['mean_prefill_tokens_per_second']:.2f} tok/s "
        f"decode={summary['mean_decode_tokens_per_second']:.2f} tok/s"
    )


def main():
    args = parse_args()
    all_cases = get_default_cases(args.model)
    if args.cases:
        case_names = set(args.cases)
        cases = [case for case in all_cases if case.name in case_names]
        missing = sorted(case_names - {case.name for case in cases})
        if missing:
            raise ValueError(f"Unknown case names: {', '.join(missing)}")
    else:
        cases = all_cases

    modes = ["eager", "graph"] if args.mode == "both" else [args.mode]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for mode_name in modes:
        mode_result = run_mode(args, mode_name, cases)
        results.append(mode_result)
        print_summary(mode_result)

    result_payload = {
        "created_at": datetime.now().isoformat(),
        "model_path": os.path.abspath(args.model),
        "tensor_parallel_size": args.tensor_parallel_size,
        "mode": args.mode,
        "cases_requested": args.cases or [case.name for case in cases],
        "results": results,
    }
    filename = f"baseline_{sanitize_model_name(args.model)}_{args.mode}_{timestamp}.json"
    output_path = output_dir / filename
    output_path.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved baseline report to {output_path}")


if __name__ == "__main__":
    main()
