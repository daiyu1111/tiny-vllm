from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanovllm import LLM
from scripts.benchmark_cases import build_text_cases
from scripts.quantize import DIRECT_TARGET_SUFFIXES, GATE_UP_WEIGHT_SUFFIXES, QKV_WEIGHT_SUFFIXES


BUILTIN_PPL_TEXTS = [
    "Nano-vLLM is a compact inference engine used to study batching, KV cache management, and quantization.",
    "Weight-only quantization should preserve the model's language modeling behavior while reducing resident memory.",
    "A stable quality report makes it easier to compare bf16 and int8 models across future kernel changes.",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Nano-vLLM INT8 quantization quality.")
    parser.add_argument("--model", required=True, help="Base model directory.")
    parser.add_argument("--int8-model", required=True, help="INT8 quantized model directory.")
    parser.add_argument(
        "--mode",
        choices=["artifact", "logits", "ppl", "generation", "all"],
        default="all",
        help="Quality check to run.",
    )
    parser.add_argument("--output", default="quant_quality_report.json", help="JSON report path.")
    parser.add_argument(
        "--ppl-source",
        choices=["builtin", "text-file", "token-ids-json", "hf-dataset"],
        default="hf-dataset",
        help="Text source for PPL evaluation.",
    )
    parser.add_argument("--hf-dataset", default="wikitext", help="Hugging Face dataset name.")
    parser.add_argument("--hf-dataset-config", default="wikitext-2-raw-v1", help="Hugging Face dataset config.")
    parser.add_argument("--hf-dataset-split", default="test", help="Hugging Face dataset split.")
    parser.add_argument("--ppl-text-file", help="Local UTF-8 text file for PPL evaluation.")
    parser.add_argument("--ppl-token-ids-json", help="JSON file containing token ids or list of token id lists.")
    parser.add_argument("--ppl-max-samples", type=int, default=16, help="Maximum PPL text samples to evaluate.")
    parser.add_argument("--ppl-max-seq-len", type=int, default=1024, help="Maximum tokens per PPL window.")
    parser.add_argument("--ppl-stride", type=int, default=512, help="Stride between PPL windows.")
    parser.add_argument("--generation-max-tokens", type=int, default=64, help="Argmax generation steps per prompt.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Runtime max model length.")
    parser.add_argument("--max-num-seqs", type=int, default=512, help="Runtime max concurrent sequences.")
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384, help="Runtime max batched tokens.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization.")
    return parser.parse_args()


def normalize_args(args):
    if args.ppl_stride <= 0:
        raise ValueError("--ppl-stride must be positive")
    if args.ppl_max_seq_len < 2:
        raise ValueError("--ppl-max-seq-len must be at least 2")
    if args.generation_max_tokens <= 0:
        raise ValueError("--generation-max-tokens must be positive")
    args.ppl_max_seq_len = min(args.ppl_max_seq_len, args.max_model_len)
    args.ppl_stride = min(args.ppl_stride, args.ppl_max_seq_len)
    return args


def load_tensors(model_dir: str | Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    files = sorted(Path(model_dir).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors files found in {model_dir}")
    for file in files:
        with safe_open(file, "pt", "cpu") as f:
            for name in f.keys():
                if name in tensors:
                    raise ValueError(f"Duplicate tensor name across safetensors files: {name}")
                tensors[name] = f.get_tensor(name)
    return tensors


def match_suffix(name: str, suffixes: dict[str, str]) -> tuple[str, str] | None:
    for shard_id, suffix in suffixes.items():
        if name.endswith(suffix):
            return name[: -len(suffix)], shard_id
    return None


def build_expected_quant_weights(base_tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    expected: dict[str, torch.Tensor] = {}
    qkv_groups: dict[str, dict[str, torch.Tensor]] = {}
    gate_up_groups: dict[str, dict[str, torch.Tensor]] = {}

    for name, tensor in base_tensors.items():
        qkv_match = match_suffix(name, QKV_WEIGHT_SUFFIXES)
        gate_up_match = match_suffix(name, GATE_UP_WEIGHT_SUFFIXES)
        if qkv_match is not None:
            prefix, shard_id = qkv_match
            qkv_groups.setdefault(prefix, {})[shard_id] = tensor
        elif gate_up_match is not None:
            prefix, shard_id = gate_up_match
            gate_up_groups.setdefault(prefix, {})[shard_id] = tensor
        elif name.endswith(DIRECT_TARGET_SUFFIXES):
            expected[name[: -len(".weight")]] = tensor

    for prefix, shards in qkv_groups.items():
        if {"q", "k", "v"} <= shards.keys():
            expected[f"{prefix}qkv_proj"] = torch.cat([shards["q"], shards["k"], shards["v"]], dim=0)
    for prefix, shards in gate_up_groups.items():
        if {"gate", "up"} <= shards.keys():
            expected[f"{prefix}gate_up_proj"] = torch.cat([shards["gate"], shards["up"]], dim=0)
    return dict(sorted(expected.items()))


def tensor_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference = reference.float().flatten()
    candidate = candidate.float().flatten()
    diff = candidate - reference
    ref_norm = torch.linalg.vector_norm(reference)
    rel_l2 = torch.linalg.vector_norm(diff) / ref_norm.clamp_min(1e-12)
    return {
        "mae": float(diff.abs().mean().item()),
        "max_error": float(diff.abs().max().item()),
        "relative_l2": float(rel_l2.item()),
        "cosine_similarity": float(F.cosine_similarity(reference, candidate, dim=0, eps=1e-12).item()),
    }


def run_artifact_check(args) -> dict[str, Any]:
    result: dict[str, Any] = {"passed": True, "errors": [], "layers": {}, "summary": {}}
    base_tensors = load_tensors(args.model)
    int8_tensors = load_tensors(args.int8_model)
    expected = build_expected_quant_weights(base_tensors)

    for name, reference in expected.items():
        q_name = f"{name}.qweight"
        s_name = f"{name}.scales"
        layer_result: dict[str, Any] = {
            "qweight": q_name,
            "scales": s_name,
            "reference_shape": list(reference.shape),
        }
        qweight = int8_tensors.get(q_name)
        scales = int8_tensors.get(s_name)
        layer_errors = []
        if qweight is None:
            layer_errors.append("missing qweight")
        if scales is None:
            layer_errors.append("missing scales")
        if qweight is not None:
            layer_result["qweight_shape"] = list(qweight.shape)
            layer_result["qweight_dtype"] = str(qweight.dtype)
            if qweight.dtype != torch.int8:
                layer_errors.append(f"qweight dtype is {qweight.dtype}, expected torch.int8")
            if tuple(qweight.shape) != tuple(reference.shape):
                layer_errors.append(f"qweight shape {tuple(qweight.shape)} != reference {tuple(reference.shape)}")
        if scales is not None:
            layer_result["scales_shape"] = list(scales.shape)
            layer_result["scales_dtype"] = str(scales.dtype)
            if scales.dtype != torch.float32:
                layer_errors.append(f"scales dtype is {scales.dtype}, expected torch.float32")
            if scales.dim() != 1 or scales.numel() != reference.shape[0]:
                layer_errors.append(f"scales shape {tuple(scales.shape)} does not match output channels {reference.shape[0]}")
        if not layer_errors and qweight is not None and scales is not None:
            dequant = qweight.float() * scales.float().unsqueeze(1)
            layer_result["reconstruction"] = tensor_metrics(reference, dequant)
        else:
            result["passed"] = False
            result["errors"].extend([f"{name}: {error}" for error in layer_errors])
        layer_result["passed"] = not layer_errors
        layer_result["errors"] = layer_errors
        result["layers"][name] = layer_result

    metrics = [
        layer["reconstruction"]
        for layer in result["layers"].values()
        if "reconstruction" in layer
    ]
    result["summary"] = {
        "expected_layers": len(expected),
        "checked_layers": len(result["layers"]),
        "passed_layers": sum(1 for layer in result["layers"].values() if layer["passed"]),
        "mean_mae": mean([metric["mae"] for metric in metrics]) if metrics else None,
        "mean_relative_l2": mean([metric["relative_l2"] for metric in metrics]) if metrics else None,
        "min_cosine_similarity": min([metric["cosine_similarity"] for metric in metrics]) if metrics else None,
    }
    return result


def runtime_kwargs(args) -> dict[str, Any]:
    return {
        "enforce_eager": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }


def load_eval_llm(args, label: str) -> LLM:
    kwargs = runtime_kwargs(args)
    if label == "int8":
        kwargs.update({"quantization": "int8", "quantized_model_path": os.path.abspath(args.int8_model)})
    return LLM(os.path.abspath(args.model), **kwargs)


def cleanup_llm(llm: LLM | None):
    if llm is not None:
        llm.exit()
        del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(math.ceil(q * len(values))) - 1))
    return float(values[idx])


def topk_overlap(base_logits: torch.Tensor, int8_logits: torch.Tensor, k: int) -> float:
    base_top = torch.topk(base_logits, k, dim=-1).indices
    int8_top = torch.topk(int8_logits, k, dim=-1).indices
    overlaps = []
    for a, b in zip(base_top, int8_top):
        overlaps.append(len(set(a.tolist()) & set(b.tolist())) / k)
    return float(mean(overlaps)) if overlaps else 0.0


def margin_bucket_agreement(base_logits: torch.Tensor, int8_logits: torch.Tensor) -> dict[str, dict[str, float | int]]:
    top2 = torch.topk(base_logits, 2, dim=-1).values
    margins = (top2[:, 0] - top2[:, 1]).float()
    agree = torch.argmax(base_logits, dim=-1) == torch.argmax(int8_logits, dim=-1)
    buckets = {
        "lt_0_01": margins < 0.01,
        "0_01_to_0_1": (margins >= 0.01) & (margins < 0.1),
        "0_1_to_1": (margins >= 0.1) & (margins < 1.0),
        "gte_1": margins >= 1.0,
    }
    result = {}
    for name, mask in buckets.items():
        count = int(mask.sum().item())
        result[name] = {
            "count": count,
            "agreement": float(agree[mask].float().mean().item()) if count else None,
        }
    return result


def collect_quality_prompts(model_path: str, max_model_len: int) -> tuple[list[str], list[list[int]]]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    cases = build_text_cases(tokenizer)
    prompts = []
    for case in cases:
        if case.kind == "text":
            prompts.extend(case.prompts)
    token_ids = [tokenizer.encode(prompt)[:max_model_len] for prompt in prompts]
    pairs = [(prompt, ids) for prompt, ids in zip(prompts, token_ids) if ids]
    return [prompt for prompt, _ in pairs], [ids for _, ids in pairs]


def get_last_logits(args, token_ids_batch: list[list[int]], label: str) -> torch.Tensor:
    llm = None
    try:
        llm = load_eval_llm(args, label)
        return llm.model_runner.call("prefill_last_logits", token_ids_batch)
    finally:
        cleanup_llm(llm)


def run_logits_check(args) -> dict[str, Any]:
    prompts, token_ids_batch = collect_quality_prompts(args.model, args.max_model_len)
    base_logits = get_last_logits(args, token_ids_batch, "bf16")
    int8_logits = get_last_logits(args, token_ids_batch, "int8")
    result: dict[str, Any] = {
        "passed": True,
        "errors": [],
        "num_prompts": len(prompts),
        "prompt_token_lengths": [len(ids) for ids in token_ids_batch],
    }
    if tuple(base_logits.shape) != tuple(int8_logits.shape):
        result["passed"] = False
        result["errors"].append(f"logits shape mismatch: bf16={tuple(base_logits.shape)} int8={tuple(int8_logits.shape)}")
        return result

    diff = int8_logits - base_logits
    row_cos = F.cosine_similarity(base_logits.float(), int8_logits.float(), dim=-1, eps=1e-12)
    rel_l2 = torch.linalg.vector_norm(diff.float()) / torch.linalg.vector_norm(base_logits.float()).clamp_min(1e-12)
    result.update(
        {
            "shape": list(base_logits.shape),
            "cosine_similarity": {
                "mean": float(row_cos.mean().item()),
                "min": float(row_cos.min().item()),
                "p05": percentile(row_cos.tolist(), 0.05),
            },
            "mae": float(diff.abs().mean().item()),
            "relative_l2": float(rel_l2.item()),
            "max_error": float(diff.abs().max().item()),
            "top1_agreement": float((base_logits.argmax(dim=-1) == int8_logits.argmax(dim=-1)).float().mean().item()),
            "top5_overlap": topk_overlap(base_logits, int8_logits, 5),
            "top10_overlap": topk_overlap(base_logits, int8_logits, 10),
            "top1_margin_buckets": margin_bucket_agreement(base_logits, int8_logits),
        }
    )
    return result


def load_ppl_token_sequences(args, tokenizer) -> tuple[str, list[list[int]], dict[str, Any]]:
    if args.ppl_source == "builtin":
        texts = BUILTIN_PPL_TEXTS[: args.ppl_max_samples]
        return "builtin", [tokenizer.encode(text) for text in texts], {"num_texts": len(texts)}
    if args.ppl_source == "text-file":
        if not args.ppl_text_file:
            raise ValueError("--ppl-text-file is required when --ppl-source=text-file")
        text = Path(args.ppl_text_file).read_text(encoding="utf-8")
        chunks = [chunk.strip() for chunk in text.splitlines() if chunk.strip()]
        if not chunks:
            chunks = [text]
        chunks = chunks[: args.ppl_max_samples]
        return "text-file", [tokenizer.encode(chunk) for chunk in chunks], {
            "path": os.path.abspath(args.ppl_text_file),
            "num_texts": len(chunks),
        }
    if args.ppl_source == "token-ids-json":
        if not args.ppl_token_ids_json:
            raise ValueError("--ppl-token-ids-json is required when --ppl-source=token-ids-json")
        payload = json.loads(Path(args.ppl_token_ids_json).read_text(encoding="utf-8"))
        if payload and isinstance(payload[0], int):
            sequences = [payload]
        else:
            sequences = payload
        return "token-ids-json", [list(map(int, seq)) for seq in sequences[: args.ppl_max_samples]], {
            "path": os.path.abspath(args.ppl_token_ids_json),
            "num_sequences": min(len(sequences), args.ppl_max_samples),
        }
    if args.ppl_source == "hf-dataset":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "PPL source 'hf-dataset' requires the optional dependency 'datasets'. "
                "Install it with: pip install datasets"
            ) from exc
        dataset = load_dataset(args.hf_dataset, args.hf_dataset_config, split=args.hf_dataset_split)
        texts = []
        for row in dataset:
            text = str(row.get("text", "")).strip()
            if text:
                texts.append(text)
            if len(texts) >= args.ppl_max_samples:
                break
        return "hf-dataset", [tokenizer.encode(text) for text in texts], {
            "dataset": args.hf_dataset,
            "config": args.hf_dataset_config,
            "split": args.hf_dataset_split,
            "num_texts": len(texts),
        }
    raise ValueError(f"Unsupported PPL source: {args.ppl_source}")


def iter_windows(token_ids: list[int], max_seq_len: int, stride: int):
    if len(token_ids) < 2:
        return
    start = 0
    while start < len(token_ids) - 1:
        end = min(start + max_seq_len, len(token_ids))
        window = token_ids[start:end]
        if len(window) >= 2:
            yield window
        if end == len(token_ids):
            break
        start += stride


def compute_model_nll(args, sequences: list[list[int]], label: str) -> dict[str, Any]:
    llm = None
    total_nll = 0.0
    total_tokens = 0
    windows = 0
    try:
        llm = load_eval_llm(args, label)
        for seq in sequences:
            for window in iter_windows(seq, args.ppl_max_seq_len, args.ppl_stride):
                logits = llm.model_runner.call("prefill_full_logits", window)
                shift_logits = logits[:-1].float()
                targets = torch.tensor(window[1:], dtype=torch.long)
                loss_sum = F.cross_entropy(shift_logits, targets, reduction="sum")
                total_nll += float(loss_sum.item())
                total_tokens += int(targets.numel())
                windows += 1
    finally:
        cleanup_llm(llm)
    loss = total_nll / total_tokens if total_tokens else float("nan")
    return {
        "loss": loss,
        "ppl": math.exp(loss) if math.isfinite(loss) else float("nan"),
        "num_tokens": total_tokens,
        "num_windows": windows,
    }


def run_ppl_check(args) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    source, sequences, source_info = load_ppl_token_sequences(args, tokenizer)
    sequences = [seq for seq in sequences if len(seq) >= 2]
    result: dict[str, Any] = {
        "passed": True,
        "errors": [],
        "source": source,
        "source_info": source_info,
        "max_seq_len": args.ppl_max_seq_len,
        "stride": args.ppl_stride,
        "num_sequences": len(sequences),
    }
    if not sequences:
        result["passed"] = False
        result["errors"].append("No PPL token sequences with at least two tokens.")
        return result

    base = compute_model_nll(args, sequences, "bf16")
    int8 = compute_model_nll(args, sequences, "int8")
    loss_delta = int8["loss"] - base["loss"]
    ppl_delta = int8["ppl"] - base["ppl"]
    result.update(
        {
            "bf16": base,
            "int8": int8,
            "loss_absolute_delta": loss_delta,
            "loss_relative_delta": loss_delta / base["loss"] if base["loss"] else None,
            "ppl_absolute_delta": ppl_delta,
            "ppl_relative_delta": ppl_delta / base["ppl"] if base["ppl"] else None,
        }
    )
    return result


def repeated_token_flag(token_ids: list[int], run_length: int = 8) -> bool:
    if len(token_ids) < run_length:
        return False
    count = 1
    for prev, cur in zip(token_ids, token_ids[1:]):
        count = count + 1 if cur == prev else 1
        if count >= run_length:
            return True
    return False


def argmax_generate(args, prompt_ids: list[int], label: str, max_tokens: int, eos_token_id: int | None) -> list[int]:
    llm = None
    generated: list[int] = []
    token_ids = list(prompt_ids)
    try:
        llm = load_eval_llm(args, label)
        for _ in range(max_tokens):
            logits = llm.model_runner.call("prefill_last_logits", [token_ids])
            next_token = int(torch.argmax(logits[0]).item())
            generated.append(next_token)
            token_ids.append(next_token)
            if eos_token_id is not None and next_token == eos_token_id:
                break
    finally:
        cleanup_llm(llm)
    return generated


def first_diff_position(a: list[int], b: list[int]) -> int | None:
    for i, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def run_generation_check(args) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    prompt_budget = max(1, args.max_model_len - args.generation_max_tokens)
    prompts, token_batches = collect_quality_prompts(args.model, prompt_budget)
    prompts = prompts[:4]
    token_batches = token_batches[:4]
    cases = []
    for prompt, prompt_ids in zip(prompts, token_batches):
        base_ids = argmax_generate(args, prompt_ids, "bf16", args.generation_max_tokens, tokenizer.eos_token_id)
        int8_ids = argmax_generate(args, prompt_ids, "int8", args.generation_max_tokens, tokenizer.eos_token_id)
        overlap_len = min(len(base_ids), len(int8_ids))
        matches = sum(1 for a, b in zip(base_ids, int8_ids) if a == b)
        token_match_rate = matches / overlap_len if overlap_len else 0.0
        cases.append(
            {
                "prompt": prompt,
                "prompt_token_count": len(prompt_ids),
                "bf16_token_ids": base_ids,
                "int8_token_ids": int8_ids,
                "bf16_text": tokenizer.decode(base_ids),
                "int8_text": tokenizer.decode(int8_ids),
                "token_match_rate": token_match_rate,
                "first_diff_position": first_diff_position(base_ids, int8_ids),
                "bf16_length": len(base_ids),
                "int8_length": len(int8_ids),
                "diagnostics": {
                    "bf16_empty": len(base_ids) == 0,
                    "int8_empty": len(int8_ids) == 0,
                    "bf16_repeated_token_run": repeated_token_flag(base_ids),
                    "int8_repeated_token_run": repeated_token_flag(int8_ids),
                    "bf16_ended_with_eos": bool(base_ids and tokenizer.eos_token_id is not None and base_ids[-1] == tokenizer.eos_token_id),
                    "int8_ended_with_eos": bool(int8_ids and tokenizer.eos_token_id is not None and int8_ids[-1] == tokenizer.eos_token_id),
                },
            }
        )
    return {
        "passed": True,
        "errors": [],
        "max_tokens": args.generation_max_tokens,
        "num_prompts": len(cases),
        "mean_token_match_rate": mean([case["token_match_rate"] for case in cases]) if cases else None,
        "cases": cases,
    }


def build_metadata(args) -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    runtime = {
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
    }
    if cuda:
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        runtime.update(
            {
                "device_index": int(device),
                "gpu_name": props.name,
                "gpu_total_memory": int(props.total_memory),
            }
        )
    return {
        "created_at": datetime.now().isoformat(),
        "model": os.path.abspath(args.model),
        "int8_model": os.path.abspath(args.int8_model),
        "mode": args.mode,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "ppl": {
            "source": args.ppl_source,
            "hf_dataset": args.hf_dataset,
            "hf_dataset_config": args.hf_dataset_config,
            "hf_dataset_split": args.hf_dataset_split,
            "max_samples": args.ppl_max_samples,
            "max_seq_len": args.ppl_max_seq_len,
            "stride": args.ppl_stride,
        },
        "runtime": runtime,
    }


def summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    failures = []
    stages = {}
    for key in ("artifact_check", "logits_check", "ppl_check", "generation_check"):
        stage = report.get(key)
        if stage is None:
            stages[key] = None
            continue
        passed = bool(stage.get("passed", False))
        stages[key] = passed
        if not passed:
            failures.extend([f"{key}: {error}" for error in stage.get("errors", [])])
    return {
        "passed": all(value is not False for value in stages.values()),
        "stages": stages,
        "failures": failures,
    }


def run_stage(name: str, fn, args, report: dict[str, Any]):
    print(f"Running {name}...")
    try:
        report[name] = fn(args)
    except Exception as exc:
        report[name] = {
            "passed": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
            "traceback": traceback.format_exc(),
        }


def main():
    args = normalize_args(parse_args())
    report: dict[str, Any] = {
        "metadata": build_metadata(args),
        "artifact_check": None,
        "logits_check": None,
        "ppl_check": None,
        "generation_check": None,
    }
    modes = ["artifact", "logits", "ppl", "generation"] if args.mode == "all" else [args.mode]
    if "artifact" in modes:
        run_stage("artifact_check", run_artifact_check, args, report)
    if "logits" in modes:
        run_stage("logits_check", run_logits_check, args, report)
    if "ppl" in modes:
        run_stage("ppl_check", run_ppl_check, args, report)
    if "generation" in modes:
        run_stage("generation_check", run_generation_check, args, report)

    report["summary"] = summarize_report(report)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = report["summary"]
    print(f"Saved quality report to {output_path}")
    print(f"Overall passed: {summary['passed']}")
    for stage, passed in summary["stages"].items():
        if passed is not None:
            print(f"  {stage}: {'passed' if passed else 'failed'}")


if __name__ == "__main__":
    main()
