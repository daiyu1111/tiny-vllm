import argparse
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from nanovllm.quantization.utils import quantize_int8_per_channel


QKV_WEIGHT_SUFFIXES = {
    "q": "q_proj.weight",
    "k": "k_proj.weight",
    "v": "v_proj.weight",
}
QKV_BIAS_SUFFIXES = {
    "q": "q_proj.bias",
    "k": "k_proj.bias",
    "v": "v_proj.bias",
}
GATE_UP_WEIGHT_SUFFIXES = {
    "gate": "gate_proj.weight",
    "up": "up_proj.weight",
}
GATE_UP_BIAS_SUFFIXES = {
    "gate": "gate_proj.bias",
    "up": "up_proj.bias",
}
DIRECT_TARGET_SUFFIXES = ("o_proj.weight", "down_proj.weight")


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize Nano-vLLM Qwen3 linear weights to INT8 weight-only format.")
    parser.add_argument("--model", required=True, type=Path, help="Source Hugging Face model directory")
    parser.add_argument("--output", required=True, type=Path, help="Output quantized model directory")
    parser.add_argument("--quantization", default="int8", choices=["int8"], help="Quantization format to generate")
    parser.add_argument("--overwrite", action="store_true", help="Remove an existing output directory before writing")
    return parser.parse_args()


def copy_model_files(model_dir: Path, output_dir: Path):
    for src in model_dir.iterdir():
        dst = output_dir / src.name
        if src.is_file():
            if src.suffix == ".safetensors" or src.name.endswith(".safetensors.index.json"):
                continue
            shutil.copy2(src, dst)
        elif src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.safetensors"), dirs_exist_ok=True)


def load_tensors(model_dir: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    files = sorted(model_dir.glob("*.safetensors"))
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


def quantize_weight(output: dict[str, torch.Tensor], name: str, weight: torch.Tensor):
    qweight, scales = quantize_int8_per_channel(weight)
    output[f"{name}.qweight"] = qweight.contiguous()
    output[f"{name}.scales"] = scales.contiguous()


def require_group(groups: dict[str, dict[str, torch.Tensor]], required: tuple[str, ...], label: str):
    for prefix, shards in groups.items():
        missing = [name for name in required if name not in shards]
        if missing:
            raise ValueError(f"Missing {label} shards for {prefix}: {missing}")


def build_quantized_tensors(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    output = {}
    qkv_weights: dict[str, dict[str, torch.Tensor]] = {}
    qkv_biases: dict[str, dict[str, torch.Tensor]] = {}
    gate_up_weights: dict[str, dict[str, torch.Tensor]] = {}
    gate_up_biases: dict[str, dict[str, torch.Tensor]] = {}

    for name, tensor in tensors.items():
        qkv_weight = match_suffix(name, QKV_WEIGHT_SUFFIXES)
        qkv_bias = match_suffix(name, QKV_BIAS_SUFFIXES)
        gate_up_weight = match_suffix(name, GATE_UP_WEIGHT_SUFFIXES)
        gate_up_bias = match_suffix(name, GATE_UP_BIAS_SUFFIXES)
        if qkv_weight is not None:
            prefix, shard_id = qkv_weight
            qkv_weights.setdefault(prefix, {})[shard_id] = tensor
        elif qkv_bias is not None:
            prefix, shard_id = qkv_bias
            qkv_biases.setdefault(prefix, {})[shard_id] = tensor
        elif gate_up_weight is not None:
            prefix, shard_id = gate_up_weight
            gate_up_weights.setdefault(prefix, {})[shard_id] = tensor
        elif gate_up_bias is not None:
            prefix, shard_id = gate_up_bias
            gate_up_biases.setdefault(prefix, {})[shard_id] = tensor
        elif name.endswith(DIRECT_TARGET_SUFFIXES):
            quantize_weight(output, name[: -len(".weight")], tensor)
        else:
            output[name] = tensor.contiguous()

    require_group(qkv_weights, ("q", "k", "v"), "qkv weight")
    require_group(gate_up_weights, ("gate", "up"), "gate/up weight")
    require_group(qkv_biases, ("q", "k", "v"), "qkv bias")
    require_group(gate_up_biases, ("gate", "up"), "gate/up bias")

    for prefix, shards in qkv_weights.items():
        packed = torch.cat([shards["q"], shards["k"], shards["v"]], dim=0)
        quantize_weight(output, f"{prefix}qkv_proj", packed)
    for prefix, shards in qkv_biases.items():
        output[f"{prefix}qkv_proj.bias"] = torch.cat([shards["q"], shards["k"], shards["v"]], dim=0).contiguous()

    for prefix, shards in gate_up_weights.items():
        packed = torch.cat([shards["gate"], shards["up"]], dim=0)
        quantize_weight(output, f"{prefix}gate_up_proj", packed)
    for prefix, shards in gate_up_biases.items():
        output[f"{prefix}gate_up_proj.bias"] = torch.cat([shards["gate"], shards["up"]], dim=0).contiguous()

    return output


def main():
    args = parse_args()
    if not args.model.is_dir():
        raise NotADirectoryError(args.model)
    if args.output.exists() and args.overwrite:
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    copy_model_files(args.model, args.output)
    tensors = load_tensors(args.model)
    quantized_tensors = build_quantized_tensors(tensors)
    save_file(quantized_tensors, args.output / "model.safetensors")
    print(f"Wrote INT8 weight-only model to {args.output}")


if __name__ == "__main__":
    main()
