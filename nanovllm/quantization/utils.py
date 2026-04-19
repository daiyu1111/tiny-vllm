from pathlib import Path

import torch


QWEN3_INT8_WEIGHT_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)


def is_qwen3_int8_target_weight(name: str) -> bool:
    return name.endswith(QWEN3_INT8_WEIGHT_SUFFIXES)


def quantize_int8_per_channel(weight: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    weight = weight.float()
    scales = weight.abs().amax(dim=1).div(127.0).clamp_min(eps)
    qweight = torch.round(weight / scales.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    return qweight, scales.to(torch.float32)


def iter_model_files(model_path: Path):
    for path in model_path.iterdir():
        if path.is_file() and path.suffix != ".safetensors":
            yield path
