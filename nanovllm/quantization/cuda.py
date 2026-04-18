from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F


_EXTENSION_NAME = "nanovllm_int8_weight_only_cuda"


def _can_use_cuda_kernel(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor) -> bool:
    return (
        torch.cuda.is_available()
        and x.is_cuda
        and qweight.is_cuda
        and scales.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16)
        and qweight.dtype == torch.int8
        and scales.dtype == torch.float32
        and x.numel() > 0
        and qweight.numel() > 0
        and scales.numel() > 0
    )


@lru_cache(maxsize=1)
def _load_extension():
    from torch.utils.cpp_extension import load

    this_dir = Path(__file__).resolve().parent
    sources = [
        str(this_dir / "csrc" / "int8_weight_only_gemm.cpp"),
        str(this_dir / "csrc" / "int8_weight_only_gemm.cu"),
    ]
    extra_cuda_cflags = [
        "-O3",
        "--use_fast_math",
    ]
    if os.name != "nt":
        extra_cuda_cflags.append("--expt-relaxed-constexpr")
    return load(
        name=_EXTENSION_NAME,
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False,
    )



def fused_int8_weight_only_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if not _can_use_cuda_kernel(x, qweight, scales):
        raise RuntimeError("CUDA fused INT8 weight-only kernel does not support the provided tensors")
    if x.shape[-1] != qweight.shape[-1]:
        raise RuntimeError(f"Incompatible shapes: x[-1]={x.shape[-1]} vs qweight[-1]={qweight.shape[-1]}")
    if scales.dim() != 1 or scales.numel() != qweight.shape[0]:
        raise RuntimeError("scales must be 1D with one value per output channel")
    if bias is not None and (bias.dim() != 1 or bias.numel() != qweight.shape[0]):
        raise RuntimeError("bias must be 1D with one value per output channel")

    x_2d = x.contiguous().view(-1, x.shape[-1])
    qweight = qweight.contiguous()
    scales = scales.contiguous()
    bias = None if bias is None else bias.contiguous()

    extension = _load_extension()
    y = extension.forward(x_2d, qweight, scales, bias)
    return y.view(*x.shape[:-1], qweight.shape[0])



def int8_weight_only_linear_fallback(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    weight = qweight.to(x.dtype) * scales.to(x.dtype).unsqueeze(1)
    return F.linear(x, weight, bias)



def apply_int8_weight_only_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if _can_use_cuda_kernel(x, qweight, scales):
        try:
            return fused_int8_weight_only_linear(x, qweight, scales, bias)
        except Exception:
            pass
    return int8_weight_only_linear_fallback(x, qweight, scales, bias)
