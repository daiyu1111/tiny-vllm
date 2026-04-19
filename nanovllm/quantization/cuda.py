from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F

from nanovllm.utils.context import get_context


_NATIVE_EXTENSION_NAME = "nanovllm_int8_weight_only_cuda"
_CUTLASS_EXTENSION_NAME = "nanovllm_int8_weight_only_cutlass"
_LOGGED_KERNEL_EVENTS: set[str] = set()
_SUPPORTED_BACKENDS = {"auto", "native", "fallback", "cutlass"}


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


def _should_log_kernel_path() -> bool:
    value = os.environ.get("NANOVLLM_INT8_LOG_PATH", "")
    return value.lower() not in ("", "0", "false", "off", "no")


def _get_backend_preference() -> str:
    value = os.environ.get("NANOVLLM_INT8_BACKEND", "native").strip().lower()
    return value if value in _SUPPORTED_BACKENDS else "native"


def _get_phase() -> str:
    return "prefill" if get_context().is_prefill else "decode"


def _log_kernel_event_once(event: str, message: str) -> None:
    if not _should_log_kernel_path() or event in _LOGGED_KERNEL_EVENTS:
        return
    _LOGGED_KERNEL_EVENTS.add(event)
    print(message)


def _log_backend_event_once(
    *,
    backend: str,
    phase: str,
    path: str,
    x: torch.Tensor,
    qweight: torch.Tensor,
    reason: str | None = None,
) -> None:
    event = f"{backend}:{phase}:{path}:{reason or ''}:{x.dtype}:{tuple(x.shape)}:{tuple(qweight.shape)}"
    message = (
        "[nanovllm.int8] "
        f"backend={backend} phase={phase} path={path} "
        f"x_dtype={x.dtype} x_shape={tuple(x.shape)} qweight_shape={tuple(qweight.shape)}"
    )
    if reason is not None:
        message += f" reason={reason}"
    _log_kernel_event_once(event, message)


def _resolve_cutlass_include_dirs() -> list[str]:
    candidates: list[Path] = []
    env_values = [
        os.environ.get("NANOVLLM_CUTLASS_INCLUDE"),
        os.environ.get("CUTLASS_INCLUDE_DIR"),
        os.environ.get("CUTLASS_PATH"),
    ]
    for value in env_values:
        if not value:
            continue
        path = Path(value).expanduser()
        candidates.append(path)
        candidates.append(path / "include")

    candidates.extend(
        [
            Path("/usr/local/include"),
            Path("/usr/local/cutlass/include"),
            Path("/opt/cutlass/include"),
            Path("/workspace/cutlass/include"),
        ]
    )

    resolved: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        header = candidate / "cutlass" / "cutlass.h"
        candidate_str = str(candidate)
        if header.is_file() and candidate_str not in seen:
            resolved.append(candidate_str)
            seen.add(candidate_str)
    return resolved


@lru_cache(maxsize=1)
def _load_native_extension():
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
        name=_NATIVE_EXTENSION_NAME,
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False,
    )


@lru_cache(maxsize=1)
def _load_cutlass_extension():
    from torch.utils.cpp_extension import load

    include_dirs = _resolve_cutlass_include_dirs()
    if not include_dirs:
        raise RuntimeError(
            "CUTLASS headers not found. Set NANOVLLM_CUTLASS_INCLUDE, CUTLASS_INCLUDE_DIR, or CUTLASS_PATH."
        )

    this_dir = Path(__file__).resolve().parent
    sources = [
        str(this_dir / "csrc" / "int8_weight_only_gemm_cutlass.cpp"),
        str(this_dir / "csrc" / "int8_weight_only_gemm_cutlass.cu"),
    ]
    extra_cuda_cflags = [
        "-O3",
        "--use_fast_math",
    ]
    if os.name != "nt":
        extra_cuda_cflags.append("--expt-relaxed-constexpr")
    return load(
        name=_CUTLASS_EXTENSION_NAME,
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=extra_cuda_cflags,
        extra_include_paths=include_dirs,
        verbose=False,
    )


def _run_extension_linear(
    extension_loader,
    *,
    backend: str,
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    if not _can_use_cuda_kernel(x, qweight, scales):
        raise RuntimeError("CUDA INT8 weight-only backend does not support the provided tensors")
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

    extension = extension_loader()
    path = extension.select_path(x_2d, qweight)
    _log_backend_event_once(
        backend=backend,
        phase=_get_phase(),
        path=path,
        x=x_2d,
        qweight=qweight,
    )
    y = extension.forward(x_2d, qweight, scales, bias)
    return y.view(*x.shape[:-1], qweight.shape[0])


def native_int8_weight_only_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return _run_extension_linear(
        _load_native_extension,
        backend="native",
        x=x,
        qweight=qweight,
        scales=scales,
        bias=bias,
    )


def cutlass_int8_weight_only_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return _run_extension_linear(
        _load_cutlass_extension,
        backend="cutlass",
        x=x,
        qweight=qweight,
        scales=scales,
        bias=bias,
    )


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
    backend = _get_backend_preference()
    if backend == "fallback":
        _log_backend_event_once(
            backend="fallback",
            phase=_get_phase(),
            path="python_fallback",
            x=x,
            qweight=qweight,
            reason="backend_forced_fallback",
        )
        return int8_weight_only_linear_fallback(x, qweight, scales, bias)

    if backend in ("cutlass", "auto"):
        try:
            return cutlass_int8_weight_only_linear(x, qweight, scales, bias)
        except Exception as exc:
            if backend == "cutlass":
                _log_backend_event_once(
                    backend="fallback",
                    phase=_get_phase(),
                    path="python_fallback",
                    x=x,
                    qweight=qweight,
                    reason=f"cutlass_unavailable {type(exc).__name__}: {exc}",
                )
                return int8_weight_only_linear_fallback(x, qweight, scales, bias)
            _log_backend_event_once(
                backend="cutlass",
                phase=_get_phase(),
                path="unavailable",
                x=x,
                qweight=qweight,
                reason=f"{type(exc).__name__}: {exc}",
            )

    if backend in ("native", "auto"):
        try:
            return native_int8_weight_only_linear(x, qweight, scales, bias)
        except Exception as exc:
            _log_backend_event_once(
                backend="fallback",
                phase=_get_phase(),
                path="python_fallback",
                x=x,
                qweight=qweight,
                reason=f"native_unavailable {type(exc).__name__}: {exc}",
            )
            return int8_weight_only_linear_fallback(x, qweight, scales, bias)

    _log_backend_event_once(
        backend="fallback",
        phase=_get_phase(),
        path="python_fallback",
        x=x,
        qweight=qweight,
        reason=f"unknown_backend {backend}",
    )
    return int8_weight_only_linear_fallback(x, qweight, scales, bias)
