import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def _load_parameter(model: nn.Module, param_name: str, loaded_weight: torch.Tensor, loaded_params: set[str], *args):
    param = model.get_parameter(param_name)
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, loaded_weight, *args)
    loaded_params.add(param_name)


def _has_parameter(model: nn.Module, param_name: str) -> bool:
    try:
        model.get_parameter(param_name)
    except AttributeError:
        return False
    return True


def _validate_quantized_loaded(model: nn.Module, loaded_params: set[str], scale_suffix: str, label: str):
    expected = {
        name
        for name, _ in model.named_parameters()
        if name.endswith(".qweight") or name.endswith(scale_suffix)
    }
    missing = sorted(expected - loaded_params)
    if missing:
        preview = ", ".join(missing[:8])
        if len(missing) > 8:
            preview += f", ... ({len(missing)} missing total)"
        raise RuntimeError(
            f"Missing {label} quantized tensors. Expected qweight/{scale_suffix[1:]} for quantized "
            f"linear layers, but did not load: {preview}"
        )


def load_model(model: nn.Module, path: str, quantization: str | None = None):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    loaded_params: set[str] = set()
    files = glob(os.path.join(path, "*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors files found in {path}")
    for file in files:
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                loaded_weight = f.get_tensor(weight_name)
                if _has_parameter(model, weight_name):
                    _load_parameter(model, weight_name, loaded_weight, loaded_params)
                    continue
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        _load_parameter(model, param_name, loaded_weight, loaded_params, shard_id)
                        break
                else:
                    _load_parameter(model, weight_name, loaded_weight, loaded_params)
    if quantization == "int8":
        _validate_quantized_loaded(model, loaded_params, ".scales", "INT8")
    if quantization == "w8a8":
        _validate_quantized_loaded(model, loaded_params, ".w_scales", "W8A8")
