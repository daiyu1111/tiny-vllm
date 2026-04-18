import torch
from torch import nn

from nanovllm.quantization.base import QuantMethod
from nanovllm.quantization.cuda import apply_int8_weight_only_linear


class Int8WeightOnlyQuantMethod(QuantMethod):

    def create_weights(self, layer: nn.Module, input_size: int, output_size: int) -> None:
        layer.qweight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=torch.int8),
            requires_grad=False,
        )
        layer.scales = nn.Parameter(
            torch.empty(output_size, dtype=torch.float32),
            requires_grad=False,
        )

    def apply(self, x: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        return apply_int8_weight_only_linear(x, layer.qweight, layer.scales, layer.bias)
