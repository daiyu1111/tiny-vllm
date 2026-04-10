import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.quantization.base import QuantMethod


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
        weight = layer.qweight.to(x.dtype) * layer.scales.to(x.dtype).unsqueeze(1)
        return F.linear(x, weight, layer.bias)
