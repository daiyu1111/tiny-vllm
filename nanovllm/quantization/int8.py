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

    def quantize_activation(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        return None

    def apply(
        self,
        x: torch.Tensor,
        layer: nn.Module,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return apply_int8_weight_only_linear(x, layer.qweight, layer.scales, layer.bias if bias is None else bias)
