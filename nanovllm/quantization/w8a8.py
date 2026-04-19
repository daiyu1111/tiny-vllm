import torch
from torch import nn

from nanovllm.quantization.base import QuantMethod
from nanovllm.quantization.cuda import apply_w8a8_linear
from nanovllm.quantization.utils import quantize_int8_per_row_dynamic


class W8A8QuantMethod(QuantMethod):

    def create_weights(self, layer: nn.Module, input_size: int, output_size: int) -> None:
        layer.qweight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=torch.int8),
            requires_grad=False,
        )
        layer.w_scales = nn.Parameter(
            torch.empty(output_size, dtype=torch.float32),
            requires_grad=False,
        )

    def quantize_activation(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_2d = x.contiguous().view(-1, x.shape[-1])
        return quantize_int8_per_row_dynamic(x_2d)

    def apply(
        self,
        x: torch.Tensor,
        layer: nn.Module,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_q, a_scales = self.quantize_activation(x)
        return apply_w8a8_linear(
            x_q,
            a_scales,
            layer.qweight,
            layer.w_scales,
            bias=bias if bias is not None else layer.bias,
            out_dtype=x.dtype,
            out_shape=(*x.shape[:-1], layer.qweight.shape[0]),
        )
