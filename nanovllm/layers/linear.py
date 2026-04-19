import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.quantization import Int8WeightOnlyQuantMethod, W8A8QuantMethod


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
        quantization: str | None = None,
    ):
        super().__init__()
        if quantization not in (None, "int8", "w8a8"):
            raise NotImplementedError(f"Unsupported linear quantization: {quantization}")
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.quantization = quantization
        if quantization == "int8":
            self.quant_method = Int8WeightOnlyQuantMethod()
        elif quantization == "w8a8":
            self.quant_method = W8A8QuantMethod()
        else:
            self.quant_method = None
        if self.quant_method is None:
            self.weight = nn.Parameter(torch.empty(output_size, input_size))
            self.weight.weight_loader = self.weight_loader
        else:
            self.quant_method.create_weights(self, input_size, output_size)
            self.qweight.weight_loader = self.qweight_loader
            if hasattr(self, "scales"):
                self.scales.weight_loader = self.scales_loader
            if hasattr(self, "w_scales"):
                self.w_scales.weight_loader = self.scales_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.bias_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def bias_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        self.weight_loader(param, loaded_weight)

    def qweight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        self.weight_loader(param, loaded_weight)

    def scales_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        self.weight_loader(param, loaded_weight)

    def apply_linear(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant_method is not None:
            return self.quant_method.apply(x, self)
        return F.linear(x, self.weight, self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quantization: str | None = None,
    ):
        super().__init__(input_size, output_size, bias, quantization=quantization)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply_linear(x)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quantization: str | None = None,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0, quantization)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def scales_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply_linear(x)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        quantization: str | None = None,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias, quantization)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

    def packed_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        for shard_id, output_size in enumerate(self.output_sizes):
            dst_offset = sum(self.output_sizes[:shard_id]) // self.tp_size
            dst_size = output_size // self.tp_size
            src_offset = sum(self.output_sizes[:shard_id])
            dst = param_data.narrow(self.tp_dim, dst_offset, dst_size)
            src = loaded_weight.narrow(self.tp_dim, src_offset, output_size)
            src = src.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
            dst.copy_(src)

    def packed_scales_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        for shard_id, output_size in enumerate(self.output_sizes):
            dst_offset = sum(self.output_sizes[:shard_id]) // self.tp_size
            dst_size = output_size // self.tp_size
            src_offset = sum(self.output_sizes[:shard_id])
            dst = param_data.narrow(0, dst_offset, dst_size)
            src = loaded_weight.narrow(0, src_offset, output_size)
            src = src.chunk(self.tp_size, 0)[self.tp_rank]
            dst.copy_(src)

    def bias_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int | None = None):
        if loaded_shard_id is None:
            self.packed_scales_loader(param, loaded_weight)
        else:
            self.weight_loader(param, loaded_weight, loaded_shard_id)

    def qweight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        self.packed_weight_loader(param, loaded_weight)

    def scales_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        self.packed_scales_loader(param, loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        quantization: str | None = None,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias, quantization)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

    def qkv_shards(self):
        q_size = self.num_heads * self.head_size
        kv_size = self.num_kv_heads * self.head_size
        return (
            (0, q_size, 0, q_size * self.tp_size),
            (q_size, kv_size, q_size * self.tp_size, kv_size * self.tp_size),
            (q_size + kv_size, kv_size, (q_size + kv_size) * self.tp_size, kv_size * self.tp_size),
        )

    def packed_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        for dst_offset, dst_size, src_offset, src_size in self.qkv_shards():
            dst = param_data.narrow(self.tp_dim, dst_offset, dst_size)
            src = loaded_weight.narrow(self.tp_dim, src_offset, src_size)
            src = src.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
            dst.copy_(src)

    def packed_scales_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        for dst_offset, dst_size, src_offset, src_size in self.qkv_shards():
            dst = param_data.narrow(0, dst_offset, dst_size)
            src = loaded_weight.narrow(0, src_offset, src_size)
            src = src.chunk(self.tp_size, 0)[self.tp_rank]
            dst.copy_(src)

    def bias_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str | None = None):
        if loaded_shard_id is None:
            self.packed_scales_loader(param, loaded_weight)
        else:
            self.weight_loader(param, loaded_weight, loaded_shard_id)

    def qweight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        self.packed_weight_loader(param, loaded_weight)

    def scales_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        self.packed_scales_loader(param, loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quantization: str | None = None,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1, quantization)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def scales_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant_method is not None:
            y = self.quant_method.apply(x, self, self.bias if self.tp_rank == 0 else None)
        else:
            y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
