import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    hf_config: AutoConfig | None = None
    eos: int = -1
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    quantization: str | None = None
    quantized_model_path: str | None = None
    quant_group_size: int = 128
    quantize_lm_head: bool = False

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        if self.quantization not in (None, "int8", "w8a8", "int4_awq"):
            raise ValueError(f"Unsupported quantization mode: {self.quantization}")
        if self.quantization == "int4_awq":
            raise NotImplementedError("int4_awq quantization is not implemented yet")
        if self.quantize_lm_head:
            raise NotImplementedError("quantize_lm_head is not implemented for the first INT8 phase")
        if self.quantized_model_path is not None:
            assert os.path.isdir(self.quantized_model_path)
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
