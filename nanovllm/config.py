import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str#模型所在的本地文件夹路径（必须真实存在）。这是唯一一个没有默认值、必须填写的参数。
    hf_config: AutoConfig | None = None#统会自动读取并保存在这里的 Hugging Face 官方模型配置对象。
    eos: int = -1#提前留好位置，等 Tokenizer 上岗后把“句号”的代码填进来。

    max_num_batched_tokens: int = 16384#每次流水线跳动（step）时，所有任务加起来最多能处理的 Token 总数。
    max_num_seqs: int = 512#系统同时最多能接待多少个并发订单。
    max_model_len: int = 4096#单条极限长度，规定 max_num_batched_tokens >= max_model_len，以及这个长度不会超过模型天生支持的最大长度限制。
    
    gpu_memory_utilization: float = 0.9#显存占用率。告诉引擎：“你可以霸占这张显卡 90% 的内存来干活，剩下的留给系统”。
    tensor_parallel_size: int = 1#张量并行的规模，也就是**“雇佣几张显卡一起干活”
    enforce_eager: bool = False#是否强制使用“急切模式”。如果设为 True，通常意味着关掉那些高级的 CUDA 计算图优化，方便调试查错。

    kvcache_block_size: int = 256#把显存切成一块一块的“小格子”（PagedAttention 机制），每个格子能存 256 个 Token 的记忆。
    num_kvcache_blocks: int = -1#总共有多少个显存小格子（通常会在启动后根据剩余显存动态计算，所以初始设为 -1）。

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
