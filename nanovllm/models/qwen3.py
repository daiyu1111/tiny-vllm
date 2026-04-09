import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,#表示输入隐藏状态的总维度，也就是每个 token 的 embedding 维度。它决定了 qkv_proj 的输入维度
        num_heads: int,#Query 头的总数，代码先保存为 self.total_num_heads，再按张量并行切到每张卡自己的 self.num_heads
        num_kv_heads: int,#Key/Value 头的总数。它可以小于 num_heads，这就是 GQA/MQA 的思路。这里同样会按 tensor parallel 切成本卡的 self.num_kv_heads，
        max_position: int = 4096 * 32,#RoPE 预计算的最大位置长度。它传给 get_rope(...) 用来生成 cos/sin cache
        head_dim: int | None = None,#每个 attention head 的维度。如果不传，就自动用 hidden_size // num_heads
        rms_norm_eps: float = 1e-06,#RMSNorm 的数值稳定项 eps。这里只在 qkv_bias=False 时给 q_norm 和 k_norm 用
        qkv_bias: bool = False,#控制 Q/K/V 线性投影是否带 bias。如果 qkv_bias=False，代码会额外对 q 和 k 做 RMSNorm；如果 qkv_bias=True，这两个 norm 就不会建，也不会执行。所以它不只是“线性层要不要 bias”，还顺带决定了是否启用 q_norm/k_norm。
        rope_theta: float = 10000,#RoPE 的底数 base
        rope_scaling: dict | None = None,#预留给 RoPE scaling 的配置。当前实现里它主要是为了接口兼容和 cache key，真正的缩放逻辑还没用上
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)#hidden_states 经 qkv_proj 生成拼在一起的 qkv
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)#按 q_size 和 kv_size 切成 q, k, v
        q = q.view(-1, self.num_heads, self.head_dim)#按 num_heads / num_kv_heads / head_dim reshape
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:#如果 qkv_bias=False，对 q/k 做 RMSNorm
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)#用 positions 做 RoPE
        o = self.attn(q, k, v)#交给 Attention(num_heads, head_dim, scaling, num_kv_heads) 做真正注意力计算，见 attention.py (line 45)
        output = self.o_proj(o.flatten(1, -1))#最后 o_proj 投回 hidden_size
        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()#融合算子

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            #第一层还没有残差输入，将输入做一次norm,并把输入存入残差
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            #具体看inputnorm的设置，这应该是个融合算子，把hidden_states+residual 和RMSNorm(hidden_states+residual)两个操作合并
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        #如果把整个 Embedding 矩阵放在一张显卡上，会吃掉大量显存。这里的“Parallel”意味着词表被按行切分到了多张 GPU 上。每张卡只负责查找自己那部分词汇的向量，最后再通过通信合并。这是典型的大模型张量并行（Tensor Parallelism）策略。
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None#
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)#融合算子，相加和norm计算合并一起
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    """模块融合映射 (packed_modules_mapping)：
        这是一个非常典型的推理优化技巧。它将原本独立的 Q、K、V 投影层打包成一个 qkv_proj，将 MLP 层中的 Gate 和 Up 投影层打包成 gate_up_proj。
        这不仅减少了模型加载时的碎片化，更重要的是在推理阶段可以通过一次矩阵乘法完成多个投影操作，大幅降低内存带宽占用和算子调度开销。"""

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:#权值共享，输入层的 Embedding 矩阵和输出层的 LM Head 矩阵将共享同一块物理显存
            #因为它们的作用互为逆操作（Token -> 向量，向量 -> Token 概率），共享权重不仅能大幅节省显存，有时还能提升模型的泛化能力。
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
