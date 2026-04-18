# Nano-vLLM 量化设计文档

## 概述

本文档用于为当前 Nano-vLLM 项目定义一条清晰、可执行、适合现有架构的量化演进路线。目标不是立即把所有量化能力一次性做完，而是先选出最适合当前仓库的第一阶段方案，并把后续扩展边界、实现路径和验收标准提前约定清楚。

结论分为三层优先级：

1. 第一优先级：在线性层引入 `W8A16`，即 INT8 weight-only 量化，激活仍保留 `fp16/bf16`。
2. 第二优先级：在第一阶段抽象稳定后，继续扩展 `W4A16`，即 AWQ/GPTQ 风格的 group-wise INT4 weight-only 量化。
3. 暂不建议第一批实现：KV cache 量化、激活量化、Embedding/LM Head 量化。

## 为什么当前项目适合先做权重量化

当前仓库的实现结构决定了，最稳妥的切入点不是整条数值链路重构，而是优先替换 `Linear` 权重表示。

### 1. 权重加载入口集中，适合统一接入量化格式

当前权重加载主要集中在 [nanovllm/utils/loader.py](/E:/llmegine/nano-vllm-main/nano-vllm-main/nanovllm/utils/loader.py)。它已经负责：

- 遍历模型目录下的 `*.safetensors`
- 根据参数名找到模块参数
- 对 packed module 调用自定义 `weight_loader`

这意味着量化权重的接入不需要把逻辑散落到整个模型里，而是可以在同一套加载流程中扩展：

- 原始 fp16/bf16 权重加载
- 量化权重及其附属张量加载
- 按 Tensor Parallel 分片后的量化权重装配

### 2. 主要 GEMM 都收敛在线性层，便于封装量化逻辑

当前项目中的大部分矩阵乘法都封装在 [nanovllm/layers/linear.py](/E:/llmegine/nano-vllm-main/nano-vllm-main/nanovllm/layers/linear.py) 中的几类层：

- `ReplicatedLinear`
- `ColumnParallelLinear`
- `MergedColumnParallelLinear`
- `QKVParallelLinear`
- `RowParallelLinear`

如果第一阶段只针对这些层做 weight-only quantization，那么改动点会高度集中。这样可以把量化的参数创建、权重加载、前向计算统一收口在 `Linear` 抽象层，不必先拆动模型定义本身。

### 3. Attention 与 KV cache 绑定较深，不适合作为首批量化目标

当前 attention 路径依赖 [nanovllm/layers/attention.py](/E:/llmegine/nano-vllm-main/nano-vllm-main/nanovllm/layers/attention.py) 中的以下机制：

- `flash_attn_varlen_func`
- `flash_attn_with_kvcache`
- Triton kernel `store_kvcache_kernel`
- 现有 `slot_mapping` 与 block table 的缓存布局

一旦对 KV cache 做量化，就不再只是“把权重存成更小格式”这么简单，而是会直接牵动：

- `k/v` 写入缓存时是否量化
- 读取缓存参与 attention 前是否反量化
- flash-attn 当前调用约定是否还能继续复用
- 现有 KV cache 内存布局和类型假设是否需要修改

这条路线明显比线性层权重量化更重、更难验证，也更容易影响 prefix cache、decode 路径和 CUDA graph 稳定性。

### 4. 当前运行时按 `hf_config.torch_dtype` 在 CUDA 上建模，适合优先替换权重表示

从 [nanovllm/engine/model_runner.py](/E:/llmegine/nano-vllm-main/nano-vllm-main/nanovllm/engine/model_runner.py) 可以看到，模型初始化时会：

- 将默认 dtype 切换为 `hf_config.torch_dtype`
- 将默认 device 切换到 CUDA
- 直接实例化模型并加载权重

这说明当前项目的默认路径是假定模型计算仍运行在标准浮点 dtype 上。对这样的实现来说，最合适的第一步是：

- 保持激活和主干计算 dtype 不变
- 只把线性层权重换成量化表示
- 在前向里按需反量化或走量化 GEMM

相比之下，如果第一阶段就引入动态激活量化、FP8 或 KV cache 量化，就会超出当前运行时设计的自然边界。

## 推荐的量化路线

### 第一阶段推荐方案：`W8A16`

当前仓库最适合先做 `W8A16`，也就是 INT8 weight-only 量化，激活保留 `fp16/bf16`。

推荐理由如下：

- 改动主要集中在 `Linear` 和 `loader`，不需要先重写 attention kernel。
- 精度风险相对可控，适合这个以可读性、轻量实现和教学价值为主的仓库。
- 对 `Qwen3-0.6B` 这类小模型，线性层权重压缩可以较稳定降低模型常驻显存。
- 它与当前 Tensor Parallel 分片方式天然兼容，权重仍然可以按 shard 加载，只是每个 shard 附带量化元数据。

对于 Nano-vLLM 这类实现，第一阶段最重要的是先把“量化模型可加载、可生成、可验证”打通，而不是立刻追求最极致的吞吐。

### 第二阶段扩展方案：`W4A16`

`W4A16` 适合在第一阶段稳定后继续引入，建议采用 AWQ/GPTQ 风格的 group-wise INT4 weight-only 方案。

选择它作为第二阶段而不是第一阶段的原因：

- 它通常能带来更大的显存收益。
- 但它需要额外设计 group size、scale、zero-point、打包格式、反量化路径。
- 如果首版就直接做 INT4，调试复杂度会显著上升。
- 需要更明确地约定离线量化产物格式，否则加载器和线性层接口容易反复修改。

因此，更合理的顺序是：

1. 先用 `W8A16` 验证整体抽象是否成立。
2. 再在同一抽象上扩展 `W4A16`。

### 首批不建议做：KV cache quantization

当前项目不适合在第一批中引入 KV cache quantization。

原因很明确：

- KV cache 由 [nanovllm/layers/attention.py](/E:/llmegine/nano-vllm-main/nano-vllm-main/nanovllm/layers/attention.py) 中的 Triton 写缓存逻辑和 flash-attn 读取逻辑共同使用。
- 一旦量化 KV，就必须补一套反量化路径，甚至可能要改变 attention 调用方式。
- 它会直接扩大改动面，波及 prefill、decode、prefix cache、cache layout 和数值稳定性。

因此，KV cache 量化应被视为后续高风险课题，而不是第一版目标。

## 落地实现方案

### 配置层

建议在 [nanovllm/config.py](/E:/llmegine/nano-vllm-main/nano-vllm-main/nanovllm/config.py) 中新增量化配置字段：

```python
quantization: None | str = None
quantized_model_path: str | None = None
quant_group_size: int = 128
quantize_lm_head: bool = False
```

推荐语义如下：

- `quantization`: 量化模式，取值为 `None | "int8" | "int4_awq"`。
- `quantized_model_path`: 量化权重目录，默认与 `model` 相同；如果不为空，则优先从该目录加载量化产物。
- `quant_group_size`: 供 INT4 group-wise quantization 使用，默认 `128`。
- `quantize_lm_head`: 是否量化 `lm_head`，第一阶段默认 `False`。

默认策略：

- `quantization=None` 时，行为必须与当前主干保持一致。
- 第一版只要求 `int8` 真正可用。
- `int4_awq` 可以先定义接口和文档规范，再在后续迭代中实现。

### 抽象层

建议新增 `nanovllm/quantization/` 模块，作为量化相关逻辑的唯一收口位置。

推荐目录结构如下：

```text
nanovllm/
  quantization/
    __init__.py
    base.py
    int8.py
    int4_awq.py
    utils.py
```

核心抽象建议定义一个统一接口，例如 `QuantMethod`，职责包括：

- 创建量化参数张量
- 加载量化权重及其辅助张量
- 执行量化版线性计算

建议接口关注以下能力：

```python
class QuantMethod:
    def create_weights(self, layer, input_size, output_size, bias, tp_dim): ...
    def load_weights(self, param_dict, loaded_tensors, shard_info): ...
    def apply(self, x, layer): ...
```

这里的关键不是接口名字必须完全一致，而是要确保：

- `Linear` 层不直接知道某种量化格式的全部细节
- `loader` 不直接知道每种量化 kernel 的细节
- 后续新增 `int4_awq` 时，不需要再重写一遍线性层体系

### 线性层接入方式

第一阶段只把量化接入到 [nanovllm/layers/linear.py](/E:/llmegine/nano-vllm-main/nano-vllm-main/nanovllm/layers/linear.py) 中的线性层：

- `ReplicatedLinear`
- `ColumnParallelLinear`
- `MergedColumnParallelLinear`
- `QKVParallelLinear`
- `RowParallelLinear`

接入原则如下：

- 保持现有类结构不变，尽量避免为了量化重写整套线性层。
- 在线性层基类中增加“是否启用量化”和“当前量化方法”的分派逻辑。
- 第一版只量化这些线性层的权重。
- `RMSNorm`、`Embedding`、`LM Head` 保持原 dtype，不纳入第一阶段范围。

推荐思路：

- `LinearBase` 负责保存量化配置、量化方法对象和参数张量。
- 各子类继续负责 TP 维度切分与分片加载。
- 真正的量化前向计算由 `QuantMethod.apply(...)` 实现。

这样做的好处是：

- 不会破坏当前 TP 分片语义。
- packed module 如 `qkv_proj`、`gate_up_proj` 仍可沿用现有映射方式。
- 量化与非量化路径可以共存，方便逐步验证。

### 权重加载格式与命名约定

建议扩展 [nanovllm/utils/loader.py](/E:/llmegine/nano-vllm-main/nano-vllm-main/nanovllm/utils/loader.py)，使其同时支持：

- 原始 fp16/bf16 `safetensors`
- 量化后的 `safetensors`

量化产物建议遵循统一命名规范：

- 权重主体：`*.qweight`
- scale：`*.scales`
- zero-point：`*.qzeros`

如果采用对称量化，`qzeros` 可以省略，但文档和代码都应允许这类字段不存在。

推荐示例：

```text
model.layers.0.self_attn.qkv_proj.qweight
model.layers.0.self_attn.qkv_proj.scales
model.layers.0.self_attn.qkv_proj.qzeros
```

需要特别注意与当前 packed module 的兼容：

- 现有模型中 `q_proj/k_proj/v_proj` 会映射到 `qkv_proj`
- `gate_proj/up_proj` 会映射到 `gate_up_proj`

因此量化权重加载时，要么：

- 沿用现有 packed 映射，在 loader 中做名义映射和 shard 分发

要么：

- 在量化产物阶段就生成与当前 packed module 对齐的名字

两种方式都可以，但第一版应选定一种并固定下来，避免后续反复变更。

### 量化产物生成方式

推荐采用“离线量化 + 在线加载”的模式。

这意味着：

- 推理启动时不现场量化原始权重
- 预先将模型导出为量化后的 `safetensors`
- 运行时只负责读取量化产物并完成前向

建议补充一个独立脚本，例如：

```text
scripts/quantize.py
```

其职责是：

- 读取原始 `safetensors`
- 逐层执行量化
- 输出 `qweight/scales/qzeros`
- 生成与 Nano-vLLM 加载器兼容的量化模型目录

第一版脚本只需要支持 `int8` 即可；`int4_awq` 可在后续扩展时补全。

### 第一阶段运行时策略

第一阶段的运行时策略应明确分阶段推进，而不是一开始就追求极致性能。

建议顺序如下：

1. 先用“反量化到运行 dtype 后再 GEMM”的朴素路径打通功能正确性。
2. 确认模型可加载、可生成、TP 可运行、数值质量在可接受范围内。
3. 若后续仍需提高吞吐，再补 Triton/int8 GEMM kernel 或更高效的 fused kernel。

也就是说，第一阶段的重点是：

- 先证明量化抽象成立
- 再证明数值与功能可接受
- 最后再做性能优化

这条顺序更符合当前仓库“轻量、可读、逐步演进”的定位。

## 第一版明确不做的事情

为了避免范围失控，第一版量化设计应明确排除以下内容：

- 不修改 [nanovllm/layers/attention.py](/E:/llmegine/nano-vllm-main/nano-vllm-main/nanovllm/layers/attention.py) 中 flash-attn 的调用协议。
- 不修改 KV cache 的内存布局。
- 不修改 `store_kvcache` 的写入逻辑。
- 不把量化扩展到采样器、调度器、块管理器或 CUDA graph 捕获流程以外的模块。
- 不在第一版引入 FP8。
- 不在第一版引入动态激活量化。
- 不在第一版引入 mixed per-layer policy。
- 不默认量化 `Embedding` 和 `LM Head`。

这些边界是为了保证第一阶段能聚焦于“线性层权重量化”本身，而不是把整个推理引擎一起改动。

## 验收标准与测试方案

### 功能正确性

至少覆盖以下检查项：

- `quantization=None` 时，行为与当前主干一致。
- 启用 `int8` 后，单卡路径可以成功完成 `generate`。
- 开启 Tensor Parallel 时，量化权重仍能按 shard 正确加载。
- packed module 路径下的 `qkv_proj` 和 `gate_up_proj` 能正确读取量化权重。

### 数值质量

至少覆盖以下场景：

- 固定 prompt 下，INT8 输出与原模型结果在语义上接近。
- 对一组基准 prompt，比对量化前后的 logits 偏差或 top-k 排序偏差。
- 不要求逐 token 完全一致，但要避免明显异常退化，例如高频乱码、重复、提前崩坏。

### 资源收益

至少记录以下指标：

- 量化前后的模型常驻显存变化。
- prefill 吞吐变化。
- decode 吞吐变化。

第一版允许小幅性能回退，因为可能会先走“反量化后再 GEMM”的朴素路径；但不应出现明显不可接受的退化。

### 回归风险覆盖

至少验证以下组合：

- `enforce_eager=True`
- `enforce_eager=False`
- prefix cache 命中场景
- CUDA graph decode 场景
- `tensor_parallel_size=1`
- `tensor_parallel_size>1`

如果第一版暂时无法完整覆盖所有组合，至少应在文档和实现说明中明确列出未覆盖项。

## 建议的实施顺序

为了减少返工，建议按以下顺序推进：

1. 在 `config.py` 中补齐量化配置字段。
2. 新增 `nanovllm/quantization/` 抽象层。
3. 改造 `linear.py`，使线性层支持可插拔量化。
4. 扩展 `loader.py`，让其支持量化权重格式。
5. 增加 `scripts/quantize.py`，形成离线量化产物生成链路。
6. 先打通 `int8` 单卡功能。
7. 再验证 Tensor Parallel、prefix cache 和 CUDA graph。
8. 最后再评估是否需要补高性能 kernel。

## 结论

对于当前 Nano-vLLM 项目，最合适的量化切入点不是 KV cache，也不是激活量化，而是线性层的 weight-only quantization。

建议明确采用以下路线：

- 第一阶段先做 `W8A16`
- 第二阶段扩展 `W4A16`
- 把 KV cache quantization 视为后续高风险优化课题

这样既能贴合当前项目的代码结构，也能在控制复杂度的前提下，为后续真正实现量化功能铺平接口和演进路径。

## INT8 Weight-Only 量化落地方案

本节对第一阶段 `W8A16` 方案做更具体的落地约定。这里的 `W8A16` 表示：

- Weight 使用 INT8 保存。
- Activation 仍使用模型原始运行 dtype。
- 对当前 `Qwen3-0.6B` 模型来说，`config.json` 中的 `torch_dtype` 是 `bfloat16`，因此 A16 实际是 bf16。
- KV cache、RMSNorm、Embedding、LM Head、采样器仍保持当前实现，不纳入第一阶段量化范围。

### 量化哪些层

第一阶段只量化 `Qwen3DecoderLayer` 内 Transformer 主干路径上的线性层，也就是当前模型中计算量和权重占比最高的 GEMM 层：

- Attention 中的 `qkv_proj`
- Attention 中的 `o_proj`
- MLP 中的 `gate_up_proj`
- MLP 中的 `down_proj`

对应到当前代码结构，这些层由 `nanovllm/layers/linear.py` 中的以下线性层实现承载：

- `QKVParallelLinear`
- `MergedColumnParallelLinear`
- `RowParallelLinear`
- 后续如果模型结构中出现普通 `ColumnParallelLinear` 或 `ReplicatedLinear`，也可以沿用同一套 INT8 weight-only 逻辑。

第一阶段明确不量化以下模块：

- `embed_tokens`
- `lm_head`
- `RMSNorm`
- RoPE
- Attention 的 `q/k/v` 激活
- KV cache
- `store_kvcache` Triton kernel
- flash-attn 调用路径
- sampler 和 logits 后处理逻辑

这样可以把改动边界限制在线性层权重表示和加载逻辑内，不改变 attention、调度器、KV cache、CUDA graph 的核心语义。

### 怎么量化

第一阶段采用对称 INT8 weight-only 量化，不使用 zero-point。

推荐粒度是 per-output-channel，也就是对每一行权重单独计算一个 scale。对于线性层权重 `weight`，其形状为：

```text
[out_features, in_features]
```

对第 `i` 个输出通道：

```python
weight_row = weight[i]
scale[i] = max(abs(weight_row)) / 127
qweight[i] = round(weight_row / scale[i]).clamp(-127, 127).to(torch.int8)
```

需要处理全零行，避免 `scale=0`：

```python
scale[i] = max(max(abs(weight_row)) / 127, eps)
```

其中 `eps` 可以取 `1e-8` 或与实现 dtype 匹配的安全下限。

推理时第一版不引入 INT8 GEMM kernel，而是使用最朴素、最容易验证的路径：

```python
dequant_weight = qweight.to(runtime_dtype) * scales[:, None].to(runtime_dtype)
output = F.linear(x, dequant_weight, bias)
```

这意味着：

- 权重常驻形式是 INT8，减少模型权重显存。
- 计算前临时反量化到 bf16/fp16。
- 激活 `x` 保持原始运行 dtype。
- 第一版性能目标不是极致吞吐，而是先打通可加载、可生成、可验证的 INT8 weight-only 路径。

后续如果需要进一步优化性能，再把 `dequantize + GEMM` 替换为 Triton fused dequant GEMM 或专用 INT8 GEMM kernel。

### 离线量化还是在线量化

第一阶段选择离线量化，不在推理启动时现场量化原始权重。

推荐流程是：

1. 使用独立脚本读取原始 bf16/fp16 `safetensors`。
2. 对目标线性层执行 INT8 per-channel 对称量化。
3. 输出 Nano-vLLM 运行时可直接加载的量化产物。
4. 推理启动时只加载 `qweight` 和 `scales`，不再依赖原始浮点权重现场量化。

在线量化只作为调试或实验手段，不作为正式推理加载路径。原因是：

- 在线量化会增加启动时间。
- 在线量化需要同时读取原始浮点权重和量化后权重，中间峰值显存/内存更高。
- 离线量化产物更容易复现、分发和做数值回归。

建议新增独立脚本：

```text
scripts/quantize.py
```

脚本职责：

- 读取原始模型目录。
- 识别需要量化的线性层。
- 处理 `q_proj/k_proj/v_proj` 到 `qkv_proj` 的 packed 映射。
- 处理 `gate_proj/up_proj` 到 `gate_up_proj` 的 packed 映射。
- 生成量化后的 `safetensors`。
- 复制或生成必要的 `config.json`、`tokenizer` 文件，使量化目录可以作为独立模型目录加载。

### 量化产物格式

第一阶段建议采用以下命名：

```text
<param_name>.qweight
<param_name>.scales
```

例如：

```text
model.layers.0.self_attn.qkv_proj.qweight
model.layers.0.self_attn.qkv_proj.scales
model.layers.0.self_attn.o_proj.qweight
model.layers.0.self_attn.o_proj.scales
model.layers.0.mlp.gate_up_proj.qweight
model.layers.0.mlp.gate_up_proj.scales
model.layers.0.mlp.down_proj.qweight
model.layers.0.mlp.down_proj.scales
```

字段约定：

- `qweight`: `torch.int8`，形状与对应浮点权重一致。
- `scales`: 建议保存为 `torch.float32`，形状为 `[out_features]`。
- 第一阶段使用对称量化，不保存 `qzeros`。

packed 层建议在离线量化阶段直接生成 packed 后的权重：

- 原始 `q_proj/k_proj/v_proj` 先按当前运行时的 `qkv_proj` 语义拼接，再量化为 `qkv_proj.qweight` 和 `qkv_proj.scales`。
- 原始 `gate_proj/up_proj` 先按当前运行时的 `gate_up_proj` 语义拼接，再量化为 `gate_up_proj.qweight` 和 `gate_up_proj.scales`。

这样运行时 loader 不需要再对多个浮点权重分片做临时拼接，量化权重的命名也能直接对齐 Nano-vLLM 当前模块结构。

### Tensor Parallel 下的处理

第一版推荐保持当前 Tensor Parallel 的切分语义：

- `QKVParallelLinear` 和 `MergedColumnParallelLinear` 仍按输出维度切分。
- `RowParallelLinear` 仍按输入维度切分。
- 切分后每个 rank 加载自己的 `qweight` shard 和对应 `scales`。

对 per-output-channel scale 来说：

- 输出维度切分时，`scales` 跟随输出通道一起切分。
- 输入维度切分时，输出通道不变，`scales` 不需要按输入维度切分。

因此离线产物可以保存完整 packed 权重，由运行时沿用当前 shard 逻辑切分；也可以由离线脚本直接输出 rank-local shard。第一阶段建议先保存完整 packed 权重，复用当前 loader 的 TP 切分习惯，减少产物数量和目录复杂度。

### 验收标准

文档与实现需要满足以下最低验收标准：

- 明确写出量化层范围和非量化层范围。
- 明确采用 INT8 对称 per-output-channel weight-only 量化。
- 明确第一阶段使用离线量化，不把在线量化作为正式路径。
- 明确第一版运行时使用 `dequantize + F.linear`。
- `quantization=None` 时行为与当前主干保持一致。
- INT8 单卡路径可以完成 `example.py` 推理。
- 固定 prompt 下，INT8 输出与原始 bf16 模型语义接近。
- 记录量化前后的模型常驻显存变化。
- Tensor Parallel、prefix cache、CUDA graph 的覆盖情况需要在实现说明中明确标记。

### 第一阶段不追求的目标

为了控制复杂度，第一阶段不追求以下目标：

- 不实现 activation quantization。
- 不实现 KV cache quantization。
- 不实现 FP8。
- 不实现 INT4。
- 不实现 mixed per-layer policy。
- 不实现高性能 fused INT8 GEMM。
- 不修改 flash-attn 接口。
- 不修改 `store_kvcache` kernel。

第一阶段的核心目标是把 INT8 weight-only 的模型格式、加载路径、线性层抽象和基础数值验证打通。性能优化应放在该路径稳定之后再做。

## 第一版 INT8 Weight-Only 实现总结

本节记录第一版 `W8A16` INT8 weight-only 量化的实际落地情况。它是对前文设计方案的实现复盘，不替代前面的方案说明。

### 实现了哪些修改

第一版实现围绕“离线量化、运行时加载 INT8 权重、前向时反量化后执行 `F.linear`”展开，改动集中在配置、量化抽象、线性层、权重加载、离线导出和 benchmark 几个位置。

1. 配置层

   在 `nanovllm/config.py` 中新增量化相关配置：

   ```python
   quantization: str | None = None
   quantized_model_path: str | None = None
   quant_group_size: int = 128
   quantize_lm_head: bool = False
   ```

   当前第一版只真正支持 `quantization="int8"`。`int4_awq` 和 `quantize_lm_head=True` 会显式抛出 `NotImplementedError`，避免用户误以为这些路径已经可用。

2. 量化抽象层

   新增 `nanovllm/quantization/`：

   - `base.py` 定义 `QuantMethod` 抽象接口。
   - `int8.py` 实现 `Int8WeightOnlyQuantMethod`。
   - `utils.py` 提供 per-output-channel 对称 INT8 量化函数。

   INT8 权重格式为：

   ```text
   qweight: torch.int8
   scales: torch.float32
   ```

   推理时执行：

   ```python
   weight = qweight.to(x.dtype) * scales.to(x.dtype).unsqueeze(1)
   output = F.linear(x, weight, bias)
   ```

3. 线性层

   在 `nanovllm/layers/linear.py` 中让 `LinearBase` 同时支持两种权重表示：

   - 非量化路径：继续使用原来的浮点 `weight`。
   - INT8 路径：创建 `qweight` 和 `scales`，并将它们注册为 `requires_grad=False` 的参数，便于现有 loader 按参数名加载。

   已接入的线性层包括：

   - `QKVParallelLinear`
   - `MergedColumnParallelLinear`
   - `RowParallelLinear`
   - `ColumnParallelLinear`
   - `ReplicatedLinear`

   对 packed 层做了专门处理：

   - `qkv_proj` 在 Tensor Parallel 下按 `q/k/v` 三段分别切分，再装入 rank-local packed 参数。
   - `gate_up_proj` 在 Tensor Parallel 下按 `gate/up` 两段分别切分，再装入 rank-local packed 参数。
   - `RowParallelLinear` 的 `qweight` 按输入维度切分，`scales` 不按输入维度切分。

4. Qwen3 模型层

   在 `nanovllm/models/qwen3.py` 中通过 `hf_config.nanovllm_quantization` 将量化模式传给 Transformer 主干线性层。

   第一版只量化以下层：

   - `self_attn.qkv_proj`
   - `self_attn.o_proj`
   - `mlp.gate_up_proj`
   - `mlp.down_proj`

   以下模块保持原始浮点实现：

   - `embed_tokens`
   - `lm_head`
   - `RMSNorm`
   - RoPE
   - Attention kernel
   - KV cache
   - sampler

5. 权重加载器

   在 `nanovllm/utils/loader.py` 中扩展加载逻辑：

   - 优先按 safetensors 中的参数名直接加载，例如 `model.layers.0.mlp.gate_up_proj.qweight`。
   - 如果模型中找不到该参数，再回退到原来的 HF 权重 packed mapping，例如 `q_proj -> qkv_proj`、`up_proj -> gate_up_proj`。
   - `quantization="int8"` 时会检查所有量化线性层的 `.qweight` 和 `.scales` 是否都已加载，缺失时直接报清晰错误。

   这个顺序很重要，因为 INT8 产物已经是 packed 后的名字。如果仍然先走旧 mapping，`gate_up_proj.qweight` 中的 `up_proj` 会被误替换，形成错误的 `gate_gate_up_proj.qweight`。

6. 离线量化脚本

   新增 `scripts/quantize.py`，用于从原始 Hugging Face safetensors 生成 Nano-vLLM 可直接加载的 INT8 目录。

   脚本做的事情包括：

   - 读取原始 `*.safetensors`。
   - 将 `q_proj/k_proj/v_proj.weight` 拼成 `qkv_proj` 后量化。
   - 将 `gate_proj/up_proj.weight` 拼成 `gate_up_proj` 后量化。
   - 将 `o_proj.weight`、`down_proj.weight` 直接量化。
   - 非目标张量原样复制。
   - 复制 tokenizer/config 等非权重文件，让量化目录可以独立保存和分发。

7. Benchmark 脚本

   新增 `bench_quant.py`，用于对比原模型和 INT8 模型的：

   - `after_model_load` 显存占用。
   - 总生成 token 数。
   - 总耗时。
   - tokens/s 吞吐。
   - INT8 相对 bf16 的吞吐比例。

### 如何测试

1. 生成 INT8 量化产物

   ```bash
   python scripts/quantize.py \
     --model Qwen3-0.6B/qwen/Qwen3-0___6B \
     --output Qwen3-0.6B/qwen/Qwen3-0___6B-int8 \
     --quantization int8
   ```

   成功时会输出：

   ```text
   Wrote INT8 weight-only model to Qwen3-0.6B/qwen/Qwen3-0___6B-int8
   ```

2. 单条生成 smoke test

   ```python
   from nanovllm import LLM, SamplingParams
   from transformers import AutoTokenizer

   base = "Qwen3-0.6B/qwen/Qwen3-0___6B"
   int8 = "Qwen3-0.6B/qwen/Qwen3-0___6B-int8"

   tokenizer = AutoTokenizer.from_pretrained(base)
   llm = LLM(
       base,
       quantization="int8",
       quantized_model_path=int8,
       enforce_eager=True,
       tensor_parallel_size=1,
   )

   prompt = tokenizer.apply_chat_template(
       [{"role": "user", "content": "introduce yourself"}],
       tokenize=False,
       add_generation_prompt=True,
       enable_thinking=False,
   )

   outputs = llm.generate([prompt], SamplingParams(temperature=0.6, max_tokens=128))
   print(outputs[0]["text"])
   ```

   该测试用于确认 INT8 模型可以加载、前向、采样并输出正常文本。

3. 快速 benchmark

   ```bash
   python bench_quant.py \
     --mode both \
     --num-seqs 16 \
     --max-input-len 512 \
     --max-output-len 256 \
     --enforce-eager
   ```

   该命令适合先做快速验证，避免一上来跑完整压力测试。

4. 完整 benchmark

   ```bash
   python bench_quant.py --mode both
   ```

   该命令会先跑原模型，再跑 INT8 模型，并打印显存和吞吐对比。

### 当前实测结果

在 `Qwen3-0.6B` 上，使用 `bench_quant.py --mode both` 的一次实测结果如下：

```text
bf16 memory after load: allocated=1.13GiB, reserved=1.48GiB, max_allocated=1.42GiB
bf16 total: 133966tok, time=38.15s, throughput=3511.81tok/s
int8 memory after load: allocated=0.74GiB, reserved=1.05GiB, max_allocated=20.94GiB
int8 total: 133966tok, time=48.28s, throughput=2774.69tok/s
int8/bf16 throughput ratio: 0.790x
```

显存结论：

- `memory_allocated` 从 `1.13GiB` 降到 `0.74GiB`，下降约 `34.5%`。
- `memory_reserved` 从 `1.48GiB` 降到 `1.05GiB`，下降约 `29%`。
- 这说明 INT8 weight-only 已经有效降低模型加载后的常驻显存。

性能结论：

- INT8 吞吐约为 bf16 的 `0.790x`。
- 这是第一版朴素实现的预期结果，不代表量化失败。
- 原因是当前前向仍然会在每次线性层计算前执行 `qweight -> runtime dtype` 的反量化，然后再调用 `F.linear`，没有使用 fused INT8 GEMM kernel。

关于 `max_allocated=20.94GiB`：

- `max_allocated` 是进程历史峰值，不等同于模型加载后的常驻显存。
- `bench_quant.py --mode both` 会在同一进程中先跑 bf16 再跑 INT8，历史峰值可能包含 warmup、CUDA graph、benchmark 中的临时峰值。
- 判断模型常驻显存收益时，应优先看 `memory_allocated` 和 `memory_reserved`。

### 量化质量评估落地方案

当前 INT8 benchmark 只验证了量化模型可以加载、可以生成、可以降低模型常驻显存，并记录了吞吐变化。它还不能证明 INT8 模型与原始 bf16 模型在 logits 分布、困惑度和生成语义上足够接近。

因此，下一步应先建立质量基线，再继续优化 fused kernel 计算路径。质量评估需要从“能跑一次”升级为“可复现、可比较、可作为回归门禁”的脚本化流程。建议新增独立脚本：

```text
bench_quant_quality.py
```

第一版脚本建议支持：

```bash
python bench_quant_quality.py \
  --model Qwen3-0.6B/qwen/Qwen3-0___6B \
  --int8-model Qwen3-0.6B/qwen/Qwen3-0___6B-int8 \
  --mode all \
  --output quant_quality_report.json
```

输出建议固定为 JSON，至少包含：

- `metadata`：模型路径、量化模型路径、dtype、设备、TP 大小、commit 或运行时间。
- `artifact_check`：量化产物格式、shape、dtype 和反量化重建误差。
- `logits_check`：logits 相似度、误差和 top-k 一致性。
- `ppl_check`：loss / ppl 对比。
- `generation_check`：固定 prompt 的生成对比和异常标记。

推荐按以下顺序分层验收：

1. P-1：量化产物一致性检查

   在加载完整引擎前，先检查离线量化产物本身是否符合格式预期。这一层成本最低，也最容易定位 `scripts/quantize.py` 或 packed 权重生成错误。

   建议检查：

   - 所有目标层都存在 `.qweight` 和 `.scales`。
   - `.qweight` dtype 为 `torch.int8`。
   - `.scales` dtype 为 `torch.float32` 或可安全转换到运行 dtype。
   - `.qweight.shape` 与对应浮点权重 shape 一致。
   - `.scales.shape == [out_features]`。
   - `qkv_proj` 的 q/k/v 拼接顺序与运行时 `QKVParallelLinear` 一致。
   - `gate_up_proj` 的 gate/up 拼接顺序与运行时 `MergedColumnParallelLinear` 一致。
   - 反量化权重与原始浮点权重的 per-layer MAE、max error、cosine similarity。

   这一层应作为硬门禁。任何目标层缺失、shape 不一致、dtype 不一致、scale 维度错误，都应直接失败。

2. P0：快速 logits 错误验证

   固定少量 prompt，对比 bf16 和 INT8 的 logits 分布，快速发现运行时加载、TP shard、packed 顺序或反量化计算错误。

   建议记录：

   - logits cosine similarity 的 mean / min / p05。
   - logits MAE。
   - logits relative L2 error。
   - logits max error，用作诊断指标，不建议单独作为硬门禁。
   - top-1 agreement。
   - top-5 / top-10 overlap。
   - top-1 margin 分桶后的 agreement，避免把原模型本来就不确定的位置误判为量化失败。

   第一版阈值建议先采用“两阶段策略”：

   - 初次落地时 record-only，跑固定 prompt 集，记录当前 bf16 vs INT8 的真实分布。
   - 连续确认结果稳定后，再冻结阈值，作为后续 fused kernel 或 INT4 扩展的回归门禁。

   这一层优先级最高，因为它反馈快，比直接跑 perplexity 更容易定位实现问题。它主要用于发现 `qkv_proj` / `gate_up_proj` packed 顺序、`scales` 切分、`RowParallelLinear` shard、loader 映射等错误。

3. P1：困惑度 / loss 评估

   在 P0 通过后，再使用更标准的语言建模指标评估量化质量。

   第一版推荐先支持本地文本文件或本地 token ids 文件，避免质量评估脚本强依赖联网下载数据集。可以再补一个可选的 `WikiText-2` perplexity 模式，因为它轻量、常见，适合作为量化质量 smoke test。后续可以补充 `C4 validation sample`、中文文本样本和业务 prompt loss 集，用于覆盖更接近实际使用的分布。

   建议记录：

   - bf16 loss / ppl
   - INT8 loss / ppl
   - relative delta
   - 参与评估的 token 数
   - 最大序列长度和 stride

   这一层用于判断 INT8 weight-only 是否造成可接受范围内的语言建模质量退化。

   P1 不建议一开始设过严阈值。第一版可以先要求 loss / ppl delta 进入可解释范围，并把真实结果写入报告；等 P0/P1 的固定数据集稳定后，再定义硬阈值。

4. P2：确定性生成质量对比

   当前 `SamplingParams` 禁止 `temperature=0`，采样器也使用随机采样。因此 P2 不能直接写成“设置 `temperature=0` 后调用 `generate`”。落地时应二选一：

   - 新增 greedy / argmax 采样模式，让 `generate` 可以在确定性路径下运行。
   - 或者在质量评估脚本中绕过 sampler，直接逐步取 `argmax(logits)` 做固定步数回放。

   建议记录：

   - bf16 输出
   - INT8 输出
   - token match rate
   - first diff position
   - 是否提前 EOS
   - 是否出现明显重复
   - 是否出现乱码或空输出
   - 人工可读的 prompt / output 摘要

   这一层不应把 token match rate 作为唯一硬门禁。自回归生成中，早期一个 token 的轻微差异就可能让后续文本完全分叉。更合理的做法是把 token match rate 和 first diff position 作为诊断指标，把乱码、异常重复、提前崩坏、明显语义漂移等作为失败信号。它用于补充 perplexity，观察真实生成路径是否稳定。

5. P3：后续任务 benchmark

   更完整的任务 benchmark 可以作为后续扩展，不作为第一阶段阻塞项。

   可选方向包括：

   - `CMMLU` / `MMLU`
   - `GSM8K`
   - `HumanEval` / `MBPP`
   - 业务 prompt 集

   P3 更适合在 P-1/P0/P1/P2 稳定后作为版本对比指标，而不应阻塞第一版 INT8 weight-only 的基础合入。

完成 P-1/P0/P1/P2 后，再进入 fused dequant GEMM 或专用 INT8 GEMM kernel 优化。性能优化完成后，必须复用同一套质量评估，确认 fused kernel 没有引入额外数值误差。后续扩展 INT4/AWQ 时，也应复用这套报告格式，只新增 INT4 特有的 group size、zero-point、packing 检查项。

### 量化质量评估实现总结

当前仓库已经新增 `bench_quant_quality.py`，用于把上面的质量评估方案落地为可执行、可复现、可保存 JSON 报告的脚本。

本次实现覆盖了 P-1/P0/P1/P2 四层检查：

1. P-1：量化产物检查

   脚本会直接读取原始模型和 INT8 模型目录中的 `*.safetensors`，按 `scripts/quantize.py` 的 packed 规则重建目标浮点权重：

   - `q_proj/k_proj/v_proj -> qkv_proj`
   - `gate_proj/up_proj -> gate_up_proj`
   - `o_proj/down_proj` 直接对应

   然后检查 INT8 产物中的 `.qweight` 和 `.scales` 是否齐全，校验 dtype、shape、scale 维度，并计算反量化权重相对原始浮点权重的：

   - MAE
   - max error
   - relative L2
   - cosine similarity

   这一层是硬检查。目标层缺失、dtype 错误、shape 不一致或 scale 维度错误都会在报告中标记失败。

2. P0：logits 分布对比

   脚本会使用 `scripts/benchmark_cases.py` 中的固定文本 prompt，分别加载 bf16 原模型和 INT8 模型，取每条 prompt 最后位置的 logits 做对比。

   当前记录的指标包括：

   - cosine similarity 的 mean / min / p05
   - MAE
   - relative L2
   - max error
   - top-1 agreement
   - top-5 / top-10 overlap
   - 按 top-1 margin 分桶后的 agreement

   第一版 logits 指标是 record-only。只有运行异常或 bf16 / INT8 logits shape 不一致会被视为失败。

3. P1：loss / perplexity 对比

   脚本支持四种 PPL 数据来源：

   - `hf-dataset`：默认使用 Hugging Face `wikitext / wikitext-2-raw-v1 / test`
   - `builtin`：使用脚本内置短文本，适合离线 smoke test
   - `text-file`：读取本地 UTF-8 文本文件
   - `token-ids-json`：读取本地 token ids JSON 文件

   `datasets` 是可选依赖，只在 `--ppl-source hf-dataset` 时 lazy import。如果环境没有安装，会提示使用：

   ```bash
   pip install datasets
   ```

   PPL 计算采用 causal LM 方式：用 `logits[:-1]` 预测 `tokens[1:]`，并记录 bf16 / INT8 的 loss、ppl、absolute delta、relative delta、参与 token 数和窗口数量。

4. P2：确定性生成对比

   当前没有修改 `SamplingParams`，也没有改变正常 `LLM.generate()` 的随机采样行为。质量脚本内部绕过 sampler，逐步调用 logits 前向并取 `argmax(logits)` 做确定性回放。

   报告会记录：

   - bf16 / INT8 生成 token ids
   - bf16 / INT8 生成文本
   - token match rate
   - first diff position
   - 是否空输出
   - 是否出现连续重复 token
   - 是否以 EOS 结束

为了支持 P0/P1/P2，`nanovllm/engine/model_runner.py` 新增了两个内部评估入口：

- `prefill_last_logits(token_ids_batch)`：返回每条序列最后位置 logits，用于 logits 对比和 argmax 生成。
- `prefill_full_logits(token_ids)`：返回整段序列逐 token logits，用于 loss / ppl 计算。

这两个入口通过 `model_runner.call(...)` 调用，能够复用现有 TP worker 机制；正常推理和 `LLM.generate()` 行为不受影响。

### 量化质量评估使用方式

在运行质量评估前，先确保已经生成 INT8 量化模型目录：

```bash
python scripts/quantize.py \
  --model Qwen3-0.6B/qwen/Qwen3-0___6B \
  --output Qwen3-0.6B/qwen/Qwen3-0___6B-int8 \
  --quantization int8
```

完整质量评估：

```bash
python bench_quant_quality.py \
  --model Qwen3-0.6B/qwen/Qwen3-0___6B \
  --int8-model Qwen3-0.6B/qwen/Qwen3-0___6B-int8 \
  --mode all \
  --output quant_quality_report.json
```

如果要使用默认 WikiText-2 PPL，需要先安装可选依赖：

```bash
pip install datasets
```

只检查量化产物格式和重建误差：

```bash
python bench_quant_quality.py \
  --model Qwen3-0.6B/qwen/Qwen3-0___6B \
  --int8-model Qwen3-0.6B/qwen/Qwen3-0___6B-int8 \
  --mode artifact \
  --output quant_quality_artifact.json
```

只做 logits 对比：

```bash
python bench_quant_quality.py \
  --model Qwen3-0.6B/qwen/Qwen3-0___6B \
  --int8-model Qwen3-0.6B/qwen/Qwen3-0___6B-int8 \
  --mode logits \
  --output quant_quality_logits.json
```

离线 PPL smoke test，不下载数据集：

```bash
python bench_quant_quality.py \
  --model Qwen3-0.6B/qwen/Qwen3-0___6B \
  --int8-model Qwen3-0.6B/qwen/Qwen3-0___6B-int8 \
  --mode ppl \
  --ppl-source builtin \
  --output quant_quality_ppl_builtin.json
```

使用本地文本文件做 PPL：

```bash
python bench_quant_quality.py \
  --model Qwen3-0.6B/qwen/Qwen3-0___6B \
  --int8-model Qwen3-0.6B/qwen/Qwen3-0___6B-int8 \
  --mode ppl \
  --ppl-source text-file \
  --ppl-text-file data/eval_text.txt \
  --output quant_quality_ppl_text.json
```

只做确定性 argmax 生成对比：

```bash
python bench_quant_quality.py \
  --model Qwen3-0.6B/qwen/Qwen3-0___6B \
  --int8-model Qwen3-0.6B/qwen/Qwen3-0___6B-int8 \
  --mode generation \
  --generation-max-tokens 64 \
  --output quant_quality_generation.json
```

多卡 TP smoke test 示例：

```bash
python bench_quant_quality.py \
  --model Qwen3-0.6B/qwen/Qwen3-0___6B \
  --int8-model Qwen3-0.6B/qwen/Qwen3-0___6B-int8 \
  --mode logits \
  --tensor-parallel-size 2 \
  --output quant_quality_tp2_logits.json
```

输出 JSON 顶层字段固定为：

- `metadata`
- `artifact_check`
- `logits_check`
- `ppl_check`
- `generation_check`
- `summary`

其中 `summary.passed` 表示本次已运行阶段的总体状态；第一版数值质量指标仍以 record-only 为主，后续可以在多次稳定运行后冻结阈值，用作 fused kernel 或 INT4/AWQ 扩展的回归门禁。

### 下一步优化方向

1. 先补质量基线，再优化计算路径

   当前最大性能瓶颈是每次前向都显式反量化权重：

   ```text
   qweight -> dequant_weight -> F.linear
   ```

   但在实现 Triton fused dequant GEMM 或专用 INT8 GEMM kernel 之前，应先完成 P0/P1/P2 质量评估，确认当前 INT8 weight-only 路径没有明显量化错误或语义退化。

   质量基线通过后，再将 `dequantize + GEMM` 融合到更高效的 kernel 中，减少临时张量和内存带宽开销。fused kernel 完成后必须重复同一套质量评估，确认性能优化没有引入额外数值误差。

2. 增强 benchmark 统计

   `bench_quant.py` 后续可以进一步拆分统计：

   - prefill throughput
   - decode throughput
   - `after_warmup`
   - `after_kv_cache`
   - `after_cuda_graph`
   - 每个阶段 reset peak 后的峰值显存

   这样可以更清楚地区分“模型权重显存收益”和“运行时临时内存峰值”。

3. 扩展回归覆盖

   当前已验证单卡 INT8 可以生成和 benchmark。后续还需要继续覆盖：

   - `tensor_parallel_size > 1`
   - `enforce_eager=False`
   - prefix cache 命中场景
   - CUDA graph decode 场景

4. 后续再考虑 INT4/AWQ

   第一版已经打通 INT8 weight-only 的模型格式、加载路径、线性层抽象和基本验证。下一阶段可以在同一抽象层上扩展 INT4/AWQ，但不建议同时引入 KV cache quantization 或 activation quantization，以免扩大改动面。

### INT8 吞吐优化具体实现方案

本节给出面向后续编码的具体落地方案，目标是在不改变当前 INT8 weight-only 模型格式和数值语义的前提下，解决现有 `qweight -> dequant_weight -> F.linear` 路径的吞吐瓶颈。推荐路线明确收敛为：

- 使用 CUDA 自定义 kernel 实现 `fused dequant + GEMM`
- 不再把后续优化主线写成 Triton 方案

这里的核心思路是：把 `int8 qweight` 的反量化过程融合进矩阵乘内部，在 CUDA kernel 的 tile 计算过程中按需完成 `int8 -> runtime dtype -> * scales -> accumulate`，而不是先在外部显式构造整块 `dequant_weight` 浮点临时张量。

#### Triton 是做什么的

Triton 是一种用于编写 GPU kernel 的 DSL 和编译工具链，适合快速开发定制矩阵计算、attention、layer norm 等高性能算子。它的定位更接近“用 Python 风格写自定义 GPU kernel”，而不是新的推理框架或量化算法。

但对当前仓库而言，下一步文档不再推荐 Triton 作为主线实现方式，而是直接规划 CUDA 自定义 kernel。原因是：

- 后续目标已经很明确：把反量化融合到 GEMM
- CUDA 实现更接近最终长期维护的执行路径
- 更容易精确控制 tile 布局、shared memory、vectorized load 和累加策略
- 后续若需要接 CUDA graph、进一步贴近 cuBLAS/CUTLASS 风格优化，也更自然

#### 优化目标与边界

这一阶段的目标不是改变量化算法，而是优化运行时执行路径：

- 保持离线量化产物格式不变，继续使用现有 `.qweight + .scales`
- 保持当前 per-channel INT8 weight-only 语义不变
- 保持输出结果与当前朴素路径在数值上等价，只允许正常浮点舍入误差
- 不引入 activation quantization
- 不引入 KV cache quantization
- 不引入 LM head quantization
- 不改变 Tensor Parallel、packed weight 和 loader 的现有约定

因此，本节优化的唯一重点是把：

```text
qweight -> dequant_weight -> F.linear
```

替换为：

```text
CUDA fused dequantize + GEMM
```

#### 模块划分

建议把实现拆成 4 个子模块，避免把 CUDA kernel、量化分派、线性层接入和验证逻辑混在一起。

##### 1. CUDA kernel 层

建议新增独立实现，例如：

```text
nanovllm/quantization/csrc/int8_weight_only_gemm.cu
nanovllm/quantization/csrc/int8_weight_only_gemm.cpp
```

首版 kernel 的输入输出建议固定为：

```python
fused_int8_weight_only_linear(
    x,        # [tokens, in_features], bf16/fp16
    qweight,  # [out_features, in_features], int8
    scales,   # [out_features], float32
    bias=None # [out_features] or None
) -> y       # [tokens, out_features], same dtype as x
```

kernel 内部行为约定如下：

- `x` 保持 `bf16/fp16`
- `qweight` 保持 `int8`
- `scales` 保持 `float32` 存储
- 线程块按 tile 处理 `x @ W^T`
- 从 global memory 分块读取 `x tile` 和 `qweight tile`
- 在寄存器或 shared memory 中把 `qweight` 转成运行时 dtype
- 按输出通道应用 `scales`
- 直接在 kernel 内完成乘加累积
- 累加器首版建议使用 `fp32`
- 输出阶段再 cast 回 `x.dtype`
- `bias` 加法行为与当前 `F.linear` 路径保持一致

关键点是：反量化后的权重只在 tile 生命周期内存在，不写回 global memory，不生成完整 `dequant_weight` 张量。

首版 kernel 的实现范围建议收敛为：

- 单卡
- eager 路径
- CUDA device only
- `x.ndim == 2`
- 输入在进入 kernel 前整理为 contiguous 的二维矩阵
- 先覆盖当前 Qwen3 INT8 weight-only 所涉及的线性层

如果后续输入是更高维张量，建议在量化方法层先 reshape 为 `[tokens, hidden]`，kernel 只负责最核心的二维矩阵乘。

##### 2. 量化方法层

`Int8WeightOnlyQuantMethod.apply(x, layer)` 的调用面建议保持不变，但内部执行路径切换为：

1. 优先尝试 CUDA fused kernel
2. 不满足条件时自动回退到当前朴素路径

建议内部抽象出一个 helper，例如：

```python
apply_int8_weight_only_linear(x, qweight, scales, bias=None) -> torch.Tensor
```

分派逻辑建议明确写成：

- 若 CUDA extension 可用、device 为 CUDA、dtype/shape 满足 kernel 约束，则走 fused CUDA kernel
- 否则回退到：

```python
weight = qweight.to(x.dtype) * scales.to(x.dtype).unsqueeze(1)
return F.linear(x, weight, bias)
```

这里必须保留回退路径，原因包括：

- 无编译扩展环境时仍要保证功能可用
- 某些小 shape 或特殊 shape 可能不值得走 fused kernel
- 调试和数值对比时需要强制走朴素路径
- 后续排查 TP、CUDA graph 或新模型兼容性问题时，回退路径是重要保底手段

如果需要调试开关，可以只新增内部布尔选项，例如“强制关闭 fused CUDA kernel”，但不要求新增公开用户配置项；用户侧仍沿用现有 `quantization="int8"` 即可。

##### 3. 线性层接入层

线性层接入原则保持当前抽象不变：

- `LinearBase.apply_linear()` 继续通过 `quant_method.apply(...)` 分派
- `RowParallelLinear.forward()` 继续复用同一量化方法，而不是单独复制一套 CUDA fused 逻辑

这意味着以下层的接入方式都不需要改变接口，只替换底层执行内核：

- `ReplicatedLinear`
- `ColumnParallelLinear`
- `MergedColumnParallelLinear`
- `QKVParallelLinear`
- `RowParallelLinear`

实现时应继续保持下面这些约束不变：

- Tensor Parallel 语义不变，kernel 只处理单个 shard 的局部矩阵乘
- `QKVParallelLinear` 的 `q/k/v` packed 顺序不变
- `MergedColumnParallelLinear` 的 `gate/up` packed 顺序不变
- loader 对 `.qweight` 和 `.scales` 的加载与切片逻辑不变

换句话说，这一阶段不碰模型格式和并行切分规则，只替换“局部 shard 上的一次线性层计算是如何执行的”。

##### 4. Benchmark 与质量门禁

性能优化不能只看 tokens/s，必须继续复用当前质量评估流程。

建议后续实现时把 benchmark 与回归门禁固定为两层：

1. 性能层：

- 继续使用 `bench_quant.py`
- 增加或细化 `prefill throughput`
- 增加或细化 `decode throughput`
- 区分 warmup 后吞吐与总吞吐
- 记录 fused 前后显存常驻收益是否保持稳定

2. 质量层：

- 继续使用 `bench_quant_quality.py --mode logits`
- 继续使用 `bench_quant_quality.py --mode ppl`
- 继续使用 `bench_quant_quality.py --mode generation`
- 先过质量，再比较吞吐

这里的核心原则是：CUDA fused kernel 只是执行路径优化，不应引入新的量化语义变化。如果 fused 后质量报告明显劣化，应优先视为实现错误，而不是“性能优化的正常代价”。

#### CUDA kernel 设计约束

为了避免实现阶段重新做产品决策，建议提前锁定以下约束：

- 输入激活 `x` 仍保持 `bf16/fp16`
- `qweight` 仍保持 `int8`
- `scales` 仍保持 `float32`
- `bias` 行为与当前实现一致
- 外部配置项不新增，继续使用现有 `quantization="int8"`
- `Int8WeightOnlyQuantMethod.apply(x, layer)` 仍是统一入口
- Tensor Parallel 只影响 shard 切分，不改变 kernel 的本地计算语义
- packed 权重格式不变，不为 CUDA kernel 重定义模型存储格式

首版实现建议优先使用“每个 thread block 计算输出矩阵一个 tile”的标准 GEMM 结构，并显式约定以下策略：

- `x` tile 从 global memory 读入 shared memory
- `qweight` tile 从 global memory 读入 shared memory 或寄存器
- `qweight` 在 tile 内转成 `fp16/bf16`
- `scales` 以输出通道维度广播到 tile
- 使用 `K` 维分块循环完成累加
- 尽量使用 vectorized load，例如 `int4`/`int8` pack 读取多个 `int8` 元素
- 对 `x` 和 `qweight` 的访问模式保持 coalesced

如果后续发现 `scales=float32` 带来明显额外带宽压力，可以再评估：

- 将 `scales` 在加载后缓存到 shared memory
- 或在保持数值可接受的前提下引入更紧凑的运行时表示

但这些都属于第二轮优化，不应与首版 fused CUDA kernel 同时推进。

#### 分阶段交付顺序

建议按以下顺序落地，避免一次性把风险叠满：

1. 先做单卡 eager 的 CUDA fused kernel  
   在最小场景下验证 kernel 功能、数值一致性和基础吞吐收益。

2. 再接到统一量化分派路径  
   让 `Int8WeightOnlyQuantMethod.apply(...)` 默认优先走 CUDA fused kernel，并保留回退。

3. 再验证所有已支持线性层  
   包括 `ReplicatedLinear`、`ColumnParallelLinear`、`MergedColumnParallelLinear`、`QKVParallelLinear`、`RowParallelLinear`。

4. 再验证 Tensor Parallel  
   先确认 shard 内 fused 计算正确，再检查 `all_reduce`、packed weight 和多卡加载路径是否仍然稳定。

5. 再评估 CUDA graph 路径是否需要专门适配  
   如果 eager 已稳定、TP 已稳定，再看该 kernel 在 graph capture 场景下是否有额外限制。

6. 最后才讨论更激进的后续路线  
   包括专用 INT8 GEMM、INT4/AWQ 扩展、运行时 scale 缓存优化等。

#### 建议的内部接口

为了让后续实现保持边界清晰，建议锁定以下内部接口形态：

```python
class Int8WeightOnlyQuantMethod(QuantMethod):
    def apply(self, x: torch.Tensor, layer: nn.Module) -> torch.Tensor: ...
```

内部再由它调用一个更聚焦的 helper：

```python
def apply_int8_weight_only_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    ...
```

其中：

- `apply()` 负责拿到 layer 上的 `qweight/scales/bias`
- helper 负责选择 CUDA fused kernel 或朴素回退路径
- CUDA extension 的 Python binding 保持“纯函数”形态，只接收张量并返回结果，避免把 layer 结构和 kernel 代码耦合在一起

这种接口划分的好处是：

- 线性层侧不需要了解 CUDA kernel 细节
- kernel 侧不需要了解模型层级结构
- 朴素路径与 fused 路径可以共用同一个入口做 A/B 对比

#### 测试与验收建议

后续真正实现这条优化路线时，至少应覆盖以下场景：

1. 单卡 `enforce_eager=True` 数值对比

- CUDA fused kernel 输出与当前朴素路径逐层对比
- 允许正常浮点误差，不允许系统性偏差

2. 质量回归

- `bench_quant_quality.py --mode logits`
- `bench_quant_quality.py --mode ppl`
- `bench_quant_quality.py --mode generation`

要求 fused 前后都跑一遍，确认没有明显语义退化。

3. 性能对比

- `bench_quant.py` 对比 fused 前后的 `prefill throughput`
- `bench_quant.py` 对比 fused 前后的 `decode throughput`
- `bench_quant.py` 对比 fused 前后的总吞吐

4. 多卡 TP smoke test

- `tensor_parallel_size=2` 的 INT8 路径可以正常加载
- 可以正常生成
- 质量检查通过

5. 回退路径验证

- CUDA extension 不可用时能自动回退
- 不支持的 dtype/shape 时能自动回退
- 回退后功能行为保持正确

验收标准建议先采用定性门槛，而不是在首版里拍脑袋定死数值：

- 吞吐应相对当前 INT8 朴素实现有明确提升
- 模型常驻显存收益不能丢失
- 质量报告不能比当前 INT8 基线出现明显恶化

如果后续多次运行结果稳定，再把吞吐提升幅度和质量阈值固化为更严格的 benchmark / CI 门禁。

### 当前 CUDA 融合实现总结

当前仓库已经按上面的路线接入了第一版 `CUDA fused dequant + GEMM`，目标是先把执行路径打通，让 INT8 weight-only 在线性层前向时不再总是显式构造完整 `dequant_weight`。

本次代码修改集中在以下几个位置：

1. CUDA fused helper

   新增 [nanovllm/quantization/cuda.py](/E:/llmegine/nano-vllm-main/nano-vllm-main/nanovllm/quantization/cuda.py)，职责包括：

   - 判断当前输入是否满足 fused CUDA kernel 的使用条件
   - 通过 `torch.utils.cpp_extension.load(...)` 在首次使用时动态编译并加载 CUDA extension
   - 提供统一入口 `apply_int8_weight_only_linear(...)`
   - 若 CUDA extension 不可用，或 shape / dtype / device 不满足要求，则自动回退到原有 `dequantize + F.linear` 路径

2. CUDA extension 入口与 kernel

   新增：

   - [nanovllm/quantization/csrc/int8_weight_only_gemm.cpp](/E:/llmegine/nano-vllm-main/nano-vllm-main/nanovllm/quantization/csrc/int8_weight_only_gemm.cpp)
   - [nanovllm/quantization/csrc/int8_weight_only_gemm.cu](/E:/llmegine/nano-vllm-main/nano-vllm-main/nanovllm/quantization/csrc/int8_weight_only_gemm.cu)

   首版 CUDA kernel 的行为是：

   - 输入 `x` 仍使用 `fp16/bf16`
   - 权重 `qweight` 仍保持 `int8`
   - `scales` 仍保持 `float32`
   - 在 kernel 内按输出元素执行 `int8 -> float -> * scale -> accumulate`
   - 不再在 Python 路径中生成完整浮点权重矩阵

   这版实现优先保证路径正确和接口稳定，还不是最终的高性能 tiled GEMM 版本。后续如果继续优化，重点会落在 shared memory、vectorized load、tile 设计和访存模式上。

3. INT8 量化执行路径切换

   [nanovllm/quantization/int8.py](/E:/llmegine/nano-vllm-main/nano-vllm-main/nanovllm/quantization/int8.py) 已经改为统一走 `apply_int8_weight_only_linear(...)`，也就是：

   - 能走 CUDA fused kernel 时优先走 fused 路径
   - 不能走时自动回退到朴素路径

4. 线性层接入

   [nanovllm/layers/linear.py](/E:/llmegine/nano-vllm-main/nano-vllm-main/nanovllm/layers/linear.py) 中原本特判的 `RowParallelLinear` INT8 路径，也已经切到同一个 helper，避免一部分层走 fused、一部分层仍停留在手写反量化逻辑。

#### 如何运行

这版实现不需要单独手写 `setup.py` 或手动预编译。运行方式仍然和当前 INT8 路径一致；区别是第一次真正走到 fused CUDA 路径时，会自动触发 extension 的 JIT 编译。

建议按下面顺序运行：

1. 先准备 INT8 量化模型

```bash
python scripts/quantize.py \
  --model Qwen3-0.6B/qwen/Qwen3-0___6B \
  --output Qwen3-0.6B/qwen/Qwen3-0___6B-int8 \
  --quantization int8
```

2. 做单条生成 smoke test

```python
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

base = "Qwen3-0.6B/qwen/Qwen3-0___6B"
int8 = "Qwen3-0.6B/qwen/Qwen3-0___6B-int8"

tokenizer = AutoTokenizer.from_pretrained(base)
llm = LLM(
    base,
    quantization="int8",
    quantized_model_path=int8,
    enforce_eager=True,
    tensor_parallel_size=1,
)

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "introduce yourself"}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

outputs = llm.generate([prompt], SamplingParams(temperature=0.6, max_tokens=64))
print(outputs[0]["text"])
```

3. 跑吞吐 benchmark

```bash
python bench_quant.py --mode both --enforce-eager
```

如果只想看 INT8：

```bash
python bench_quant.py --mode int8 --enforce-eager
```

4. 跑质量回归

```bash
python bench_quant_quality.py \
  --model Qwen3-0.6B/qwen/Qwen3-0___6B \
  --int8-model Qwen3-0.6B/qwen/Qwen3-0___6B-int8 \
  --mode logits \
  --output quant_quality_logits.json
```

完整质量评估：

```bash
python bench_quant_quality.py \
  --model Qwen3-0.6B/qwen/Qwen3-0___6B \
  --int8-model Qwen3-0.6B/qwen/Qwen3-0___6B-int8 \
  --mode all \
  --output quant_quality_report.json
```

#### 运行时注意事项

- 首次走 fused CUDA 路径时，会触发 `torch.utils.cpp_extension.load(...)` 动态编译，因此第一次运行通常会比后续慢。
- 需要本地 Python 环境中可正常导入 `torch`，并且具备可用的 CUDA / NVCC 编译环境，否则会自动回退到 `dequantize + F.linear`。
- 当前 fused kernel 只在 CUDA、`fp16/bf16` 激活、`int8` 权重、`float32` scales 的组合下尝试启用。
- 当前版本的回退策略是静默回退；如果你想观察是否真的命中了 fused kernel，后续可以再补调试日志或显式开关。
- 这版 kernel 的重点是先完成“反量化融合到 GEMM”的执行路径，吞吐可能会优于当前朴素实现，但还不应把它视为最终性能版。
