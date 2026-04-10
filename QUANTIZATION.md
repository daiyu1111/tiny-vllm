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

### 下一步优化方向

1. 优先优化计算路径

   当前最大性能瓶颈是每次前向都显式反量化权重：

   ```text
   qweight -> dequant_weight -> F.linear
   ```

   下一步应优先实现 Triton fused dequant GEMM 或专用 INT8 GEMM kernel，将反量化和矩阵乘法融合，减少临时张量和内存带宽开销。

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