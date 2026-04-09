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
