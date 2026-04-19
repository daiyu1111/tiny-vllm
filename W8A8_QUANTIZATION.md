# Nano-vLLM W8A8 量化实现设计

## 背景与目标

当前仓库已经落地了 `W8A16` 路径：权重离线量化为 `int8`，运行时沿用浮点激活，在 `Linear` 层内走 fused INT8 weight-only kernel 或 `dequantize + F.linear` fallback。这个版本已经打通了：

- `Config.quantization="int8"` 的配置入口
- `LinearBase` 及 TP 线性层的量化参数注册与前向分发
- `scripts/quantize.py` 的离线量化产物生成
- `nanovllm/quantization/cuda.py` 的 backend 分发
- `bench_quant.py` 与 `bench_quant_quality.py` 的性能、质量验证链路

`W8A8` 的目标不是重复这条链路，而是在它的基础上继续向前推进一层：让激活也进入 `int8` 计算路径，从而把线性层 GEMM 真正推进到 `int8 x int8 -> int32` 的高性能实现上，而不是继续依赖“权重量化后再反量化回浮点”的执行方式。

本文档只讨论如何在当前 Nano-vLLM 仓库中实现 `W8A8`，范围固定为线性层推理路径，不展开通用 AWQ、SmoothQuant 教程，也不覆盖 KV cache、Embedding、LM head 的量化。

默认设计结论如下：

- 以现有 `W8A16` 为基线扩展到 `W8A8`
- 第一目标是高性能 kernel 路线，不以 Python fallback 为主线
- 范围限定在线性层，不覆盖 KV cache、Embedding、LM head
- 激活量化采用运行时动态量化，权重量化保持离线静态量化
- Tensor Parallel、packed QKV / gate-up、现有 loader 命名约定全部延续

## 为什么是 W8A8，为什么仍然只做 Linear

当前仓库最适合承接 `W8A8` 的位置仍然是 `Linear` 抽象层，而不是 attention 或 KV cache。

原因很直接：

- 主计算热点仍然集中在 `nanovllm/layers/linear.py` 的几类线性层。
- 现有 `LinearBase` 已经具备量化参数注册、量化方法分发、TP 分片加载等骨架能力。
- `scripts/quantize.py` 已经处理了 `qkv_proj` 与 `gate_up_proj` 的 packed 权重拼接逻辑。
- `nanovllm/utils/loader.py` 已经承担了量化权重加载与校验职责，适合继续扩展 `W8A8` 命名与校验规则。
- `nanovllm/quantization/cuda.py` 已经有 backend 分发和日志路径，可继续复用为 `W8A8` 的统一入口。

相比之下，KV cache 量化会直接牵动：

- `attention.py` 中的 KV 写入路径
- flash-attn 的读取约定
- prefill / decode 两条 cache 数据流
- Triton cache layout 与 dtype 假设

这会让问题从“线性层量化”扩大成“注意力子系统重构”。因此，`W8A8` 仍然只做 `Linear`，不碰 KV cache。

## 数据格式与接口约定

### 配置层

建议在现有 `Config.quantization` 上新增 `w8a8` 模式：

```python
quantization: str | None = None  # None | "int8" | "w8a8" | "int4_awq"
```

第一版不建议把 `W8A8` 隐藏在额外布尔开关里，而是直接作为显式模式暴露。这样可以让：

- 模型构建逻辑明确知道当前是 `W8A16` 还是 `W8A8`
- loader 明确知道应该校验哪些量化张量
- benchmark / quality 工具直接按模式做对比

如果后续需要和现有命名体系更强一致，也可以把模式名改为 `int8_w8a8`，但本文默认使用 `w8a8`。

### QuantMethod 抽象

当前 `QuantMethod` 只有两项职责：

- `create_weights(layer, input_size, output_size)`
- `apply(x, layer)`

对 `W8A16` 来说这已经够用，但对 `W8A8` 还不够，因为运行时需要显式描述激活量化元信息。建议把抽象扩展成“既管理权重侧元数据，也管理激活侧量化约定”的统一接口。接口不要求机械照抄，但职责必须明确：

```python
class QuantMethod(ABC):
    def create_weights(self, layer: nn.Module, input_size: int, output_size: int) -> None: ...
    def quantize_activation(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
    def apply(self, x: torch.Tensor, layer: nn.Module) -> torch.Tensor: ...
```

其中：

- `create_weights(...)` 负责注册权重侧参数，例如 `qweight`、`w_scales`
- `quantize_activation(...)` 负责运行时把浮点激活量化成 `int8` 与 `a_scales`
- `apply(...)` 负责选择 fused kernel 并完成 epilogue

这样可以保证：

- `LinearBase` 只负责分派，不直接知道 `W8A8` 的细节
- `loader` 不需要理解 activation quantization
- 后续如果扩展到 block-wise activation scaling，也只需要替换量化方法内部实现

### 量化权重格式

`W8A8` 继续沿用“权重离线量化、激活运行时量化”的划分，因此模型目录中只保存权重侧产物，不保存激活量化结果。

推荐的权重格式如下：

- `*.qweight`: `torch.int8`
- `*.w_scales`: `torch.float32` 或 `torch.float16`
- `*.bias`: 保持原始浮点

示例：

```text
model.layers.0.self_attn.qkv_proj.qweight
model.layers.0.self_attn.qkv_proj.w_scales
model.layers.0.self_attn.qkv_proj.bias
```

命名上不建议继续复用 `*.scales` 这个模糊名字。对 `W8A8` 而言，显式区分：

- `w_scales`: 离线权重 scale
- `a_scales`: 运行时激活 scale

会让 kernel、调试日志和 benchmark 代码都更清晰。

### 激活量化格式

激活不离线存盘，而是在前向时动态生成：

- 输入 `x` 先 reshape 成 `2D`，形状为 `[m, k]`
- 对每一行做对称动态量化
- 输出：
  - `x_q`: `torch.int8`，形状 `[m, k]`
  - `a_scales`: `torch.float32` 或 `torch.float16`，形状 `[m]`

第一版默认采用 `per-token/per-row` scale，不做 zero-point，不做 group-wise activation quantization，不做额外 clipping policy。

## 运行时实现思路

### 推荐执行链路

`W8A8` 的推荐执行链路固定如下：

1. 输入 `x` 在前向入口 reshape 为二维矩阵 `[m, k]`
2. 对每一行做动态对称 INT8 量化，得到 `x_q` 和 `a_scales`
3. 权重侧直接读取离线存储的 `qweight` 和 `w_scales`
4. kernel 内完成 `int8 x int8 -> int32 accum`
5. epilogue 中执行：
   - `accum_fp = accum_int32 * a_scales * w_scales`
   - 若存在 bias，则在同一 epilogue 中完成融合
6. 输出张量恢复为运行时 dtype，即 `fp16` 或 `bf16`
7. 最后 reshape 回原始 batch 维度

可用伪代码表示为：

```python
x_2d = x.contiguous().view(-1, x.shape[-1])
x_q, a_scales = dynamic_quantize_per_row(x_2d)
y = w8a8_linear(x_q, a_scales, qweight, w_scales, bias, out_dtype=x.dtype)
return y.view(*x.shape[:-1], qweight.shape[0])
```

### 为什么不推荐“激活先反量化回浮点再走 F.linear”

不推荐把 `W8A8` 实现成下面这种路径：

```python
x_q, a_scales = quantize(x)
x_dequant = x_q.to(x.dtype) * a_scales[:, None]
weight_dequant = qweight.to(x.dtype) * w_scales[:, None]
y = F.linear(x_dequant, weight_dequant, bias)
```

原因是这会直接丢掉 `W8A8` 最重要的收益：

- GEMM 主体仍然在浮点路径执行
- 激活量化只增加一次量化和一次反量化开销
- 很难在 prefill / decode 上获得真实吞吐提升
- 实际效果更接近“多做了一遍格式转换的 `W8A16`”

因此本文默认把 fused kernel 视为 `W8A8` 的主线实现，而不是可选优化项。

### 核心数值决策

以下决策在第一版文档中固定，不留给实现阶段再选择：

- 权重：离线对称 per-output-channel INT8
- 激活：运行时对称动态 INT8，默认 per-token/per-row scale
- accumulation：`int32`
- output dtype：跟随当前运行时 `fp16/bf16`
- bias：保持浮点，在 epilogue 融合
- zero-point：第一版不做
- group-wise activation quant：第一版不做
- `lm_head`：不纳入范围
- fallback：可以存在，但文档主线不以 fallback 为中心

## 仓库改动点

### 配置层

配置层只做模式扩展与参数兜底：

- `Config.quantization` 接受 `"w8a8"`
- 与 `int8` 一样，`quantized_model_path` 继续指向量化模型目录
- 如果后续需要额外控制 activation quant 形态，可新增保守参数，例如：
  - `act_quant_granularity: str = "per_token"`
  - `w8a8_backend: str | None = None`

第一版不建议把很多 tuning 开关直接暴露到公共配置里。先固定 `per-token` 和默认 backend，文档即可落地。

### 量化抽象层

在 `nanovllm/quantization/` 下新增 `W8A8QuantMethod`，职责如下：

- 注册 `qweight` 与 `w_scales`
- 定义 `dynamic_quantize_per_row(x)` 这样的激活量化入口
- 调用新的 `apply_w8a8_linear(...)`

现有 `Int8WeightOnlyQuantMethod` 可以继续保留。`LinearBase` 根据 `quantization` 选择：

- `None` -> 浮点权重
- `"int8"` -> `W8A16`
- `"w8a8"` -> `W8A8QuantMethod`

### 线性层

`LinearBase` 和现有 TP 线性层仍然是唯一接入面，不新增另一套平行的线性层体系。

第一版的要求是：

- `ReplicatedLinear`
- `ColumnParallelLinear`
- `MergedColumnParallelLinear`
- `QKVParallelLinear`
- `RowParallelLinear`

全部复用同一套 `W8A8` 权重格式和运行时 activation quantization 逻辑。

其中最关键的是继续兼容现有 packed 模块：

- `q_proj/k_proj/v_proj -> qkv_proj`
- `gate_proj/up_proj -> gate_up_proj`

这意味着 loader、离线脚本和线性层参数名仍然要沿用现在的 packed 规则，不能为了 `W8A8` 再引入一套新命名。

### loader 与离线量化脚本

loader 与 `scripts/quantize.py` 的职责边界保持不变：

- `scripts/quantize.py` 负责离线生成权重侧张量
- `loader.py` 负责把张量按参数名加载进模型

`W8A8` 对这两处的新增要求如下：

1. `scripts/quantize.py` 支持 `--quantization w8a8`
2. 继续复用现有的 packed QKV / gate-up 拼接逻辑
3. 为目标线性层生成：
   - `qweight`
   - `w_scales`
   - 原始 bias
4. `loader.py` 在 `quantization == "w8a8"` 时校验所有量化线性层的 `qweight` / `w_scales` 是否完整加载

权重量化策略保持与当前 `W8A16` 一致，即按 output channel 做对称静态 INT8 量化。这样可以最大化复用已有离线脚本逻辑，降低 `W8A8` 第一版的非必要变量。

### CUDA 扩展与 backend 分发

现有 `nanovllm/quantization/cuda.py` 已经提供了：

- backend 选择
- extension lazy load
- kernel path 日志
- fallback 兜底

`W8A8` 应继续沿用这套模式，新增独立入口，例如：

```python
apply_w8a8_linear(...)
native_w8a8_linear(...)
cutlass_w8a8_linear(...)
```

建议的 backend 策略是：

- `v1`: 单 backend 打通，优先 native CUDA 或已有最容易接的 CUTLASS 路线
- `v2`: native / CUTLASS 双 backend 补齐
- `v3`: 针对 decode 小 batch、小 `m` 形状继续细化 kernel

kernel 接口职责固定为：

- 输入：
  - `x_q[int8]`
  - `a_scales[float]`
  - `qweight[int8]`
  - `w_scales[float]`
  - `bias[float | None]`
- 计算：
  - `int8 x int8 -> int32 accum`
- 输出：
  - `fp16/bf16`

第一版不要求同时覆盖所有 shape 最优路径，但要确保 prefill 和 decode 都能命中真正的 `W8A8` kernel，而不是悄悄退回浮点 GEMM。

### benchmark 与质量验证

现有 benchmark 链路应直接扩展为三方对比：

- bf16 baseline
- `W8A16`
- `W8A8`

重点复用：

- `bench_quant.py` 的吞吐和显存统计框架
- `bench_quant_quality.py` 的 artifact / logits / ppl / generation 检查框架

新增后需要支持：

- `--w8a8-model`
- 运行时 `quantization="w8a8"`
- 输出中明确区分 backend / path

## 风险与阶段拆分

### v1：先打通高性能单卡主路径

第一阶段只要求：

- 单卡
- 单 backend
- 权重对称 per-channel INT8
- 激活动态 per-token/per-row INT8
- `int8 x int8 -> int32` fused kernel
- 输出回到 `fp16/bf16`

这一阶段的核心是先让 `W8A8` 真正进入高性能 kernel 路径，而不是追求所有 shape 上都最优。

### v2：补齐 backend 与工程鲁棒性

第二阶段补：

- `native` / `cutlass` 双 backend
- 更完整的 shape dispatch
- 更清晰的日志、报错与 fallback 说明
- 更完善的 benchmark 自动化

### v3：继续优化 decode 与更激进的量化策略

第三阶段再考虑：

- block-wise activation scaling
- persistent kernel
- 更细的 decode 路径优化
- 更激进的 epilogue 融合

在这之前，不建议同时引入 zero-point、group-wise activation quant、KV cache quantization 等额外变量。

## 现有实现的复用关系

为了避免 `W8A8` 方案变成另一套平行系统，文档中应明确以下复用关系：

- `LinearBase` / TP 线性层仍然是唯一接入面
- `scripts/quantize.py` 继续负责离线权重量化，只扩展模式选择和产物命名
- `nanovllm/quantization/cuda.py` 的 backend 分发机制继续复用
- `bench_quant_quality.py` 扩成 bf16 vs `W8A16` vs `W8A8` 三方比较
- `loader.py` 继续承担量化张量完整性校验

`W8A8` 的实现重点是扩展当前链路，而不是新开一条完全独立的量化系统。

## 验收标准

文档落地后，`W8A8` 至少需要满足以下验收标准。

### 1. 产物检查

`w8a8` 模型目录能生成完整的：

- `qweight`
- `w_scales`
- `bias`

并且命名兼容 packed `qkv_proj` 与 `gate_up_proj`。

### 2. 加载检查

当 `quantization="w8a8"` 时：

- 所有量化线性层都能完成参数加载
- 缺失 `qweight` 或 `w_scales` 时直接报清晰错误
- packed 模块不会因旧名字映射导致误装载

### 3. 数值检查

固定输入下，对单层或整模型输出比较 bf16 与 `W8A8`，至少记录：

- cosine similarity
- relative L2
- top-k logits overlap

第一版不要求完全逼近 bf16，但必须证明结果稳定且无明显异常漂移。

### 4. 端到端推理检查

`example.py` 或等价最小生成脚本需要能够完成：

- prefill
- decode
- 正常文本输出

并且不出现：

- NaN
- 空输出
- 明显异常重复

### 5. 性能检查

至少分别比较 prefill 和 decode：

- `W8A8` 对比 bf16
- `W8A8` 对比当前 `W8A16`

并要求日志中能够区分实际命中的 backend 与 kernel path，避免把 fallback 结果误认为 `W8A8` 主路径性能。

## 实现默认值

除非后续实现明确推翻，本文默认采用以下约定：

- 文档文件名：`W8A8_QUANTIZATION.md`
- 文风：工程设计说明，不做大段方案比较
- 量化模式命名：`w8a8`
- 前提：仓库已经具备 `W8A16`、CUDA INT8 backend、离线量化脚本和质量 benchmark
- 默认只写“如何在当前仓库里实现”，不展开通用量化算法综述

以上约定的目的是把 `W8A8` 收敛成一个能直接指导实现的工程规格，而不是开放式讨论题。

## 当前已完成的修改

下面这部分记录当前仓库里已经落地的 `W8A8` v1 改动，目的是把“设计方案”与“当前代码状态”区分开。

### 1. 配置与模式

- `Config.quantization` 已支持 `"w8a8"`。
- 现有 `"int8"` 仍表示原来的 `W8A16` 路径，没有复用到 `W8A8`。
- `quantized_model_path` 继续沿用，不新增额外模型目录语义。

### 2. Python 侧量化主链

- 新增 `nanovllm/quantization/w8a8.py`，实现 `W8A8QuantMethod`。
- `LinearBase` 已支持三种模式：
  - `None`
  - `"int8"`
  - `"w8a8"`
- `W8A8` 路径中：
  - 权重参数注册为 `.qweight` 和 `.w_scales`
  - 激活在前向时做动态对称 `per-token/per-row` INT8 量化
  - 前向调用 `apply_w8a8_linear(...)`

### 3. 权重量化产物

- `scripts/quantize.py` 已支持：
  - `--quantization int8`
  - `--quantization w8a8`
- `w8a8` 产物格式为：
  - `<param>.qweight`
  - `<param>.w_scales`
  - `<param>.bias`
- QKV 和 gate/up 仍然沿用当前 packed 规则：
  - `q_proj/k_proj/v_proj -> qkv_proj`
  - `gate_proj/up_proj -> gate_up_proj`

### 4. 加载与校验

- `loader.py` 已增加 `w8a8` 的完整性校验。
- 当 `quantization="w8a8"` 时，会检查量化线性层的：
  - `.qweight`
  - `.w_scales`
- 旧 `int8` 路径的 `.scales` 校验保留不变。

### 5. CUDA 与 fallback

- `nanovllm/quantization/cuda.py` 已新增：
  - `apply_w8a8_linear(...)`
  - `native_w8a8_linear(...)`
  - `w8a8_linear_fallback(...)`
- 已新增 `W8A8` 的 C++ / CUDA 扩展入口：
  - `nanovllm/quantization/csrc/w8a8_gemm.cpp`
  - `nanovllm/quantization/csrc/w8a8_gemm.cu`
- 当前 native kernel 是一条语义正确、实现朴素的通用 `int32 accumulation` 路径，重点是先打通功能，不是最终性能版本。
- 当前仍保留 Python fallback，便于数值比对和 kernel 不可用时兜底。

### 6. Benchmark 与质量脚本

- `bench_quant.py` 已支持 `w8a8` 模型路径与三方对比：
  - bf16
  - int8 (`W8A16`)
  - w8a8
- `bench_quant_quality.py` 已扩展为可同时检查：
  - int8
  - w8a8
- artifact / logits / ppl / generation 检查逻辑已经能识别 `w8a8` 的 `.w_scales` 格式。

### 7. 当前状态说明

截至目前，`W8A8` 代码已经完成：

- 配置接线
- 权重量化产物格式
- loader 校验
- 线性层前向分发
- Python fallback
- native CUDA 扩展入口
- benchmark / quality 脚本接线

但当前 native `W8A8` kernel 仍然是第一版通用实现，后续如果目标是吞吐优化，还需要继续做更高性能的 tile / WMMA / shape dispatch 版本。

## 如何使用

下面给出当前 `W8A8` 的建议使用方式。

### 1. 生成 `W8A8` 量化模型

```bash
python scripts/quantize.py \
  --model /mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B \
  --output /mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B-w8a8 \
  --quantization w8a8
```

生成后，模型目录中应包含类似下面的量化张量：

```text
model.layers.0.self_attn.qkv_proj.qweight
model.layers.0.self_attn.qkv_proj.w_scales
model.layers.0.self_attn.qkv_proj.bias
```

### 2. 在运行时加载 `W8A8`

Python 侧使用方式与现有 `int8` 类似，只是把量化模式改成 `w8a8`：

```python
from nanovllm import LLM

llm = LLM(
    "/mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B",
    quantization="w8a8",
    "/mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B-w8a8",
    enforce_eager=True,
    tensor_parallel_size=1,
)
```

这里仍然需要：

- `model` 指向原始模型目录，用于读取配置和 tokenizer
- `quantized_model_path` 指向 `w8a8` 量化产物目录

### 3. 选择 backend

当前 `W8A8` 支持两个 backend：

- `native`
- `fallback`

默认是 `native`。可以通过环境变量切换：

```bash
set NANOVLLM_W8A8_BACKEND=native
```

或：

```bash
set NANOVLLM_W8A8_BACKEND=fallback
```

如果希望打印命中的 `W8A8` backend / path，可以打开日志：

```bash
set NANOVLLM_W8A8_LOG_PATH=1
```

### 4. 跑性能对比

可以直接用 benchmark 脚本比较 bf16、`W8A16`、`W8A8`：

```bash
python bench_quant.py \
  --model /path/to/base-model \
  --int8-model /path/to/int8-model \
  --w8a8-model /path/to/w8a8-model \
  --mode all
```

### 5. 跑质量检查

可以直接用质量脚本做 artifact、logits、ppl、generation 对比：

```bash
python bench_quant_quality.py \
  --model /path/to/base-model \
  --int8-model /path/to/int8-model \
  --w8a8-model /path/to/w8a8-model \
  --mode all \
  --output quant_quality_report.json
```

### 6. 当前建议的验证顺序

建议按下面顺序使用当前 `W8A8`：

1. 先生成 `w8a8` 量化产物
2. 用 `fallback` backend 验证数值链路是否通
3. 再切换到 `native` backend
4. 跑 `bench_quant_quality.py` 看质量
5. 跑 `bench_quant.py` 看 prefill / decode 性能

如果 native kernel 在某些环境下不可用，当前实现会退回 fallback，这种情况更适合先查日志再继续定位。

## 服务器实际路径示例

以下命令按当前服务器目录修正：

- 工作目录：`/mnt/workspace/nano-vllm-main`
- 原始模型目录：`/mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B`
- INT8 模型目录：`/mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B-int8`
- W8A8 模型目录：`/mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B-w8a8`

### 1. 生成 `W8A8` 量化模型

```bash
cd /mnt/workspace/nano-vllm-main

python scripts/quantize.py \
  --model /mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B \
  --output /mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B-w8a8 \
  --quantization w8a8
```

### 2. 运行时加载 `W8A8`

```python
from nanovllm import LLM

llm = LLM(
    "/mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B",
    quantization="w8a8",
    quantized_model_path="/mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B-w8a8",
    enforce_eager=True,
    tensor_parallel_size=1,
)
```

### 3. 跑性能对比

```bash
cd /mnt/workspace/nano-vllm-main

python bench_quant.py \
  --model /mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B \
  --int8-model /mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B-int8 \
  --w8a8-model /mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B-w8a8 \
  --mode all
```

### 4. 跑质量检查

```bash
cd /mnt/workspace/nano-vllm-main

python bench_quant_quality.py \
  --model /mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B \
  --int8-model /mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B-int8 \
  --w8a8-model /mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B-w8a8 \
  --mode all \
  --output quant_quality_report.json
```
