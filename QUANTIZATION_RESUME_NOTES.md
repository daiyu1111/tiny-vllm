# Nano-vLLM 量化成果讲稿与简历素材

## 简历可直接使用版

### 保守版

- 设计并实现 Nano-vLLM 的 INT8 weight-only (`W8A16`) 与 `W8A8` 量化链路，打通离线量化、量化权重加载、Tensor Parallel 分片适配、运行时 kernel 分派与质量评测闭环。
- 对 Qwen3-0.6B 的主干线性层完成量化接入，模型加载显存由 `1.13 GiB / 1.48 GiB` 降至 `0.74 GiB / 1.05 GiB`。
- 建立 artifact / logits / PPL / generation 四层质量评测，INT8 的 PPL 相对 bf16 仅上升 `0.85%`，decode 吞吐达到 bf16 的 `1.156x`。

### 进攻版

- 以 Nano-vLLM 现有 `LinearBase + Tensor Parallel + packed module` 体系为基础，落地 `W8A16(INT8 weight-only)` 与 `W8A8` 两条量化路径，在不改动 attention / KV cache 主路径的前提下完成从量化产物生成到运行时推理的工程闭环。
- 为 `qkv_proj`、`o_proj`、`gate_up_proj`、`down_proj` 接入量化权重格式与 TP 分片加载，兼容 `q_proj/k_proj/v_proj -> qkv_proj`、`gate_proj/up_proj -> gate_up_proj` 的 packed 权重映射。
- 在性能与质量两侧建立基线：量化后模型加载显存降低约 `34%`，INT8 在 decode 阶段超过 bf16，且 WikiText-2 上 PPL 仅增加 `+0.85%`，说明量化链路已经具备可部署的稳定性。

### 一句话总结

我把 Nano-vLLM 的量化工作做成了一个完整工程链路：不仅能把权重压成 INT8/W8A8，还能稳定加载、支持 TP、做质量回归，并明确给出“INT8 当前最成熟、W8A8 已跑通但性价比暂不如 INT8”的实验结论。

## 项目背景与目标

Nano-vLLM 本身是一个轻量化推理引擎，已经具备连续批处理、抢占式调度、Prefix Cache、CUDA Graph、Tensor Parallel 等一整套运行时机制。对这样一个系统做量化，目标不是单点写一个量化 kernel，而是把下面几件事全部打通：

- 能离线生成量化权重产物；
- 能在运行时正确加载量化权重，并兼容现有 TP 与 packed module 逻辑；
- 能保持生成质量在可接受范围内；
- 能通过 benchmark 和质量脚本给出可复现、可解释的结果；
- 最后能明确回答：量化到底省了什么、快了什么、慢了什么、为什么会这样。

因此，这次量化工作的目标不是“立刻做到端到端吞吐全面超过 bf16”，而是先打通一条可加载、可运行、可验证、可 benchmark 的完整量化工程链路。

## 量化方案怎么做

### 两条量化路径

本次工作最终落地了两条路径：

- `W8A16`：INT8 weight-only，权重使用 `int8` 保存，激活保持运行时原始 dtype。对当前 Qwen3-0.6B 来说，A16 实际是 `bf16`。
- `W8A8`：权重使用 `int8` 保存，激活在运行时动态量化为 `int8`，在线性层中执行 `int8 x int8 -> int32 accum` 的计算路径。

### 量化了哪些权重

只量化 `Qwen3DecoderLayer` 主干路径里占计算与权重比例最高的线性层：

- `self_attn.qkv_proj`
- `self_attn.o_proj`
- `mlp.gate_up_proj`
- `mlp.down_proj`

这些层分别由现有的：

- `QKVParallelLinear`
- `MergedColumnParallelLinear`
- `RowParallelLinear`

承载，因此量化直接接入 `LinearBase` 体系，而不是额外再造一套线性层框架。

### 没有量化哪些部分

第一版明确不碰这些路径：

- KV cache
- Embedding
- LM head
- attention kernel 本体
- flash-attn 调用接口

原因很直接：这些部分和 KV cache 布局、prefix cache、prefill/decode 两条执行路径绑定很深。如果第一版就同时改线性层和 attention/KV cache，复杂度会指数上升，很难把“性能问题”和“数值正确性问题”分开定位。

### 兼容现有架构的方式

量化是沿着 Nano-vLLM 现有工程边界接入的，不是重写运行时：

- 通过 `Config.quantization` 与 `quantized_model_path` 选择量化路径；
- 通过 `LinearBase` 分派 `Int8WeightOnlyQuantMethod` 与 `W8A8QuantMethod`；
- 通过 `loader.py` 在原加载流程里支持量化张量；
- 继续沿用 packed 权重语义：
  - `q_proj / k_proj / v_proj -> qkv_proj`
  - `gate_proj / up_proj -> gate_up_proj`
- 继续沿用 TP 分片方式：
  - 输出维切分的层，`qweight` 与 scale 按输出通道切分；
  - 输入维切分的层，`qweight` 按输入维切分，scale 不按输入维切分。

也就是说，量化工作是在“保留原运行时架构”的基础上完成的，不是另起一套推理框架。

## 关键实现细节

### W8A16：INT8 weight-only 是怎么做的

`W8A16` 采用对称 INT8、per-output-channel 的权重量化方式。对每个输出通道单独计算一个 scale，离线生成：

- `qweight: torch.int8`
- `scales: torch.float32`

核心公式可以概括为：

```python
scale[i] = max(abs(weight_row)) / 127
qweight[i] = round(weight_row / scale[i]).clamp(-127, 127).to(torch.int8)
```

运行时不做在线量化，而是直接加载离线产物。前向计算走两种可能路径：

- 优先走 fused kernel；
- 如果 backend 不可用，则走 fallback：先反量化，再执行 `F.linear`。

也就是说，`W8A16` 的本质是“权重常驻 INT8，激活保持 bf16”，重点先解决模型常驻显存和量化工程闭环问题。

### W8A8：权重端怎么做

`W8A8` 的权重量化与 `W8A16` 仍然保持一致，仍然是离线对称 INT8，只是命名上明确区分为：

- `qweight`
- `w_scales`

这样可以在实现与调试时把：

- `w_scales`：权重侧 scale
- `a_scales`：激活侧 scale

分得很清楚，避免运行时混淆。

### W8A8：激活是怎么量化的

`W8A8` 的核心变化在于激活也进入了 INT8 计算路径，但激活不离线存盘，而是在运行时动态量化。

具体做法是：

1. 先把输入 `x` reshape 成二维矩阵 `[m, k]`；
2. 按行做基于 `abs_max` 的对称动态量化，也就是 per-token / per-row 量化；
3. 得到：
   - `x_q: torch.int8`
   - `a_scales: torch.float32`
4. 再和离线保存的：
   - `qweight: torch.int8`
   - `w_scales: torch.float32`
     一起送进 `W8A8` kernel。

这里的 scale 计算方式是：先对每一行激活取 `abs().amax()`，再除以 `127`，得到该行的动态量化 scale。
也就是说，当前实现采用的是基于 per-row `abs_max` 的对称动态量化。

它的数值路径可以概括为：

```python
x_2d = x.contiguous().view(-1, x.shape[-1])
x_q, a_scales = quantize_int8_per_row_dynamic(x_2d)
y = w8a8_linear(x_q, a_scales, qweight, w_scales, bias, out_dtype=x.dtype)
```

kernel 内部的核心思路是：

- 主体计算走 `int8 x int8 -> int32 accum`
- epilogue 阶段再乘上 `a_scales * w_scales`
- 同时融合 bias
- 最终输出恢复成运行时 dtype，也就是 `bf16/fp16`

这条路径比 `W8A16` 更激进，因为它不是简单把权重压小，而是真正把线性层 GEMM 推进到 INT8 计算链路里。

### 为什么能兼容现有系统

这一版能稳定落地，关键在于没有去破坏现有运行时的深层假设：

- 不改 attention / flash-attn / KV cache 路径；
- 不改 Prefix Cache 与 block table 的逻辑；
- 不改现有 `LinearBase` / TP / packed module 框架；
- 只在线性层与 loader 这一层做量化能力扩展。

这让问题边界足够清晰：如果质量有问题，大概率是 packed 顺序、scale 对齐或 kernel 行为；如果性能有问题，大概率是 prefill 形状、激活量化开销或 backend 路径选择，而不是整个推理系统一起失控。

## 实验结果与成就

下面的数字来自当前仓库中的 `bench_quant.py` 与 `quant_quality_report.json`。

### 显存

模型加载后的显存占用如下：


| 模式 | after-load allocated / reserved | runtime peak allocated |
| ---- | ------------------------------- | ---------------------- |
| bf16 | `1.13 GiB / 1.48 GiB`           | `19.60 GiB`            |
| int8 | `0.74 GiB / 1.05 GiB`           | `19.47 GiB`            |
| w8a8 | `0.74 GiB / 1.05 GiB`           | `19.46 GiB`            |

这里最重要的结论是：

- 量化显著降低了模型加载后的常驻权重显存；
- runtime peak 没有同步大幅下降，不是量化没生效，而是因为推理时的大头已经是 KV cache 和运行时 buffer。

换句话说，这次量化最直接的显存收益体现在“权重存储”，而不是“把整条推理链路的峰值显存一起打下来”。

### 吞吐与 TTFT


| 模式 | total throughput | prefill throughput | decode throughput | TTFT      |
| ---- | ---------------- | ------------------ | ----------------- | --------- |
| bf16 | `3069.97 tok/s`  | `38256.06 tok/s`   | `3530.05 tok/s`   | `2.69 s`  |
| int8 | `2634.29 tok/s`  | `12067.17 tok/s`   | `4082.19 tok/s`   | `12.38 s` |
| w8a8 | `2699.93 tok/s`  | `17385.02 tok/s`   | `3614.86 tok/s`   | `6.95 s`  |

相对 bf16：

- INT8 total throughput：`0.858x`
- W8A8 total throughput：`0.879x`
- INT8 decode throughput：`1.156x`
- W8A8 decode throughput：`1.024x`

这个结果很有代表性：

- `INT8` 在 decode 阶段已经超过了 bf16；
- `W8A8` 在 decode 上也略优于 bf16；
- 但 total throughput 与 TTFT 都被 prefill 明显拖住了。

所以这次工作不是“量化完全没收益”，而是收益主要已经出现在 decode 侧；只是端到端体验还被 prefill 阶段压着。

### 质量检查

质量评测目前已经打通四层链路：

- artifact check
- logits check
- PPL check
- generation check

#### 1. Artifact check

- `112 / 112` 目标层全部通过；
- mean relative L2 reconstruction error：`0.00949`
- min cosine similarity：`0.999903`

这说明量化产物本身是对的，不是文件错、shape 错或者 packed 顺序错。

#### 2. Logits check

- INT8：cosine mean `0.999398`，top1 agreement `1.0`
- W8A8：cosine mean `0.996619`，top1 agreement `1.0`

这说明在测试 prompt 上，量化后的下一 token 选择和 bf16 是一致的，INT8 尤其稳定。

#### 3. PPL

WikiText-2 上的最新 PPL 结果如下：


| 模式 | Loss     | PPL       | 相对 bf16 变化 |
| ---- | -------- | --------- | -------------- |
| bf16 | `3.6475` | `38.3786` | `-`            |
| int8 | `3.6559` | `38.7034` | `+0.85%`       |
| w8a8 | `3.6797` | `39.6332` | `+3.27%`       |

这组数字很关键，因为它说明：

- INT8 几乎保住了原模型语言建模能力；
- W8A8 也可用，但质量退化明显大于 INT8。

#### 4. Generation

generation check 已通过，没有出现明显异常，如：

- 空输出
- 提前崩坏
- 重复 token 失控

但长链生成中，W8A8 比 INT8 更容易在后半段逐步分叉，这和 PPL 结果是一致的。

### 结论：这次量化的成就是什么

如果把这次量化工作的成就收敛成几句话，可以这样说：

1. 我已经把 Nano-vLLM 的量化做成了一条完整工程链路，而不只是一个实验脚本。
2. INT8 weight-only 已经是一个成熟结果：
   - 降低加载显存；
   - 保住主要质量；
   - decode 性能已经超过 bf16。
3. W8A8 已经证明“实现链路跑通”，但当前实现下性价比还不如 INT8。

## 为什么会得到这个结果

这是面试时最值得讲清楚的一段，因为它体现你不是只会报数字，而是能解释系统行为。

### 1. 量化只覆盖了 Linear / GEMM，没有覆盖 attention / KV cache

这次量化工作的优化重点在线性层，因此收益主要落在线性层 GEMM 上；但 attention、KV cache 读写、flash-attn、norm、sampling 这些成本仍然存在。
因此，不可能期待“只改 Linear，就让整个推理系统所有阶段一起飞”。

### 2. benchmark 的 total throughput 对 prefill 很敏感

当前 benchmark 的总吞吐统计方式，把 prefill 时间和 decode 时间一起算进总耗时，但指标本身又很容易被 prefill 阶段放大影响。
所以一旦 prefill 变慢，总吞吐就会看起来比较难看。

### 3. 小模型下 bf16 本来就很强

Qwen3-0.6B 这个规模在 A10 上，bf16 已经很能打。
这意味着量化不是“天然稳赢”，特别是在 prefill 这种大 shape 场景下，bf16 tensor core 路径本来就可能很高效。

### 4. W8A8 的动态激活量化会额外引入 prefill 成本

`W8A8` 虽然理论上更完整，但它也多做了一步运行时激活量化。
在 prefill 阶段，输入 token 数多、矩阵更大，这部分动态量化的成本会放大，所以它没有把理论收益全部转成端到端吞吐优势。

### 5. 省下的权重显存没有自动变成更大 batch 上限

量化后虽然加载显存下降了，但当前实验配置下 `max_num_batched_tokens` 没变，runtime peak 又主要由 KV cache 主导。
所以省下来的权重显存，并没有直接换成更高 batch 上限或更大端到端吞吐。

## 面试表达模板

### 一句话版本

我在 Nano-vLLM 里做了两条量化路径：一条是 INT8 weight-only 的 `W8A16`，一条是激活也进 INT8 计算链路的 `W8A8`。我不仅做了量化本身，还把离线量化、运行时加载、TP 兼容、benchmark 和质量评测都打通了。最终结果是 INT8 已经比较成熟，显著降低加载显存、保住质量，而且 decode 性能已经超过 bf16。

### 三分钟版本

这个项目里我做的不是单点写一个量化 kernel，而是把 Nano-vLLM 的量化工程链路补完整。
我先在现有 `LinearBase + Tensor Parallel + packed module` 框架上落了 `W8A16`，也就是 INT8 weight-only，把 `qkv_proj`、`o_proj`、`gate_up_proj`、`down_proj` 这些主干线性层做成离线量化产物，运行时通过 loader 和 quantization dispatch 加载。然后我再往前推进到 `W8A8`，让激活在运行时按行动态量化成 `int8`，和离线保存的 `qweight + w_scales` 一起进入 `int8 x int8 -> int32 accum` 的 kernel 路径。

我刻意没有碰 attention、KV cache 和 flash-attn 接口，因为第一版如果同时动线性层和 KV cache，问题边界会很难收敛。最后我用 benchmark 和 quality report 把结果闭环起来。实验上，量化把模型加载显存从 `1.13/1.48 GiB` 降到了 `0.74/1.05 GiB`。INT8 在 decode 上已经达到 bf16 的 `1.156x`，而且 PPL 相对 bf16 只上升了 `0.85%`，说明质量保持得很好。现在的问题主要在 prefill，所以 total throughput 和 TTFT 还没赢过 bf16。整体判断是：INT8 已经是一个成熟且均衡的方案，W8A8 链路也打通了，但当前性价比还不如 INT8。

### 深挖细节版本

如果面试官继续追问，我会重点展开这几点：

1. **为什么只量化线性层**

   - 因为 Nano-vLLM 的主要 GEMM 都收敛在 `LinearBase` 体系里，而 attention / KV cache 改动面太大。
2. **W8A16 怎么量化**

   - 对称 INT8、per-output-channel；
   - 离线生成 `qweight + scales`；
   - 运行时优先走 fused kernel，不行就 fallback 到 `dequant + F.linear`。
3. **W8A8 激活怎么量化**

   * **W8A8** 激活量化是**运行时动态量化**

   - 输入先 reshape 成二维；
   - 对每一行先取 `abs_max`，再按 `abs_max / 127` 计算动态 scale；
   - 按行动态量化出 `x_q + a_scales`；
   - 和 `qweight + w_scales` 一起进 kernel；
   - epilogue 里乘 `a_scales * w_scales` 并融合 bias。
4. **为什么 decode 变快但 total 没变快**

   - 因为量化收益主要在 Linear/GEMM 上，decode 阶段更吃这个；
   - prefill 还受 attention、激活量化和大 shape 影响；
   - benchmark 总吞吐又对 prefill 很敏感，所以端到端结果被 prefill 拉住了。
5. **最终结论**

   - 当前应主推 INT8，不主推 W8A8 作为默认部署方案；
   - 如果继续做，下一步应该重点优化 prefill，而不是继续在质量上纠缠。

## 面试时的结果解读口径

### 成就

- 完整打通量化工程链路；
- 显著降低加载显存；
- 保住 INT8 质量；
- decode 阶段已经看到量化性能收益。

### 不足

- prefill 仍然是瓶颈；
- TTFT 变差；
- W8A8 当前实现成本高于收益。

### 原因

- attention / KV cache 未量化；
- prefill 中激活量化开销明显；
- benchmark workload 对 prefill 敏感；
- 小模型下 bf16 本身已经很强。

### 最终结论

当前仓库里最值得对外主推的是 `INT8 weight-only (W8A16)`，因为它在显存、质量和 decode 性能之间给出了最均衡的结果；`W8A8` 更适合作为已经跑通的进阶路线，而不是默认部署方案。
