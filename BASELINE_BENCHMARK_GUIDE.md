# Nano-vLLM 量化前基线测试指导

## 概述

本文档用于指导后续实现一套“量化前基线测试代码”，目标是在量化改造开始之前，先稳定记录当前实现的吞吐、延迟、显存和输出质量，作为后续量化版本的对照基线。

这里先做的是“基线测试框架与记录规范”，不是直接实现量化功能。基线测试也必须尽量复用当前项目已有的推理入口，而不是另起一套与实际引擎行为不一致的测试逻辑。

建议新增一份固定的基线测试指导文档，放在仓库根目录，文件名使用：

```text
BASELINE_BENCHMARK_GUIDE.md
```

## 当前仓库中可复用的测试与观测入口

在正式实现基线测试代码之前，需要先明确当前仓库已经具备哪些可直接复用的能力。

### 1. `bench.py` 已经提供了批量吞吐测试雏形

[bench.py](/E:/llmegine/nano-vllm-main/nano-vllm-main/bench.py) 已经能完成一轮批量生成，并输出：

- 总输出 token 数
- 总耗时
- 总吞吐 `tokens/s`

但它当前仍然不够作为量化前后对比的正式基线工具，原因包括：

- 输入样本完全由随机 token 构造，不便于质量回归。
- 没有区分 prefill 和 decode 的吞吐。
- 没有记录显存信息。
- 没有保存结构化结果文件。
- 没有固定 case 集合和统一输出 schema。

因此，`bench.py` 可以继续保留为轻量示例，但后续正式的量化前后对比应迁移到新的基线脚本。

### 2. `LLMEngine.generate()` 已经暴露了 prefill / decode 观测点

从 [nanovllm/engine/llm_engine.py](/E:/llmegine/nano-vllm-main/nano-vllm-main/nanovllm/engine/llm_engine.py) 可以看到：

- `step()` 已经返回 `num_tokens`
- prefill 阶段用正数表示本轮处理的 token 数
- decode 阶段用负数表示本轮新生成的 token 数
- `generate()` 中已经临时计算过 `Prefill tok/s` 和 `Decode tok/s`

这说明当前引擎内部已经有天然的分段性能观测点。后续如果需要补更细粒度的统计，应优先在这条主路径上增加可选统计返回，而不是复制一份新的调度循环。

### 3. `ModelRunner` 的行为决定了测试必须区分 eager / graph 模式

从 [nanovllm/engine/model_runner.py](/E:/llmegine/nano-vllm-main/nano-vllm-main/nanovllm/engine/model_runner.py) 可以看到，模型初始化阶段会涉及：

- warmup
- KV cache 分配
- CUDA graph 捕获
- `enforce_eager=True/False` 两种不同运行方式

这意味着性能测试不能只记录单一模式，否则量化前后很难判断变化究竟来自量化本身，还是来自运行模式差异。

因此，基线测试至少要覆盖：

- `enforce_eager=True`
- `enforce_eager=False`

### 4. 当前 `SamplingParams` 禁止 greedy，质量对比不能依赖逐 token 完全一致

从 [nanovllm/sampling_params.py](/E:/llmegine/nano-vllm-main/nano-vllm-main/nanovllm/sampling_params.py) 可见，当前实现要求：

- `temperature > 1e-10`
- 不允许真正的 greedy sampling

这意味着量化前后的质量对比不应建立在“逐 token 完全一致”之上，而应采用更稳妥的比较方法，例如：

- 固定 prompt 下的输出文本样例
- 输出长度统计
- 语义可读性人工比对
- 后续可扩展的 logits 或 top-k 偏差接口

## 后续要实现什么测试代码

后续基线测试代码建议拆成两个脚本，而不是把所有逻辑塞进一个 benchmark 文件中。

### 1. `scripts/benchmark_cases.py`

职责：

- 管理固定测试用例
- 提供可重复的 prompts / prompt_token_ids / sampling 配置
- 统一维护 seed、长度区间、请求数等输入定义

不负责：

- 不实际运行 benchmark
- 不负责计时
- 不负责显存采集
- 不负责写结果文件

它的核心价值是保证每次 benchmark 使用同一套输入集合，避免测试结果因为输入变化而不可比。

### 2. `scripts/benchmark_baseline.py`

职责：

- 执行基线测试
- 调用 `LLM` 主路径完成推理
- 记录吞吐、耗时、显存和输出质量
- 生成控制台摘要和 JSON 结果文件

不负责：

- 不在内部随机生成新的基准样本
- 不重新实现调度或生成逻辑
- 不绕开 `LLM` / `LLMEngine` 主入口

### 3. `bench.py` 的定位

建议保留 [bench.py](/E:/llmegine/nano-vllm-main/nano-vllm-main/bench.py) 作为轻量吞吐示例，但后续正式对比应统一迁移到新的基线测试脚本中。这样可以避免：

- 指标口径不一致
- 输入集合不一致
- 结果格式不一致

## 需要记录的指标

为了让量化前后具备可比性，建议把指标分成三类。

### 一、性能指标

后续实现时至少记录以下性能指标：

- 总耗时
- 总输出 token 数
- 总吞吐 `tokens/s`
- prefill 吞吐 `prefill tokens/s`
- decode 吞吐 `decode tokens/s`
- 单次请求平均完成时间
- 首 token 延迟近似值

其中“首 token 延迟”第一版可以采用简化近似：

- 使用“prefill 阶段结束时间”近似 TTFT
- 先不要求做到逐请求级别的精确首 token 打点

这样可以先把基础对比能力建立起来，避免一开始就把统计系统做得过重。

### 二、资源指标

后续实现时至少记录以下显存指标：

- 模型加载后显存占用
- warmup 后显存占用
- KV cache 分配后的常驻显存
- 正式生成期间峰值显存

这些指标可以帮助后续回答两个关键问题：

- 量化是否真的降低了模型常驻显存
- 量化是否改变了运行期的峰值显存行为

### 三、质量基线指标

后续实现时建议记录以下质量信息：

- 固定测试 prompt 的生成文本
- 每条输出的 token 数
- 关键 prompt 的扩展比较接口

第一版建议最小可用质量基线是：

- 记录文本
- 记录输出长度

如果后续实现方便，再进一步扩展：

- 保存 logits 摘要
- 保存 top-k token 及其分数
- 记录与基线版本的偏差

## 测试输入集合设计

基线测试不能只依赖完全随机输入。建议同时保留“真实文本 case”和“合成 token case”两套输入集合。

### 1. 真实文本 case

真实文本用例用于观察质量回归和典型场景行为，至少包含四类 case：

- 短输入、短输出：验证低延迟场景
- 短输入、长输出：验证 decode 主导场景
- 长输入、短输出：验证 prefill 主导场景
- 长输入、长输出：验证综合吞吐场景

这些 case 建议由固定 prompt 构成，并在 `benchmark_cases.py` 中写死：

- prompt 内容
- sampling 参数
- 预期场景标签
- 用例说明

### 2. 合成 token case

合成 token case 用于稳定压测吞吐，减少 tokenizer 和自然语言分布波动的影响。

建议采用固定 seed 生成：

- `prompt_token_ids`
- 输入长度分布
- 输出长度分布
- 请求数量

合成 case 的目标不是看文本质量，而是给性能提供一个稳定、可重复的压力场景。

### 3. 输入集合的记录要求

无论是真实文本还是合成 token case，都应把以下元数据写入结果文件：

- case 名称
- case 类型：`text` 或 `synthetic_token`
- seed
- 请求数
- 输入长度区间
- 输出长度区间
- sampling 参数

这样后续量化版才能严格复现实验条件。

## 测试执行矩阵

为了保证量化前后可对齐，后续基线测试至少需要支持以下执行矩阵。

### 最小必跑矩阵

- 单卡，`enforce_eager=False`
- 单卡，`enforce_eager=True`

这是默认最小基线。即使环境只有一张卡，也应先保证这两组结果可以稳定记录。

### 条件允许时补充

- `tensor_parallel_size=2`
- 更高 TP 配置

但要注意，多卡不是第一阶段的必需前提。文档应明确：

- 单卡结果是最小可交付基线
- 多卡结果是条件允许时的增强项

## 结果输出格式

后续基线测试代码应把结果落成结构化文件，建议统一使用 JSON。

### 输出目录

建议固定输出到：

```text
artifacts/baseline/
```

### 文件命名

推荐输出文件名格式：

```text
baseline_<model>_<mode>_<timestamp>.json
```

例如：

```text
baseline_qwen3-0.6b_eager_20260409_153000.json
```

### 推荐 JSON 内容

建议结果文件至少包含以下字段：

- 测试时间
- 机器信息
- GPU 名称
- CUDA 版本
- PyTorch 版本
- 模型路径
- 测试参数
- case 列表
- 每个 case 的性能指标
- 每个 case 的显存指标
- 关键输出样例
- 汇总统计

### 控制台输出与文件输出分工

建议分工如下：

- 控制台：打印简洁摘要
- JSON：保存完整结果

这样既方便日常查看，也方便后续自动比较量化前后结果。量化版本也应沿用同样的结果 schema，避免后期再做结果格式迁移。

## 推荐实现方式

为了降低后续实现成本，建议先把实现边界写清楚。

### 1. 复用现有 `LLM` 主入口

后续测试代码应围绕 `LLM` 与 `LLMEngine` 主路径构建，不应绕开真实推理入口。这样才能确保：

- 测到的是实际运行路径
- eager / CUDA graph / scheduler / sampler 行为一致
- 量化前后比较有效

### 2. 不修改 `LLM.generate()` 的外部接口

建议不改变当前 `LLM.generate()` 的对外调用方式。测试脚本应优先采用：

- 外层计时
- 外层收集输出
- 外层保存结果

如果需要更细粒度的分段统计，应优先考虑：

- 在 `LLMEngine.step()` 中增加可选统计返回
- 或在 `generate()` 中增加默认关闭的统计选项

不建议复制调度循环来“模拟 generate”，否则后续容易和真实行为漂移。

### 3. 显存统计的推荐来源

后续记录显存时，优先使用以下 API：

- `torch.cuda.memory_allocated()`
- `torch.cuda.max_memory_allocated()`
- `torch.cuda.mem_get_info()`

如果需要统一口径，建议在文档里定义：

- “当前占用”使用 `memory_allocated`
- “峰值占用”使用 `max_memory_allocated`
- “总空闲/总容量”使用 `mem_get_info`

### 4. 分阶段实现策略

建议把实现拆成两个阶段。

第一阶段：

- 实现脚本级总耗时、总吞吐、显存记录
- 支持固定 case 集合
- 生成结构化 JSON 结果

第二阶段：

- 增加 step 级 prefill/decode 统计
- 增加更细粒度延迟统计
- 预留 logits / top-k 质量对比接口

第一版不要求接入复杂 profiler，也不要求一次性实现所有高精度统计。

## 实现验收标准

后续基线测试代码完成后，至少应满足以下验收条件：

- 运行一次基线脚本后，能够生成一份 JSON 结果文件。
- 结果中至少包含总吞吐、prefill 吞吐、decode 吞吐、总耗时、峰值显存。
- 同一组固定 case 多次运行时，输入集合和采样参数保持一致。
- 单卡模式下，`enforce_eager=True/False` 都能得到结果。
- 输出结果中包含至少 3 条固定文本样例，供量化前后人工比对。
- 若环境支持多卡，TP 模式结果与单卡结果使用同一结果 schema。

## 建议的后续落地顺序

为了让后续实现更顺畅，建议按以下顺序推进：

1. 新增 `scripts/benchmark_cases.py`，固定测试输入集合。
2. 新增 `scripts/benchmark_baseline.py`，实现外层执行和 JSON 记录。
3. 第一版先记录总吞吐和显存。
4. 第二版再扩展 prefill / decode 分段统计。
5. 最后再补更细粒度延迟和质量对比字段。

## 结论

在量化改造之前，先建立一套稳定的基线测试与记录规范是必要的。对于当前 Nano-vLLM 项目，最合适的做法不是直接重写 benchmark，而是在已有 `bench.py` 和 `LLMEngine` 观测点基础上，补一套：

- 固定输入集合
- 结构化结果输出
- 明确指标口径
- 可与量化版本共享的 benchmark schema

这样后续无论是实现 `W8A16` 还是继续扩展 `W4A16`，都能基于同一套基线数据做可解释的性能和质量对比。
