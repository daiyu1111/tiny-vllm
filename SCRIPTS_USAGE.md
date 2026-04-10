# Scripts 使用说明

## `scripts/` 目录里有什么

当前 `scripts/` 目录下主要有两个脚本：

### 1. `scripts/benchmark_cases.py`

这个脚本不是用来直接跑 benchmark 的，而是用来定义和组织测试用例。

它负责：

- 定义 `BenchmarkCase` 数据结构
- 构造固定的文本测试集
- 构造固定 seed 的合成 token 压测集
- 提供 `get_default_cases(model_path)` 给其他脚本调用

当前默认内置的 case 包括：

- `short_input_short_output`
- `short_input_long_output`
- `long_input_short_output`
- `long_input_long_output`
- `synthetic_throughput`

你可以把它理解成“测试样本仓库”，而不是“执行器”。

### 2. `scripts/benchmark_baseline.py`

这个脚本才是真正用来执行基线测试的。

它负责：

- 加载模型
- 调用 `LLM` / `LLMEngine` 主路径执行推理
- 读取 `benchmark_cases.py` 提供的默认 case
- 记录吞吐、prefill / decode 指标、显存和输出样例
- 将结果保存为 JSON 文件

如果你要实际跑量化前基线测试，应该运行这个脚本，而不是运行 `benchmark_cases.py`。

## 我该怎么用

### 第一步：进入项目根目录

建议始终从项目根目录运行，而不是停留在 `script/` 或 `scripts/` 子目录中：

```bash
cd /mnt/workspace/nano-vllm-main
```

### 第二步：准备模型路径

你必须明确本地模型路径，例如：

```bash
/mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B
```

脚本运行时必须传 `--model`，否则会报错：

```bash
error: the following arguments are required: --model
```

### 第三步：运行最基本的基线测试

最推荐先跑这一条：

```bash
python scripts/benchmark_baseline.py \
  --model /mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B \
  --mode both
```

这条命令会做这些事：

- 加载默认 benchmark cases
- 跑 `eager` 模式
- 跑 `graph` 模式
- 打印摘要结果
- 把完整结果写入 `artifacts/baseline/`

## 常见使用方式

### 1. 跑全部默认 case

```bash
python scripts/benchmark_baseline.py \
  --model /mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B
```

默认 `--mode` 是 `both`，也就是同时跑 `eager` 和 `graph`。

### 2. 只跑 eager

```bash
python scripts/benchmark_baseline.py \
  --model /mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B \
  --mode eager
```

### 3. 只跑 graph

```bash
python scripts/benchmark_baseline.py \
  --model /mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B \
  --mode graph
```

### 4. 只跑某一个 case

例如只跑短输入短输出场景：

```bash
python scripts/benchmark_baseline.py \
  --model /mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B \
  --case short_input_short_output
```

### 5. 跑多个 case

```bash
python scripts/benchmark_baseline.py \
  --model /mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B \
  --case short_input_short_output \
  --case synthetic_throughput
```

### 6. 指定张量并行数

```bash
python scripts/benchmark_baseline.py \
  --model /mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B \
  --tensor-parallel-size 1
```

### 7. 指定输出目录

```bash
python scripts/benchmark_baseline.py \
  --model /mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B \
  --output-dir artifacts/baseline
```

## 输出结果在哪里

运行完成后，结果会写到：

```bash
artifacts/baseline/
```

输出文件名类似：

```bash
baseline_qwen3-0___6b_both_20260409_153000.json
```

JSON 中会包含：

- 模型路径
- 运行模式
- 每个 case 的吞吐
- prefill 吞吐
- decode 吞吐
- 显存统计
- 样例输出文本
- 汇总统计

## 常见问题

### 1. 为什么 `python benchmark_cases.py` 没有任何输出

因为 `benchmark_cases.py` 只是“测试用例定义模块”，里面没有 `main()`，也没有 `print()`。

它本来就不是直接执行用的，正确用法是让 `benchmark_baseline.py` 去调用它。

### 2. 为什么 `python benchmark_baseline.py` 会报必须传 `--model`

因为脚本设计上要求你显式告诉它“测哪个模型”。

正确写法：

```bash
python scripts/benchmark_baseline.py --model <模型目录>
```

### 3. 为什么 `--case` 不能单独写

因为 `--case` 后面必须跟一个 case 名字。

错误写法：

```bash
python scripts/benchmark_baseline.py --case
```

正确写法：

```bash
python scripts/benchmark_baseline.py \
  --model <模型目录> \
  --case short_input_short_output
```

### 4. 为什么会报 `ModuleNotFoundError: No module named 'nanovllm'`

通常是因为你不在项目根目录执行，或者路径环境不对。

建议统一这样做：

```bash
cd /mnt/workspace/nano-vllm-main
python scripts/benchmark_baseline.py --model <模型目录>
```

## 最推荐的起步命令

如果你现在只是想先把基线跑起来，最推荐直接执行这一条：

```bash
cd /mnt/workspace/nano-vllm-main

python scripts/benchmark_baseline.py \
  --model /mnt/workspace/nano-vllm-main/Qwen3-0.6B/qwen/Qwen3-0___6B \
  --mode both
```

跑完以后再检查：

```bash
ls artifacts/baseline
```

如果目录下出现了新的 JSON 文件，就说明基线测试已经跑通。
