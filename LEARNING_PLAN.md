# Nano-vLLM 项目结构与学习计划

## 一、项目概述

**Nano-vLLM** 是一个从头实现的轻量级 vLLM 推理引擎，约 1,200 行代码，具有以下特点：

- 🚀 快速离线推理（性能媲美 vLLM）
- 📖 代码简洁易读
- ⚡ 支持 Prefix Caching、Tensor Parallelism、CUDA Graph 等优化

---

## 二、项目结构

```
nano-vllm/
├── nanovllm/                      # 核心代码目录
│   ├── __init__.py                # 导出 LLM 类
│   ├── llm.py                     # LLM 引擎入口（LLMEngine 包装）
│   ├── config.py                  # 配置类（Config）
│   ├── sampling_params.py         # 采样参数配置
│   │
│   ├── engine/                    # 引擎核心模块
│   │   ├── llm_engine.py          # 主引擎：进程管理、请求调度
│   │   ├── model_runner.py        # 模型执行：CUDA Graph、KV Cache 管理
│   │   ├── scheduler.py           # 调度器：Prefill/Decode 调度
│   │   ├── block_manager.py       # 块管理器：KV Cache 块分配
│   │   └── sequence.py            # 序列定义：Sequence 状态管理
│   │
│   ├── layers/                    # 神经网络层
│   │   ├── linear.py              # 张量并行线性层
│   │   ├── attention.py           # Flash Attention 封装
│   │   ├── sampler.py             # 采样器（Gumbel-Max）
│   │   ├── activation.py          # 激活函数（SiLU & Mul）
│   │   ├── layernorm.py           # RMSNorm
│   │   ├── rotary_embedding.py    # RoPE 旋转编码
│   │   └── embed_head.py          # 词嵌入与 LM Head
│   │
│   ├── models/                    # 模型实现
│   │   └── qwen3.py               # Qwen3 模型架构
│   │
│   └── utils/                     # 工具模块
│       ├── context.py             # 上下文管理（Prefill/Decode 状态）
│       └── loader.py              # 模型权重加载
│
├── example.py                     # 使用示例
├── bench.py                       # 性能测试脚本
└── pyproject.toml                 # 项目配置
```

---

## 三、核心架构

```
┌─────────────────────────────────────────────────────────────┐
│                        LLM (用户接口)                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     LLMEngine (主引擎)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Tokenizer   │  │ Scheduler   │  │ ModelRunner (TP=0)  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                           │                   │              │
│                           │          ┌────────┴────────┐     │
│                           │          │ ModelRunner     │     │
│                           │          │ (TP Worker 1-7) │     │
│                           ▼          └─────────────────┘     │
│                    ┌─────────────┐                           │
│                    │BlockManager │                           │
│                    │ KV Cache 管理│                          │
│                    └─────────────┘                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Qwen3ForCausalLM                         │
│  ┌───────────┐  ┌───────────────┐  ┌───────────────────┐   │
│  │Embedding  │  │Decoder Layers │  │  LM Head + Sampler│   │
│  └───────────┘  └───────────────┘  └───────────────────┘   │
│                      │  Attention + MLP                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 四、学习计划

### 📌 阶段一：基础理解（1-2 天）

**目标**：理解项目整体架构和使用方式


| 序号 | 学习内容 | 阅读文件                          | 关键问题                 |
| ---- | -------- | --------------------------------- | ------------------------ |
| 1    | 项目使用 | `example.py`, `README.md`         | 如何初始化和生成？       |
| 2    | 入口流程 | `llm.py`, `llm_engine.py`         | LLMEngine 如何管理进程？ |
| 3    | 配置系统 | `config.py`, `sampling_params.py` | 有哪些可配置项？         |

**实践任务**：

- [X]  运行 `example.py` 查看生成效果
- [X]  修改参数观察输出变化

---

### 📌 阶段二：调度系统（2-3 天）

**目标**：理解请求调度和 KV Cache 管理


| 序号 | 学习内容     | 阅读文件                 | 关键问题                     |
| ---- | ------------ | ------------------------ | ---------------------------- |
| 1    | 序列管理     | `sequence.py`            | Sequence 有哪些状态？        |
| 2    | 调度逻辑     | `scheduler.py`           | Prefill 和 Decode 如何调度？ |
| 3    | 块管理       | `block_manager.py`       | KV Cache 块如何分配/回收？   |
| 4    | Prefix Cache | `block_manager.py:36-82` | 如何实现前缀缓存？           |

**关键概念**：

- **Block Table**：每个序列的 KV Cache 块映射表
- **Prefix Caching**：通过哈希复用相同前缀的 KV Cache
- **抢占调度**：当显存不足时如何处理

**实践任务**：

- [X]  绘制调度流程图
- [X]  添加日志观察调度过程

---

### 📌 阶段三：模型执行（3-4 天）

**目标**：理解模型推理和 CUDA 优化


| 序号 | 学习内容   | 阅读文件                  | 关键问题                      |
| ---- | ---------- | ------------------------- | ----------------------------- |
| 1    | 模型执行   | `model_runner.py`         | 如何组织 Prefill/Decode？     |
| 2    | CUDA Graph | `model_runner.py:216-251` | 如何捕获和重放计算图？        |
| 3    | KV Cache   | `model_runner.py:100-118` | 如何分配 KV Cache？           |
| 4    | 张量并行   | `model_runner.py:26-48`   | 多 GPU 如何通信？             |
| 5    | 上下文管理 | `context.py`              | Prefill/Decode 状态如何传递？ |

**关键概念**：

- **Prefill 阶段**：处理提示词，使用 varlen attention
- **Decode 阶段**：逐 token 生成，使用 kvcache attention
- **CUDA Graph**：减少 kernel 启动开销

**实践任务**：

- [ ]  对比 enable/disable CUDA Graph 的性能
- [ ]  绘制数据流图

---

### 📌 阶段四：模型架构（2-3 天）

**目标**：理解 Qwen3 模型实现


| 序号 | 学习内容  | 阅读文件                     | 关键问题                   |
| ---- | --------- | ---------------------------- | -------------------------- |
| 1    | 整体架构  | `models/qwen3.py`            | DecoderLayer 如何组织？    |
| 2    | Attention | `layers/attention.py`        | Flash Attention 如何集成？ |
| 3    | MLP       | `qwen3.py:90-116`            | MoE 还是 Dense？           |
| 4    | 位置编码  | `layers/rotary_embedding.py` | RoPE 如何实现？            |
| 5    | 采样器    | `layers/sampler.py`          | 如何实现温度采样？         |

**实践任务**：

- [ ]  绘制模型架构图
- [ ]  尝试添加新模型支持

---

### 📌 阶段五：张量并行（2-3 天）

**目标**：理解分布式推理实现


| 序号 | 学习内容   | 阅读文件           | 关键问题                   |
| ---- | ---------- | ------------------ | -------------------------- |
| 1    | 并行线性层 | `layers/linear.py` | Column/Row Parallel 区别？ |
| 2    | QKV 投影   | `linear.py:96-128` | QKV 如何切分？             |
| 3    | 进程通信   | `model_runner.py`  | 如何同步多卡？             |
| 4    | 权重加载   | `utils/loader.py`  | 如何加载切分权重？         |

**关键概念**：

- **Column Parallel**：输出维度切分（QKV 投影）
- **Row Parallel**：输入维度切分（输出投影）
- **AllReduce**：合并部分结果

---

### 📌 阶段六：深入优化（3-5 天）

**目标**：理解性能优化技术


| 序号 | 学习内容      | 关键问题                                  |
| ---- | ------------- | ----------------------------------------- |
| 1    | Triton Kernel | `attention.py:10-40` store_kvcache_kernel |
| 2    | Torch Compile | `sampler.py:10` @torch.compile            |
| 3    | 共享内存通信  | `model_runner.py:68-83`                   |
| 4    | 显存管理      | `model_runner.py:100-118`                 |

---

## 五、代码阅读顺序推荐

```
1. example.py                    → 了解使用方式
2. sampling_params.py            → 了解采样配置
3. sequence.py                   → 了解序列定义
4. config.py                     → 了解全局配置
5. llm_engine.py                 → 了解主流程
6. scheduler.py + block_manager.py → 理解调度
7. context.py                    → 理解上下文传递
8. model_runner.py               → 理解模型执行
9. models/qwen3.py               → 理解模型架构
10. layers/*.py                  → 理解各层实现
11. layers/linear.py             → 理解张量并行
12. utils/loader.py              → 理解权重加载
```

---

## 六、关键知识点

### 6.1 核心数据结构


| 结构       | 作用             | 位置               |
| ---------- | ---------------- | ------------------ |
| `Sequence` | 表示一个生成请求 | `sequence.py`      |
| `Block`    | KV Cache 块      | `block_manager.py` |
| `Context`  | 前向传播上下文   | `context.py`       |

### 6.2 核心算法


| 算法                | 描述                      | 位置                      |
| ------------------- | ------------------------- | ------------------------- |
| **调度算法**        | Prefill 优先，Decode 抢占 | `scheduler.py:24-58`      |
| **Prefix Cache**    | 基于哈希的块复用          | `block_manager.py:59-82`  |
| **CUDA Graph**      | 捕获静态图减少开销        | `model_runner.py:216-251` |
| **Gumbel-Max 采样** | 无核采样实现              | `sampler.py:14`           |

### 6.3 优化技术


| 技术            | 效果             | 实现位置              |
| --------------- | ---------------- | --------------------- |
| Prefix Caching  | 复用相同前缀     | `block_manager.py`    |
| Tensor Parallel | 多 GPU 推理      | `layers/linear.py`    |
| CUDA Graph      | 减少 kernel 启动 | `model_runner.py`     |
| Torch Compile   | 算子融合         | `sampler.py`          |
| Flash Attention | 加速 Attention   | `layers/attention.py` |

---

## 七、扩展阅读

- [vLLM 官方文档](https://docs.vllm.ai/)
- [Flash Attention 论文](https://arxiv.org/abs/2205.14135)
- [Megatron-LM 张量并行](https://github.com/NVIDIA/Megatron-LM)
- [CUDA Graph 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)

---

## 八、学习检查清单

### 基础理解

- [ ]  能解释 LLM.generate 的完整流程
- [ ]  能画出系统架构图
- [ ]  理解多进程如何协作

### 调度系统

- [ ]  理解 Prefill 和 Decode 的区别
- [ ]  能解释 Block Table 的作用
- [ ]  理解 Prefix Cache 原理

### 模型执行

- [ ]  理解 CUDA Graph 如何加速
- [ ]  理解 KV Cache 如何管理
- [ ]  理解张量并行如何通信

### 模型架构

- [ ]  理解 Qwen3 的 Decoder 结构
- [ ]  理解 RoPE 位置编码
- [ ]  理解采样器实现

---

*生成时间：2026-03-25*
