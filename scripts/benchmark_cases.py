from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
import sys

from transformers import AutoTokenizer, PreTrainedTokenizerBase

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanovllm import SamplingParams


@dataclass
class BenchmarkCase:
    name: str
    kind: str
    description: str
    prompts: list[str] | list[list[int]]
    sampling_params: list[SamplingParams]
    metadata: dict


def _format_chat_prompt(tokenizer: PreTrainedTokenizerBase, content: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return content


def build_text_cases(tokenizer: PreTrainedTokenizerBase) -> list[BenchmarkCase]:
    prompt_sets = [
        (
            "short_input_short_output",
            "短输入、短输出，观察低延迟场景。",
            [
                "用一句话介绍你自己。",
                "北京和上海最大的区别是什么？请简短回答。",
                "列出三个适合晨练的轻运动。",
            ],
            64,
        ),
        (
            "short_input_long_output",
            "短输入、长输出，观察 decode 主导场景。",
            [
                "请写一篇关于张量并行与数据并行差异的中文说明，要求结构清晰。",
                "请系统介绍一下 Transformer 推理时 prefill 和 decode 两个阶段分别在做什么。",
                "请解释大模型推理中 KV Cache 的作用、收益和限制。",
            ],
            256,
        ),
        (
            "long_input_short_output",
            "长输入、短输出，观察 prefill 主导场景。",
            [
                "下面是一段关于大模型推理引擎设计的说明，请阅读后只用两句话总结重点：\n"
                "一个高吞吐推理引擎通常需要在调度、显存管理、算子融合和通信层面协同优化。"
                "Prefill 阶段关注大批量 prompt 的并行编码，Decode 阶段关注低延迟逐 token 生成。"
                "Paged KV Cache 可以显著提升内存利用率，但也会给 block 管理和调度带来复杂性。"
                "如果进一步引入量化，则需要在精度、显存、吞吐与工程复杂度之间平衡。",
                "请阅读以下长说明，并只输出一个精简结论：\n"
                "权重量化通常优先于激活量化，因为它对现有推理路径侵入更小。"
                "如果项目已经将主要 GEMM 抽象收敛在线性层，那么 weight-only quantization 往往可以更快打通。"
                "反过来，KV Cache 量化通常需要同时处理写入、读取、反量化和注意力算子兼容性问题。",
                "请概括下面的系统设计描述：\n"
                "基线测试在量化前非常关键，因为没有稳定基线就无法判断量化到底带来了显存收益还是只是改变了调度行为。"
                "因此需要对吞吐、延迟、显存和输出质量建立统一记录格式，并尽量复用真实推理入口。",
            ],
            48,
        ),
        (
            "long_input_long_output",
            "长输入、长输出，观察综合吞吐场景。",
            [
                "下面是一段关于推理系统优化路线的长背景，请基于它写一篇完整的中文说明，包含问题背景、设计权衡和后续路线：\n"
                "一个教学向但追求较高性能的推理项目，通常会先实现最必要的执行链路，再逐步补调度优化、KV Cache 管理、张量并行和 CUDA Graph。"
                "当它开始考虑量化时，最容易落地的往往不是最激进的方案，而是对主干矩阵乘法侵入最小的方案。"
                "因此，建立可重复的 benchmark 与结果记录体系，会成为后续所有性能优化工作的共同基础。",
                "请阅读以下材料，并给出一份结构化总结：\n"
                "Prefill 阶段通常吞吐较高，但受 prompt 长度和批大小影响明显；"
                "Decode 阶段每步 token 数少，但对端到端体验影响更直接；"
                "显存监控不仅要看峰值，还要看模型加载后和 KV Cache 分配后的常驻状态；"
                "如果没有固定测试用例，同一个优化在不同输入分布下可能会呈现完全不同的收益。",
            ],
            192,
        ),
    ]

    cases = []
    for name, description, prompts, max_tokens in prompt_sets:
        formatted_prompts = [_format_chat_prompt(tokenizer, prompt) for prompt in prompts]
        sampling_params = [SamplingParams(temperature=0.6, max_tokens=max_tokens) for _ in formatted_prompts]
        input_lengths = [len(tokenizer.encode(prompt)) for prompt in formatted_prompts]
        cases.append(
            BenchmarkCase(
                name=name,
                kind="text",
                description=description,
                prompts=formatted_prompts,
                sampling_params=sampling_params,
                metadata={
                    "request_count": len(formatted_prompts),
                    "input_length_min": min(input_lengths),
                    "input_length_max": max(input_lengths),
                    "output_length_max": max_tokens,
                    "seed": None,
                },
            )
        )
    return cases


def build_synthetic_case(
    seed: int = 0,
    num_seqs: int = 128,
    min_input_len: int = 128,
    max_input_len: int = 512,
    min_output_len: int = 128,
    max_output_len: int = 512,
    vocab_high: int = 10000,
) -> BenchmarkCase:
    rng = Random(seed)
    prompts = [
        [rng.randint(0, vocab_high) for _ in range(rng.randint(min_input_len, max_input_len))]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=rng.randint(min_output_len, max_output_len),
        )
        for _ in range(num_seqs)
    ]
    return BenchmarkCase(
        name="synthetic_throughput",
        kind="synthetic_token",
        description="固定 seed 的合成 token 吞吐压测。",
        prompts=prompts,
        sampling_params=sampling_params,
        metadata={
            "seed": seed,
            "request_count": num_seqs,
            "input_length_min": min_input_len,
            "input_length_max": max_input_len,
            "output_length_min": min_output_len,
            "output_length_max": max_output_len,
            "vocab_high": vocab_high,
        },
    )


def get_default_cases(model_path: str) -> list[BenchmarkCase]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    cases = build_text_cases(tokenizer)
    cases.append(build_synthetic_case())
    return cases
