# tiny-vLLM

A lightweight vLLM implementation built 

## Key Features

* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.
* 🚀 W8A16,W8A8

## Model Download

To download the model weights manually, use the following command:

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:

```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

- Hardware: NVIDIA A10 (20GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens


| Inference Engine | Output Tokens     | Time (s) | Throughput (tokens/s) |
| ---------------- | ----------------- | -------- | --------------------- |
| tiny-vLLM        | 133966tok, 38.43s | 38.43s,  | 3486.41tok/s          |
|                  |                   |          |                       |

## Quantization Results

The repository includes INT8 weight-only and W8A8 quantization paths for Qwen3-0.6B. The following results were measured on an NVIDIA A10 with `bench_quant.py` and `bench_quant_quality.py`.

### Memory

Quantization clearly reduces model load-time memory:

| Mode | Memory After Load (allocated / reserved) | Runtime Peak Allocated |
| ---- | ---------------------------------------- | ---------------------- |
| bf16 | 1.13 GiB / 1.48 GiB                      | 19.60 GiB              |
| int8 | 0.74 GiB / 1.05 GiB                      | 19.47 GiB              |
| w8a8 | 0.74 GiB / 1.05 GiB                      | 19.46 GiB              |

Load-time memory drops significantly after quantization. Runtime peak memory changes less because KV cache and other runtime buffers dominate the total footprint.

### Throughput and TTFT

| Mode | Total Throughput | Prefill Throughput | Decode Throughput | TTFT |
| ---- | ---------------- | ------------------ | ----------------- | ---- |
| bf16 | 3069.97 tok/s    | 38256.06 tok/s     | 3530.05 tok/s     | 2.69 s |
| int8 | 2634.29 tok/s    | 12067.17 tok/s     | 4082.19 tok/s     | 12.38 s |
| w8a8 | 2699.93 tok/s    | 17385.02 tok/s     | 3614.86 tok/s     | 6.95 s |

Relative to bf16:

- INT8 total throughput: `0.858x`
- W8A8 total throughput: `0.879x`
- INT8 decode throughput: `1.156x`
- W8A8 decode throughput: `1.024x`

Quantized kernels already help decode, but prefill remains the main bottleneck in the current implementation. As a result, TTFT and end-to-end throughput do not yet beat bf16 on this workload.

### Quality Check

Quality was evaluated with artifact validation, logits comparison, greedy generation, and perplexity on WikiText-2.

Artifact check:

- `112 / 112` target layers passed
- Mean relative L2 reconstruction error: `0.00949`
- Minimum cosine similarity: `0.999903`

Logits check:

- INT8: cosine mean `0.999398`, top-1 agreement `1.0`
- W8A8: cosine mean `0.996619`, top-1 agreement `1.0`

Perplexity:

| Mode | Loss | PPL | PPL Delta vs bf16 |
| ---- | ---- | --- | ----------------- |
| bf16 | 3.6475 | 38.3786 | - |
| int8 | 3.6559 | 38.7034 | +0.85% |
| w8a8 | 3.6797 | 39.6332 | +3.27% |

Greedy generation remained stable in all checks, with INT8 staying closer to bf16 than W8A8 over long decoding runs.

### Takeaway

INT8 is currently the strongest trade-off in this repository:

- Lower load-time memory than bf16
- Better decode throughput than bf16
- Very small quality regression

W8A8 is functional and reasonably accurate, but in the current implementation it is less attractive than INT8 because it improves quality less and does not provide a larger end-to-end speedup.
