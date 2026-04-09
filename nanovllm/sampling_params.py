from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0#温度参数。控制回答的“创造力”或“发散程度”。值越大越随机，值越小越死板。
    max_tokens: int = 64#字数限制。规定模型这一条回答最多只能吐出多少个 Token（默认只让说 64 个字）。
    ignore_eos: bool = False#是否无视“句号”。正常情况下模型输出结束符（EOS）就会停下，如果设为 True，它就会一直结结巴巴往下编，直到达到 max_tokens 的上限。

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
#__post_init__ 规定 temperature 必须大于 1e-10。这意味着这个引擎强制要求使用“随机采样”，不允许使用绝对死板的“贪心解码 (greedy sampling)”。