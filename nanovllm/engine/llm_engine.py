import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        #Config 是从外部导入进来的一个类，这段代码的目的就是把外部传入的参数整理好，用来实例化 config 这个组件。
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        #多进程
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id#结束符，并写入配制参数
        self.scheduler = Scheduler(config)#这份完善后的配置说明书 config 交给实例Scheduler化的调度器self.scheduler
        atexit.register(self.exit)#"at exit"（在退出时）。调用self.exit

    def exit(self):
        self.model_runner.call("exit")#主进程向子进程发出退出信号，子进程会自己用自己的退出逻辑
        del self.model_runner#理主进程（包工头自己）的内存
        for p in self.ps:#根据花名册，join 的意思是**“阻塞并等待，直到这个进程（线程）完全结束”**。
            p.join()

    #每当有一个用户带着他的问题（prompt）来找大模型时，这个函数就会被调用
    # prompt：用户想问的问题，人类文字（str），也可以是已经被翻译好的数字代码（list[int]）。
    #sampling_params：temperature，max_tokens等
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):#是文字就token化
            prompt = self.tokenizer.encode(prompt)
        #把翻译好的数字题目（prompt）和用户的定制要求（sampling_params）一起装进这个标准的文件袋里，
        # 并给它盖上一个专属的流水号。在引擎里，每一个正在处理的请求都被称为一个 Sequence（序列）。
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)#self.scheduler把这个工单放进等待队列里

    def step(self):
        # 1. 调度分配：车间主任（Scheduler）决定这一步要处理哪些请求（seqs）
        # 返回的 is_prefill 标志用来区分是“读题阶段”（处理长 prompt）还是“作答阶段”（逐字生成）
        seqs, is_prefill = self.scheduler.schedule()

        # 2. 并行计算：包工头（Model Runner）拉起所有 GPU 工人执行前向传播
        # 根据送来的序列和所处阶段，算出这些请求对应的下一个字的代码（token_ids）
        token_ids = self.model_runner.call("run", seqs, is_prefill)

        # 3. 验收更新：车间主任把新算出来的字填进各自的工单里
        # 同时检查这些请求是不是已经输出了“句号”（EOS），如果完成则打上 is_finished 标记
        self.scheduler.postprocess(seqs, token_ids)

        # 4. 挑出成品：遍历刚才处理的工单，把所有盖了 is_finished 完工章的挑出来
        # 提取出它们的订单号（seq_id）和最终生成的完整内容（completion_token_ids）
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]

        # 5. 统计业绩 (KPI)：计算这一拍处理了多少个 Token，用于评估引擎速度
        # 巧妙之处：如果是 prefill 阶段，就累加所有 prompt 的长度；
        # 如果是 decode 阶段，为了区分，用负数表示（一次生成 len(seqs) 个新字）
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:

        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
