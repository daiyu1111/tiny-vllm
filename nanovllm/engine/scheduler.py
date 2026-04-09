from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs # 最多同时运行几个序列，厨房同时最多做几道菜（并行度）
        self.max_num_batched_tokens = config.max_num_batched_tokens# 最多处理多少token，一次最多处理多少食材（GPU内存限制）
        self.eos = config.eos # 结束符token_id
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()# 尚未分配KV缓存的等待队列
        self.running: deque[Sequence] = deque()# 已经分配了KV缓存（`block_manager.allocate(seq)`），可以进行增量解码的序列


    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []#本次要执行的序列列表
        num_seqs = 0#当前批大小，不超过（max_num_seqs）
        num_batched_tokens = 0#当期批总token数（不能超过max_num_batched_tokens）
        while self.waiting and num_seqs < self.max_num_seqs:#只要还有等待的序列，且没到达最大批大小就一直取
            seq = self.waiting[0]# 查看队列头部，先不弹出，先检查资源够不够
            # 检查资源约束
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break# 资源不足，停止调度
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens#只计新token__，缓存的token不占用当前批处理容量
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):# 检查是否能追加token（是否需要新块）
                if self.running:
                    self.preempt(self.running.pop())# 抢占其他序列
                else:
                    self.preempt(seq)# 抢占自己
                    break
            else:# 可以追加
                num_seqs += 1
                self.block_manager.may_append(seq) # 处理块追加
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
