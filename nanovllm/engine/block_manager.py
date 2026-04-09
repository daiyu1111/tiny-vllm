from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size# 每个块能存多少token（256）
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]# 创建一堆空块
        self.hash_to_block_id: dict[int, int] = dict() # 哈希值 -> 块编号的映射（快速查找）
        self.free_block_ids: deque[int] = deque(range(num_blocks))# 空闲块的队列
        self.used_block_ids: set[int] = set()# 正在使用的块集合

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table# 确保序列还没有分配块
        h = -1#初始化哈希值，-1 表示"没有前一个块的哈希"
        cache_miss = False#缓存未命中标志。
        for i in range(seq.num_blocks):
            """seq.num_blocks` 是序列需要的块数。比如：
            - 序列有 500 个token，`block_size=256`
            - 需要 2 个块：第一个块满256，第二个块244个token
            """
            token_ids = seq.block(i)#返回第 i 个块的 token 列表
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            """
            这一行有点复杂，分解一下：
            1. `if len(token_ids) == self.block_size`：检查这个块是否满了（256个token）
            2. 如果满了 → 计算哈希值：`self.compute_hash(token_ids, h)`
            - 参数1：`token_ids` 当前块的token列表
            - 参数2：`h` 前一个块的哈希值（用于链式哈希）
            3. 如果没满 → 哈希值为 -1（表示这个块不参与缓存）
            """
            block_id = self.hash_to_block_id.get(h, -1)#用哈希值 h 查找块编号 ,如果找到 → 返回 block_id ,如果没找到 → 返回 -1

            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:#如果没有hash对应的块编号，或者找到的块的内容不一致，就缓存未命中
                cache_miss = True
            
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size#`num_cached_tokens` 是一个计数器，用来记录：这个序列有多少个 token 是从缓存中复用的__（而不是新分配的）

                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:#判断是否需要开新块
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
