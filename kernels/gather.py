"""게더 컴팩션 베이스라인의 호스트 쪽 (스펙 4절). 복사 자체는 gpu_ops.copy_slots (Triton) 또는 copy_slots_torch.

시퀀스마다 살아 있는 토큰이 자기 블록들의 앞쪽 슬롯 0..n-1 에 빈틈없이 놓인다. 그래서 커널은
live_mask 없이 seq_lens = n (토큰 수)로 돌린다. 축출이 일어나면 살아 있는 토큰을 새 블록으로
모두 옮기고 옛 블록을 반납한다 (R-KV의 k[:, :, keep] 컴팩션과 같은 의미, 시퀀스 전체를 다시 모음).

주의: 같은 스텝에 여러 시퀀스를 한 번에 복사하려면, 모든 시퀀스의 compact() 를 먼저 부르고
복사를 마친 뒤에 release() 해야 한다. 먼저 반납하면 다른 시퀀스의 새 블록이 아직 안 읽은 원본 블록과 겹친다.
"""
import torch

from kernels.block_table import pad_block_tables


class GatherTable:
    def __init__(self, num_blocks: int, block_size: int = 8):
        self.bs = block_size
        self.free = list(range(num_blocks - 1, -1, -1))
        self.rows: list[list[int]] = []        # 시퀀스마다 블록 리스트
        self.toks: list[list[int]] = []        # 시퀀스마다 슬롯 순서대로의 토큰 인덱스
        self.in_use = 0
        self.peak = 0                          # 할당된 블록 수의 최댓값 (컴팩션 중 옛·새 블록 공존 포함)

    def _alloc(self) -> int:
        if not self.free:
            raise MemoryError("물리 블록이 바닥났다 (num_blocks를 늘린다)")
        self.in_use += 1
        self.peak = max(self.peak, self.in_use)
        return self.free.pop()

    def add_seq(self) -> int:
        self.rows.append([])
        self.toks.append([])
        return len(self.rows) - 1

    def _slot(self, row, i):
        return row[i // self.bs] * self.bs + i % self.bs

    def append(self, sid: int, token: int) -> int:
        row, toks = self.rows[sid], self.toks[sid]
        if len(toks) % self.bs == 0:
            row.append(self._alloc())
        toks.append(token)
        return self._slot(row, len(toks) - 1)

    def compact(self, sid: int, gone) -> tuple[list[int], list[int], list[int]]:
        """gone 토큰을 빼고 나머지를 새 블록으로 옮길 계획을 세운다. 반환: (src 슬롯, dst 슬롯, 반납할 옛 블록)."""
        gone = set(gone)
        old_row, old_toks = self.rows[sid], self.toks[sid]
        keep = [i for i, t in enumerate(old_toks) if t not in gone]
        assert len(keep) == len(old_toks) - len(gone), "없는 토큰을 축출하려 했다"
        new_row = [self._alloc() for _ in range(-(-len(keep) // self.bs))]
        src = [self._slot(old_row, i) for i in keep]
        dst = [self._slot(new_row, j) for j in range(len(keep))]
        self.rows[sid] = new_row
        self.toks[sid] = [old_toks[i] for i in keep]
        return src, dst, old_row

    def release(self, blocks) -> None:
        self.free.extend(blocks)
        self.in_use -= len(blocks)

    def kernel_inputs(self, sids=None, device="cpu", pad_to=64):
        """block_tables [B, W] int32, seq_lens [B] int32 (토큰 수). live_mask는 없다(모두 live)."""
        sids = range(len(self.rows)) if sids is None else sids
        rows = [self.rows[i] for i in sids]
        seq_lens = torch.tensor([len(self.toks[i]) for i in sids], dtype=torch.int32)
        return pad_block_tables(rows, self.bs, pad_to).to(device), seq_lens.to(device)


def copy_slots_torch(key_cache, value_cache, src, dst):
    """copy_slots의 정답 구현. 캐시 [num_blocks, bs, Hkv, D] 에서 src 슬롯 행을 dst 슬롯으로."""
    k = key_cache.view(-1, *key_cache.shape[2:])
    v = value_cache.view(-1, *value_cache.shape[2:])
    k[dst] = k[src]
    v[dst] = v[src]
