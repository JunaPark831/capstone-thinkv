"""CT 블록 테이블: 호스트 쪽 자료구조 (스펙 docs/e10_ct_kernel_design.md 3절).

어텐션 커널은 이 상태를 직접 읽지 않는다. kernel_inputs()가 만드는
block_tables / live_mask / seq_lens 세 텐서만 받는다.

슬롯 번호는 전역이다: slot = 물리 블록 번호 * block_size + 블록 안 위치.
vLLM의 slot_mapping과 같은 규칙이라 쓰기 커널(reshape_and_cache_flash)에 그대로 넘긴다.
"""
import heapq
from dataclasses import dataclass, field

import numpy as np
import torch

T, E, R = 0, 1, 2          # 사고 유형 (스펙 3절 type 필드)
FREE = -1                  # 아직 어느 시퀀스에도 할당되지 않은 블록


@dataclass
class Segment:
    start: int                                       # 구간 첫 토큰의 논리 인덱스
    type: int
    slots: list[int] = field(default_factory=list)   # 이 구간의 살아 있는 토큰 슬롯


@dataclass
class Seq:
    blocks: list[int] = field(default_factory=list)          # 블록 테이블 한 행 (할당 순서)
    segments: list[Segment] = field(default_factory=list)
    reuse: dict[int, list[int]] = field(default_factory=dict)  # 유형 -> 축출 슬롯 힙
    partial: dict[int, int] = field(default_factory=dict)      # 유형 -> 덜 찬 블록
    seg_of: dict[int, Segment] = field(default_factory=dict)   # 살아 있는 슬롯 -> 구간


class BlockTable:
    def __init__(self, num_blocks: int, block_size: int = 8):
        self.bs = block_size
        self.free = list(range(num_blocks - 1, -1, -1))    # pop()이 작은 번호부터 준다
        self.type = np.full(num_blocks, FREE, np.int8)
        self.num_filled = np.zeros(num_blocks, np.int32)
        self.evict_mask = np.zeros((num_blocks, block_size), bool)
        self.live_mask = np.zeros((num_blocks, block_size), np.int8)
        self.seqs: list[Seq] = []

    def add_seq(self) -> int:
        self.seqs.append(Seq())
        return len(self.seqs) - 1

    def append(self, sid: int, start: int, ttype: int) -> int:
        """새 토큰 하나의 슬롯을 정하고 표시한다. 반환값이 slot_mapping 원소다.

        start가 마지막 구간의 시작과 다르면 새 구간을 연다.
        자리 고르는 순서(스펙 3절 쓰기 경로): 같은 유형 블록의 축출 슬롯 → 같은 유형의 덜 찬 블록 → 새 블록.
        """
        s = self.seqs[sid]
        if not s.segments or s.segments[-1].start != start:
            s.segments.append(Segment(start, ttype))
        seg = s.segments[-1]
        assert seg.type == ttype, "한 구간 안에서 유형이 바뀌었다"

        heap = s.reuse.setdefault(ttype, [])
        if heap:
            slot = heapq.heappop(heap)
        else:
            p = s.partial.get(ttype)
            if p is None:
                if not self.free:
                    raise MemoryError("물리 블록이 바닥났다 (num_blocks를 늘린다)")
                p = self.free.pop()
                self.type[p] = ttype
                s.blocks.append(p)
                s.partial[ttype] = p
            slot = p * self.bs + int(self.num_filled[p])
            self.num_filled[p] += 1
            if self.num_filled[p] == self.bs:
                del s.partial[ttype]

        b, o = divmod(slot, self.bs)
        self.evict_mask[b, o] = False
        self.live_mask[b, o] = 1
        seg.slots.append(slot)
        s.seg_of[slot] = seg
        return slot

    def evict(self, sid: int, slots) -> None:
        """축출 표시만 한다(soft-mark). 메모리는 다음 같은 유형 토큰이 덮어쓸 때까지 그대로 둔다."""
        s = self.seqs[sid]
        gone = set(slots)
        touched = {id(s.seg_of[x]): s.seg_of[x] for x in gone}   # 없는 슬롯이면 KeyError
        for seg in touched.values():
            seg.slots = [x for x in seg.slots if x not in gone]
        for x in gone:
            del s.seg_of[x]
            b, o = divmod(x, self.bs)
            self.evict_mask[b, o] = True
            self.live_mask[b, o] = 0
            heapq.heappush(s.reuse.setdefault(int(self.type[b]), []), x)

    def kernel_inputs(self, sids=None, device="cpu", pad_to=64):
        """커널 인자 세 개: block_tables [B, W] int32, live_mask [num_blocks, bs] int8, seq_lens [B] int32.

        seq_lens는 토큰 수가 아니라 슬롯 수(블록 수 * bs)다. 빈 슬롯·죽은 슬롯은 live_mask가 거른다.
        W는 pad_to 슬롯 단위로 올림한다. vLLM 커널은 타일(16~32슬롯) 단위로 블록 테이블을
        마스크 없이 읽기 때문에, 블록 수가 홀수면 행 끝 다음 칸을 읽는다. 채움 값 0은 유효한 블록 번호다.
        """
        sids = range(len(self.seqs)) if sids is None else sids
        rows = [self.seqs[i].blocks for i in sids]
        seq_lens = np.array([len(r) * self.bs for r in rows], np.int32)
        return (
            pad_block_tables(rows, self.bs, pad_to).to(device),
            torch.from_numpy(self.live_mask.copy()).to(device),
            torch.from_numpy(seq_lens).to(device),
        )

    def blocks_in_use(self) -> int:
        return int((self.type != FREE).sum())


def pad_block_tables(rows, bs, pad_to=64) -> torch.Tensor:
    """블록 번호 리스트들을 [B, W] int32로. W는 pad_to 슬롯 단위로 올림, 빈 칸은 0 (kernel_inputs 설명 참고)."""
    per = max(1, pad_to // bs)
    width = max(1, -(-max(len(r) for r in rows) // per)) * per
    bt = np.zeros((len(rows), width), np.int32)
    for i, r in enumerate(rows):
        bt[i, : len(r)] = r
    return torch.from_numpy(bt)
