"""V3·V4 (스펙 5절): 블록 테이블 불변식. 실행: 루트에서 .venv-kvpress/bin/python -m pytest tests/kernels

검사는 block_table.py의 상태를 그대로 믿지 않는다. 테스트가 따로 들고 있는 그림자 모델
(슬롯 -> 그 슬롯에 살아 있는 토큰의 유형)과 대조한다.
"""
import random

import numpy as np
import pytest
import torch

from kernels.block_table import FREE, BlockTable


def check_I3(bt: BlockTable):
    """V4. live_mask == 채워짐 & ~evict_mask, 할당 안 된 블록은 전부 0."""
    filled = np.arange(bt.bs)[None, :] < bt.num_filled[:, None]
    assert np.array_equal(bt.live_mask.astype(bool), filled & ~bt.evict_mask)
    assert not bt.evict_mask[bt.type == FREE].any()


def check_against_shadow(bt: BlockTable, alive: dict, first_type: dict):
    """I1과 상태 일치. alive: 슬롯 -> (sid, 유형). first_type: 블록 -> 처음 쓰일 때 유형."""
    live_slots = set(np.flatnonzero(bt.live_mask.reshape(-1)).tolist())
    assert live_slots == set(alive)
    for slot, (sid, t) in alive.items():
        b = slot // bt.bs
        assert bt.type[b] == t == first_type[b]          # I1: 살아 있는 토큰 유형 == 블록 유형
        assert b in bt.seqs[sid].blocks
    for b, t in first_type.items():
        assert bt.type[b] == t                           # I1: 블록 유형은 바뀌지 않는다
    owners = [b for s in bt.seqs for b in s.blocks]
    assert len(owners) == len(set(owners))               # 블록은 한 시퀀스에만 속한다
    for sid, s in enumerate(bt.seqs):
        in_segs = [x for seg in s.segments for x in seg.slots]
        assert sorted(in_segs) == sorted(x for x, (q, _) in alive.items() if q == sid)


def random_timeline(bt: BlockTable, rng: random.Random, n_ops: int, n_seqs: int, check_every_op: bool):
    alive, first_type = {}, {}
    cur = []                                             # 시퀀스마다 [다음 토큰 인덱스, 구간 시작, 유형]
    for _ in range(n_seqs):
        bt.add_seq()
        cur.append([0, 0, rng.randrange(3)])
    for _ in range(n_ops):
        sid = rng.randrange(n_seqs)
        if rng.random() < 0.7:
            st = cur[sid]
            if rng.random() < 0.1:                       # 새 구간, 유형 무작위
                st[1], st[2] = st[0], rng.randrange(3)
            slot = bt.append(sid, st[1], st[2])
            assert slot not in alive                     # I2: 살아 있는 슬롯을 덮어쓰지 않는다
            b = slot // bt.bs
            first_type.setdefault(b, st[2])
            alive[slot] = (sid, st[2])
            st[0] += 1
        else:
            mine = [x for x, (q, _) in alive.items() if q == sid]
            if not mine:
                continue
            gone = rng.sample(mine, rng.randint(1, len(mine)))
            bt.evict(sid, gone)
            for x in gone:
                del alive[x]
        if check_every_op:
            check_I3(bt)
    check_I3(bt)
    check_against_shadow(bt, alive, first_type)


@pytest.mark.parametrize("bs", [8, 16])
def test_V4_I3_after_every_op(bs):
    for seed in range(50):
        random_timeline(BlockTable(4096, bs), random.Random(seed), n_ops=300, n_seqs=3, check_every_op=True)


@pytest.mark.parametrize("bs", [8, 16])
def test_V3_I1_I2_1000_timelines(bs):
    for seed in range(1000):
        random_timeline(BlockTable(4096, bs), random.Random(seed), n_ops=200, n_seqs=4, check_every_op=False)


def test_slot_order_rules():
    """스펙 3절 쓰기 경로 1→2→3 순서를 손으로 따라간다."""
    bt = BlockTable(16, block_size=4)
    sid = bt.add_seq()
    r = [bt.append(sid, 0, 2) for _ in range(6)]         # R 6개: 블록 0 가득, 블록 1에 2개
    assert r == [0, 1, 2, 3, 4, 5]
    e = bt.append(sid, 6, 1)                             # E: R 블록에 빈자리가 있어도 새 블록(사고 인식 페이징)
    assert e == 8
    bt.evict(sid, [1, 3])
    assert bt.append(sid, 7, 2) == 1                     # 1순위: 같은 유형 블록의 축출 슬롯, 작은 번호부터
    assert bt.append(sid, 7, 2) == 3
    assert bt.append(sid, 7, 2) == 6                     # 2순위: 같은 유형의 덜 찬 블록
    assert bt.append(sid, 7, 2) == 7
    assert bt.append(sid, 7, 2) == 12                    # 3순위: 새 블록 (블록 3)
    assert bt.append(sid, 12, 1) == 9                    # E는 자기 블록 2의 빈자리로
    bt.evict(sid, [9])
    assert bt.append(sid, 12, 1) == 9


def test_kernel_inputs_padding():
    bt = BlockTable(64, block_size=8)
    a, b = bt.add_seq(), bt.add_seq()
    for i in range(24):                                  # 3블록
        bt.append(a, 0, 2)
    bt.append(b, 0, 0)                                   # 1블록
    tables, live, seq_lens = bt.kernel_inputs()
    assert tables.shape == (2, 8) and tables.dtype == torch.int32 and seq_lens.dtype == torch.int32
    assert tables[0, :3].tolist() == bt.seqs[a].blocks and tables[1, 0] == bt.seqs[b].blocks[0]
    assert seq_lens.tolist() == [24, 8]
    assert live.shape == (64, 8) and live.dtype == torch.int8 and int(live.sum()) == 25


def test_evict_unknown_slot_raises():
    bt = BlockTable(8, block_size=8)
    sid = bt.add_seq()
    bt.append(sid, 0, 0)
    with pytest.raises(KeyError):
        bt.evict(sid, [5])                               # 쓰인 적 없는 슬롯
    bt.evict(sid, [0])
    with pytest.raises(KeyError):
        bt.evict(sid, [0])                               # 두 번 축출
