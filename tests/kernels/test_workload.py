"""workload.py: 축출 규칙이 스펙 4절대로인지, 그리고 타임라인을 블록 테이블에 돌려도 V5가 성립하는지."""
import pytest
import torch

from kernels.block_table import T, BlockTable
from kernels.reference import attention, paged_attention
from kernels.workload import drive, timeline

SMALL = dict(prompt_len=20, n_decode=600, pattern="RETETRETE", budget=64, tau=16, retain=(8, 4))
PAPER = dict(prompt_len=100, n_decode=6000, pattern="RETETRETE", budget=1024, tau=128, retain=(64, 32, 16, 8, 4))


@pytest.mark.parametrize("cfg", [SMALL, PAPER], ids=["small", "paper"])
def test_rules(cfg):
    P, tau, retain, budget = cfg["prompt_len"], cfg["tau"], cfg["retain"], cfg["budget"]
    steps = timeline(**cfg, seed=0)
    seg_of, segs = {}, {}                      # 토큰 -> 구간 시작, 구간 시작 -> 살아 있는 토큰 집합
    seg_type = {}
    alive = set(range(P))
    fired_case1 = 0
    for i, st in enumerate(steps):
        before = {s: len(v) for s, v in segs.items()}
        segs.setdefault(st.start, set()).add(st.token)
        seg_type[st.start] = st.type
        seg_of[st.token] = st.start
        alive.add(st.token)
        for x in st.evict:
            assert x in alive and x >= P               # 살아 있던 토큰만, 프롬프트는 축출 안 함
            assert seg_of[x] != st.start               # 열린 구간은 건드리지 않음
            alive.remove(x)
            segs[seg_of[x]].remove(x)
        closed = [s for s in segs if s != st.start]
        for s in closed:
            assert len(segs[s]) in (tau, *retain)      # 닫힌 구간 길이는 tau 또는 보존 스케줄 값
        if len(alive) > budget:                        # 예산 초과가 남았다면 더 줄일 구간이 없어야 한다
            assert all(len(segs[s]) <= retain[-1] for s in closed)
        if i % tau == 0 and i > 0 and seg_type[steps[i - 1].start] == T:   # 경우 1: 방금 T 구간이 닫혔다
            fired_case1 += 1
            for s, n in before.items():
                if s != steps[i - 1].start and n > retain[-1]:
                    assert len(segs[s]) < n
    assert fired_case1 > 0


def test_deterministic():
    a = timeline(**SMALL, seed=3)
    b = timeline(**SMALL, seed=3)
    c = timeline(**SMALL, seed=4)
    assert a == b and a != c


@pytest.mark.parametrize("bs", [8, 16])
def test_V5_on_workload(bs):
    """타임라인을 BlockTable에 돌리면서 중간중간 paged_attention == 살아 있는 토큰만의 attention."""
    HKV, GQA, D = 2, 2, 16
    gen = torch.Generator().manual_seed(0)
    nb = 256
    kc, vc = torch.zeros(nb, bs, HKV, D), torch.zeros(nb, bs, HKV, D)
    kv = {}

    def on_write(tok, slot):
        kv[tok] = (torch.randn(HKV, D, generator=gen), torch.randn(HKV, D, generator=gen))
        kc[slot // bs, slot % bs], vc[slot // bs, slot % bs] = kv[tok]

    steps = timeline(**SMALL, seed=1)
    for cut in (50, 200, 400, len(steps)):              # 자른 지점마다 처음부터 다시 돌린다
        bt = BlockTable(nb, bs)
        kc.zero_(), vc.zero_(), kv.clear()
        sid = bt.add_seq()
        slot_of = drive(bt, sid, SMALL["prompt_len"], steps[:cut], on_write)
        tables, live, seq_lens = bt.kernel_inputs()
        q = torch.randn(1, HKV * GQA, D, generator=gen)
        got = paged_attention(q, kc, vc, tables, seq_lens, live, D ** -0.5)
        toks = sorted(slot_of)
        want = attention(q[0], torch.stack([kv[t][0] for t in toks]), torch.stack([kv[t][1] for t in toks]), D ** -0.5)
        torch.testing.assert_close(got[0], want, atol=1e-5, rtol=1e-5)
        assert int(live.sum()) == len(slot_of)
