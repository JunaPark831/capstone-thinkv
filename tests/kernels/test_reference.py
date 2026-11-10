"""V5 (스펙 5절): 블록 테이블의 논리 상태로 계산한 어텐션이 정답과 같다.

정답 쪽은 블록 테이블을 거치지 않는다. 테스트가 토큰마다 K·V를 따로 들고 있다가
살아 있는 토큰만 모아 attention()에 넣는다. 비교 대상은 물리 풀 + kernel_inputs()로 계산한 paged_attention()이다.
축출된 슬롯의 옛 K·V는 풀에 그대로 남아 있으므로 live_mask가 틀리면 값이 달라진다.
"""
import random

import pytest
import torch

from kernels.block_table import BlockTable
from kernels.reference import attention, paged_attention

HKV, GQA, D = 2, 3, 16
SCALE = D ** -0.5


def build(bs, seed, n_ops=300, n_seqs=3, num_blocks=512):
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    bt = BlockTable(num_blocks, bs)
    kc = torch.zeros(num_blocks, bs, HKV, D)
    vc = torch.zeros(num_blocks, bs, HKV, D)
    alive = []                                   # 시퀀스마다 {토큰 인덱스: (slot, k, v)}
    cur = []

    def write(sid, start, t):
        slot = bt.append(sid, start, t)
        k, v = torch.randn(HKV, D, generator=gen), torch.randn(HKV, D, generator=gen)
        kc[slot // bs, slot % bs], vc[slot // bs, slot % bs] = k, v
        alive[sid][cur[sid][0]] = (slot, k, v)
        cur[sid][0] += 1

    for sid in range(n_seqs):
        bt.add_seq()
        alive.append({})
        cur.append([0, 0, 2])
        for _ in range(rng.randint(1, 5)):       # 프롬프트: 토큰 0..P-1, 축출하지 않는다
            write(sid, 0, 2)
        cur[sid][1] = cur[sid][0]
    for _ in range(n_ops):
        sid = rng.randrange(n_seqs)
        st = cur[sid]
        if rng.random() < 0.7:
            if rng.random() < 0.1:
                st[1], st[2] = st[0], rng.randrange(3)
            write(sid, st[1], st[2])
        else:
            seg_of = bt.seqs[sid].seg_of
            cand = [i for i, (slot, _, _) in alive[sid].items() if seg_of[slot].start != 0]   # 프롬프트 제외
            if not cand:
                continue
            gone = rng.sample(cand, rng.randint(1, len(cand)))
            bt.evict(sid, [alive[sid][i][0] for i in gone])
            for i in gone:
                del alive[sid][i]
    return bt, kc, vc, alive


def dense(q, alive):
    out = []
    for b, toks in enumerate(alive):
        ks = torch.stack([toks[i][1] for i in sorted(toks)])
        vs = torch.stack([toks[i][2] for i in sorted(toks)])
        out.append(attention(q[b], ks, vs, SCALE))
    return torch.stack(out)


@pytest.mark.parametrize("bs", [8, 16])
def test_V5_paged_matches_dense(bs):
    missed = 0
    for seed in range(100):
        bt, kc, vc, alive = build(bs, seed)
        tables, live, seq_lens = bt.kernel_inputs()
        q = torch.randn(len(alive), HKV * GQA, D, generator=torch.Generator().manual_seed(1000 + seed))
        got = paged_attention(q, kc, vc, tables, seq_lens, live, SCALE)
        want = dense(q, alive)
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
        # 마스크를 빼면 죽은·빈 슬롯이 섞여 달라져야 한다 (이 테스트가 마스크에 민감하다는 확인)
        no_mask = paged_attention(q, kc, vc, tables, seq_lens, None, SCALE)
        missed += torch.allclose(no_mask, want, atol=1e-3)
    assert missed == 0


def test_attention_gqa_head_mapping():
    """쿼리 헤드 h가 KV 헤드 h // GQA를 본다: KV 헤드 1의 값만 1이면 헤드 3~5 출력만 1."""
    q = torch.randn(HKV * GQA, D)
    k = torch.randn(4, HKV, D)
    v = torch.zeros(4, HKV, D)
    v[:, 1] = 1.0
    out = attention(q, k, v, SCALE)
    assert torch.allclose(out[:GQA], torch.zeros(GQA, D)) and torch.allclose(out[GQA:], torch.ones(GQA, D))
