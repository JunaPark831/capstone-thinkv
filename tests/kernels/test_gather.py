"""게더 베이스라인 호스트 로직: 한 스텝에 여러 시퀀스를 같이 컴팩션해도 어텐션이 정답과 같다."""
import pytest
import torch

from kernels.gather import GatherTable, copy_slots_torch
from kernels.reference import attention, paged_attention
from kernels.workload import timeline

HKV, GQA, D = 2, 2, 16
CFG = dict(prompt_len=20, n_decode=600, pattern="RETETRETE", budget=64, tau=16, retain=(8, 4))


def run(bs, n_seqs, cut, num_blocks=1024):
    gen = torch.Generator().manual_seed(cut)
    gt = GatherTable(num_blocks, bs)
    kc, vc = torch.zeros(num_blocks, bs, HKV, D), torch.zeros(num_blocks, bs, HKV, D)
    kv = [{} for _ in range(n_seqs)]
    tls = [timeline(**CFG, seed=s) for s in range(n_seqs)]       # 축출 시점은 같고 남기는 토큰만 다르다

    def write(sid, tok):
        slot = gt.append(sid, tok)
        kv[sid][tok] = (torch.randn(HKV, D, generator=gen), torch.randn(HKV, D, generator=gen))
        kc[slot // bs, slot % bs], vc[slot // bs, slot % bs] = kv[sid][tok]

    for sid in range(n_seqs):
        gt.add_seq()
        for p in range(CFG["prompt_len"]):
            write(sid, p)
    for i in range(cut):
        for sid in range(n_seqs):
            write(sid, tls[sid][i].token)
        if tls[0][i].evict:
            plans = [gt.compact(sid, tls[sid][i].evict) for sid in range(n_seqs)]
            src = [x for p in plans for x in p[0]]
            dst = [x for p in plans for x in p[1]]
            # GPU 복사 커널은 슬롯마다 병렬이라 읽기·쓰기 순서가 없다. torch 복사는 원본을 먼저 다 읽어서
            # 겹쳐도 맞게 나오므로, 겹치지 않는다는 것을 직접 확인한다.
            assert not set(src) & set(dst)
            copy_slots_torch(kc, vc, src, dst)                     # 모든 시퀀스를 한 번에 복사한 뒤
            for p in plans:
                gt.release(p[2])                                   # 반납
            for sid in range(n_seqs):
                for t in tls[sid][i].evict:
                    del kv[sid][t]
    return gt, kc, vc, kv


@pytest.mark.parametrize("bs", [8, 16])
def test_gather_attention_matches_dense(bs):
    for cut in (17, 100, 333, 600):
        gt, kc, vc, kv = run(bs, n_seqs=4, cut=cut)
        tables, seq_lens = gt.kernel_inputs()
        q = torch.randn(4, HKV * GQA, D, generator=torch.Generator().manual_seed(7))
        got = paged_attention(q, kc, vc, tables, seq_lens, None, D ** -0.5)
        for sid in range(4):
            toks = sorted(kv[sid])
            assert gt.toks[sid] == toks                            # 컴팩션 뒤에도 논리 순서 유지
            want = attention(q[sid], torch.stack([kv[sid][t][0] for t in toks]),
                             torch.stack([kv[sid][t][1] for t in toks]), D ** -0.5)
            torch.testing.assert_close(got[sid], want, atol=1e-5, rtol=1e-5)
        assert gt.in_use + len(gt.free) == 1024 and len(set(gt.free)) == len(gt.free)
        assert gt.in_use == sum(len(r) for r in gt.rows)
        assert gt.peak >= gt.in_use
