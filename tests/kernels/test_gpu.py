"""서버(GPU)에서만 도는 검증. 스펙 6절 순서대로 위에서 아래로 실행된다 (-x로 첫 실패에서 멈춘다).

  3순서: test_store_kv, test_step3_port_no_mask  -- 이식 자체가 맞는가 (모든 슬롯 live, V1)
  4순서: test_step4_ct_mask (V1), test_step4_permutation (V2)
  5순서: test_copy_slots, test_gather_attention  -- 게더 베이스라인
허용 오차는 bf16에서 atol = rtol = 2e-2 (스펙 V1). 실측 오차를 출력하니 로그에서 확인해 기록한다.
"""
import random

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("GPU 없음", allow_module_level=True)

from kernels.block_table import BlockTable, pad_block_tables  # noqa: E402
from kernels.gather import GatherTable, copy_slots_torch  # noqa: E402
from kernels.gpu_ops import PagedDecode, copy_slots, store_kv  # noqa: E402
from kernels.reference import attention, paged_attention  # noqa: E402
from kernels.workload import timeline  # noqa: E402

DEV = "cuda"
BF = torch.bfloat16
DIMS = {"llama8b": (32, 8, 128), "qwen1.5b": (12, 2, 128)}     # (Hq, Hkv, D)
TOL = dict(atol=2e-2, rtol=2e-2)
WL = dict(prompt_len=50, n_decode=1200, pattern="RETETRETE", budget=256, tau=128)


def report(name, got, want):
    err = (got.float() - want.float()).abs().max().item()
    print(f"\n[{name}] max abs err = {err:.3e}")
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got.float(), want.float(), **TOL)


# ---------------------------------------------------------------- 3순서

@pytest.mark.parametrize("bs", [8, 16])
def test_store_kv(bs):
    hkv, d, n = 8, 128, 300
    kc = torch.zeros(64, bs, hkv, d, dtype=BF, device=DEV)
    vc = torch.zeros_like(kc)
    slots = torch.randperm(64 * bs, device=DEV)[:n]
    k, v = torch.randn(n, hkv, d, dtype=BF, device=DEV), torch.randn(n, hkv, d, dtype=BF, device=DEV)
    store_kv(k, v, kc, vc, slots)
    assert torch.equal(kc.view(-1, hkv, d)[slots], k) and torch.equal(vc.view(-1, hkv, d)[slots], v)


@pytest.mark.parametrize("model", DIMS)
@pytest.mark.parametrize("bs", [8, 16])
@pytest.mark.parametrize("B", [1, 8, 80])          # 작은 배치는 3D 경로, 80은 2D 경로 (두 모델 모두)
def test_step3_port_no_mask(model, bs, B):
    """vLLM 커널을 마스크 없이: 토큰 수 seq_lens, 흩어진 블록. reference와 같아야 이식이 맞다."""
    hq, hkv, d = DIMS[model]
    g = torch.Generator(device="cpu").manual_seed(B * 100 + bs)
    lens = [int(x) for x in torch.randint(1, 3000, (B,), generator=g)]
    need = [-(-n // bs) for n in lens]
    perm = torch.randperm(sum(need) + 7, generator=g).tolist()       # 블록을 뒤섞어 흩어 놓는다
    rows, at = [], 0
    for k in need:
        rows.append(perm[at: at + k])
        at += k
    nb = len(perm)
    kc = torch.randn(nb, bs, hkv, d, dtype=BF, device=DEV)
    vc = torch.randn(nb, bs, hkv, d, dtype=BF, device=DEV)
    tables = pad_block_tables(rows, bs).to(DEV)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device=DEV)
    q = torch.randn(B, hq, d, dtype=BF, device=DEV)
    got = PagedDecode(B, hq, hkv, d)(q, kc, vc, tables, seq_lens, max(lens))
    want = paged_attention(q.float(), kc.float(), vc.float(), tables, seq_lens, None, d ** -0.5)
    report(f"step3 {model} bs={bs} B={B}", got, want)


# ---------------------------------------------------------------- 4순서

def build_ct(model, bs, B, seed=0):
    """타임라인 B개를 BlockTable에 돌리며 store_kv로 실제로 쓴다. 살아 있는 토큰의 K·V도 따로 들고 있다."""
    hq, hkv, d = DIMS[model]
    tls = [timeline(**WL, seed=seed * 1000 + b) for b in range(B)]
    nb = B * (-(-(WL["prompt_len"] + WL["n_decode"]) // bs) + 2)
    kc = torch.zeros(nb, bs, hkv, d, dtype=BF, device=DEV)      # 0으로 시작: 안 쓴 슬롯도 유한해야 한다
    vc = torch.zeros_like(kc)
    bt = BlockTable(nb, bs)
    kv = [{} for _ in range(B)]
    slot_of = [{} for _ in range(B)]

    def write(toks):                          # toks: [(b, token, start, type)]
        slots = [bt.append(b, s, t) for b, _, s, t in toks]
        k = torch.randn(len(toks), hkv, d, dtype=BF, device=DEV)
        v = torch.randn(len(toks), hkv, d, dtype=BF, device=DEV)
        store_kv(k, v, kc, vc, torch.tensor(slots, device=DEV))
        for i, (b, tok, _, _) in enumerate(toks):
            kv[b][tok] = (k[i], v[i])
            slot_of[b][tok] = slots[i]

    for _ in range(B):
        bt.add_seq()
    for p in range(WL["prompt_len"]):
        write([(b, p, 0, 2) for b in range(B)])
    for i in range(WL["n_decode"]):
        write([(b, tls[b][i].token, tls[b][i].start, tls[b][i].type) for b in range(B)])
        for b in range(B):
            if tls[b][i].evict:
                bt.evict(b, [slot_of[b].pop(t) for t in tls[b][i].evict])
                for t in tls[b][i].evict:
                    del kv[b][t]
    tables, live, seq_lens = bt.kernel_inputs(device=DEV)
    return bt, kc, vc, kv, tables, live, seq_lens


def dense(q, kv, d):
    out = []
    for b, toks in enumerate(kv):
        ks = torch.stack([toks[t][0] for t in sorted(toks)]).float()
        vs = torch.stack([toks[t][1] for t in sorted(toks)]).float()
        out.append(attention(q[b].float(), ks, vs, d ** -0.5))
    return torch.stack(out)


@pytest.mark.parametrize("model", DIMS)
@pytest.mark.parametrize("bs", [8, 16])
@pytest.mark.parametrize("B", [1, 80])
def test_step4_ct_mask(model, bs, B):
    hq, hkv, d = DIMS[model]
    bt, kc, vc, kv, tables, live, seq_lens = build_ct(model, bs, B)
    q = torch.randn(B, hq, d, dtype=BF, device=DEV)
    run = PagedDecode(B, hq, hkv, d)
    got = run(q, kc, vc, tables, seq_lens, int(seq_lens.max()), live_mask=live)
    report(f"step4 vs reference {model} bs={bs} B={B}",
           got, paged_attention(q.float(), kc.float(), vc.float(), tables, seq_lens, live, d ** -0.5))
    report(f"step4 vs dense {model} bs={bs} B={B}", got, dense(q, kv, d))
    # 마스크를 빼면 죽은 슬롯의 옛 K·V가 섞여 달라져야 한다
    no_mask = run(q, kc, vc, tables, seq_lens, int(seq_lens.max()))
    assert not torch.allclose(no_mask.float(), got.float(), **TOL)
    print(f"  slots/live = {int(seq_lens.sum())}/{int(live.sum())}")


@pytest.mark.parametrize("model", DIMS)
def test_step4_permutation(model):
    """V2: 블록 테이블 행 안의 블록 순서를 섞어도 출력이 같다 (어텐션은 순열 불변, 논문 §C.3)."""
    hq, hkv, d = DIMS[model]
    B, bs = 8, 8
    bt, kc, vc, kv, tables, live, seq_lens = build_ct(model, bs, B, seed=1)
    q = torch.randn(B, hq, d, dtype=BF, device=DEV)
    run = PagedDecode(B, hq, hkv, d)
    base = run(q, kc, vc, tables, seq_lens, int(seq_lens.max()), live_mask=live)
    rng = random.Random(0)
    rows = [rng.sample(bt.seqs[b].blocks, len(bt.seqs[b].blocks)) for b in range(B)]
    shuffled = pad_block_tables(rows, bs).to(DEV)
    assert not torch.equal(shuffled, tables)
    report(f"V2 permutation {model}", run(q, kc, vc, shuffled, seq_lens, int(seq_lens.max()), live_mask=live), base)


# ---------------------------------------------------------------- 5순서

def test_copy_slots():
    kc = torch.randn(256, 8, 8, 128, dtype=BF, device=DEV)
    vc = torch.randn_like(kc)
    perm = torch.randperm(256 * 8, device=DEV)
    src, dst = perm[:700], perm[700:1400]
    want_k, want_v = kc.clone(), vc.clone()
    copy_slots_torch(want_k, want_v, src, dst)
    copy_slots(kc, vc, src, dst)
    assert torch.equal(kc, want_k) and torch.equal(vc, want_v)


@pytest.mark.parametrize("model", DIMS)
def test_gather_attention(model):
    """게더 방식: 축출 때마다 copy_slots로 모으고, 마스크 없이 토큰 수 seq_lens로 돌린다."""
    hq, hkv, d = DIMS[model]
    B, bs = 8, 8
    tls = [timeline(**WL, seed=b) for b in range(B)]
    nb = B * 2 * (-(-(WL["prompt_len"] + WL["n_decode"]) // bs) + 2)
    kc = torch.zeros(nb, bs, hkv, d, dtype=BF, device=DEV)
    vc = torch.zeros_like(kc)
    gt = GatherTable(nb, bs)
    kv = [{} for _ in range(B)]

    def write(toks):
        slots = [gt.append(b, tok) for b, tok in toks]
        k = torch.randn(len(toks), hkv, d, dtype=BF, device=DEV)
        v = torch.randn(len(toks), hkv, d, dtype=BF, device=DEV)
        store_kv(k, v, kc, vc, torch.tensor(slots, device=DEV))
        for i, (b, tok) in enumerate(toks):
            kv[b][tok] = (k[i], v[i])

    for _ in range(B):
        gt.add_seq()
    for p in range(WL["prompt_len"]):
        write([(b, p) for b in range(B)])
    for i in range(WL["n_decode"]):
        write([(b, tls[b][i].token) for b in range(B)])
        if tls[0][i].evict:
            plans = [gt.compact(b, tls[b][i].evict) for b in range(B)]
            src = torch.tensor([x for p in plans for x in p[0]], device=DEV)
            dst = torch.tensor([x for p in plans for x in p[1]], device=DEV)
            copy_slots(kc, vc, src, dst)
            for p in plans:
                gt.release(p[2])
            for b in range(B):
                for t in tls[b][i].evict:
                    del kv[b][t]
    tables, seq_lens = gt.kernel_inputs(device=DEV)
    q = torch.randn(B, hq, d, dtype=BF, device=DEV)
    got = PagedDecode(B, hq, hkv, d)(q, kc, vc, tables, seq_lens, int(seq_lens.max()))
    report(f"gather {model}", got, dense(q, kv, d))
