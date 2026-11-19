"""M3: C++ 융합 어텐션(fused) 확인. 세 가지로 본다.

1. 무작위 값: 저장 상태를 손으로 만들어 '풀어서 float32로 계산한 정답'과 비교 (atol = rtol = 1e-3)
2. 실제 8B 생성 중 같은 상태에서 비교: 매 층·매 스텝마다 C++ 어텐션과 float32 정답을 둘 다 계산 (같은 허용 오차)
3. 같은 토큰을 넣고 점수 비교: 원본(ThinKVPress)이 만든 토큰을 C++ 어텐션 엔진에 그대로 넣고 매 스텝 다음 토큰 점수 차이를 잰다
   (샘플링 생성은 작은 차이로도 문장이 갈라져서 C++ 어텐션 확인에 쓸 수 없다)
실행(루트에서): .venv-kvpress/bin/python -m pytest ondevice/tests/test_ventuno_attn.py -q -s"""
import pytest
import torch
from transformers import AutoConfig, DynamicCache

from kernels.block_table import E, R, T
from ondevice import ventuno_cache
from ondevice.measure_scale_waste import MODEL
from ondevice.tests.conftest import N, SHORT, sample
from ondevice.ventuno_attn import fused
from ondevice.ventuno_cache import Engine, attach, eager
from thinkv.press import ThinKVPress

TOL = 10.0      # 다음 토큰 점수 차이의 최댓값 허용 (2026-11-19 사용자 결정, 조각 모델 비교와 같은 기준)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fused_matches_dense(dtype):
    cfg = AutoConfig.from_pretrained(MODEL)
    eng = Engine.from_config(cfg, dtype, 256)
    L, hq = eng.layers[0], cfg.num_attention_heads
    hkv, _, d = L.pk.shape
    torch.manual_seed(0)
    rnd = lambda *s: (torch.randn(*s) * 3).to(dtype)

    eng.prefill(0, rnd(hkv, 40, d), rnd(hkv, 40, d))
    for i, (kind, typ) in enumerate([("nvfp4", R), ("ternary", T), ("nvfp4", E), ("nvfp4", R), ("ternary", T)]):
        L.bk.copy_(rnd(hkv, 16, d)), L.bv.copy_(rnd(hkv, 16, d))
        L.nb = 16
        L.quant(kind, i, typ)
    L.evict(0, 5)                                   # 구간 0은 5개만, 구간 1은 3개만 남긴다
    L.evict(1, 3)
    pool, seg = L.segs[2]
    pool.evict(list(seg.slots))                     # 구간 2를 통째로 비우면 키 스케일 행이 반납되고
    assert pool.free
    L.bk.copy_(rnd(hkv, 16, d)), L.bv.copy_(rnd(hkv, 16, d))
    L.nb = 16
    L.quant("nvfp4", 5, R)                          # 다음 묶음이 그 행과 빈칸을 다시 쓴다
    assert not pool.free
    L.bk[:, :7] = rnd(hkv, 7, d)
    L.nb = 7

    q, scaling = rnd(hq, d), d ** -0.5
    K, V = L.dense(torch.float32)                   # 정답: 같은 저장값을 float32로 풀어서 계산
    ref = eager(q.float(), K, V, scaling)
    torch.testing.assert_close(fused(L, q, scaling), ref, atol=1e-3, rtol=1e-3)


def test_fused_on_real_8b(model, tok, prompt, monkeypatch):
    """생성은 임시 어텐션(torch)으로 하고, 매번 그 자리(같은 저장 상태)에서 C++ 어텐션도 계산해 정답과 비교한다."""
    kw = dict(tau=64, retain=(2, 1))                # 묶음 네 개짜리 구간에서 2개만 남기므로 적어도 두 묶음이 통째로 퇴출된다
    eng = Engine.from_config(model.config, torch.bfloat16, N, attn="torch", budget=prompt.shape[1] + 64, **kw)
    cur, errs, freed = {}, [], set()
    attend = eng.attend

    def attend_and_remember(l, *a):
        cur["L"] = eng.layers[l]
        return attend(l, *a)

    def eager_and_check(q, K, V, scaling):          # Engine.attend 안에서 불린다: 이번 토큰을 넣은 뒤, 압축·퇴출 전
        L = cur["L"]
        ref = eager(q.float(), *L.dense(torch.float32), scaling)
        out = fused(L, q, scaling)
        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)
        errs.append((out - ref).abs().max())
        freed.update(g for p in L.pools.values() for g in p.free)
        return eager(q, K, V, scaling)

    eng.attend = attend_and_remember
    monkeypatch.setattr(ventuno_cache, "eager", eager_and_check)
    attach(model, eng)
    try:
        sample(model, prompt, tok)
    finally:
        model.set_attn_implementation("eager_sparsity")
    errs = torch.stack(errs)
    print(f"\n비교 {len(errs)}번, 최대 오차 {errs.max():.2e}, 반납된 키 스케일 행 번호 {sorted(freed)}")
    assert len(errs) == model.config.num_hidden_layers * (N - 1)
    assert freed                                    # 키 스케일 행 반납·재사용이 실제 데이터에서도 일어났다


def forced_logits(model, prompt, forced):
    """토큰 열을 그대로 넣으며(샘플링 없음) 매 스텝 다음 토큰 점수를 모은다."""
    cache, out, ids = DynamicCache(), [], prompt
    with torch.no_grad():
        for t in forced:
            out.append(model(ids, past_key_values=cache, use_cache=True).logits[0, -1].float())
            ids = torch.tensor([[t]])
    return out


@pytest.mark.parametrize("labels", ["periodic", "sparsity"])
def test_fused_logits_with_same_tokens(model, tok, prompt, labels):
    budget = prompt.shape[1] + 64
    press = ThinKVPress(budget=budget, labels=labels, **SHORT)
    with press(model):
        ref = sample(model, prompt, tok, output_logits=True, return_dict_in_generate=True)
    forced = ref.sequences[0, prompt.shape[1]:].tolist()

    eng = Engine.from_config(model.config, torch.bfloat16, N, budget=budget, labels=labels, **SHORT)   # 기본값 fused
    attach(model, eng)
    try:
        seen = forced_logits(model, prompt, forced)
    finally:
        model.set_attn_implementation("eager_sparsity")
    diff = torch.stack([(a - b[0].float()).abs().max() for a, b in zip(seen, ref.logits)])
    top1 = sum(int(a.argmax() == b[0].argmax()) for a, b in zip(seen, ref.logits)) / len(seen)
    ours, orig = "".join(eng.plan.log["labels"]), "".join(press.last_log["labels"])
    print(f"\n{labels}: 스텝 {len(seen)}, 점수 차이 최대 {diff.max():.3f}({int(diff.argmax())}번째 스텝) 중앙값 {diff.median():.3f}, "
          f"처음 16스텝 최대 {diff[:16].max():.3f}, 1등 토큰 일치 {top1:.1%}, 라벨 원본 {orig} / 우리 {ours}")
    assert orig and len(seen) == len(forced) and torch.isfinite(diff).all()
    assert diff.max() <= TOL
