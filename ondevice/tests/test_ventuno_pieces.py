"""M3.5: 조각의 입출력 모양이 고정인지, 그리고 조각 실행(조각 + RoPE + 엔진 + C++ 커널)이
원래 모델(HF + ThinKVPress)과 같은 토큰을 넣었을 때 다음 토큰 점수(logit)가 얼마나 다른지 본다.
허용 오차는 실측(200스텝 최대 7.6)을 보고 사용자가 '전체 최대 차이 10 이하'로 정했다(2026-11-19).
실행(루트에서): .venv-kvpress/bin/python -m pytest ondevice/tests/test_ventuno_pieces.py -q -s"""
import pytest
import torch

from ondevice.measure_scale_waste import aime_prompt
from ondevice.tests.conftest import N, SHORT, sample
from ondevice.ventuno_cache import Engine
from ondevice.ventuno_pieces import CHUNK, Rope, TorchRunner, generate, pieces
from thinkv.press import ThinKVPress

TOL = 10.0      # 다음 토큰 점수 차이의 최댓값 허용 (2026-11-19 사용자 결정)


def test_piece_shapes(model):
    cfg, ps = model.config, pieces(model)
    run, d = TorchRunner(ps), cfg.hidden_size // cfg.num_attention_heads
    assert len(ps) == cfg.num_hidden_layers + 1                # 8B: 33조각
    for n in (1, CHUNK):
        h = torch.randn(n, cfg.hidden_size, dtype=torch.bfloat16)
        a = torch.randn(n, cfg.num_attention_heads * d, dtype=torch.bfloat16)
        for i, x in ((0, None), (5, a)):
            h2, q, k, v = run(i, h, x)
            assert h2.shape == h.shape and q.shape == a.shape
            assert k.shape == v.shape == (n, cfg.num_key_value_heads * d)
        assert run(len(ps) - 1, h, a).shape == (n, cfg.vocab_size)
    with pytest.raises(AssertionError):
        run(0, torch.zeros(5, cfg.hidden_size, dtype=torch.bfloat16))


def test_logits_vs_original(model, tok):
    prompt = aime_prompt(tok, 4)
    assert CHUNK < prompt.shape[1] < 2 * CHUNK                  # prefill 묶음이 둘이고, 둘째 묶음은 채워서 맞춘다
    budget = prompt.shape[1] + 64
    press = ThinKVPress(budget=budget, labels="sparsity", **SHORT)
    with press(model):
        ref = sample(model, prompt, tok, output_logits=True, return_dict_in_generate=True)
    forced = ref.sequences[0, prompt.shape[1]:].tolist()

    eng = Engine.from_config(model.config, torch.bfloat16, N, attn="fused", budget=budget, labels="sparsity", **SHORT)
    rot = model.model.rotary_emb
    seen = []

    def pick(logits, step):                                     # 원래 모델이 고른 토큰을 그대로 넣는다
        seen.append(logits.float())
        return forced[step]

    out, T = generate(TorchRunner(pieces(model)), model.model.embed_tokens, Rope(rot.inv_freq, rot.attention_scaling),
                      eng, prompt[0].tolist(), len(forced), pick)
    diff = torch.stack([(a - b[0].float()).abs().max() for a, b in zip(seen, ref.logits)])
    top1 = sum(int(a.argmax() == b[0].argmax()) for a, b in zip(seen, ref.logits)) / len(seen)
    same_plan = eng.plan.log["labels"] == press.last_log["labels"] and eng.plan.log["cache_len"] == press.last_log["cache_len"]
    print(f"\n스텝 {len(seen)}, 점수 차이 최대 {diff.max():.4f} 중앙값 {diff.median():.4f}, 1등 토큰 일치 {top1:.1%}")
    print("스텝별 최대 차이(16스텝마다):", [round(x, 3) for x in diff[::16].tolist()])
    print("라벨·퇴출 뒤 길이도 같음:", same_plan, "| 라벨", "".join(eng.plan.log["labels"]), "/", "".join(press.last_log["labels"]))
    print({k: round(v, 2) for k, v in T.items()})
    assert out == forced and torch.isfinite(diff).all()
    assert diff.max() <= TOL
