"""M2: 압축 저장 + 장부 + 옮겨 온 퇴출 계획이 ThinKVPress(가짜 양자화 + 복사)와
생성 토큰·라벨·퇴출 뒤 길이까지 같은지 본다(8B, bf16).
실행(루트에서): .venv-kvpress/bin/python -m pytest ondevice/tests/test_ventuno_cache.py -q -s"""
import pytest
import torch

from kernels.block_table import R
from ondevice.tests.conftest import N, SHORT, sample
from ondevice.ventuno_cache import Engine, Pool, attach
from thinkv.press import ThinKVPress


@pytest.mark.parametrize("labels", ["periodic", "sparsity"])
def test_same_as_thinkvpress(model, tok, prompt, labels):
    budget = prompt.shape[1] + 64
    press = ThinKVPress(budget=budget, labels=labels, **SHORT)
    with press(model):
        ref = sample(model, prompt, tok)
    log = press.last_log

    # 비트 단위로 비교하므로 원본과 같은 계산을 하는 임시 어텐션(torch)을 쓴다. C++ 쪽은 test_ventuno_attn.py가 본다
    eng = Engine.from_config(model.config, torch.bfloat16, N, attn="torch", budget=budget, labels=labels, **SHORT)
    attach(model, eng)
    try:
        out = sample(model, prompt, tok)
    finally:
        model.set_attn_implementation("eager_sparsity")

    print(labels, "".join(log["labels"]), "최종 길이", log["cache_len"][-1])
    assert log["cache_len"][-1] < prompt.shape[1] + N        # 퇴출이 실제로 일어났다
    if labels == "periodic":
        assert "T" in log["labels"]                           # 세 유형(R·E·T)이 모두 나왔다
    assert eng.plan.log["labels"] == log["labels"]
    assert eng.plan.log["cache_len"] == log["cache_len"]
    assert torch.equal(out, ref)


def test_scale_row_returned_when_group_empty():
    """묶음의 토큰이 모두 퇴출되면 키 스케일 행(묶음 번호)을 반납하고, 다음 묶음이 그 행을 다시 쓴다."""
    p = Pool("nvfp4", 8, 128, 64)
    x = torch.randn(8, 16, 128, dtype=torch.bfloat16)
    p.put(0, R, x, x)
    p.put(0, R, x, x)
    first = p.bt.seqs[0].segments[0].slots[:16]
    p.evict(first[:15])
    assert p.free == []                                       # 하나라도 남으면 반납하지 않는다
    p.evict(first[15:])
    assert p.free == [0]
    p.put(1, R, x, x)
    assert p.free == [] and p.live == [16, 16]
    assert sorted(p.bt.seqs[0].segments[-1].slots) == sorted(first)   # 빈칸도 다시 쓴다
