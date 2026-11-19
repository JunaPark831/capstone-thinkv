"""모델을 쓰는 테스트가 함께 쓰는 준비물. 모델은 보드와 같은 8B 하나만 쓴다(2026-11-19 결정), bf16, CPU.
8B를 한 번만 올리도록 세션 단위로 만든다(RAM 약 16GB)."""
import pytest
import torch

from ondevice.measure_scale_waste import MODEL, aime_prompt

SHORT = dict(tau=16, retain=(8, 4))     # 200토큰 안에 퇴출이 여러 번 일어나도록 줄인 설정: 구간 16토큰, 닫힌 구간은 8 → 4개
N = 200


@pytest.fixture(scope="session")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="session")
def model():
    import thinkv.press  # noqa: F401  "eager_sparsity" 어텐션(희소도 기록)을 등록한다
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="eager_sparsity").eval()


@pytest.fixture(scope="session")
def prompt(tok):
    return aime_prompt(tok)


def sample(model, ids, tok, **kw):
    """GPU 실험과 같은 샘플링(온도 0.6, top-p 0.95), 시드 0."""
    torch.manual_seed(0)
    with torch.no_grad():
        return model.generate(ids, max_new_tokens=N, do_sample=True, temperature=0.6, top_p=0.95,
                              pad_token_id=tok.eos_token_id, **kw)
