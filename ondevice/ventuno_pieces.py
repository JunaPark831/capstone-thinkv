"""M3.5 모델 자르기: 층 단위 NPU 조각 + 조각 실행기 + 생성 루프.

조각 i (0 ≤ i < 층 수 L): [층 i-1의 출력 투영 + 잔차 + MLP] + [층 i의 정규화 + Q·K·V 투영]  → h, q, k, v
조각 L (마지막)        : [층 L-1의 출력 투영 + 잔차 + MLP] + 최종 정규화 + 출력층              → logits
그래서 NPU 호출은 토큰당 L+1번이다(8B는 33번). 입력 행 수는 decode 1, prefill 128 두 가지로 고정한다.
RoPE, 어텐션, KV 캐시는 CPU(ventuno_cache.Engine)가 한다.

보드에서는 run(i, h, a)만 NPU 호출로 바꾸면 된다. 나머지(이 파일의 generate, Engine)는 그대로 쓴다.
"""
import time

import torch
import torch.nn.functional as F
from torch import nn

CHUNK = 128   # prefill 묶음 크기 (스펙 결정 6의 초기값)


class Piece(nn.Module):
    """NPU 조각 하나. 가중치는 복사하지 않고 HF 모델의 층 모듈을 가리킨다."""

    def __init__(self, prev=None, cur=None, norm=None, head=None):
        super().__init__()
        self.prev, self.cur, self.norm, self.head = prev, cur, norm, head

    def forward(self, h, a=None):
        """h [n, hidden] 잔차 흐름, a [n, Hq*D] 앞 층의 어텐션 출력(조각 0은 없음)."""
        if self.prev is not None:
            h = h + self.prev.self_attn.o_proj(a)
            h = h + self.prev.mlp(self.prev.post_attention_layernorm(h))
        if self.cur is None:
            return self.head(self.norm(h))
        x, at = self.cur.input_layernorm(h), self.cur.self_attn
        return h, at.q_proj(x), at.k_proj(x), at.v_proj(x)


def pieces(model):
    """HF Llama 계열 모델 → 조각 L+1개."""
    m, ls = model.model, model.model.layers
    L = len(ls)
    return [Piece(ls[i - 1] if i else None, ls[i] if i < L else None,
                  m.norm if i == L else None, model.lm_head if i == L else None) for i in range(L + 1)]


class TorchRunner:
    """Mac용 조각 실행기: 조각을 PyTorch로 돌린다. 보드에서는 같은 모양의 run(i, h, a)를 NPU 호출로 바꾼다."""

    def __init__(self, pieces):
        self.pieces = pieces

    @torch.no_grad()
    def __call__(self, i, h, a=None):
        assert h.shape[0] in (1, CHUNK), "NPU 조각은 행 1개(decode)와 128개(prefill)로만 컴파일된다"
        return self.pieces[i](h, a)


class Rope:
    """transformers의 LlamaRotaryEmbedding + apply_rotary_pos_emb와 같은 계산.
    보드에는 inv_freq와 scale 두 값만 가져가면 된다: Rope(model.model.rotary_emb.inv_freq, model.model.rotary_emb.attention_scaling)"""

    def __init__(self, inv_freq, scale=1.0):
        self.inv_freq, self.scale = inv_freq.float(), scale

    def __call__(self, x, pos):
        """x [n, H, D], pos [n] 정수 위치."""
        f = pos[:, None].float() * self.inv_freq
        e = torch.cat([f, f], -1)
        cos, sin = (e.cos() * self.scale).to(x.dtype)[:, None], (e.sin() * self.scale).to(x.dtype)[:, None]
        x1, x2 = x.chunk(2, -1)
        return x * cos + torch.cat([-x2, x1], -1) * sin


def prefill_attn(q, L, s, scaling):
    """prefill 묶음의 어텐션(PyTorch SDPA, 스펙 결정 6). q [128, Hq, D], 키는 지금까지 쌓인 프롬프트 KV 전체."""
    mask = torch.arange(L.pk.shape[1]) <= torch.arange(s, s + q.shape[0])[:, None]   # 자기 위치까지만 본다
    o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], L.pk[None], L.pv[None], attn_mask=mask,
                                       scale=scaling, enable_gqa=True)
    return o[0].transpose(0, 1).reshape(q.shape[0], -1)


def sampler(temperature=0.6, top_p=0.95, seed=0):
    """GPU 실험과 같은 샘플링(온도 0.6, top-p 0.95). 보드에 transformers 없이 쓰려고 직접 썼다."""
    g = torch.Generator().manual_seed(seed)

    def pick(logits, step):
        p, idx = (logits.float() / temperature).softmax(-1).sort(descending=True)
        p[p.cumsum(0) - p > top_p] = 0           # 누적 확률이 top_p를 넘은 뒤의 토큰은 버린다
        return int(idx[torch.multinomial(p, 1, generator=g)])

    return pick


@torch.no_grad()
def generate(run, embed, rope, engine, ids, max_new, pick, eos=None):
    """조각 실행기 run으로 생성한다. ids: 프롬프트 토큰 리스트, pick(logits [vocab], step) → 다음 토큰.
    돌려주는 것: (생성 토큰 리스트, 시간 초 {"first": 첫 토큰까지, "npu": 조각 실행 합, "cpu": 어텐션·KV 합, "decode": decode 전체})."""
    hkv, _, d = engine.layers[0].pk.shape
    scaling, L, P = d ** -0.5, len(engine.layers), len(ids)
    T = {"npu": 0.0, "cpu": 0.0}

    def timed(key, f, *a):
        t = time.perf_counter()
        out = f(*a)
        T[key] += time.perf_counter() - t
        return out

    t0 = time.perf_counter()
    for s in range(0, P, CHUNK):                 # prefill: 128토큰 묶음. 마지막 묶음은 채워서 모양을 맞추고 채운 자리는 버린다
        n = min(CHUNK, P - s)
        h, a, pos = embed(torch.tensor(ids[s:s + n] + [0] * (CHUNK - n))), None, torch.arange(s, s + CHUNK)
        for i in range(L):
            h, q, k, v = timed("npu", run, i, h, a)
            q, k, v = rope(q.view(CHUNK, -1, d), pos), rope(k.view(CHUNK, hkv, d), pos), v.view(CHUNK, hkv, d)
            engine.prefill(i, k[:n].transpose(0, 1), v[:n].transpose(0, 1))
            a = timed("cpu", prefill_attn, q, engine.layers[i], s, scaling)
    logits = timed("npu", run, L, h[n - 1:n], a[n - 1:n])[0]   # 출력층은 마지막 진짜 토큰 한 줄만
    T["first"] = time.perf_counter() - t0

    out = []
    for step in range(max_new):
        tok = pick(logits, step)
        out.append(tok)
        if tok == eos or step == max_new - 1:
            break
        h, a, pos = embed(torch.tensor([tok])), None, torch.tensor([P + step])
        for i in range(L):
            h, q, k, v = timed("npu", run, i, h, a)
            q, k = rope(q.view(1, -1, d), pos)[0], rope(k.view(1, hkv, d), pos)[0]
            a = timed("cpu", engine.attend, i, q, k, v.view(hkv, d), scaling).reshape(1, -1)
        logits = timed("npu", run, L, h, a)[0]
    T["decode"] = time.perf_counter() - t0 - T["first"]
    return out, T
