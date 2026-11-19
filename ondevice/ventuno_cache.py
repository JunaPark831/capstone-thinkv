"""M2 장부 연결: 생성 토큰의 KV를 진짜 양자화해 풀 두 개에 담고, 퇴출은 빈칸 표시만 한다.

층마다 담는 것
  프롬프트 KV   [Hkv, P, D]   모델 정밀도. 양자화·퇴출 없음
  열린 묶음     [Hkv, 16, D]  모델 정밀도. 아직 16개가 안 찬 최근 토큰
  풀 두 개      nvfp4 풀(R·E 구간), ternary 풀(T 구간). 장부는 kernels/block_table.BlockTable

토큰 하나(decode)를 처리하는 순서는 층마다 ThinKVPress와 같다: 버퍼에 추가 → 어텐션 → (16개가 차면) 양자화 → 퇴출.
퇴출 계획(Plan)은 thinkv/press.py의 plan_step을 옮긴 것이다. press.py는 kvpress를 불러서 보드에 깔기 무겁다.
"""
import random

import torch

from kernels.block_table import BlockTable, E, R, T
from ondevice.ventuno_attn import fused
from ondevice.ventuno_quant import BITS, G, decode, encode
from thinkv.kmeans import select_representative_tokens

TYPE = {"T": T, "E": E, "R": R}       # 장부에는 R과 E를 다른 유형으로 넘긴다 (빈칸을 서로 재사용하지 않음, 2026-11-19 결정)


def phi_win256_mid(q, k, scaling):
    """thinkv/press.py의 같은 이름 함수와 같은 계산. q [Hq, D] 이번 토큰, k [Hkv, n, D] 논리 순서 키(이번 토큰 포함).
    최근 256칸(앞 4칸, 마지막 8칸 제외)에서 점수가 최댓값의 1% 미만인 칸의 비율 = 희소도."""
    H, Hkv, n = q.shape[0], k.shape[0], k.shape[1]
    g = n - 1
    lo, hi = max(4, g - 255), g - 8
    if hi < lo:
        return torch.full((), float("nan"))
    s = (q.view(Hkv, H // Hkv, -1) @ k[:, lo:hi + 1].transpose(-1, -2)) * scaling
    row = s.float().amax(1).softmax(-1).mean(0)
    return (row < 0.01 * row.amax()).float().mean()


class Plan:
    """ThinKVPress.plan_step을 옮긴 것: 구간 나누기, 라벨, 예산, 보존 단계.

    원본과 다른 점 두 가지. 퇴출 동작을 ("evict", 구간 번호, 새 길이)로 돌려준다(원본은 논리 시작 위치).
    양자화는 언제나 TBQ다(T 구간은 ternary, R·E 구간은 nvfp4).
    """

    def __init__(self, theta=(0.2955, 0.4932), tau=128, budget=1024, labels="sparsity", retain=(64, 32, 16, 8, 4), seed=0):
        assert tau % G == 0, "구간 길이가 16의 배수여야 구간이 닫힐 때 모든 토큰이 양자화돼 있다"
        self.theta, self.tau, self.budget, self.labels, self.retain = theta, tau, budget, labels, retain
        self.rng = random.Random(seed)
        self.segments = []      # [{type, level, len, closed}]
        self.phi = []           # 측정 층의 희소도. decode 토큰마다 하나 (원본의 PHI)
        self.log = {"labels": [], "phi_block": [], "cache_len": []}

    def step(self, t, cache_len):
        """t번째 decode 토큰에서 할 일 목록. cache_len: 이 토큰까지 붙은, 퇴출 전 길이(프롬프트 포함)."""
        before = [s["len"] for s in self.segments]
        acts, bumped = [], set()
        m = (t - 1) // self.tau + 1
        if (t - 1) % self.tau == 0:                       # 구간 m 시작: 앞 구간을 닫는다
            if m > 1:
                prev = self.segments[-1]
                prev["closed"] = True
                if prev["type"] == "T":                   # 경우 1: 전환 구간이 끝나면 그 앞 구간들을 한 단계씩 줄인다
                    for i, s in enumerate(self.segments[:-1]):
                        if self._bump(s):
                            bumped.add(i)
            self.segments.append({"type": self._label(m), "level": 0, "len": 0, "closed": False})
        open_ = self.segments[-1]
        open_["len"] += 1                                 # 이번 토큰은 열린 구간에 들어간다
        if open_["len"] % G == 0:                         # 방금 찬 마지막 16개를 양자화
            acts.append(("quant", "ternary" if open_["type"] == "T" else "nvfp4"))
        total = cache_len - sum(before[i] - self.segments[i]["len"] for i in bumped)
        if self.budget is not None:                       # 경우 2: 예산을 넘으면 T → E → R 순, 오래된 구간부터 줄인다
            while total > self.budget:
                cand = [i for i, s in enumerate(self.segments) if s["closed"] and s["level"] < len(self.retain)]
                if not cand:
                    break                                 # 모두 바닥이면 포기한다. 그래서 예산은 상한이 아니다
                typ = next(x for x in "TER" if any(self.segments[i]["type"] == x for i in cand))
                i = min(i for i in cand if self.segments[i]["type"] == typ)
                total -= self._bump(self.segments[i])
                bumped.add(i)
        acts += [("evict", i, self.segments[i]["len"]) for i in sorted(bumped)]
        self.log["cache_len"].append(total)
        return acts

    def _bump(self, s):
        """보존 단계를 하나 올리고, 줄어든 토큰 수를 돌려준다."""
        if s["level"] >= len(self.retain):
            return 0
        s["level"] += 1
        new = min(s["len"], self.retain[s["level"] - 1])
        d = s["len"] - new
        s["len"] = new
        return d

    def _label(self, m):
        phi = float("nan")
        if self.labels == "periodic":
            lab = "RRRRRRREET"[(m - 1) % 10]
        elif self.labels == "random":
            lab = self.rng.choices("ERT", weights=(0.19, 0.70, 0.11))[0]
        elif m == 1:
            lab = "R"
        else:
            a = (m - 2) * self.tau
            phi = torch.stack(self.phi[a:a + self.tau]).nanmean().item()
            if phi != phi:
                lab = "R"
            else:
                lab = "E" if phi < self.theta[0] else "R" if phi < self.theta[1] else "T"
        self.log["labels"].append(lab)
        self.log["phi_block"].append(phi)
        return lab


def _grow(t, n, cap):
    """첫 축이 n칸 이상이 되게 늘린다(두 배씩, 최대 cap). 처음부터 잡지 않고 쓰는 만큼만 잡는다(2026-11-19 결정)."""
    if t.shape[0] >= n:
        return t
    out = t.new_zeros((min(max(n, 2 * t.shape[0]), cap),) + t.shape[1:])
    out[: t.shape[0]] = t
    return out


class Pool:
    """한 가지 비트 폭(nvfp4 또는 ternary)의 저장소와 장부.

    슬롯별  k, v [S, Hkv, D*bits/8] 코드 묶음 · vs [S, Hkv, D/16] 값 스케일 · gid [S] 묶음 번호
    묶음별  ks [n_gid, Hkv, D] 키 스케일 · kg, vg [n_gid] glob
    키 스케일은 슬롯이 아니라 묶음 번호로 찾으므로, 빈칸을 토큰 하나씩 재사용해도 된다.
    묶음 번호(키 스케일 행)는 그 묶음의 토큰이 모두 퇴출될 때만 반납한다(결정 A).
    """

    def __init__(self, kind, hkv, d, max_tokens):
        self.kind, nb = kind, d * BITS[kind] // 8
        self.bt = BlockTable(max_tokens // G + 3, G)    # 장부만 최대 생성 길이로 잡는다(블록당 수십 바이트)
        self.bt.add_seq()
        self.cap = self.bt.live_mask.size               # 슬롯 번호의 상한
        u8 = dict(dtype=torch.uint8)
        self.k, self.v = torch.empty(0, hkv, nb, **u8), torch.empty(0, hkv, nb, **u8)
        self.vs = torch.empty(0, hkv, d // G, **u8)
        self.gid = torch.empty(0, dtype=torch.int32)
        self.ks = torch.empty(0, hkv, d, **u8)
        self.kg, self.vg = torch.empty(0), torch.empty(0)
        self.live = []          # 묶음 번호별 살아 있는 토큰 수
        self.free = []          # 반납된 묶음 번호

    def put(self, seg, ttype, k, v):
        """토큰 16개 k·v [Hkv, 16, D]를 양자화해 장부가 준 슬롯 16개에 쓴다."""
        kp, ks, kg = encode(k, -2, self.kind)
        vp, vs, vg = encode(v, -1, self.kind)
        slots = torch.tensor([self.bt.append(0, seg, ttype) for _ in range(G)])
        g = self.free.pop() if self.free else len(self.live)
        if g == len(self.live):
            self.live.append(0)
        self.live[g] = G
        n = int(slots.max()) + 1
        self.k, self.v, self.vs, self.gid = (_grow(t, n, self.cap) for t in (self.k, self.v, self.vs, self.gid))
        self.ks, self.kg, self.vg = (_grow(t, g + 1, self.cap) for t in (self.ks, self.kg, self.vg))
        self.k[slots], self.v[slots], self.vs[slots] = kp.transpose(0, 1), vp.transpose(0, 1), vs.transpose(0, 1)
        self.gid[slots] = g
        self.ks[g], self.kg[g], self.vg[g] = ks, kg, vg

    def evict(self, slots):
        """빈칸 표시만 한다. 묶음의 마지막 토큰이 나가면 그 묶음 번호를 반납한다."""
        self.bt.evict(0, slots)
        for g in self.gid[slots].tolist():
            self.live[g] -= 1
            if self.live[g] == 0:
                self.free.append(g)

    def dense(self, slots, dtype):
        """슬롯들을 풀어 K, V [Hkv, n, D]로 돌려준다."""
        s = torch.tensor(slots)
        g = self.gid[s].long()
        K = decode(self.k[s], self.ks[g], self.kg[g][:, None, None], -2, self.kind, dtype)
        V = decode(self.v[s], self.vs[s], self.vg[g][:, None, None], -1, self.kind, dtype)
        return K.transpose(0, 1), V.transpose(0, 1)


class Layer:
    """층 하나의 KV: 프롬프트 + 열린 묶음 + 풀 두 개."""

    def __init__(self, hkv, d, dtype, max_tokens):
        self.pk, self.pv = torch.empty(hkv, 0, d, dtype=dtype), torch.empty(hkv, 0, d, dtype=dtype)
        self.bk, self.bv = torch.zeros(hkv, G, d, dtype=dtype), torch.zeros(hkv, G, d, dtype=dtype)
        self.nb = 0                                     # 열린 묶음에 든 토큰 수
        self.pools = {kind: Pool(kind, hkv, d, max_tokens) for kind in BITS}
        self.segs = []                                  # 구간 순서대로 (풀, 장부의 Segment)

    def __len__(self):
        return self.pk.shape[1] + sum(len(s.slots) for _, s in self.segs) + self.nb

    def quant(self, kind, seg, ttype):
        """열린 묶음(16개)을 풀에 넣는다. seg: 퇴출 계획의 구간 번호."""
        pool = self.pools[kind]
        pool.put(seg, ttype, self.bk, self.bv)
        if seg == len(self.segs):
            self.segs.append((pool, pool.bt.seqs[0].segments[-1]))
        self.nb = 0

    def evict(self, seg, new):
        """구간 seg를 new개로 줄인다. 남길 토큰은 원본처럼 그 구간의 (풀어 낸) 키로 K-means를 돌려 고른다."""
        pool, s = self.segs[seg]
        K, _ = pool.dense(s.slots, self.pk.dtype)
        keep = set(select_representative_tokens(K, new).tolist())
        pool.evict([x for j, x in enumerate(s.slots) if j not in keep])

    def dense(self, dtype=None):
        """논리 순서(프롬프트 → 구간들 → 열린 묶음)로 풀어 낸 K, V [Hkv, n, D]."""
        dtype = dtype or self.pk.dtype
        parts = [(self.pk, self.pv)] + [p.dense(s.slots, dtype) for p, s in self.segs if s.slots]
        parts.append((self.bk[:, : self.nb], self.bv[:, : self.nb]))
        return [torch.cat(x, 1).to(dtype) for x in zip(*parts)]


def eager(q, K, V, scaling):
    """transformers의 eager_attention_forward와 같은 연산(비트 비교용). q [Hq, D], K·V [Hkv, n, D] → [Hq, D]."""
    r = q.shape[0] // K.shape[0]
    K, V = (x[:, None].expand(-1, r, -1, -1).reshape(q.shape[0], *x.shape[1:]) for x in (K, V))
    w = torch.matmul(q[None, :, None], K[None].transpose(2, 3)) * scaling
    w = torch.softmax(w, -1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(w, V[None])[0, :, 0]


class Engine:
    """CPU 쪽: 층마다 KV 저장 + 퇴출 + decode 어텐션.

    attn="fused": C++ 융합 커널 (ventuno_attn). 기본값이고 보드에서 쓰는 길이다
    attn="torch": 풀어서 PyTorch로 계산 (M2 임시 구현). ThinKVPress와 비트 단위로 비교하는 테스트에서만 쓴다
    """

    def __init__(self, layers, hkv, d, dtype, max_tokens, measure_layer=24, attn="fused", **plan):
        self.layers = [Layer(hkv, d, dtype, max_tokens) for _ in range(layers)]
        self.plan, self.measure, self.attn = Plan(**plan), measure_layer, attn
        self.t, self.acts = 0, []
        self.scale_log = []     # decode 토큰마다 (살아 있는 생성 토큰 수, 키 스케일 바이트), 모든 층 합

    @classmethod
    def from_config(cls, cfg, dtype, max_tokens, **kw):
        """층 수·헤드 수를 모델 설정에서 읽는다(코드에 박지 않는다)."""
        d = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        return cls(cfg.num_hidden_layers, cfg.num_key_value_heads, d, dtype, max_tokens, **kw)

    def prefill(self, l, k, v):
        """프롬프트 K·V [Hkv, n, D](RoPE 뒤)를 이어 붙인다. 128토큰 묶음으로 여러 번 와도 된다."""
        L = self.layers[l]
        L.pk, L.pv = torch.cat([L.pk, k], 1), torch.cat([L.pv, v], 1)

    def attend(self, l, q, k, v, scaling):
        """decode 토큰 하나. q [Hq, D], k·v [Hkv, D] (RoPE 뒤) → 어텐션 출력 [Hq, D]."""
        L = self.layers[l]
        L.bk[:, L.nb], L.bv[:, L.nb] = k, v
        L.nb += 1
        if self.attn == "torch" or l == self.measure:
            # ponytail: 측정 층에서는 fused여도 전부 푼다. 느리면 최근 264칸만 풀게 바꾼다
            K, V = L.dense()
        if self.attn == "torch":
            out = eager(q, K, V, scaling)
        else:
            out = fused(L, q, scaling).to(q.dtype)
        if l == self.measure:
            self.plan.phi.append(phi_win256_mid(q, K, scaling))
        if l == 0:                                      # 계획은 층 0에서 한 번 세우고 모든 층이 같이 쓴다
            self.t += 1
            self.acts = self.plan.step(self.t, len(L))
        seg = len(self.plan.segments) - 1
        for a in self.acts:
            if a[0] == "quant":
                L.quant(a[1], seg, TYPE[self.plan.segments[seg]["type"]])
            else:
                L.evict(a[1], a[2])
        if l == len(self.layers) - 1:
            self.scale_log.append(self.scale_bytes())
        return out

    def scale_bytes(self):
        """(살아 있는 생성 토큰 수, 키 스케일이 차지한 바이트), 모든 층 합. 키 스케일 낭비를 재는 데 쓴다."""
        live = rows = 0
        for L in self.layers:
            for p in L.pools.values():
                live += int(p.bt.live_mask.sum())
                rows += sum(c > 0 for c in p.live) * p.ks.shape[1] * p.ks.shape[2]
        return live, rows


def attach(model, engine):
    """Mac 비교용: HF 모델의 어텐션을 엔진으로 바꾼다. HF는 자기 캐시를 계속 쌓지만 엔진은 그중 이번 토큰만 쓴다.
    되돌릴 때는 model.set_attn_implementation(원래 이름)."""
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.masking_utils import eager_mask
    from transformers.models.llama.modeling_llama import eager_attention_forward

    def attn(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        l = module.layer_idx
        if query.shape[2] > 1:                          # prefill: 프롬프트가 한 번에 온다
            engine.prefill(l, key[0], value[0])
            return eager_attention_forward(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kw)
        out = engine.attend(l, query[0, :, 0], key[0, :, -1], value[0, :, -1], scaling)
        return out[None, None], None

    AttentionInterface.register("ondevice", attn)
    AttentionMaskInterface.register("ondevice", eager_mask)   # 없으면 prefill 마스크가 달라져 첫 토큰부터 어긋난다
    model.set_attn_implementation("ondevice")
