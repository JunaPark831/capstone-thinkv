import random
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from kvpress.presses.base_press import BasePress, is_prefilling
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.masking_utils import eager_mask
from transformers.models.llama.modeling_llama import eager_attention_forward

from thinkv.kmeans import select_representative_tokens
from thinkv.quant import FQ

PHI = [] #MEASURE_LAYER 24에서의 희소도값,,decoding step마다 누적한다.
MEASURE_LAYER = 24 #from calibration

def phi_win256_mid(q, k, scaling):
    # q (1,H,1,d), k (1,Hkv,n,d): RoPE 적용 후, 이번 스텝까지 포함
    # 현재 토큰의 쿼리 q, 지금까지 캐신된 키 k
    H, Hkv, n = q.shape[1], k.shape[1], k.shape[2]
    g = n - 1 #직전 스텝
    lo, hi = max(4, g - 255), g - 8         # 싱크(앞) 0~3, 마지막 8칸 제외
    if hi < lo:
        return torch.full((), float("nan"), device=q.device)
    s = (q[0, :, 0].view(Hkv, H // Hkv, -1) @ k[0, :, lo:hi + 1].transpose(-1, -2)) * scaling
    row = s.float().amax(1).softmax(-1).mean(0)          # 식3 max-pool -> 식4 softmax -> KV그룹 평균
    return (row < 0.01 * row.amax()).float().mean() #0.01이하인것 수 반환,즉 희소도값 반환

def attn(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    #attention수행하면서 layer 24의 희소도값 누적하는 것 추가한것.
    out, w = eager_attention_forward(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kw)
    if module.layer_idx == MEASURE_LAYER and query.shape[2] == 1:
        PHI.append(phi_win256_mid(query, key, scaling))
    return out, w


AttentionInterface.register("eager_sparsity", attn)
AttentionMaskInterface.register("eager_sparsity", eager_mask)


@dataclass
class ThinKVPress(BasePress):
    theta: tuple = (0.2955, 0.4932)
    layer: int = 24
    tau: int = 128
    group: int = 16
    budget: int | None = 1024          # None = TBE 끔
    quant: str | None = "tbq"          # "tbq" | "nvfp4_all" | "kivi2" | None(TBE만)
    labels: str = "sparsity"           # "sparsity" | "random" | "periodic"
    retain: tuple = (64, 32, 16, 8, 4)
    seed: int = 0

    def __post_init__(self):
        self.last_log = None
        self.reset()

    def reset(self):
        global MEASURE_LAYER
        MEASURE_LAYER = self.layer
        PHI.clear()
        self.prompt_len = None
        self.step = defaultdict(int)     # 레이어별 디코드 스텝 수
        self.segments = []               # [{type, level, len, closed}]
        self.actions = {}                # t -> [("quant", kind, n), ("evict", start, old_len, new_len)]
        self.rng = random.Random(self.seed)
        self.log = {"labels": [], "phi_block": [], "cache_len": []}

    def forward_hook(self, module, inp, kwargs, out):
        cache = kwargs["past_key_values"]
        l = module.layer_idx
        q_len = kwargs["hidden_states"].shape[1]
        if is_prefilling(kwargs["cache_position"], q_len):
            if l == 0:
                self.reset()
                self.prompt_len = q_len
            return out
        self.step[l] += 1
        t = self.step[l]
        if l == 0:
            self.actions[t] = self.plan_step(t, cache.layers[0].keys.shape[2])
        self.apply_step(cache.layers[l], self.actions[t])
        if l == 0:
            assert sum(s["len"] for s in self.segments) == cache.layers[0].keys.shape[2] - self.prompt_len
        if l == len(cache.layers) - 1:
            self.actions.pop(t)          # 마지막 레이어가 쓰고 버림
        return out

    def plan_step(self, t, cache_len):   # cache_len: 이번 토큰까지 붙은, 퇴출 전 길이
        before = [s["len"] for s in self.segments]                    # ① 기존 구간들의 퇴출 전 길이
        starts = [self.prompt_len + sum(before[:i]) for i in range(len(before))]
        acts, bumped = [], set()
        m = (t - 1) // self.tau + 1
        if (t - 1) % self.tau == 0:                                   # ② 구간 m 시작: 앞 구간 닫기
            if m > 1:
                prev = self.segments[-1]
                prev["closed"] = True
                if prev["type"] == "T":                               #    경우 1
                    for i, s in enumerate(self.segments[:-1]):
                        if self._bump(s):
                            bumped.add(i)
            self.segments.append({"type": self._label(m), "level": 0, "len": 0, "closed": False})
        open_ = self.segments[-1]
        open_["len"] += 1                                             # ③ 이번 토큰은 열린 구간에
        if self.quant and open_["len"] % self.group == 0:             # ④ TBQ: 방금 찬 마지막 16칸
            acts.append(("quant", self._kind(open_["type"]), self.group))
        total = cache_len - sum(before[i] - self.segments[i]["len"] for i in bumped)
        if self.budget is not None:                                   # ⑤ 경우 2
            while total > self.budget:
                cand = [i for i, s in enumerate(self.segments) if s["closed"] and s["level"] < len(self.retain)]
                if not cand:
                    break                                             #    모두 바닥이면 포기
                typ = next(x for x in "TER" if any(self.segments[i]["type"] == x for i in cand))
                i = min(i for i in cand if self.segments[i]["type"] == typ)
                total -= self._bump(self.segments[i])
                bumped.add(i)
        for i in sorted(bumped):                                      # ⑥ 바뀐 구간마다 evict
            acts.append(("evict", starts[i], before[i], self.segments[i]["len"]))
        self.log["cache_len"].append(total)
        return acts

    def _bump(self, s):                  # 한 단계 줄이고 줄어든 토큰 수를 돌려줌
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
            phi = torch.stack(PHI[a:a + self.tau]).nanmean().item()
            if phi != phi:
                lab = "R"
            else:
                lab = "E" if phi < self.theta[0] else "R" if phi < self.theta[1] else "T"
        self.log["labels"].append(lab)
        self.log["phi_block"].append(phi)
        return lab

    def _kind(self, typ):
        if self.quant == "tbq":
            return "ternary" if typ == "T" else "nvfp4"
        return "kivi2" if self.quant == "kivi2" else "nvfp4"

    def apply_step(self, layer, acts):
        K, V = layer.keys, layer.values                               # (1, Hkv, n, d)
        for a in acts:
            if a[0] == "quant":
                _, kind, n = a
                K[..., -n:, :] = FQ[kind](K[..., -n:, :], -2)         # 키: 토큰 16개 묶음
                V[..., -n:, :] = FQ[kind](V[..., -n:, :], -1)         # 값: 채널 16개 묶음
        ev = [a for a in acts if a[0] == "evict"]
        if ev:
            keep = torch.ones(K.shape[2], dtype=torch.bool, device=K.device)
            for _, start, old, new in ev:
                idx = select_representative_tokens(K[0, :, start:start + old], new)
                keep[start:start + old] = False
                keep[start + idx] = True
            layer.keys, layer.values = K[:, :, keep], V[:, :, keep]

    @contextmanager
    def __call__(self, model):
        try:
            with super().__call__(model):
                yield
        finally:
            self.last_log = self.log     # reset()이 self.log를 새로 만들기 전에 보관
            self.reset()

