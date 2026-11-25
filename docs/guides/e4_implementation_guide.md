# E4 구현 안내 — ThinKV(분류기 + TBQ + TBE)를 kvpress 위에 짜는 법 (2026-11-05)

확정값은 `docs/design/e4_tbe_tbq_design.md` 6절. 이 문서는 "어디에 어떤 코드를 두고, 어떤 순서로 무엇이 일어나야 하는가"만 적는다. 환경: kvpress 0.5.5, transformers 5.2.0, torch 2.14, `.venv-kvpress`.

## 0. 먼저 알아둘 사실 (코드를 읽어 확인한 것)

- kvpress의 press는 **각 레이어 `self_attn`의 forward hook**이다(`BasePress.__call__`이 `register_forward_hook(..., with_kwargs=True)`). 훅 인자 `kwargs`에 `hidden_states`, `past_key_values`, `cache_position`이 있고, 캐시는 `cache.layers[layer_idx].keys / .values`(모양 `(1, Hkv, n, d)`)로 직접 읽고 바꿀 수 있다. `DecodingPress`가 디코드 중에 하는 일이 정확히 이것이다: `cache_layer.keys = keys[:, :, keep]` 식으로 갈아끼운다.
- 프리필/디코드 구분은 `kvpress.presses.base_press.is_prefilling(kwargs["cache_position"], q_len)`.
- 0단계에서 `DecodingPress(KnormPress)`가 `eval/baseline.py`의 `model.generate(...)` 호출 그대로 퇴출을 수행했다(`results/baseline_knorm.jsonl`에 8000토큰 생성 뒤 `cache_len` 1343). 즉 **같은 generate 호출로 디코드 중 캐시를 줄여도 RoPE 위치·마스크가 깨지지 않는다.** 마스크 모양 오류가 나면 `attention_mask` 인자만 빼면 된다(단일 시퀀스라 불필요).
- 희소도 측정은 `calib/paper_calibration/collect_calibration.py`의 방식(`AttentionInterface.register("eager_sparsity", attn)` + `attn_implementation="eager_sparsity"`)을 그대로 쓴다. 어텐션 함수 안에서 받는 `key`는 **이번 스텝까지 갱신된 전체 캐시(RoPE 적용 후)**라 분류기 측정에 바로 쓸 수 있다. 측정은 레이어 24 한 곳만.
- 캘리브레이션의 레이어 번호는 0부터 센 `module.layer_idx`다. L* = {24}는 `layer_idx == 24`.

## 1. 파일 배치 (4개)

```
thinkv/
  quant.py      # 가짜 양자화: fq_nvfp4, fq_ternary, fq_kivi2 (+ 자기 검사)
  kmeans.py     # 구간 안 대표 토큰 고르기: select_representatives
  press.py      # ThinKVPress(BasePress): 분류기 버퍼, 구간 표, TBQ·TBE 상태 기계
eval/run_e4.py  # 설정 9개 × AIME 30 × 시드 8, 재개 가능, jsonl 기록
```

## 2. 시간 축: 스텝·구간·라벨이 언제 생기나

기호: τ = 128, 디코드 스텝 t = 1, 2, … (t번째 생성 토큰이 입력으로 들어가 캐시에 붙는 forward). 스텝 t가 끝나면 캐시 길이 = `prompt_len + t`(퇴출 전 기준).

- **구간 m**(1부터) = 생성 토큰 (m−1)·τ+1 … m·τ. 스텝 t가 속한 구간은 `m = (t−1)//τ + 1`, 구간의 첫 스텝은 `(t−1) % τ == 0`.
- **φ_t** = 스텝 t에서 레이어 24가 잰 희소도(아래 4절). 어텐션 함수 안에서 재므로 **레이어 24의 forward 중**에 생긴다. 같은 스텝에서 레이어 0~23의 훅은 φ_t가 생기기 전에, 레이어 24~31의 훅은 생긴 후에 돈다.
- **라벨(확정 1a, 전방 적용)**: 구간 m의 라벨 = `classify(mean(φ over 구간 m−1))`. 구간 m−1의 φ는 스텝 (m−1)·τ까지 모두 생겼으므로, 구간 m의 첫 스텝 t = (m−1)·τ+1의 **어느 레이어 훅에서도** 사용할 수 있다. 버퍼를 꼬리에서 자르지 말고 **인덱스로** 잘라라: `phi[(m−2)·τ : (m−1)·τ]`(phi[i]가 스텝 i+1). 꼬리로 자르면 레이어 0~23과 24~31이 다른 128개를 본다.
  - 구간 1은 앞 블록이 없다 → **R로 둔다**(가장 높은 정밀도, 안전한 쪽). 논문은 i=0에서 프롬프트 마지막 토큰의 희소도를 잰다고 볼 수 있으나, 블록 평균 정의와 맞지 않아 R 고정이 단순하다.
  - 대안(소급 적용: 블록 m의 평균으로 블록 m 자신을 라벨)은 구간이 끝날 때까지 128토큰을 풀 정밀도로 들고 있어야 해서 논문의 16토큰 버퍼 규칙이 사라진다. 전방 적용이 논문 구조를 유지하므로 기본값. 바꾸려면 `plan_step`의 라벨 결정 위치만 바뀐다.
- **TBQ**(16토큰 버퍼): 열린 구간에 쌓인 토큰 수가 16의 배수가 되는 스텝에 **캐시의 마지막 16칸**을 그 구간의 라벨 비트로 양자화한다. 퇴출은 열린 구간을 절대 건드리지 않으므로 "마지막 16칸 = 방금 찬 그룹"이 항상 성립한다. 절대 위치를 추적할 필요가 없다.
- **TBE 경우 1**: 라벨이 T인 구간 m−1이 닫히는 순간(= 구간 m의 첫 스텝)에 **그 앞의 모든 닫힌 구간(1 … m−2)**을 한 단계 더 줄인다. 닫힌 T 구간 자신은 그대로 두고(단계 0), 경우 2가 가장 먼저 집어간다.
- **TBE 경우 2**: 매 스텝 끝에 `cache_len > k`인 동안 반복: 닫힌 구간 중 중요도가 가장 낮은 유형(T < E < R) 가운데 가장 오래된 구간을 한 단계 줄인다. 모두 최소 단계(4토큰)면 멈춘다(그러면 캐시가 k를 넘는다. 논문도 최소 4토큰을 둔다. 그림 11(a): min R = 0이면 정확도가 47%로 떨어지므로 바닥을 없애지 말 것).
- **보존 단계**: 0 = 128(원본) → 1 = 64 → 2 = 32 → 3 = 16 → 4 = 8 → 5 = 4. 각 단계 전환마다 **현재 남은 토큰에 대해** K-means를 다시 돈다(64 → 32는 64개 위에서 K=32).

## 3. 상태 기계: 한 레이어(0번)가 계획하고 모든 레이어가 같은 계획을 실행

레이어별로 표를 따로 두면 32개가 어긋날 여지가 생긴다. 라벨이 전역(확정 2a)이고 퇴출 길이가 레이어와 무관하므로 **구간 표는 하나**만 두고, 스텝마다 **레이어 0의 훅이 계획**을 세우고(`plan_step`), **모든 레이어의 훅이 그 계획을 실행**한다(`apply_step`). 레이어 0의 훅은 그 스텝에서 가장 먼저 돈다.

```python
@dataclass
class ThinKVPress(BasePress):
    theta: tuple = (0.2955, 0.4932)          # calibration_chosen.json
    layer: int = 24
    tau: int = 128
    group: int = 16
    budget: int | None = 1024                # None = TBE 끔 (TBQ만, 균일 양자화)
    quant: str | None = "tbq"                # "tbq" | "nvfp4_all" | "kivi2" | None(TBE만)
    labels: str = "sparsity"                 # "sparsity" | "random" | "periodic"
    retain: tuple = (64, 32, 16, 8, 4)
    seed: int = 0

    def __post_init__(self): self.reset()

    def reset(self):
        self.prompt_len = None
        self.step = defaultdict(int)         # 레이어별 디코드 스텝 수
        self.segments = []                   # [{type, level, len, closed}]
        self.actions = {}                    # t -> [("quant", kind, n), ("evict", start, old_len, new_len)]
        self.phi = PHI; PHI.clear()          # 어텐션 함수가 채우는 모듈 전역 리스트
        global MEASURE_LAYER; MEASURE_LAYER = self.layer
        self.log = {"labels": [], "phi_block": [], "cache_len": []}

    def forward_hook(self, module, inp, kwargs, out):
        cache = kwargs["past_key_values"]; l = module.layer_idx
        q_len = kwargs["hidden_states"].shape[1]
        if is_prefilling(kwargs["cache_position"], q_len):
            if l == 0: self.reset(); self.prompt_len = q_len
            return out
        self.step[l] += 1; t = self.step[l]
        if l == 0: self.actions[t] = self.plan_step(t, cache.layers[0].keys.shape[2])
        self.apply_step(cache.layers[l], self.actions[t])
        if l == len(cache.layers) - 1: self.actions.pop(t)       # 마지막 레이어가 쓰고 버림
        return out

    @contextmanager
    def __call__(self, model):
        with super().__call__(model): yield
        self.reset()
```

`plan_step(t, cache_len)` — **이 스텝의 퇴출 전 칸 번호 체계**로 모든 action을 적는다(한 스텝에 여러 구간을 줄여도 start는 모두 "줄이기 전" 기준). 그래야 `apply_step`이 keep 마스크 하나로 끝난다.

```python
def plan_step(self, t, cache_len):                               # cache_len: 이번 토큰까지 붙은, 퇴출 전 길이
    before = [s["len"] for s in self.segments]                   # ① 기존 구간들의 퇴출 전 길이 스냅샷
    starts = [self.prompt_len + sum(before[:i]) for i in range(len(before))]
    acts, bumped = [], set()
    m = (t - 1) // self.tau + 1
    if (t - 1) % self.tau == 0:                                  # ② 구간 m 시작: 앞 구간 닫기
        if m > 1:
            prev = self.segments[-1]; prev["closed"] = True
            if prev["type"] == "T":                              #    경우 1: 앞선 닫힌 구간 전부 한 단계
                for i, s in enumerate(self.segments[:-1]):
                    if self._bump(s): bumped.add(i)
        self.segments.append({"type": self._label(m), "level": 0, "len": 0, "closed": False})
    open_ = self.segments[-1]; open_["len"] += 1                 # ③ 이번 토큰은 열린 구간에
    if self.quant and open_["len"] % self.group == 0:            # ④ TBQ: 방금 찬 마지막 16칸
        acts.append(("quant", self._kind(open_["type"]), self.group))
    total = cache_len - sum(before[i] - self.segments[i]["len"] for i in bumped)
    if self.budget is not None:                                  # ⑤ 경우 2
        while total > self.budget:
            cand = [i for i, s in enumerate(self.segments) if s["closed"] and s["level"] < len(self.retain)]
            if not cand: break                                   #    모두 바닥(4토큰)이면 포기
            typ = next(x for x in "TER" if any(self.segments[i]["type"] == x for i in cand))
            i = min(i for i in cand if self.segments[i]["type"] == typ)   # 그 유형의 가장 오래된 것 (확정 9b)
            total -= self._bump(self.segments[i]); bumped.add(i)
    for i in sorted(bumped):                                     # ⑥ 바뀐 구간마다 evict action (좌표는 ①의 스냅샷 기준)
        acts.append(("evict", starts[i], before[i], self.segments[i]["len"]))
    self.log["cache_len"].append(total)
    return acts

def _bump(self, s):            # 한 단계 줄이고 줄어든 토큰 수를 돌려줌 (len을 바로 갱신)
    if s["level"] >= len(self.retain): return 0
    s["level"] += 1; new = min(s["len"], self.retain[s["level"] - 1])
    d = s["len"] - new; s["len"] = new; return d
```

열린 구간은 ①의 스냅샷에 (새로 만든 경우) 없거나 절대 bump되지 않으므로, 좌표 체계는 "이번 스텝 퇴출 전 캐시"로 일관된다. `_kind`: `quant == "tbq"`면 R·E → `"nvfp4"`, T → `"ternary"`; `"nvfp4_all"`·`"kivi2"`는 라벨 무시.

`apply_step(layer, acts)` — 레이어마다 자기 키로 K-means를 돌리지만 **남기는 개수는 같으므로** 표는 하나로 유지된다.

```python
def apply_step(self, layer, acts):
    K, V = layer.keys, layer.values                               # (1, Hkv, n, d)
    for a in acts:
        if a[0] == "quant":
            _, kind, n = a
            K[..., -n:, :] = FQ[kind](K[..., -n:, :], axis=-2)   # 키: 토큰 16개 묶음(채널별)
            V[..., -n:, :] = FQ[kind](V[..., -n:, :], axis=-1)   # 값: 채널 16개 묶음(토큰별)
    ev = [a for a in acts if a[0] == "evict"]
    if ev:
        keep = torch.ones(K.shape[2], dtype=torch.bool, device=K.device)
        for _, start, old, new in ev:
            idx = select_representatives(K[0, :, start:start + old], new)   # (old,) 중 new개, 정렬
            keep[start:start + old] = False; keep[start + idx] = True
        layer.keys, layer.values = K[:, :, keep], V[:, :, keep]
```

불변식(매 스텝 레이어 0에서 assert 하나): `sum(s["len"] for s in segments) == layer.keys.shape[2] - prompt_len`.

## 4. 분류기: 어텐션 함수와 φ

`collect_calibration.attn`을 복사해 **레이어 24, q_len == 1일 때만** 재고, 채널은 `win256_mid@max1` 하나만 계산한다(`replay_masked.measure`의 정의를 슬라이스로 옮긴 것).

```python
PHI = []            # 스텝마다 float 하나. press.reset()이 비운다.
MEASURE_LAYER = 24  # press.reset()이 self.layer를 여기에 써 넣는다. 함수 안에 24를 박아 두면 θ와 레이어가 따로 놀 수 있다.

def phi_win256_mid(q, k, scaling):            # q (1,H,1,d), k (1,Hkv,n,d) — RoPE 적용 후, 이번 스텝까지 포함
    H, Hkv, n = q.shape[1], k.shape[1], k.shape[2]
    g = n - 1
    lo, hi = max(4, g - 255), g - 8               # 싱크 0~3, 마지막 8칸(자기 포함) 제외
    if hi < lo: return float("nan")
    s = (q[0, :, 0].view(Hkv, H // Hkv, -1) @ k[0, :, lo:hi + 1].transpose(-1, -2)) * scaling
    row = s.float().amax(1).softmax(-1).mean(0)   # 식3 max-pool → 식4 softmax(구간 안) → KV그룹 평균
    return (row < 0.01 * row.amax()).float().mean().item()

def attn(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    out, w = eager_attention_forward(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kw)
    if module.layer_idx == MEASURE_LAYER and query.shape[2] == 1:
        PHI.append(phi_win256_mid(query, key, scaling))
    return out, w

AttentionInterface.register("eager_sparsity", attn)
AttentionMaskInterface.register("eager_sparsity", eager_mask)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager_sparsity")
```

`_label(m)`: `labels == "sparsity"`이면 `m == 1 → "R"`, 아니면 `φ̄ = nanmean(PHI[(m-2)τ:(m-1)τ])`에 대해 `φ̄ < θ1 → "E"`, `< θ2 → "R"`, 그 외 `"T"`. `"random"`이면 시드 고정 RNG로 E/R/T를 (0.19, 0.70, 0.11)에서 뽑는다(모의 분포). `"periodic"`은 `"RRRRRRREET"[(m-1) % 10]`. `φ̄`와 라벨은 `self.log`에 남겨라(아래 7절의 이유).

**주의(측정 공간이 캘리브레이션과 다르다).** θ는 FullKV·bf16 캐시의 원래 위치 공간에서 구했다. 실행 중에는 (i) 마지막 256칸이 "칸 번호" 기준이라 K-means로 줄어든 옛 구간이 섞이고, (ii) 키가 역양자화된 값이다. 이것이 배포된 시스템이 보는 값이므로 그대로 두되, φ̄ 분포가 θ에서 밀리면 블록 로그로 바로 보인다. 싱크 제외 "앞 4칸"은 프롬프트를 퇴출하지 않기(확정 3a) 때문에 유효하다.

## 5. 가짜 양자화 (`thinkv/quant.py`)

입력은 `(…, n, d)` 슬라이스(bf16). float32로 올려 계산하고 bf16으로 돌려놓는다. `axis=-2`는 토큰 축 16개 묶음(키, 채널별 스케일), `axis=-1`은 채널 축 16개 묶음(값, 토큰별 스케일). `unflatten(axis, (-1, 16))`으로 묶으면 두 경우가 같은 코드다(16칸 슬라이스, head_dim 128 둘 다 16으로 나누어떨어진다).

- **E4M3 스케일(확정 5b)**: `s8 = s.to(torch.float8_e4m3fn).float()`. E4M3의 최소 서브노멀은 약 0.002라 작은 그룹이 0으로 죽는다. NVFP4처럼 **텐서당 fp32 스케일 한 단계를 더** 둬라: `glob = x.abs().amax() / (6 * 448)`, `s = e4m3(g_abs / 6 / glob) * glob`. 호출(= 레이어 × 16토큰 슬라이스) 단위가 "텐서"다.
- **NVFP4(확정 6b, R·E 4비트)**: 그룹 최대 `g_abs`로 `s` 계산 → `y = x / s`(|y| ≤ 6) → E2M1 격자 {0, .5, 1, 1.5, 2, 3, 4, 6}에 **가장 가까운 값**으로 반올림(경계 .25, .75, 1.25, 1.75, 2.5, 3.5, 5에 `torch.bucketize`) → `q * s`.
- **Ternary(확정 7a, T 2비트)**: 그룹별 `s = mean(|x|)`를 E4M3로(위와 같은 2단계), `q = sign(x) · [|x| > 0.5 s]`, 결과 `q * s`.
- **KIVI식 2비트(대조군)**: 그룹별 비대칭 균등: `mn, mx` → `scale = (mx − mn)/3`, `q = round((x − mn)/scale).clamp(0, 3)`, 복원 `q·scale + mn`. 축은 키 채널별/값 토큰별 그대로. (KIVI 원논문은 2비트에 그룹 32를 쓴다. 16으로 맞추되 기록해 둘 것.)
- **균일 4비트(대조군)**: 라벨 무시하고 모든 그룹에 NVFP4.

자기 검사(파일 끝 `__main__`): `x = torch.randn(2, 8, 16, 128)`에 대해 상대 RMSE가 NVFP4 < 0.15, ternary < 0.8, kivi2 < 0.5이고, `axis=-2`와 `axis=-1`의 결과가 서로 다르며(스케일 축이 실제로 다른지), 입력 모양이 보존되는지 assert.

## 6. K-means 대표 토큰 (`thinkv/kmeans.py`)

`select_representatives(keys, K)`: `keys (Hkv, n, d)` → 헤드별로 다른 토큰을 고르면 캐시가 울퉁불퉁해져 kvpress 구조로 못 담는다. **레이어당 토큰 집합 하나**: 토큰을 헤드 연결 벡터 `(n, Hkv·d)`(float32)로 보고 K-means를 돈다(D.4 "구간의 키에 K-means"의 레이어 단위 해석. 기록해 둘 것).

- 초기화: `linspace(0, n−1, K)` 위치의 토큰(결정적, 기기 무관). 반복 10회(확정 8a). `torch.cdist` → `argmin` → 군집 평균.
- 대표: 군집마다 중심에 가장 가까운 **실제 토큰**(확정 8a). 빈 군집이 생기면 아직 안 뽑힌 토큰 중 자기 중심에서 가장 먼 것을 넣어 **정확히 K개**를 맞춘다.
- 반환은 **정렬된** 인덱스(순서 보존, RoPE는 이미 키에 적용돼 있어 재회전 없음, kvpress 기본과 같음). 값은 같은 인덱스로 자른다.
- 자기 검사: `n=128, K=64`에서 반환이 64개·중복 없음·정렬, 그리고 `K == n`이면 전체 반환.

## 7. 평가 러너 (`eval/run_e4.py`)

`eval/baseline.py`의 `baseline()`을 그대로 재사용한다(프롬프트 템플릿, 샘플링 0.6/0.95, `math_verify` 채점, `cache_len`). 바뀌는 것은 세 가지뿐: 모델을 `attn_implementation="eager_sparsity"`로 한 번만 로드(설정 9개 모두 같은 어텐션 → 수치 조건 동일), `max_new_tokens=32768`, 결과에 `press.log` 요약을 덧붙임.

```python
SETTINGS = {
  "fullkv":   None,
  "kivi2":    dict(quant="kivi2",     budget=None),
  "uniform4": dict(quant="nvfp4_all", budget=None),
  "snapkv":   "snapkv",   # DecodingPress(SnapKVPress(window_size=64), compression_interval=128, target_size=1024)
  "thinkv":   dict(quant="tbq",  budget=1024, labels="sparsity"),
  "tbq_only": dict(quant="tbq",  budget=None, labels="sparsity"),
  "tbe_only": dict(quant=None,   budget=1024, labels="sparsity"),
  "random":   dict(quant="tbq",  budget=1024, labels="random"),
  "periodic": dict(quant="tbq",  budget=1024, labels="periodic"),
}
```

- 루프: 설정 × 문제 30 × 시드 0~7. 한 줄 = 한 생성, `results/e4/{setting}.jsonl`에 바로 append. (설정, 문제, 시드)가 이미 있으면 건너뜀 → 재개·샤딩(`--shard i/N`을 (문제, 시드) 쌍에 적용)은 `collect_calibration.py`와 같은 방식.
- 기록: `correct, gen_len, truncated, seconds, cache_len(끝), input_len` + `labels`(구간별 R/E/T 문자열), `phi_block`(블록 평균 목록), `mean_cache_len`(스텝 평균), `avg_bits`(생성 토큰 기준 R·E 4, T 2 가중 평균), `n_evictions`.
- 집계: 문제별 8회 평균 → pass@1, 설정별 30문제 평균. 생성 길이 비율(논문 그림 10d)도 같이.
- 데이터: AIME 2024에서 15, 2025에서 15 — Cai et al. 2025(R-KV)의 목록을 따른다(논문 E.1). R-KV 저장소의 데이터 파일을 그대로 쓰고, 못 찾으면 HF의 AIME 2024(30)·2025(30)에서 시드 0으로 15개씩 뽑고 그 사실을 기록.
- 예산 k = 1024는 **프롬프트 포함 전체 캐시 길이**로 센다(kvpress `target_size`와 같은 뜻). 프롬프트 100~150토큰이면 생성분 예산은 ~900. 생성분만 세고 싶으면 `plan_step`의 `total`에서 `prompt_len`을 빼면 된다. 어느 쪽인지 기록할 것.

**시간.** HF generate 단일 스트림은 A100에서 8B bf16 25~30 tok/s다(측정 없이도 메모리 대역폭 한계). AIME 평균 생성 9,020토큰(논문) × 240회 ≈ 216만 토큰 → **설정당 약 24시간**, 9개면 한 GPU로 9일. 앞서 말한 "10시간 안팎"은 낮게 잡은 것이었다. 줄이는 수단은 두 가지: (1) 샤딩으로 인스턴스 여러 대에 나눠 돌리기(스크립트가 재개 가능하면 공짜), (2) **라벨이 없는 설정(fullkv, kivi2, uniform4)**은 같은 프롬프트 시드 8개를 **배치 8**로 한 번에 생성(생성 길이가 달라도 generate가 패딩 처리; 양자화는 배치 차원에 그대로 적용). `tbq_only`는 라벨이 시퀀스마다 다르므로 φ(`q[0]`, `k[0]`에 배치 차원 추가), 구간 표, 그룹별 양자화 종류가 전부 원소별이 되고 먼저 끝난 시퀀스의 패딩 토큰이 구간에 섞인다 → 손질이 아니라 다시 짜는 일이다. 퇴출 설정은 캐시 길이가 원소마다 달라져 단일 스트림으로 둔다. 순서는 1회 생성(시드 0)으로 9개 설정을 먼저 다 보고, 그다음 시드 1~7을 채우는 것이 안전하다.

## 8. 스모크 테스트 (A100 올리기 전에 Mac에서)

0단계 환경의 Qwen2.5-1.5B(28레이어, 같은 `eager_attention_forward`)로 `max_new_tokens=400, tau=16, group=16, budget=96, retain=(8, 4), layer=24, theta=(0.3, 0.5)`를 돌려 다음을 확인한다. θ는 의미 없고 **기계 동작**만 본다(τ=16이라 보존 단계도 16보다 작게 둔다).
1. 매 스텝 불변식(3절) 통과, `cache_len`이 `budget + tau` 안에서 톱니 모양.
2. 라벨에 T가 한 번은 나오도록 `labels="periodic"`으로 경우 1이 실제로 발화하는지(앞 구간들의 level이 한꺼번에 오르는지).
3. 양자화 켠/끈 두 실행의 `cache_len` 궤적이 같고(TBQ는 길이를 안 바꿈), 출력 텍스트는 달라지는지.
4. `thinkv` 설정으로 생성이 EOS까지 끝나고 텍스트가 깨지지 않는지(2비트 T가 많으면 깨질 수 있다. 그때는 확정 14의 메모대로 R을 FP8로 올리는 실험을 추가).
5. `snapkv` 설정(`DecodingPress(SnapKVPress(window_size=64), compression_interval=128, target_size=1024)`)도 EOS까지 돌려 `cache_len ≤ 1024 + 128`인지. 0단계가 증명한 것은 `DecodingPress(Knorm)`뿐이고, SnapKV는 디코드 중 hidden-state 창과 `position_embeddings`가 필요해 따로 확인해야 한다.

## 9. 미리 알아둘 결과 기대치

- 논문 그림 10(f): AIME에서 R/E/T = 29/30/41 %, 즉 **논문에서는 Transition이 가장 흔하다.** 우리 분류기(#7)의 모의 분포는 E/R/T = 19/70/11 %. T가 적으니 평균 비트는 논문의 3.2~3.4가 아니라 **3.8 근처**가 나오고, 2비트 손실도 적게 나타난다. 결과 비교 때 이 차이를 먼저 적어야 한다.
- 분류기의 한 스텝 라벨은 잡음에 가깝고 블록 평균도 전환율이 ~37 %다(`calibration_chosen.json`의 decode_sim). 무작위 라벨·주기 라벨 설정과 차이가 없으면 그것이 결론이다. 그래서 12번을 (c)로 고른 것이 맞다.

## 10. 기본값으로 둔 열린 점 (바꾸고 싶으면 그 자리만)

| 점 | 기본값 | 어디 |
|---|---|---|
| 라벨 적용 방향 | 전방(블록 m−1 평균 → 구간 m), 구간 1 = R | `_label` |
| 측정 공간 | 실행 중 캐시의 칸 번호 공간, 역양자화된 키 | `phi_win256_mid` |
| 경우 2 대상 | 닫힌 구간만(열린 구간은 길이가 아직 변함) | `plan_step` cand |
| 경우 1 범위 | 닫힌 T 구간보다 앞선 구간 전부, T 구간 자신은 제외 | `plan_step` |
| 경우 2 시점 | 매 스텝 `cache_len > k`(§4.3 문장). 그림 10(b)처럼 "예산 아래 선제 퇴출"로 하려면 구간 경계에서 `cache_len + τ ≤ k`가 될 때까지 | `plan_step` while |
| k의 범위 | 프롬프트 포함 전체 캐시 | `plan_step` total |
| K-means 단위 | 레이어당 토큰 집합 하나(헤드 연결 벡터) | `select_representatives` |
| NVFP4 스케일 | 그룹 E4M3 × 텐서 fp32 2단계 | `quant.py` |
| KIVI 그룹 | 16(원논문 2비트는 32) | `fq_kivi2` |
| 주기 라벨 패턴 | `RRRRRRREET` 반복 | `_label` |
