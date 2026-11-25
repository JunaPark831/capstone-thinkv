# ondevice: VENTUNO Q에서 ThinKV 돌리기 (보드 담당 팀원용 안내)

모델은 DeepSeek-R1-Distill-Llama-8B다. 일은 둘로 나뉜다.

- **NPU**는 가중치 계산을 맡는다. 모델을 층 단위 조각 33개로 잘라 올린다.
- **CPU**는 나머지를 맡는다. KV 캐시를 진짜 2·4비트로 저장하고, 퇴출할 때는 빈칸 표시만 하며, 어텐션을 C++로 계산한다.

설계는 `docs/design/ondevice_npu_cpu_design.md`에 있다.

## 파일

| 파일 | 하는 일 |
|---|---|
| `ventuno_quant.py` | 양자화해서 코드·스케일로 만들고, 다시 푼다 |
| `ventuno_cache.py` | 층별 KV 저장(프롬프트, 열린 묶음, 풀 두 개), 퇴출 계획, `Engine` |
| `ventuno_attn.cpp`, `ventuno_attn.py` | decode 어텐션. 압축된 KV를 읽으면서 바로 계산한다(NEON, OpenMP) |
| `ventuno_pieces.py` | 모델을 NPU 조각으로 자르기, 조각 실행기, 생성 루프 `generate` |
| `measure_scale_waste.py` | 키 스케일 낭비 측정 |
| `tests/` | Mac 테스트 |

## 준비물

- 파이썬 패키지는 `torch`, `numpy`만 있으면 된다. 리포의 `kernels/block_table.py`, `thinkv/quant.py`, `thinkv/kmeans.py`도 쓰므로 리포를 통째로 가져간다. 이 세 파일도 torch·numpy만 쓴다.
- C++ 컴파일러(`g++` 등)가 필요하다.
  - 어텐션을 처음 부를 때 자동으로 컴파일돼 임시 폴더에 `ventuno_attn_*.so`로 저장된다.
  - OpenMP가 되면 KV 헤드 8개를 코어에 나눠 돈다.
  - 보드용 최적화 옵션은 환경변수로 더한다: `VENTUNO_CXXFLAGS="-mcpu=native"`.
  - 다른 컴파일러를 쓰려면 `CXX=clang++`처럼 지정한다.

## Mac에서 확인하기 (8B, bf16, RAM 약 16GB)

```bash
.venv-kvpress/bin/python -m pytest ondevice/tests -q -s
.venv-kvpress/bin/python -m ondevice.measure_scale_waste --max-new 4096
```

## 보드에서 할 일

### 1. 조각을 NPU용으로 컴파일

`pieces(model)`이 PyTorch 모듈 33개를 돌려준다. 행 수 n은 1(decode)과 128(prefill) 두 가지다. 마지막 조각은 n = 1만 있으면 된다.

| 조각 | 입력 | 출력 |
|---|---|---|
| 0 | `h [n, 4096]` | `h [n, 4096]`, `q [n, 4096]`, `k [n, 1024]`, `v [n, 1024]` |
| 1 ~ 31 | `h [n, 4096]`, `a [n, 4096]` | 위와 같음 |
| 32 | `h [1, 4096]`, `a [1, 4096]` | `logits [1, 128256]` |

`a`는 앞 층의 어텐션 출력이다. 조각 0에는 `a`가 없다. 컴파일 도구(QAIRT 직접 사용 또는 ExecuTorch Qualcomm 백엔드)는 아직 정하지 않았다. 정해지면 그 도구의 내보내기(ONNX 또는 torch.export)를 붙인다.

### 2. NPU 실행기 만들기

함수 하나만 만들면 된다: `run(i, h, a) → 조각 i의 출력`. 모양과 순서는 위 표와 같고, 텐서는 bf16으로 주고받는다. Mac에서는 `TorchRunner`가 같은 일을 한다. `TorchRunner`를 그대로 보드 CPU에서 돌리면 "전부 CPU" 대조군이 된다.

### 3. CPU 쪽 준비물 꺼내기 (Mac이나 변환용 PC에서 한 번)

```python
rot = model.model.rotary_emb
torch.save({"embed": model.model.embed_tokens.weight, "inv_freq": rot.inv_freq,
            "scale": rot.attention_scaling, "ids": ids}, "cpu_side.pt")   # ids: 토크나이저로 만든 프롬프트 토큰 리스트
```

### 4. 실행

```python
import torch
from ondevice.ventuno_cache import Engine
from ondevice.ventuno_pieces import Rope, generate, sampler

c = torch.load("cpu_side.pt")
embed = torch.nn.Embedding.from_pretrained(c["embed"])
eng = Engine(32, 8, 128, torch.bfloat16, max_tokens=4096, attn="fused")   # 층 32, KV 헤드 8, 헤드 차원 128
toks, T = generate(run, embed, Rope(c["inv_freq"], c["scale"]), eng, c["ids"], 4096, sampler(), eos=128001)
```

`max_tokens`는 최대 생성 길이다. 장부 크기만 이 값으로 잡고, KV 저장 공간은 쓰는 만큼 늘어난다. 압축 설정(구간 128토큰, 예산 1024, 보존 64·32·16·8·4, 희소도 라벨)의 기본값은 GPU 실험과 같다.

### 5. 측정

| 항목 | 재는 법 |
|---|---|
| 첫 토큰 지연 | `T["first"]` |
| 토큰당 지연 | `T["decode"] / len(toks)` |
| NPU 시간과 CPU 어텐션 시간의 비율 | `T["npu"]`과 `T["cpu"]` |
| 최대 상주 메모리 | `/usr/bin/time -v python ...`의 "Maximum resident set size" |
| 키 스케일 낭비 | `eng.scale_log` (생성 토큰마다 살아 있는 토큰 수와 키 스케일 바이트) |

대조군은 둘이다. 하나는 같은 코드를 `TorchRunner`로 돌린 "전부 CPU"이고, 다른 하나는 벤더 NPU 경로(GenieX)다.
