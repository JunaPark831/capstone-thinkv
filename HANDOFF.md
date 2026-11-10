# HANDOFF — ThinKV 재현 (kvpress)

마지막 갱신: 2026-10-01

## 지금 상태

**0단계(환경 + FullKV 기준선) 완료. 다음은 1단계(희소도 3봉 확인, E3, 분기점 A).** 1단계 안내서는 `docs/stage1.html`(볼트에도 같은 파일). 1단계 코드는 아직 한 줄도 없다.

## 관련 문서

리포 안:

| 파일 | 내용 |
|---|---|
| `docs/stage0.html` | 0단계 안내서 사본(볼트 원본 그대로). 실제 구현과 다른 곳은 아래 "검증한 사실"이 정본 |
| `docs/stage1.html` | 1단계 안내서. 희소도 정의, 수집·분석 스크립트 설계, 검증한 코드 조각, 분기점 A 판정, 완료 체크리스트 |

Obsidian 볼트 `캡스톤디자인/`:

| 문서 | 내용 |
|---|---|
| `version2/구현/ThinKV 구현 계획.md` | 전체 설계의 기준. 부품별 설계(§3), 실험 매트릭스 E0~E11(§4), 코드 구조(§5) |
| `version2/구현/v2 로드맵.md` | 주 단위 일정, 분기점 A·B·C |
| `version2/구현/ThinKV 0단계 — 환경과 FullKV 기준선.html` | 0단계 안내서 원본 |
| `version2/구현/ThinKV 1단계 — 희소도 3봉 확인.html` | 1단계 안내서 (`docs/stage1.html`과 같은 파일) |
| `version2/프레임워크/kvpress.md` | kvpress 조사 노트. `DecodingPress` 동작, 함정 목록 |
| `version1/번역/논문2 한국어 완역 - ThinKV.md` | 논문 번역. Algorithm 1은 509행 근처, 희소도 정의는 각주 2 |

## 환경

- 머신: Apple M5 Pro, 48GB, NVIDIA GPU 없음 → 장치는 MPS
- 가상환경: `.venv-kvpress` (Python 3.12.13). 루트의 `.venv`는 torch도 없는 빈 환경이라 쓰지 않는다.
- 버전: torch 2.14.0, transformers 5.2.0, kvpress 0.5.5, math-verify 0.9.0, datasets 5.0.1, scipy 1.18.1, matplotlib 3.11.2, pandas 2.3.3
- 모델: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, bf16. 28층, 쿼리 헤드 12, KV 헤드 2
- 데이터: 평가 `alessiodevoto/math500`(split `test`), 캘리브레이션 `simplescaling/s1K-1.1`(split `train`)
- git: `main`에 첫 커밋(2026-10-01). `.venv*`, `__pycache__`, `.DS_Store`는 `.gitignore`

## 코드

| 파일 | 역할 |
|---|---|
| `eval/baseline.py` | MATH-500 앞 10문제를 press별로 생성·채점. 결과 `results/baseline_<press>.jsonl` |
| `eval/compare.py` | 결과 JSONL을 모두 읽어 요약표 출력 + `results/compare.png`(생성 길이·KV 캐시·속도 3칸) |

실행은 프로젝트 루트에서 `.venv-kvpress/bin/python eval/baseline.py`, 그다음 `eval/compare.py`.

## 0단계 결과 (2026-09-30 ~ 10-01)

설정: MATH-500 앞 10문제, 시드 0, 샘플 1, `max_new_tokens=8000`, temperature 0.6, top_p 0.95. knorm = `DecodingPress(KnormPress(), compression_interval=512, target_size=1024)`.

| 지표 | none | knorm |
|---|---|---|
| 정답 | 9/10 | 6/10 |
| 8000토큰에서 잘림 | 1 (`prealgebra/1139`) | 5 |
| 생성 길이 최소/평균/최대 | 935 / 2769 / 8000 | 935 / 4696 / 8000 |
| 초당 토큰 (중앙값) | 44.0 | 36.5 |
| 총 시간 | 약 11분 | 약 21분 |
| 압축이 걸린 문제 / 그 문제들의 KV 유지율 | — | 8 / 평균 0.41 (8000까지 간 문제는 약 0.16~0.17) |

- **캐시 확인 통과.** none은 10줄 모두 `cache_len == input_len + gen_len − 1`. knorm은 모든 줄이 "압축 안 걸림(none 공식과 같음)" 또는 "1024~1535 안" 둘 중 하나.
- **knorm은 none이 3천 토큰 안팎에서 끝낸 문제 4개를 전부 8000까지 생성했다**(`algebra/1349`, `number_theory/515`, `precalculus/927`, `intermediate_algebra/1994`). 공격적인 축출로 결론을 못 내는 현상으로 보이지만, 생성 텍스트를 저장하지 않아 반복인지 확인하지 못했다. `number_theory/515`는 잘렸는데도 채점은 정답.
- **none 속도 두 개는 오염됐다.** `algebra/2584`(33.2)와 `number_theory/572`(34.4)는 같은 시각에 어시스턴트가 같은 GPU로 스모크 테스트를 돌렸다. 속도 비교에서 빼거나 중앙값을 쓴다.
- none 재실행의 요약 수치(9/10, 생성 길이 935/2769/8000)는 옛 `baseline.jsonl`과 같았다. 옛 파일은 지웠다.

## 검증한 사실

- **MPS에서 kvpress 디코드 압축이 동작한다.** 테스트 값(32스텝마다 64개)으로 1문제 300토큰: none 캐시 369, knorm 캐시 75.
- **FullKV 캐시 길이 = 입력 길이 + 생성 길이 − 1.** 마지막으로 뽑은 토큰은 다시 모델에 들어가지 않는다.
- **boxed 지시문은 math500이 넣는다.** `question` 500개가 전부 `Remember to put your final answer within \boxed{}.`로 끝난다. **s1K에는 없다** — 1단계에서는 직접 붙인다.
- **math500 실제 컬럼:** `question, solution, answer, subject, level, unique_id, answer_prefix, context, max_new_tokens`. 데이터셋의 `max_new_tokens`(4096)는 쓰지 않는다.
- **채점은 math-verify.** 표기 차이(`\left(`, `0.5` vs `\frac{1}{2}`)는 맞게, 진짜 오답은 틀리게 판정하는 것을 확인했다.
- **`pad_token_id=tokenizer.eos_token_id`를 넘기면 "Setting pad_token_id..." 메시지가 사라지고 출력은 같다.** 이 토크나이저는 pad와 eos가 둘 다 151643.
- **`DecodingPress` 상태는 문제 사이에 안 넘어간다.** `with`가 `baseline()` 안에서 문제마다 열리고 나갈 때 `finally: self.reset()`(kvpress `decoding_press.py:181-192`). press 객체도 문제마다 새로 만든다.
- **1단계용 (2026-10-01 확인):**
  - eager로 로드하면 `layer.self_attn`의 forward hook `output[1]`이 어텐션 가중치 `(1, 12, q_len, kv_len)`. 디코드 스텝은 `q_len == 1`.
  - 희소도 배열 모양은 `(28, gen_len − 1)`(20·600토큰으로 확인).
  - hook 안에서 `.item()`을 부르면 초당 21토큰, 텐서로 쌓으면 초당 약 36토큰(eager, 600토큰).
  - `simplescaling/s1K-1.1`은 1000문제, `cot_type`이 math 877 / science 108 / crossword 15.
  - `gaussian_kde` + `find_peaks(prominence=0.05)`: 인위적 1봉 데이터가 `bw=0.1`에서 2봉으로 잡혔다. 폭 여러 개로 확인해야 한다.

## 결정 사항

| 결정 | 이유 |
|---|---|
| kvpress `pipeline`·`evaluate.py`를 쓰지 않고 `model.generate`를 직접 부른다 | 그 경로는 그리디 고정이라 E0 샘플링 조건을 못 맞춘다. 구현 계획 §5에 기록(2026-09-30) |
| press 비교는 새 파일이 아니라 `baseline.py`의 press 옵션으로 | 압축 여부만 다르고 프롬프트·샘플링·채점·시간 측정이 같아야 비교가 공정하다 |
| kvpress는 `make_press()` 안에서 지연 import하고, `PRESS_NAMES`에서 `"none"`을 항상 먼저 | `import kvpress`만 해도 어텐션 함수가 전역으로 패치된다 |
| 결과는 `results/baseline_<press>.jsonl`, 문제·시드마다 한 줄씩 즉시 append | 계획서 §5에도 반영(2026-09-30). 중간에 멈춰도 남고 `(id, seed)`로 건너뛴다 |
| 결과 경로는 `__file__` 기준 | 어디서 실행해도 `results/`에 저장된다 |
| 시드별 원자료를 저장하고 pass@1은 분석 때 계산 | 중간 재개, 분산 확인, press 간 같은 시드끼리 비교 |
| 로컬은 시드 `[0]` 하나, 서버에서 계획서 설정(시드 3 × 샘플 8, 16k) | 로컬 목적은 환경·저장·속도 확인이지 정확도 비교가 아니다 |
| dtype은 bf16, 인자 이름은 `dtype` | bf16 문제 시 fp16 말고 fp32로 |
| 동기화는 `sync_device(device)`로 mps/cuda 분기 | GPU 서버로 옮겨도 그대로 동작 |
| 비교 그림의 글자는 영어 | matplotlib 기본 폰트에서 한글이 깨진다 |
| (1단계 제안) 수집 `calib/collect_sparsity.py`와 분석 `calib/calibrate.py`를 나눈다 | KDE 폭을 바꿔 분석을 여러 번 돌려야 하는데 생성을 다시 할 수 없다. 계획서 §5는 한 파일이라, 채택하면 계획서에 적는다 |
| (1단계 제안) 캘리브레이션은 s1K math, 평가는 MATH-500 | 캘리브레이션과 평가 데이터가 겹치지 않게 |

## 다음 할 일

### 1. 1단계 — 희소도 3봉 확인 (E3, 분기점 A)

`docs/stage1.html`을 따라간다. 순서만 요약:

1. **분기점 A 판정 기준을 결과 보기 전에 계획서에 적는다**(예: 3봉 비율 50% 이상인 층이 4개 이상, KDE 폭 세 가지에서 유지).
2. `calib/collect_sparsity.py`: eager 로드, kvpress import 금지, `self_attn` hook으로 스텝별 희소도 `(28, gen_len − 1)` 수집, 문제마다 `calib/sparsity/{idx:04d}.pt`(희소도 + `token_ids`)로 저장. P=1(300토큰)로 모양 확인 → P=10(약 20~30분).
3. `calib/calibrate.py`: 층별 KDE → 3봉 층 → L* 4개 → 골 평균 θ₁·θ₂ → `calib/calib.json`. 그림 두 장(28층 KDE = E3 그림, 그림 3 재현).
4. 분기점 A 판정을 구현 계획·로드맵에 기록. 3봉이면 P=100(약 3~4시간)으로 최종 `calib.json`.

로드맵상 W2(9/23~9/29) 항목이라 이미 밀려 있다.

### 2. 병행

- **GPU 서버 결정.** Mac 속도(초당 약 44토큰)로는 E0 설정(MATH-500 × 시드 3 × 샘플 8, 16k)이 불가능하다. 24GB급 NVIDIA GPU. 로드맵 "실험 GPU 확정" 미정.
- **E3′ 키워드 베이스라인**(로드맵 W2): 1단계에서 저장한 `token_ids`를 재사용한다.
- (선택) knorm이 8000까지 간 이유를 보려면 생성 텍스트를 저장해야 한다. 지금 `baseline.py`는 텍스트를 남기지 않는다.

### 3. 이후

2단계 온라인 분류기 → 3단계 TBE → 4단계 TBQ(fake-quant) → 5단계 조립·재현 판정(E4~E6) → 6단계 CT 커널(Triton, GPU 필수). 완료 기준은 구현 계획 §3.

## 함정

- `import kvpress`는 전역 어텐션 패치. 캘리브레이션(1단계)과 none 실행에서는 import하지 않는다.
- 어텐션 가중치는 eager에서만 나온다. sdpa(기본값)에서는 hook의 `output[1]`이 `None`.
- `DecodingPress`의 상태는 `with press(model)`을 나갈 때 초기화된다. 문제마다 `with`를 새로 연다(지금 구조가 그렇다).
- `ScorerPress`의 top-k는 토큰 순서를 섞는다. TBE에서 세그먼트 표를 유지하려면 인덱스를 정렬한 뒤 gather한다.
- kvpress의 `is_prefilling`은 휴리스틱이다. ThinKV의 스텝 카운트는 직접 센다.
- KDE 봉우리 수는 폭(bandwidth)에 따라 바뀐다. `bw_method` 숫자는 절대 폭이 아니라 표준편차 배율.
- MPS는 같은 시드여도 비트 단위 재현을 보장하지 않는다. 수집 결과는 지우지 않는다.
- 같은 GPU로 다른 작업을 돌리면 `tok_per_s`가 오염된다. 실험 중에는 GPU를 쓰는 테스트를 돌리지 않는다.
- 추론 모델 pass@1은 분산이 크다. 로컬 10문제 결과로 결론을 내지 않는다.

## 작업 방식

- 코드는 사용자가 직접 작성한다. 어시스턴트는 변경안을 제시하고 임시 복사본에서 실행해 검증한다. 예외는 사용자가 명시적으로 맡긴 파일뿐이다(2026-10-01: `eval/compare.py`는 어시스턴트가 작성. 2026-11-11: E10 `kernels/`·`tests/kernels/` 전체는 어시스턴트가 작성하고 사용자가 점검 — 교수님 미팅 일정 때문).
- 설계 판단은 Obsidian의 `ThinKV 구현 계획.md`를 기준으로 한다. 계획과 달라지면 계획서에 이유와 날짜를 적는다.
