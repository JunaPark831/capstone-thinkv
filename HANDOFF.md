# HANDOFF — ThinKV 재현과 온디바이스 구현

마지막 갱신: 2026-11-20 (그 전 판은 2026-10-01, 0단계 직후)

## 지금 상태

**Mac에서 할 수 있는 일은 끝났다. 다음은 보드(VENTUNO Q) 실행과 측정이고, 보드 담당 팀원이 한다.** 진행 현황과 결과 요약은 `README.md`가 정본이다.

| 단계 | 상태 | 정본 문서 |
|---|---|---|
| 0. 환경과 기준선 | 완료 | 이 문서 아래 "0단계 기록" |
| 1. 희소도 3봉 확인 | 완료. 미리 정한 기준과 논문 Algorithm 1 모두에서 3봉이 나오지 않았다 | `docs/e3_branch_a_criteria.md`, `docs/e3_branch_a_2type_fallback.md`, `docs/e3_paper_calibration_spec.md` |
| 2. 사고 유형 분류기 | 완료. 대체 임계값 사용 | `docs/e4_implementation_guide.md` |
| 3. TBQ + TBE | 구현 완료. 평가는 AIME 2024 10문제를 한 번씩만 돌렸다 | `docs/e4_tbe_tbq_design.md`, `thinkv/result/` |
| 4. CT 커널 벤치마크 | 1차 측정 완료(RTX 4090) | `docs/e10_ct_kernel_design.md` 9절 |
| 5. 온디바이스 구현 | 완료. `ondevice/tests` 19개 통과(8B, Mac) | `docs/ondevice_npu_cpu_design.md`, `ondevice/HANDOFF.md` |
| 6. 보드 실행과 측정 | 예정 | `ondevice/README.md` |

## 관련 문서

| 파일 | 내용 |
|---|---|
| `README.md` | 진행 현황, 단계별 결과, 다이어그램 |
| `docs/ondevice_npu_cpu_design.md` | 보드용 NPU + CPU 분업 설계, 결정 사항, Mac 실측, 백로그 |
| `ondevice/HANDOFF.md` | 온디바이스 코드 작업 규칙과 확정 사항 |
| `ondevice/README.md` | 보드 담당 팀원용 실행·측정 안내 |
| `docs/e10_ct_kernel_design.md` | CT 커널 설계와 벤치마크 결과, 백로그 |
| `docs/e3_*.md`, `docs/e4_*.md` | 희소도 분석과 TBQ·TBE 설계 |
| `docs/stage0.html`, `docs/stage1.html` | 0·1단계 안내서 |

설계의 기준은 Obsidian 볼트 `캡스톤디자인/version2/구현/ThinKV 구현 계획.md`다. 계획과 달라지면 계획서에 이유와 날짜를 적는다.

## 환경

- 로컬: Apple Silicon Mac(48GB), NVIDIA GPU 없음. 가상환경은 `.venv-kvpress`(Python 3.12).
- GPU: vast.ai에서 빌려 쓴다(희소도 수집은 A100, CT 커널 벤치마크는 RTX 4090).
- 보드: Arduino VENTUNO Q(Qualcomm IQ-8275, RAM 16GB).
- 모델: 지금은 `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`(32층, 쿼리 헤드 32, KV 헤드 8, 헤드 차원 128). 0단계는 R1-Distill-Qwen-1.5B로 했다.
- git: 작업 브랜치 `e10-ct-kernel`, 2026-11-20에 `main`에 합쳤다. 저장소는 공개다.

## 코드

| 폴더 | 역할 |
|---|---|
| `eval/` | 0단계 기준선 실행과 비교 |
| `calib/` | 희소도 수집, 봉우리 분석, 분류 임계값 산출물 |
| `thinkv/` | TBQ(`quant.py`, 값을 양자화한 뒤 16비트로 되돌려 저장), TBE(`press.py`), K-means, 평가 러너 |
| `kernels/`, `tests/kernels/` | 블록 장부(`block_table.py`), vLLM Triton 커널 이식, 벤치마크 |
| `ondevice/` | 보드용 구현: 진짜 2·4비트 저장, C++ 어텐션, NPU 조각, 테스트 |

## 다음 할 일

1. **보드 실행과 측정**(보드 담당 팀원). 조각을 NPU용으로 컴파일하고, 속도와 메모리를 잰다. 순서는 `ondevice/README.md`.
2. **NPU 호출 왕복 시간 확인.** 토큰당 33번 호출을 감당하지 못하면 가중치 계산도 CPU로 옮기는 방안을 검토한다(2026-11-17 팀원 제안, 미결).
3. **백로그**(착수 조건은 각 문서):
   - 퇴출 때 다시 묶어 재양자화하기, 1B급 모델 + 무압축 KV 대 8B + ThinKV 비교 (`docs/ondevice_npu_cpu_design.md` 11절)
   - 논문과의 차이 정리, 가중치·활성값·KV 정밀도별 prefill·decode 속도 측정 (`docs/e10_ct_kernel_design.md` 10·11절)
4. **열려 있는 것**: TBQ + TBE 평가는 10문제뿐이라 결론을 낼 수 없다. 균일 2비트·균일 4비트·TBQ만 설정은 실행이 오염되어 다시 돌려야 한다(`thinkv/result/`의 `.contaminated`).

## 0단계 기록 (2026-10-01 기준, 그대로 보존)

아래 세 절(결과, 검증한 사실, 결정 사항)은 0단계 때 쓴 것이고 R1-Distill-Qwen-1.5B 기준이다.

### 0단계 결과 (2026-09-30 ~ 10-01)

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

### 검증한 사실

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

### 결정 사항

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

- 2026-11-19부터 `ondevice/` 코드는 구현 세션이 직접 쓰고, 단계마다 흐름 설명 → 코드 → 테스트 순서로 사용자와 짚어 간다. 그 전에는 사용자가 직접 옮겨 적었다.
- 스펙 문서와 커밋은 설계 세션이 맡는다. 커밋·푸시는 사용자가 시킬 때 한다.
- 설계 판단은 Obsidian의 `ThinKV 구현 계획.md`를 기준으로 한다. 계획과 달라지면 계획서에 이유와 날짜를 적는다.
