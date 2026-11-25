# E3 분기점 A 실패와 2유형 fallback 계획

- 결정 시각: 2026-10-08 15:30 KST
- 상태: E3의 사전 등록된 3봉 기준은 통과하지 못했다.
- 결정: 7B에서 완료된 P=8을 고정 자료로 사용하고, 원래의 R/E/T 3유형 캘리브레이션을 중단한다. 4번 레이어의 반복적인 2봉 신호를 후보로 삼아 `|T|=2` 축소 버전을 탐색한다.
- 성격: 이 문서는 사전 등록 기준을 변경하지 않는다. 2유형 분석은 3봉 실패를 확인한 뒤 선택한 **사후 탐색적 fallback**이다.

## 1. 기존 사전 기준과 판정

원래의 분기점 A 기준은 `docs/criteria/e3_branch_a_criteria.md`에 고정되어 있다.

- 각 프롬프트·레이어에 KDE를 적용한다.
- Scott factor ×0.5, ×1, ×2를 비교한다.
- `find_peaks(..., prominence=0.05)`에서 봉우리가 정확히 3개일 때 3봉으로 판정한다.
- 동일한 레이어가 세 bandwidth 모두에서 3봉 비율 50% 이상이어야 한다.
- 위 조건을 만족하는 공통 레이어가 4개 이상이어야 분기점 A를 통과한다.

이 기준은 결과를 본 뒤 완화하거나 변경하지 않는다.

## 2. 사용 데이터

2유형 fallback의 고정 입력은 통합 Qwen-7B run에서 처음 8개 파일이다. `0008.pt`~`0010.pt`는 원래 P=8 fallback 결정 이후 수집되어 이 고정 입력에는 포함하지 않는다.

```text
calib/sparsity/qwen7b_s1k_p11_max35000/
├── 0000.pt
├── 0001.pt
├── 0002.pt
├── 0003.pt
├── 0004.pt
├── 0005.pt
├── 0006.pt
└── 0007.pt  # P=8 고정 입력의 마지막 파일
```

수집 조건:

- 모델: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- 데이터: `simplescaling/s1K-1.1`, `cot_type == "math"`, `shuffle(seed=0)`의 앞 8문제
- 샘플링: seed 0, temperature 0.6, top-p 0.95
- 생성 상한: 35,000토큰
- 어텐션: eager
- 희소도: 각 헤드에서 어텐션 최댓값의 1% 미만인 KV 위치의 비율을 전체 헤드·KV 위치에 대해 평균

8개 결과는 모두 EOS로 자연 종료했다. 생성 길이는 5,055~24,925토큰, 평균 13,437.25토큰, 총 107,498토큰이다. 따라서 이 P=8 결과의 3봉 부재는 생성 상한에서 잘린 것만으로 설명되지 않는다.

## 3. 3봉 실패 근거

최신 통합 Qwen-7B P=11 근거 기록과 그림:

```text
calib/criteria_artifacts/qwen7b_s1k_p11_max35000/criteria_result.json
calib/criteria_artifacts/qwen7b_s1k_p11_max35000/criteria_result.md
calib/criteria_artifacts/qwen7b_s1k_p11_max35000/e3_kde_scott_x0_5.png
calib/criteria_artifacts/qwen7b_s1k_p11_max35000/e3_kde_scott_x1.png
calib/criteria_artifacts/qwen7b_s1k_p11_max35000/e3_kde_scott_x2.png
calib/criteria_artifacts/qwen7b_s1k_p11_max35000/e3_three_mode_fractions.png
```

기존 P=8 artifact 디렉터리는 P=11 통합 결과로 대체했다. 아래 P=8 수치와 fallback 결정은 당시 검증한 기록을 그대로 보존한다.

P=8에는 bandwidth별로 8 × 28 = 224개의 프롬프트·레이어 조합이 있다.

| Bandwidth | 정확히 3봉인 조합 | 3봉 비율 50% 이상인 레이어 |
|---|---:|---|
| Scott ×0.5 | 0/224 | 없음 |
| Scott ×1 | 0/224 | 없음 |
| Scott ×2 | 0/224 | 없음 |

남은 두 프롬프트가 모든 레이어에서 3봉이어도 레이어별 최대치는 2/10 = 20%다. 따라서 원래 P=10의 50% 기준은 수학적으로 달성할 수 없다. 이 근거로 3유형 분기점 A 실패를 확정하고 P=8에서 fallback 설계로 넘어간다.

## 4. 2봉 후보

4번 레이어(0부터 시작하는 인덱스)를 2유형 후보로 사용한다.

| Bandwidth | 4번 레이어가 정확히 2봉인 프롬프트 |
|---|---:|
| Scott ×0.5 | 5/8 = 62.5% |
| Scott ×1 | 4/8 = 50.0% |
| Scott ×2 | 3/8 = 37.5% |

참고로 Scott ×0.5에서 3번 레이어와 7번 레이어가 각각 1/8, Scott ×1에서 7번 레이어가 1/8의 2봉을 보였다. 반복성이 가장 높은 후보는 4번 레이어다.

이 결과는 세 bandwidth에서 모두 50%를 넘지 않는다. 따라서 “안정적인 2봉이 입증됐다”고 주장하지 않고, **4번 레이어를 후속 2유형 실험의 후보로 선택했다**고 기록한다.

## 5. 3유형 `calibrate.py` 처리

원래 계획한 3유형 `calibrate.py`는 실행하지 않는다.

3봉이 없으므로 다음 값을 데이터에서 정당하게 구할 수 없다.

- 3유형용 `L*` 레이어 4개
- 3봉 사이 골 두 개 `θ₁`, `θ₂`
- 희소도만으로 정한 R/E/T 세 라벨

결과를 맞추기 위해 bandwidth, prominence 또는 통과 비율을 사후 조정하지 않는다. `calib/criteria_checker.py`의 FAIL 기록을 원래 분기점 A의 결과로 보존한다.

## 6. 다음 구현: `calibrate_2type.py`

새 파일의 목표:

```text
calib/calibrate_2type.py
```

입력:

```text
calib/sparsity/qwen7b_s1k_p11_max35000/0000.pt ... 0007.pt
```

출력:

```text
calib/calib_2type.json
calib/two_type_artifacts/layer04_kde_valleys.png
calib/two_type_artifacts/layer04_theta_timeline_*.png
```

### 고정 분석 설정

- 후보 레이어: 4
- KDE 기준: Scott 기본 factor(×1)
- 평가 구간: `[0, 1]`
- 평가 격자: 512점
- 밀도 정규화: 각 KDE를 자신의 최댓값으로 나눔
- 봉우리 탐지: `scipy.signal.find_peaks`
- prominence: 0.05
- 사용할 프롬프트: Scott ×1에서 정확히 2봉인 프롬프트
- 문제별 임계값: 두 봉우리 사이 KDE 최솟값
- 최종 임계값: 문제별 임계값의 중앙값

Scott ×1을 쓰는 이유는 사전 3봉 분석의 기준 bandwidth이고, 좁거나 넓게 조정한 보조 폭이 아니기 때문이다. 이 선택 역시 2유형 fallback의 사후 탐색 설정으로 기록한다.

### 문제별 골 계산

```python
peaks, _ = find_peaks(density, prominence=0.05)

if len(peaks) == 2:
    left, right = peaks
    between = density[left : right + 1]
    valley_idx = left + np.argmin(between)
    theta_p = xs[valley_idx]
```

Scott ×1에서 정확히 2봉인 프롬프트가 현재 4개이므로 `theta_per_problem`은 4개여야 한다.

```python
final_theta = float(np.median(theta_per_problem))
```

중앙값을 사용하는 이유는 P=8의 적은 표본에서 특정 문제의 골 위치가 최종 임계값을 과도하게 움직이는 것을 줄이기 위해서다.

### 출력 JSON

```json
{
  "schema_version": 1,
  "method": "branch_a_two_type_fallback",
  "analysis_status": "post_hoc_exploratory",
  "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
  "P": 8,
  "source_dir": "calib/sparsity/qwen7b_s1k_p11_max35000",
  "source_indices": [0, 1, 2, 3, 4, 5, 6, 7],
  "num_thought_types": 2,
  "layer": 4,
  "bandwidth": "scott_x1",
  "prominence": 0.05,
  "qualifying_prompt_indices": [],
  "theta_per_problem": [],
  "theta": 0.0,
  "original_three_type_criterion_passed": false
}
```

`theta`를 계산했다는 사실만으로 두 사고 유형이 의미 있게 분리됐다고 결론 내리지 않는다.

## 7. E3′: 두 군집의 의미 확인

임계값을 구한 뒤에는 라벨을 바로 Reasoning/Execution으로 부르지 않고 중립적으로 시작한다.

```python
label = "LOW" if sparsity < theta else "HIGH"
```

희소도와 토큰의 정렬은 다음과 같다.

```text
sparsity[4, j] ↔ token_ids[j]
token_ids의 마지막 토큰에는 대응하는 다음 디코드 hook 값이 없음
```

따라서 분석에는 `token_ids[:-1]`과 `sparsity[4]`를 맞춘다. 128스텝 이동평균을 함께 그리되, 분류 자체에 이동평균을 사용할지는 별도 결정으로 남긴다.

저장된 토큰을 디코딩해 다음 표현과 LOW/HIGH 구간의 관계를 확인한다.

- 풀이 계획과 논리 설명: Reasoning 후보
- 계산식과 정해진 절차 수행: Execution 후보
- `Wait`, `However`, `Alternatively`, 재검토와 방향 변경: Transition 후보
- 최종 답 정리와 `\boxed{}` 주변

확인 질문:

1. LOW/HIGH가 텍스트상 서로 다른 행동과 대응하는가?
2. 같은 행동이 임계값 양쪽에 무작위로 섞이지 않는가?
3. Transition이 독립 군집이 아니라 다른 유형에 합쳐졌다고 설명할 근거가 있는가?
4. 4번 레이어의 2봉이 일부 문제에만 나타나는 이유가 생성 길이, 문제 유형 또는 특정 구간 비율 때문인가?

## 8. 온라인 분류기로 넘어가는 조건

다음 항목을 확인한 뒤에만 2유형 온라인 분류기로 이동한다.

- `calib_2type.json`과 KDE/골 그림이 생성됨
- 문제별 `theta_p`와 최종 중앙값 `theta`가 `[0, 1]` 안에 있음
- 각 `theta_p`가 해당 문제의 두 봉우리 사이에 있음
- E3′ 그림에서 LOW/HIGH가 해석 가능한 텍스트 행동 차이를 보임
- 결과 문서에 2유형이 원래 3유형 재현이 아니라 fallback임을 명시함

온라인 분류기의 최초 형태:

```python
if layer4_sparsity < theta:
    thought_type = "LOW"
else:
    thought_type = "HIGH"
```

텍스트 분석 전에는 LOW/HIGH를 R/E/T 이름으로 바꾸지 않는다.

## 9. 해석과 범위 제한

이 분기의 결론은 다음 범위로 제한한다.

- 1.5B와 7B 로컬 실험에서 원 논문의 3봉 구조를 재현하지 못했다.
- 7B P=8에서는 4번 레이어가 가장 반복적인 2봉 후보였다.
- 2유형 fallback은 ThinKV의 완전한 R/E/T 재현이 아니라 SLM 조건에서 가능한 축소 버전을 탐색하는 실험이다.
- 향후 더 큰 모델이나 다른 신호에서 3봉이 관찰되더라도 이번 FAIL 기록은 삭제하거나 소급 변경하지 않는다.

## 10. 실행 순서

```text
[완료] P=8 스냅샷 고정 및 무결성 검증
[완료] 3봉 기준 FAIL 확인
[완료] 4번 레이어의 2봉 빈도 확인
[다음] calib/calibrate_2type.py 작성
[다음] theta 및 KDE/골 그림 저장
[다음] E3′ LOW/HIGH 토큰 구간 시각화
[다음] 두 군집의 의미를 확인한 뒤 2유형 온라인 분류기 구현
```
