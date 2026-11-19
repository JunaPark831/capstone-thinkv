# 더 큰 모델 캘리브레이션 — 사전 기록 (R1-Distill-Qwen-14B)

- 작성 시각: 2026-11-04 (이 모델의 희소도·KDE 결과를 보기 전)
- 결정(사용자): 모델 크기 가설을 보기 위해 **14B를 먼저** 돌리고, 결과를 본 뒤 70B 진행 여부를 정한다.
- 관계: `docs/e3_paper_calibration_spec.md` §1~§5(논문 Algorithm 1 명세)를 **그대로** 쓴다. 바뀌는 것은 모델 하나다. 그 문서는 다른 세션이 계속 덧붙이고 있어 이 실험은 이 파일에 따로 적는다.
- 이 문서의 §1~§8은 결과를 본 뒤 고치지 않는다. 고칠 일이 생기면 아래에 수정 기록을 덧붙인다.

## 1. 가설

- **모델 크기 가설:** 8B(s1K P=9, 논문 식 변형 4개 × 워밍업 2 = 8조합 전부 L* 공집합)에서 3봉이 안 나온 이유가 모델 크기라면, 더 큰 모델에서는 나온다.
- 논문 근거: 그림 14에 R1-Distill-Qwen-14B가 AIME 레이어 14, LiveCodeBench 레이어 15·3에서 세 띠로 갈라진 그림이 있다. 단, 논문 캘리브레이션은 s1K 100개이고 그림은 AIME·LCB라 데이터가 다르다.
- 반증 가능성: 14B도 FAIL이면 "8B → 14B에서 개선 없음"으로 기록하고 크기 가설은 약해진다. 측정 정의(논문이 정하지 않은 KV 위치 범위, KDE 폭)가 원인일 가능성은 이 실험으로 닫히지 않는다.
- 미리 밝히는 기대(결과 아님): 8B가 논문 그림 3a와 같은 모델인데도 실패했으므로 14B에서 뒤집힐 가능성은 낮다고 본다. 그래도 가장 싸게 닫을 수 있는 한 단계라 돌린다.

## 2. 바뀌는 것 하나와 바뀌지 않는 것

**바뀌는 것:** 모델 `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` (Qwen2 구조, 48층, 쿼리 헤드 40, KV 헤드 8, head_dim 128, 그룹 크기 G=5. 설정 파일에서 확인).

**바뀌지 않는 것(8B P=15 실행과 동일):**

| 항목 | 값 |
|---|---|
| 데이터 | `simplescaling/s1K-1.1` train 전체 1000문제 `shuffle(seed=0)` 앞 15개. `idx`가 8B와 같다 |
| 최대 생성 길이 | 32768 |
| 샘플링 | temperature 0.6, top-p 0.95, 문제마다 `torch.manual_seed(0)` |
| 프롬프트 | 채팅 템플릿 + `\nPlease reason step by step, and put your final answer within \boxed{}.` |
| 측정 | 논문 식 2~4(그룹 안 쿼리 헤드 max pooling → softmax 재정규화 → 그룹 평균 → 행 최댓값 1% 미만 비율), bf16 점수 → fp32 softmax |
| 변형 5개 | `full_nosink`(위치 1..n−1), `win1024`, `win256`, `full_literal`(참고, 0..n−1), `sink_mass`(진단) |
| 수집 코드 | `calib/paper_calibration/collect_calibration.py` **수정 없음**. `--model`만 바꾼다 |
| 저장 | 문제마다 `{idx:04d}.pt`, 희소도 `(5, 48, steps)` fp16 + `token_ids` + 메타. 있으면 건너뜀 |

출력 폴더: `calib/sparsity/s1k_qwen14b_p15_max32768/`. 8B 폴더와 섞지 않는다.

## 3. 분석 규칙 (지금 고정. 오늘 8B 분석에서 여러 번 바뀐 설정을 여기서 묶는다)

**주 결과 — 논문 문자 그대로** (`calibrate_paper.py` 기본값):
- 가우시안 KDE, Scott ×1, `[0,1]` 512점 격자, `find_peaks` prominence 없음, 정확히 3봉, `L*` = 모든 문제의 교집합, θ = 골의 평균.
- 조합: 변형 3개 주(`full_nosink`, `win1024`, `win256`) + `full_literal` 참고 × 워밍업 2개(전체 스텝 / 앞 1100 제외) = **8조합**. 8B P=9 표와 같은 형식으로 보고한다.

**보조 — 잔물결 필터** (`sweep_prominence.py`, 사용자가 오늘 정한 주 단계):
- 위와 같되 `find_peaks(d/d.max(), prominence=0.05)`. 같은 8조합. 0.02·0.1·0.2 단계도 함께 보고한다.
- Scott 배율 스윕(×0.5~×8)과 "봉우리다운 것만" 필터(L1~L3)는 **8B에서 전부 FAIL이었으므로 14B에서는 기본으로 돌리지 않는다.** 주·보조 결과를 본 뒤 사용자가 요청하면 별도 사전 기록 후 돌린다.

**항상 보고하는 것:**
- 레이어별 정확히 3봉인 문제 수의 최댓값, 봉우리 수 중앙값, 1봉 비율(변형·워밍업·규칙별).
- EOS 종료 수 / 32768 상한에서 잘린 수, 생성 길이 최소·평균·최대. 8B(평균 약 1.4만)와 비교.
- 2/3 규칙(15개 중 10개 이상에서 정확히 3봉)을 만족하는 레이어, 모든 문제에서 봉우리 수가 같은 k≥2 레이어(백로그 확인용).
- 논문 그림 14가 지목한 **레이어 14**의 128스텝 이동평균 시계열과 히스토그램(문제별). 데이터가 AIME가 아니라 직접 비교는 아니고 참고다.
- 8B와의 짝 비교: 8B 완료분 9개(idx 0,1,2,3,4,5,6,9,12)와 같은 idx끼리 레이어별 봉우리 수를 나란히 둔다. 15개 중 6개(7,8,10,11,13,14)는 8B 짝이 없다.

## 4. 판정

- **성공:** 주 결과 8조합 중 하나라도 `L*`가 비어 있지 않다. 그러면 그 레이어의 문제별 히스토그램을 그려 **눈에 보이는 경계**가 있는지 사용자가 판단한다. 보조(prominence 0.05)에서도 `L*`가 남는지 병기한다. 주에서만 통과하고 보조에서 비면 "잔물결 통과"로 적고 구조의 증거로 쓰지 않는다(8B 음성 대조군 경고와 같은 규칙).
- **실패:** 8조합 모두 공집합. FAIL로 기록하고 멈춘 뒤 사용자에게 보고한다. 규칙을 완화하지 않는다.
- **P=15의 지위:** 논문 P=100이 아닌 예비 확인이다. 교집합은 문제가 늘수록 줄기만 하므로 **15개에서 비면 100개에서도 반드시 빈다.** 15개에서 안 비어도 100개에서 비지 않는다는 뜻은 아니다.
- **중간 분석:** 완료 파일이 9개 이상 모이면 8B 때처럼 미리 Algorithm 1을 돌린다. 그 시점에 모두 비면 단조성에 따라 FAIL이 확정되므로 남은 생성을 멈출 수 있다(비용 절약). 비지 않으면 15개를 채운다. 샤드 1개라 파일은 idx 순서로 완료되므로 9개 시점은 idx 0~8이고, 그때 8B와 짝이 맞는 문제는 **0~6의 7개뿐**이다(8B 짝 중 9, 12는 아직 없다). 중간에 멈추면 짝 비교는 7개로 보고하고, 15개를 채우면 9개로 보고한다.
- **크기 가설의 해석(미리):**
  - 14B FAIL → "같은 측정 정의로 8B·14B 모두 FAIL". 70B 진행은 사용자 결정이며 별도 사전 기록을 쓴다(§7).
  - 14B 성공 → 8B는 같은 정의로 FAIL이었으므로 크기 쪽 증거가 된다. 단, 측정 정의가 14B에 우연히 맞았을 가능성은 P=15로 배제하지 못한다.

## 5. 실행 절차 (vast.ai, 8B 실행과 같은 방식)

1. **인스턴스:** A100 80GB 또는 H100 80GB **1장**. 디스크 **60GB 이상**(가중치 bf16 약 29.5GB + HF 캐시. vast.ai 기본 디스크는 작아 부족하면 내려받다 죽는다). 끝나면 **destroy**(stop만 하면 저장소 요금이 계속 나간다).
2. **환경:** `pip install "transformers==5.2.0" accelerate datasets scipy tqdm` (torch는 이미지 기본). kvpress는 설치하지 않는다.
3. **파일:** `collect_calibration.py` 하나를 업로드한다(리포에 아직 untracked라 clone으로는 안 온다).
4. **스모크(필수, 본 실행 폴더와 다른 곳):**
   ```bash
   python collect_calibration.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --P 2 --max-new 2048 --out /workspace/smoke14b
   nvidia-smi   # 생성 중 다른 터미널에서, 최대 메모리 기록 (8B 때 빠뜨린 항목)
   ```
   기록할 것: `tok/s`, `eos=`, nvidia-smi 최대 메모리, 첫 `.pt`를 열어 `sparsity.shape == (5, 48, steps)`와 `variants` 순서.
5. **본 실행:**
   ```bash
   nohup python collect_calibration.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --P 15 --max-new 32768 \
     --out /workspace/s1k_qwen14b_p15_max32768 > run14b.log 2>&1 &
   ```
   **샤드 1개.** 80GB 한 장에 37GB짜리 프로세스 두 개는 32K 컨텍스트에서 빠듯하고, 8B 실행에서 샤딩이 합계 처리량을 늘리지 못했다.
6. **내려받기:** `/workspace/s1k_qwen14b_p15_max32768/*.pt` → 이 Mac의 `calib/sparsity/s1k_qwen14b_p15_max32768/`. 중간 분석을 하려면 완료분만 먼저 받아도 된다(`weights_only=True`로 로드).
7. **분석:**
   ```bash
   python calib/paper_calibration/calibrate_paper.py --dir calib/sparsity/s1k_qwen14b_p15_max32768 --variant full_nosink --out calib/paper_calibration/results_14b/full_nosink_skip0.json
   # 변형 4개 × --skip-first {0, 1100}
   python -I calib/paper_calibration/sweep_prominence.py --dir calib/sparsity/s1k_qwen14b_p15_max32768 --out calib/paper_calibration/prominence_14b
   ```

### 5-a. 붙여넣기 명령 (사용자 결정 2026-11-04: 로컬 스모크 없이 vast.ai에서 바로. `<PORT>`·`<HOST>`는 인스턴스 카드의 SSH 주소)

Mac에서:
```bash
scp -i ~/.ssh/vast_key -P <PORT> calib/paper_calibration/collect_calibration.py root@<HOST>:/workspace/
ssh -i ~/.ssh/vast_key -p <PORT> root@<HOST>
```

인스턴스에서:
```bash
cd /workspace && export HF_HOME=/workspace/hf          # 가중치 30GB를 큰 디스크에 받는다
pip install -q "transformers==5.2.0" accelerate datasets scipy tqdm
python -c "import torch,transformers;print(torch.__version__,transformers.__version__,torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv && df -h /workspace
nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 30 > mem14b.log &   # 30초마다 메모리 기록
python collect_calibration.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --P 2 --max-new 2048 --out /workspace/smoke14b 2>&1 | tee smoke14b.log
python -c "import torch;d=torch.load('/workspace/smoke14b/0000.pt',weights_only=True);print(d['sparsity'].shape,d['variants'],d['ended_by_eos'])"
sort -n mem14b.log | tail -1                           # 스모크 최대 메모리
nohup python collect_calibration.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --P 15 --max-new 32768 --out /workspace/s1k_qwen14b_p15_max32768 > run14b.log 2>&1 &
tail -f run14b.log                                     # 문제마다 "[idx] N tok, X tok/s, eos=" 한 줄
```

Mac에서 내려받기(완료분만 먼저 받아도 된다):
```bash
mkdir -p calib/sparsity/s1k_qwen14b_p15_max32768
scp -i ~/.ssh/vast_key -P <PORT> 'root@<HOST>:/workspace/s1k_qwen14b_p15_max32768/*.pt' calib/sparsity/s1k_qwen14b_p15_max32768/
scp -i ~/.ssh/vast_key -P <PORT> 'root@<HOST>:/workspace/*.log' calib/sparsity/s1k_qwen14b_p15_max32768/
```

중간·최종 분석(Mac):
```bash
mkdir -p calib/paper_calibration/results_14b
for v in full_nosink win1024 win256 full_literal; do for s in 0 1100; do
  .venv-kvpress/bin/python calib/paper_calibration/calibrate_paper.py --dir calib/sparsity/s1k_qwen14b_p15_max32768 --variant $v --skip-first $s --out calib/paper_calibration/results_14b/${v}_skip${s}.json
done; done
.venv-kvpress/bin/python -I calib/paper_calibration/sweep_prominence.py --dir calib/sparsity/s1k_qwen14b_p15_max32768 --out calib/paper_calibration/prominence_14b
```

끝나면 vast.ai 콘솔 Instances에서 **DESTROY**(휴지통). Stop은 저장소 요금이 계속 나간다.

## 6. 비용·시간 추정 (미측정 가정, 스모크 후 갱신)

- 메모리: 가중치 29.5GB + 32K KV 캐시 6.0GB(48층 × KV 8헤드 × 128 × 2바이트 × 32768) + 활성값·eager 어텐션 → 약 36~38GB. 80GB 한 장에 여유.
- 속도: 8B 단일 프로세스 2K 컨텍스트 24 tok/s, 긴 컨텍스트 6~12 tok/s(샤드 3개 동시). 14B는 가중치가 1.8배라 단일 프로세스 **8~15 tok/s** 가정.
- 토큰: 8B 평균 생성 약 1.4만 × 15 = 약 21만(14B의 생성 길이는 다를 수 있다).

| 시나리오 | GPU 시간(셋업·스모크·다운로드 1시간 포함) | A100 $0.75~1.1/h | H100 $1.6~2.3/h |
|---|---|---|---|
| 낙관(15 tok/s) | 약 5시간 | $4~6 | $8~12 |
| 중심(10 tok/s) | 약 7시간 | $5~8 | $11~16 |
| 비관(8 tok/s) | 약 8.5시간 | $6~9 | $14~20 |
| 최악(15개 전부 32K, 8 tok/s) | 약 18시간 | $14~20 | $29~41 |

시세는 장터라 매일 바뀐다(위 범위는 2026-10 트래커 기준). 빌리기 전에 확인한다.

## 7. 70B 진행 조건 (미리 적어 두는 것, 확정 아님)

- 14B 결과를 본 뒤 사용자가 정한다. 진행하면 **별도 사전 기록**을 쓴다.
- 모델: `deepseek-ai/DeepSeek-R1-Distill-Llama-70B`(80층, KV 8헤드). 가중치 bf16 약 141GB + 32K KV 10GB → **80GB 2장**, 디스크 **200GB 이상**.
- 코드 변경 3곳(사용자 작성, 제안): ① `device_map="auto"`로 두 장에 나눠 싣기, ② 입력을 첫 장치로 보내기(`.to(device)` 대신), ③ 레이어별 버퍼가 두 GPU에 흩어지므로 바깥 `torch.stack` 전에 `.cpu()`.
- 속도 4~7 tok/s(장치 간 순차 실행) 가정, P=15 약 10~14시간, $25~45.
- 논문의 70B 그림(3b)은 LiveCodeBench(코드)라 s1K와 데이터가 다르다. 데이터 선택도 그때 정한다.

## 8. 한계

- 크기와 측정 정의가 교란된다. 논문 KDE 폭·커널은 미상이라 저자 설정에서 다를 가능성은 남는다.
- P=15 예비. 14B의 생성 길이 분포가 8B와 다르면 "앞 1100 제외" 규칙의 효과도 달라진다.
- 14B는 Qwen2 구조라 `collect_calibration.py`의 `eager_attention_forward`(Llama 모듈에서 import)가 Qwen2에서도 같은 방식으로 불리는지 로컬에서 확인한다(아래 검증 기록).

---

# 검증 기록 (2026-11-04, 데이터 수집 전, 코드 검증만)

로컬(Mac, transformers 5.2.0, CPU)에서 수집 스크립트를 고치지 않고 Qwen2 구조로 돌아가는지 확인했다. 스크립트 사본이 아니라 `collect_calibration.py`의 `attn`·`measure`·`BUF`를 그대로 import해 썼다.

1. **Qwen2 경로:** 무작위 초기화한 작은 `Qwen2ForCausalLM`(2층, 쿼리 헤드 4, KV 헤드 2)에 `attn_implementation="eager_sparsity"`를 등록하고 7토큰 프롬프트에서 5토큰을 생성했다. 두 레이어 모두 디코드 스텝 4개(= 생성 토큰 − 1, 스크립트의 assert 조건)에서 변형 5개짜리 값이 쌓였고, 쌓은 결과 모양은 `(5, 2, 4)`다. n ≤ 256인 구간에서 `win256`·`win1024`가 `full_nosink`와 같은 것도 확인했다. Qwen2의 어텐션이 `sliding_window` 인자를 넘기지만 Llama의 `eager_attention_forward`가 `**kwargs`로 받아 문제없다. (무작위 작은 모델이라 희소도 값 자체는 전부 0이다. 어텐션이 거의 균등해서 최댓값의 1% 미만인 위치가 없는 것으로, 값의 의미는 없다.)
2. **14B 설정:** `Qwen2ForCausalLM`, 48층, 쿼리 헤드 40, KV 헤드 8, `use_sliding_window=False`(슬라이딩 윈도 없음, 측정 정의에 영향 없음), dtype bf16.
3. **14B 토크나이저:** 채팅 템플릿이 `<｜begin▁of▁sentence｜><｜User｜>…<｜Assistant｜><think>\n`로 시작하고, 위치 0은 BOS(id 151646)다. `full_nosink`가 빼는 위치 0이 8B(Llama BOS)와 같은 역할이다. eos = pad = 151643으로 스크립트의 `pad_token_id=eos_token_id`가 그대로 맞는다. 생성 프롬프트가 `<think>\n`을 포함하므로 8B와 같이 사고 과정부터 생성한다.

검증 스크립트는 리포 밖(세션 스크래치)에 두었고 리포에는 넣지 않았다.
