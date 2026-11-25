# E10 B단계: 읽기 안내서 (2026-11-09)

스펙(`docs/design/e10_ct_kernel_design.md`) 6절 1순서에 해당한다. **Mac에서 하고 GPU는 필요 없다.** 받은 코드는 저장소 밖(`~/dev/ref`)에 두고 **실행하지 않고 읽기만** 한다.

목적은 둘이다.
1. 3순서(vLLM 커널 이식)에서 막히지 않도록 Triton 어텐션 커널의 구조를 머리에 넣는다.
2. 스펙 7절의 R2·R3·R4·R5를 읽기만으로 확인한다.

예상 시간: 튜토리얼 1~2시간, LightLLM 1~2시간, vLLM 1~2시간. 끝나는 조건은 8절 보고 양식을 채워 어시스턴트에게 보내는 것이다.

## 0. 준비 (5분)

```bash
mkdir -p ~/dev/ref && cd ~/dev/ref
git clone --depth 1 https://github.com/ModelTC/lightllm
git clone --depth 1 --branch v0.30.0 https://github.com/vllm-project/vllm
```

- 태그를 못 찾으면 `git ls-remote --tags https://github.com/vllm-project/vllm | tail -20`으로 가장 가까운 태그를 고른다. 실제로 쓴 태그를 보고서에 적는다.
- `pip install`이나 실행은 하지 않는다.

## B1. Triton 튜토리얼 (1~2시간)

위치: <https://triton-lang.org/main/getting-started/tutorials/index.html> 의 `02-fused-softmax`, `03-matrix-multiplication`. 읽기만 한다(돌리려면 GPU가 필요하다. 3순서에서 서버를 다시 빌릴 때 첫 10분에 돌려 본다).

| 튜토리얼 | 눈여겨볼 것 | 우리 커널과의 연결 |
|---|---|---|
| 02 | `program_id`와 `tl.arange`로 이 프로그램의 담당 구간을 정하는 법 · `tl.load(..., mask=, other=)`로 범위 밖을 처리하는 법 · 최댓값을 빼고 `exp`하는 이유 | 우리가 넣을 `tl.where(valid, qk, -inf)`가 같은 종류의 도구다 |
| 03 | 오프셋으로 포인터를 계산하는 법 · K 방향 누산기 루프 · `tl.dot` | 어텐션의 `QK^T`와 `PV`가 이 구조다 |

스스로 답해 볼 질문: **한 프로그램은 무엇 하나를 계산하는가? 그 프로그램은 자기 담당 위치를 어떻게 아는가?**

## B2. LightLLM `flash_decoding` (1~2시간)

```bash
cd ~/dev/ref
find lightllm -name "*flash_decoding*"
```

3파일이 나온다(래퍼 + stage1 + stage2, 합쳐 274줄). 래퍼 → stage1 → stage2 순서로 읽는다.

읽고 나서 답할 질문:
1. stage1의 프로그램(grid)은 무엇 단위로 나뉘는가?
2. stage1이 만드는 **중간 결과**는 무엇인가? (부분 출력과 부분 softmax 정규화 값)
3. stage2는 그 중간 결과들을 어떻게 하나로 합치는가? (log-sum-exp)
4. 토큰 위치를 물리 메모리 위치로 바꾸는 **간접 테이블**은 어디서 쓰이는가? (계획서 노트: `Req_to_tokens[req, seq]`, 토큰 단위) vLLM은 이것이 블록 단위다. 차이를 한 줄로 적어 둔다.

1~3을 **두 문장**으로 요약한 것이 보고 1번이다.

## B3. vLLM `triton_unified_attention.py` (1~2시간)

```bash
cd ~/dev/ref
F=$(find vllm -name triton_unified_attention.py -path "*ops*" | head -1); echo "$F"
head -20 "$F"                                       # 라이선스 헤더 (복사할 때 유지해야 한다)
grep -n "^import\|^from" "$F"                       # R4
grep -n "physical_block_idx" "$F"                   # 우리가 손댈 지점
grep -n "block_size\|BLOCK_SIZE" "$F" | head -40    # R2
grep -n "seq_len" "$F" | head -40                   # R3
```

여러 파일이 잡히면 `attention/ops` 안의 것을 쓴다. 옛 경로(`vllm/attention/ops/`)를 가리키는 글이 많으니 주의한다.

읽고 나서 답할 질문:
1. `physical_block_idx`를 계산하는 줄은 어디인가? 그 값이 K, V 주소 계산에 쓰이는 곳은?
2. 점수(`qk`)에 마스크를 거는 곳(`tl.where` 등)은 어디인가? **우리의 `live_mask` 로드가 들어갈 자리 후보다.**
3. (R2) Python 래퍼 함수에 `block_size` 제약(assert, 2의 거듭제곱, 최솟값 등)이 있는가? 블록 크기 8이 통과하는가?
4. (R3) `seq_len`이 쓰이는 줄을 분류한다. 루프 상한 / causal 마스크 / 그 외. 우리가 슬롯 수로 바꿔 넘겨도 되는가?
5. (R4) import 중 `vllm` 내부 모듈이 있는가? 있다면 무엇인가? (계획서는 `triton`과 `torch`뿐이라고 주장했다. 미확인)

## B4. torch 핀 (R5, 10분)

```bash
cd ~/dev/ref/vllm
grep -rn "torch" requirements/cuda.txt
grep -rni "triton" requirements/
grep -n "torch" pyproject.toml | head
```

서버는 torch 2.14.1+cu132, Triton 3.8.0이다. 핀이 같은 minor면 낙관해도 되고, 다르면 3순서에서 서버 torch를 vLLM 핀에 맞출지 판단한다.

## 8. 보고 양식 (복사해서 채운다)

```
[B 보고]
vLLM 태그:
1. flash decoding 2단계 (두 문장):
2. physical_block_idx: <파일>:<줄번호>  <그 줄 내용>
   qk 마스크가 걸리는 곳(후보): <줄번호>
3. import 목록:
   vllm 내부 모듈: 있음 / 없음 (있으면 이름)
4. torch 핀:      triton 핀:
(R2) block_size 제약:
(R3) seq_len 사용처:
막힌 곳:
```

R2·R3은 읽다가 보이는 만큼만 적어도 된다. 1~4는 필수다.

## 다음

보고를 받으면 어시스턴트가 R2~R5를 스펙 7절에 기록하고 **C(Mac 코딩)** 를 시작하게 한다.
