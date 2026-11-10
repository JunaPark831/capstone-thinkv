"""합성 ThinKV 타임라인 (스펙 4절). 커널 벤치마크와 V5에 같은 축출 순서를 준다.

규칙은 e4_tbe_tbq_design.md 1절과 E4 체크리스트의 plan_step을 따른다.
- 프롬프트 P토큰(구간 0, 유형 R로 둔다)은 축출하지 않는다. 예산 k에는 포함한다.
- 그 뒤 tau토큰짜리 구간이 이어지고, 구간 m(1부터)의 유형은 pattern[(m-1) % len(pattern)].
- 경우 1: T 구간이 닫히면 그보다 앞선 구간 전부(프롬프트 제외)를 한 단계 줄인다.
- 경우 2: 살아 있는 토큰 수가 예산을 넘는 동안, 닫힌 구간 중 T→E→R 순으로 가장 오래된 것을 한 단계 줄인다.
  모두 최소(retain[-1])면 멈춘다.
- 한 단계 = 지금 길이보다 작은 retain 값 중 가장 큰 것으로. 열린 구간은 건드리지 않는다.
- 어느 토큰을 남기는지는 커널 성능과 무관하므로 구간 안에서 무작위(시드 고정). 실제 TBE는 K-means다.
"""
import random
from dataclasses import dataclass

from kernels.block_table import E, R, T

RETAIN = (64, 32, 16, 8, 4)
TYPES = {"T": T, "E": E, "R": R}


@dataclass
class Step:
    token: int          # 논리 토큰 인덱스 (프롬프트 다음부터)
    start: int          # 이 토큰이 속한 구간의 시작 (BlockTable.append 인자)
    type: int
    evict: list[int]    # 이 토큰을 쓴 뒤 축출할 토큰 인덱스들


def timeline(prompt_len, n_decode, pattern="RETETRETE", budget=1024, tau=128, retain=RETAIN, seed=0):
    rng = random.Random(seed)
    segs = []                                   # [start, type, 살아 있는 토큰 리스트, 닫힘]
    total = prompt_len
    steps = []

    def shrink(seg):
        nonlocal total
        smaller = [r for r in retain if r < len(seg[2])]
        if not smaller:
            return []
        keep = set(rng.sample(seg[2], smaller[0]))
        gone = [x for x in seg[2] if x not in keep]
        seg[2] = [x for x in seg[2] if x in keep]
        total -= len(gone)
        return gone

    for i in range(n_decode):
        tok = prompt_len + i
        evict = []
        if i % tau == 0:
            if segs:
                segs[-1][3] = True
                if segs[-1][1] == T:                                  # 경우 1
                    for s in segs[:-1]:
                        evict += shrink(s)
            segs.append([tok, TYPES[pattern[(i // tau) % len(pattern)]], [], False])
        segs[-1][2].append(tok)
        total += 1
        while total > budget:                                         # 경우 2
            cand = [s for t in (T, E, R) for s in segs if s[3] and s[1] == t and len(s[2]) > retain[-1]]
            if not cand:
                break
            evict += shrink(cand[0])
        steps.append(Step(tok, segs[-1][0], segs[-1][1], evict))
    return steps


def drive(bt, sid, prompt_len, steps, on_write=None):
    """타임라인을 BlockTable(CT 방식)에 적용한다. on_write(token, slot)으로 K·V를 쓸 자리를 받는다.
    반환: 토큰 -> 슬롯 (살아 있는 것만)."""
    slot_of = {}
    for p in range(prompt_len):
        slot_of[p] = bt.append(sid, 0, R)
        if on_write:
            on_write(p, slot_of[p])
    for st in steps:
        slot_of[st.token] = bt.append(sid, st.start, st.type)
        if on_write:
            on_write(st.token, slot_of[st.token])
        if st.evict:
            bt.evict(sid, [slot_of.pop(x) for x in st.evict])
    return slot_of
