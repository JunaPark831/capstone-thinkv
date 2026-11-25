"""'봉우리다운' 봉우리만 남기는 필터 (논문에 없는 우리의 정의). 사전 기록: docs/criteria/e3_paper_calibration_spec.md "봉우리다운 것만".

peaklike(d, prom, dist, dip): d = KDE 밀도(512점 격자, 0~1), 반환 = 남은 봉우리의 격자 인덱스.
  prom: 전체 최댓값 대비 prominence 하한
  dist: 이웃한 봉우리 사이 최소 거리(희소도 단위). 더 가까우면 낮은 쪽을 버린다
  dip : 이웃한 두 봉우리 사이 골이 낮은 쪽 봉우리보다 이 비율 이상 낮아야 한다. 못 지키는 쌍은 낮은 쪽을 버리고 반복
"""
import numpy as np
from scipy.signal import find_peaks

from calibrate_paper import XS


def peaklike(d, prom=0.0, dist=0.0, dip=0.0):
    dn = d / d.max()
    kw = {}
    if prom > 0:
        kw["prominence"] = prom
    if dist > 0:
        kw["distance"] = max(1, int(round(dist * (len(XS) - 1))))
    pk = [int(i) for i in find_peaks(dn, **kw)[0]]
    while dip > 0 and len(pk) >= 2:
        dips = [1 - dn[a:b + 1].min() / min(dn[a], dn[b]) for a, b in zip(pk[:-1], pk[1:])]
        k = int(np.argmin(dips))
        if dips[k] >= dip:
            break
        a, b = pk[k], pk[k + 1]
        pk.pop(k if dn[a] < dn[b] else k + 1)       # 낮은 쪽 봉우리 제거
    return np.array(pk, int)


def valleys_between(d, pk):
    return [float(XS[a + int(np.argmin(d[a:b + 1]))]) for a, b in zip(pk[:-1], pk[1:])]


def peaks_by_mass(d, min_mass=0.02, dist=0.05):
    """높이 조건 없이: 거리 >= dist 인 국소 최댓값 중, 구역(이웃 골 사이) KDE 질량이 전체의 min_mass 미만인 봉우리를 제거하며 반복."""
    kw = {"distance": max(1, int(round(dist * (len(XS) - 1))))} if dist > 0 else {}
    pk = [int(i) for i in find_peaks(d / d.max(), **kw)[0]]
    total = d.sum()
    while len(pk) >= 2:
        bounds = [0] + [a + int(np.argmin(d[a:b + 1])) for a, b in zip(pk[:-1], pk[1:])] + [len(d)]
        mass = [d[bounds[k]:bounds[k + 1]].sum() / total for k in range(len(pk))]
        k = int(np.argmin(mass))
        if mass[k] >= min_mass:
            break
        pk.pop(k)
    return np.array(pk, int)
