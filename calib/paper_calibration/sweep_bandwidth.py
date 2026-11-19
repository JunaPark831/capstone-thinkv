"""KDE 폭(Scott 배수)을 바꿔 가며 봉우리 수와 Algorithm 1 결과를 다시 센다. 사전 기록: docs/e3_paper_calibration_spec.md "KDE 폭 스윕 사전 기록".

사용: python -I sweep_bandwidth.py --dir DATA_DIR --out RESULT.json
조합: 변형 4 x 워밍업 {0, 1100} x 폭 {0.5, 1, 2, 3, 5, 8}. 판정 규칙(정확히 3봉, 교집합)과 그 밖의 설정은 calibrate_paper.py 와 같다.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks, peak_prominences

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_paper import XS, kde_density  # noqa: E402

VARIANTS = ["full_nosink", "win1024", "win256", "full_literal"]
MULTS = [0.5, 1, 2, 3, 5, 8]
SKIPS = [0, 1100]


def one(job):
    dirpath, variant, skip = job
    series = []
    for f in sorted(Path(dirpath).glob("*.pt")):
        r = torch.load(f, weights_only=True)   # 원격에서 받은 파일이라 pickle 실행을 막는다
        series.append(r["sparsity"][r["variants"].index(variant)].float().numpy())
    P, L = len(series), series[0].shape[0]
    out = {}
    for m in MULTS:
        n = np.zeros((P, L), int)
        three, valleys = [], {}
        for p, arr in enumerate(series):
            for l in range(L):
                d = kde_density(arr[l, skip:].astype(np.float64), m)
                if d is None:
                    continue
                pk = find_peaks(d)[0]
                n[p, l] = len(pk)
                if len(pk) == 3:
                    prom = float((peak_prominences(d, pk)[0] / d.max()).min())
                    va = [float(XS[a + int(np.argmin(d[a:b + 1]))]) for a, b in zip(pk[:-1], pk[1:])]
                    three.append(prom)
                    valleys[(p, l)] = va
        lstar = sorted(set.intersection(*[set(np.where(n[p] == 3)[0].tolist()) for p in range(P)]))
        theta = [float(np.mean([valleys[(p, l)][j] for l in lstar for p in range(P)])) for j in range(2)] if lstar else None
        out[str(m)] = {
            "L_star": lstar, "theta": theta,
            "max_prompts_exactly3": int(max((n[:, l] == 3).sum() for l in range(L))),
            "mode_share": {"1": float((n == 1).mean()), "2": float((n == 2).mean()), "3": float((n == 3).mean()), "4+": float((n >= 4).mean())},
            "median_modes": float(np.median(n)),
            "n_exactly3": len(three), "prom_ge_0.1": int(sum(t >= 0.1 for t in three)), "prom_ge_0.05": int(sum(t >= 0.05 for t in three)),
            "same_k_ge2_layers": {int(l): int(n[0, l]) for l in range(L) if len(set(n[:, l])) == 1 and n[0, l] >= 2},
            "n_modes": n.tolist(),
        }
    return variant, skip, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs = [(a.dir, v, s) for v in VARIANTS for s in SKIPS]
    with ProcessPoolExecutor(max_workers=min(8, len(jobs))) as ex:
        results = list(ex.map(one, jobs))
    res = {f"{v}|skip{s}": o for v, s, o in results}
    Path(a.out).write_text(json.dumps(res, indent=1))
    P = len(list(Path(a.dir).glob("*.pt")))
    for v, s, o in results:
        print(f"\n== {v} | 앞 {s}스텝 제외 | P={P}")
        print(" 폭   |L*| 최대문제수/P  봉우리수 비율 1/2/3/4+ (%)   중앙값  정확히3봉(prom>=0.1)  전문제 동일 k>=2 레이어")
        for m in MULTS:
            r = o[str(m)]
            ms = r["mode_share"]
            print(f"x{m:<4} {len(r['L_star']):3d}   {r['max_prompts_exactly3']}/{P}          "
                  f"{ms['1'] * 100:4.0f}/{ms['2'] * 100:3.0f}/{ms['3'] * 100:3.0f}/{ms['4+'] * 100:3.0f}      {r['median_modes']:4.1f}   "
                  f"{r['n_exactly3']:3d} ({r['prom_ge_0.1']:3d})            {r['same_k_ge2_layers'] or '-'}")
            if r["L_star"]:
                print(f"      ** L*={r['L_star']} theta={r['theta']}")
    print("\nsaved", a.out)


if __name__ == "__main__":
    main()
