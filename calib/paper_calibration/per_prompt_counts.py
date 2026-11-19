"""채널마다 문제별·레이어별 봉우리 수(필터 없음 / 봉우리다움 L2)를 센다. 합치지 않는다.
사용: python -I per_prompt_counts.py --dir DATA_DIR --out RESULT.json
"""
import argparse, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, torch
from scipy.signal import find_peaks
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_paper import kde_density
from peaklike import peaklike, peaks_by_mass
from calibrate_paper import XS
L2 = (0.05, 0.05, 0.30)   # 기본. --prom/--dist/--dip 로 바꾼다

def one(job):
    dirpath, ch, bw, flt = job
    files = sorted(Path(dirpath).glob("*.pt"))
    raw, l2, pos = [], [], []
    for f in files:
        r = torch.load(f, weights_only=True)
        arr = r["sparsity"][r["variants"].index(ch)].float().numpy()
        a, b, c = [], [], []
        for l in range(arr.shape[0]):
            d = kde_density(arr[l].astype(np.float64), bw)
            if d is None:
                a.append(0); b.append(0); c.append([]); continue
            a.append(len(find_peaks(d / d.max())[0]))
            pk = peaks_by_mass(d, flt[0], flt[1]) if flt[2] == 'mass' else peaklike(d, *flt)
            b.append(len(pk)); c.append([round(float(XS[i]), 3) for i in pk] + [round(float(d[i] / d.max()), 3) for i in pk])
        raw.append(a); l2.append(b); pos.append(c)
    return ch, np.array(raw), np.array(l2), pos

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True); ap.add_argument("--bw", type=float, default=1.0); ap.add_argument("--prom", type=float, default=0.05); ap.add_argument("--dist", type=float, default=0.05); ap.add_argument("--dip", type=float, default=0.30); ap.add_argument("--mass", type=float, default=None, help="질량 기준 모드: 구역 질량 하한(예 0.02)"); a = ap.parse_args()
    flt = (a.mass, a.dist, "mass") if a.mass is not None else (a.prom, a.dist, a.dip)
    files = sorted(Path(a.dir).glob("*.pt")); names = [f.stem for f in files]
    chans = [c for c in torch.load(files[0], weights_only=True)["variants"] if c != "sink_mass"]
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(one, [(a.dir, c, a.bw, flt) for c in chans]))
    out = {}
    print(f"KDE Scott x{a.bw} | 필터 {flt} | 문제 {names}\n각 채널: [필터 없음] 문제별 정확히3봉 레이어 수 | L* ; [L2] 문제별 ≥2봉 레이어 수 / 정확히3봉 레이어 수 | L*")
    for ch, raw, l2, pos in res:
        P, L = raw.shape
        ls_raw = [l for l in range(L) if (raw[:, l] == 3).all()]
        ls_l2 = [l for l in range(L) if (l2[:, l] == 3).all()]
        r3 = [int((raw[p] == 3).sum()) for p in range(P)]
        g2 = [int((l2[p] >= 2).sum()) for p in range(P)]
        e3 = [int((l2[p] == 3).sum()) for p in range(P)]
        out[ch] = {"raw_exactly3_per_prompt": r3, "raw_L_star": ls_raw, "L2_ge2_per_prompt": g2, "L2_exactly3_per_prompt": e3, "L2_L_star": ls_l2,
                   "raw_median_modes": float(np.median(raw)), "L2_share_1": float((l2 == 1).mean()),
                   "L2_ge2_cells": [(names[p], int(l), int(l2[p, l])) for p in range(P) for l in range(L) if l2[p, l] >= 2],
                   "exactly3_detail": [(names[p], int(l), pos[p][l][:3], pos[p][l][3:]) for p in range(P) for l in range(L) if l2[p, l] == 3]}
        print(f"{ch:18s} [없음] {r3} | L*={ls_raw or '∅'} (중앙값 {np.median(raw):.0f}봉)   [L2] ≥2봉 {g2} / 정확히3 {e3} | L*={ls_l2 or '∅'} (1봉 {(l2 == 1).mean() * 100:.0f}%)")
    Path(a.out).write_text(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
