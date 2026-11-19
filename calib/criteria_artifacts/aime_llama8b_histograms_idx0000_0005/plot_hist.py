"""희소도 분포(히스토그램 + KDE ×1)를 직접 그리고, '경계가 보이는 정도'를 prominence 단계별로 센다.
사용: python plot_hist.py <out_dir>   (스텝 >= 1100, 입력은 scratchpad의 denom_0005.pt, denom_0000.pt)
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

SP = Path("/private/tmp/claude-501/-Users-juna-dev-thinkv-kvpress/d25dc06d-5491-466d-99ed-8d51d68034cc/scratchpad")
XS = np.linspace(0, 1, 512)
MIN = 1100
VARS = ["paper", "paper_nosink", "paper_win256", "paper_win1024"]
PROMS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]   # 0.0 = 제약 없음(raw)


def kde(x):
    return gaussian_kde(x)(XS)               # Scott ×1


def npeaks(d, prom):
    dn = d / d.max()
    return len(find_peaks(dn, prominence=prom)[0]) if prom > 0 else len(find_peaks(dn)[0])


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    data = {k: torch.load(SP / f"denom_{k}.pt", weights_only=False) for k in ("0005", "0000")}

    # 그림 A: 레이어 10, 변형 4개 × 궤적 2개
    fig, ax = plt.subplots(4, 2, figsize=(12, 11))
    for j, (k, r) in enumerate(data.items()):
        names, sp = r["variants"], r["sparsity"].float().numpy()
        for i, v in enumerate(VARS):
            x = sp[names.index(v), 10, MIN:]
            a = ax[i, j]
            a.hist(x, bins=80, range=(0, 1), density=True, alpha=0.5)
            d = kde(x)
            a.plot(XS, d, lw=1.3)
            a.set_title(f"traj {k} | {v} | layer10 | prom0.05 peaks={npeaks(d, 0.05)} raw={npeaks(d, 0)}", fontsize=8)
            a.set_xlim(0, 1)
    fig.suptitle(f"sparsity distribution (steps>={MIN}): histogram + KDE(Scott x1)")
    fig.tight_layout()
    fig.savefig(out / "hist_layer10.png", dpi=100)
    plt.close(fig)

    # 그림 B: 32층 격자 (변형 2개, 궤적 0000)
    r = data["0000"]
    names, sp = r["variants"], r["sparsity"].float().numpy()
    for v in ("paper_win256", "paper_win1024"):
        fig, ax = plt.subplots(4, 8, figsize=(24, 10))
        for l in range(32):
            x = sp[names.index(v), l, MIN:]
            a = ax[l // 8, l % 8]
            a.hist(x, bins=60, range=(0, 1), density=True, alpha=0.5)
            d = kde(x)
            a.plot(XS, d, lw=1)
            a.set_title(f"L{l} p05={npeaks(d, 0.05)} raw={npeaks(d, 0)}", fontsize=8)
            a.set_xlim(0, 1)
        fig.suptitle(f"traj 0000 | {v} | per-layer histogram + KDE (steps>={MIN})")
        fig.tight_layout()
        fig.savefig(out / f"hist_grid_{v}.png", dpi=80)
        plt.close(fig)

    # 표: 경계 깊이(prominence)별 '봉우리 ≥2, ≥3' 레이어 수
    res = {}
    print("prominence 단계별로, 봉우리가 ≥2 / ≥3개인 레이어 수 (32층 중), Scott ×1, steps>=%d" % MIN)
    print(f"{'traj':5s} {'variant':14s} " + " ".join(f"p>={p:<4}" for p in PROMS))
    for k, r in data.items():
        names, sp = r["variants"], r["sparsity"].float().numpy()
        for v in VARS:
            row = []
            for p in PROMS:
                c = [npeaks(kde(sp[names.index(v), l, MIN:]), p) for l in range(32)]
                row.append((sum(n >= 2 for n in c), sum(n >= 3 for n in c)))
            res[f"{k}/{v}"] = {str(p): {"ge2": a, "ge3": b} for p, (a, b) in zip(PROMS, row)}
            print(f"{k:5s} {v:14s} " + " ".join(f"{a:2d}/{b:<2d}  " for a, b in row))
    print("(각 칸: ≥2봉 레이어 수 / ≥3봉 레이어 수)")
    (out / "separation_by_prominence.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
