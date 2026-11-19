"""replay_sparsity.py 결과를 docs/e3_paper_exact_criteria.md §5-6 규칙으로 분석한다.

사용: python analyze_paper_exact.py <replay.pt> <out_dir>
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

XS = np.linspace(0, 1, 512)
MULTS = [0.5, 1, 2]


def kde_curve(x, mult):
    try:
        f = gaussian_kde(x).factor
        d = gaussian_kde(x, bw_method=f * mult)(XS)
    except (np.linalg.LinAlgError, ValueError):  # 분산 0 등
        return None
    return d / d.max()


def modes(x, mult):
    d = kde_curve(x, mult)
    if d is None:
        return 0, 0, None
    return len(find_peaks(d)[0]), len(find_peaks(d, prominence=0.05)[0]), d


def main(src, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    r = torch.load(src, weights_only=False)
    names, sp = r["variants"], r["sparsity"].float().numpy()   # (V, L, T)
    V, L, T = sp.shape
    res = {"src": src, "steps": T, "variants": {}}
    curves = {}
    for v, name in enumerate(names):
        per_layer = []
        for l in range(L):
            x = sp[v, l]
            row = {"layer": l, "mean": float(x.mean()), "frac_gt_0.98": float((x > 0.98).mean())}
            for m in MULTS:
                raw, prom, d = modes(x, m)
                row[f"modes_raw_x{m}"], row[f"modes_prom0.05_x{m}"] = raw, prom
                curves[(name, l, m)] = d
            per_layer.append(row)
        for kind in ("raw", "prom0.05"):
            key = f"modes_{kind}"
            ok = [p["layer"] for p in per_layer if all(p[f"{key}_x{m}"] == 3 for m in MULTS)]
            res["variants"].setdefault(name, {})[f"layers_exactly3_all_bw_{kind}"] = ok
        res["variants"][name]["per_layer"] = per_layer
        res["variants"][name]["frac_gt_0.98_overall"] = float((sp[v] > 0.98).mean())
    (out_dir / "result.json").write_text(json.dumps(res, indent=1))

    print(f"steps={T}")
    for name in names:
        vv = res["variants"][name]
        print(f"\n== {name}  frac>0.98 overall {vv['frac_gt_0.98_overall']:.3f}")
        print(" exactly-3 at all 3 bw  raw:", vv["layers_exactly3_all_bw_raw"], " prom0.05:", vv["layers_exactly3_all_bw_prom0.05"])
        print(" layer: modes_raw(x0.5,x1,x2) | modes_prom(x0.5,x1,x2) | mean | frac>.98")
        for p in vv["per_layer"]:
            a = tuple(p[f"modes_raw_x{m}"] for m in MULTS)
            b = tuple(p[f"modes_prom0.05_x{m}"] for m in MULTS)
            print(f"  L{p['layer']:02d}: {a} | {b} | {p['mean']:.3f} | {p['frac_gt_0.98']:.3f}")

    # 그림 1: 레이어 10 시계열 (128스텝 이동평균), 변형 4개
    k = np.ones(128) / 128
    fig, ax = plt.subplots(V, 1, figsize=(14, 2.6 * V), sharex=True)
    for v, name in enumerate(names):
        x = sp[v, 10]
        ax[v].plot(x, lw=0.3, alpha=0.4)
        ax[v].plot(np.arange(127, T), np.convolve(x, k, "valid"), lw=1.2)
        ax[v].set_ylim(0, 1)
        ax[v].set_ylabel(name, fontsize=8)
    ax[0].set_title("layer 10 sparsity over decode steps (raw + MA128)")
    fig.tight_layout()
    fig.savefig(out_dir / "layer10_timeseries.png", dpi=110)

    # 그림 2: paper 변형 32층 KDE (Scott x1)
    pi = names.index("paper")
    fig, ax = plt.subplots(4, 8, figsize=(24, 10))
    for l in range(L):
        a = ax[l // 8, l % 8]
        d = curves[("paper", l, 1)]
        if d is not None:
            a.plot(XS, d)
        a.set_title(f"L{l} raw modes={res['variants']['paper']['per_layer'][l]['modes_raw_x1']}", fontsize=8)
    fig.suptitle("paper variant: per-layer KDE (Scott x1)")
    fig.tight_layout()
    fig.savefig(out_dir / "paper_kde_scott_x1.png", dpi=90)
    print("\nsaved to", out_dir)


if __name__ == "__main__":
    main(*sys.argv[1:])
