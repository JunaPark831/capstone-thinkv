"""문제마다, 레이어마다 따로: (문제, 레이어) 조합 하나당 패널 하나로 히스토그램 + KDE + 봉우리를 그린다.

사용: python -I plot_layer_panels.py --dir DATA_DIR --out OUT_DIR [--variants win1024,full_nosink,win256] [--bw-mult 1.5] [--prom 0.05] [--skip 0]
산출: OUT_DIR/{변형}/prompt_{문제}.png (문제당 레이어 32개 패널), OUT_DIR/{변형}/layer_overlay.png (레이어마다 문제 9개 KDE를 겹침, 합치지 않음), OUT_DIR/index.html
봉우리 규칙: find_peaks(d/d.max(), prominence=prom) 로 남은 것이 주황 큰 점, 필터에서 버려진 요철은 회색 작은 점. 골은 남은 봉우리 사이 KDE 최솟값.
"""
import argparse
import html
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_paper import XS, kde_density  # noqa: E402
from peaklike import peaklike  # noqa: E402

NAMES = ["Execution", "Reasoning", "Transition"]


def analyze(x, bw, prom, dist=0.0, dip=0.0):
    d = kde_density(x, bw)
    if d is None:
        return None, [], [], []
    dn = d / d.max()
    raw = find_peaks(dn)[0]
    kept = peaklike(d, prom, dist, dip)
    va = [float(XS[a + int(np.argmin(d[a:b + 1]))]) for a, b in zip(kept[:-1], kept[1:])]
    return d, raw, kept, va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="win1024,full_nosink,win256")
    ap.add_argument("--bw-mult", type=float, default=1.5)
    ap.add_argument("--prom", type=float, default=0.05)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--dist", type=float, default=0.0, help="이웃한 봉우리 최소 거리(희소도 단위)")
    ap.add_argument("--dip", type=float, default=0.0, help="이웃한 봉우리 사이 골이 낮은 쪽 봉우리보다 이 비율 이상 낮아야 함")
    a = ap.parse_args()
    out = Path(a.out)
    files = sorted(Path(a.dir).glob("*.pt"))
    data = [torch.load(f, weights_only=True) for f in files]   # 원격에서 받은 파일이라 pickle 실행을 막는다
    names = [f.stem for f in files]
    L = data[0]["sparsity"].shape[1]
    sections = []
    for variant in a.variants.split(","):
        vi = data[0]["variants"].index(variant)
        vdir = out / variant
        vdir.mkdir(parents=True, exist_ok=True)
        series = [r["sparsity"][vi].float().numpy() for r in data]
        summary = []
        for p, (nm, arr) in enumerate(zip(names, series)):
            fig, ax = plt.subplots(4, 8, figsize=(24, 10.5))
            kept_counts = []
            for l in range(L):
                g = ax[l // 8, l % 8]
                x = arr[l, a.skip:].astype(np.float64)
                d, raw, kept, va = analyze(x, a.bw_mult, a.prom, a.dist, a.dip)
                g.hist(x, bins=60, range=(0, 1), density=True, alpha=0.4, color="#7aa5ff")
                if d is not None:
                    g.plot(XS, d, lw=1.2, color="#1f3f9f")
                    for i in raw:
                        if i not in kept:
                            g.plot([XS[i]], [d[i]], "o", color="#999", ms=3)
                    for i in kept:
                        g.plot([XS[i]], [d[i]], "o", color="#d9630a", ms=6)
                    for v in va:
                        g.axvline(v, color="#1f7a4d", ls="--", lw=1)
                n = len(kept)
                kept_counts.append(n)
                g.set_xlim(0, 1)
                g.set_title(f"L{l}  봉우리 {n}개 (요철 포함 {len(raw)}개)", fontsize=8, color="#c00000" if n >= 2 else "black")
                g.tick_params(labelsize=6)
                if n >= 2:
                    for sp in g.spines.values():
                        sp.set_edgecolor("#d9630a")
                        sp.set_linewidth(2.2 if n >= 3 else 1.6)
            fig.suptitle(f"{variant} | 문제 {nm} (스텝 {arr.shape[1]}) | KDE Scott x{a.bw_mult} | 주황 큰 점=봉우리다운 봉우리(prominence>={a.prom}, 거리>={a.dist}, 골 깊이>={a.dip}), "
                         f"회색 작은 점=걸러진 요철, 초록 점선=골 | 봉우리 2개 이상인 레이어는 주황 테두리 | x축 0~1", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.965))
            fig.savefig(vdir / f"prompt_{nm}.png", dpi=62)
            plt.close(fig)
            summary.append((nm, arr.shape[1], kept_counts))

        # 레이어마다 문제 9개를 겹쳐 그림 (합치지 않고 문제별 KDE 곡선)
        fig, ax = plt.subplots(4, 8, figsize=(24, 10.5))
        cmap = plt.get_cmap("tab10")
        for l in range(L):
            g = ax[l // 8, l % 8]
            for p, (nm, arr) in enumerate(zip(names, series)):
                d, raw, kept, va = analyze(arr[l, a.skip:].astype(np.float64), a.bw_mult, a.prom, a.dist, a.dip)
                if d is not None:
                    g.plot(XS, d / d.max(), lw=1, color=cmap(p % 10), alpha=0.85, label=nm)
                    for i in kept:
                        g.plot([XS[i]], [d[i] / d.max()], "o", color=cmap(p % 10), ms=4)
            g.set_xlim(0, 1)
            g.set_ylim(0, 1.05)
            g.set_title(f"L{l}", fontsize=9)
            g.tick_params(labelsize=6)
            if l == 0:
                g.legend(fontsize=6, ncol=2)
        fig.suptitle(f"{variant} | 레이어마다 문제 {len(names)}개의 KDE를 겹침(합치지 않음, 각 곡선은 꼭대기=1로 정규화) | Scott x{a.bw_mult} | 점=봉우리다운 봉우리(prominence>={a.prom}, 거리>={a.dist}, 골 깊이>={a.dip})", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        fig.savefig(vdir / "layer_overlay.png", dpi=62)
        plt.close(fig)
        sections.append((variant, summary))
        multi = sum(1 for _, _, c in summary for n in c if n >= 2)
        print(f"{variant}: (문제, 레이어) {len(names) * L}개 중 봉우리 2개 이상 {multi}개, 3개 이상 {sum(1 for _, _, c in summary for n in c if n >= 3)}개")

    body = ["<!doctype html><meta charset='utf-8'><title>레이어별·문제별 KDE</title>",
            "<style>body{font:15px/1.6 -apple-system,'Apple SD Gothic Neo',sans-serif;margin:24px;max-width:1700px}img{width:100%;border:1px solid #ccc;margin:6px 0 22px}h2{margin-top:40px}</style>",
            f"<h1>문제마다·레이어마다 KDE (Scott x{a.bw_mult}, prominence >= {a.prom})</h1>",
            "<p>패널 하나 = (문제 1개, 레이어 1개). 주황 큰 점 = 필터를 통과한 봉우리, 회색 작은 점 = 걸러진 요철. 봉우리가 2개 이상인 레이어는 주황 테두리입니다.</p>"]
    for variant, summary in sections:
        body.append(f"<h2>{html.escape(variant)}</h2><h3>레이어마다 문제를 겹친 그림</h3><img src='{variant}/layer_overlay.png'>")
        for nm, steps, c in summary:
            body.append(f"<h3>문제 {nm} (스텝 {steps}) — 봉우리 2개 이상인 레이어: {[l for l, n in enumerate(c) if n >= 2] or '없음'}</h3>"
                        f"<img loading='lazy' src='{variant}/prompt_{nm}.png'>")
    (out / "index.html").write_text("\n".join(body), encoding="utf-8")
    print("saved to", out)


if __name__ == "__main__":
    main()
