"""정확히 3봉으로 세어진 (문제, 레이어) 시계열을 뽑아 그림·표로 만든다 (calibrate_paper.py 와 같은 KDE·봉우리 규칙).

사용: python -I plot_three_modes.py --dir DATA_DIR --out OUT_DIR [--skip 0]
산출: three_modes_{변형}_skip{N}.png (3봉인 시계열 전부), mode_count_heatmap.png (문제×레이어 봉우리 수),
      three_modes_list.csv (3봉 목록: 봉우리·골 위치, 가장 약한 봉우리의 prominence)
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "DejaVu Sans"]  # 한글 제목 (macOS 기본 폰트)
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from scipy.signal import find_peaks, peak_prominences

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_paper import XS, kde_density  # noqa: E402  (같은 KDE 설정을 재사용)

VARIANTS = ["full_nosink", "win1024", "win256", "full_literal"]


def analyze(x):
    """x: 시계열 -> (density, 봉우리 인덱스, 골 위치, 가장 약한 봉우리의 상대 prominence)."""
    d = kde_density(x)
    if d is None:
        return None, [], [], None
    pk = find_peaks(d)[0]
    valleys = [float(XS[a + int(np.argmin(d[a:b + 1]))]) for a, b in zip(pk[:-1], pk[1:])]
    prom = float((peak_prominences(d, pk)[0] / d.max()).min()) if len(pk) else None
    return d, pk, valleys, prom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip", type=int, default=0, help="그림에 쓸 워밍업(앞 스텝 제외 수). 표는 0과 1100 둘 다 만든다")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(a.dir).glob("*.pt"))
    data = [torch.load(f, weights_only=True) for f in files]   # 원격에서 받은 파일이라 pickle 실행을 막는다
    names = [f.stem for f in files]
    P, L = len(data), data[0]["sparsity"].shape[1]

    rows, counts = [], {}
    found = {}  # (variant, skip) -> [(p, l, x, d, pk, valleys, prom)]
    for v in VARIANTS:
        vi = data[0]["variants"].index(v)
        for skip in sorted({0, 1100, a.skip}):
            cnt = np.zeros((P, L), int)
            for p, r in enumerate(data):
                arr = r["sparsity"][vi].float().numpy()
                for l in range(L):
                    x = arr[l, skip:].astype(np.float64)
                    d, pk, valleys, prom = analyze(x)
                    cnt[p, l] = len(pk)
                    if len(pk) == 3:
                        found.setdefault((v, skip), []).append((p, l, x, d, pk, valleys, prom))
                        rows.append({"variant": v, "skip": skip, "prompt": names[p], "layer": l, "steps_used": len(x),
                                     "mode1": round(float(XS[pk[0]]), 4), "mode2": round(float(XS[pk[1]]), 4),
                                     "mode3": round(float(XS[pk[2]]), 4), "valley1": round(valleys[0], 4),
                                     "valley2": round(valleys[1], 4), "min_rel_prominence": round(prom, 4)})
            counts[(v, skip)] = cnt

    with open(out / "three_modes_list.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # 그림 1: 3봉으로 세어진 시계열 전부 (변형별 한 장, 지정한 워밍업)
    for v in VARIANTS:
        items = sorted(found.get((v, a.skip), []), key=lambda t: (t[1], t[0]))
        if not items:
            continue
        ncol = 6
        nrow = -(-len(items) // ncol)
        fig, ax = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.7 * nrow), squeeze=False)
        for k, (p, l, x, d, pk, valleys, prom) in enumerate(items):
            g = ax[k // ncol, k % ncol]
            g.hist(x, bins=60, range=(0, 1), density=True, alpha=0.45, color="#7aa5ff")
            g.plot(XS, d, lw=1.2, color="#1f3f9f")
            g.plot(XS[pk], d[pk], "o", color="#d9630a", ms=4)
            for vv in valleys:
                g.axvline(vv, color="#1f7a4d", ls="--", lw=1)
            g.set_xlim(0, 1)
            g.set_title(f"{names[p]}  L{l}  n={len(x)}\n가장 약한 봉 prominence={prom:.2f}", fontsize=8)
            g.tick_params(labelsize=6)
        for k in range(len(items), nrow * ncol):
            ax[k // ncol, k % ncol].axis("off")
        fig.suptitle(f"{v} | 워밍업 앞 {a.skip}스텝 제외 | 정확히 3봉으로 세어진 (문제, 레이어) {len(items)}개 / {P * L}개 "
                     f"(파랑 막대=히스토그램, 선=KDE, 주황점=봉우리, 초록 점선=골, x축 0~1 고정)", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out / f"three_modes_{v}_skip{a.skip}.png", dpi=75)
        plt.close(fig)

    # 그림 2: 문제 x 레이어 봉우리 수 (3인 칸을 굵은 테두리로 표시)
    skips = sorted({0, 1100})
    fig, ax = plt.subplots(len(VARIANTS), len(skips), figsize=(15 * len(skips) / 1.2, 3.1 * len(VARIANTS)), squeeze=False)
    for i, v in enumerate(VARIANTS):
        for j, skip in enumerate(skips):
            g = ax[i, j]
            c = counts[(v, skip)]
            g.imshow(np.minimum(c, 12), cmap="YlGnBu", vmin=0, vmax=12, aspect="auto")
            for p in range(P):
                for l in range(L):
                    g.text(l, p, str(c[p, l]), ha="center", va="center", fontsize=6, color="black" if c[p, l] < 9 else "white")
                    if c[p, l] == 3:
                        g.add_patch(Rectangle((l - 0.5, p - 0.5), 1, 1, fill=False, ec="#d9630a", lw=1.8))
            g.set_xticks(range(0, L, 2))
            g.set_xticklabels(range(0, L, 2), fontsize=6)
            g.set_yticks(range(P))
            g.set_yticklabels(names, fontsize=6)
            n3 = int((c == 3).sum())
            full = [l for l in range(L) if (c[:, l] == 3).all()]
            g.set_title(f"{v} | 앞 {skip}스텝 제외 | 3봉 칸 {n3}/{P * L}, 모든 문제에서 3봉인 레이어 {full or '없음'}", fontsize=9)
    fig.suptitle("봉우리 수 (행=문제, 열=레이어). 주황 테두리=정확히 3봉. 한 열이 전부 테두리여야 L*에 든다", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "mode_count_heatmap.png", dpi=75)
    plt.close(fig)

    for (v, skip), items in sorted(found.items()):
        print(f"{v:13s} skip={skip:<4d} 3봉 {len(items):3d}개, 가장 약한 봉 prominence 중앙값 {np.median([t[6] for t in items]):.3f}, "
              f"0.1 이상인 것 {sum(t[6] >= 0.1 for t in items)}개")
    print("saved to", out)


if __name__ == "__main__":
    main()
