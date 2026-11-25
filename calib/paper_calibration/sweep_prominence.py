"""prominence 필터를 건 find_peaks 로 봉우리를 다시 센다. 사전 기록: docs/criteria/e3_paper_calibration_spec.md "prominence 필터 적용 재측정 사전 기록".

사용: python -I sweep_prominence.py --dir DATA_DIR --out OUT_DIR [--prom-main 0.05]
대상 1: Algorithm 1 (변형 4 x 워밍업 {0,1100} x prominence 단계), 대상 2: 분위수 방식의 φ̄ (quantile_calibration.py 와 같은 정의).
KDE 는 가우시안 Scott x1, 512점 격자(calibrate_paper.py 와 같음). 봉우리 = find_peaks(d/d.max(), prominence=p).
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_paper import XS, kde_density  # noqa: E402
from quantile_calibration import ma, top_layers  # noqa: E402

VARIANTS = ["full_nosink", "win1024", "win256", "full_literal"]
SKIPS = [0, 1100]
PROMS = [0.0, 0.02, 0.05, 0.1, 0.2]       # 0 = 기존과 같은지 확인하는 점검용


def peaks_valleys(d, prom):
    """필터를 통과한 봉우리 위치(격자 인덱스)와, 이웃한 봉우리 사이 KDE 최솟값 위치."""
    dn = d / d.max()
    pk = find_peaks(dn, prominence=prom)[0] if prom > 0 else find_peaks(dn)[0]
    va = [float(XS[a + int(np.argmin(d[a:b + 1]))]) for a, b in zip(pk[:-1], pk[1:])]
    return pk, va


def one(job):
    dirpath, variant, skip, bw = job
    series = []
    for f in sorted(Path(dirpath).glob("*.pt")):
        r = torch.load(f, weights_only=True)   # 원격에서 받은 파일이라 pickle 실행을 막는다
        series.append(r["sparsity"][r["variants"].index(variant)].float().numpy())
    P, L = len(series), series[0].shape[0]
    cnt = {p: np.zeros((P, L), int) for p in PROMS}
    val = {p: {} for p in PROMS}
    for i, arr in enumerate(series):
        for l in range(L):
            d = kde_density(arr[l, skip:].astype(np.float64), bw)
            if d is None:
                continue
            for p in PROMS:
                pk, va = peaks_valleys(d, p)
                cnt[p][i, l] = len(pk)
                if len(pk) == 3:
                    val[p][(i, l)] = va
    out = {}
    for p in PROMS:
        n = cnt[p]
        lstar = sorted(set.intersection(*[set(np.where(n[i] == 3)[0].tolist()) for i in range(P)]))
        theta = [float(np.mean([val[p][(i, l)][j] for l in lstar for i in range(P)])) for j in range(2)] if lstar else None
        per_layer = [int((n[:, l] == 3).sum()) for l in range(L)]
        out[str(p)] = {
            "L_star": lstar, "theta": theta, "max_prompts_exactly3": max(per_layer),
            "layers_ge_2_3": [l for l, c in enumerate(per_layer) if c >= 6],
            "mode_share": {k: float(v) for k, v in zip(["1", "2", "3", "4+"], [(n == 1).mean(), (n == 2).mean(), (n == 3).mean(), (n >= 4).mean()])},
            "n_exactly3": int((n == 3).sum()),
            "same_k_ge2_layers": {int(l): int(n[0, l]) for l in range(L) if len(set(n[:, l])) == 1 and n[0, l] >= 2},
            "n_modes": n.tolist(),
        }
    return variant, skip, out


def phi_part(dirpath, prom_main, bw):
    """분위수 방식의 φ̄ (win1024, 4개 레이어, 128스텝 이동평균). 공유 레이어(a)와 문제별 자기 레이어(b) 둘 다 만든다."""
    files = sorted(Path(dirpath).glob("*.pt"))
    names = [f.stem for f in files]
    U = []
    for f in files:
        r = torch.load(f, weights_only=True)
        U.append(r["sparsity"][r["variants"].index("win1024")].float().numpy())
    sel, _ = top_layers(U, 128, 4)
    own = [top_layers([u], 128, 4)[0] for u in U]                 # 문제마다 그 문제만의 변동 상위 4개
    phi = [ma(u[sel].astype(np.float64).mean(0), 128) for u in U]
    phi_own = [ma(u[s].astype(np.float64).mean(0), 128) for u, s in zip(U, own)]
    pool = np.concatenate(phi)
    th = [float(t) for t in np.quantile(pool, [1 / 3, 2 / 3])]
    res = {"bw_mult": bw, "selected_layers": sel, "own_layers": dict(zip(names, own)), "theta_quantile_pooled": th,
           "pooled": {}, "per_prompt": {}, "per_prompt_own_layers": {}}
    dpool = kde_density(pool, bw)
    for p in PROMS:
        pk, va = peaks_valleys(dpool, p)
        res["pooled"][str(p)] = {"n_peaks": len(pk), "peaks": [float(XS[i]) for i in pk], "valleys": va}
    for key, series in (("per_prompt", phi), ("per_prompt_own_layers", phi_own)):
        for name, x in zip(names, series):
            d = kde_density(x, bw)
            res[key][name] = {"own_quantile_theta": [float(t) for t in np.quantile(x, [1 / 3, 2 / 3])]}
            for p in PROMS:
                pk, va = peaks_valleys(d, p)
                res[key][name][str(p)] = {"n_peaks": len(pk), "peaks": [float(XS[i]) for i in pk], "valleys": va}
    return res, names, phi, phi_own, own, pool, dpool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prom-main", type=float, default=0.05)
    ap.add_argument("--bw-mult", type=float, default=1.0, help="KDE 폭 = Scott x bw-mult (1 = scipy 기본)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pm = str(a.prom_main)

    jobs = [(a.dir, v, s, a.bw_mult) for v in VARIANTS for s in SKIPS]
    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(one, jobs))
    P = len(list(Path(a.dir).glob("*.pt")))
    phi_res, names, phi, phi_own, own, pool, dpool = phi_part(a.dir, a.prom_main, a.bw_mult)
    (out / "result.json").write_text(json.dumps({"algorithm1": {f"{v}|skip{s}": o for v, s, o in results}, "phi": phi_res}, indent=1))

    print(f"== Algorithm 1 (KDE Scott x{a.bw_mult}, prominence 필터), P={P}. 각 칸: |L*| / 최대 문제수(/{P}) / 6이상 레이어 수 / 1봉 비율% / 같은 k>=2 레이어")
    for v, s, o in results:
        print(f"\n{v} | 앞 {s}스텝 제외")
        for p in PROMS:
            r = o[str(p)]
            same = r["same_k_ge2_layers"]
            tag = " (점검용, 필터 없음)" if p == 0 else (" <= 주 결과" if str(p) == pm else "")
            print(f"  prom>={p:<4}  |L*|={len(r['L_star'])}  최대 {r['max_prompts_exactly3']}/{P}  6/9이상 레이어 {len(r['layers_ge_2_3'])}개  "
                  f"1봉 {r['mode_share']['1'] * 100:3.0f}%  3봉 시계열 {r['n_exactly3']:3d}개  같은 k>=2 레이어 {same or '-'}{tag}")
            if r["L_star"]:
                print(f"      ** L*={r['L_star']} theta={r['theta']}")

    # 문제별 요약 (Algorithm 1): 주 변형 3개 + 참고, 문제마다 정확히 3봉인 레이어 수
    print(f"\n== 문제별: 정확히 3봉인 레이어 수 (prominence >= {pm}, 필터 없음일 때는 괄호)")
    nm = [f.stem for f in sorted(Path(a.dir).glob("*.pt"))]
    for v in VARIANTS:
        for sk in SKIPS:
            o = {(vv, ss): oo for vv, ss, oo in results}[(v, sk)]
            n_f, n_0 = np.array(o[pm]["n_modes"]), np.array(o["0.0"]["n_modes"])
            print(f"  {v:13s} 앞{sk:<4d} " + "  ".join(f"{nm[i]}:{int((n_f[i] == 3).sum())}({int((n_0[i] == 3).sum())})" for i in range(len(nm))))

    print(f"\n== φ̄, 문제별 (Scott x{a.bw_mult}). (a) 공유 레이어 {phi_res['selected_layers']}, (b) 문제별 자기 레이어")
    for key, label in (("per_prompt", "(a) 공유"), ("per_prompt_own_layers", "(b) 자기")):
        print(f"  {label}: prominence별 문제별 봉우리 수 / 정확히 3봉인 문제의 골")
        for p in PROMS:
            row = [phi_res[key][n][str(p)] for n in names]
            tag = " (점검용)" if p == 0 else (" <= 주 결과" if str(p) == pm else "")
            print(f"    prom>={p:<4} 봉우리 수 {[r['n_peaks'] for r in row]}{tag}")
            for n, r in zip(names, row):
                if r["n_peaks"] >= 2 and p > 0:
                    print(f"        {n}: 봉우리 {np.round(r['peaks'], 3).tolist()} 골 {np.round(r['valleys'], 3).tolist()}")
    print("  (b) 문제별 자기 레이어:", phi_res["own_layers"])

    # 그림 1: 문제 x 레이어 봉우리 수 (주 prominence)
    fig, ax = plt.subplots(len(VARIANTS), len(SKIPS), figsize=(15 * len(SKIPS) / 1.2, 3.1 * len(VARIANTS)), squeeze=False)
    names_all = [f.stem for f in sorted(Path(a.dir).glob("*.pt"))]
    lookup = {(v, s): o for v, s, o in results}
    for i, v in enumerate(VARIANTS):
        for j, s in enumerate(SKIPS):
            g = ax[i, j]
            c = np.array(lookup[(v, s)][pm]["n_modes"])
            g.imshow(np.minimum(c, 6), cmap="YlGnBu", vmin=0, vmax=6, aspect="auto")
            for p_ in range(c.shape[0]):
                for l in range(c.shape[1]):
                    g.text(l, p_, str(c[p_, l]), ha="center", va="center", fontsize=6, color="black" if c[p_, l] < 4 else "white")
                    if c[p_, l] == 3:
                        g.add_patch(Rectangle((l - 0.5, p_ - 0.5), 1, 1, fill=False, ec="#d9630a", lw=1.8))
            g.set_xticks(range(0, c.shape[1], 2))
            g.set_xticklabels(range(0, c.shape[1], 2), fontsize=6)
            g.set_yticks(range(c.shape[0]))
            g.set_yticklabels(names_all, fontsize=6)
            r = lookup[(v, s)][pm]
            g.set_title(f"{v} | 앞 {s}스텝 제외 | 1봉 {r['mode_share']['1'] * 100:.0f}%, 3봉 칸 {r['n_exactly3']}, 모든 문제 3봉 레이어 {r['L_star'] or '없음'}", fontsize=9)
    fig.suptitle(f"prominence >= {pm} 필터 후 봉우리 수 (행=문제, 열=레이어, 주황 테두리=정확히 3봉)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "fig_filtered_peak_counts.png", dpi=75)
    plt.close(fig)

    # 그림 2: 문제마다 따로 — φ̄(자기 레이어) 히스토그램 + KDE + 필터 통과 봉우리·골 (점선 회색: 공유 레이어 KDE)
    ncol = 3
    nrow = -(-len(names) // ncol)
    fig, ax = plt.subplots(nrow, ncol, figsize=(6 * ncol, 3.2 * nrow), squeeze=False)
    for i, (nme, x, xs_) in enumerate(zip(names, phi_own, phi)):
        g = ax[i // ncol, i % ncol]
        d_own = kde_density(x, a.bw_mult)
        d_sh = kde_density(xs_, a.bw_mult)
        g.hist(x, bins=60, density=True, alpha=0.4, color="#7aa5ff")
        g.plot(XS, d_own, lw=1.8, color="#1f3f9f", label="자기 레이어")
        if d_sh is not None:
            g.plot(XS, d_sh, lw=1, color="#888", ls="--", label="공유 레이어")
        pk, va = peaks_valleys(d_own, a.prom_main)
        for k in pk:
            g.plot([XS[k]], [d_own[k]], "o", color="#d9630a", ms=7)
        for v_ in va:
            g.axvline(v_, color="#1f7a4d", ls="--", lw=1.3)
        lo_, hi_ = float(x.min()), float(x.max())
        g.set_xlim(max(0, lo_ - 0.05), min(1, hi_ + 0.05))
        n02 = len(peaks_valleys(d_own, 0.02)[0])
        g.set_title(f"{nme} 레이어 {own[i]}  봉우리: prom0.05 {len(pk)}개, prom0.02 {n02}개, 필터 없음 {len(peaks_valleys(d_own, 0)[0])}개", fontsize=8)
        g.tick_params(labelsize=7)
        if i == 0:
            g.legend(fontsize=7)
    for k in range(len(names), nrow * ncol):
        ax[k // ncol, k % ncol].axis("off")
    fig.suptitle(f"문제마다 따로: φ̄(문제별 변동 상위 4개 레이어 평균 win1024 희소도의 128스텝 이동평균) 분포, KDE = Scott x{a.bw_mult}, "
                 f"주황 점=prominence >= {a.prom_main} 봉우리, 초록 점선=골", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig_phi_per_prompt.png", dpi=70)
    plt.close(fig)
    print("\nsaved to", out)


if __name__ == "__main__":
    main()
