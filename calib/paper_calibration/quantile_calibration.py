"""분위수 임계값 대체 방법 (논문 Algorithm 1 아님). 사전 기록: docs/e3_paper_calibration_spec.md "분위수 임계값 실행 사전 기록".

사용: python -I quantile_calibration.py --dir DATA_DIR --out OUT_DIR [--variant win1024] [--k 4] [--win 128]
산출: result.json, fig_timeseries.png (문제별 φ̄ 시계열 + θ 띠), fig_distribution.png (합친 분포·KDE·θ·레이어 점수)
"""
import argparse
import json
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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_paper import XS, kde_density  # noqa: E402

COL = ["#2f6fed", "#1f7a4d", "#d9630a"]       # Execution, Reasoning, Transition
NAMES = ["Execution", "Reasoning", "Transition"]


def ma(x, w):
    return np.convolve(x, np.ones(w) / w, "valid")


def top_layers(U, w, k, skip_prompt=None):
    """U: 문제별 (L, steps). 문제들에 걸친 '이동평균의 시간 표준편차' 평균이 큰 레이어 k개."""
    sc = np.mean([[ma(u[l].astype(np.float64), w).std() for l in range(u.shape[0])]
                  for i, u in enumerate(U) if i != skip_prompt], axis=0)
    return sorted(np.argsort(-sc)[:k].tolist()), sc


def modes_valleys(x):
    d = kde_density(x)
    if d is None:
        return None, [], []
    pk = find_peaks(d)[0]
    va = [float(XS[a + int(np.argmin(d[a:b + 1]))]) for a, b in zip(pk[:-1], pk[1:])]
    return d, [float(XS[i]) for i in pk], va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="win1024")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--win", type=int, default=128)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(a.dir).glob("*.pt"))
    names = [f.stem for f in files]
    U = []
    for f in files:
        r = torch.load(f, weights_only=True)          # 원격에서 받은 파일이라 pickle 실행을 막는다
        U.append(r["sparsity"][r["variants"].index(a.variant)].float().numpy())
    P = len(U)

    sel, score = top_layers(U, a.win, a.k)
    phi = [ma(u[sel].astype(np.float64).mean(0), a.win) for u in U]      # 분류 값 φ̄ (스텝 t는 t+win-1 번째 스텝)
    pool = np.concatenate(phi)
    th = np.quantile(pool, [1 / 3, 2 / 3])
    cls = [np.digitize(p, th) for p in phi]                              # 0=E, 1=R, 2=T

    # 진단: KDE 봉우리/골 (논문 방식 규칙)
    d_pool, m_pool, v_pool = modes_valleys(pool)
    per = []
    for i, p in enumerate(phi):
        d, mo, va = modes_valleys(p)
        c = cls[i]
        samp = c[::a.win]                                                # 128스텝 간격으로 본 분류
        rho = float(spearmanr(np.arange(len(p)), p)[0])
        third = len(p) // 3
        per.append({
            "prompt": names[i], "steps": int(U[i].shape[1]), "n_phi": int(len(p)), "kde_modes": mo, "kde_valleys": va,
            "frac_E_R_T": [float((c == k).mean()) for k in range(3)],
            "switches_per_refresh": float((np.diff(samp) != 0).mean()) if len(samp) > 1 else None,
            "spearman_phi_vs_step": rho,
            "frac_T_first_third": float((c[:third] == 2).mean()), "frac_T_last_third": float((c[-third:] == 2).mean()),
            "frac_E_first_third": float((c[:third] == 0).mean()), "frac_E_last_third": float((c[-third:] == 0).mean()),
        })
    loo = [top_layers(U, a.win, a.k, skip_prompt=i)[0] for i in range(P)]
    res = {"variant": a.variant, "k": a.k, "win": a.win, "P": P, "selected_layers": sel,
           "layer_score": [float(s) for s in score], "theta": [float(t) for t in th],
           "pooled_kde_modes": m_pool, "pooled_kde_valleys": v_pool, "per_prompt": per,
           "leave_one_out_top_layers": loo,
           "pool_frac_E_R_T": [float((np.concatenate(cls) == k).mean()) for k in range(3)]}
    (out / "result.json").write_text(json.dumps(res, indent=1))

    # 그림 1: 문제별 φ̄ 시계열과 θ 띠
    ncol = 3
    nrow = -(-P // ncol)
    fig, ax = plt.subplots(nrow, ncol, figsize=(6 * ncol, 2.9 * nrow), squeeze=False)
    lo, hi = float(min(p.min() for p in phi)), float(max(p.max() for p in phi))
    pad = 0.03 * (hi - lo)
    for i, p in enumerate(phi):
        g = ax[i // ncol, i % ncol]
        g.axhspan(lo - pad, th[0], color=COL[0], alpha=0.12)
        g.axhspan(th[0], th[1], color=COL[1], alpha=0.12)
        g.axhspan(th[1], hi + pad, color=COL[2], alpha=0.12)
        g.plot(np.arange(len(p)) + a.win - 1, p, lw=0.8, color="#222")
        g.axhline(th[0], color=COL[0], lw=1)
        g.axhline(th[1], color=COL[2], lw=1)
        g.set_ylim(lo - pad, hi + pad)
        e = per[i]
        g.set_title(f"{names[i]}  스텝 {e['steps']}  E/R/T={e['frac_E_R_T'][0]:.2f}/{e['frac_E_R_T'][1]:.2f}/{e['frac_E_R_T'][2]:.2f}  "
                    f"스텝과의 상관 {e['spearman_phi_vs_step']:+.2f}", fontsize=8)
        g.tick_params(labelsize=7)
    for k in range(P, nrow * ncol):
        ax[k // ncol, k % ncol].axis("off")
    fig.suptitle(f"φ̄ = 레이어 {sel} 평균 희소도({a.variant})의 {a.win}스텝 이동평균 | θ1={th[0]:.3f}, θ2={th[1]:.3f} (합친 분포의 33%/67%) | "
                 f"파랑=Execution, 초록=Reasoning, 주황=Transition 구간", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "fig_timeseries.png", dpi=70)
    plt.close(fig)

    # 그림 2: 합친 분포 + KDE(봉우리·골) + 문제별 KDE, 레이어 점수
    fig, ax = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [1.5, 1]})
    g = ax[0]
    g.hist(pool, bins=80, density=True, alpha=0.4, color="#7aa5ff", label="φ̄ 히스토그램(9개 문제 합침)")
    for i, p in enumerate(phi):
        d, _, _ = modes_valleys(p)
        if d is not None:
            g.plot(XS, d, lw=0.7, alpha=0.5, color="#888")
    g.plot(XS, d_pool, lw=2, color="#1f3f9f", label="합친 KDE (Scott ×1)")
    for m in m_pool:
        g.plot([m], [np.interp(m, XS, d_pool)], "o", color="#d9630a", ms=6)
    for v in v_pool:
        g.axvline(v, color="#1f7a4d", ls="--", lw=1.2)
    g.axvline(th[0], color=COL[0], lw=2, label=f"θ1 {th[0]:.3f} (33%)")
    g.axvline(th[1], color=COL[2], lw=2, label=f"θ2 {th[1]:.3f} (67%)")
    g.set_xlim(max(0, lo - 0.05), min(1, hi + 0.05))
    g.set_title(f"φ̄ 분포. 합친 KDE 봉우리 {len(m_pool)}개(주황 점), 골(초록 점선) {len(v_pool)}개. 회색 선=문제별 KDE", fontsize=10)
    g.legend(fontsize=8)
    g = ax[1]
    colr = [COL[2] if l in sel else "#aab" for l in range(len(score))]
    g.bar(range(len(score)), score, color=colr)
    g.set_xlabel("레이어")
    g.set_ylabel("128스텝 이동평균의 시간 표준편차 (문제 평균)")
    g.set_title(f"레이어 점수. 선택된 4개(주황) = {sel}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_distribution.png", dpi=75)
    plt.close(fig)

    print(f"선택 레이어 {sel}  θ1={th[0]:.4f} θ2={th[1]:.4f}  φ̄ 범위 {lo:.3f}~{hi:.3f}")
    print(f"합친 분포 KDE 봉우리 {len(m_pool)}개 {np.round(m_pool, 3).tolist()}  골 {np.round(v_pool, 3).tolist()}")
    print("문제별: 스텝, KDE 봉우리 수, E/R/T 비율, 스텝과의 Spearman, T 비율(앞1/3→뒤1/3), 128스텝 간격 전환율")
    for e in per:
        print(f"  {e['prompt']} {e['steps']:6d} 봉우리 {len(e['kde_modes']):2d}  "
              f"{e['frac_E_R_T'][0]:.2f}/{e['frac_E_R_T'][1]:.2f}/{e['frac_E_R_T'][2]:.2f}  rho {e['spearman_phi_vs_step']:+.2f}  "
              f"T {e['frac_T_first_third']:.2f}->{e['frac_T_last_third']:.2f}  E {e['frac_E_first_third']:.2f}->{e['frac_E_last_third']:.2f}  "
              f"전환 {e['switches_per_refresh'] if e['switches_per_refresh'] is None else round(e['switches_per_refresh'], 2)}")
    cnt = {}
    for t in loo:
        for l in t:
            cnt[l] = cnt.get(l, 0) + 1
    print("한 문제씩 빼고 다시 고른 상위 4개에 든 횟수(9회 중):", dict(sorted(cnt.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
