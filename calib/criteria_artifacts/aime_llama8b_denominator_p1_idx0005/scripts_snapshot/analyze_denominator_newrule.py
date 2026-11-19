"""docs/e3_denominator_variants.md §3-4 의 사전 기준대로 replay_sparsity_v2 결과를 분석한다.

사용: python analyze_denominator.py <replay_v2.pt> <out_dir> [min_step=1100]
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
NEW = ["paper_nosink", "paper_win256", "paper_win1024"]
REF = ["legacy_perhead", "paper"]
S1_FRAC, S1_LAYERS, S2_CORR, S3_LAYERS = 0.20, 8, 0.30, 4


def peak_counts(x):
    out = {}
    for m in MULTS:
        try:
            d = gaussian_kde(x, bw_method=gaussian_kde(x).factor * m)(XS)
            d = d / d.max()
            out[m] = (len(find_peaks(d)[0]), len(find_peaks(d, prominence=0.05)[0]))
        except (np.linalg.LinAlgError, ValueError):
            out[m] = (0, 0)
    return out


def main(src, out_dir, min_step=1100):
    min_step = int(min_step)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    r = torch.load(src, weights_only=False)
    names, sp = r["variants"], r["sparsity"].float().numpy()
    V, L, T = sp.shape
    idx = {n: i for i, n in enumerate(names)}
    step = np.arange(T)
    n_kv = r["input_len"] + step + 1                      # 스텝 t에서 KV 길이
    ker = np.ones(128) / 128
    res = {"src": src, "steps": T, "min_step": min_step, "variants": {}}

    print(f"steps={T}  min_step={min_step}  (판정은 step>={min_step}만)")
    for name in REF + NEW:
        x_all = sp[idx[name]]                               # (L, T)
        rows = []
        for l in range(L):
            x = x_all[l, min_step:]
            ma = np.convolve(x, ker, "valid")
            pk = peak_counts(x)
            rows.append({
                "layer": l,
                "frac_gt_0.98": float((x > 0.98).mean()),
                "corr_step": float(np.corrcoef(step[min_step:], x)[0, 1]) if x.std() > 0 else 0.0,
                "ma_p5_p95_width": float(np.percentile(ma, 95) - np.percentile(ma, 5)),
                "modes_raw": [pk[m][0] for m in MULTS],
                "modes_prom": [pk[m][1] for m in MULTS],
            })
        s1 = [q["layer"] for q in rows if q["frac_gt_0.98"] < S1_FRAC]
        s2 = [q["layer"] for q in rows if q["layer"] in s1 and abs(q["corr_step"]) < S2_CORR]
        # 개정(2026-11-04, 사용자 지시): S3 = Scott ×1, prominence 없음, 정확히 3봉. 구 기준은 병기한다.
        s3 = [q["layer"] for q in rows if q["modes_raw"][1] == 3]
        s3_old = [q["layer"] for q in rows if all(k == 3 for k in q["modes_raw"])]
        s3p = [q["layer"] for q in rows if all(k == 3 for k in q["modes_prom"])]
        hist_x1 = {}
        for q in rows:
            hist_x1[q["modes_raw"][1]] = hist_x1.get(q["modes_raw"][1], 0) + 1
        v = {
            "S1_layers": s1, "S1_pass": len(s1) >= S1_LAYERS,
            "S2_layers": s2, "S2_pass": len(s1) > 0 and len(s2) * 2 >= len(s1),
            "S3_layers": s3, "S3_pass": len(s3) >= S3_LAYERS,
            "S3_old_layers_3bw_raw": s3_old, "S3_old_pass": len(s3_old) >= S3_LAYERS,
            "S3_old_layers_3bw_prom0.05": s3p,
            "modes_x1_histogram": dict(sorted(hist_x1.items())),
            "D_width_median": float(np.median([q["ma_p5_p95_width"] for q in rows])),
            "D_width_max": float(max(q["ma_p5_p95_width"] for q in rows)),
            "frac_gt_0.98_overall": float((x_all[:, min_step:] > 0.98).mean()),
            "per_layer": rows,
        }
        res["variants"][name] = v
        print(f"\n== {name}: S1 {len(s1)}/{L} layers (>= {S1_LAYERS} → {v['S1_pass']}) | S2 {len(s2)} (→ {v['S2_pass']}) | "
              f"S3(×1,no prominence) {len(s3)} layers (>= {S3_LAYERS} → {v['S3_pass']}) [old 3bw: {len(s3_old)}, old prom: {len(s3p)}] | "
              f"×1 mode-count histogram {v['modes_x1_histogram']} | "
              f"D width median {v['D_width_median']:.3f} max {v['D_width_max']:.3f} | frac>.98 {v['frac_gt_0.98_overall']:.3f}")

    # 판정 (사전 표를 기계적으로 적용)
    new = [res["variants"][n] for n in NEW]
    if any(v["S1_pass"] and v["S2_pass"] and v["S3_pass"] for v in new):
        verdict = "지지됨 (S1·S2·S3 충족 변형 있음)"
    elif any(v["S1_pass"] and v["S2_pass"] for v in new):
        verdict = "부분 지지 (S1·S2 충족, S3 불충족)"
    elif all(not v["S1_pass"] for v in new):
        verdict = "기각 (세 변형 모두 S1 불충족)"
    else:
        verdict = "사전 표에 없는 경우: S1 충족 변형이 있으나 S2 불충족. 판정 보류, 사실만 기록"
    res["verdict"] = verdict
    print("\n판정(개정 S3 적용):", verdict)
    ctrl = res["variants"]["paper"]["S3_pass"]   # 음성 대조군: 포화된 행 전체 변형. 여기서 S3가 통과하면 S3는 신호 증거가 못 된다.
    res["negative_control_paper_S3_pass"] = bool(ctrl)
    print(f"음성 대조군(paper, 행 전체·포화) S3 통과 여부: {ctrl} "
          f"({len(res['variants']['paper']['S3_layers'])}개 레이어)  ← True면 개정 S3는 이 데이터에서 신호를 가르지 못한다")

    # 4절 보고: 무거운 위치 수 k(t)
    print("\n무거운 위치 수 k = (1-s)*m (레이어 10, 구간별 중앙값). m = 행에 넣은 위치 수")
    m_of = {"paper": n_kv, "paper_nosink": n_kv - 1, "paper_win256": np.minimum(256, n_kv), "paper_win1024": np.minimum(1024, n_kv)}
    bins = [(0, 500), (500, 1500), (1500, 3300), (3300, T)]
    res["k_layer10"] = {}
    for name, m in m_of.items():
        k = (1 - sp[idx[name], 10]) * m
        med = [float(np.median(k[a:b])) for a, b in bins if b > a]
        res["k_layer10"][name] = med
        print(f"  {name:14s} " + "  ".join(f"[{a}-{b}) {md:8.1f}" for (a, b), md in zip(bins, med)))
    print("  (n 중앙값)       " + "  ".join(f"[{a}-{b}) {np.median(n_kv[a:b]):8.1f}" for a, b in bins if b > a))

    # sink_mass
    sm = sp[idx["sink_mass"]]
    res["sink_mass"] = {"mean_all": float(sm.mean()), "layer10_mean": float(sm[10].mean()),
                        "layer10_corr_step": float(np.corrcoef(step, sm[10])[0, 1]),
                        "per_layer_mean": [float(v) for v in sm.mean(1)]}
    print(f"\nsink_mass: 전체 평균 {sm.mean():.3f}, 레이어10 평균 {sm[10].mean():.3f} (스텝 상관 {res['sink_mass']['layer10_corr_step']:+.2f}); "
          f"레이어별 평균 min {sm.mean(1).min():.3f} max {sm.mean(1).max():.3f}")

    (out_dir / "result.json").write_text(json.dumps(res, indent=1))

    # 그림 1: 레이어 10 시계열
    show = ["paper"] + NEW + ["sink_mass"]
    fig, ax = plt.subplots(len(show), 1, figsize=(14, 2.4 * len(show)), sharex=True)
    for a, n in zip(ax, show):
        x = sp[idx[n], 10]
        a.plot(x, lw=0.3, alpha=0.4)
        a.plot(np.arange(127, T), np.convolve(x, ker, "valid"), lw=1.2)
        a.axvline(min_step, color="gray", ls=":", lw=0.8)
        a.set_ylim(0, 1)
        a.set_ylabel(n, fontsize=8)
    ax[0].set_title("layer 10 over decode steps (raw + MA128); dotted = judgment start")
    fig.tight_layout()
    fig.savefig(out_dir / "layer10_timeseries.png", dpi=110)

    # 그림 2: 모든 레이어의 MA128 겹쳐 그리기 (변형별)
    fig, ax = plt.subplots(len(NEW) + 1, 1, figsize=(14, 2.6 * (len(NEW) + 1)), sharex=True)
    for a, n in zip(ax, ["paper"] + NEW):
        for l in range(L):
            a.plot(np.arange(127, T), np.convolve(sp[idx[n], l], ker, "valid"), lw=0.7, alpha=0.7)
        a.axvline(min_step, color="gray", ls=":", lw=0.8)
        a.set_ylim(0, 1)
        a.set_ylabel(n, fontsize=8)
    ax[0].set_title("all layers MA128 per variant")
    fig.tight_layout()
    fig.savefig(out_dir / "all_layers_ma128.png", dpi=110)
    print("\nsaved to", out_dir)


if __name__ == "__main__":
    main(*sys.argv[1:])
