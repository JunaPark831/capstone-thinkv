"""문턱(1/2/5/10%) × 변형 채널마다 (1) 논문 그대로 Algorithm 1, (2) prominence 0.05 필터, (3) 키워드 라벨 AUC 를 한 표로 낸다.
사전 기록: docs/criteria/e3_paper_calibration_spec.md "문턱 1% → 2·5·10% 재측정".

사용: HF_HUB_OFFLINE=1 python -I threshold_analysis.py --dir THRESH_DIR --out OUT_DIR
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
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_paper import calibrate, kde_density  # noqa: E402
from keyword_signal import LAB, auc, label_tokens, token_text  # noqa: E402

VARS = ["win1024", "win256", "full_nosink", "full_literal"]
PCTS = ["1", "2", "5", "10"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--W", type=int, default=64)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--vars", default=None, help="범위 이름 쉼표 목록 (기본: 문턱 재생용)")
    ap.add_argument("--pcts", default=None, help="정의 이름 쉼표 목록")
    a = ap.parse_args()
    global VARS, PCTS
    if a.vars:
        VARS = a.vars.split(",")
    if a.pcts:
        PCTS = a.pcts.split(",")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.dir).glob("*.pt"))
    data = [torch.load(f, weights_only=True) for f in files]
    names = [f.stem for f in files]
    P, L = len(data), data[0]["sparsity"].shape[1]
    tok = AutoTokenizer.from_pretrained(a.model)
    labs = []
    for r in data:
        text, starts, _ = token_text(tok, r["token_ids"])
        lab, _, _ = label_tokens(text, starts, len(r["token_ids"]), a.W, "A")
        labs.append(lab[:-1])
    print(f"P={P}, 파일 {names}, 회귀(1% 채널 대 저장값 최대 차이): " +
          ", ".join(f"{n}:{max(list(r.get("regression_max_abs_diff_vs_saved_1pct", r.get("regression_max_abs_diff_vs_prev_replay_1pct", {0: 0})).values()) or [0]):.3f}" for n, r in zip(names, data)))

    rows = {}
    print(f"\n{'채널':16s} {'Alg1 L*':8s} {'최대문제수':6s} {'봉우리중앙값':6s} | {'prom0.05 1봉%':12s} {'3봉시계열':6s} {'최대문제수':6s} | {'TvsE AUC중앙값':12s} {'>=0.65':6s} {'<=0.35':6s} {'상위AUC(잔차,문제별)':24s} {'RvsE >=0.65':6s}")
    for v in VARS:
        for p in PCTS:
            ch = f"{v}@{p}"
            ci = data[0]["variants"].index(ch)
            series = [r["sparsity"][ci].float().numpy() for r in data]
            # (1) 논문 그대로: Scott x1, prominence 없음, 정확히 3봉 교집합
            res = calibrate(series, T=3, skip=0)
            n_modes = np.array(res["n_modes"])
            # (2) prominence 0.05
            cnt = np.zeros((P, L), int)
            for i, arr in enumerate(series):
                for l in range(L):
                    d = kde_density(arr[l].astype(np.float64))
                    if d is not None:
                        cnt[i, l] = len(find_peaks(d / d.max(), prominence=0.05)[0])
            # (3) 키워드 AUC
            te, ted, tem, re_ = [], [], [], []
            for l in range(L):
                raw = [x[l].astype(np.float64) for x in series]
                det = [x - uniform_filter1d(x, 1025, mode="nearest") for x in raw]
                pos = np.concatenate([x[lab == LAB["T"]] for x, lab in zip(raw, labs)])
                neg = np.concatenate([x[lab == LAB["E"]] for x, lab in zip(raw, labs)])
                te.append(auc(pos, neg))
                ted.append(auc(np.concatenate([x[lab == LAB["T"]] for x, lab in zip(det, labs)]),
                               np.concatenate([x[lab == LAB["E"]] for x, lab in zip(det, labs)])))
                per = [auc(x[lab == LAB["T"]], x[lab == LAB["E"]]) for x, lab in zip(raw, labs) if (lab == LAB["T"]).sum() >= 20 and (lab == LAB["E"]).sum() >= 20]
                tem.append(float(np.median(per)) if per else None)
                re_.append(auc(np.concatenate([x[lab == LAB["R"]] for x, lab in zip(raw, labs)]), neg))
            hi = [l for l in range(L) if te[l] is not None and te[l] >= 0.65]
            lo = [l for l in range(L) if te[l] is not None and te[l] <= 0.35]
            best = sorted(range(L), key=lambda l: -(te[l] or 0))[:3]
            med_hi = [tem[l] for l in hi if tem[l] is not None]
            gate = "신호 있음" if len(hi) >= 4 and med_hi and np.median(med_hi) >= 0.60 else ("신호 있음(반대)" if len(lo) >= 4 else "신호 없음")
            rows[ch] = {"alg1_L_star": res["L_star"], "alg1_theta": res["theta"], "alg1_max_prompts_exactly3": int(max(res["layer_prompts_exactly_T"])),
                        "alg1_median_modes": float(np.median(n_modes)), "prom05_share1": float((cnt == 1).mean()), "prom05_n_exactly3": int((cnt == 3).sum()),
                        "prom05_max_prompts_exactly3": int(max((cnt[:, l] == 3).sum() for l in range(L))), "prom05_L_star": [l for l in range(L) if (cnt[:, l] == 3).all()],
                        "TvsE_pooled": te, "TvsE_detrended": ted, "TvsE_per_prompt_median": tem, "RvsE_pooled": re_,
                        "layers_ge_0.65": hi, "layers_le_0.35": lo, "gate": gate, "mean_sparsity": float(np.mean([x.mean() for x in series]))}
            print(f"{ch:16s} {len(res['L_star']):<8d} {rows[ch]['alg1_max_prompts_exactly3']:<6d} {rows[ch]['alg1_median_modes']:<6.0f} | "
                  f"{rows[ch]['prom05_share1'] * 100:<12.0f} {rows[ch]['prom05_n_exactly3']:<6d} {rows[ch]['prom05_max_prompts_exactly3']:<6d} | "
                  f"{np.median([x for x in te if x is not None]):<12.3f} {len(hi):<6d} {len(lo):<6d} "
                  f"{str([(l, round(te[l], 2), round(ted[l], 2), round(tem[l], 2) if tem[l] else None) for l in best]):24s} "
                  f"{sum(1 for x in re_ if x is not None and x >= 0.65):<6d}  → {gate}")
    (out / "result.json").write_text(json.dumps({"W": a.W, "P": P, "files": names, "rows": rows}, indent=1))

    # 그림: 채널별 TvsE AUC (레이어 32개) — 변형별 한 패널, 문턱 4개 선
    nrow = -(-len(VARS) // 2)
    fig, ax = plt.subplots(nrow, 2, figsize=(15, 3.5 * nrow), sharey=True, squeeze=False)
    for g, v in zip(ax.ravel(), VARS):
        pass
    for g, v in zip(ax.ravel(), VARS):
        for p, col in zip(PCTS, ["#2f6fed", "#1f7a4d", "#d9630a", "#7a1fa2", "#888"]):
            y = [x if x is not None else np.nan for x in rows[f"{v}@{p}"]["TvsE_pooled"]]
            g.plot(range(L), y, "-o", ms=3, lw=1, color=col, label=str(p))
        g.axhline(0.5, color="#555", lw=0.8)
        g.axhline(0.65, color="#999", lw=0.8, ls="--")
        g.axhline(0.35, color="#999", lw=0.8, ls="--")
        g.set_ylim(0.3, 0.75)
        g.set_title(f"{v}: Transition vs Execution AUC (문제 합침), 정의별", fontsize=10)
        g.set_xlabel("레이어")
        g.legend(fontsize=8, title="정의")
    fig.suptitle(f"희소도 문턱 1/2/5/10% 별 키워드 라벨 분리도 (W={a.W}). 0.5=구분 없음, 점선=gate", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig_auc_by_threshold.png", dpi=80)
    plt.close(fig)
    print("saved to", out)


if __name__ == "__main__":
    main()
