"""키워드 라벨(논문 표 7)로 디코드 스텝에 사고 유형을 붙이고, 라벨별 희소도 분포가 갈리는지 AUC 로 본다.
사전 기록: docs/criteria/e3_paper_calibration_spec.md "키워드 라벨 신호 검증". 키워드는 검증용 라벨이며 분류기가 아니다.

사용: HF_HUB_OFFLINE=1 python -I keyword_signal.py --dir DATA_DIR --out OUT_DIR [--W 64]
산출: result.json, fig_auc.png (레이어별 AUC), fig_best_layers.png (상위 4개 레이어의 라벨별 히스토그램)
"""
import argparse
import json
import re
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
from scipy.stats import rankdata
from transformers import AutoTokenizer

KW = {  # 논문 표 7. 긴 구절을 먼저 두어 regex 가 우선 매칭하게 한다
    "T": ["wait no", "but wait", "earlier i said that", "wait", "hmm", "alternatively"],
    "R": ["think", "approach", "remember", "find", "okay", "suppose", "verify"],
    "E": ["now", "the steps are"],
}
LAB = {"E": 0, "R": 1, "T": 2, None: -1}
NAMES = {0: "Execution", 1: "Reasoning", 2: "Transition"}
VARIANTS = ["win1024", "win256", "full_nosink"]
PAIRS = [("T", "E"), ("R", "E"), ("T", "R")]   # (양성, 음성): 논문 방향이면 AUC > 0.5


def token_text(tok, ids):
    pieces = [tok.decode([int(i)]) for i in ids]
    starts = np.cumsum([0] + [len(p) for p in pieces[:-1]])
    return "".join(pieces), starts, pieces


def label_tokens(text, starts, n, W, scheme):
    """토큰 인덱스별 라벨(-1/0/1/2)과 키워드 적중 수."""
    pat = "|".join(rf"\b{re.escape(k)}\b" for typ in ("T", "R", "E") for k in KW[typ])
    typ_of = {k: t for t in KW for k in KW[t]}
    hits = []          # (토큰 인덱스, 유형, 키워드)
    counts = {}
    for m in re.finditer(pat, text, flags=re.IGNORECASE):
        k = m.group(0).lower()
        t = typ_of[k]
        ti = int(np.searchsorted(starts, m.start(), side="right") - 1)
        hits.append((ti, t, k))
        counts[k] = counts.get(k, 0) + 1
    lab = np.full(n, -1, int)
    hits.sort()
    for j, (ti, t, _) in enumerate(hits):
        end = min(n, ti + W, hits[j + 1][0] if j + 1 < len(hits) else n)
        lab[ti:end] = LAB[t]
    if scheme == "B":  # 미라벨 중 수식 줄 -> Execution
        pos = 0
        for line in text.split("\n"):
            if ("=" in line or "\\boxed" in line or "\\frac" in line):
                a = int(np.searchsorted(starts, pos, side="right") - 1)
                b = int(np.searchsorted(starts, pos + len(line), side="right"))
                seg = lab[a:b]
                seg[seg == -1] = 0
            pos += len(line) + 1
    return lab, counts, len(hits)


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return None
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--W", type=int, default=64)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    files = sorted(Path(a.dir).glob("*.pt"))
    data = [torch.load(f, weights_only=True) for f in files]   # 원격에서 받은 파일이라 pickle 실행을 막는다
    names = [f.stem for f in files]
    L = data[0]["sparsity"].shape[1]

    texts = [token_text(tok, r["token_ids"]) for r in data]
    res = {"W": a.W, "gate": {}, "schemes": {}}
    for scheme in ("A", "B"):
        for W in ([a.W] if scheme == "B" else sorted({32, a.W, 128})):
            labs, kwc, nh = [], {}, []
            for (text, starts, _), r in zip(texts, data):
                lab, c, h = label_tokens(text, starts, len(r["token_ids"]), W, scheme)
                labs.append(lab[:-1])        # sparsity 는 길이 n-1, 인덱스 s = 쿼리 토큰 ids[s]
                for k, v in c.items():
                    kwc[k] = kwc.get(k, 0) + v
                nh.append(h)
            key = f"{scheme}_W{W}"
            S = {"keyword_counts": dict(sorted(kwc.items(), key=lambda kv: -kv[1])), "hits_per_prompt": dict(zip(names, nh)),
                 "label_steps": {NAMES[k]: int(sum((l == k).sum() for l in labs)) for k in range(3)},
                 "unlabeled_steps": int(sum((l == -1).sum() for l in labs)), "variants": {}}
            for v in VARIANTS:
                vi = data[0]["variants"].index(v)
                per_layer = []
                for l in range(L):
                    raw = [r["sparsity"][vi, l].float().numpy().astype(np.float64) for r in data]
                    det = [x - uniform_filter1d(x, 1025, mode="nearest") for x in raw]
                    row = {}
                    for pname, (p_, n_) in zip(["TvsE", "RvsE", "TvsR"], PAIRS):
                        P_, N_ = LAB[p_], LAB[n_]
                        pooled = auc(np.concatenate([x[lab == P_] for x, lab in zip(raw, labs)]),
                                     np.concatenate([x[lab == N_] for x, lab in zip(raw, labs)]))
                        pooled_det = auc(np.concatenate([x[lab == P_] for x, lab in zip(det, labs)]),
                                         np.concatenate([x[lab == N_] for x, lab in zip(det, labs)]))
                        per = [auc(x[lab == P_], x[lab == N_]) for x, lab in zip(raw, labs)
                               if (lab == P_).sum() >= 20 and (lab == N_).sum() >= 20]
                        row[pname] = {"pooled": pooled, "pooled_detrended": pooled_det,
                                      "per_prompt_median": float(np.median(per)) if per else None, "n_prompts": len(per)}
                    row["median_by_label"] = {NAMES[k]: float(np.median(np.concatenate([x[lab == k] for x, lab in zip(raw, labs)])))
                                              for k in range(3) if sum((lab == k).sum() for lab in labs) > 0}
                    per_layer.append(row)
                S["variants"][v] = per_layer
            res["schemes"][key] = S
            # 요약 출력
            print(f"\n== 안 {scheme}, W={W}: 라벨 스텝 {S['label_steps']}, 미라벨 {S['unlabeled_steps']}, 키워드 상위 {list(S['keyword_counts'].items())[:8]}")
            for v in VARIANTS:
                pl = S["variants"][v]
                te = [r["TvsE"]["pooled"] for r in pl]
                ted = [r["TvsE"]["pooled_detrended"] for r in pl]
                tem = [r["TvsE"]["per_prompt_median"] for r in pl]
                n_hi = sum(1 for x in te if x is not None and x >= 0.65)
                n_lo = sum(1 for x in te if x is not None and x <= 0.35)
                best = sorted(range(L), key=lambda l: -(te[l] or 0))[:4]
                print(f"  {v:12s} TvsE 합친 AUC: 중앙값 {np.median([x for x in te if x is not None]):.3f}, >=0.65 레이어 {n_hi}개, <=0.35 레이어 {n_lo}개 | "
                      f"상위4 {[(l, round(te[l], 3), round(ted[l], 3) if ted[l] is not None else None, round(tem[l], 3) if tem[l] is not None else None) for l in best]} (레이어, 합친, 잔차, 문제별중앙값)")
                for pname in ("RvsE", "TvsR"):
                    vals = [r[pname]["pooled"] for r in pl if r[pname]["pooled"] is not None]
                    print(f"  {'':12s} {pname} 합친 AUC 중앙값 {np.median(vals):.3f}, >=0.65 {sum(x >= 0.65 for x in vals)}개, <=0.35 {sum(x <= 0.35 for x in vals)}개")

    # gate: 안 A, W=a.W, win1024
    pl = res["schemes"][f"A_W{a.W}"]["variants"]["win1024"]
    te = [r["TvsE"]["pooled"] for r in pl]
    hi = [l for l in range(L) if te[l] is not None and te[l] >= 0.65]
    lo = [l for l in range(L) if te[l] is not None and te[l] <= 0.35]
    med_hi = [pl[l]["TvsE"]["per_prompt_median"] for l in hi if pl[l]["TvsE"]["per_prompt_median"] is not None]
    gate = "신호 있음" if len(hi) >= 4 and med_hi and float(np.median(med_hi)) >= 0.60 else ("신호 있음(방향 반대)" if len(lo) >= 4 else "신호 없음")
    res["gate"] = {"rule": "win1024, scheme A, TvsE pooled AUC>=0.65 in >=4 layers and median per-prompt AUC>=0.60", "layers_ge_0.65": hi,
                   "layers_le_0.35": lo, "verdict": gate}
    print(f"\n*** GATE (win1024, 안 A, W={a.W}): {gate} | AUC>=0.65 레이어 {hi} | AUC<=0.35 레이어 {lo}")
    (out / "result.json").write_text(json.dumps(res, indent=1))

    # 그림 1: 레이어별 AUC (안 A, W=a.W), 변형 3개 x 쌍 3개, 원시/잔차
    S = res["schemes"][f"A_W{a.W}"]
    fig, ax = plt.subplots(len(VARIANTS), 1, figsize=(14, 3.2 * len(VARIANTS)), sharex=True)
    for g, v in zip(ax, VARIANTS):
        pl = S["variants"][v]
        xs = np.arange(L)
        for off, (pname, col) in enumerate(zip(["TvsE", "RvsE", "TvsR"], ["#d9630a", "#1f7a4d", "#2f6fed"])):
            y = [r[pname]["pooled"] if r[pname]["pooled"] is not None else np.nan for r in pl]
            yd = [r[pname]["pooled_detrended"] if r[pname]["pooled_detrended"] is not None else np.nan for r in pl]
            g.bar(xs + (off - 1) * 0.27, np.array(y) - 0.5, width=0.27, bottom=0.5, color=col, alpha=0.85, label=f"{pname} 원시")
            g.plot(xs + (off - 1) * 0.27, yd, "k_", ms=6, mew=1.2, label="잔차(위치 제거)" if off == 0 else None)
        g.axhline(0.5, color="#555", lw=0.8)
        g.axhline(0.65, color="#999", lw=0.8, ls="--")
        g.axhline(0.35, color="#999", lw=0.8, ls="--")
        g.set_ylim(0.2, 0.8)
        g.set_ylabel("AUC")
        g.set_title(f"{v}: 라벨 쌍별 AUC (0.5=구분 없음, >0.5=논문 방향: E<R<T). 막대=문제 합침, 검은 선=위치 추세 제거 후", fontsize=9)
        if v == VARIANTS[0]:
            g.legend(fontsize=7, ncol=4)
    ax[-1].set_xlabel("레이어")
    fig.suptitle(f"키워드 라벨(논문 표 7, W={a.W}) 기준 희소도 분리도 | 라벨 스텝 {S['label_steps']} | 점선=gate 0.65/0.35", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "fig_auc.png", dpi=80)
    plt.close(fig)

    # 그림 2: win1024 TvsE 상위 4개 레이어의 라벨별 히스토그램
    pl = S["variants"]["win1024"]
    best = sorted(range(L), key=lambda l: -(pl[l]["TvsE"]["pooled"] or 0))[:4]
    vi = data[0]["variants"].index("win1024")
    labs = []
    for (text, starts, _), r in zip(texts, data):
        lab, _, _ = label_tokens(text, starts, len(r["token_ids"]), a.W, "A")
        labs.append(lab[:-1])
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.6))
    for g, l in zip(ax, best):
        for k, col in zip(range(3), ["#2f6fed", "#1f7a4d", "#d9630a"]):
            x = np.concatenate([r["sparsity"][vi, l].float().numpy()[lab == k] for r, lab in zip(data, labs)])
            if len(x):
                g.hist(x, bins=60, range=(0, 1), density=True, alpha=0.45, color=col, label=f"{NAMES[k]} (n={len(x)})")
        g.set_title(f"win1024 L{l}  TvsE AUC {pl[l]['TvsE']['pooled']:.3f} (잔차 {pl[l]['TvsE']['pooled_detrended']:.3f})", fontsize=9)
        g.set_xlim(0, 1)
        g.legend(fontsize=7)
    fig.suptitle("TvsE AUC 상위 4개 레이어: 라벨별 희소도 분포 (문제 9개 합침)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig_best_layers.png", dpi=80)
    plt.close(fig)
    print("saved to", out)


if __name__ == "__main__":
    main()
