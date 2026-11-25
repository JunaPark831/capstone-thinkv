"""레이어마다 9개 문제를 이어 붙인(pooled) 희소도 분포에서 봉우리를 세고, 합침 때문에 생기는 가짜 봉우리를 점검한다.
사전 기록: docs/criteria/e3_paper_calibration_spec.md "문제 전체 합친(pooled) 분포의 3봉 확인".

사용: HF_HUB_OFFLINE=1 python -I pooled_peaks.py --dir THRESH_DIR --out OUT_DIR
산출: result.json, 표(stdout), 갤러리 index.html + 채널별 32레이어 격자 그림
"""
import argparse
import html
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
from scipy.signal import find_peaks
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_paper import XS, kde_density  # noqa: E402
from keyword_signal import LAB, label_tokens, token_text  # noqa: E402
from peaklike import peaklike, valleys_between  # noqa: E402

VARS = ["win1024", "win256", "full_nosink", "full_literal"]
PCTS = [1, 2, 5, 10]
SKIPS = [0, 1100]
L2 = (0.05, 0.05, 0.30)
PRIMARY = ["win1024@1", "win256@1", "full_nosink@1"]
NAMES3 = ["E", "R", "T"]


def basins(x, valleys):
    return np.digitize(x, valleys)   # 0..k-1 구역


def analyze_channel(job):
    dirpath, ch, skip = job
    files = sorted(Path(dirpath).glob("*.pt"))
    series, pid, pos = [], [], []
    for i, f in enumerate(files):
        r = torch.load(f, weights_only=True)
        arr = r["sparsity"][r["variants"].index(ch)].float().numpy()[:, skip:]
        series.append(arr)
        n = arr.shape[1]
        pid.append(np.full(n, i))
        pos.append((np.arange(n) + skip) / (n + skip))
    pid, pos = np.concatenate(pid), np.concatenate(pos)
    L = series[0].shape[0]
    rows = []
    for l in range(L):
        x = np.concatenate([s[l] for s in series]).astype(np.float64)
        d = kde_density(x)
        if d is None:
            rows.append({"layer": l, "n_raw": 0, "n_L2": 0})
            continue
        raw = find_peaks(d / d.max())[0]
        pk = peaklike(d, *L2)
        va = valleys_between(d, pk)
        row = {"layer": l, "n_raw": int(len(raw)), "n_L2": int(len(pk)), "peaks": [float(XS[i]) for i in pk], "valleys": va, "basins": []}
        if len(pk) >= 2:
            b = basins(x, va)
            for k in range(len(pk)):
                m = b == k
                comp = np.bincount(pid[m], minlength=len(files)) / max(1, m.sum())
                row["basins"].append({"share": float(m.mean()), "top_prompt_share": float(comp.max()), "top_prompt": int(comp.argmax()),
                                      "pos_median": float(np.median(pos[m])) if m.any() else None, "idx_mask_sum": int(m.sum())})
            row["flag_prompt"] = any(bb["top_prompt_share"] >= 0.70 for bb in row["basins"])
            row["flag_warmup"] = any(bb["pos_median"] is not None and bb["pos_median"] < 0.10 for bb in row["basins"])
        rows.append(row)
    return ch, skip, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--vars", default=None)
    ap.add_argument("--pcts", default=None)
    ap.add_argument("--primary", default=None)
    a = ap.parse_args()
    global VARS, PCTS, PRIMARY
    if a.vars:
        VARS = a.vars.split(",")
    if a.pcts:
        PCTS = a.pcts.split(",")
    if a.primary:
        PRIMARY = a.primary.split(",")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    chans = [f"{v}@{p}" for v in VARS for p in PCTS]
    jobs = [(a.dir, ch, s) for ch in chans for s in SKIPS]
    with ProcessPoolExecutor(max_workers=8) as ex:
        results = {(ch, s): rows for ch, s, rows in ex.map(analyze_channel, jobs)}

    files = sorted(Path(a.dir).glob("*.pt"))
    names = [f.stem for f in files]
    data = [torch.load(f, weights_only=True) for f in files]
    tok = AutoTokenizer.from_pretrained(a.model)
    labs = []
    for r in data:
        text, starts, _ = token_text(tok, r["token_ids"])
        lab, _, _ = label_tokens(text, starts, len(r["token_ids"]), 64, "A")
        labs.append(lab[:-1])

    print(f"P={len(files)} | 합친 분포, KDE Scott x1, 봉우리다움 L2={L2}. 각 칸: L2 정확히3봉 레이어 수(깨끗한 것) / ≥2봉 레이어 수 / 필터 없음 봉우리 중앙값")
    summary, candidates = {}, []
    for s in SKIPS:
        print(f"\n== 앞 {s}스텝 제외")
        for ch in chans:
            rows = results[(ch, s)]
            n3 = [r for r in rows if r.get("n_L2") == 3]
            clean3 = [r for r in n3 if not r.get("flag_prompt") and not r.get("flag_warmup")]
            n2 = [r for r in rows if r.get("n_L2", 0) >= 2]
            med_raw = float(np.median([r.get("n_raw", 0) for r in rows]))
            summary[f"{ch}|skip{s}"] = {"exactly3_L2": [r["layer"] for r in n3], "clean3_L2": [r["layer"] for r in clean3], "ge2_L2": [r["layer"] for r in n2], "median_raw": med_raw}
            flag = ""
            if n3:
                flag = " | 3봉 레이어 " + ", ".join(f"L{r['layer']}" + ("(문제편중)" if r.get("flag_prompt") else "") + ("(초반)" if r.get("flag_warmup") else "") for r in n3)
            print(f"  {ch:16s} 3봉 {len(n3)}개(깨끗 {len(clean3)}개) / ≥2봉 {len(n2)}개 / 필터없음 중앙값 {med_raw:.0f}{flag}")
            for r in clean3:
                candidates.append((ch, s, r))
    (out / "result.json").write_text(json.dumps({"L2": L2, "summary": summary, "rows": {f"{ch}|skip{s}": rows for (ch, s), rows in results.items()}}, indent=1))

    # 후보 상세: 구역별 문제 구성·위치·키워드 라벨
    print("\n== 깨끗한 3봉 후보 상세 (구역별: 점유율, 최다 문제 비율, 위치 중앙값, 키워드 라벨 E/R/T 비율)")
    for ch, s, r in candidates:
        l = r["layer"]
        x = np.concatenate([d["sparsity"][d["variants"].index(ch)].float().numpy()[l, s:] for d in data]).astype(np.float64)
        lab = np.concatenate([lb[s:] for lb in labs])
        b = basins(x, r["valleys"])
        parts = []
        for k, bb in enumerate(r["basins"]):
            m = b == k
            lc = [float((lab[m] == LAB[t]).mean()) for t in NAMES3]
            parts.append(f"구역{k}(봉 {r['peaks'][k]:.2f}): 점유 {bb['share']:.2f}, 최다문제 {names[bb['top_prompt']]} {bb['top_prompt_share']:.2f}, 위치 {bb['pos_median']:.2f}, E/R/T {lc[0]:.2f}/{lc[1]:.2f}/{lc[2]:.2f}")
        print(f"  {ch} 앞{s} L{l}: " + " | ".join(parts))
    if not candidates:
        print("  (없음)")

    # 그림: 주 채널 + 후보 채널, 32레이어 격자, 문제별 색 누적 히스토그램 + KDE + L2 봉우리
    to_plot = []
    for s in SKIPS:
        for ch in PRIMARY:
            to_plot.append((ch, s))
    for ch, s, _ in candidates:
        if (ch, s) not in to_plot:
            to_plot.append((ch, s))
    cmap = plt.get_cmap("tab10")
    body = ["<!doctype html><meta charset='utf-8'><title>합친 분포 봉우리</title>",
            "<style>body{font:15px/1.6 -apple-system,'Apple SD Gothic Neo',sans-serif;margin:24px;max-width:1700px}img{width:100%;border:1px solid #ccc;margin:6px 0 22px}</style>",
            f"<h1>레이어마다 9개 문제를 합친 희소도 분포 (KDE Scott x1, 봉우리다움 L2={L2})</h1><p>막대는 문제별로 색을 나눈 누적 히스토그램(어느 문제에서 온 질량인지), 선은 합친 KDE, 주황 점은 L2 봉우리, 초록 점선은 골. 제목이 빨간 레이어는 L2 봉우리 2개 이상.</p>"]
    for ch, s in to_plot:
        rows = results[(ch, s)]
        ci = data[0]["variants"].index(ch)
        fig, ax = plt.subplots(4, 8, figsize=(24, 10.5))
        for l in range(32):
            g = ax[l // 8, l % 8]
            per = [d["sparsity"][ci].float().numpy()[l, s:] for d in data]
            g.hist(per, bins=60, range=(0, 1), stacked=True, color=[cmap(i % 10) for i in range(len(per))], alpha=0.75, label=names if l == 0 else None)
            x = np.concatenate(per).astype(np.float64)
            d = kde_density(x)
            if d is not None:
                ymax = g.get_ylim()[1]
                g.plot(XS, d / d.max() * ymax * 0.95, lw=1.3, color="#1f3f9f")
                r = rows[l]
                for pk_x in r.get("peaks", []):
                    g.plot([pk_x], [np.interp(pk_x, XS, d) / d.max() * ymax * 0.95], "o", color="#d9630a", ms=6)
                for v in r.get("valleys", []):
                    g.axvline(v, color="#1f7a4d", ls="--", lw=1)
            r = rows[l]
            tag = "".join(["(문제편중)" if r.get("flag_prompt") else "", "(초반)" if r.get("flag_warmup") else ""])
            g.set_title(f"L{l} L2 봉우리 {r.get('n_L2', 0)}개 (요철 포함 {r.get('n_raw', 0)}){tag}", fontsize=8, color="#c00000" if r.get("n_L2", 0) >= 2 else "black")
            g.set_xlim(0, 1)
            g.tick_params(labelsize=6)
            if l == 0:
                g.legend(fontsize=5, ncol=3)
        fig.suptitle(f"{ch} | 앞 {s}스텝 제외 | 9개 문제 합침 | 색=문제", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        fn = f"pooled_{ch.replace('@', '_p')}_skip{s}.png"
        fig.savefig(out / fn, dpi=62)
        plt.close(fig)
        sm = summary[f"{ch}|skip{s}"]
        body.append(f"<h2>{html.escape(ch)} | 앞 {s}스텝 제외 — L2 정확히 3봉 레이어 {sm['exactly3_L2'] or '없음'} (깨끗한 것 {sm['clean3_L2'] or '없음'}), ≥2봉 {sm['ge2_L2'] or '없음'}</h2><img loading='lazy' src='{fn}'>")
    (out / "index.html").write_text("\n".join(body), encoding="utf-8")
    print("\nsaved to", out)


if __name__ == "__main__":
    main()
