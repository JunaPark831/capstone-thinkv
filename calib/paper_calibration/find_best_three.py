"""모든 (채널, 문제, 레이어)에서 '적절한 3봉'을 찾아 순위를 매기고 갤러리를 만든다. 사전 기록: spec "단일 문제 3봉 후보 탐색".
사용: HF_HUB_OFFLINE=1 python -I find_best_three.py --out OUT_DIR
"""
import argparse, html, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "DejaVu Sans"]; matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np, torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_paper import kde_density, XS
from peaklike import peaks_by_mass, valleys_between
from keyword_signal import label_tokens, token_text, LAB
MASKED = "calib/sparsity/s1k_llama8b_p15_masked"; THRESH = "calib/sparsity/s1k_llama8b_p15_thresholds"
MIN_SHARE, MIN_GAP, MIN_POS = 0.10, 0.08, 0.10
COL = ["#2f6fed", "#1f7a4d", "#d9630a"]

def scan(job):
    src, ch = job
    out = []
    for f in sorted(Path(src).glob("*.pt")):
        r = torch.load(f, weights_only=True); arr = r["sparsity"][r["variants"].index(ch)].float().numpy()
        for l in range(arr.shape[0]):
            x = arr[l].astype(np.float64); d = kde_density(x)
            if d is None: continue
            pk = peaks_by_mass(d, 0.02, 0.05)
            if len(pk) != 3: continue
            va = valleys_between(d, pk); b = np.digitize(x, va); n = len(x)
            shares = [float((b == k).mean()) for k in range(3)]
            posm = [float(np.median(np.where(b == k)[0]) / n) if (b == k).any() else 0.0 for k in range(3)]
            gaps = [float(XS[pk[1]] - XS[pk[0]]), float(XS[pk[2]] - XS[pk[1]])]
            dn = d / d.max(); dips = [float(1 - dn[a:c + 1].min() / min(dn[a], dn[c])) for a, c in zip(pk[:-1], pk[1:])]
            rho = float(spearmanr(np.arange(n), x)[0])
            out.append({"src": src, "channel": ch, "prompt": f.stem, "layer": l, "peaks": [float(XS[i]) for i in pk], "valleys": va,
                        "shares": shares, "pos_median": posm, "gaps": gaps, "dips": dips, "rho_step": rho, "n": n,
                        "ok": min(shares) >= MIN_SHARE and min(gaps) >= MIN_GAP and min(posm) >= MIN_POS})
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--top", type=int, default=12); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    chm = [c for c in torch.load(next(Path(MASKED).glob("*.pt")), weights_only=True)["variants"] if c != "sink_mass"]
    jobs = [(MASKED, c) for c in chm] + [(THRESH, f"full_literal@{p}") for p in (1, 2, 5, 10)]
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = [c for part in ex.map(scan, jobs) for c in part]
    ok = sorted([c for c in res if c["ok"]], key=lambda c: -min(c["shares"]))
    print(f"정확히 3봉 셀 {len(res)}개 중 '적절' 기준 통과 {len(ok)}개 (구역 질량 ≥{MIN_SHARE}, 간격 ≥{MIN_GAP}, 위치 중앙값 ≥{MIN_POS})")
    # 키워드 라벨
    tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    labs = {}
    for f in sorted(Path(MASKED).glob("*.pt")):
        r = torch.load(f, weights_only=True); text, starts, _ = token_text(tok, r["token_ids"])
        labs[f.stem] = label_tokens(text, starts, len(r["token_ids"]), 64, "A")[0][:-1]
    print(f"{'순위':3s} {'채널':18s} {'문제':5s} {'L':3s} {'봉우리':22s} {'골':14s} {'구역 질량':18s} {'위치중앙값':18s} {'골깊이':12s} {'ρ(스텝)':8s} {'구역별 T 비율':18s}")
    top = ok[: a.top]
    for i, c in enumerate(top):
        r = torch.load(Path(c["src"]) / f"{c['prompt']}.pt", weights_only=True); x = r["sparsity"][r["variants"].index(c["channel"]), c["layer"]].float().numpy()
        b = np.digitize(x, c["valleys"]); lab = labs[c["prompt"]]
        tfrac = [float((lab[b == k] == LAB["T"]).mean()) if (b == k).any() else 0 for k in range(3)]; c["T_frac"] = tfrac
        flag = " 위치추세" if abs(c["rho_step"]) > 0.6 else ""
        print(f"{i+1:3d} {c['channel']:18s} {c['prompt']:5s} {c['layer']:<3d} {str([round(p,2) for p in c['peaks']]):22s} {str([round(v,2) for v in c['valleys']]):14s} {str([round(s,2) for s in c['shares']]):18s} {str([round(p,2) for p in c['pos_median']]):18s} {str([round(v,2) for v in c['dips']]):12s} {c['rho_step']:+.2f}{flag:6s} {str([round(t,2) for t in tfrac]):18s}")
    (out / "candidates.json").write_text(json.dumps({"all_exactly3": res, "ok_sorted": ok}, indent=1))
    # 갤러리: 후보마다 분포 + 시계열
    body = ["<!doctype html><meta charset='utf-8'><title>단일 문제 3봉 후보</title><style>body{font:15px/1.6 -apple-system,'Apple SD Gothic Neo',sans-serif;margin:24px;max-width:1500px}img{width:100%;border:1px solid #ccc;margin:6px 0 20px}</style>",
            f"<h1>'적절한 3봉' 후보 상위 {len(top)}개 (R3 봉우리, Scott x1) — 왼쪽: 분포(골=초록 점선), 오른쪽: 시계열(띠 색 = 세 구역, 검은 선 = 128스텝 이동평균)</h1>"]
    for i, c in enumerate(top):
        r = torch.load(Path(c["src"]) / f"{c['prompt']}.pt", weights_only=True); x = r["sparsity"][r["variants"].index(c["channel"]), c["layer"]].float().numpy()
        d = kde_density(x.astype(np.float64)); v1, v2 = c["valleys"]
        fig, ax = plt.subplots(1, 2, figsize=(16, 3.8), gridspec_kw={"width_ratios": [1, 2.2]})
        g = ax[0]; g.hist(x, bins=70, range=(0, 1), density=True, alpha=0.45, color="#7aa5ff"); g.plot(XS, d, color="#1f3f9f", lw=1.5)
        for p in c["peaks"]: g.plot([p], [np.interp(p, XS, d)], "o", color="#d9630a", ms=7)
        for v in (v1, v2): g.axvline(v, color="#1f7a4d", ls="--", lw=1.5)
        g.set_xlim(0, 1); g.set_title(f"#{i+1} {c['channel']} 문제 {c['prompt']} L{c['layer']} | 구역 질량 {[round(s,2) for s in c['shares']]}", fontsize=9)
        g = ax[1]; n = len(x); lo, hi = 0, 1
        g.axhspan(lo, v1, color=COL[0], alpha=0.12); g.axhspan(v1, v2, color=COL[1], alpha=0.12); g.axhspan(v2, hi, color=COL[2], alpha=0.12)
        g.plot(x, lw=0.3, color="#555", alpha=0.5); g.plot(np.arange(127, n), np.convolve(x, np.ones(128) / 128, "valid"), lw=1.1, color="black")
        g.axhline(v1, color=COL[0], lw=1); g.axhline(v2, color=COL[2], lw=1); g.set_ylim(0, 1)
        g.set_title(f"시계열 | 스텝 상관 {c['rho_step']:+.2f} | 구역 위치 중앙값 {[round(p,2) for p in c['pos_median']]} | 구역별 T 라벨 비율 {[round(t,2) for t in c['T_frac']]}", fontsize=9)
        fig.tight_layout(); fn = f"cand_{i+1:02d}.png"; fig.savefig(out / fn, dpi=70); plt.close(fig)
        body.append(f"<h3>#{i+1} {html.escape(c['channel'])} · 문제 {c['prompt']} · 레이어 {c['layer']} · 봉우리 {[round(p,2) for p in c['peaks']]} · 골 {[round(v,2) for v in c['valleys']]}</h3><img src='{fn}'>")
    (out / "index.html").write_text("\n".join(body), encoding="utf-8"); print("saved to", out)

if __name__ == "__main__":
    main()
