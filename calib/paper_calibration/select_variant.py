"""win256 / win1024 중 봉우리가 더 많은 쪽을 고르고(변형 단위, 레이어 단위) Algorithm 1 을 다시 적용한다.
사전 기록: docs/criteria/e3_paper_calibration_spec.md "봉우리가 더 많은 변형 고르기".

사용: python -I select_variant.py --dir DATA_DIR --out RESULT.json [--bw-mult 1.0]
봉우리 = find_peaks(d/d.max(), prominence=p), KDE = 가우시안 Scott x bw-mult, 512점 격자(calibrate_paper.py 와 같음).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_paper import XS, kde_density  # noqa: E402

VARS = ["win1024", "win256"]            # 동률이면 앞의 것(win1024)을 고른다
PROMS = [0.0, 0.02, 0.05]
SKIPS = [0, 1100]


def peaks(d, prom):
    dn = d / d.max()
    return find_peaks(dn, prominence=prom)[0] if prom > 0 else find_peaks(dn)[0]


def lstar_report(n, valleys=None):
    """n: (P, L) 봉우리 수 행렬 -> L*, 2/3 규칙 레이어, 같은 k>=2 레이어, 레이어별 최대 문제수."""
    P, L = n.shape
    cnt = [(n[:, l] == 3).sum() for l in range(L)]
    lstar = [l for l in range(L) if cnt[l] == P]
    return {"L_star": lstar, "layers_ge_2_3": [l for l in range(L) if cnt[l] >= int(np.ceil(2 * P / 3))],
            "max_prompts_exactly3": int(max(cnt)),
            "same_k_ge2_layers": {l: int(n[0, l]) for l in range(L) if len(set(n[:, l])) == 1 and n[0, l] >= 2},
            "share_1": float((n == 1).mean()), "n_exactly3": int((n == 3).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bw-mult", type=float, default=1.0)
    a = ap.parse_args()
    files = sorted(Path(a.dir).glob("*.pt"))
    data = [torch.load(f, weights_only=True) for f in files]   # 원격에서 받은 파일이라 pickle 실행을 막는다
    P, L = len(data), data[0]["sparsity"].shape[1]
    res = {}
    for skip in SKIPS:
        cnt = {v: {p: np.zeros((P, L), int) for p in PROMS} for v in VARS}
        val = {v: {} for v in VARS}      # (문제, 레이어) -> prominence 0.05 에서 정확히 3봉일 때의 골
        for v in VARS:
            vi = data[0]["variants"].index(v)
            for i, r in enumerate(data):
                arr = r["sparsity"][vi].float().numpy()
                for l in range(L):
                    d = kde_density(arr[l, skip:].astype(np.float64), a.bw_mult)
                    if d is None:
                        continue
                    for p in PROMS:
                        pk = peaks(d, p)
                        cnt[v][p][i, l] = len(pk)
                        if p == 0.05 and len(pk) == 3:
                            val[v][(i, l)] = [float(XS[x + int(np.argmin(d[x:y + 1]))]) for x, y in zip(pk[:-1], pk[1:])]
        out = {"by_variant": {}, "choice": {}}
        for p in PROMS:
            tot = {v: int(cnt[v][p].sum()) for v in VARS}
            best = max(VARS, key=lambda v: (tot[v], -VARS.index(v)))   # 동률이면 win1024
            out["by_variant"][str(p)] = {"total_peaks": tot, "chosen_variant": best, **{v: lstar_report(cnt[v][p]) for v in VARS}}
            # 레이어 단위: 레이어마다 문제들에 걸친 봉우리 수 합이 더 큰 변형 (같은 레이어는 모든 문제에서 같은 변형)
            pick = {l: max(VARS, key=lambda v: (int(cnt[v][p][:, l].sum()), -VARS.index(v))) for l in range(L)}
            comb = np.stack([cnt[pick[l]][p][:, l] for l in range(L)], axis=1)
            rep = lstar_report(comb)
            rep["layer_variant"] = pick
            rep["n_layers_picked_win256"] = sum(1 for l in range(L) if pick[l] == "win256")
            if p == 0.05 and rep["L_star"]:
                rep["theta"] = [float(np.mean([val[pick[l]][(i, l)][j] for l in rep["L_star"] for i in range(P)])) for j in range(2)]
            out["choice"][str(p)] = rep
        res[f"skip{skip}"] = out

    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"P={P}, KDE Scott x{a.bw_mult}, 변형 {VARS}")
    for skip in SKIPS:
        o = res[f"skip{skip}"]
        print(f"\n== 앞 {skip}스텝 제외")
        for p in PROMS:
            bv = o["by_variant"][str(p)]
            ch = o["choice"][str(p)]
            print(f" prominence >= {p}: 봉우리 수 합 {bv['total_peaks']} → 변형 단위 선택 {bv['chosen_variant']}")
            for v in VARS:
                r = bv[v]
                print(f"     {v:8s} |L*|={len(r['L_star'])} 최대 {r['max_prompts_exactly3']}/{P} 6/9이상 {r['layers_ge_2_3']} 1봉 {r['share_1'] * 100:.0f}% "
                      f"3봉 시계열 {r['n_exactly3']}개 같은 k>=2 {r['same_k_ge2_layers'] or '-'}")
            print(f"     레이어 단위 선택(win256 {ch['n_layers_picked_win256']}개 레이어): |L*|={len(ch['L_star'])} 최대 {ch['max_prompts_exactly3']}/{P} "
                  f"6/9이상 {ch['layers_ge_2_3']} 1봉 {ch['share_1'] * 100:.0f}% 3봉 시계열 {ch['n_exactly3']}개 같은 k>=2 {ch['same_k_ge2_layers'] or '-'}")
            if ch.get("theta"):
                print(f"     ** L*={ch['L_star']} theta={ch['theta']}")
    print("\nsaved", a.out)


if __name__ == "__main__":
    main()
