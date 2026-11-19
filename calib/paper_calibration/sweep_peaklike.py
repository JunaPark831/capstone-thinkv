"""봉우리다움 필터 단계별로 Algorithm 1 을 다시 적용한다 (win256 / win1024, 변형 단위·레이어 단위 선택).
사전 기록: docs/e3_paper_calibration_spec.md "봉우리다운 것만".

사용: python -I sweep_peaklike.py --dir DATA_DIR --out RESULT.json [--bw 0.5,1.0]
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_paper import kde_density  # noqa: E402
from peaklike import peaklike  # noqa: E402
from select_variant import lstar_report  # noqa: E402

VARS = ["win1024", "win256"]          # 동률이면 앞의 것
SKIPS = [0, 1100]
LEVELS = {"기준(prom0.02)": (0.02, 0, 0), "L1": (0.02, 0.05, 0.15), "L2(주)": (0.05, 0.05, 0.30), "L3": (0.05, 0.10, 0.50)}
PRIMARY = "L2(주)"


def one(job):
    dirpath, variant, skip, bw = job
    series = []
    for f in sorted(Path(dirpath).glob("*.pt")):
        r = torch.load(f, weights_only=True)   # 원격에서 받은 파일이라 pickle 실행을 막는다
        series.append(r["sparsity"][r["variants"].index(variant)].float().numpy())
    P, L = len(series), series[0].shape[0]
    cnt = {k: np.zeros((P, L), int) for k in LEVELS}
    for i, arr in enumerate(series):
        for l in range(L):
            d = kde_density(arr[l, skip:].astype(np.float64), bw)
            if d is None:
                continue
            for k, (p, ds, dp) in LEVELS.items():
                cnt[k][i, l] = len(peaklike(d, p, ds, dp))
    return bw, variant, skip, {k: v.tolist() for k, v in cnt.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bw", default="0.5,1.0")
    a = ap.parse_args()
    bws = [float(x) for x in a.bw.split(",")]
    jobs = [(a.dir, v, s, b) for b in bws for v in VARS for s in SKIPS]
    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(one, jobs))
    cnt = {(b, v, s): {k: np.array(m) for k, m in o.items()} for b, v, s, o in results}
    P = next(iter(cnt.values()))[PRIMARY].shape[0]
    L = next(iter(cnt.values()))[PRIMARY].shape[1]
    out = {}
    for b in bws:
        for s in SKIPS:
            print(f"\n===== Scott x{b} | 앞 {s}스텝 제외 | P={P} =====")
            for k in LEVELS:
                tot = {v: int(cnt[(b, v, s)][k].sum()) for v in VARS}
                best = max(VARS, key=lambda v: (tot[v], -VARS.index(v)))
                pick = {l: max(VARS, key=lambda v: (int(cnt[(b, v, s)][k][:, l].sum()), -VARS.index(v))) for l in range(L)}
                comb = np.stack([cnt[(b, pick[l], s)][k][:, l] for l in range(L)], axis=1)
                rep_c = lstar_report(comb)
                tag = " <= 주 단계" if k == PRIMARY else ""
                print(f" [{k}]{tag} 봉우리 수 합 {tot} → 변형 단위 선택 {best}")
                rec = {"total_peaks": tot, "chosen_variant": best, "layer_level": {**rep_c, "n_layers_picked_win256": sum(1 for l in range(L) if pick[l] == "win256")}}
                for v in VARS:
                    r = lstar_report(cnt[(b, v, s)][k])
                    rec[v] = r
                    print(f"     {v:8s} |L*|={len(r['L_star'])} 최대 {r['max_prompts_exactly3']}/{P} 6/9이상 {r['layers_ge_2_3']} "
                          f"1봉 {r['share_1'] * 100:3.0f}% 3봉 시계열 {r['n_exactly3']:3d}개 같은 k>=2 {r['same_k_ge2_layers'] or '-'}")
                print(f"     레이어 단위 선택(win256 {rec['layer_level']['n_layers_picked_win256']}개 레이어): |L*|={len(rep_c['L_star'])} 최대 {rep_c['max_prompts_exactly3']}/{P} "
                      f"6/9이상 {rep_c['layers_ge_2_3']} 1봉 {rep_c['share_1'] * 100:3.0f}% 3봉 시계열 {rep_c['n_exactly3']}개 같은 k>=2 {rep_c['same_k_ge2_layers'] or '-'}")
                out[f"bw{b}|skip{s}|{k}"] = rec
    Path(a.out).write_text(json.dumps({"levels": {k: list(v) for k, v in LEVELS.items()}, "results": out}, indent=1))
    print("\nsaved", a.out)


if __name__ == "__main__":
    main()
