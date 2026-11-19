"""사용자 결정에 따른 대체 캘리브레이션: 정확히 3봉이 3개 문제에서 겹친 (채널, 레이어)를 L*로, 그 쌍의 골 평균을 θ로.
사용: python -I finalize_calibration.py --dir MASKED_DIR --out calibration_chosen.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_paper import kde_density, XS
from peaklike import peaks_by_mass, valleys_between
RULE = {"kde": "gaussian Scott x1, 512 grid", "peaks": "mass>=0.02, distance>=0.05, no height condition (R3)"}
CANDS = {"primary": ("full_mid@max1", [5, 24]), "reference": ("win1024@max1", [0])}
LABELS = ["Execution", "Reasoning", "Transition"]   # 희소도 낮은 순
TAU = 128

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    files = sorted(Path(a.dir).glob("*.pt")); names = [f.stem for f in files]; data = [torch.load(f, weights_only=True) for f in files]
    out = {"rule": RULE, "tau": TAU, "labels_low_to_high": LABELS, "source_dir": a.dir, "prompts": names, "calibrations": {}}
    for key, (ch, lstar) in CANDS.items():
        ci = data[0]["variants"].index(ch)
        pairs = []
        for l in lstar:
            for i, r in enumerate(data):
                d = kde_density(r["sparsity"][ci, l].float().numpy().astype(np.float64))
                pk = peaks_by_mass(d, 0.02, 0.05)
                if len(pk) == 3:
                    pairs.append({"layer": l, "prompt": names[i], "peaks": [round(float(XS[k]), 4) for k in pk], "valleys": [round(v, 4) for v in valleys_between(d, pk)]})
        theta = [float(np.mean([p["valleys"][j] for p in pairs])) for j in range(2)]
        # 디코드 모의: φ = L* 평균 희소도, τ=128 블록 평균으로 분류
        sim = {}
        tot = np.zeros(3)
        for i, r in enumerate(data):
            phi = r["sparsity"][ci][lstar].float().numpy().mean(0)
            nb = len(phi) // TAU
            blk = phi[: nb * TAU].reshape(nb, TAU).mean(1) if nb else phi[None].mean(1)
            lab = np.digitize(blk, theta)
            frac = [float((lab == k).mean()) for k in range(3)]
            first = [int(np.argmax(lab == k)) if (lab == k).any() else None for k in range(3)]
            sim[names[i]] = {"blocks": int(len(lab)), "frac_E_R_T": frac, "switches": int((np.diff(lab) != 0).sum()),
                             "first_block_E_R_T": first, "last_nonT_block": int(np.where(lab != 2)[0].max()) if (lab != 2).any() else None}
            tot += np.array(frac) * len(lab)
        out["calibrations"][key] = {"channel": ch, "range": "positions 4..g-8 (sink block 0-3 and last 8 excluded)" if "mid" in ch else "positions max(1,g-1023)..g",
                                    "L_star": lstar, "theta": theta, "pairs": pairs, "n_pairs": len(pairs),
                                    "decode_sim": sim, "overall_frac_E_R_T": (tot / tot.sum()).tolist()}
        print(f"\n[{key}] {ch} L*={lstar}  θ1={theta[0]:.4f} θ2={theta[1]:.4f}  (쌍 {len(pairs)}개)")
        for p in pairs:
            print(f"   L{p['layer']} 문제 {p['prompt']}: 봉우리 {p['peaks']} 골 {p['valleys']}")
        print(f"   디코드 모의(τ=128 블록, L* 평균): 전체 E/R/T 비율 {[round(x, 3) for x in out['calibrations'][key]['overall_frac_E_R_T']]}")
        for n, s in sim.items():
            print(f"     {n}: 블록 {s['blocks']:4d}  E/R/T {[round(x, 2) for x in s['frac_E_R_T']]}  전환 {s['switches']:3d}회  마지막 비-T 블록 {s['last_nonT_block']}")
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("\nsaved", a.out)

if __name__ == "__main__":
    main()
