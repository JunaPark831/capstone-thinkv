"""논문 Algorithm 1 (offline calibration) 그대로. 명세: docs/e3_paper_calibration_spec.md

사용: python calibrate_paper.py --dir DIR --variant full_nosink --out RESULT.json [--T 3]
      python calibrate_paper.py --selftest
DIR 의 *.pt 는 collect_calibration.py 출력(희소도 'sparsity' (V, L, steps), 'variants')이다.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

XS = np.linspace(0, 1, 512)   # KDE 평가 격자 (명세 3절)


def kde_density(x, bw_mult=1.0):
    """가우시안 커널, 폭 = Scott × bw_mult (기본 1 = scipy 기본). 분산 0 등 KDE가 안 되면 None."""
    try:
        k = gaussian_kde(x)
        if bw_mult != 1.0:
            k.set_bandwidth(k.factor * bw_mult)
        return k(XS)
    except (np.linalg.LinAlgError, ValueError):
        return None


def modes_and_valleys(x):
    """Alg. 18행: 내부 국소 극대(prominence 없음). 27행: 연속한 두 모드 사이 KDE 최솟값."""
    d = kde_density(x)
    if d is None:
        return [], []
    m = find_peaks(d)[0]
    valleys = [float(XS[a + int(np.argmin(d[a:b + 1]))]) for a, b in zip(m[:-1], m[1:])]
    return [float(XS[i]) for i in m], valleys


def calibrate(series, T=3, skip=0):
    """series: 프롬프트별 (L, steps) 배열 리스트 (한 변형). Algorithm 1 의 14~30행."""
    P, L = len(series), series[0].shape[0]
    n_modes = np.zeros((P, L), int)
    valleys = [[None] * L for _ in range(P)]
    for p, arr in enumerate(series):                      # 4~12행의 수집은 끝난 상태. 14~23행
        for l in range(L):
            mo, va = modes_and_valleys(arr[l, skip:].astype(np.float64))   # skip=0 이 논문 문자 그대로(모든 스텝)
            n_modes[p, l] = len(mo)
            valleys[p][l] = va
    layer_sets = [{l for l in range(L) if n_modes[p, l] == T} for p in range(P)]
    lstar = sorted(set.intersection(*layer_sets)) if layer_sets else []        # 24행
    theta = None
    if lstar:                                                                   # 25~30행
        theta = [float(np.mean([valleys[p][l][j] for l in lstar for p in range(P)])) for j in range(T - 1)]
    return {"P": P, "L": L, "T": T, "L_star": lstar, "theta": theta,
            "status": "OK" if lstar else "FAIL_EMPTY_INTERSECTION",
            "layer_prompts_exactly_T": [int((n_modes[:, l] == T).sum()) for l in range(L)],
            "mode_count_histogram": {int(k): int((n_modes == k).sum()) for k in np.unique(n_modes)},
            "n_modes": n_modes.tolist()}


def load_series(dirpath, variant):
    files = sorted(Path(dirpath).glob("*.pt"))
    series, meta = [], []
    for f in files:
        r = torch.load(f, weights_only=True)   # 원격 인스턴스에서 받은 파일이라 pickle 코드 실행을 막는다 (텐서·숫자·문자열만 있음)
        series.append(r["sparsity"][r["variants"].index(variant)].float().numpy())
        meta.append({"file": f.name, "steps": series[-1].shape[1], "ended_by_eos": r.get("ended_by_eos")})
    return series, meta


def selftest():
    """합성 데이터: 레이어 0~2 는 모든 프롬프트에서 3수준(0.45/0.70/0.90), 레이어 3 은 일부 프롬프트만,
    나머지는 1봉. L* 가 [0,1,2] 이고 θ 가 두 골 안에 있어야 한다."""
    rng = np.random.default_rng(0)

    def banded(n=6000):
        out, lv = [], rng.permutation([0.45, 0.70, 0.90] * 30)
        for v in lv:
            out.append(rng.normal(v, 0.03, rng.integers(60, 120)))
        return np.clip(np.concatenate(out)[:n], 0, 1)

    P, L, n = 6, 8, 6000
    series = []
    for p in range(P):
        arr = np.stack([banded() if l < 3 or (l == 3 and p < 3) else np.clip(rng.normal(0.8, 0.05, n), 0, 1) for l in range(L)])
        series.append(arr)
    res = calibrate(series)
    assert res["L_star"] == [0, 1, 2], res["L_star"]
    t1, t2 = res["theta"]
    assert 0.50 < t1 < 0.65 and 0.75 < t2 < 0.85, res["theta"]
    # 교집합이 비는 경우: 레이어 0 을 한 프롬프트에서 1봉으로 망가뜨리면 L* 에서 빠져야 한다
    series[0][0] = np.clip(rng.normal(0.8, 0.05, n), 0, 1)
    assert calibrate(series)["L_star"] == [1, 2]
    print("selftest ok:", res["L_star"], [round(t, 3) for t in res["theta"]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--variant")
    ap.add_argument("--out")
    ap.add_argument("--T", type=int, default=3)
    ap.add_argument("--skip-first", type=int, default=0, help="KDE 에서 제외할 앞쪽 스텝 수. 0 = 논문 문자 그대로")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        raise SystemExit
    series, meta = load_series(a.dir, a.variant)
    res = calibrate(series, a.T, a.skip_first)
    res.update({"variant": a.variant, "skip_first": a.skip_first, "files": meta, "labels_low_to_high": ["Execution", "Reasoning", "Transition"],
                "n_ended_by_eos": sum(bool(m["ended_by_eos"]) for m in meta)})
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    top = sorted(range(res["L"]), key=lambda l: -res["layer_prompts_exactly_T"][l])[:8]
    print(f"[{a.variant}] P={res['P']} status={res['status']} L*={res['L_star']} theta={res['theta']}")
    print("mode-count histogram (prompt×layer):", res["mode_count_histogram"])
    print("layers by #prompts with exactly T modes:", [(l, res["layer_prompts_exactly_T"][l]) for l in top])
