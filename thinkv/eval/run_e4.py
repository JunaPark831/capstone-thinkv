"""E4 러너: 설정 9개 × AIME 30문제 × 시드 8. 한 줄 = 한 생성, thinkv/result/{설정}.jsonl 에 바로 append.
사용: python thinkv/eval/run_e4_tbe_tbq.py [--settings thinkv,fullkv] [--seeds 0-7] [--shard 0/1] [--max-new 32768]
      python thinkv/eval/run_e4_tbe_tbq.py --summary
이미 있는 (설정, 문제, 시드)는 건너뛰므로 재개·샤딩(--shard i/N, (문제, 시드) 쌍 기준)이 된다."""

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent        # thinkv/eval
ROOT = HERE.parent.parent                     # 레포 루트
sys.path.insert(0, str(ROOT))                 # import thinkv
#sys.path.append(str(ROOT / "eval"))           # baseline.py 가 레포 루트 eval/ 에 있는 경우

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import thinkv.press as tp                     # import 하면 eager_sparsity 어텐션이 등록된다
from baseline import baseline, get_device     # 프롬프트 템플릿·샘플링 0.6/0.95·math_verify 채점·cache_len
from thinkv.press import ThinKVPress

SETTINGS = {
    "fullkv":   None,
    "kivi2":    dict(quant="kivi2",     budget=None),
    "uniform4": dict(quant="nvfp4_all", budget=None),
    "snapkv":   "snapkv",
    "thinkv":   dict(quant="tbq",  budget=1024, labels="sparsity"),
    "tbq_only": dict(quant="tbq",  budget=None, labels="sparsity"),
    "tbe_only": dict(quant=None,   budget=1024, labels="sparsity"),
    "random":   dict(quant="tbq",  budget=1024, labels="random"),
    "periodic": dict(quant="tbq",  budget=1024, labels="periodic"),
}
BITS = {  # quant 설정 -> 구간 라벨별 비트 (스케일 비용 제외)
    "tbq":       {"R": 4, "E": 4, "T": 2},
    "nvfp4_all": dict.fromkeys("RET", 4),
    "kivi2":     dict.fromkeys("RET", 2),
    None:        dict.fromkeys("RET", 16), #양자화없음
}


def problems():
    """AIME 2024·2025 에서 시드 0으로 15개씩 (논문 E.1). R-KV 저장소에는 aime24 전체 30개 파일뿐이라 같은 방식으로 뽑는다."""
    out = []
    for ds, split, year in (("HuggingFaceH4/aime_2024", "train", 2024), ("math-ai/aime25", "test", 2025)):
        rows = list(load_dataset(ds)[split])
        for i in sorted(random.Random(0).sample(range(len(rows)), 15)):
            out.append({"unique_id": f"aime{year}/{rows[i]['id']}", "answer": str(rows[i]["answer"]),
                        "question": rows[i]["problem"] + "\nRemember to put your final answer within \\boxed{}."})
    return out


def make_press(name, seed):
    cfg = SETTINGS[name]
    if cfg is None:
        return None
    if cfg == "snapkv":
        from kvpress import DecodingPress, SnapKVPress
        return DecodingPress(SnapKVPress(window_size=64), compression_interval=128, target_size=1024)
    return ThinKVPress(seed=seed, **cfg)


def log_summary(press):
    """press 가 남긴 로그를 한 줄 요약으로. 로그가 없는 설정(fullkv·snapkv)은 16비트."""
    lg = getattr(press, "last_log", None)
    if not lg:
        return {"avg_bits": 16.0}
    cl, bit = lg["cache_len"], BITS[press.quant]
    toks = [min(press.tau, len(cl) - i * press.tau) for i in range(len(lg["labels"]))]   # 구간별 생성 토큰 수
    return {"labels": "".join(lg["labels"]),
            "phi_block": [None if x != x else round(x, 4) for x in lg["phi_block"]],
            "mean_cache_len": sum(cl) / max(len(cl), 1),
            "avg_bits": sum(bit[x] * n for x, n in zip(lg["labels"], toks)) / max(sum(toks), 1),
            "n_evictions": sum(b < a + 1 for a, b in zip(cl, cl[1:]))}


def finished(path):
    """이미 끝난 (문제, 시드) 집합."""
    if not path.exists():
        return set()
    return {(r["id"], r["seed"]) for r in map(json.loads, filter(str.strip, path.open()))}


def summarize(out):
    import pandas as pd
    df = pd.concat([pd.read_json(p, lines=True) for p in sorted(out.glob("*.jsonl"))])
    g = (df.groupby(["setting", "id"])
           .agg(pass1=("correct", "mean"), gen_len=("gen_len", "mean"), avg_bits=("avg_bits", "mean"),
                mean_cache=("mean_cache_len", "mean"), truncated=("truncated", "mean"), seeds=("seed", "count"))
           .groupby("setting").mean())                      # 문제별 시드 평균 -> 설정별 문제 평균
    g["problems"] = df.groupby("setting")["id"].nunique()
    if "fullkv" in g.index:
        g["len_ratio"] = g["gen_len"] / g.loc["fullkv", "gen_len"]     # 논문 그림 10(d)
    print(g.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default=",".join(SETTINGS))
    ap.add_argument("--seeds", default="0-7")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=30, help="앞에서 몇 문제만 (스모크용)")
    ap.add_argument("--max-new", type=int, default=32768)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--out", default=str(HERE.parent / "result"))
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.summary:
        return summarize(out)

    probs = problems()
    (out / "problems.json").write_text(json.dumps([p["unique_id"] for p in probs]))
    shard, nshard = map(int, a.shard.split("/"))
    lo, _, hi = a.seeds.partition("-")
    device = get_device()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map=device,
                                                 attn_implementation="eager_sparsity")

    for seed in range(int(lo), int(hi or lo) + 1):         # 시드 0 으로 9개 설정을 먼저 다 본 뒤 나머지 시드
        for name in a.settings.split(","):
            path = out / f"{name}.jsonl"
            done = finished(path)
            for i, item in enumerate(probs[:a.limit]):
                if (seed * len(probs) + i) % nshard != shard or (item["unique_id"], seed) in done:
                    continue
                press = make_press(name, seed)
                if not isinstance(press, ThinKVPress):
                    tp.MEASURE_LAYER = -1                  # φ 가 필요 없는 설정은 측정을 끈다
                r = baseline(item, tok, model, device, a.max_new, seed, press=press, press_name=name)
                r.update(setting=name, correct=bool(r["correct"]), **log_summary(press))
                with path.open("a") as fh:
                    fh.write(json.dumps(r) + "\n")
                print(f"{name:9s} seed{seed} {item['unique_id']:14s} ok={r['correct']!s:5s} gen={r['gen_len']:5d} "
                      f"cache={r['cache_len']:5d} {r['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()