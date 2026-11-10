"""E10 벤치마크 (스펙 4절): 무압축(fullkv) / 게더(gather) / CT 를 같은 합성 타임라인·같은 배치로 비교한다.

실행 (서버, 저장소 루트):  python -m kernels.bench --quick   # 1~2분 점검
                          python -m kernels.bench           # 본 측정
결과: --out(기본 results/e10_bench.jsonl)에 (방식, 모델, 배치, 시드)마다 한 줄 append, 끝에 요약표 출력.

측정 정의:
- attn_ms        마지막 스텝 상태에서 한 층 decode 어텐션. triton.testing.do_bench 중앙값(L2 비우고 반복)
- evict_ms_total 타임라인 동안 축출 이벤트마다 GPU가 한 일의 시간 합. 게더 = copy_slots, CT = live_mask 칸을 0으로
- step_ms        layers * (attn_ms + evict_ms_total / steps). 층마다 같은 타임라인이라고 둔다(스펙 8절)
- kv_bytes_peak  할당된 물리 블록 수의 최댓값 * 블록 바이트(K+V, bf16). 게더는 컴팩션 중 옛·새 블록 공존분 포함
- 세 방식에 공통인 일(새 토큰 store_kv, 호스트 블록 테이블 계산, 인덱스 H2D 복사)은 시간에서 뺀다.
- 풀은 세 방식 모두 같은 크기로 미리 잡으므로 torch_max_alloc은 방식 비교용이 아니다(기록만 한다).
"""
import argparse
import json
import statistics
from pathlib import Path

import torch
import triton

from kernels.block_table import R, BlockTable
from kernels.gather import GatherTable
from kernels.gpu_ops import PagedDecode, copy_slots
from kernels.workload import timeline

MODELS = {
    "llama8b": dict(hq=32, hkv=8, d=128, layers=32),     # R1-Distill-Llama-8B
    "qwen1.5b": dict(hq=12, hkv=2, d=128, layers=28),    # R1-Distill-Qwen-1.5B
}
DEV = "cuda"


def gpu_ms(fn):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b)


def run(method, model, B, args, seed):
    m, bs = MODELS[model], args.block_size
    tls = [timeline(args.prompt, args.steps, args.pattern, args.budget, args.tau, seed=seed * 1000 + b)
           for b in range(B)]
    nb = B * (2 * -(-(args.prompt + args.steps) // bs) + 2)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    kc = torch.randn(nb, bs, m["hkv"], m["d"], dtype=torch.bfloat16, device=DEV)   # 값은 시간과 무관, 유한하기만 하면 된다
    vc = torch.randn_like(kc)
    live_dev = torch.ones(nb, bs, dtype=torch.int8, device=DEV)   # CT 축출 비용 측정용. 측정 직전에 호스트 상태로 다시 올린다

    if method == "ct":
        tab, slot_of = BlockTable(nb, bs), [{} for _ in range(B)]
        for b in range(B):
            tab.add_seq()
            for p in range(args.prompt):
                slot_of[b][p] = tab.append(b, 0, R)
    else:
        tab = GatherTable(nb, bs)
        for b in range(B):
            tab.add_seq()
            for p in range(args.prompt):
                tab.append(b, p)

    # 첫 호출의 Triton 컴파일·캐시 로드가 축출 시간에 섞이지 않게 같은 모양으로 한 번씩 미리 부른다
    w = torch.arange(2, device=DEV)
    copy_slots(kc, vc, w[:1], w[1:])
    live_dev.view(-1).index_fill_(0, w[:1], 1)
    torch.cuda.synchronize()

    evict_ms = []
    for i in range(args.steps):
        for b in range(B):
            st = tls[b][i]
            if method == "ct":
                slot_of[b][st.token] = tab.append(b, st.start, st.type)
            else:
                tab.append(b, st.token)
        if method == "fullkv" or not any(tls[b][i].evict for b in range(B)):
            continue
        if method == "ct":
            gone = []
            for b in range(B):
                s = [slot_of[b].pop(t) for t in tls[b][i].evict]
                tab.evict(b, s)
                gone += s
            idx = torch.tensor(gone, device=DEV)
            evict_ms.append(gpu_ms(lambda: live_dev.view(-1).index_fill_(0, idx, 0)))
        else:
            plans = [tab.compact(b, tls[b][i].evict) for b in range(B)]
            src = torch.tensor([x for p in plans for x in p[0]], device=DEV)
            dst = torch.tensor([x for p in plans for x in p[1]], device=DEV)
            evict_ms.append(gpu_ms(lambda: copy_slots(kc, vc, src, dst)))
            for p in plans:                    # 복사가 끝난 뒤 반납 (gather.py 주의 참고)
                tab.release(p[2])

    if method == "ct":
        tables, live, seq_lens = tab.kernel_inputs(device=DEV)
        blocks_peak, live_tokens = tab.blocks_in_use(), int(live.sum())
    else:
        tables, seq_lens = tab.kernel_inputs(device=DEV)
        live, blocks_peak, live_tokens = None, tab.peak, int(seq_lens.sum())
    q = torch.randn(B, m["hq"], m["d"], dtype=torch.bfloat16, device=DEV)
    out = torch.empty_like(q)
    attn = PagedDecode(B, m["hq"], m["hkv"], m["d"])
    max_len = int(seq_lens.max())
    attn_ms = triton.testing.do_bench(lambda: attn(q, kc, vc, tables, seq_lens, max_len, live, out),
                                      warmup=25, rep=200, return_mode="median")
    total = sum(evict_ms)
    return dict(
        method=method, model=model, batch=B, seed=seed, block_size=bs, steps=args.steps, prompt=args.prompt,
        budget=args.budget, tau=args.tau, pattern=args.pattern, path="3D" if B <= attn.threshold else "2D",
        attn_ms=attn_ms, evict_events=len(evict_ms), evict_ms_total=total,
        evict_ms_mean=total / len(evict_ms) if evict_ms else 0.0,
        evict_ms_median=statistics.median(evict_ms) if evict_ms else 0.0,
        evict_ms_max=max(evict_ms, default=0.0),
        step_ms=m["layers"] * (attn_ms + total / args.steps),
        slots_scanned=int(seq_lens.sum()), live_tokens=live_tokens,
        kv_bytes_peak=blocks_peak * 2 * bs * m["hkv"] * m["d"] * 2,
        torch_max_alloc=torch.cuda.max_memory_allocated(),
        torch_version=torch.__version__, triton_version=triton.__version__,
        gpu=torch.cuda.get_device_name(0),
    )


def summary(rows):
    print("\n| 모델 | 배치 | 경로 | 방식 | attn ms | 축출 ms/스텝 | 스텝 ms (×층) | 축출 스텝 정지 ms (×층) "
          "| 훑는 슬롯/살아 있는 토큰 | KV MB | CT 대비 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for (model, B) in sorted({(r["model"], r["batch"]) for r in rows}):
        group = {r["method"]: r for r in rows if r["model"] == model and r["batch"] == B}
        ct = group.get("ct")
        for meth in ("fullkv", "gather", "ct"):
            r = group.get(meth)
            if not r:
                continue
            ratio = f"{r['step_ms'] / ct['step_ms']:.2f}×" if ct else "-"
            stall = MODELS[model]["layers"] * r["evict_ms_mean"]
            print(f"| {model} | {B} | {r['path']} | {meth} | {r['attn_ms']:.4f} | "
                  f"{r['evict_ms_total'] / r['steps']:.5f} | {r['step_ms']:.3f} | {stall:.3f} | "
                  f"{r['slots_scanned']}/{r['live_tokens']} | {r['kv_bytes_peak'] / 2**20:.1f} | {ratio} |")
    print("\nCT 대비 = 그 방식의 스텝 ms ÷ CT 스텝 ms (1보다 크면 CT가 빠르다). 스텝 ms는 축출 비용을 전체 스텝에 나눈 평균이고,"
          "\n축출 스텝 정지 = 축출이 일어나는 그 스텝에 한 번에 더해지는 시간(층 수 × 이벤트 평균)이다.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32, 64])
    ap.add_argument("--methods", nargs="+", default=["fullkv", "gather", "ct"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--steps", type=int, default=8192)
    ap.add_argument("--prompt", type=int, default=128)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--tau", type=int, default=128)
    ap.add_argument("--pattern", default="RETETRETE")
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--out", default="results/e10_bench.jsonl")
    ap.add_argument("--quick", action="store_true", help="steps 1024, budget 512, 배치 1·8 — 동작 점검용")
    args = ap.parse_args()
    if args.quick:
        args.steps, args.budget, args.batches = 1024, 512, [1, 8]
        args.out = args.out.replace(".jsonl", "_quick.jsonl")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in args.models:
        for B in args.batches:
            for seed in args.seeds:
                for meth in args.methods:
                    r = run(meth, model, B, args, seed)
                    rows.append(r)
                    with open(args.out, "a") as f:
                        f.write(json.dumps(r) + "\n")
                    print(f"{model} B={B} {meth}: attn {r['attn_ms']:.4f} ms, evict {r['evict_events']}회 "
                          f"{r['evict_ms_total']:.2f} ms, step {r['step_ms']:.3f} ms", flush=True)
    summary(rows)


if __name__ == "__main__":
    main()
