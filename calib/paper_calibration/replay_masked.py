"""청크 teacher-forcing 재생으로 (범위 6: 싱크 블록·로컬 창 제외 'mid' 포함) × (정의 4: 최댓값 대비 1%/5%, 누적 질량 90%/95%) 희소도를 잰다.
사전 기록: docs/criteria/e3_paper_calibration_spec.md "위치 마스킹 + 누적 질량 정의 재측정".

사용: HF_HUB_OFFLINE=1 python -I replay_masked.py --dir DATA_DIR --out OUT_DIR [--chunk 512] [--only 0009] [--ref THRESH_DIR]
"""
import argparse
import time
from pathlib import Path

import torch
from transformers import AttentionInterface, AttentionMaskInterface, AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import eager_mask
from transformers.models.llama.modeling_llama import eager_attention_forward

INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."
SINK, LOCAL = 4, 8                        # 제외: 위치 0..3, 최근 8개(자기 자신 포함)
RANGES = ["full_nosink", "full_mid", "win1024", "win1024_mid", "win256", "win256_mid"]
MEASURES = ["max1", "max5", "mass90", "mass95"]
CHANNELS = [f"{r}@{m}" for r in RANGES for m in MEASURES] + ["sink_mass"]
BUF = {}
STATE = {"measure": False}


def measure(q, k, w, scaling):
    H, Hkv = q.shape[1], k.shape[1]
    G, T, n = H // Hkv, q.shape[2], k.shape[2]
    P0 = n - T
    s = torch.matmul(q[0].view(Hkv, G, T, -1), k[0].unsqueeze(1).transpose(-1, -2)) * scaling
    pooled = s.float().amax(1)                                                # (Hkv,T,n)
    del s
    g = torch.arange(P0, P0 + T, device=q.device).unsqueeze(1)
    j = torch.arange(n, device=q.device).unsqueeze(0)
    masks = {
        "full_nosink": (j >= 1) & (j <= g),
        "full_mid": (j >= SINK) & (j <= g - LOCAL),
        "win1024": (j >= 1) & (j >= g - 1023) & (j <= g),
        "win1024_mid": (j >= SINK) & (j >= g - 1023) & (j <= g - LOCAL),
        "win256": (j >= 1) & (j >= g - 255) & (j <= g),
        "win256_mid": (j >= SINK) & (j >= g - 255) & (j <= g - LOCAL),
    }
    outs = []
    for r in RANGES:
        m = masks[r]
        row = torch.softmax(pooled.masked_fill(~m.unsqueeze(0), float("-inf")), -1).mean(0)   # (T,n), 범위 밖은 0
        cnt = m.sum(-1).float()
        rmax = row.amax(-1, keepdim=True)
        outs.append(((row < 0.01 * rmax) & m).sum(-1).float() / cnt)
        outs.append(((row < 0.05 * rmax) & m).sum(-1).float() / cnt)
        cs = row.sort(-1, descending=True).values.cumsum(-1)                                  # 범위 밖 0은 뒤로 감
        for frac in (0.90, 0.95):
            kk = (cs < frac).sum(-1).float() + 1.0                                            # 질량 frac 에 이르는 데 필요한 위치 수
            outs.append(1.0 - kk / cnt)
        del row, cs
    outs.append(w[0, :, :, 0].float().mean(0))
    return torch.stack(outs, 0).cpu()


def attn(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    out, w = eager_attention_forward(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kw)
    if STATE["measure"]:
        BUF.setdefault(module.layer_idx, []).append(measure(query, key, w, scaling))
    return out, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=None, help="직전 재생 폴더(1% 채널 회귀 대조)")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--only", default=None)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    AttentionInterface.register("eager_sparsity", attn)
    AttentionMaskInterface.register("eager_sparsity", eager_mask)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map=device, attn_implementation="eager_sparsity")
    L = model.config.num_hidden_layers
    files = sorted(Path(a.dir).glob("*.pt"))
    if a.only:
        files = [f for f in files if f.stem in a.only.split(",")]
    for f in files:
        dst = out / f.name
        if dst.exists():
            print(f"skip {f.name} (exists)", flush=True)
            continue
        r = torch.load(f, weights_only=True)
        prompt = tok.apply_chat_template([{"role": "user", "content": r["problem"] + INSTRUCTION}], tokenize=False, add_generation_prompt=True)
        inp = tok(prompt, return_tensors="pt").input_ids[0]
        assert inp.shape[0] == r["input_len"]
        ids = r["token_ids"]
        seq = torch.cat([inp, ids[:-1].to(inp.dtype)])
        input_len = int(r["input_len"])
        BUF.clear()
        t0, pos, pkv = time.time(), 0, None
        with torch.no_grad():
            while pos < len(seq):
                end = min(pos + a.chunk, len(seq))
                if pos < input_len < end:
                    end = input_len
                STATE["measure"] = pos >= input_len
                o = model(input_ids=seq[pos:end].view(1, -1).to(device), past_key_values=pkv, use_cache=True)
                pkv = o.past_key_values
                pos = end
                if device == "mps":
                    torch.mps.empty_cache()
        STATE["measure"] = False
        sp = torch.stack([torch.cat(BUF[l], dim=1) for l in range(L)], 0).permute(1, 0, 2).contiguous()
        assert sp.shape[2] == len(ids) - 1
        reg = {}
        if a.ref and (Path(a.ref) / f.name).exists():
            ref = torch.load(Path(a.ref) / f.name, weights_only=True)
            for v in ("full_nosink", "win1024", "win256"):
                reg[v] = float((ref["sparsity"][ref["variants"].index(f"{v}@1")].float() - sp[CHANNELS.index(f"{v}@max1")].float()).abs().max())
        torch.save({"idx": r["idx"], "problem": r["problem"], "cot_type": r.get("cot_type"), "model_name": a.model, "variants": CHANNELS,
                    "sparsity": sp.half(), "token_ids": ids, "input_len": input_len, "ended_by_eos": r.get("ended_by_eos"), "src": f.name,
                    "chunk": a.chunk, "sink_excluded": SINK, "local_excluded": LOCAL, "regression_max_abs_diff_vs_prev_replay_1pct": reg}, dst)
        dt = time.time() - t0
        print(f"[{f.stem}] {len(ids)} tok, {len(seq) / dt:.1f} tok/s, {dt / 60:.1f} min | 직전 재생 1% 대비 최대 차이 {reg}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
