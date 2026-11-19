"""저장된 token_ids 를 청크 teacher-forcing 으로 재생하며 희소도를 문턱 4개(1/2/5/10%) × 변형 4개로 다시 잰다.
사전 기록: docs/e3_paper_calibration_spec.md "문턱 1% → 2·5·10% 재측정". 1% 채널은 저장값과 대조한다(회귀 확인).

사용: HF_HUB_OFFLINE=1 python -I replay_thresholds.py --dir DATA_DIR --out OUT_DIR [--chunk 512] [--only 0009]
출력: OUT_DIR/{idx}.pt — 'variants' 17채널(예: win1024@5), 'sparsity' (17, L, n-1) fp16, 'token_ids', 'regression' (1% 채널 대 저장값 최대 차이)
"""
import argparse
import time
from pathlib import Path

import torch
from transformers import AttentionInterface, AttentionMaskInterface, AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import eager_mask
from transformers.models.llama.modeling_llama import eager_attention_forward

INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."   # collect_calibration.py 와 동일
THRESH = [0.01, 0.02, 0.05, 0.10]
VARS = ["full_nosink", "win1024", "win256", "full_literal"]
CHANNELS = [f"{v}@{int(round(t * 100))}" for v in VARS for t in THRESH] + ["sink_mass"]
BUF = {}
STATE = {"measure": False}


def measure(q, k, w, scaling):
    """q (1,H,T,d) 청크 쿼리, k (1,Hkv,n,d) 전체 키(과거 + 청크), w (1,H,T,n) 모델의 softmax 가중치. 반환 (C, T)."""
    H, Hkv = q.shape[1], k.shape[1]
    G, T, n = H // Hkv, q.shape[2], k.shape[2]
    P0 = n - T                                                                # 청크 앞의 키 수 = 첫 쿼리의 전역 위치
    s = torch.matmul(q[0].view(Hkv, G, T, -1), k[0].unsqueeze(1).transpose(-1, -2)) * scaling   # (Hkv,G,T,n) softmax 전
    pooled = s.float().amax(1)                                                # 식 3: (Hkv,T,n)
    del s
    g = torch.arange(P0, P0 + T, device=q.device).unsqueeze(1)                # (T,1) 쿼리 전역 위치
    j = torch.arange(n, device=q.device).unsqueeze(0)                         # (1,n) 키 위치
    causal = j <= g
    masks = {"full_literal": causal, "full_nosink": causal & (j >= 1),
             "win1024": causal & (j >= 1) & (j >= g - 1023), "win256": causal & (j >= 1) & (j >= g - 255)}
    outs = []
    for v in VARS:
        m = masks[v]
        row = torch.softmax(pooled.masked_fill(~m.unsqueeze(0), float("-inf")), -1).mean(0)   # 식 4 재정규화 -> 그룹 평균 (T,n)
        rmax = row.amax(-1, keepdim=True)
        cnt = m.sum(-1).float()
        for t in THRESH:
            outs.append(((row < t * rmax) & m).sum(-1).float() / cnt)        # 위치 집합 안에서 '최댓값의 t 미만' 비율
        del row
    outs.append(w[0, :, :, 0].float().mean(0))                                # sink_mass: 위치 0 이 받는 확률(헤드 평균)
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
        r = torch.load(f, weights_only=True)   # 원격에서 받은 파일이라 pickle 실행을 막는다
        prompt = tok.apply_chat_template([{"role": "user", "content": r["problem"] + INSTRUCTION}], tokenize=False, add_generation_prompt=True)
        inp = tok(prompt, return_tensors="pt").input_ids[0]
        assert inp.shape[0] == r["input_len"], f"{f.name}: 프롬프트 길이 {inp.shape[0]} != 저장된 {r['input_len']}"
        ids = r["token_ids"]
        seq = torch.cat([inp, ids[:-1].to(inp.dtype)])                        # 프롬프트 + 생성 토큰(마지막 제외) = 쿼리 열
        input_len = int(r["input_len"])
        BUF.clear()
        t0, pos, pkv = time.time(), 0, None
        with torch.no_grad():
            while pos < len(seq):
                end = min(pos + a.chunk, len(seq))
                if pos < input_len < end:
                    end = input_len                                           # 프롬프트/생성 경계에서 청크를 끊어 측정 범위를 정확히
                STATE["measure"] = pos >= input_len
                o = model(input_ids=seq[pos:end].view(1, -1).to(device), past_key_values=pkv, use_cache=True)
                pkv = o.past_key_values
                pos = end
                if device == "mps":
                    torch.mps.empty_cache()
        STATE["measure"] = False
        sp = torch.stack([torch.cat(BUF[l], dim=1) for l in range(L)], 0).permute(1, 0, 2).contiguous()   # (C, L, steps)
        assert sp.shape[2] == len(ids) - 1, (sp.shape, len(ids))
        old, names = r["sparsity"].float(), r["variants"]
        reg = {v: float((old[names.index(v)] - sp[CHANNELS.index(f"{v}@1")].float()).abs().max()) for v in VARS}
        reg["sink_mass"] = float((old[names.index("sink_mass")] - sp[CHANNELS.index("sink_mass")].float()).abs().max())
        torch.save({"idx": r["idx"], "problem": r["problem"], "cot_type": r.get("cot_type"), "model_name": a.model, "variants": CHANNELS,
                    "sparsity": sp.half(), "token_ids": ids, "input_len": input_len, "ended_by_eos": r.get("ended_by_eos"),
                    "src": f.name, "chunk": a.chunk, "regression_max_abs_diff_vs_saved_1pct": reg}, dst)
        dt = time.time() - t0
        print(f"[{f.stem}] {len(ids)} tok, {len(seq) / dt:.1f} tok/s, {dt / 60:.1f} min | 1% 채널 대 저장값 최대 차이 {reg}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
