"""저장된 token_ids를 teacher forcing으로 재생하며 희소도 4변형을 동시에 잰다.

사용: python replay_sparsity_v2.py <in.pt> <out.pt> [max_steps] [prev_replay.pt]
사전 기준: docs/e3_paper_exact_criteria.md (§3 변형 0~3), docs/e3_denominator_variants.md (§2 변형 4~7)
"""
import sys

import torch
from transformers import AttentionInterface, AttentionMaskInterface, AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import eager_mask
from transformers.models.llama.modeling_llama import eager_attention_forward

VARIANTS = ["legacy_perhead", "paper", "pergroup_maxpool", "headavg_nomax",
            "paper_nosink", "paper_win256", "paper_win1024", "sink_mass"]
INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."  # collect_sparsity.py와 동일


def frac_below(row):  # row: (..., n) -> (...)  행 최댓값의 1% 미만 비율
    return (row < 0.01 * row.amax(-1, keepdim=True)).float().mean(-1)


def measure(q, k, w, scaling):
    """q (1,H,1,d), k (1,Hkv,n,d), w = 모델이 낸 softmax 가중치 (1,H,1,n)."""
    H, Hkv = q.shape[1], k.shape[1]
    G = H // Hkv
    wf = w[0, :, 0, :].float()                                   # (H, n)
    legacy = frac_below(wf).mean()
    s = (q[0, :, 0, :].view(Hkv, G, -1) @ k[0].transpose(-1, -2)) * scaling   # (Hkv, G, n) softmax 전, bf16
    pooled = s.float().amax(1)                                   # Eq.(3): (Hkv, n)
    a = pooled.softmax(-1)                                       # Eq.(4)
    paper = frac_below(a.mean(0))                                # 그룹 평균 행 하나 -> 1% 판정
    pergroup = frac_below(a).mean()                              # 그룹마다 1% 판정 -> 평균
    headavg = frac_below(wf.mean(0))                             # maxpool 없이 모든 헤드 평균

    def subset(p):  # 위치 부분집합만으로 softmax 재정규화 -> 그룹 평균 -> 1% 판정
        return frac_below(p.softmax(-1).mean(0))

    nosink = subset(pooled[:, 1:])
    win256 = subset(pooled[:, -256:])
    win1024 = subset(pooled[:, -1024:])
    sink_mass = w[0, :, 0, 0].float().mean()                     # 위치 0이 받는 확률(진단용)
    return torch.stack([legacy, paper, pergroup, headavg, nosink, win256, win1024, sink_mass])


def main(src, dst, max_steps=None, prev=None):
    import time
    t0 = time.time()
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    d = torch.load(src, weights_only=False)
    model_name, ids = d["model_name"], d["token_ids"]
    steps = min(int(max_steps), len(ids) - 1) if max_steps else len(ids) - 1  # 빈 값/None = 전체

    buf = {}

    def attn(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        out, w = eager_attention_forward(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kw)
        if query.shape[2] == 1:  # prefill 제외
            buf.setdefault(module.layer_idx, []).append(measure(query, key, w, scaling))
        return out, w

    AttentionInterface.register("eager_sparsity", attn)
    AttentionMaskInterface.register("eager_sparsity", eager_mask)

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map=device, attn_implementation="eager_sparsity"
    )
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": d["problem"] + INSTRUCTION}], tokenize=False, add_generation_prompt=True
    )
    inp = tok(prompt, return_tensors="pt").input_ids.to(device)
    assert inp.shape[1] == d["input_len"], f"프롬프트 길이 {inp.shape[1]} != 저장된 {d['input_len']}"

    with torch.no_grad():
        pkv = model(input_ids=inp, use_cache=True).past_key_values
        for j in range(steps):
            pkv = model(input_ids=ids[j].view(1, 1).to(device), past_key_values=pkv, use_cache=True).past_key_values
            if j % 500 == 0:
                print(f"step {j}/{steps}", flush=True)

    L = model.config.num_hidden_layers
    assert sorted(buf) == list(range(L)) and all(len(buf[i]) == steps for i in buf)
    out = torch.stack([torch.stack(buf[i]) for i in range(L)]).cpu()   # (L, steps, V)
    out = out.permute(2, 0, 1).contiguous()                            # (V, L, steps)
    torch.save({"variants": VARIANTS, "sparsity": out.half(), "token_ids": ids[: steps + 1], "src": src,
                "model_name": model_name, "input_len": d["input_len"]}, dst)

    old = d["sparsity"][:, :steps].float()
    new = out[0].float()
    diff = (old - new).abs()
    print("replay legacy vs stored: max|diff|", diff.max().item(), "mean|diff|", diff.mean().item())
    corr = [torch.corrcoef(torch.stack([old[l], new[l]]))[0, 1].item() for l in range(L)]
    print("per-layer corr min/median", min(corr), sorted(corr)[L // 2])
    if prev:  # 회귀 확인: 변형 0~3이 직전 재생과 같은가
        p = torch.load(prev, weights_only=False)["sparsity"][:4, :, :steps].float()
        d4 = (p - out[:4].float()).abs()
        print("regression vs prev (variants 0-3): max|diff|", d4.max().item(), "mean|diff|", d4.mean().item())
    print("elapsed s", round(time.time() - t0))
    for v, name in enumerate(VARIANTS):
        x = out[v].float()
        print(f"{name:18s} mean {x.mean():.3f}  min {x.min():.3f}  max {x.max():.3f}  frac>0.98 {(x > 0.98).float().mean():.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
