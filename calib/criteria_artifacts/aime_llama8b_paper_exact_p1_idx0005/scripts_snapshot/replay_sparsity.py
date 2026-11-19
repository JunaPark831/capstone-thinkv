"""저장된 token_ids를 teacher forcing으로 재생하며 희소도 4변형을 동시에 잰다.

사용: python replay_sparsity.py <in.pt> <out.pt> [max_steps]
사전 기준: docs/e3_paper_exact_criteria.md (§3 변형 정의, §4 재생 방식)
"""
import sys

import torch
from transformers import AttentionInterface, AttentionMaskInterface, AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import eager_mask
from transformers.models.llama.modeling_llama import eager_attention_forward

VARIANTS = ["legacy_perhead", "paper", "pergroup_maxpool", "headavg_nomax"]
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
    a = s.float().amax(1).softmax(-1)                            # Eq.(3)(4): (Hkv, n)
    paper = frac_below(a.mean(0))                                # 그룹 평균 행 하나 -> 1% 판정
    pergroup = frac_below(a).mean()                              # 그룹마다 1% 판정 -> 평균
    headavg = frac_below(wf.mean(0))                             # maxpool 없이 모든 헤드 평균
    return torch.stack([legacy, paper, pergroup, headavg])


def main(src, dst, max_steps=None):
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    d = torch.load(src, weights_only=False)
    model_name, ids = d["model_name"], d["token_ids"]
    steps = len(ids) - 1 if max_steps is None else min(int(max_steps), len(ids) - 1)

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
    out = torch.stack([torch.stack(buf[i]) for i in range(L)]).cpu()   # (L, steps, 4)
    out = out.permute(2, 0, 1).contiguous()                            # (4, L, steps)
    torch.save({"variants": VARIANTS, "sparsity": out.half(), "token_ids": ids[: steps + 1], "src": src,
                "model_name": model_name, "input_len": d["input_len"]}, dst)

    old = d["sparsity"][:, :steps].float()
    new = out[0].float()
    diff = (old - new).abs()
    print("replay legacy vs stored: max|diff|", diff.max().item(), "mean|diff|", diff.mean().item())
    corr = [torch.corrcoef(torch.stack([old[l], new[l]]))[0, 1].item() for l in range(L)]
    print("per-layer corr min/median", min(corr), sorted(corr)[L // 2])
    for v, name in enumerate(VARIANTS):
        x = out[v].float()
        print(f"{name:18s} mean {x.mean():.3f}  min {x.min():.3f}  max {x.max():.3f}  frac>0.98 {(x > 0.98).float().mean():.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
