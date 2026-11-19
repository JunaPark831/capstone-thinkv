"""논문 Algorithm 1 의 수집 단계 (명세: docs/e3_paper_calibration_spec.md).

사용: python collect_calibration.py --out DIR [--P 100] [--max-new 32768] [--shard 0/1]
                                    [--model deepseek-ai/DeepSeek-R1-Distill-Llama-8B] [--device auto]
문제마다 DIR/{idx:04d}.pt 로 저장한다. 이미 있으면 건너뛰므로 재개·샤딩(--shard i/N)이 된다.
요구: transformers==5.2.0 (AttentionInterface 등록 API), torch, datasets, tqdm
"""
import argparse
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AttentionInterface, AttentionMaskInterface, AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import eager_mask
from transformers.models.llama.modeling_llama import eager_attention_forward  # Llama·Qwen2 동일 구현

VARIANTS = ["full_nosink", "win1024", "win256", "full_literal", "sink_mass"]
INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."
BUF = {}  # layer_idx -> [텐서(5,) per 디코드 스텝]


def frac_below(row):  # 행 최댓값의 1% 미만인 위치의 비율
    return (row < 0.01 * row.amax(-1, keepdim=True)).float().mean(-1)


def measure(q, k, w, scaling):
    """q (1,H,1,d), k (1,Hkv,n,d), w = 모델이 낸 softmax 가중치 (1,H,1,n)."""
    H, Hkv = q.shape[1], k.shape[1]
    s = (q[0, :, 0, :].view(Hkv, H // Hkv, -1) @ k[0].transpose(-1, -2)) * scaling   # (Hkv, G, n) softmax 전, bf16
    pooled = s.float().amax(1)                                                       # 논문 식 3: (Hkv, n)

    def row(p):  # 위치 집합에 대해 softmax 재정규화(식 4) -> 그룹 평균 -> 1% 판정
        return frac_below(p.softmax(-1).mean(0))

    body = pooled[:, 1:]                                                             # 위치 0(BOS 싱크) 제외
    return torch.stack([row(body), row(body[:, -1024:]), row(body[:, -256:]), row(pooled),
                        w[0, :, 0, 0].float().mean()])


def attn(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    out, w = eager_attention_forward(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kw)
    if query.shape[2] == 1:  # prefill 제외
        BUF.setdefault(module.layer_idx, []).append(measure(query, key, w, scaling))
    return out, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--P", type=int, default=100)
    ap.add_argument("--max-new", type=int, default=32768)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    shard, nshard = map(int, a.shard.split("/"))
    device = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    AttentionInterface.register("eager_sparsity", attn)
    AttentionMaskInterface.register("eager_sparsity", eager_mask)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map=device, attn_implementation="eager_sparsity")
    L = model.config.num_hidden_layers

    ds = load_dataset("simplescaling/s1K-1.1", split="train").shuffle(seed=0).select(range(a.P))
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(ds):
        path = out_dir / f"{idx:04d}.pt"
        if idx % nshard != shard or path.exists():
            continue
        BUF.clear()
        torch.manual_seed(0)
        prompt = tok.apply_chat_template([{"role": "user", "content": item["question"] + INSTRUCTION}], tokenize=False, add_generation_prompt=True)
        inp = tok(prompt, return_tensors="pt").to(device)
        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=a.max_new, do_sample=True, temperature=0.6, top_p=0.95,
                                 eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
        input_len = inp["input_ids"].shape[1]
        ids = gen[0, input_len:].cpu()
        assert sorted(BUF) == list(range(L)) and all(len(BUF[i]) == len(ids) - 1 for i in BUF), "희소도 길이 불일치"
        sp = torch.stack([torch.stack(BUF[i]) for i in range(L)]).cpu().permute(2, 0, 1).contiguous()  # (5, L, steps)
        torch.save({"idx": idx, "problem": item["question"], "cot_type": item["cot_type"], "model_name": a.model,
                    "dataset_name": "simplescaling/s1K-1.1", "variants": VARIANTS, "sparsity": sp.half(), "token_ids": ids,
                    "input_len": input_len, "max_new_tokens": a.max_new, "ended_by_eos": bool(ids[-1] == tok.eos_token_id),
                    "seed": 0, "temperature": 0.6, "top_p": 0.95}, path)
        print(f"[{idx}] {len(ids)} tok, {len(ids) / (time.time() - t0):.1f} tok/s, eos={bool(ids[-1] == tok.eos_token_id)}", flush=True)


if __name__ == "__main__":
    main()
