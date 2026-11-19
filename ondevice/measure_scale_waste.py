"""키 스케일 낭비 측정 (2026-11-19 결정: AIME 1문제, GPU 실험과 같은 압축 설정, 약 4,000토큰).

키 스케일은 16토큰 묶음마다 [Hkv, D] 바이트(8B는 1,024)이고, 묶음의 토큰이 모두 퇴출될 때만 반납한다.
묶음이 꽉 차 있으면 토큰당 64바이트다. 퇴출 뒤 묶음에 토큰이 조금만 남으면 이 값이 커진다(최악 1,024).
값이 크면 '퇴출 때 남은 토큰을 다시 16개씩 묶어 양자화하는 방식'을 앞당긴다.

실행(루트에서): .venv-kvpress/bin/python -m ondevice.measure_scale_waste [--max-new 4096] [--attn torch]
"""
import argparse
import random

import torch

from ondevice.ventuno_cache import Engine, attach

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"


def aime_prompt(tok, k=0):
    """GPU 실험(thinkv/eval/run_e4.py)이 뽑은 AIME 2024 문제 15개 중 k번째, 같은 템플릿.
    k=0은 128토큰(prefill 한 묶음), k=4는 146토큰(두 묶음)이다."""
    from datasets import load_dataset

    rows = list(load_dataset("HuggingFaceH4/aime_2024")["train"])
    i = sorted(random.Random(0).sample(range(len(rows)), 15))[k]
    q = rows[i]["problem"] + "\nRemember to put your final answer within \\boxed{}." + "\nPlease reason step by step."
    text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt").input_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--attn", default="fused", choices=["fused", "torch"])
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="eager").eval()
    ids = aime_prompt(tok)
    eng = Engine.from_config(model.config, torch.bfloat16, a.max_new, attn=a.attn)   # τ 128, 예산 1024, 보존 64·32·16·8·4, sparsity 라벨
    attach(model, eng)
    torch.manual_seed(0)
    with torch.no_grad():
        model.generate(ids, max_new_tokens=a.max_new, do_sample=True, temperature=0.6, top_p=0.95,
                       pad_token_id=tok.eos_token_id)

    per = [b / n for n, b in eng.scale_log if n]
    print(f"프롬프트 {ids.shape[1]}토큰, 생성 {len(eng.scale_log)}토큰, 라벨 {''.join(eng.plan.log['labels'])}")
    print("생성 토큰 수 | 살아 있는 토큰(층 평균) | 토큰당 키 스케일 바이트")
    L = len(eng.layers)
    for t in list(range(255, len(eng.scale_log), 256)) + [len(eng.scale_log) - 1]:
        n, b = eng.scale_log[t]
        print(f"{t + 1:6d} | {n / L:8.1f} | {b / max(n, 1):7.1f}")
    print(f"최댓값 {max(per):.1f}, 마지막 {per[-1]:.1f}  (묶음이 꽉 차 있으면 64, 최악 1024)")


if __name__ == "__main__":
    main()
