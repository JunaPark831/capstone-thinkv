"""E4 러너가 쓰는 생성·채점 함수. eval/baseline.py (0단계)의 baseline()을 그대로 가져왔고, 0단계 파일은 건드리지 않는다.
달라진 점은 하나: 결과에 생성 텍스트(output)를 저장한다. 프롬프트 템플릿·샘플링(0.6/0.95)·채점·cache_len은 0단계와 같다."""
import contextlib
import time

import torch
from math_verify import parse, verify


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sync_device(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def baseline(split_item, tokenizer, model, device, max_new_tokens, seed, press=None, press_name="none"):
    message = [{"role": "user", "content": split_item["question"] + "\nPlease reason step by step."}]
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    tokenized_prompt = tokenizer(prompt, return_tensors="pt").to(device=device)
    count_input_tokens = tokenized_prompt.input_ids.size()[1]

    sync_device(device)
    start = time.perf_counter()

    torch.manual_seed(seed)
    context_press = press(model) if press is not None else contextlib.nullcontext()
    with torch.no_grad(), context_press:
        out = model.generate(tokenized_prompt["input_ids"], max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=0.6, top_p=0.95, eos_token_id=tokenizer.eos_token_id,
                             pad_token_id=tokenizer.eos_token_id, attention_mask=tokenized_prompt["attention_mask"],
                             return_dict_in_generate=True)

    output = out.sequences
    cache_len = out.past_key_values.layers[0].keys.shape[2]
    sync_device(device)
    time_taken = time.perf_counter() - start

    truncated = output[0, -1].item() != tokenizer.eos_token_id
    gen_len = output.size()[1] - count_input_tokens
    decoded_output = tokenizer.decode(output[0][count_input_tokens:], skip_special_tokens=True)
    pred_correct = verify(parse(f"${split_item['answer']}$"), parse(decoded_output))

    return {
        "id": split_item["unique_id"],
        "press": press_name,
        "correct": pred_correct,
        "gen_len": gen_len,
        "truncated": truncated,
        "seconds": time_taken,
        "tok_per_s": gen_len / time_taken,
        "seed": seed,
        "input_len": count_input_tokens,
        "cache_len": cache_len,
        "max_new_tokens": max_new_tokens,
        "output": decoded_output,      # 생성 텍스트: 나중에 품질(깨짐·반복)을 직접 읽어 보려고 저장한다
    }
