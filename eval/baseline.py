#FullKV 기준 결과
import torch
import time
import json, os
import contextlib
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from math_verify import parse, verify
from tqdm import tqdm


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def sync_device(device: str):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()

def get_model(model_name: str, dtype: torch.dtype, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map=device)
    return tokenizer, model

def get_dataset():
    dataset = load_dataset("alessiodevoto/math500")
    return dataset

#print(dataset)


#print(split.column_names)
#print(split[0])
#columns: ['question', 'solution', 'answer', 'subject', 'level', 'unique_id', 'answer_prefix', 'context', 'max_new_tokens']



def baseline(split_item: dict, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, device: str, max_new_tokens: int, seed: int, press=None, press_name: str = "none") -> dict:
    problem = split_item["question"]
    id = split_item["unique_id"]
    message = [
        {
            "role" : "user",
            "content" : problem + "\nPlease reason step by step."
            #put your final answer within \\boxed{}. 
            #Dataset already contains the generation prompt.
        }
    ]

    prompt = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )

    tokenized_prompt = tokenizer(prompt, return_tensors="pt").to(device=device)

    count_input_tokens = tokenized_prompt.input_ids.size()[1]

    sync_device(device)
    start = time.perf_counter()

    torch.manual_seed(seed)
    context_press = press(model) if press is not None else contextlib.nullcontext()
    with torch.no_grad(), context_press:
        out = model.generate(tokenized_prompt["input_ids"], max_new_tokens=max_new_tokens, do_sample=True, temperature=0.6, top_p=0.95, eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.eos_token_id, attention_mask=tokenized_prompt["attention_mask"],return_dict_in_generate=True)

    output = out.sequences
    cache_len = out.past_key_values.layers[0].keys.shape[2]
    sync_device(device)
    end = time.perf_counter()

    time_taken = end - start
    truncated = output[0, -1].item() != tokenizer.eos_token_id
    gen_len = output.size()[1] - count_input_tokens

    decoded_output = tokenizer.decode(output[0][count_input_tokens:], skip_special_tokens=True)

    gold = parse(f"${split_item['answer']}$")
    pred = parse(decoded_output)
    pred_correct = verify(gold, pred)

    result = {
        "id": id, 
        "press": press_name, 
        "correct": pred_correct, 
        "gen_len": gen_len, 
        "truncated": truncated, 
        "seconds": time_taken, 
        "tok_per_s": gen_len / time_taken,
        "seed": seed,
        "input_len": count_input_tokens,
        "cache_len": cache_len,
        "max_new_tokens": max_new_tokens
    }
    return result

def make_press(press_name: str):
    if press_name == "none":
        return None
    from kvpress import DecodingPress, KnormPress
    if press_name == "knorm":
        return DecodingPress(base_press=KnormPress(),compression_interval=512, target_size = 1024)
    raise ValueError(f"Unknown press name: {press_name}")

if __name__ == "__main__":
    device = get_device()

    dataset = get_dataset()
    split = dataset["test"]
    
    dtype = torch.bfloat16
    model_name = "deepseek-ai/Deepseek-R1-Distill-Qwen-1.5B"
    tokenizer, model = get_model(model_name, dtype, device)

    RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
    os.makedirs(RESULTS_DIR, exist_ok=True)

    SEEDS = [0]
    PRESS_NAMES = ["none", "knorm"]
    for press_name in tqdm(PRESS_NAMES, desc="Presses"):
        path = os.path.join(RESULTS_DIR, f"baseline_{press_name}.jsonl")

        done = set()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r= json.loads(line)
                        done.add((r["id"], r["seed"]))

        for i in tqdm(range(10), desc=f"Items ({press_name})"):
            item = split[i]
            unique_id = item["unique_id"]
            for seed in SEEDS:
                if (unique_id, seed) in done:
                    tqdm.write(f"{press_name} skip {unique_id} with seed {seed}")
                    continue
                press = make_press(press_name)
                result = baseline(item, tokenizer, model, device, max_new_tokens=8000, seed=seed, press=press, press_name=press_name)
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    tqdm.write(f"done {press_name} {unique_id} with seed {seed}: {result['correct']}, {result['seconds']:.2f}s, {result['tok_per_s']:.2f} tok/s, cache {result['cache_len']}")
