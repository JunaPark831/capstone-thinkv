import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, concatenate_datasets
from pathlib import Path
from tqdm.auto import tqdm

def load_data(P: int):
    if P % 2 != 0:
        raise ValueError("P must be even. 2024, 2025 를 같은 수로 뽑기 위해 P는 짝수여야 한다!")

    dataset = load_dataset("allenai/aime-2022-2025", split="train")

    aime_2024 = dataset.filter(lambda x: x["year"] == 2024)
    aime_2025 = dataset.filter(lambda x: x["year"] == 2025)

    assert len(aime_2024) == 30
    assert len(aime_2025) == 30

    per_year = P // 2

    aime_2024 = aime_2024.shuffle(seed=0).select(range(per_year))
    aime_2025 = aime_2025.shuffle(seed=0).select(range(per_year))

    selected = concatenate_datasets([aime_2024, aime_2025]).shuffle(seed=0)

    return selected

def make_hook(i, buf):
    #수집기 만드는 함수
    def hook(module, args, output):
        w = output[1]
        if w.shape[2] != 1:
            return #prefill단계 제외
        row = w[0,:,0,:].float()  #i번째 layer의 attention weight
        smaller = row < 0.01 * row.amax(-1, keepdim=True)  # 최대값 대비 1% 미만인 값인지 아닌지
        sparsity = smaller.float().mean() #미만 비율 = 희소도
        buf[i].append(sparsity) 
    return hook


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

if __name__ == "__main__":
    device = get_device()
    #model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    assert "kvpress" not in sys.modules

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype = torch.bfloat16, device_map = device,
        attn_implementation="eager"
        )

    assert model.config._attn_implementation == "eager"
    assert "kvpress" not in sys.modules

    ########실험 세팅###########
    n_layers = model.config.num_hidden_layers
    print(f"number of layers: {n_layers}")

    buf = [[] for _ in range(n_layers)]
    #sparsity 수집기 등록
    handles = [layer.self_attn.register_forward_hook(make_hook(i,buf)) for i, layer in enumerate(model.model.layers)]


    ##################실측1########################

    P = 10
    MAX_NEW_TOKENS = 35000
    dataset = load_data(P)

    assert len(dataset) == P
    assert sum(item["year"] == 2024 for item in
  dataset) == P//2
    assert sum(item["year"] == 2025 for item in
  dataset) == P//2

    output_dir = (
        Path(__file__).resolve().parent
        / "sparsity"
        / "aime_llama8b_p10_max35000"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    
    try:
        for idx, item in enumerate(tqdm(dataset, desc="Collecting sparsity")):
            output_path = output_dir / f"{idx:04d}.pt"

            if output_path.exists():
                tqdm.write(f"이미 존재하는 파일 건너뜀: {output_path}")
                continue

            #sparsity 수집기 등록
            
        
            instruction = (
                "\nPlease reason step by step, "
                "and put your final answer within \\boxed{}."
            )

            problem = item["problem"]
            message = [
                {
                    "role" : "user",
                    "content" : problem + instruction
                }
            ]

            for b in buf:
                b.clear()

            torch.manual_seed(0)
        
            prompt = tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
            ).to(device=device)
        
            out = model.generate(
                input_ids = inputs["input_ids"],
                attention_mask = inputs["attention_mask"],
                max_new_tokens = MAX_NEW_TOKENS,
                do_sample = True,
                temperature = 0.6,
                top_p = 0.95,
                eos_token_id = tokenizer.eos_token_id,
                pad_token_id = tokenizer.eos_token_id
            )
            input_len = inputs["input_ids"].shape[1]

                
            if not all(buf):
                raise RuntimeError(f"{idx}번 문제 희소도 수집 실패. 최소 2개 이상 토큰 생성 필요!!")
            
            sparsity = torch.stack([torch.stack(layer_buf) for layer_buf in buf]).cpu()

            token_ids = out[0, input_len:].cpu()

            assert sparsity.shape[0] == model.config.num_hidden_layers, f"희소도 shape {sparsity.shape}와 layer 수 {model.config.num_hidden_layers} 불일치"
            assert sparsity.shape[1] == len(token_ids) - 1, f"희소도 shape {sparsity.shape}와 생성 토큰 수 {len(token_ids)} 불일치"

            torch.save(
                {
                    "idx": idx,
                    "source_id": item["id"],
                    "year": int(item["year"]),
                    "problem": problem,
                    "answer": item["answer"],
                    "model_name": model_name,
                    "dataset_name": "allenai/aime-2022-2025",
                    "sparsity": sparsity.half(),
                    "token_ids": token_ids,
                    "input_len": input_len,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "seed": 0,
                    "temperature": 0.6,
                    "top_p": 0.95
                },
                output_path
            )
    
    
    finally:
        for handle in handles:
            handle.remove()