#!/usr/bin/env bash
# GPU 인스턴스(vast.ai) 환경 준비 — E4 러너용.  thinkv/ 폴더를 올린 디렉토리에서:  bash thinkv/setup_gpu.sh
# torch 는 이미지 기본(CUDA 빌드)을 그대로 쓴다(다시 설치하지 않는다). 가중치는 큰 디스크(/workspace)에 받는다.
# 패키지 버전은 Mac 에서 검증한 것과 같다. 어느 단계든 실패하면 거기서 멈춘다.
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/workspace/hf}"
mkdir -p "$HF_HOME"
grep -q "HF_HOME=" ~/.bashrc 2>/dev/null || echo "export HF_HOME=$HF_HOME" >> ~/.bashrc   # 새 셸(tmux)에서도 같은 캐시를 쓴다

echo "== 1/4 GPU·디스크"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h "$HF_HOME" | tail -1
python --version

echo "== 2/4 패키지 설치 (torch 는 건드리지 않는다)"
python -m pip install -q "transformers==5.2.0" "kvpress==0.5.5" "math-verify==0.9.0" "datasets==5.0.1" pandas tqdm accelerate

echo "== 3/4 설치 확인: 버전 + CUDA 에서 thinkv 가 도는지"
python - <<'EOF'
import importlib.metadata as m
import torch

assert m.version("transformers") == "5.2.0", m.version("transformers")
assert m.version("kvpress") == "0.5.5", m.version("kvpress")
assert torch.cuda.is_available(), "CUDA 를 못 찾았다 (nvidia-smi 는 보이는데 torch 가 CPU 빌드면 이미지가 잘못된 것)"
print("torch", torch.__version__, "| cuda", torch.version.cuda, "|", torch.cuda.get_device_name(0),
      f"| {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB")

import thinkv.press                                   # eager_sparsity 어텐션 등록까지 포함해서 import 가 되는지
from thinkv.kmeans import select_representative_tokens
from thinkv.quant import FQ

x = torch.randn(1, 8, 16, 128, device="cuda", dtype=torch.bfloat16)
for kind, f in FQ.items():                            # float8 캐스트·bucketize 가 CUDA 에서 도는지
    for ax in (-2, -1):
        y = f(x, ax)
        err = ((y.float() - x.float()).norm() / x.float().norm()).item()
        assert y.shape == x.shape and y.dtype == x.dtype and err < 0.8, (kind, ax, err)
idx = select_representative_tokens(torch.randn(8, 128, 128, device="cuda"), 64)   # scatter_reduce·cdist 가 CUDA 에서 도는지
assert idx.numel() == 64 and idx.unique().numel() == 64
print("thinkv quant·kmeans on CUDA: ok")
EOF

echo "== 4/4 모델·데이터 미리 받기 (여기서 실패하면 네트워크·디스크 문제)"
python - <<'EOF'
from datasets import load_dataset
from huggingface_hub import snapshot_download

p = snapshot_download("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
print("모델:", p)
for name in ("HuggingFaceH4/aime_2024", "math-ai/aime25"):
    d = load_dataset(name)
    print(name, {k: len(v) for k, v in d.items()})
EOF

echo
echo "준비 끝. 다음: 파일럿"
echo "  python thinkv/eval/run_e4.py --limit 2 --seeds 0 --max-new 2000 --out thinkv/result_pilot"
