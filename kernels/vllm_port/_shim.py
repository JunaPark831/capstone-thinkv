"""vLLM 내부 모듈 대체 (스펙 7절 R4).

kernels/vllm_port/ 의 나머지 세 파일은 vLLM v0.30.0 (commit ced6857afa) 의
vllm/v1/attention/ops/ 에서 본문을 그대로 복사했다 (Apache-2.0, 파일 머리의 SPDX 유지).
바꾼 곳은 두 가지뿐이다:
  1. vllm.* import 블록 -> 이 파일
  2. triton_unified_attention.py 에 live_mask 인자 ([ThinKV CT] 주석으로 표시)
원본과 비교: diff ~/dev/ref/vllm/vllm/v1/attention/ops/<파일> kernels/vllm_port/<파일>
"""
import logging
from enum import IntEnum
from types import SimpleNamespace

import torch
import triton
import triton.language as tl

__all__ = ["triton", "tl", "envs", "init_logger", "current_platform", "KVQuantMode",
           "FP8_DTYPE", "get_fp8_min_max", "is_quantized_kv_cache"]

envs = SimpleNamespace(VLLM_BATCH_INVARIANT=False)
init_logger = logging.getLogger


def _cap():
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


current_platform = SimpleNamespace(
    fp8_dtype=lambda: torch.float8_e4m3fn,
    is_device_capability_family=lambda cap: _cap() // 10 == cap // 10,
    has_device_capability=lambda cap: _cap() >= cap,
    is_rocm=lambda: False,
    is_xpu=lambda: False,
)

FP8_DTYPE = torch.float8_e4m3fn


def get_fp8_min_max():
    info = torch.finfo(FP8_DTYPE)
    return info.min, info.max


def is_quantized_kv_cache(kv_cache_dtype: str) -> bool:     # vllm/utils/torch_utils.py 와 같다
    return (
        kv_cache_dtype.startswith("fp8")
        or kv_cache_dtype.endswith("per_token_head")
        or kv_cache_dtype.startswith("nvfp4")
    )


class KVQuantMode(IntEnum):        # vllm/v1/kv_cache_interface.py 와 같은 값
    NONE = 0
    FP8_PER_TENSOR = 1
    INT8_PER_TOKEN_HEAD = 2
    FP8_PER_TOKEN_HEAD = 3
    INT4_PER_TOKEN_HEAD = 4
    NVFP4 = 5
    TURBOQUANT_K8V4 = 6
