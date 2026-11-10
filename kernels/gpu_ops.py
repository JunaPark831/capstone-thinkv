"""GPU 연산 (Triton). 서버에서만 import 한다 (Mac에는 Triton이 없다).

- store_kv:     slot_mapping대로 새 토큰 K·V를 풀에 쓴다 (vLLM triton_reshape_and_cache_flash 그대로)
- PagedDecode:  decode 어텐션 (vLLM unified_attention 그대로 + live_mask). vLLM triton_attn 백엔드처럼
                배치가 작으면 3D(split, 16구간), 크면 2D 경로를 탄다
- copy_slots:   게더 베이스라인의 순차형 복사 (src 슬롯 행 -> dst 슬롯 행)
"""
import torch
import triton
import triton.language as tl

from kernels.vllm_port.triton_reshape_and_cache_flash import triton_reshape_and_cache_flash
from kernels.vllm_port.triton_unified_attention import unified_attention

NUM_SEGMENTS = 16     # vLLM v1/attention/backends/triton_attn.py NUM_PAR_SOFTMAX_SEGMENTS
MIN_GRID_2D = 128     # 같은 파일 MIN_LAUNCH_GRID_SIZE_2D: 배치 >= 128 // Hkv 면 2D 경로


def store_kv(key, value, key_cache, value_cache, slot_mapping):
    """key·value [N, Hkv, D], 캐시 [num_blocks, bs, Hkv, D], slot_mapping [N] int64."""
    one = torch.ones((), dtype=torch.float32, device=key.device)   # fp8 캐시에서만 읽는 스케일
    triton_reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, "auto", one, one)


class PagedDecode:
    """vLLM 백엔드가 미리 잡아 두는 3D 버퍼를 같은 모양으로 들고 unified_attention을 부른다."""

    def __init__(self, max_batch, num_q_heads, num_kv_heads, head_size, device="cuda"):
        self.threshold = MIN_GRID_2D // num_kv_heads
        hp = triton.next_power_of_2(head_size)
        n = self.threshold
        f32 = dict(dtype=torch.float32, device=device)
        self.segm_out = torch.empty(n, num_q_heads, NUM_SEGMENTS, hp, **f32)
        self.segm_max = torch.empty(n, num_q_heads, NUM_SEGMENTS, **f32)
        self.segm_expsum = torch.empty(n, num_q_heads, NUM_SEGMENTS, **f32)
        self.cu = torch.arange(max_batch + 1, dtype=torch.int32, device=device)
        self.one = torch.ones(max_batch, num_kv_heads, **f32)          # k/v descale (bf16 캐시에선 안 읽힘)

    def __call__(self, q, key_cache, value_cache, block_tables, seq_lens, max_seq_len, live_mask=None, out=None):
        """q [B, Hq, D] -> [B, Hq, D]. seq_lens는 CT면 슬롯 수, 무압축·게더면 토큰 수."""
        B = q.shape[0]
        out = torch.empty_like(q) if out is None else out
        unified_attention(
            q=q, k=key_cache, v=value_cache, out=out,
            cu_seqlens_q=self.cu[: B + 1], max_seqlen_q=1,
            seqused_k=seq_lens, max_seqlen_k=max_seq_len,
            softmax_scale=q.shape[-1] ** -0.5, causal=True, window_size=(-1, -1),
            block_table=block_tables, softcap=0,
            q_descale=None, k_descale=self.one[:B], v_descale=self.one[:B],
            seq_threshold_3D=self.threshold, num_par_softmax_segments=NUM_SEGMENTS,
            softmax_segm_output=self.segm_out, softmax_segm_max=self.segm_max,
            softmax_segm_expsum=self.segm_expsum,
            live_mask=live_mask,
        )
        return out


@triton.jit
def _copy_slots_kernel(k_ptr, v_ptr, src_ptr, dst_ptr, ROW: tl.constexpr, TILE: tl.constexpr):
    i = tl.program_id(0)
    offs = tl.program_id(1) * TILE + tl.arange(0, TILE)
    m = offs < ROW
    s = tl.load(src_ptr + i).to(tl.int64) * ROW
    d = tl.load(dst_ptr + i).to(tl.int64) * ROW
    tl.store(k_ptr + d + offs, tl.load(k_ptr + s + offs, mask=m), mask=m)
    tl.store(v_ptr + d + offs, tl.load(v_ptr + s + offs, mask=m), mask=m)


def copy_slots(key_cache, value_cache, src, dst):
    """src·dst: 같은 길이의 GPU 정수 텐서. 서로 겹치면 안 된다 (슬롯마다 병렬이라 순서가 없다)."""
    assert key_cache.is_contiguous() and value_cache.is_contiguous() and src.numel() == dst.numel()
    row = key_cache.shape[2] * key_cache.shape[3]
    tile = min(1024, triton.next_power_of_2(row))
    if src.numel():
        _copy_slots_kernel[(src.numel(), triton.cdiv(row, tile))](
            key_cache, value_cache, src, dst, ROW=row, TILE=tile)
