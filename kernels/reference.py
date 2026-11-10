"""정답 어텐션 (PyTorch, float32). 커널 검증 V1·V2·V5의 기준.

paged_attention()은 Triton 커널과 **같은 인자**(물리 풀, block_tables, live_mask, slot 단위 seq_lens)를 받는다.
서버에서는 같은 입력을 커널에 넣어 이 함수 출력과 비교하면 되고, 바뀌는 것은 커널 본체뿐이다.
"""
import torch


def attention(q, k, v, scale):
    """decode 한 토큰의 어텐션. q [Hq, D], k·v [n, Hkv, D] -> [Hq, D] float32.

    GQA: 쿼리 헤드 h는 KV 헤드 h // (Hq // Hkv)를 본다 (vLLM num_queries_per_kv와 같은 규칙).
    n은 1 이상이어야 한다 (프롬프트는 축출하지 않으므로 항상 성립, 스펙 3절).
    """
    g = q.shape[0] // k.shape[1]
    k = k.float().repeat_interleave(g, dim=1)
    v = v.float().repeat_interleave(g, dim=1)
    p = (torch.einsum("hd,nhd->hn", q.float(), k) * scale).softmax(-1)
    return torch.einsum("hn,nhd->hd", p, v)


def paged_attention(q, key_cache, value_cache, block_tables, seq_lens, live_mask, scale):
    """q [B, Hq, D], key/value_cache [num_blocks, bs, Hkv, D], block_tables [B, W],
    seq_lens [B] (슬롯 수), live_mask [num_blocks, bs] (1 = 살아 있음) -> [B, Hq, D] float32.
    live_mask=None이면 모든 슬롯이 살아 있다 (무압축·게더 방식, 6절 3순서의 마스크 없는 이식 검증).
    """
    bs = key_cache.shape[1]
    out = []
    for b in range(q.shape[0]):
        n = int(seq_lens[b])
        blocks = block_tables[b, : -(-n // bs)].long()
        k = key_cache[blocks].flatten(0, 1)[:n]
        v = value_cache[blocks].flatten(0, 1)[:n]
        if live_mask is not None:
            live = live_mask[blocks].flatten()[:n].bool()
            k, v = k[live], v[live]
        out.append(attention(q[b], k, v, scale))
    return torch.stack(out)
