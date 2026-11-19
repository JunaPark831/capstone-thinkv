#구간 안 대표 토큰 고르기: select representative tokens in the interval

import torch

#반환값은 keys에서 남길 key의 인덱스
#iters는 군집 업데이트 횟수
def select_representative_tokens(keys, K, iters = 10):
    #Hkv : KV 헤드 수
    #n : 토큰 수
    #d : 토큰 차원 수 (채널)
    head_kv, n_tokens, d_dimension = keys.shape
    if K >= n_tokens:
        #K가 토큰 수보다 크거나 같으면 그냥 그대로 반환
        return torch.arange(n_tokens, device = keys.device)

    #head concatenation
    x = keys.permute(1,0,2).reshape(n_tokens, head_kv * d_dimension).float()

    #초기 군집 K개 생성
    ar = torch.arange(n_tokens, device = x.device)
    cen = x[torch.linspace(0, n_tokens - 1, K, device=x.device).round().long()]

    #군집 업데이트
    for _ in range(iters):
        a = torch.cdist(x,cen).argmin(1)
        cnt = torch.bincount(a, minlength=K)
        mean = torch.zeros_like(cen).index_add_(0, a, x) / cnt.clamp_min(1)[:, None]
        cen = torch.where(cnt[:, None] > 0, mean, cen)  #빈 군집은 중심 유지
    a = torch.cdist(x, cen).argmin(1)
    dd = (x - cen[a]).norm(dim=1)     #자기 중심까지의 거리

    #군집마다 중심에 가장 가까운 실제 토큰 하나
    mn = torch.full((K,), float("inf"), device=x.device).scatter_reduce(0, a, dd, "amin")
    first = torch.full((K,), n_tokens, device=x.device).scatter_reduce(0, a, torch.where(dd == mn[a], ar, n_tokens), "amin")
    picked = first[first < n_tokens]

    m = K - picked.numel()
    if m:
        #빈 군집 수만큼, 안 뽑힌 토큰 중 자기 중심에서 가장 먼 것으로 채움 -> 정확히 K
        #free = dd.clone()
        #free[picked] = -1
        #picked = torch.cat([picked, free.topk(m).indices])

        #빈 군집 수만큼, 토큰이 가장 많은 군집이 더 가져감 (그 안에서 중심에서 먼 것부터) -> 정확히 K개 
        cnt = torch.bincount(a, minlength=K)  # 마지막 배정 기준 군집 크기
        free = cnt[a] + dd / (dd.max() + 1) # 정수부 = 군집 크기, 소수부 = 중심에서의 거리(0~1)
        free[picked] = -1
        picked = torch.cat([picked, free.topk(m).indices])
    return picked.sort().values

if __name__ == "__main__":
    idx = select_representative_tokens(torch.randn(8, 128, 128), 64)
    assert idx.numel() == 64 and idx.unique().numel() == 64 and (idx[1:] > idx[:-1]).all()
    assert select_representative_tokens(torch.randn(8, 16, 128), 16).numel() == 16   # K == n 이면 전부
    same = select_representative_tokens(torch.zeros(8, 128, 128), 64)                # 전부 같은 벡터
    assert same.numel() == 64 and same.unique().numel() == 64
    print("kmeans ok")