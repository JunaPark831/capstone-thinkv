"""
    양자화
    묶음 & 스케일 값 & glob
    역양자화
"""

import torch

from thinkv.quant import BOUND, GRID, G
#scale로 나누어서 GRID의 최댓값으로 범위를 맞춘다.
#그때를 위해 BOUND 필요,,경계선임
#G는 묶음 내 수 G = 16,,값 16개 묶음

BITS = {"nvfp4": 4, "ternary" : 2}
E4M3 = torch.arange(256, dtype = torch.uint8).view(torch.float8_e4m3fn).float() #2^8 = 256
#4바이트대신 E4M3라는 (8bit = 1byte)로 스케일 표현함..각 묶음마다 스케일값이 있을텐데 묶음이 굉장히 많기에..표현을 위한 스케일을 1byte로 줄여서 저장한다.
#E4M3 = 부호 1비트(+/-) + 지수 4비트(크기)(2^(x-bias)) + 가수 3비트(정밀도)(1+(각각 1/2, 1/4, 1/8))
#스케일 = 부호 * 지수 * 가수
#ex) 0 1111 110 => + 2^(8) * (1+1/2+1/4) = 448
#E4M3는 +/- 0.00195 ~ 448.까지 나타낼 수 있다. 
#0-126 -> 양수 (0-448)
#127 -> NaN
#128-254 -> 음수
#255 -> NaN
#여기에 모든 E4M3이 들어가 있다.

#E4M3 8비트를 int8 정수 비트로 우선 저장해둠. 1:1 대응됨.


#byte랑 glob통해서 -> 실제 연산에 쓸 스케일 (바이트) 반환.      scale for each 묶음
def scale_of(byte, glob):
    #E4M3값 반환
    #byte(8비트)-> 실제로 쓰일 스케일
    # *glob는 448범위안으로 좁혔으니 다시 펼쳐준다.(좁히거나)...448이상의 스케일값이 나오는것 가능하게 해줌(glob)
    return (E4M3[byte.long()]*glob).clamp_min(1e-12)
    #현재 byte는 8비트...long통해서 64비트로 늘려서 인덱스로 사용가능

def _scale(s_raw): #1바이트내로 표현한 scale값이 아닌 float32로 표현된 scale이다. (s_raw)
    #scale 을 바이트화시킴
    glob = (s_raw.amax()/448).clamp_min(1e-12) 
    #scale값중에서 제일 큰값을 448로 나눔. 모든 scale값들을 E4M3로 1 바이트안에 담아야 하는데 
    # 표현할 수 있는게 448까지이기에 이 범위안으로 만들기 위해 제일 큰 값에 448로 나누어서 glob를 둔다.
    #scale / glob -> (0-448)안으로 맞추어진다!!!)
    byte = (s_raw/glob).to(torch.float8_e4m3fn).view(torch.uint8)
    #glob통해서 8바이트 내로 표현가능한 범위로 바꾸어준다.
    return byte, glob, scale_of(byte,glob) #scale바이트, glob, 실제 스케일값 in E4M3 (8비트,,1바이트)

# 실제 양자화 수행,,,실제 비트 수 줄이는 코드 
# 바이트 단위로 간다. 
# 한 바이트 당
# nvfp4 -> 2 코드 들어감
# tenary -> 4 코드 들어감
def pack(code, bits):
    per = 8 // bits
    c = code.to(torch.uint8).unflatten(-1, (-1, per))
    out = c[...,0].clone()
    for j in range(1,per):
        out |= c[..., j] << (bits*j) 
    return out

#실제 역양자화 코드
def unpack(packed, bits):
    """pack의 반대. [..., D*bits/8] uint8 -> [..., D] uint8."""
    per = 8 // bits
    return torch.stack([(packed >> (bits * j)) & (2**bits - 1) for j in range(per)], -1).flatten(-2)

def values(code, kind):
    """코드 -> 격자 값 float32 (스케일을 곱하기 전)."""
    if kind == "nvfp4":
        mag, neg = GRID[(code & 7).long()], code >> 3
    else:
        mag, neg = (code & 1).float(), code >> 1
    return torch.where(neg.bool(), -mag, mag)

# 양자화!
# x
# -> key or value cache
# -> buffer가 다 차면 encode진행함   16 X [8 헤드 , 128 hk]
# axis는 묶음을 한 토큰 내 채널로 묶을지(value) 한 채널의 여러 토큰별로할지(key)
# kind는 4, 2 비트 중 어떤걸로 줄일지.
def encode(x, axis, kind):
    """x [..., n, D] -> (코드 묶음 [..., n, D*bits/8], 스케일 바이트, glob).

    axis=-2 (키):  n == 16. 채널마다 토큰 16개가 스케일 하나.  스케일 [..., D]
    axis=-1 (값):  토큰마다 채널 16개가 스케일 하나.           스케일 [..., n, D/16]
    """
    g = x.float().unflatten(axis, (-1, G))
    #캐시를 묶음(G=16)으로 나눔

    #4비트 양자화
    if kind == "nvfp4":
        #각 묶음의 scale값 뽑음
        byte, glob, s = _scale(g.abs().amax(axis, keepdim=True) / 6)

        #각 묶음에 각각의 scale값을 나누어 준다. -> y
        #GRID안으로 이제 맞추어주기 위해.
        y = g / s

        #BOUND경계따라 가장 가까운 격자로! (번호임) 
        #격자의 index임
        #음수이면 8더해준다. 1xxx로 맞추어줌.
        code = torch.bucketize(y.abs().contiguous(), BOUND) | ((y < 0) << 3) 

    #2비트 양자화
    else:
        byte, glob, s = _scale(g.abs().mean(axis, keepdim=True))
        code = (g.abs() > 0.5 * s).long() | ((g < 0) << 1)

    #code는 64 bit인 상태...이제 이거를 실제 bit수를 줄인다.
    packed = pack(code.flatten(axis - 1, axis), BITS[kind])
    return packed, (byte.flatten(-3) if axis == -2 else byte.squeeze(-1)), glob

#dtype으로 역양자화한다.
def decode(packed, scale, glob, axis, kind, dtype=torch.float32):
    """encode의 반대. 코드 묶음 [..., n, D*bits/8] -> [..., n, D].

    scale: 키는 [..., 1, D] 또는 [..., n, D] (encode가 준 것에 토큰 축을 끼운 모양), 값은 [..., n, D/16].
    glob:  스칼라, 또는 scale과 브로드캐스트되는 텐서 (묶음이 서로 다른 토큰을 한 번에 풀 때).
    """
    q = values(unpack(packed, BITS[kind]), kind)
    s = scale_of(scale, glob)
    if axis == -1:
        s = s.repeat_interleave(G, -1)
    return (q * s).to(dtype)