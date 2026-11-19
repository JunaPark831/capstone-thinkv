#가짜 양자화: fq_nvfp4, fq_ternary, fq_kivi2 (+ 자기 검사)
import torch

G=16
GRID = torch.tensor([0,.5,1,1.5,2,3,4,6])
BOUND = torch.tensor([.25,.75,1.25,1.75,2.5,3.5,5])

def _e4m3(s,glob):
    #s를 E4M3FN으로 맞추는 단계
    t = s / glob
    device_ = t.device
    if device_.type == "mps":
        t = t.cpu()
    t = t.to(torch.float8_e4m3fn).float()
    return t.to(device_) * glob

def fq_nvfp4(x, axis):
    #R, E to 4-bit NVFP4
    #axis = -1 은 값 (한 토큰 의 채널 전체)
    #axis = -2 는 키 (한 채널 상 토큰 전체)
    original_x_datatype = x.dtype
    group = x.float().unflatten(axis, (-1,G))

    #6으로 나누어주어서 x/s의 절댓값 범위를 6이하로 두는것!
    s_raw = group.abs().amax(axis, keepdim=True)/6

    #s_raw를 E4M3FN으로 맞추기 위해서 절댓값 범위를 448로 두는것.
    #clamp_min(1e-12)로 0이 되는것을 방지 1e-12미만은 다 1e-12로 두는것.
    glob = (s_raw.amax() / 448).clamp_min(1e-12)

    #s_raw, glob통해서 s를 E4M3FN표현가능 범위에 두고 담길수 있게 반올림하는 단게
    s = _e4m3(s_raw, glob).clamp_min(1e-12)

    #양자화 진행
    y = group / s
    idx = torch.bucketize(y.abs(), BOUND.to(x.device))
    quant = GRID.to(x.device)[idx] * y.sign()

    #양자화 복원값 반환
    return (quant * s).flatten(axis-1, axis).to(original_x_datatype)

def fq_ternary(x, axis):
    #T to ternar (2-bit)
    original_x_datatype = x.dtype
    group = x.float().unflatten(axis, (-1,G))

    #격자에 맞추는것이 아닌 {-s, 0, s} 중 하나로만 보내면 되니 mean사용
    #최댓값 활용하면 그쪽으로 빠질것 -> mean으로 대표값 설정하고 그거 위아래로 판별
    s_raw = group.abs().mean(axis, keepdim = True)

    glob = (s_raw.amax()/ 448).clamp_min(1e-12)
    s = _e4m3(s_raw, glob).clamp_min(1e-12)
    quant = group.sign() * (group.abs() > 0.5*s)

    return (quant* s).flatten(axis-1, axis).to(original_x_datatype)
    

def fq_kivi2(x, axis):
    #라벨 구분 X
    #모두 Ternary로 보내고, 그 중 2개만 선택해서 대표값으로 보내는것
    original_x_datatype = x.dtype   
    g = x.float().unflatten(axis, (-1,G))
    mn = g.amin(axis, keepdim=True)
    scale = ((g.amax(axis, keepdim=True) - mn) / 3).clamp_min(1e-12)
    q = ((g - mn) / scale).round().clamp(0, 3)
    return (q * scale + mn).flatten(axis - 1, axis).to(original_x_datatype)

FQ = {"nvfp4": fq_nvfp4, "ternary": fq_ternary, "kivi2": fq_kivi2}

if __name__ == "__main__":
    x = torch.randn(2, 8, 16, 128)
    for f, lim in ((fq_nvfp4, .15), (fq_ternary, .8), (fq_kivi2, .5)):
        for ax in (-2, -1):
            y = f(x, ax); assert y.shape == x.shape
            assert ((y - x).norm() / x.norm()) < lim, (f.__name__, ax)
        assert not torch.equal(f(x, -2), f(x, -1))
    print("quant ok")