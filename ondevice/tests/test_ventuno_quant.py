"""V1: decode(encode(x)) 가 thinkv.quant.FQ[kind](x) 와 비트 단위로 같다.
실행: 루트에서 .venv-kvpress/bin/python -m pytest ondevice/tests -q"""
import pytest
import torch

from ondevice import ventuno_quant as codec
from thinkv.quant import FQ


def bits(t):
    """부동소수 텐서를 정수 비트 그대로 본다 (-0.0과 +0.0도 구분된다)."""
    return t.contiguous().view(torch.int16 if t.dtype == torch.bfloat16 else torch.int32)


@pytest.mark.parametrize("kind", ["nvfp4", "ternary"])
@pytest.mark.parametrize("axis", [-2, -1], ids=["key", "value"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_v1_bit_exact(kind, axis, dtype):
    torch.manual_seed(0)
    x = torch.randn(1, 8, 16, 128) * torch.logspace(-3, 2, 128)      # 채널마다 크기가 크게 다르다
    x[0, 0, :, :16] = 0                                              # 스케일이 0이 되는 경우 (1e-12 처리)
    x[0, 1, 3, 5], x[0, 2, 4, 7] = -0.0, 1e-30
    x = x.to(dtype)
    packed, scale, glob = codec.encode(x, axis, kind)
    assert packed.dtype == scale.dtype == torch.uint8
    assert packed.shape == (1, 8, 16, 128 * codec.BITS[kind] // 8)
    assert scale.shape == ((1, 8, 128) if axis == -2 else (1, 8, 16, 8))
    y = codec.decode(packed, scale.unsqueeze(-2) if axis == -2 else scale, glob, axis, kind, dtype)
    ref = FQ[kind](x, axis)
    assert torch.signbit(ref).logical_and(ref == 0).any()            # 정답에 -0.0이 실제로 들어 있다
    assert torch.equal(bits(y), bits(ref))


def test_pack_roundtrip():
    for b in (2, 4):
        c = torch.randint(0, 2**b, (3, 5, 128), dtype=torch.uint8)
        assert torch.equal(codec.unpack(codec.pack(c, b), b), c)