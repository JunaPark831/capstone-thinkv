"""M3: C++ 융합 어텐션(ventuno_attn.cpp)을 ctypes로 부른다.

처음 불릴 때 컴파일해 임시 폴더에 .so를 만들고, 소스나 옵션이 바뀌면 다시 만든다(따로 빌드 명령이 없다).
OpenMP를 켜 보고 안 되면(Mac 기본 clang) 끄고 만든다. 보드 컴파일러 옵션은 VENTUNO_CXXFLAGS로 더한다(예: -mcpu=native).
"""
import ctypes
import hashlib
import os
import subprocess
import tempfile
from functools import cache

import torch

from ondevice.ventuno_quant import BITS, E4M3

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ventuno_attn.cpp")
ptr, i32, i64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64


class CPool(ctypes.Structure):          # ventuno_attn.cpp의 struct Pool과 같은 순서
    _fields_ = [(n, ptr) for n in ("k", "v", "vs", "ks", "live", "gid", "kg", "vg")] + [(n, i64) for n in ("S", "n_gid", "bits")]


@cache
def _lib():
    flags = ["-O3", "-std=c++17", "-shared", "-fPIC"] + os.environ.get("VENTUNO_CXXFLAGS", "").split()
    key = hashlib.sha1(open(SRC, "rb").read() + " ".join(flags).encode()).hexdigest()[:12]
    so = os.path.join(tempfile.gettempdir(), f"ventuno_attn_{key}.so")
    if not os.path.exists(so):
        cmd = [os.environ.get("CXX", "c++"), *flags, SRC, "-o", so]
        if subprocess.run(cmd + ["-fopenmp"], capture_output=True).returncode:
            subprocess.run(cmd, check=True)
    lib = ctypes.CDLL(so)
    for f in (lib.attend_bf16, lib.attend_f32):
        f.argtypes = [ptr, i32, i32, i32, ctypes.c_float, ptr, ptr, i64, ptr, ptr, i64, ptr, i32, ptr, ptr]
        f.restype = None
    return lib


def fused(L, q, scaling):
    """ventuno_cache.Layer 하나에 대한 decode 어텐션. q [Hq, D] → [Hq, D] float32."""
    hkv, P, d = L.pk.shape
    assert d % 64 == 0, "unpack이 16바이트씩 읽는다"
    fn = {torch.bfloat16: _lib().attend_bf16, torch.float32: _lib().attend_f32}[L.pk.dtype]
    ps = list(L.pools.values())
    cp = (CPool * len(ps))(*[CPool(p.k.data_ptr(), p.v.data_ptr(), p.vs.data_ptr(), p.ks.data_ptr(),
                                   p.bt.live_mask.ctypes.data, p.gid.data_ptr(), p.kg.data_ptr(), p.vg.data_ptr(),
                                   p.k.shape[0], p.ks.shape[0], BITS[p.kind]) for p in ps])
    q = q.float().contiguous()
    out = torch.empty_like(q)
    fn(q.data_ptr(), q.shape[0], hkv, d, scaling, L.pk.data_ptr(), L.pv.data_ptr(), P,
       L.bk.data_ptr(), L.bv.data_ptr(), L.nb, cp, len(ps), E4M3.data_ptr(), out.data_ptr())
    return out
