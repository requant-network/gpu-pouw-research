"""Exhaustive constant-row BF16 scale tests and random rational-oracle tests."""
import ctypes
import hashlib
import json
import pathlib
from datetime import datetime,timezone

import numpy as np
import torch
from experiments.protocol_norms import exact_norms

assert __debug__
library=ctypes.CDLL(str(pathlib.Path('build/libcanonical_norms.so').resolve()))
library.requant_norms.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
library.requant_norms.restype=ctypes.c_int


def evaluate(codes,scales):
    m,k=codes.shape
    a=torch.from_numpy(codes).to('cuda')
    s=torch.from_numpy(scales).to('cuda')
    out=torch.empty((m,2),dtype=torch.uint16,device='cuda')
    assert library.requant_norms(a.data_ptr(),s.data_ptr(),out.data_ptr(),m,k,torch.cuda.current_stream().cuda_stream)==0
    return out.cpu().numpy()


constant_checks=invalid_checks=0
for q in (0,1,3,127,-128):
    bits=np.arange(0x7f80,dtype=np.uint16)
    floats=(bits.astype(np.uint32)<<16).view(np.float32)
    with np.errstate(over='ignore'):
        products=np.float32(abs(q))*floats
    raw=products.view(np.uint32)
    rounded=((raw+0x7fff+((raw>>16)&1))>>16).astype(np.uint32)
    grid=(rounded+2)&0xfffc
    valid=(rounded<0x7f80)&(grid<0x7f80)
    codes=np.full((len(bits),1024),q,np.int8)
    scales=np.repeat(bits[:,None],128,axis=1)
    actual=evaluate(codes,scales)
    expected=np.stack([grid,rounded],axis=1).astype(np.uint16)
    assert np.array_equal(actual[valid],expected[valid]),q
    assert np.all((actual[~valid,0]==0xffff)|(actual[~valid,1]>=0x7f80)),q
    constant_checks+=int(valid.sum());invalid_checks+=int((~valid).sum())
    print('constant row',q,'all finite BF16 scale codes checked',flush=True)
rng=np.random.default_rng(24681357);random_checks=0
for k in (1024,1536,4096,65536):
    m=64
    codes=rng.integers(-128,128,(m,k),dtype=np.int16).astype(np.int8)
    exponents=rng.integers(0,241,(m,k//8),dtype=np.uint16)
    mantissas=rng.integers(0,128,exponents.shape,dtype=np.uint16)
    scales=(exponents<<7)|mantissas
    scales[0].fill(0);scales[1].fill(1);codes[2].fill(0)
    actual=evaluate(codes,scales)
    expected=np.array([exact_norms(v,s) for v,s in zip(codes,scales)],np.uint16)
    assert np.array_equal(actual,expected),k
    random_checks+=m
    print('random rational oracle',k,'all rows checked',flush=True)
record=dict(utc=datetime.now(timezone.utc).isoformat(),seed=24681357,
            constant_valid_rows=constant_checks,constant_overflow_or_grid_rejections=invalid_checks,
            random_rational_oracle_rows=random_checks,
            source_sha256={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in
                           ('validate_canonical_norms.py','cuda/canonical_norms.cu','experiments/protocol_norms.py')},
            scope='Every nonnegative finite BF16 scale code in constant rows for q in {0,1,3,127,-128}, plus heterogeneous full-exponent rows. Constant-row exact norm is abs(q*scale). Unsupported finite-output/grid overflow rows must be rejected. No proof validation.')
pathlib.Path('results/canonical-norm-validation.json').write_text(json.dumps(record,indent=2)+'\n')
