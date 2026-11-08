"""E4M3 Tensor Core GEMM, preparation cost, and low-rank noise peeling."""
import ctypes
import argparse
import hashlib
import json
import pathlib
import time
from datetime import datetime, timezone

import cupy as cp
import numpy as np
from experiments.quantization import encode, decode, fixture

SOURCE = r'''
#include <cuda_fp8.h>
extern "C" __global__ void quantize(const float* a,const float* noise,unsigned char* q,float* decoded,int n,float sigma){
 int i=blockIdx.x*256+threadIdx.x;
 if(i<n){float x=sigma==0 ? a[i] : __fadd_rn(a[i],__fmul_rn(sigma,noise[i]));__nv_fp8_e4m3 v(x);q[i]=v.__x;if(decoded)decoded[i]=(float)v;}
}
'''
quantize = cp.RawKernel(SOURCE, 'quantize')
lib = ctypes.CDLL(str(pathlib.Path('build/libfp8.so').resolve()))
lib.fp8_create.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_int]
lib.fp8_create.restype=ctypes.c_void_p
lib.fp8_run.argtypes=[ctypes.c_void_p]*6
lib.fp8_run.restype=ctypes.c_int
lib.fp8_destroy.argtypes=[ctypes.c_void_p]
lib.fp8_error.restype=ctypes.c_char_p


def error(reference, actual):
    norm = np.linalg.norm(reference)
    value = np.linalg.norm(reference-actual)
    return float(value/norm if norm else value)


def main():
    assert __debug__
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=20261109)
    parser.add_argument('--output',default='results/gpu-study.json')
    args=parser.parse_args()
    rng=np.random.default_rng(args.seed)
    # All Stage-0 bit fixtures must pass through the actual CUDA encoder.
    _,_,bits,_,expected=fixture()
    fa=cp.asarray(np.array(bits,np.uint32).view(np.float32))
    q=cp.empty(len(bits),cp.uint8);decoded=cp.empty(len(bits),cp.float32)
    quantize(((len(bits)+255)//256,),(256,),(fa,cp.zeros_like(fa),q,decoded,np.int32(len(bits)),np.float32(0)))
    assert cp.asnumpy(q).tolist()==expected
    cases=[]
    for n in (256,512,1024,2048):
        scale=cp.ones(1,cp.float32)
        handles=[lib.fp8_create(n,scale.data.ptr,fp8) for fp8 in (0,1,2)]
        assert all(handles),lib.fp8_error().decode()
        workspace=cp.empty(32*1024*1024,cp.uint8)
        out=cp.empty((n,n),cp.float32)
        qa,qb=(cp.empty((n,n),cp.uint8) for _ in range(2))
        decoded_a,decoded_b=(cp.empty((n,n),cp.float32) for _ in range(2))
        def gemm(fp8,a,b):
            status=lib.fp8_run(handles[fp8],a.data.ptr,b.data.ptr,out.data.ptr,workspace.data.ptr,cp.cuda.get_current_stream().ptr)
            assert status==0,status
        try:
            for pattern in ('gaussian','outlier','rank_one','zero'):
                a=rng.normal(0,.7,(n,n)).astype(np.float32)
                b=rng.normal(0,.7,(n,n)).astype(np.float32)
                if pattern=='outlier':a[0,0]=100
                if pattern=='rank_one':a.fill(1);b.fill(1)
                if pattern=='zero':a.fill(0);b.fill(0)
                u=rng.choice([-1.,1.],(n,2)).astype(np.float32)
                v=rng.choice([-1.,1.],(2,n)).astype(np.float32)
                noise=u@v
                packed_b=np.ascontiguousarray(b.T)
                pinned_a=np.frombuffer(cp.cuda.alloc_pinned_memory(a.nbytes),np.float32).reshape(a.shape)
                pinned_b=np.frombuffer(cp.cuda.alloc_pinned_memory(b.nbytes),np.float32).reshape(b.shape)
                pinned_out=np.frombuffer(cp.cuda.alloc_pinned_memory(b.nbytes),np.float32).reshape(b.shape)
                pinned_a[:]=a;pinned_b[:]=packed_b
                da,db=cp.asarray(a),cp.asarray(packed_b)
                half_a,half_b=da.astype(cp.float16),db.astype(cp.float16)
                de,det=cp.asarray(noise),cp.asarray(np.ascontiguousarray(noise.T))
                reference=a.astype(np.float64)@b.astype(np.float64)
                gemm(0,da,db)
                assert error(reference,cp.asnumpy(out).T.astype(np.float64))<5e-5
                gemm(2,half_a,half_b)
                half_error=error(reference,cp.asnumpy(out).T.astype(np.float64))
                assert half_error<.001
                for sigma in (0.,.015625,.25,1.):
                    def prepare(with_decode=False):
                        quantize(((n*n+255)//256,),(256,),(da,de,qa,decoded_a if with_decode else np.uint64(0),np.int32(n*n),np.float32(sigma)))
                        quantize(((n*n+255)//256,),(256,),(db,det,qb,decoded_b if with_decode else np.uint64(0),np.int32(n*n),np.float32(-sigma)))
                    prepare(with_decode=True)
                    # CPU quantizer checks a deterministic subset for every workload.
                    flat=(a+np.float32(sigma)*noise).ravel()
                    observed=cp.asnumpy(qa).ravel()
                    indices=np.linspace(0,n*n-1,257,dtype=int)
                    assert all(int(observed[i])==encode(float(flat[i])) for i in indices)
                    aq=cp.asnumpy(decoded_a).astype(np.float64)
                    bq=cp.asnumpy(decoded_b).T.astype(np.float64)
                    gemm(1,qa,qb)
                    actual=cp.asnumpy(out).T.astype(np.float64)
                    independent=aq@bq
                    accumulation_error=error(independent,actual)
                    assert accumulation_error<5e-5,accumulation_error
                    # Algebraic peeling evaluated in float64; quantization residual remains.
                    e=np.float64(sigma)*noise.astype(np.float64)
                    peeled=actual-e@b.astype(np.float64)+a.astype(np.float64)@e+e@e
                    du,dv=cp.asarray(np.float32(sigma)*u),cp.asarray(v)
                    def peel():
                        return out.T-du@(dv@db.T)+(da@du)@dv+(du@(dv@du))@dv
                    assert np.allclose(cp.asnumpy(peel()),peeled,rtol=2e-4,atol=.02)
                    samples={mode:[] for mode in ('fp32_gemm','fp16_gemm','fp8_gemm','quantize_fp8_gemm','quantize_fp8_gemm_peel','transfer_quantize_fp8_gemm','transfer_prepacked_quantize_fp8_gemm','transfer_pinned_quantize_fp8_gemm')}
                    for iteration in range(35):
                        modes=list(samples)
                        if iteration%2:modes.reverse()
                        for mode in modes:
                            cp.cuda.get_current_stream().synchronize()
                            start=time.perf_counter()
                            if mode=='fp32_gemm':gemm(0,da,db)
                            elif mode=='fp16_gemm':gemm(2,half_a,half_b)
                            else:
                                if mode.startswith('transfer'):
                                    host_a=pinned_a if 'pinned' in mode else a
                                    host_b=pinned_b if 'pinned' in mode else packed_b if 'prepacked' in mode else np.ascontiguousarray(b.T)
                                    ta,tb=cp.asarray(host_a),cp.asarray(host_b)
                                    quantize(((n*n+255)//256,),(256,),(ta,de,qa,np.uint64(0),np.int32(n*n),np.float32(sigma)))
                                    quantize(((n*n+255)//256,),(256,),(tb,det,qb,np.uint64(0),np.int32(n*n),np.float32(-sigma)))
                                elif mode.startswith('quantize_'):prepare()
                                gemm(1,qa,qb)
                                if mode.endswith('_peel'):peel()
                                if mode.startswith('transfer'):
                                    if 'pinned' in mode:out.get(out=pinned_out)
                                    else:cp.asnumpy(out)
                            cp.cuda.get_current_stream().synchronize()
                            elapsed=(time.perf_counter()-start)*1000
                            if iteration==0 and mode not in ('fp32_gemm','fp16_gemm'):
                                assert np.array_equal(cp.asnumpy(out).T.astype(np.float64),actual)
                            if iteration>=5:samples[mode].append(elapsed)
                    cases.append(dict(n=n,pattern=pattern,noise_scale=sigma,
                                      product_error=error(reference,actual),peeled_error=error(reference,peeled),
                                      error_metric='absolute_frobenius' if pattern=='zero' else 'relative_frobenius',
                                      fp32_accumulation_relative_error=accumulation_error,quantizer_subset_checks=len(indices),samples_ms=samples))
                    cases[-1]['fp16_product_error']=half_error
                print(n,pattern,'complete',flush=True)
        finally:
            for handle in handles:lib.fp8_destroy(handle)
    output=dict(utc=datetime.now(timezone.utc).isoformat(),seed=args.seed,
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (pathlib.Path('gpu_study.py'),pathlib.Path('cuda/fp8_matmul.cu'),pathlib.Path('experiments/quantization.py'))},
                device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),runtime=cp.cuda.runtime.runtimeGetVersion(),
                driver=cp.cuda.runtime.driverGetVersion(),numpy=np.__version__,cupy=cp.__version__,
                encoder_fixture_checks=len(bits),warmups=5,samples=30,cases=cases,
                scope='cuBLASLt E4M3/FP16/FP32 inputs, FP32 accumulation/output, unit FP8 scales, no FP8 fast-accum flag. Wall timings include Python/ctypes enqueue and synchronization. Noise and rank-2 factors precomputed and excluded. Peeling uses GPU thin-matrix products. No protocol sampler, hash, ticket or verifier.')
    pathlib.Path(args.output).write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':main()
