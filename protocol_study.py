"""Run pinned Pearl kernels against exact rational norm/scale references."""
import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import pathlib
import struct
import subprocess
import sys
import time
from datetime import datetime,timezone

import numpy as np
import torch
from experiments.protocol_norms import exact_norms,fixture,scales_from_norms

REVISION='b7160514ed3d862fb48e850ba9a70a545cefcb1e'


def main():
    assert __debug__
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('upstream',type=pathlib.Path)
    parser.add_argument('--seed',type=int,default=20261111)
    parser.add_argument('--output',default='results/protocol-study.json')
    args=parser.parse_args()
    assert subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip()==REVISION
    # The tested modules come directly from this revision, without patching upstream.
    sys.path.insert(0,str(args.upstream/'miner/pearl-gemm/src'))
    from pearl_gemm import (pre_quant,pre_quant_output_shapes,tensor_hash_plus_stats_b,
                            tensor_hash_workspace_bytes,TensorHashConfig,noisy_quant,pack_noise_factor)
    from pearl_gemm.protocol_constants import R
    assert torch.cuda.get_device_capability()==(12,0)
    library=ctypes.CDLL(str(pathlib.Path('build/libcanonical_norms.so').resolve()))
    library.requant_norms.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
    library.requant_norms.restype=ctypes.c_int
    library.requant_inject.argtypes=[ctypes.c_void_p]*2+[ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
    library.requant_inject.restype=ctypes.c_int
    rng=np.random.default_rng(args.seed)
    fixture_values,fixture_scales,fixture_bits=fixture(pathlib.Path('fixtures/norm-boundary.bin'))
    cases=[]
    for kind,k in [('known_boundary',1024),('boundary_permutations',1024),('random',1024),('random',4096),('random',16384),('random',65536)]:
        m=64
        if kind=='known_boundary':
            input_bits=np.tile(np.array(fixture_bits,np.uint16),(m,1))
        elif kind=='boundary_permutations':
            blocks=np.array(fixture_bits,np.uint16).reshape(-1,8)
            input_bits=np.stack([blocks[rng.permutation(len(blocks))].reshape(-1) for _ in range(m)])
        else:
            floats=rng.normal(0,.7,(m,k)).astype(np.float32)
            bits=floats.view(np.uint32)
            input_bits=((bits+0x7fff+((bits>>16)&1))>>16).astype(np.uint16)
        inputs=(input_bits.astype(np.uint32)<<16).view(np.float32)
        x=torch.from_numpy(inputs).to(device='cuda',dtype=torch.bfloat16)
        code_shape,scale_shape=pre_quant_output_shapes(m,k)
        codes=torch.empty(code_shape,dtype=torch.int8,device='cuda')
        scales=torch.empty(scale_shape,dtype=torch.bfloat16,device='cuda')
        pre_quant(x,codes,scales)
        host_codes=codes.cpu().numpy();host_scales=scales.view(torch.uint16).cpu().numpy()
        if kind=='known_boundary':
            assert np.array_equal(host_codes[0],np.array(fixture_values,np.int8))
            assert np.array_equal(host_scales[0],np.array(fixture_scales,np.uint16))
        norms=[exact_norms(v,s) for v,s in zip(host_codes,host_scales)]
        expected=np.array([scales_from_norms(*n) for n in norms],np.uint16)
        # This file lets the unchanged Rust functions cross-check every tested row.
        filename=f'results/norm-inputs-{kind}-{k}.bin'
        pathlib.Path(filename).write_bytes(struct.pack('<II',m,k)+host_codes.tobytes()+host_scales.astype('<u2').tobytes())
        key=torch.zeros(32,dtype=torch.uint8,device='cuda')
        root_codes=torch.empty_like(key);root_scales=torch.empty_like(key)
        config=TensorHashConfig()
        roots=torch.empty(tensor_hash_workspace_bytes(m,k,config),dtype=torch.uint8,device='cuda')
        stats=torch.empty(2*m*k//512,dtype=torch.float32,device='cuda')
        def commit():
            tensor_hash_plus_stats_b(codes,scales,key,root_codes,root_scales,roots,stats,config=config)
        factor=torch.ones((R,k),dtype=torch.float8_e4m3fn,device='cuda')
        packed=pack_noise_factor(factor)
        alpha=torch.empty(m,dtype=torch.bfloat16,device='cuda');beta=torch.empty_like(alpha)
        noise=torch.empty((m,R),dtype=torch.float8_e4m3fn,device='cuda')
        primed=torch.empty((m,k),dtype=torch.float8_e4m3fn,device='cuda')
        peel=torch.empty((m,2*R),dtype=torch.bfloat16,device='cuda')
        def prepare(selected_stats=stats):
            noisy_quant(codes,scales,key,selected_stats,packed,factor,alpha,beta,noise,primed,peel)
        commit();prepare();torch.cuda.synchronize()
        actual=np.stack([alpha.view(torch.uint16).cpu().numpy(),beta.view(torch.uint16).cpu().numpy()],axis=1)
        mismatch=np.nonzero(np.any(actual!=expected,axis=1))[0]
        original_prime=primed.view(torch.uint8).cpu().numpy().copy()
        canonical=torch.empty((m,2),dtype=torch.uint16,device='cuda')
        fixed_stats=torch.empty_like(stats)
        def exact_gpu_norms():
            assert library.requant_norms(codes.data_ptr(),scales.data_ptr(),canonical.data_ptr(),m,k,
                                         torch.cuda.current_stream().cuda_stream)==0
        def inject_norms():
            assert library.requant_inject(canonical.data_ptr(),fixed_stats.data_ptr(),m,k,
                                          torch.cuda.current_stream().cuda_stream)==0
        exact_gpu_norms()
        assert np.array_equal(canonical.cpu().numpy(),np.array(norms,np.uint16))
        inject_norms();prepare(fixed_stats);torch.cuda.synchronize()
        fixed=np.stack([alpha.view(torch.uint16).cpu().numpy(),beta.view(torch.uint16).cpu().numpy()],axis=1)
        assert np.array_equal(fixed,expected),'Canonical norm adapter did not reproduce the scale chain'
        fixed_prime=primed.view(torch.uint8).cpu().numpy()
        changed_codes=int(np.count_nonzero(fixed_prime!=original_prime))
        graphs={}
        for name in ('original','canonical'):
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                pre_quant(x,codes,scales);commit()
                if name=='canonical':exact_gpu_norms();inject_norms();prepare(fixed_stats)
                else:prepare()
            graph.replay();torch.cuda.synchronize()
            observed=np.stack([alpha.view(torch.uint16).cpu().numpy(),beta.view(torch.uint16).cpu().numpy()],axis=1)
            assert np.array_equal(observed,expected if name=='canonical' else actual)
            graphs[name]=graph
        graph_samples={name:{'wall_ms':[],'device_elapsed_ms':[]} for name in graphs}
        for iteration in range(35):
            modes=list(graphs)
            if iteration%2:modes.reverse()
            for name in modes:
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start=time.perf_counter();begin.record();graphs[name].replay();end.record();end.synchronize()
                wall=(time.perf_counter()-start)*1000;elapsed=begin.elapsed_time(end)
                if iteration>=5:
                    graph_samples[name]['wall_ms'].append(wall)
                    graph_samples[name]['device_elapsed_ms'].append(elapsed)
        samples={mode:[] for mode in ('pre_quant','commit_stats','noisy_quant','canonical_norms',
                                      'canonical_noisy_quant','pre_quant_commit_noisy_quant','canonical_full_prep')}
        for iteration in range(35):
            modes=list(samples)
            if iteration%2:modes.reverse()
            for mode in modes:
                torch.cuda.synchronize();start=time.perf_counter()
                if mode in ('pre_quant','pre_quant_commit_noisy_quant','canonical_full_prep'):pre_quant(x,codes,scales)
                if mode in ('commit_stats','pre_quant_commit_noisy_quant','canonical_full_prep'):commit()
                if mode in ('noisy_quant','pre_quant_commit_noisy_quant'):prepare()
                if mode in ('canonical_norms','canonical_noisy_quant','canonical_full_prep'):exact_gpu_norms()
                if mode in ('canonical_noisy_quant','canonical_full_prep'):inject_norms();prepare(fixed_stats)
                torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
                if iteration>=5:samples[mode].append(elapsed)
        cases.append(dict(kind=kind,m=m,k=k,canonical_norms=norms,canonical_alpha_beta=expected.tolist(),
                          actual_alpha_beta=actual.tolist(),mismatching_rows=mismatch.tolist(),
                          fixed_alpha_beta=fixed.tolist(),gpu_canonical_norms_exact=True,changed_fp8_codes=changed_codes,
                          original_fp8_sha256=hashlib.sha256(original_prime.tobytes()).hexdigest(),
                          canonical_fp8_sha256=hashlib.sha256(fixed_prime.tobytes()).hexdigest(),
                          graph_samples_ms=graph_samples,
                          input_sha256=hashlib.sha256(pathlib.Path(filename).read_bytes()).hexdigest(),
                          samples_ms=samples))
        print(kind,k,'mismatching rows',len(mismatch),'of',m,flush=True)
    upstream_paths=['miner/pearl-gemm/src/pearl_gemm/tensor_hash_plus_stats/_block_stats.py',
                    'miner/pearl-gemm/src/pearl_gemm/noisy_quant/_quantization_ops.py',
                    'zk-pow/src/v4/api/prequant.rs','zk-pow/src/v4/circuit/scale_stark/stark.rs']
    record=dict(utc=datetime.now(timezone.utc).isoformat(),revision=REVISION,seed=args.seed,warmups=5,samples=30,
                device=torch.cuda.get_device_name(),torch=torch.__version__,torch_cuda=torch.version.cuda,
                packages={p:importlib.metadata.version(p) for p in ('nvidia-cutlass-dsl','quack-kernels','apache-tvm-ffi','cuda-python','numpy')},
                source_sha256={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in
                               ('protocol_study.py','experiments/protocol_norms.py','fixtures/norm-boundary.bin','cuda/canonical_norms.cu')},
                upstream_source_sha256={p:hashlib.sha256(subprocess.check_output(['git','-C',str(args.upstream),'show',REVISION+':'+p])).hexdigest() for p in upstream_paths},
                cases=cases,scope='Actual upstream prequant, commitment/stats and noising kernels. Fixed zero key and finite all-one factors isolate scale generation, not a valid proof/sampler/lottery. Independent integer GPU norms also feed a restricted synthetic-FP32-stats adapter, validated on these rows only, not a production fix. Timings include Python launch and synchronization, exclude JIT and allocation. Mismatches are recorded, not asserted away.')
    pathlib.Path(args.output).write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
