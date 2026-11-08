"""Offline full GPU nonce pipeline and the pinned native plain-proof verifier."""
import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import pathlib
import subprocess
import sys
import time
from datetime import datetime,timezone

import numpy as np
import torch

REVISION='b7160514ed3d862fb48e850ba9a70a545cefcb1e'


def main():
    assert __debug__
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('upstream',type=pathlib.Path)
    parser.add_argument('--seed',type=int,default=20261111)
    parser.add_argument('--output',default='results/nonce-study.json')
    args=parser.parse_args();assert subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip()==REVISION
    for package in ('miner-base','pearl-gemm'):sys.path.insert(0,str(args.upstream/'miner'/package/'src'))
    import pearl_mining as pm
    from miner_base.commitment import commit_planes,commitment_keys
    from miner_base.commitment_hash import a_keys,noise_line_key,noise_seed_b,operand_digest
    from miner_base.mining_config import default_mining_config
    from miner_base.transcript import bits_to_target
    from pearl_gemm import (pre_quant,tensor_hash_plus_stats_b,tensor_hash_plus_stats,TensorHashConfig,
        tensor_hash_workspace_bytes,noise_lines,LABEL_F1,LABEL_F2,pack_noise_factor,noisy_quant,noisy_quant_b,
        mixed_gemm,HitSignal,HitSignalConfig)
    from pearl_gemm.protocol_constants import R
    rng=np.random.default_rng(args.seed);m=n=256
    parent=pm.BlockHeader(pm.IncompleteBlockHeader(1,b'\x11'*32,b'\x33'*32,1699999000,0x207fffff),b'\x44'*32)
    parent_bytes=bytes(parent.to_bytes())
    def header(iteration):return pm.IncompleteBlockHeader(1,bytes(parent.block_hash()),b'\x22'*32,1700000000+iteration,0x207fffff)
    def device_bytes(data):return torch.frombuffer(bytearray(data),dtype=torch.uint8).cuda()
    library=ctypes.CDLL(str(pathlib.Path('build/libcanonical_norms.so').resolve()))
    library.requant_norms.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int,ctypes.c_int,ctypes.c_void_p];library.requant_norms.restype=ctypes.c_int
    library.requant_inject.argtypes=[ctypes.c_void_p]*2+[ctypes.c_int,ctypes.c_int,ctypes.c_void_p];library.requant_inject.restype=ctypes.c_int
    def canonical(codes,scales,stats,workspace):
        rows,k=codes.shape;stream=torch.cuda.current_stream().cuda_stream
        assert library.requant_norms(codes.data_ptr(),scales.data_ptr(),workspace.data_ptr(),rows,k,stream)==0
        assert library.requant_inject(workspace.data_ptr(),stats.data_ptr(),rows,k,stream)==0
    cases=[]
    for pattern,k,repair in [('gaussian',1024,False),('gaussian',2048,False),('norm_boundary',1024,False),('norm_boundary',1024,True)]:
        setup_start=time.perf_counter();config=default_mining_config(k,R);hash_config=TensorHashConfig()
        a=rng.normal(0,.7,(m,k)).astype(np.float32);b=rng.normal(0,.7,(n,k)).astype(np.float32)
        if pattern=='norm_boundary':
            bits=np.frombuffer(pathlib.Path('fixtures/norm-boundary.bin').read_bytes(),'<u2',1024,1280)
            a[:]=((bits.astype(np.uint32)<<16).view(np.float32))[None,:]
        x=torch.from_numpy(a).to(device='cuda',dtype=torch.bfloat16);y=torch.from_numpy(b).to(device='cuda',dtype=torch.bfloat16)
        reference=x.cpu().float().numpy().astype(np.float64)@y.cpu().float().numpy().astype(np.float64).T
        a_codes=torch.empty((m,k),device='cuda',dtype=torch.int8);a_scales=torch.empty((m,k//8),device='cuda',dtype=torch.bfloat16)
        b_codes=torch.empty((n,k),device='cuda',dtype=torch.int8);b_scales=torch.empty((n,k//8),device='cuda',dtype=torch.bfloat16)
        pre_quant(x,a_codes,a_scales);pre_quant(y,b_codes,b_scales)
        a_cpu,a_scale_cpu=a_codes.cpu(),a_scales.cpu();b_cpu,b_scale_cpu=b_codes.cpu(),b_scales.cpu()
        key_a,key_b=commitment_keys(bytes(header(0).to_bytes()),parent_bytes)
        key_a_dev,key_b_dev=device_bytes(key_a),device_bytes(key_b)
        b_roots=[torch.empty(32,device='cuda',dtype=torch.uint8) for _ in range(2)]
        a_roots=[torch.empty(32,device='cuda',dtype=torch.uint8) for _ in range(2)]
        b_scratch=torch.empty(tensor_hash_workspace_bytes(n,k,hash_config),device='cuda',dtype=torch.uint8)
        a_scratch=torch.empty(tensor_hash_workspace_bytes(m,k,hash_config),device='cuda',dtype=torch.uint8)
        b_stats=torch.empty(2*n*k//512,device='cuda',dtype=torch.float32);a_stats=torch.empty(2*m*k//512,device='cuda',dtype=torch.float32)
        tensor_hash_plus_stats_b(b_codes,b_scales,key_b_dev,*b_roots,b_scratch,b_stats,config=hash_config)
        committed_b=commit_planes([b_cpu,b_scale_cpu],key_b,config.b_hash_id)
        gpu_b_digest=operand_digest([bytes(v.cpu().numpy()) for v in b_roots],key_b)
        assert gpu_b_digest==committed_b.digest
        seed_b=noise_seed_b(gpu_b_digest,key_b,config.p_b(n));seed_b_dev=device_bytes(seed_b);key_noise_b=device_bytes(noise_line_key(seed_b))
        bases=[]
        for label in (LABEL_F1,LABEL_F2):
            lines=torch.empty((k,R),device='cuda',dtype=torch.float8_e4m3fn);noise_lines(key_noise_b,label,lines);bases.append(lines.t().contiguous())
        f1,f2=bases
        norm_a=torch.empty((m,2),device='cuda',dtype=torch.uint16);norm_b=torch.empty((n,2),device='cuda',dtype=torch.uint16)
        if repair:canonical(b_codes,b_scales,b_stats,norm_b)
        alpha_b=torch.empty(n,device='cuda',dtype=torch.bfloat16);beta_b=torch.empty_like(alpha_b)
        eb=torch.empty((n,R),device='cuda',dtype=torch.float8_e4m3fn);bp=torch.empty((n,k),device='cuda',dtype=torch.float8_e4m3fn)
        b_peel=torch.empty((n,2*R),device='cuda',dtype=torch.bfloat16);gram=torch.empty((R,R),device='cuda',dtype=torch.float32)
        noisy_quant_b(b_codes,b_scales,key_noise_b,b_stats,pack_noise_factor(f2),f1,alpha_b,beta_b,eb,bp,b_peel,gram)
        inv_alpha_b=torch.reciprocal(alpha_b.float()).contiguous()
        keys_a=torch.empty(96,device='cuda',dtype=torch.uint8);packed_f1=pack_noise_factor(f1)
        try:
            tensor_hash_plus_stats(a_codes,a_scales,key_a_dev,seed_b_dev,*a_roots,a_scratch,keys_a,a_stats,p_a=config.p_a(m),config=hash_config)
        except ValueError as error:
            finalize_error=str(error)
            assert len(config.p_a(m))==9 and '11-byte' in finalize_error
        else:
            raise AssertionError('Pinned finalize incompatibility unexpectedly disappeared')
        alpha_a=torch.empty(m,device='cuda',dtype=torch.bfloat16);beta_a=torch.empty_like(alpha_a)
        ea=torch.empty((m,R),device='cuda',dtype=torch.float8_e4m3fn);ap=torch.empty((m,k),device='cuda',dtype=torch.float8_e4m3fn)
        a_peel=torch.empty((m,2*R),device='cuda',dtype=torch.bfloat16);out=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
        threshold=device_bytes(bits_to_target(0x207fffff).to_bytes(32,'little'));signal=HitSignal(HitSignalConfig(max_m=m,max_k=k))
        torch.cuda.synchronize();setup_ms=(time.perf_counter()-setup_start)*1000
        def matrix_proof(opening,mutate=False):
            p=opening.proof;leaves=[bytes(v) for v in p.leaf_data]
            if mutate:
                leaf=bytearray(leaves[0]);leaf[0]^=1;leaves[0]=bytes(leaf)
            return pm.MatrixMerkleProof(pm.MerkleProof(leaves,list(p.leaf_indices),bytes(p.root),[bytes(v) for v in p.siblings],p.total_leaves),list(opening.row_indices))
        def proof(comm_a,rows,cols,mutate=False):
            av,ass=comm_a.open(rows);bv,bss=committed_b.open(cols)
            return pm.PlainProofV4(parent,pm.CommonParams(k,R,pm.Quant.Fp8E4M3Prequant,pm.Device.B200),
                pm.OperandParams(m,pm.HashId.Blake3Chunk1024,config.tile_rows),pm.OperandParams(n,pm.HashId.Blake3Chunk1024,config.tile_cols),
                matrix_proof(av,mutate),matrix_proof(bv),matrix_proof(ass),matrix_proof(bss))
        samples={name:[] for name in ('gpu_nonce_wall_ms','proof_build_ms','native_plain_verify_ms','complete_local_nonce_ms')}
        results=[];max_error=0
        for iteration in range(35):
            start_all=time.perf_counter();h=header(iteration);ka,_=commitment_keys(bytes(h.to_bytes()),parent_bytes)
            key_a_dev.copy_(torch.frombuffer(bytearray(ka),dtype=torch.uint8));signal.reset_hit();torch.cuda.synchronize()
            start_gpu=time.perf_counter()
            pre_quant(x,a_codes,a_scales)
            # The pin emits 9-byte pA but its GPU finalize accepts only 11/13 bytes.
            # Bridge key finalization while preserving both upstream implementations.
            tensor_hash_plus_stats_b(a_codes,a_scales,key_a_dev,*a_roots,a_scratch,a_stats,config=hash_config)
            digest_a=operand_digest([bytes(v.cpu().numpy()) for v in a_roots],ka)
            keys_a.copy_(device_bytes(a_keys(digest_a,seed_b,ka,config.p_a(m)).to_bytes()))
            if repair:canonical(a_codes,a_scales,a_stats,norm_a)
            noisy_quant(a_codes,a_scales,keys_a[32:64],a_stats,packed_f1,f2,alpha_a,beta_a,ea,ap,a_peel)
            mixed_gemm(ap,bp,a_peel,b_peel,alpha_a,inv_alpha_b,keys_a[64:96],threshold,out,signal,a_codes,a_scales,seed_b_dev)
            torch.cuda.synchronize();gpu_ms=(time.perf_counter()-start_gpu)*1000
            hit=signal.read_hit();assert hit is not None and hit.valid
            if hit.codes is not None:assert torch.equal(hit.codes,a_cpu) and torch.equal(hit.scales,a_scale_cpu)
            rows=list(range(hit.tile_row*hit.ltile_rows,(hit.tile_row+1)*hit.ltile_rows))
            cols=list(range(hit.tile_column*hit.ltile_cols,(hit.tile_column+1)*hit.ltile_cols))
            start=time.perf_counter();committed_a=commit_planes([a_cpu,a_scale_cpu],ka,config.a_hash_id)
            assert bytes(keys_a.cpu().numpy())==a_keys(committed_a.digest,seed_b,ka,config.p_a(m)).to_bytes()
            artifact=proof(committed_a,rows,cols);build_ms=(time.perf_counter()-start)*1000
            start=time.perf_counter();accepted,message=pm.verify_plain_proof_v4(h,artifact);verify_ms=(time.perf_counter()-start)*1000
            complete_ms=(time.perf_counter()-start_all)*1000
            if pattern=='gaussian' or repair:assert accepted,message
            if iteration==0:
                bad_header=pm.IncompleteBlockHeader(h.version,h.prev_block,h.merkle_root,h.timestamp+10000,h.nbits)
                assert not pm.verify_plain_proof_v4(bad_header,artifact)[0]
                assert not pm.verify_plain_proof_v4(h,proof(committed_a,rows,cols,True))[0]
            result=out.cpu().float().numpy().astype(np.float64)
            relative=float(np.linalg.norm(result-reference)/np.linalg.norm(reference));assert np.isfinite(relative)
            max_error=max(max_error,relative)
            results.append(dict(iteration=iteration,accepted=accepted,message=message,rows=rows,columns=cols,
                                proof_bytes=len(artifact.to_bytes()),header_sha256=hashlib.sha256(bytes(h.to_bytes())).hexdigest()))
            if iteration>=5:
                for name,value in zip(samples,(gpu_ms,build_ms,verify_ms,complete_ms)):samples[name].append(value)
        cases.append(dict(pattern=pattern,k=k,m=m,n=n,canonical_repair=repair,setup_ms=setup_ms,
                          upstream_finalize_rejection=finalize_error,p_a_bytes=len(config.p_a(m)),
                          accepted=sum(c['accepted'] for c in results),attempts=len(results),
                          max_useful_output_relative_error=max_error,negative_controls_rejected=2,samples_ms=samples,verification_results=results))
        print(pattern,k,'repair',repair,'accepted',sum(c['accepted'] for c in results))
    record=dict(utc=datetime.now(timezone.utc).isoformat(),revision=REVISION,seed=args.seed,warmups=5,samples=30,
                source_sha256={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in ('nonce_study.py','cuda/canonical_norms.cu','fixtures/norm-boundary.bin')},
                native_packages={p:importlib.metadata.version(p) for p in ('py-pearl-mining','pearl-blake3','torch','nvidia-cutlass-dsl')},cases=cases,
                compatibility_bridge='Pinned miner-base emits dense pA of 9 bytes; pinned GPU finalize rejects it (expects 11/13). Harness uses unchanged GPU hash/stats plus canonical host a_keys, not the incompatible GPU finalize.',
                scope='Offline synthetic headers and easy target 0x207fffff. Actual upstream keyed commitments, rank-32 noise, FP8 GEMM, peeling, ticket publication, Merkle openings and native full plain-proof verification. Fresh header per iteration; job-constant B preparation excluded from nonce timing. GPU nonce wall timer includes host key-finalization bridge. Complete local timer includes header/key preparation, proof building and verification, excludes correctness diagnostics and ZK-certificate generation. No network or reward claim.')
    pathlib.Path(args.output).write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
