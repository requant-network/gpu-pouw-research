"""Paired direct-norm GPU pipelines, GPU key finalization and native share-target checks."""
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
from experiments.protocol_norms import exact_norms,scales_from_norms

import numpy as np
import torch

REVISION='b7160514ed3d862fb48e850ba9a70a545cefcb1e'


def main():
    assert __debug__
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('upstream',type=pathlib.Path)
    parser.add_argument('--seed',type=int,default=20261111)
    parser.add_argument('--output',default='results/conformance-study.json')
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--patch',default='patches/conformance.patch')
    parser.add_argument('--share-nbits',type=lambda s:int(s,0),default=0x207fffff)
    args=parser.parse_args();assert subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip()==REVISION
    subprocess.run(['git','-C',str(args.upstream),'apply','--reverse','--check',str(pathlib.Path(args.patch).resolve())],check=True)
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
        # Direct canonical BF16 pairs go into the patched noising API.
    cases=[];warmups=1 if args.quick else 5;repetitions=1 if args.quick else 30
    for pattern,k,repair in [('gaussian',1024,False),('gaussian',1024,True),('gaussian',2048,False),('gaussian',2048,True),('norm_boundary',1024,False),('norm_boundary',1024,True)]:
        rng=np.random.default_rng(args.seed+k+(0 if pattern=='gaussian' else 1))
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
        if repair:
            canonical(b_codes,b_scales,b_stats,norm_b)
            observed=norm_b.cpu().numpy()
            for row in (0,1,127,255):assert tuple(observed[row])==exact_norms(b_cpu[row].numpy(),b_scale_cpu.view(torch.uint16)[row].numpy())
            expected_b=np.array([scales_from_norms(int(a),int(b)) for a,b in observed],np.uint16)
        alpha_b=torch.empty(n,device='cuda',dtype=torch.bfloat16);beta_b=torch.empty_like(alpha_b)
        eb=torch.empty((n,R),device='cuda',dtype=torch.float8_e4m3fn);bp=torch.empty((n,k),device='cuda',dtype=torch.float8_e4m3fn)
        b_peel=torch.empty((n,2*R),device='cuda',dtype=torch.bfloat16);gram=torch.empty((R,R),device='cuda',dtype=torch.float32)
        noisy_quant_b(b_codes,b_scales,key_noise_b,b_stats,pack_noise_factor(f2),f1,alpha_b,beta_b,eb,bp,b_peel,gram,canonical_norms=norm_b.view(torch.bfloat16) if repair else None)
        if repair:
            assert np.array_equal(alpha_b.view(torch.uint16).cpu().numpy(),expected_b[:,0])
            assert np.array_equal(beta_b.view(torch.uint16).cpu().numpy(),expected_b[:,1])
        inv_alpha_b=torch.reciprocal(alpha_b.float()).contiguous()
        keys_a=torch.empty(96,device='cuda',dtype=torch.uint8);packed_f1=pack_noise_factor(f1)
        tensor_hash_plus_stats(a_codes,a_scales,key_a_dev,seed_b_dev,*a_roots,a_scratch,keys_a,a_stats,p_a=config.p_a(m),config=hash_config)
        finalize_error=None
        alpha_a=torch.empty(m,device='cuda',dtype=torch.bfloat16);beta_a=torch.empty_like(alpha_a)
        ea=torch.empty((m,R),device='cuda',dtype=torch.float8_e4m3fn);ap=torch.empty((m,k),device='cuda',dtype=torch.float8_e4m3fn)
        a_peel=torch.empty((m,2*R),device='cuda',dtype=torch.bfloat16);out=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
        work=config.tile_rows*config.tile_cols*k
        share_bound=min((1<<256)-1,bits_to_target(args.share_nbits)*work)
        threshold=device_bytes(share_bound.to_bytes(32,'little'));signal=HitSignal(HitSignalConfig(max_m=m,max_k=k))
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
        results=[];max_error=0;negative_controls=0;public_artifacts={};artifact_saved=False
        for iteration in range(warmups+repetitions):
            start_all=time.perf_counter();h=header(iteration);ka,_=commitment_keys(bytes(h.to_bytes()),parent_bytes)
            key_a_dev.copy_(torch.frombuffer(bytearray(ka),dtype=torch.uint8));signal.reset_hit();torch.cuda.synchronize()
            start_gpu=time.perf_counter()
            pre_quant(x,a_codes,a_scales)
            tensor_hash_plus_stats(a_codes,a_scales,key_a_dev,seed_b_dev,*a_roots,a_scratch,keys_a,a_stats,p_a=config.p_a(m),config=hash_config)
            if repair:canonical(a_codes,a_scales,a_stats,norm_a)
            noisy_quant(a_codes,a_scales,keys_a[32:64],a_stats,packed_f1,f2,alpha_a,beta_a,ea,ap,a_peel,canonical_norms=norm_a.view(torch.bfloat16) if repair else None)
            mixed_gemm(ap,bp,a_peel,b_peel,alpha_a,inv_alpha_b,keys_a[64:96],threshold,out,signal,a_codes,a_scales,seed_b_dev)
            torch.cuda.synchronize();gpu_ms=(time.perf_counter()-start_gpu)*1000
            if repair:
                observed=norm_a.cpu().numpy()
                for row in (0,1,127,255):assert tuple(observed[row])==exact_norms(a_cpu[row].numpy(),a_scale_cpu.view(torch.uint16)[row].numpy())
                expected_a=np.array([scales_from_norms(int(a),int(b)) for a,b in observed],np.uint16)
                assert np.array_equal(alpha_a.view(torch.uint16).cpu().numpy(),expected_a[:,0])
                assert np.array_equal(beta_a.view(torch.uint16).cpu().numpy(),expected_a[:,1])
            hit=signal.read_hit()
            if hit is None:
                complete_ms=(time.perf_counter()-start_all)*1000
                results.append(dict(iteration=iteration,found=False,accepted=False,header_sha256=hashlib.sha256(bytes(h.to_bytes())).hexdigest()))
                if iteration>=warmups:
                    for name,value in zip(samples,(gpu_ms,0.0,0.0,complete_ms)):samples[name].append(value)
                continue
            assert hit.valid
            if hit.codes is not None:assert torch.equal(hit.codes,a_cpu) and torch.equal(hit.scales,a_scale_cpu)
            rows=list(range(hit.tile_row*hit.ltile_rows,(hit.tile_row+1)*hit.ltile_rows))
            cols=list(range(hit.tile_column*hit.ltile_cols,(hit.tile_column+1)*hit.ltile_cols))
            start=time.perf_counter();committed_a=commit_planes([a_cpu,a_scale_cpu],ka,config.a_hash_id)
            assert bytes(keys_a.cpu().numpy())==a_keys(committed_a.digest,seed_b,ka,config.p_a(m)).to_bytes()
            artifact=proof(committed_a,rows,cols);build_ms=(time.perf_counter()-start)*1000
            start=time.perf_counter();accepted,message=pm.verify_plain_proof_v4(h,artifact,args.share_nbits);verify_ms=(time.perf_counter()-start)*1000
            complete_ms=(time.perf_counter()-start_all)*1000
            # Retain every rejection as evidence rather than losing the campaign.
            if negative_controls==0:
                bad_header=pm.IncompleteBlockHeader(h.version,h.prev_block,h.merkle_root,h.timestamp+10000,h.nbits)
                assert not pm.verify_plain_proof_v4(bad_header,artifact)[0]
                assert not pm.verify_plain_proof_v4(h,proof(committed_a,rows,cols,True))[0]
                negative_controls=2
            result=out.cpu().float().numpy().astype(np.float64)
            relative=float(np.linalg.norm(result-reference)/np.linalg.norm(reference));assert np.isfinite(relative)
            max_error=max(max_error,relative)
            share_verdicts={}
            for bits in (0x20000001,0x1f000001,0x1e000001):
                ok,why=pm.verify_plain_proof_v4(h,artifact,bits)
                share_verdicts[hex(bits)]=dict(accepted=ok,message=why)
            if not artifact_saved:
                folder=pathlib.Path('artifacts')/f'conformance-{pathlib.Path(args.patch).stem}-{args.seed}-{args.share_nbits:08x}-{pattern}-{k}-{int(repair)}'
                folder.mkdir(parents=True,exist_ok=True)
                (folder/'proof.plain').write_bytes(bytes(artifact.to_bytes()))
                (folder/'header.json').write_text(json.dumps(dict(version=h.version,prev_block=bytes(h.prev_block).hex(),merkle_root=bytes(h.merkle_root).hex(),timestamp=h.timestamp,nbits=h.nbits),indent=2)+'\n')
                public_artifacts={p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
                artifact_saved=True
            results.append(dict(found=True,share_verdicts=share_verdicts,iteration=iteration,accepted=accepted,message=message,rows=rows,columns=cols,
                                proof_bytes=len(artifact.to_bytes()),header_sha256=hashlib.sha256(bytes(h.to_bytes())).hexdigest()))
            if iteration>=warmups:
                for name,value in zip(samples,(gpu_ms,build_ms,verify_ms,complete_ms)):samples[name].append(value)
        cases.append(dict(public_artifacts_sha256=public_artifacts,found=sum(c.get('found',False) for c in results),pattern=pattern,k=k,m=m,n=n,canonical_repair=repair,setup_ms=setup_ms,
                          upstream_finalize_rejection=finalize_error,p_a_bytes=len(config.p_a(m)),
                          accepted=sum(c['accepted'] for c in results),attempts=len(results),
                          max_useful_output_relative_error=max_error,negative_controls_rejected=negative_controls,samples_ms=samples,verification_results=results))
        print(pattern,k,'repair',repair,'accepted',sum(c['accepted'] for c in results))
    modified_names=subprocess.check_output(['git','-C',str(args.upstream),'diff','--name-only'],text=True).splitlines()
    record=dict(patch=args.patch,upstream_modified_sha256={p:hashlib.sha256((args.upstream/p).read_bytes()).hexdigest() for p in modified_names},share_nbits=hex(args.share_nbits),share_target_rule="min(2^256-1, compact_target * h * w * k)",utc=datetime.now(timezone.utc).isoformat(),revision=REVISION,seed=args.seed,warmups=warmups,samples=repetitions,
                source_sha256={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in ('conformance_study.py','prepare_conformance.py',args.patch,'cuda/canonical_norms.cu','fixtures/norm-boundary.bin','experiments/protocol_norms.py')},
                native_packages={p:importlib.metadata.version(p) for p in ('py-pearl-mining','pearl-blake3','torch','nvidia-cutlass-dsl')},cases=cases,
                compatibility_bridge='Isolated attributed patch: dense pA=9/MoE pA=11; patch identity distinguishes historical lane-strided lottery from corrected contiguous row-major extraction; all A keys computed on GPU and compared byte-for-byte with canonical host derivation. Canonical variant consumes exact BF16 norm pairs directly; no stats injection or host key bridge.',
                scope='Paired B payloads across boundary variants. Correctness qualification only: incidental latency samples are not a fair dedicated performance campaign. Optional hard share targets are checked by the unchanged native verifier on the same proof; this does not establish bitwise ticket equality. Offline synthetic headers and easy target 0x207fffff. Actual upstream keyed commitments, rank-32 noise, FP8 GEMM, peeling, ticket publication, Merkle openings and native full plain-proof verification. Fresh header per iteration; job-constant B preparation excluded from nonce timing. GPU nonce wall timer uses GPU key finalization and includes direct-norm API validation synchronization when selected. Complete local timer includes header/key preparation, proof building and verification, excludes correctness diagnostics and ZK-certificate generation. No network or reward claim.')
    pathlib.Path(args.output).write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
