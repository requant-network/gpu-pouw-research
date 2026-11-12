"""Independent dense/MoE finalizer and wide-exponent direct-norm qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from experiments.protocol_norms import scales_from_norms


def main():
    assert __debug__
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('checkout',type=Path)
    args=parser.parse_args()
    subprocess.run(['git','-C',str(args.checkout),'apply','--reverse','--check',str(Path('patches/conformance.patch').resolve())],check=True)
    for package in ('miner-base','pearl-gemm'):sys.path.insert(0,str(args.checkout/'miner'/package/'src'))
    from miner_base.commitment_hash import a_keys,operand_digest
    from miner_base.params import OperandParams,MoeParams,HashId,encode_p_a
    from pearl_gemm.tensor_hash_plus_stats._host import _compile_finalize
    from pearl_gemm.tensor_hash_plus_stats._finalize_kernel import p_a_words
    from pearl_gemm._utils._stream import get_stream
    from pearl_gemm.noisy_quant._quantization_ops import _scale_chain_norms
    from pearl_gemm.noisy_quant._host import _scale_constants
    from pearl_gemm import noisy_quant
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    import cuda.bindings.driver as cuda_drv
    rng=np.random.default_rng(20261112);finalizers=[]
    def device(data):return torch.frombuffer(bytearray(data),dtype=torch.uint8).cuda()
    for moe in (False,True):
        for iteration in range(128):
            raw=[rng.bytes(32) for _ in range(4)];routing=rng.bytes(64) if moe else b''
            if iteration<2:raw=[bytes([iteration*255])*32]*4
            p_a=encode_p_a(OperandParams(256,HashId.BLAKE3_CHUNK_1024,4),MoeParams(8,HashId.BLAKE3_CHUNK_256,HashId.BLAKE3_CHUNK_512) if moe else None)
            tensors=[device(data) for data in raw];out=torch.empty(96,dtype=torch.uint8,device='cuda')
            route=device(routing) if moe else None
            compiled=_compile_finalize(torch.cuda.get_device_capability(),p_a_words(p_a),len(p_a))
            arguments=[from_dlpack(t.view(torch.uint32)) for t in tensors]
            arguments += [None if route is None else from_dlpack(route.view(torch.uint32)),from_dlpack(out.view(torch.uint32)),get_stream(0)]
            compiled(*arguments)
            expected=a_keys(operand_digest(raw[:2],raw[2]),raw[3],raw[2],p_a,routing).to_bytes()
            actual=bytes(out.cpu().numpy());assert actual==expected,(moe,iteration)
            finalizers.append(dict(moe=moe,roots=[v.hex() for v in raw[:2]],key=raw[2].hex(),seed_b=raw[3].hex(),
                                   routing=routing.hex(),p_a=p_a.hex(),a_keys=actual.hex()))
    invalid=0
    for value in (b'',bytes(8),bytes(10),bytes(12),bytes(13),bytearray(9),None):
        try:p_a_words(value)
        except ValueError:invalid+=1
        else:raise AssertionError('Malformed pA accepted')
    m,k=256,1024
    operands=[torch.zeros((m,k),dtype=torch.int8,device='cuda'),torch.ones((m,k//8),dtype=torch.bfloat16,device='cuda'),
              torch.zeros(32,dtype=torch.uint8,device='cuda'),torch.zeros(2*m*k//512,dtype=torch.float32,device='cuda'),
              torch.zeros((k,32),dtype=torch.int8,device='cuda'),torch.ones((32,k),device='cuda').to(torch.float8_e4m3fn),
              torch.empty(m,dtype=torch.bfloat16,device='cuda'),torch.empty(m,dtype=torch.bfloat16,device='cuda'),
              torch.empty((m,32),dtype=torch.float8_e4m3fn,device='cuda'),torch.empty((m,k),dtype=torch.float8_e4m3fn,device='cuda'),
              torch.empty((m,64),dtype=torch.bfloat16,device='cuda')]
    bad=[torch.ones((m,2),dtype=torch.float32,device='cuda'),torch.ones((m,1),dtype=torch.bfloat16,device='cuda')]
    for value in (float('nan'),float('inf'),-1.0):
        norms=torch.ones((m,2),dtype=torch.bfloat16,device='cuda');norms[0,0]=value;bad.append(norms)
    norms=torch.ones((m,2),dtype=torch.bfloat16,device='cuda');norms.view(torch.int16)[0,0]=0x3f81;bad.append(norms)
    norm_rejections=0
    for norms in bad:
        try:noisy_quant(*operands,canonical_norms=norms)
        except (ValueError,TypeError):norm_rejections+=1
        else:raise AssertionError('Invalid canonical norm tensor accepted')
    pairs=np.array([(code,other) for code in range(0,0x7b81,4) for other in (code,0x3f80)],np.uint16)
    expected=np.array([scales_from_norms(int(a),int(b)) for a,b in pairs],np.uint16)
    data=torch.from_numpy(pairs).cuda().view(torch.bfloat16);output=torch.empty_like(data)
    @cute.kernel
    def scale_kernel(norms:cute.Tensor,out:cute.Tensor,n:cutlass.Constexpr,delta:cutlass.Constexpr,over:cutlass.Constexpr):
        t,_,_=cute.arch.thread_idx();block,_,_=cute.arch.block_idx();i=block*128+t
        if i<n:
            alpha,beta=_scale_chain_norms(norms[(i,0)].to(cutlass.Float32),norms[(i,1)].to(cutlass.Float32),delta,over)
            out[(i,0)]=alpha.to(cutlass.BFloat16);out[(i,1)]=beta.to(cutlass.BFloat16)
    @cute.jit
    def launch(norms:cute.Tensor,out:cute.Tensor,stream:cuda_drv.CUstream,n:cutlass.Constexpr,delta:cutlass.Constexpr,over:cutlass.Constexpr):
        scale_kernel(norms,out,n,delta,over).launch(grid=((n+127)//128,1,1),block=(128,1,1),stream=stream)
    delta,over=_scale_constants(torch.cuda.get_device_capability());arguments=(from_dlpack(data),from_dlpack(output),get_stream(0))
    function=cute.compile(launch,*arguments,n=len(pairs),delta=delta,over=over);function(*arguments)
    actual=output.view(torch.uint16).cpu().numpy();mismatch=np.flatnonzero(np.any(actual!=expected,axis=1))
    record=dict(finalizer_checks=len(finalizers),finalizer_examples=finalizers,invalid_p_a_rejected=invalid,
                invalid_norm_tensors_rejected=norm_rejections,norm_scale_checks=len(pairs),norm_scale_mismatches=len(mismatch),
                mismatches=[dict(norms=pairs[i].tolist(),expected=expected[i].tolist(),actual=actual[i].tolist()) for i in mismatch[:32]],
                norm_input_sha256=hashlib.sha256(pairs.tobytes()).hexdigest(),norm_expected_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
                norm_actual_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                    ('conformance_validation.py','experiments/protocol_norms.py','patches/conformance.patch')},
                scope='Finalizer compares full96-byte GPU keys with canonical CPU BLAKE3 for128 dense/128 MoE trials, including zero/all-one inputs. Direct scale chain compared with independent rational arithmetic over finite BF16 grid norms through2^120 and two infinity-norm families. Does not claim exhaustive arbitrary-pair coverage or a general production security proof.')
    Path('results/conformance-validation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(len(finalizers),'exact finalizers;',len(pairs),'direct scale pairs;',len(mismatch),'mismatches')
    assert invalid==7 and norm_rejections==6 and not len(mismatch)


if __name__=='__main__':main()
