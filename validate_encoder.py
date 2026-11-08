"""Check both noised operands against the independent CPU FP8 encoder."""
import hashlib
import json
import pathlib
from datetime import datetime, timezone

import cupy as cp
import numpy as np
from experiments.quantization import encode
from gpu_study import quantize

assert __debug__, 'Run without -O'
rng=np.random.default_rng(424242)
checks=0
for sigma in (0.,.015625,.25,1.):
    for sign in (-1.,1.):
        a=rng.normal(0,.7,32768).astype(np.float32)
        a[:8]=[0.,-0.,448.,-448.,1e10,-1e10,2**-10,-2**-10]
        noise=rng.choice([-2.,0.,2.],len(a)).astype(np.float32)
        q=cp.empty(len(a),cp.uint8)
        scale=np.float32(sigma*sign)
        quantize(((len(a)+255)//256,),(256,),
                 (cp.asarray(a),cp.asarray(noise),q,np.uint64(0),np.int32(len(a)),scale))
        # Two explicit FP32 operations match the CUDA round-to-nearest operations.
        noised=a if scale==0 else a+scale*noise
        expected=np.array([encode(float(x)) for x in noised],np.uint8)
        assert np.array_equal(cp.asnumpy(q),expected)
        checks+=len(a)
result=dict(utc=datetime.now(timezone.utc).isoformat(),seed=424242,exact_encoder_checks=checks,
            positive_and_negative_noise=True,signed_zero_and_saturation=True,
            source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           (pathlib.Path('validate_encoder.py'),pathlib.Path('gpu_study.py'),pathlib.Path('experiments/quantization.py'))})
pathlib.Path('results/encoder-validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(checks,'exact encoder checks passed')
