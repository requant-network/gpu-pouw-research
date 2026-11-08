"""Compare every stored GPU test input with the compiled upstream Rust norms."""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--study',type=pathlib.Path,default=pathlib.Path('results/protocol-study.json'))
parser.add_argument('--inputs-dir',type=pathlib.Path,default=pathlib.Path('results'))
parser.add_argument('--output',type=pathlib.Path,default=pathlib.Path('results/rust-norm-confirmation.json'))
args=parser.parse_args()
assert __debug__
study=json.loads(args.study.read_text());records=[]
binary=pathlib.Path('build/rust-norms/target/release/requant-upstream-norms'+('.exe' if os.name=='nt' else ''))
for case in study['cases']:
    filename=f"norm-inputs-{case['kind']}-{case['k']}.bin"
    path=args.inputs_dir/filename
    assert hashlib.sha256(path.read_bytes()).hexdigest()==case['input_sha256']
    actual=json.loads(subprocess.check_output([str(binary.resolve()),str(path)],text=True))
    assert actual==case['canonical_norms']
    records.append(dict(kind=case['kind'],k=case['k'],rows=len(actual),all_norms_equal=True,input_sha256=case['input_sha256']))
output=dict(revision=study['revision'],seed=study['seed'],norms_checked=sum(x['rows'] for x in records),cases=records,
            source_sha256={'confirm_rust_norms.py':hashlib.sha256(pathlib.Path('confirm_rust_norms.py').read_bytes()).hexdigest()},
            build_provenance='rust-norm-build.json',study_sha256=hashlib.sha256(args.study.read_bytes()).hexdigest())
args.output.write_text(json.dumps(output,indent=2)+'\n')
print(output['norms_checked'],'rows match unmodified upstream Rust exact_norms')
