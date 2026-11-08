"""Bind native verifier builds and executed study records to a clean upstream pin."""
import argparse
import hashlib
import importlib.metadata
import json
import pathlib
import subprocess
import zipfile
from datetime import datetime, timezone

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('upstream',type=pathlib.Path)
parser.add_argument('wheels',type=pathlib.Path)
args=parser.parse_args()
def command(*values):return subprocess.check_output(values,text=True).strip()
def digest(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
revision=command('git','-C',str(args.upstream),'rev-parse','HEAD')
assert revision=='b7160514ed3d862fb48e850ba9a70a545cefcb1e'
assert not command('git','-C',str(args.upstream),'status','--porcelain')
results=[pathlib.Path('results')/name for name in ('nonce-study.json','nonce-replication.json')]
for p in results:assert json.loads(p.read_text())['revision']==revision
wheels=[p for p in args.wheels.glob('*.whl') if p.name.startswith(('py_pearl_mining-','pearl_blake3-'))]
assert len(wheels)==2
installed_native={}
for wheel in wheels:
    package='py-pearl-mining' if wheel.name.startswith('py_pearl_mining-') else 'pearl-blake3'
    distribution=importlib.metadata.distribution(package)
    with zipfile.ZipFile(wheel) as archive:
        members=[name for name in archive.namelist() if name.endswith('.so')]
        assert members,wheel.name
        for name in members:
            expected=hashlib.sha256(archive.read(name)).hexdigest()
            assert digest(distribution.locate_file(name))==expected,(wheel.name,name)
            installed_native[name]=expected
record=dict(utc=datetime.now(timezone.utc).isoformat(),revision=revision,upstream_clean=True,
    rustc=command('rustc','--version'),maturin=command('maturin','--version'),
    native_build_flags=['maturin','build','--release','--locked','--jobs','3'],rustc_bootstrap='1',
    cargo_lock_sha256={str(p.relative_to(args.upstream)):digest(p) for p in args.upstream.glob('*/Cargo.lock')},wheel_sha256={p.name:digest(p) for p in wheels},
    source_sha256={'collect_pearl_build.py':digest('collect_pearl_build.py')},
    result_sha256={p.name:digest(p) for p in results},
    native_packages={p:importlib.metadata.version(p) for p in ('py-pearl-mining','pearl-blake3')},
    installed_native_sha256=installed_native,installed_native_matches_wheels=True,
    scope='Both native wheels built from the clean pinned upstream source, release/locked, with Rust 1.98.1 and RUSTC_BOOTSTRAP=1. Full native plain-proof verifier executed; no ZK certificate generated or verified.')
pathlib.Path('results/pearl-build.json').write_text(json.dumps(record,indent=2)+'\n')
