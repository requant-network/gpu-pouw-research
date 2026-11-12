"""Run a reproducible CPU baseline and generate the future CUDA check fixture."""
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/cpu-baseline.json')
    args = parser.parse_args()
    from experiments.quantization import experiment, fixture
    result = experiment()
    result['environment'] = {'python': platform.python_version(), 'os': platform.system(),
                             'architecture': platform.machine(), 'processor': platform.processor()}
    result['generated_at_utc'] = datetime.now(timezone.utc).isoformat()
    sources = [ROOT / 'run.py', *sorted((ROOT / 'experiments').glob('*.py')), ROOT / 'cuda/probe.cu']
    result['source_sha256'] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sources}
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    import struct
    first, second, a, b, expected = fixture()
    payload = struct.pack('<3I', len(a), first, second)
    for values in (a, b, expected):
        payload += struct.pack(f'<{len(values)}I', *values)
    (output.parent / 'gpu-fixture.bin').write_bytes(payload)
    print(json.dumps({'result': output.relative_to(ROOT).as_posix(), 'checks': 'passed', 'gpu_execution': 'not run'}))

if __name__ == '__main__':
    main()
