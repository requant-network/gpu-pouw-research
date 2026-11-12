"""Qualify the pinned native FP8 ZK certificate on one public synthetic proof."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def worker(folder, destination):
    import resource
    import pearl_mining as pm
    data = json.loads((folder / 'header.json').read_text())
    header = pm.IncompleteBlockHeader(data['version'], bytes.fromhex(data['prev_block']),
                                      bytes.fromhex(data['merkle_root']), data['timestamp'], data['nbits'])
    plain = pm.PlainProofV4.from_bytes((folder / 'proof.plain').read_bytes())
    assert pm.verify_plain_proof_v4(header, plain)[0]
    phases = {}
    def phase(name, function):
        print('PHASE ' + name, flush=True)
        start = time.monotonic()
        result = function()
        phases[name] = time.monotonic() - start
        return result
    prover = phase('prover_setup', lambda: pm.Fp8Prover.setup(pm.Device.B200))
    public, proof = phase('prove', lambda: prover.prove(header, plain))
    verifier = phase('verifier_setup', lambda: pm.Fp8Verifier.generate(public))
    phase('verify', lambda: verifier.verify_block(header, [], public, proof))
    bad_header = pm.IncompleteBlockHeader(data['version'], bytes.fromhex(data['prev_block']),
                                         bytes.fromhex(data['merkle_root']), data['timestamp'] + 1, data['nbits'])
    damaged = bytearray(proof); damaged[len(damaged) // 2] ^= 1
    wrong_public = bytearray(public); wrong_public[len(wrong_public) // 2] ^= 1
    for h, p, z in ((bad_header, public, proof), (header, public, bytes(damaged)),
                    (header, bytes(wrong_public), proof)):
        try:
            verifier.verify_block(h, [], p, z)
        except Exception:
            pass
        else:
            raise AssertionError('Altered certificate accepted')
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'statement.bin').write_bytes(bytes(public))
    (destination / 'proof.zk').write_bytes(bytes(proof))
    print('RESULT ' + json.dumps(dict(accepted=True, negative_controls_rejected=3,
          phases_seconds=phases, proof_bytes=len(proof), statement_bytes=len(public),
          process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)), flush=True)


def main():
    assert __debug__
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact', type=Path)
    parser.add_argument('--destination', type=Path, default=Path('artifacts/zk-certificate'))
    parser.add_argument('--output', type=Path, default=Path('results/zk-certificate.json'))
    parser.add_argument('--max-seconds', type=int, default=900)
    parser.add_argument('--memory-gib', type=int, default=48)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.artifact, args.destination)
        return
    import resource
    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (args.memory_gib << 30,) * 2)
    command = [sys.executable, __file__, str(args.artifact), '--worker', '--destination', str(args.destination)]
    start = time.monotonic()
    try:
        result = subprocess.run(command, capture_output=True, timeout=args.max_seconds, preexec_fn=limits)
        code, stdout, stderr = result.returncode, result.stdout, result.stderr
        timed_out = False
    except subprocess.TimeoutExpired as error:
        code, stdout, stderr = None, error.stdout or b'', error.stderr or b''
        timed_out = True
    log = Path('build/zk-certificate.log'); log.parent.mkdir(exist_ok=True)
    log.write_bytes(stdout + b'\n' + stderr)
    lines = stdout.decode(errors='replace').splitlines()
    complete = [json.loads(line[7:]) for line in lines if line.startswith('RESULT ')]
    record = dict(exit_code=code, timed_out=timed_out, wall_seconds=time.monotonic() - start,
                  memory_limit_gib=args.memory_gib, timeout_seconds=args.max_seconds,
                  completed_phases=[line[6:] for line in lines if line.startswith('PHASE ')],
                  result=complete[-1] if complete else None,
                  source_sha256={'zk_certificate_study.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                  input_artifacts_sha256={p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in (args.artifact / 'header.json', args.artifact / 'proof.plain')},
                  output_artifacts_sha256={p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in args.destination.glob('*') if p.is_file()},
                  scope='Unmodified pinned native CPU certificate API, synthetic public proof. One feasibility observation with explicit address-space/time budget; failure within that budget is not a universal impossibility claim. Full raw process output stays private under build/.')
    args.output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
