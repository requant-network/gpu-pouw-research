"""Prepare an isolated, pinned Pearl copy with the conformance patch."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

REVISION = 'b7160514ed3d862fb48e850ba9a70a545cefcb1e'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('upstream', type=Path)
    parser.add_argument('--destination', type=Path, default=Path('build/pearl-conformance-v2'))
    args = parser.parse_args()
    assert subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip() == REVISION
    patch = Path('patches/conformance.patch').resolve()
    destination = args.destination
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--no-hardlinks', str(args.upstream.resolve()), str(destination)], check=True)
        subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', REVISION], check=True)
        subprocess.run(['git', '-C', str(destination), 'apply', '--check', str(patch)], check=True)
        subprocess.run(['git', '-C', str(destination), 'apply', str(patch)], check=True)
    assert subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip() == REVISION
    subprocess.run(['git', '-C', str(destination), 'apply', '--reverse', '--check', str(patch)], check=True)
    names = subprocess.check_output(['git', '-C', str(destination), 'diff', '--name-only'], text=True).splitlines()
    assert len(names) == 7
    record = dict(revision=REVISION,
                  source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                 for p in ('prepare_conformance.py', 'patches/conformance.patch')},
                  patched_source_sha256={p: hashlib.sha256((destination / p).read_bytes()).hexdigest() for p in names})
    Path('results').mkdir(exist_ok=True)
    Path('results/conformance-preparation.json').write_text(json.dumps(record, indent=2) + '\n')
    print('Prepared seven pinned source changes; original upstream is unchanged')


if __name__ == '__main__':
    main()
