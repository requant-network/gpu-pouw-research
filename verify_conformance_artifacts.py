"""Recheck retained plain and ZK artifacts with the unchanged native verifier."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def header(data,pm):
    return pm.IncompleteBlockHeader(data['version'],bytes.fromhex(data['prev_block']),
                                    bytes.fromhex(data['merkle_root']),data['timestamp'],data['nbits'])


def main():
    assert __debug__
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='results/conformance-native-replay.json')
    parser.add_argument('--skip-zk',action='store_true')
    args=parser.parse_args()
    import pearl_mining as pm
    observations=[]
    for path in sorted(Path('results').glob('conformance-*.json')):
        record=json.loads(path.read_text())
        for case in record.get('cases',[]):
            if not case['public_artifacts_sha256']:continue
            folder=Path(next(iter(case['public_artifacts_sha256']))).parent
            data=json.loads((folder/'header.json').read_text());h=header(data,pm)
            proof=pm.PlainProofV4.from_bytes((folder/'proof.plain').read_bytes())
            expected=next(t for t in case['verification_results'] if t.get('found'))
            assert hashlib.sha256(bytes(h.to_bytes())).hexdigest()==expected['header_sha256']
            accepted,message=pm.verify_plain_proof_v4(h,proof,int(record['share_nbits'],0))
            assert accepted==expected['accepted'],(folder,message)
            observations.append(dict(artifact=folder.as_posix(),share_nbits=record['share_nbits'],accepted=accepted,message=message))
    certificates=[]
    if not args.skip_zk:
        for path in sorted(Path('results').glob('zk-certificate*.json')):
            record=json.loads(path.read_text());input_folder=Path(next(iter(record['input_artifacts_sha256']))).parent
            output_folder=Path(next(iter(record['output_artifacts_sha256']))).parent
            h=header(json.loads((input_folder/'header.json').read_text()),pm)
            public=(output_folder/'statement.bin').read_bytes();proof=(output_folder/'proof.zk').read_bytes()
            start=time.monotonic();verifier=pm.Fp8Verifier.generate(public);verifier.verify_block(h,[],public,proof)
            damaged=bytearray(proof);damaged[len(damaged)//2]^=1
            try:verifier.verify_block(h,[],public,bytes(damaged))
            except Exception:pass
            else:raise AssertionError('Altered ZK proof accepted')
            certificates.append(dict(artifact=output_folder.as_posix(),accepted=True,altered_rejected=True,seconds=time.monotonic()-start))
            del verifier
    result=dict(plain_verdicts=observations,zk_certificates=certificates,
                source_sha256={'verify_conformance_artifacts.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                scope='Fresh CPU replay of published plain artifacts at their declared share targets, including rejected historical controls; original pinned native verifier. Optional ZK setup/replay checks both new certificates and altered proofs. No GPU execution.')
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(len(observations),'plain artifacts and',len(certificates),'ZK certificates replayed')


if __name__=='__main__':main()
