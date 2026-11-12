"""Replay conformance evidence, source bindings and the lottery layout invariant."""
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import struct
from blake3 import blake3
from experiments.protocol_norms import scales_from_norms


def fold(words):
    value=0
    for word in words:
        value=(value*0x9e3779b1+word)&0xffffffff
        value=((value<<13)|(value>>19))&0xffffffff
    return value


def layout_check():
    rng=random.Random(20261112);checks=0
    for width in (64,128,192,256):
        matrix=[[rng.getrandbits(32) for _ in range(width)] for _ in range(64)]
        for row_half in range(2):
            for warp in range(4):
                messages={}
                for lane in range(32):
                    chunk=lane&3;row=16*warp+(lane>>2)+8*row_half
                    words=[]
                    for block in range(width//32):
                        for owner in range(4):
                            for column in range(2):
                                nb=chunk*(width//32)+block
                                words.append(matrix[row][8*nb+2*owner+column])
                    message=4*warp+2*row_half+(lane>>4)
                    messages.setdefault(message,[None]*16)[lane&15]=fold(words)
                for message,actual in messages.items():
                    flat=sum(matrix[4*message:4*message+4],[])
                    size=len(flat)//16
                    expected=[fold(flat[i*size:(i+1)*size]) for i in range(16)]
                    assert actual==expected,(width,message)
                    checks+=16
    return checks


def binding(name,expected):
    path=Path(name)
    assert not path.is_absolute() and ':' not in name and '..' not in path.parts
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name


def main():
    assert __debug__
    rows=[];canonical_found=canonical_accepted=historical_rejections=0;artifacts=set()
    paths=sorted(Path('results').glob('conformance-*.json'))+sorted(Path('results').glob('zk-certificate*.json'))
    assert len(paths)==13
    for path in paths:
        record=json.loads(path.read_text())
        for name,expected in record.get('source_sha256',{}).items():binding(name,expected)
        for key in ('input_artifacts_sha256','output_artifacts_sha256'):
            for name,expected in record.get(key,{}).items():binding(name,expected);artifacts.add(name)
        if path.name=='conformance-validation.json':
            assert record['finalizer_checks']==len(record['finalizer_examples'])==256
            assert record['invalid_p_a_rejected']==7 and record['invalid_norm_tensors_rejected']==6
            for example in record['finalizer_examples']:
                decode=lambda key:bytes.fromhex(example[key])
                roots=b''.join(bytes.fromhex(v) for v in example['roots'])
                key=decode('key');routing=decode('routing');p_a=decode('p_a')
                assert len(routing)==(64 if example['moe'] else 0) and len(p_a)==(11 if example['moe'] else 9)
                h=blake3(roots,key=key).digest()
                seed=blake3(h+routing+decode('seed_b')+key+p_a,key=blake3(b'pearl/v4/FP8/seed-A').digest()).digest()
                expected=seed+b''.join(blake3(label,key=seed).digest() for label in (b'pearl/v4/FP8/noise-line',b'pearl/v4/FP8/jackpot'))
                assert expected==decode('a_keys')
            pairs=[(a,b) for a in range(0,0x7b81,4) for b in (a,0x3f80)]
            packed=lambda values:b''.join(struct.pack('<HH',*v) for v in values)
            expected=packed(scales_from_norms(a,b) for a,b in pairs)
            assert record['norm_scale_checks']==len(pairs) and record['norm_scale_mismatches']==0
            assert hashlib.sha256(packed(pairs)).hexdigest()==record['norm_input_sha256']
            assert hashlib.sha256(expected).hexdigest()==record['norm_expected_sha256']==record['norm_actual_sha256']
        if path.name=='conformance-native-replay.json':
            assert len(record['plain_verdicts'])==48 and len(record['zk_certificates'])==2
            assert all(c['accepted'] and c['altered_rejected'] for c in record['zk_certificates'])
        if path.name.startswith('zk-certificate'):
            assert record['exit_code']==0 and not record['timed_out']
            assert record['result']['accepted'] and record['result']['negative_controls_rejected']==3
        if 'cases' not in record:continue
        assert record['warmups']==5 and record['samples']==30 and len(record['cases'])==6
        assert record['seed'] in (20261112,20261212)
        historical='historical' in path.name
        for case in record['cases']:
            trials=case['verification_results'];assert len(trials)==35 and case['attempts']==35
            found=sum(t.get('found',False) for t in trials);accepted=sum(t['accepted'] for t in trials)
            assert found==case['found'] and accepted==case['accepted'] and 0<=accepted<=found<=35
            assert case['p_a_bytes']==9
            for trial in trials:
                if not trial.get('found'):assert not trial['accepted']
                else:assert len(trial['share_verdicts'])==3
            for values in case['samples_ms'].values():assert len(values)==30 and all(math.isfinite(v) and v>=0 for v in values)
            for name,expected in case['public_artifacts_sha256'].items():binding(name,expected);artifacts.add(name)
            if case['canonical_repair'] and not historical:
                assert accepted==found
                canonical_found+=found;canonical_accepted+=accepted
            if historical:historical_rejections+=found-accepted
            rows.append(dict(record=path.name,seed=record['seed'],patch=record['patch'],share_nbits=record['share_nbits'],
                             pattern=case['pattern'],k=case['k'],canonical_norms=case['canonical_repair'],
                             attempts=35,found=found,accepted=accepted,rejected=found-accepted,no_hit=35-found))
    assert len(rows)==48 and canonical_found==canonical_accepted and historical_rejections>0
    with Path('results/conformance-summary.csv').open('w',newline='') as output:
        writer=csv.DictWriter(output,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    print(len(rows),'cases;',canonical_accepted,'canonical candidates accepted;',historical_rejections,
          'historical candidates rejected;',len(artifacts),'public artifact hashes;',layout_check(),'layout word checks')


if __name__=='__main__':main()
