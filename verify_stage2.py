"""Verify Stage-2 source provenance and derive median/p95 tables."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median

files=set()
for pattern in ('binary-*.json','large-ntt*.json','protocol-*.json','canonical-norm-validation.json','rust-norm-*.json','stage2-build.json','ckks-*.json','bootstrap*.json','nonce-*.json','*-build.json'):
    files.update(Path('results').glob(pattern))
assert files,'No Stage-2 records'
rows=[]
for path in sorted(files):
    record=json.loads(path.read_text())
    for build in record.get('builds',{}).values():
        for name,expected in build['source_sha256'].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,(path,name)
    for name,expected in record.get('source_sha256',{}).items():
        assert not Path(name).is_absolute() and ':' not in name
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,(path,name)
    for name,expected in record.get('result_sha256',{}).items():
        assert hashlib.sha256((Path('results')/name).read_bytes()).hexdigest()==expected,(path,name)
    if 'oracle_record_sha256' in record:
        assert hashlib.sha256((Path('results')/record['oracle_record']).read_bytes()).hexdigest()==record['oracle_record_sha256']
    if 'builder_sha256' in record:
        assert hashlib.sha256(Path('build_rust_norms.py').read_bytes()).hexdigest()==record['builder_sha256']
    if 'study_sha256' in record:
        study='protocol-replication.json' if 'replication' in path.name else 'protocol-study.json'
        assert hashlib.sha256((Path('results')/study).read_bytes()).hexdigest()==record['study_sha256']
    groups=[('resident',record.get('cases',[])),('multiplication',record.get('multiplication_cases',[])),('proof',record.get('proof_cases',[]))]
    if 'samples_ms' in record:groups.append(('application',[record]))
    for category,cases in groups:
        for case in cases:
            samples={}
            if 'samples_ms' in case:samples.update({(category,variant):values for variant,values in case['samples_ms'].items()})
            if isinstance(case.get('samples'),dict):
                samples.update({(scope,variant):values for variant,scopes in case['samples'].items() for scope,values in scopes.items()})
            samples.update({('graph_'+scope,variant):values for variant,scopes in case.get('graph_samples_ms',{}).items() for scope,values in scopes.items()})
            for (scope,variant),values in samples.items():
                assert record['samples']==30 and record['warmups']==5
                assert len(values)==30 and all(math.isfinite(x) and x>=0 for x in values)
                row={'file':path.name,'seed':record.get('seed','')}
                row.update({key:case.get(key,record.get(key,'')) for key in ('n','k','m','limbs','batch','pattern','kind','canonical_repair')})
                row.update(variant=variant,scope=scope,median_ms=median(values),p95_ms=sorted(values)[math.ceil(.95*len(values))-1],repetitions=len(values))
                rows.append(row)
assert rows,'No timing records found'
with Path('results/stage2-summary.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(rows)
print(len(files),'Stage-2 records verified;',len(rows),'latency summaries derived')
