"""Validate recorded source hashes and derive latency summaries (standard library)."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median

rows=[]
for filename in ('gpu-study.json','gpu-replication.json'):
    record=json.loads((Path('results')/filename).read_text())
    assert record['samples']==30 and record['warmups']==5
    for name,expected in record['source_sha256'].items():
        assert not Path(name).is_absolute() and ':' not in name
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
    for case in record['cases']:
        if 'samples_ms' in case:
            samples={('transfer_wall_ms' if k.startswith('transfer') else 'resident_wall_ms',k):v for k,v in case['samples_ms'].items()}
            samples.update({('binding_transfer_wall_ms',k):v for k,v in case.get('full_path_samples_ms',{}).items()})
        else:
            samples={(scope,variant):values for variant,scopes in case['samples'].items() for scope,values in scopes.items()}
        for (scope,variant),values in samples.items():
            assert len(values)==30 and all(math.isfinite(x) and x>=0 for x in values)
            ordered=sorted(values)
            rows.append(dict(seed=record['seed'],n=case['n'],batch=case.get('batch',''),pattern=case.get('pattern',''),
                             noise_scale=case.get('noise_scale',''),variant=variant,scope=scope,
                             median_ms=median(values),p95_ms=ordered[math.ceil(.95*len(values))-1],
                             min_ms=min(values),max_ms=max(values),repetitions=len(values)))
with Path('results/gpu-summary.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
print(f'{len(rows)} latency summaries; all recorded source hashes match')
