"""Produce a small verified evidence bundle; large weights stay on the remote."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results_policy_decorator'
subprocess.run([sys.executable,'summarize_policy_decorator.py','--runs',str(ROOT/'runs'),
                '--out',str(OUT)],cwd=ROOT,check=True)
files=[p for p in OUT.glob('*') if p.name!='bundle_manifest.json']
for arm in ['offline','native','warm','random','warm_qam_replay','random_qam_replay',
            'policy_decorator_qam_replay','smoke_policy_decorator_qam_replay']:
    files += [p for p in (ROOT/'runs'/arm).glob('*') if p.suffix in ['.json','.jsonl']]
files += [ROOT/'runs'/p for p in ['policy_decorator_source_manifest.json','POLICY_DECORATOR_CHECKS_PASSED.json',
    'policy_decorator_suite_status.json','policy_decorator_suite.jsonl','policy_decorator_unit.log',
    'source_manifest.json','replay_source_manifest.json','pip_freeze.txt']]
files += [ROOT/p for p in ['policy_decorator_agent.py','run_policy_decorator.py','test_policy_decorator.py',
    'launch_policy_decorator.py','summarize_policy_decorator.py','export_policy_decorator.py',
    'POLICY_DECORATOR_PROTOCOL.md','POLICY_DECORATOR_SOURCE_AUDIT.json','CHART_CONTRACT_PD.md']]
files=sorted(set(files))
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
manifest=OUT/'bundle_manifest.json'
manifest.write_text(json.dumps(hashes,indent=2)+'\n')
archive=ROOT/'results_policy_decorator_bundle.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in files+[manifest]:tar.add(p,arcname=str(p.relative_to(ROOT)))
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,
    sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(files)+1)))
