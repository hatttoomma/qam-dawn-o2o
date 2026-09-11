"""Validate and bundle lightweight BC experiment evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results_bc'
subprocess.run([sys.executable,'summarize_bc.py','--runs',str(ROOT/'runs'),'--out',str(OUT)],cwd=ROOT,check=True)
files=[p for p in OUT.glob('*') if p.name!='bundle_manifest.json']
arms=['offline','native','warm','random','warm_qam_replay','random_qam_replay','policy_decorator_qam_replay',
    'bc_offline','bc_native','bc_dawn_online','bc_dawn_qam','bc_policy_decorator',
    'smoke_bc_offline','smoke_bc_native','smoke_bc_dawn_online','smoke_bc_dawn_qam','smoke_bc_policy_decorator']
for arm in arms:files += [p for p in (ROOT/'runs'/arm).glob('*') if p.suffix in ['.json','.jsonl']]
files += [ROOT/'runs'/p for p in ['source_manifest.json','replay_source_manifest.json','policy_decorator_source_manifest.json',
    'bc_pretrain_source_manifest.json','bc_online_source_manifest.json','BC_PRETRAIN_CHECKS_PASSED.json','BC_ONLINE_CHECKS_PASSED.json',
    'bc_unit.log','bc_pretrain_suite_status.json','bc_pretrain_suite.jsonl','bc_online_suite_status.json','bc_online_suite.jsonl','pip_freeze.txt']]
files += [ROOT/p for p in ['bc_pretrain.py','run_bc_online.py','test_bc_pretrain.py','launch_bc_pretrain.py','launch_bc_online.py',
    'summarize_bc.py','export_bc.py','BC_PROTOCOL.md','CHART_CONTRACT_BC.md']]
files=sorted(set(files));manifest=OUT/'bundle_manifest.json'
manifest.write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
archive=ROOT/'results_bc_bundle.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in files+[manifest]:tar.add(p,arcname=str(p.relative_to(ROOT)))
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,
    sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(files)+1)))
