"""Generate the task2 report and a portable evidence archive without model weights."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parent
subprocess.run([sys.executable,'summarize_task2.py'],cwd=ROOT,check=True)
paths=[]
for folder in ['runs/task2','results_task2']:
    paths += [p for p in (ROOT/folder).rglob('*') if p.is_file()
        and p.name!='bundle_manifest.json' and p.suffix in ('.json','.jsonl','.log','.txt','.md','.csv','.png','.pdf')]
paths += [ROOT/n for n in ['run_task2.py','audit_task2.py','launch_task2.py','TASK2_PROTOCOL.md',
                           'summarize_task2.py','export_task2.py','CHART_CONTRACT_TASK2.md']]
paths=sorted(set(paths))
manifest={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths}
mp=ROOT/'results_task2/bundle_manifest.json';mp.write_text(json.dumps(manifest,indent=2)+'\n')
paths.append(mp)
archive=ROOT/'results_task2_bundle.tar.gz'
with tarfile.open(archive,'w:gz') as tf:
    for p in paths:tf.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
print(json.dumps(dict(archive=str(archive),files=len(paths),bytes=archive.stat().st_size,
    sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
