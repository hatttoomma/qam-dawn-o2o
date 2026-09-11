"""Export verified five-task results and lightweight raw evidence, retaining remote weights."""
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from launch_cube5 import digest

ROOT=Path(__file__).resolve().parent
for script in ['audit_cube5_checkpoints.py','summarize_cube5.py']:
    subprocess.run([sys.executable,script],cwd=ROOT,check=True)
paths=[]
for folder in ['runs/cube5','results_cube5']:
    paths += [p for p in (ROOT/folder).rglob('*') if p.is_file() and p.name!='bundle_manifest.json'
        and p.suffix in ('.json','.jsonl','.log','.txt','.md','.csv','.png','.pdf','.npz')
        and p.name not in ('launcher.log','suite_status.json','suite.jsonl','report.log','export_result.json')]
validation=json.loads((ROOT/'results_cube5/validation.json').read_text())
paths += [ROOT/n for n in validation['sources']]
paths += [ROOT/n for n in json.loads((ROOT/'runs/cube5/source_manifest.json').read_text())]
paths += [ROOT/'export_cube5.py']
paths=sorted(set(paths))
manifest={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in paths}
mp=ROOT/'results_cube5/bundle_manifest.json';mp.write_text(json.dumps(manifest,indent=2)+'\n');paths.append(mp)
archive=ROOT/'results_cube5_bundle.tar.gz'
with tarfile.open(archive,'w:gz') as tf:
    for p in paths:tf.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
result=dict(archive=str(archive),files=len(paths),bytes=archive.stat().st_size,sha256=digest(archive))
(ROOT/'runs/cube5/export_result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
