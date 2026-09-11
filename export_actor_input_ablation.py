"""Export validated actor-input results and lightweight reproducibility evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parent
for script in ['audit_actor_input_warmup.py','summarize_actor_input_ablation.py']:
    subprocess.run([sys.executable,script],cwd=ROOT,check=True)
paths=[]
for folder in ['runs/actor_input_ablation','results_actor_input_ablation']:
    paths += [p for p in (ROOT/folder).rglob('*') if p.is_file() and p.name!='bundle_manifest.json'
              and p.suffix in ('.json','.jsonl','.log','.txt','.md','.csv','.png','.pdf','.npz')]
validation=json.loads((ROOT/'results_actor_input_ablation/validation.json').read_text())
paths += [ROOT/n for n in validation['sources']]
paths += [ROOT/n for n in json.loads((ROOT/'runs/actor_input_ablation/source_manifest.json').read_text())]
paths += [ROOT/'export_actor_input_ablation.py']
paths=sorted(set(paths))
manifest={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths}
mp=ROOT/'results_actor_input_ablation/bundle_manifest.json';mp.write_text(json.dumps(manifest,indent=2)+'\n');paths.append(mp)
archive=ROOT/'results_actor_input_ablation_bundle.tar.gz'
with tarfile.open(archive,'w:gz') as tf:
    for p in paths:tf.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
print(json.dumps(dict(archive=str(archive),files=len(paths),bytes=archive.stat().st_size,
    sha256=hashlib.sha256(archive.read_bytes()).hexdigest())),flush=True)
