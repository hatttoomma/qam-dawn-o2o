"""Export validated continuation evidence without large model checkpoints."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parent
subprocess.run([sys.executable,'summarize_warm_continuation.py'],cwd=ROOT,check=True)
paths=[]
for folder in ['runs/warm_extended','results_warm_continuation']:
    paths += [p for p in (ROOT/folder).rglob('*') if p.is_file() and p.name != 'bundle_manifest.json'
              and p.suffix in ('.json','.jsonl','.md','.csv','.png','.pdf')]
paths += [ROOT/p for p in ['runs/WARM_CONTINUATION_CHECKS_PASSED.json',
    'runs/warm_continuation_suite.jsonl','runs/warm_continuation_suite_status.json',
    'runs/warm_extended.log','runs/warm_continuation_launcher.log',
    'run_warm_continuation.py','launch_warm_continuation.py','summarize_warm_continuation.py',
    'export_warm_continuation.py','WARM_CONTINUATION_PROTOCOL.md','CHART_CONTRACT_CONTINUATION.md']]
paths=sorted(set(paths))
manifest={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths}
mp=ROOT/'results_warm_continuation/bundle_manifest.json'
mp.write_text(json.dumps(manifest,indent=2)+'\n')
paths.append(mp)
archive=ROOT/'results_warm_continuation_bundle.tar.gz'
with tarfile.open(archive,'w:gz') as tf:
    for p in paths:tf.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
print(json.dumps(dict(archive=str(archive),files=len(paths),bytes=archive.stat().st_size,
                     sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
