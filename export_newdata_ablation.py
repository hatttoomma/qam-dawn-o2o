"""Export completed, audited new-data ablation evidence without model weights."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
subprocess.run([sys.executable, 'summarize_newdata_ablation.py'], cwd=ROOT, check=True)
paths = []
for folder in ['runs/newdata_ablation', 'results_newdata_ablation']:
    paths += [p for p in (ROOT / folder).rglob('*') if p.is_file() and p.name != 'bundle_manifest.json'
              and p.suffix in ('.json', '.jsonl', '.log', '.txt', '.md', '.csv', '.png', '.pdf', '.npz')]
validation = json.loads((ROOT / 'results_newdata_ablation/validation.json').read_text())
paths += [ROOT / n for n in validation['sources']]
paths += [ROOT / n for n in ['run_newdata_ablation.py', 'smoke_newdata_reference.py', 'launch_newdata_ablation.py',
    'NEWDATA_ABLATION_PROTOCOL.md', 'summarize_newdata_ablation.py', 'export_newdata_ablation.py', 'CHART_CONTRACT_NEWDATA_ABLATION.md']]
paths = sorted(set(paths))
manifest = {str(p.relative_to(ROOT)): dict(bytes=p.stat().st_size,
    sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths}
mp = ROOT / 'results_newdata_ablation/bundle_manifest.json';mp.write_text(json.dumps(manifest, indent=2) + '\n')
paths.append(mp)
archive = ROOT / 'results_newdata_ablation_bundle.tar.gz'
with tarfile.open(archive, 'w:gz') as tf:
    for p in paths: tf.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
print(json.dumps(dict(archive=str(archive), files=len(paths), bytes=archive.stat().st_size,
    sha256=hashlib.sha256(archive.read_bytes()).hexdigest())), flush=True)
