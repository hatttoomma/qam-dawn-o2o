"""Export both tasks' verified scale evidence without model weights."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
subprocess.run([sys.executable, 'summarize_scale_extremes.py'], cwd=ROOT, check=True)
paths = []
for folder in ['runs/scale_extremes', 'results_scale_extremes']:
    paths += [p for p in (ROOT / folder).rglob('*') if p.is_file() and p.name != 'bundle_manifest.json'
              and p.suffix in ('.json', '.jsonl', '.log', '.txt', '.md', '.csv', '.png', '.pdf', '.npz')]
validation = json.loads((ROOT / 'results_scale_extremes/validation.json').read_text())
paths += [ROOT / n for n in validation['sources']]
paths += [p for p in (ROOT / 'runs/task2/scale_sweep/smoke_reference').glob('*')
          if p.is_file() and p.suffix in ('.json', '.jsonl')]
paths += [ROOT / n for n in ['run_scale_extremes.py', 'launch_scale_extremes.py', 'SCALE_EXTREMES_PROTOCOL.md',
                            'summarize_scale_extremes.py', 'export_scale_extremes.py', 'audit_scale_warmup.py',
                            'CHART_CONTRACT_SCALE_EXTREMES.md']]
paths = sorted(set(paths))
manifest = {str(p.relative_to(ROOT)): dict(bytes=p.stat().st_size,
            sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths}
mp = ROOT / 'results_scale_extremes/bundle_manifest.json';mp.write_text(json.dumps(manifest, indent=2) + '\n')
paths.append(mp)
archive = ROOT / 'results_scale_extremes_bundle.tar.gz'
with tarfile.open(archive, 'w:gz') as tf:
    for p in paths: tf.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
print(json.dumps(dict(archive=str(archive), files=len(paths), bytes=archive.stat().st_size,
                     sha256=hashlib.sha256(archive.read_bytes()).hexdigest())), flush=True)
