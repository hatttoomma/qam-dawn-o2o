"""Export verified scale-sweep results and raw evidence, excluding weights."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
subprocess.run([sys.executable, 'summarize_task2_scale.py'], cwd=ROOT, check=True)
paths = []
for folder in ['runs/task2/scale_sweep', 'results_task2_scale']:
    paths += [p for p in (ROOT / folder).rglob('*') if p.is_file()
        and p.name != 'bundle_manifest.json' and p.suffix in ('.json', '.jsonl', '.log', '.txt', '.md', '.csv', '.png', '.pdf')]
# Include immutable reference evidence, but do not re-export unrelated historical runs.
for folder in ['offline', 'native', 'warm']:
    paths += [p for p in (ROOT / 'runs/task2' / folder).glob('*') if p.is_file()
              and p.suffix in ('.json', '.jsonl')]
paths += [ROOT / 'runs/task2' / n for n in ['source_manifest.json', 'CHECKS_PASSED.json', 'TASK_AUDIT_PASSED.json']]
paths += [ROOT / n for n in ['run_task2_scale.py', 'launch_task2_scale.py', 'TASK2_SCALE_PROTOCOL.md',
                             'summarize_task2_scale.py', 'export_task2_scale.py', 'CHART_CONTRACT_TASK2_SCALE.md']]
paths = sorted(set(paths))
manifest = {str(p.relative_to(ROOT)): dict(bytes=p.stat().st_size,
            sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths}
mp = ROOT / 'results_task2_scale/bundle_manifest.json'
mp.write_text(json.dumps(manifest, indent=2) + '\n')
paths.append(mp)
archive = ROOT / 'results_task2_scale_bundle.tar.gz'
with tarfile.open(archive, 'w:gz') as tf:
    for p in paths: tf.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
print(json.dumps(dict(archive=str(archive), files=len(paths), bytes=archive.stat().st_size,
                     sha256=hashlib.sha256(archive.read_bytes()).hexdigest())), flush=True)
