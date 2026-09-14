"""Read-only remote snapshot, executed by fetch_status.py over SSH."""
from pathlib import Path
import base64, gzip, hashlib, json, os, subprocess, sys, time

root = Path(sys.argv[1])
full = sys.argv[2] == 'full'
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

result = dict(root=str(root), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              full=full, files={}, checkpoint_audit={}, source_audit={})
for p in sorted((root / 'results').rglob('*.json')):
    try:
        value = json.loads(p.read_text())
        if not full and isinstance(value, dict):
            value = {k: v for k, v in value.items() if k not in ('records', 'source_files')}
        result['files'][str(p.relative_to(root))] = value
    except Exception as e:
        result['files'][str(p.relative_to(root))] = {'read_error': str(e)}
for name in ('inputs.json', 'launch.json', 'adapter_hashes.json', 'source_manifest.json', 'storage.json', 'logging_repair.json'):
    p = root / name
    if p.exists(): result[name] = json.loads(p.read_text())
if full:
    for p in sorted((root / 'results').glob('task*/*/*.jsonl')):
        rows = []
        for line in p.read_text().splitlines():
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: pass  # An active writer may leave a partial last line.
        result['files'][str(p.relative_to(root))] = rows
    for p in sorted((root / 'results').glob('task*/*/CHECKS_PASSED.json')):
        done = json.loads((p.parent / 'DONE.json').read_text())
        model = p.parent / 'final.pkl'
        digest = sha(model)
        result['checkpoint_audit'][str(model.relative_to(root))] = dict(
            sha256=digest, expected=done['checkpoint_sha256'], matched=digest == done['checkpoint_sha256'])
    manifests = [result['adapter_hashes.json'], result['source_manifest.json'].get('files', result['source_manifest.json'])]
    for manifest in manifests:
        for name, expected in manifest.items():
            p = Path(name) if name.startswith('/') else root / name
            digest = sha(p)
            result['source_audit'][name] = dict(sha256=digest, expected=expected, matched=digest == expected)
for p in sorted((root / 'results').rglob('stdout.log')):
    with p.open('rb') as f:
        f.seek(max(0, p.stat().st_size - 2000))
        result.setdefault('logs', {})[str(p.relative_to(root))] = f.read().decode(errors='replace')
suite = result['files'].get('results/suite_status.json', {})
result['process_liveness'] = {}
for key in ('pid', 'child_pid'):
    pid = suite.get(key)
    if pid:
        proc = Path('/proc') / str(pid)
        result['process_liveness'][key] = dict(pid=pid, alive=proc.exists(),
            command=(proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace') if proc.exists() else '')
result['gpu'] = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used',
                                '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
result['disk'] = subprocess.run(['df', '-h', str(root), '/root/autodl-tmp'], capture_output=True, text=True).stdout
print(base64.b64encode(gzip.compress(json.dumps(result).encode())).decode())
