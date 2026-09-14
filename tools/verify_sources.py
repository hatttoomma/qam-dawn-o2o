"""Verify the uploaded source snapshot and final-run provenance without GPU or SSH."""
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / 'provenance/20260914'
SNAPSHOT = ROOT / 'reports/100k_20260914/final_complete_snapshot.json'
SNAPSHOT_SHA256 = '70ee02f29c9dbc6883f3447e8a1f7809d5d62f66468db246ee888d0f9da68a87'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(path, expected):
    if not path.is_file():
        raise RuntimeError(f'Missing source: {path.relative_to(ROOT)}')
    if sha256(path) != expected:
        raise RuntimeError(f'Source checksum mismatch: {path.relative_to(ROOT)}')


def main():
    remote = json.loads((PROVENANCE / 'remote_sources.json').read_text())['files']
    for relative, spec in remote.items():
        check(ROOT / relative, spec['sha256'])

    local = json.loads((PROVENANCE / 'local_additions.json').read_text())['files']
    packaging = json.loads((PROVENANCE / 'packaging_changes.json').read_text())
    for relative, digest in local.items():
        if relative in packaging:
            if packaging[relative]['original_sha256'] != digest:
                raise RuntimeError(f'Packaging origin mismatch: {relative}')
            digest = packaging[relative]['repository_sha256']
        check(ROOT / relative, digest)

    check(SNAPSHOT, SNAPSHOT_SHA256)
    snapshot = json.loads(SNAPSHOT.read_text())
    directories = {'antmaze_dawn': 'scale100k/antmaze',
                   'antmaze_native': 'scale100k/antmaze_native',
                   'cube': 'scale100k/cube'}
    run_source_checks = 0
    original_cube = Path('/root/autodl-tmp/qam_dawn_o2o')
    for name, queue in snapshot['queues'].items():
        directory = ROOT / directories[name]
        for filename in ('inputs.json', 'adapter_hashes.json', 'source_manifest.json'):
            if json.loads((directory / filename).read_text()) != queue[filename]:
                raise RuntimeError(f'Final-run metadata mismatch: {name}/{filename}')
        for relative, spec in queue['source_audit'].items():
            if not spec['matched'] or spec['sha256'] != spec['expected']:
                raise RuntimeError(f'Failed original source audit: {name}/{relative}')
            path = Path(relative)
            if path.is_absolute():
                path = ROOT / path.relative_to(original_cube)
            else:
                path = directory / path
            check(path, spec['sha256'])
            run_source_checks += 1

    python_files = 0
    ignored = {'.git', 'venv', '.venv', '__pycache__', '.pytest_cache'}
    for path in ROOT.rglob('*.py'):
        if any(part in ignored for part in path.relative_to(ROOT).parts):
            continue
        ast.parse(path.read_bytes(), filename=str(path))
        python_files += 1

    print(json.dumps({'status': 'passed', 'remote_source_files': len(remote),
                      'local_addition_files': len(local), 'report_path_adaptations': len(packaging),
                      'final_run_source_checks': run_source_checks,
                      'python_files_syntax_checked': python_files,
                      'final_snapshot_sha256': SNAPSHOT_SHA256}, indent=2))


if __name__ == '__main__':
    main()
