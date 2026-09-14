"""Fetch all three queues from the two authorized machines; no credentials stored."""
import argparse, base64, concurrent.futures, gzip, json, shlex, subprocess, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'output/scale100k_live'
QUEUES = {
    'antmaze_dawn': dict(machine='antmaze', port=45557, socket='/tmp/icra_antmaze_balanced45557_dns.sock',
        root='/root/autodl-tmp/qam_antmaze_scale100k_dawn', python='/root/autodl-tmp/antmaze_qam_venv/bin/python'),
    'antmaze_native': dict(machine='antmaze', port=45557, socket='/tmp/icra_antmaze_balanced45557_dns.sock',
        root='/root/autodl-tmp/qam_antmaze_scale100k_native', python='/root/autodl-tmp/antmaze_qam_venv/bin/python'),
    'cube': dict(machine='cube', port=39459, socket='/tmp/icra_cube39459_scale100.sock',
        root='/root/qam_dawn_outputs/scale100k_20260914', python='/root/autodl-tmp/qam_dawn_o2o/venv/bin/python'),
}
parser = argparse.ArgumentParser()
parser.add_argument('--full', action='store_true')
parser.add_argument('--machine', choices=['all', 'antmaze', 'cube'], default='all')
parser.add_argument('--output', type=Path, default=OUT / 'remote_status.json')
args = parser.parse_args()
source = (HERE / 'remote_snapshot.py').read_text()

def fetch(item):
    name, cfg = item
    cmd = ['ssh', '-o', 'ConnectTimeout=15', '-o', 'BatchMode=yes', '-p', str(cfg['port'])]
    if Path(cfg['socket']).exists(): cmd += ['-S', cfg['socket']]
    remote = [cfg['python'], '-c', source, cfg['root'], 'full' if args.full else 'compact']
    cmd += ['root@connect.bjb1.seetacloud.com', shlex.join(remote)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    if r.returncode: raise RuntimeError(name + ': ' + r.stderr.strip())
    return name, json.loads(gzip.decompress(base64.b64decode(r.stdout)))

result = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), full=args.full, queues={}, errors={})
items = [(n, c) for n, c in QUEUES.items() if args.machine in ('all', c['machine'])]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    futures = {pool.submit(fetch, item): item[0] for item in items}
    for future in concurrent.futures.as_completed(futures):
        try:
            name, data = future.result()
            result['queues'][name] = data
        except Exception as e: result['errors'][futures[future]] = str(e)
args.output.parent.mkdir(parents=True, exist_ok=True)
temp = args.output.with_suffix('.tmp')
temp.write_text(json.dumps(result, indent=2)); temp.replace(args.output)
print(json.dumps(dict(snapshot=str(args.output), errors=result['errors'],
    suites={name: data['files'].get('results/suite_status.json') for name, data in result['queues'].items()}), indent=2))
if result['errors']: raise SystemExit(1)
