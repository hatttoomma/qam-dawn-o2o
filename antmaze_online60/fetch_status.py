"""Read the authorized remote suite; SSH reads any password from the terminal."""
import argparse
import base64
import gzip
import json
from pathlib import Path
import shlex
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--hostname')
parser.add_argument('--socket', default='/tmp/icra_antmaze_balanced45557_dns.sock')
parser.add_argument('--full', action='store_true', help='Include episode records, metrics, and checkpoint verification.')
parser.add_argument('--output', default='/Users/riverwang/Documents/ICRA2026/output/antmaze_large_online60_20260914/remote_status.json')
args = parser.parse_args()
source = '''
from pathlib import Path
import json,os,subprocess,time,hashlib,gzip,base64
root=Path('/root/autodl-tmp/qam_antmaze_online60')
result={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'files':{},'file_sha256':{},'full':FULL}
for p in sorted((root/'results').rglob('*.json')):
 if not FULL and p.name in ('source_verification.json','requested_flags.json'):continue
 try:
  raw=p.read_bytes()
  value=json.loads(raw)
  if isinstance(value,dict) and not FULL:
   value.pop('records',None);value.pop('source_files',None)
  result['files'][str(p.relative_to(root))]=value
  result['file_sha256'][str(p.relative_to(root))]=hashlib.sha256(raw).hexdigest()
 except Exception as e:result['files'][str(p.relative_to(root))]={'read_error':str(e)}
if FULL:
 for folder in sorted((root/'results').glob('task*/*')):
  for p in sorted(folder.glob('*.jsonl')):
   raw=p.read_bytes();rows=[]
   for line in raw.splitlines():
    try:rows.append(json.loads(line))
    except json.JSONDecodeError:pass
   result['files'][str(p.relative_to(root))]=rows
   result['file_sha256'][str(p.relative_to(root))]=hashlib.sha256(raw).hexdigest()
 result['checkpoint_audit']={}
 for task in (1,2,3,4,5):
  for relative,metadata,key in (
    (f'results/task{task}/dawn/final.pkl',f'results/task{task}/dawn/DONE.json','checkpoint_sha256'),):
   if metadata not in result['files']:continue
   h=hashlib.sha256()
   with (root/relative).open('rb') as f:
    for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
   expected=result['files'][metadata][key]
   result['checkpoint_audit'][relative]={'sha256':h.hexdigest(),'expected':expected,'matched':h.hexdigest()==expected}
for name in ('install_tuna.log','download_verified.log','system_deps.log'):
 p=root/name
 if p.exists():result[name]=p.read_text(errors='replace')[-2000:]
for p in sorted((root/'results').rglob('stdout.log')):
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-2400));result[str(p.relative_to(root))]=f.read().decode(errors='replace')
result['gpu']=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
result['processes']=subprocess.run(['ps','-eo','pid,ppid,etime,stat,args'],capture_output=True,text=True).stdout
result['processes']='\\n'.join(x for x in result['processes'].splitlines() if ('qam_antmaze' in x or '/antmaze_qam_venv/' in x) and 'python -c' not in x and 'sshd:' not in x)
print(base64.b64encode(gzip.compress(json.dumps(result).encode())).decode())
'''
source = 'FULL = ' + repr(args.full) + '\n' + source
cmd = ['ssh', '-o', 'ConnectTimeout=15', '-p', '45557']
if args.socket != 'none' and Path(args.socket).exists():
    cmd += ['-S', args.socket]
if args.hostname:
    cmd += ['-o', f'Hostname={args.hostname}', '-o', 'HostKeyAlias=[connect.bjb1.seetacloud.com]:45557']
cmd += ['root@connect.bjb1.seetacloud.com', '/root/miniconda3/bin/python -c ' + shlex.quote(source)]
destination = Path(args.output)
destination.parent.mkdir(parents=True, exist_ok=True)
temp = destination.with_suffix('.tmp')
with temp.open('w') as f:
    result = subprocess.run(cmd, stdout=f)
if result.returncode:
    raise SystemExit(result.returncode)
data = json.loads(gzip.decompress(base64.b64decode(temp.read_text())))
temp.write_text(json.dumps(data, indent=2))
temp.replace(destination)
print(json.dumps({'snapshot':str(destination),'utc':data['utc'],
    'suite':data['files'].get('results/suite_status.json'),'gpu':data['gpu']},ensure_ascii=False,indent=2))
