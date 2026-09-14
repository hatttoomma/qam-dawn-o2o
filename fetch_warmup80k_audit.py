"""Read-only retrieval of five-task experiment evidence over a fresh SSH connection."""
import base64
import json
from pathlib import Path
import shlex
import subprocess
import zlib

REMOTE = r'''
import pathlib,json,base64,zlib,hashlib,subprocess,time,math
root=pathlib.Path('/root/autodl-tmp/qam_dawn_o2o')
old=root/'runs/warmup80k_150k_20260911'
new=root/'runs/warmup80k_150k_task234_20260911'
files={};checks=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
for label,parent in [('task15_suite',old),('task234_suite',new)]:
 for n in ['suite_config.json','source_manifest.json','runtime_compatibility.json','suite_status.json','pip_freeze.txt','suite.jsonl','storage.json','results.json','results.csv','all5_results.json','all5_results.csv','all5_qam_native_comparison.csv']:
  p=parent/n
  if p.exists():files[label+'/'+n]=p.read_text()
 manifest=json.loads((parent/'source_manifest.json').read_text())
 for name,expected in manifest.items():assert sha(root/name)==expected,(label,name)
 checks.append(dict(suite=label,source_files_checked=len(manifest)))
for task in range(1,6):
 parent=old if task in (1,5) else new
 for stage in ['collector','warm']:
  folder=parent/f'task{task}_{stage}'
  for n in ['actual_agent_config.json','config.json','requested_config.json','initial_hashes.json','warmup.json','shared_initialization.json','task_manifest.json','backup_math.json','actor_input_checks.json','DONE.json','CHECKS_PASSED.json']:
   p=folder/n
   if p.exists():files[f'task{task}_{stage}/'+n]=p.read_text()
  for p in folder.glob('eval_*.json'):
   e=json.loads(p.read_text());records=e.pop('records')
   assert len(records)==e['episodes']==100
   assert abs(sum(x['success'] for x in records)/100-e['success'])<1e-10
   assert abs(sum(x['return_'] for x in records)/100-e['return_mean'])<1e-8
   assert [x['reset_seed'] for x in records]==list(range(500000,500100))
   ids=[(x['episode'],x['reset_seed'],x['initial_hash']) for x in records]
   e['record_ids_hash']=hashlib.sha256(json.dumps(ids,sort_keys=True).encode()).hexdigest()
   e['raw_records_sha256']=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
   e['raw_record_aggregates_verified']=True
   files[f'task{task}_{stage}/'+p.name]=json.dumps(e)
  for n in ['restore_checks.jsonl','evaluation_integrity.jsonl']:
   if (folder/n).exists():files[f'task{task}_{stage}/'+n]=(folder/n).read_text()
  if (folder/'metrics.jsonl').exists():
   metrics=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines() if line.strip()]
   assert all(math.isfinite(v) for row in metrics for v in row.values() if isinstance(v,(int,float)))
   if stage=='warm':assert all(row['updates']==(row['step']-80000)//4 for row in metrics)
   files[f'task{task}_{stage}/metrics_verified.json']=json.dumps(dict(rows=len(metrics),all_finite=True,update_schedule_verified=stage=='warm',last=metrics[-1]))
 off=root/('runs/offline' if task==1 else 'runs/task2/offline' if task==2 else f'runs/cube5/task{task}_offline')
 for n in ['config.json','DONE.json','task_manifest.json']:
  if (off/n).exists():files[f'task{task}_offline/'+n]=(off/n).read_text()
 done=json.loads((off/'DONE.json').read_text());assert sha(off/'final.pkl')==done['checkpoint_sha256']
 check=dict(task=task,offline_checkpoint_sha256=done['checkpoint_sha256'])
 folder=parent/f'task{task}_warm'
 if (folder/'DONE.json').exists():
  online=json.loads((folder/'DONE.json').read_text())
  assert online['steps']==150000 and online['updates']==17500
  assert json.loads((folder/'CHECKS_PASSED.json').read_text())['status']=='passed'
  assert sha(folder/'final.pkl')==online['checkpoint_sha256']
  check['final_checkpoint_verified']=online['checkpoint_sha256']
 checks.append(check)
 native=root/('runs/native' if task==1 else 'runs/task2/native' if task==2 else f'runs/cube5/task{task}_native')
 for n in ['config.json','DONE.json']:
  files[f'task{task}_native/'+n]=(native/n).read_text()
 e=json.loads((native/'eval_050000_100.json').read_text());records=e.pop('records')
 assert len(records)==e['episodes']==100
 assert abs(sum(x['success'] for x in records)/100-e['success'])<1e-10
 assert abs(sum(x['return_'] for x in records)/100-e['return_mean'])<1e-8
 e['raw_record_aggregates_verified']=True
 e['record_ids_hash']=hashlib.sha256(json.dumps([(x['episode'],x['reset_seed'],x['initial_hash']) for x in records],sort_keys=True).encode()).hexdigest()
 files[f'task{task}_native/eval_050000_100.json']=json.dumps(e)
for n in ['run_warmup80k.py','run_warmup80k_task234.py','launch_warmup80k.py','launch_warmup80k_task234.py','run.py','actor_input_agent.py','run_actor_input_ablation.py','warmup80k_task234_extension_diff.json']:
 files['source/'+n]=(root/n).read_text()
files['remote_verification.json']=json.dumps(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),checks=checks,gpu=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True,capture_output=True).stdout.strip()))
print(base64.b64encode(zlib.compress(json.dumps(files).encode())).decode())
'''


if __name__ == '__main__':
    out = Path(__file__).resolve().parents[2] / 'output/warmup80k_150k_task234_20260911/settings_audit'
    out.mkdir(parents=True, exist_ok=True)
    raw = out / 'retrieved_payload.b64'
    command = ['ssh', '-S', 'none', '-o', 'ControlMaster=no', '-o', 'ConnectTimeout=10',
               '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2',
               '-p', '39459', 'root@connect.bjb1.seetacloud.com',
               shlex.join(['/root/autodl-tmp/qam_dawn_o2o/venv/bin/python', '-c', REMOTE])]
    with raw.open('wb') as stream:
        subprocess.run(command, stdout=stream, check=True, timeout=180)
    files = json.loads(zlib.decompress(base64.b64decode(raw.read_bytes())))
    for name, value in files.items():
        path = out / name
        assert path.resolve().is_relative_to(out.resolve())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    print(f'Retrieved {len(files)} evidence files into {out}')
    print(files['remote_verification.json'])
