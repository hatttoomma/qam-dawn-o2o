"""Fetch a compact, credential-free progress and result audit from the current machine."""
import argparse
import base64
import gzip
import json
from pathlib import Path
import subprocess
import sys

LOCAL = Path(__file__).resolve().parents[2] / 'output/utd00625_warmup80k_150k_20260911'
REMOTE = r'''
from pathlib import Path
import base64,gzip,hashlib,json,subprocess,time
root=Path('/root/autodl-tmp/qam_dawn_o2o')
suite=root/'runs/utd00625_warmup80k_150k_20260911'
files={}
for name in ('suite_config.json','suite_status.json','launcher_pid.json','PREREQUISITES_VERIFIED.json','PREFLIGHT_PASSED.json','source_manifest.json','storage.json','runtime_compatibility.json','all_results.json','all_results.csv','final_comparison.csv','paired_comparison.json','paired_comparison.csv','suite.jsonl'):
    path=suite/name
    if path.exists(): files[name]=path.read_text()
tasks={}
for task in range(1,6):
    folder=suite/f'task{task}_warm'
    state={}
    for name in ('progress.json','checkpoint_state.json','DONE.json','CHECKS_PASSED.json','FAILED.json','MATCHED_CONFIG_PASSED.json','utd_schedule_audit.json','actual_agent_config.json','requested_config.json','shared_initialization.json','initial_hashes.json','warmup.json','restore_checks.jsonl'):
        path=folder/name
        if path.exists():
            files[f'task{task}_warm/{name}']=path.read_text()
            if name in ('progress.json','DONE.json','FAILED.json','MATCHED_CONFIG_PASSED.json'): state[name]=json.loads(path.read_text())
    evaluations=[]
    for path in sorted(folder.glob('eval_*_100*.json')):
        e=json.loads(path.read_text())
        records=e.pop('records')
        assert len(records)==e['episodes']==100
        assert abs(sum(r['success'] for r in records)/100-e['success'])<1e-10
        assert abs(sum(r['return_'] for r in records)/100-e['return_mean'])<1e-8
        ids=[{k:r[k] for k in ('episode','reset_seed','initial_hash')} for r in records]
        e['episode_ids_sha256']=hashlib.sha256(json.dumps(ids,sort_keys=True).encode()).hexdigest()
        e['raw_records_sha256']=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
        e['raw_record_aggregates_verified']=True
        files[f'task{task}_warm/{path.name}']=json.dumps(e,indent=2)+'\n'
        evaluations.append({k:e[k] for k in ('step','success','return_mean','deterministic_residual')})
    state['evaluations']=evaluations
    state['complete']=(folder/'CHECKS_PASSED.json').exists()
    if state['complete']:
        h=hashlib.sha256()
        with (folder/'final.pkl').open('rb') as f:
            for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
        assert h.hexdigest()==state['DONE.json']['checkpoint_sha256']
        state['final_checkpoint_sha256_verified']=True
    log=suite/f'task{task}_warm.log'
    if log.exists(): state['log_tail']=log.read_text(errors='replace').splitlines()[-3:]
    tasks[str(task)]=state
ps=subprocess.check_output(['ps','-eo','pid,etime,args'],text=True)
processes=[s for s in ps.splitlines() if ('launch_utd00625.py' in s or 'run_utd00625.py' in s) and 'ps -eo' not in s and 'sshd:' not in s]
state=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),tasks=tasks,processes=processes)
for name in ('suite_status.json','PREFLIGHT_PASSED.json'):
    if (suite/name).exists(): state[name]=json.loads((suite/name).read_text())
state['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
state['launcher_tail']=(suite/'launcher.log').read_text(errors='replace').splitlines()[-6:] if (suite/'launcher.log').exists() else []
smoke=suite/'smoke_task1_warm.log'
if smoke.exists(): state['smoke_tail']=smoke.read_text(errors='replace').splitlines()[-4:]
print(base64.b64encode(gzip.compress(json.dumps(dict(state=state,files=files),ensure_ascii=False).encode())).decode(),flush=True)
'''


def main():
    parser=argparse.ArgumentParser()
    saved=LOCAL/'monitor_state.json'
    latest_socket=json.loads(saved.read_text()).get('ssh_control_socket') if saved.exists() else None
    parser.add_argument('--socket',default=latest_socket or '/tmp/icra_qam_39459_utd.sock')
    parser.add_argument('--hostname',help='Optional verified real IP for the same remote host when local fake-IP routing fails')
    args=parser.parse_args()
    if args.socket != 'none' and not Path(args.socket).exists():
        parser.error('No reusable SSH socket. Run --socket none in an interactive terminal for fresh authentication.')
    if args.socket == 'none' and not sys.stderr.isatty():
        parser.error('Fresh authentication needs an interactive terminal; credentials are never read from files.')
    LOCAL.mkdir(parents=True,exist_ok=True)
    packet=LOCAL/'latest_packet.b64'
    command=['ssh','-S',args.socket,'-o','ConnectTimeout=12','-o','ServerAliveInterval=15',
        '-o','ServerAliveCountMax=2','-p','39459','root@connect.bjb1.seetacloud.com',
        '/root/autodl-tmp/qam_dawn_o2o/venv/bin/python','-']
    if args.socket != 'none':
        command[1:1]=['-o','BatchMode=yes']
    if args.hostname:
        command[1:1]=['-o',f'Hostname={args.hostname}','-o','HostKeyAlias=[connect.bjb1.seetacloud.com]:39459']
    with packet.open('w') as f:
        subprocess.run(command,input=REMOTE,text=True,stdout=f,check=True,timeout=180)
    decoded=gzip.decompress(base64.b64decode(packet.read_bytes())).decode()
    payload=json.loads(decoded)
    (LOCAL/'latest_packet.json').write_text(decoded)
    for name,content in payload['files'].items():
        path=LOCAL/name
        assert path.resolve().is_relative_to(LOCAL.resolve())
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(content)
    state=payload['state']
    state['ssh_control_socket']=args.socket
    (LOCAL/'monitor_state.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(dict(utc=state['utc'],suite=state.get('suite_status.json'),
        tasks={t:{'progress':s.get('progress.json'),'complete':s['complete'],'failed':s.get('FAILED.json'),
                    'matched_config':s.get('MATCHED_CONFIG_PASSED.json',{}).get('status')}
               for t,s in state['tasks'].items()},processes=state['processes'],gpu=state['gpu'],
        launcher_tail=state['launcher_tail'],smoke_tail=state.get('smoke_tail')),indent=2))


if __name__=='__main__': main()
