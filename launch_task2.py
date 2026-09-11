"""Audit task2, smoke test both adapters, then run shared offline and two online arms."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
RUNS=ROOT/'runs/task2'
RUNS.mkdir(parents=True,exist_ok=True)


def load(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def record(event,**fields):
    value=dict(event=event,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**fields)
    with (RUNS/'suite.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
    p=RUNS/'suite_status.json.tmp';p.write_text(json.dumps(value,indent=2)+'\n');p.replace(RUNS/'suite_status.json')
    print(json.dumps(value),flush=True)


def execute(name,args):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',XLA_PYTHON_CLIENT_MEM_FRACTION='.65',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl',
        PYTHONUNBUFFERED='1',JAX_COMPILATION_CACHE_DIR=str(ROOT/'jax_cache'))
    with (RUNS/(name+'.log')).open('a') as out:
        p=subprocess.Popen([sys.executable,*args],cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT)
        record('started',name=name,pid=p.pid,gpu=0,command=args)
        code=p.wait()
    record('finished' if code==0 else 'failed',name=name,exit_code=code)
    if code:raise RuntimeError(name+' failed; see '+str(RUNS/(name+'.log')))


def train(stage,smoke=False):
    name=('smoke_' if smoke else '')+stage
    args=['run_task2.py','--stage',stage,'--out',str(RUNS/name)]
    if smoke:
        args+=['--offline-steps','32','--online-steps','50','--warmup','10','--native-start','10',
               '--eval-episodes','1','--final-episodes','1','--block','16','--log-interval','16',
               '--offline-checkpoint',str(RUNS/'smoke_offline/final.pkl')]
    execute(name,args)


def smoke_checks():
    offline=load(RUNS/'smoke_offline/DONE.json')
    native=load(RUNS/'smoke_native/DONE.json')
    warm=load(RUNS/'smoke_warm/DONE.json')
    init=load(RUNS/'smoke_warm/initial_hashes.json')
    assert offline['step']==32
    assert native['steps']==warm['steps']==50
    assert native['updates']==41 and warm['updates']==10
    assert init['critic']==offline['q_hash'] and init['target']==offline['target_q_hash']
    assert native['initial_flow_hash']==warm['final_flow_hash']==offline['flow_hash']
    assert native['final_flow_hash']!=offline['flow_hash']
    assert warm['final_critic_hash']!=init['critic'] and warm['final_actor_hash']!=init['actor']
    a=load(RUNS/'smoke_native/eval_000000_001.json')
    b=load(RUNS/'smoke_warm/eval_000000_001.json')
    assert a['records']==b['records']
    manifests=[load(RUNS/name/'task_manifest.json') for name in ['smoke_offline','smoke_native','smoke_warm']]
    assert manifests[0]==manifests[1]==manifests[2]
    (RUNS/'CHECKS_PASSED.json').write_text(json.dumps(dict(status='passed',
        task_audit='TASK_AUDIT_PASSED.json',offline_updates=32,online_steps=50,
        native_updates=41,dawn_updates=10,task2_critic_inherited=True,
        task2_initial_evaluations_identical=True,frozen_flow_unchanged=True),indent=2)+'\n')


def main():
    historical=load(ROOT/'runs/source_manifest.json')
    for n,h in historical.items():assert digest(ROOT/n)==h,n
    sources=dict(historical)
    for n in ['run_task2.py','audit_task2.py','launch_task2.py','TASK2_PROTOCOL.md']:
        sources[n]=digest(ROOT/n)
    manifest=RUNS/'source_manifest.json'
    if manifest.exists():assert load(manifest)==sources
    else:manifest.write_text(json.dumps(sources,indent=2)+'\n')
    if not (RUNS/'pip_freeze.txt').exists():
        with (RUNS/'pip_freeze.txt').open('w') as f:subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
    if not (RUNS/'TASK_AUDIT_PASSED.json').exists():execute('task_audit',['audit_task2.py'])
    assert load(RUNS/'TASK_AUDIT_PASSED.json')['status']=='passed'
    if not (RUNS/'CHECKS_PASSED.json').exists():
        for stage in ['offline','native','warm']:train(stage,True)
        smoke_checks()
    record('checks_passed_starting_formal_training')
    for stage in ['offline','native','warm']:train(stage)
    record('all_training_complete')
    # Reporting is installed while the long training runs. It does not touch models.
    if (ROOT/'export_task2.py').exists():execute('report',['export_task2.py'])
    record('task2_suite_complete')


if __name__=='__main__':
    lock=(RUNS/'suite.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:main()
    except BaseException as e:
        record('suite_failed',type=type(e).__name__,message=str(e))
        raise
