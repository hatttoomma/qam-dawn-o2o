"""Run checks, shared offline training, then three online arms on two GPUs."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parent
RUNS=ROOT/'runs'
RUNS.mkdir(exist_ok=True)
LOCK=(RUNS/'suite.lock').open('w')
fcntl.flock(LOCK,fcntl.LOCK_EX|fcntl.LOCK_NB)
EVENT_LOCK=threading.Lock()


def record(event,**kwargs):
    value=dict(event=event,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kwargs)
    with EVENT_LOCK:
        with (RUNS/'suite.jsonl').open('a') as f:
            f.write(json.dumps(value)+'\n')
        temp=RUNS/'suite_status.json.tmp'
        temp.write_text(json.dumps(value,indent=2)+'\n')
        temp.replace(RUNS/'suite_status.json')
        print(json.dumps(value),flush=True)


def run(name,argv,gpu):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),XLA_PYTHON_CLIENT_MEM_FRACTION='.65',
             OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
             JAX_COMPILATION_CACHE_DIR=str(ROOT/'jax_cache'),MUJOCO_GL='egl',PYTHONUNBUFFERED='1')
    with (RUNS/f'{name}.log').open('a') as out:
        process=subprocess.Popen([sys.executable,*argv],cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT)
        record('started',name=name,pid=process.pid,gpu=gpu,command=argv)
        code=process.wait()
        record('finished' if code==0 else 'failed',name=name,exit_code=code,gpu=gpu)
        if code:
            raise RuntimeError(f'{name} exited with status {code}; see runs/{name}.log')
    return gpu


def train(name,stage,gpu,smoke=False):
    out=RUNS/name
    args=['run.py','--stage',stage,'--out',str(out)]
    if smoke:
        args+=['--offline-steps','32','--online-steps','50','--warmup','10','--native-start','10',
               '--eval-episodes','1','--final-episodes','1','--block','16','--log-interval','16',
               '--offline-checkpoint',str(RUNS/'smoke_offline/final.pkl')]
    run(name,args,gpu)
    return gpu


def check_smoke():
    warm=json.loads((RUNS/'smoke_warm/initial_hashes.json').read_text())
    cold=json.loads((RUNS/'smoke_random/initial_hashes.json').read_text())
    assert warm['actor']==cold['actor']
    assert warm['critic']!=cold['critic']
    w=json.loads((RUNS/'smoke_warm/warmup.json').read_text())
    c=json.loads((RUNS/'smoke_random/warmup.json').read_text())
    assert w['replay_hash']==c['replay_hash']
    for name in ['smoke_warm','smoke_random']:
        x=json.loads((RUNS/name/'DONE.json').read_text())
        assert x['steps']==50 and x['updates']==10
        assert x['initial_hashes']['flow']==x['final_flow_hash']
        assert x['initial_hashes']['critic']!=x['final_critic_hash']
        assert x['initial_hashes']['actor']!=x['final_actor_hash']
    (RUNS/'CHECKS_PASSED.json').write_text(json.dumps(dict(smoke='passed',seed=0),indent=2)+'\n')


def main():
    record('suite_started',pid=os.getpid())
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in ROOT.rglob('*.py') if not any(x in p.parts for x in ('venv','runs','jax_cache'))}
    (RUNS/'source_manifest.json').write_text(json.dumps(sources,indent=2)+'\n')
    with (RUNS/'pip_freeze.txt').open('w') as f:
        subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
    run('unit_tests',['-m','pytest','-q','test_design.py'],0)
    train('smoke_offline','offline',0,True)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(train,'smoke_native','native',0,True),pool.submit(train,'smoke_warm','warm',1,True)]
        for f in futures:
            f.result()
    train('smoke_random','random',1,True)
    check_smoke()
    record('checks_passed')
    train('offline','offline',0)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        pending={pool.submit(train,'native','native',0):'native',pool.submit(train,'warm','warm',1):'warm'}
        first=next(concurrent.futures.as_completed(pending))
        gpu=first.result()
        remaining=[f for f in pending if f is not first]
        remaining.append(pool.submit(train,'random','random',gpu))
        for f in remaining:
            f.result()
    record('all_training_complete')


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        record('suite_failed',type=type(exc).__name__,message=str(exc))
        raise
