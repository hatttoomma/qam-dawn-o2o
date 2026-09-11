"""Run the two additional replay ablations without changing the original suite."""
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
LOCK=(RUNS/'replay_suite.lock').open('w')
fcntl.flock(LOCK,fcntl.LOCK_EX|fcntl.LOCK_NB)
EVENT_LOCK=threading.Lock()


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def record(event,**fields):
    x=dict(event=event,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**fields)
    with EVENT_LOCK:
        with (RUNS/'replay_suite.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
        tmp=RUNS/'replay_suite_status.json.tmp'
        tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(RUNS/'replay_suite_status.json')
        print(json.dumps(x),flush=True)


def run(name,stage,gpu,smoke=False):
    argv=[sys.executable,'run_qam_replay.py','--stage',stage,'--out',str(RUNS/name)]
    if smoke:argv+=['--online-steps','50','--warmup','10','--eval-episodes','1','--final-episodes','1']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),XLA_PYTHON_CLIENT_MEM_FRACTION='.65',
             OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl',
             PYTHONUNBUFFERED='1',JAX_COMPILATION_CACHE_DIR=str(ROOT/'jax_cache'))
    with (RUNS/(name+'.log')).open('a') as f:
        p=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        record('started',name=name,pid=p.pid,gpu=gpu,command=argv)
        code=p.wait()
    record('finished' if code==0 else 'failed',name=name,exit_code=code,gpu=gpu)
    if code:raise RuntimeError(f'{name} failed: {code}')


def main():
    unit=json.loads((RUNS/'REPLAY_UNIT_PASSED.json').read_text())
    assert unit['status']=='passed'
    for p,h in unit['sources'].items():assert digest(ROOT/p)==h,p
    source_files=['run.py','dawn_agent.py','qam_replay.py','dawn_replay_agent.py',
                  'run_qam_replay.py','test_qam_replay.py','launch_qam_replay.py']
    source_files+=[str(p.relative_to(ROOT)) for p in (ROOT/'vendor/qam').rglob('*.py')]
    sources={p:digest(ROOT/p) for p in source_files}
    path=RUNS/'replay_source_manifest.json'
    if path.exists():assert json.loads(path.read_text())==sources
    else:path.write_text(json.dumps(sources,indent=2)+'\n')
    record('replay_suite_started',pid=os.getpid())
    run('cache_offline_base','cache',0)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(run,'smoke_'+arm+'_qam_replay',arm,gpu,True)
                 for gpu,arm in enumerate(['warm','random'])]
        for f in futures:f.result()
    for arm in ['warm','random']:
        out=RUNS/('smoke_'+arm+'_qam_replay')
        initial=json.loads((out/'initial_hashes.json').read_text())
        reference=json.loads((RUNS/arm/'initial_hashes.json').read_text())
        assert initial==reference,arm
        done=json.loads((out/'DONE.json').read_text())
        assert done['steps']==50 and done['updates']==10
        assert done['replay_stats']['replay_size']==1000050
        assert done['replay_stats']['sampled_rows']==10240
        assert done['final_flow_hash']==initial['flow']
        assert done['final_critic_hash']!=initial['critic'] and done['final_actor_hash']!=initial['actor']
    (RUNS/'REPLAY_CHECKS_PASSED.json').write_text(json.dumps(dict(status='passed',seed=0),indent=2)+'\n')
    record('replay_checks_passed')
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(run,arm+'_qam_replay',arm,gpu) for gpu,arm in enumerate(['warm','random'])]
        for f in futures:f.result()
    record('replay_all_training_complete')


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        record('replay_suite_failed',type=type(exc).__name__,message=str(exc))
        raise
