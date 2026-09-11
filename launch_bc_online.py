"""Smoke-test BC adapters while BC trains, then run four arms on two GPUs."""
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
ROOT=Path(__file__).resolve().parent;RUNS=ROOT/'runs'
lock=(RUNS/'bc_online_suite.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
event_lock=threading.Lock()
METHODS=['native','dawn_online','dawn_qam','policy_decorator']


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def record(event,**fields):
    row=dict(event=event,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**fields)
    with event_lock:
        with (RUNS/'bc_online_suite.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        p=RUNS/'bc_online_suite_status.json.tmp';p.write_text(json.dumps(row,indent=2)+'\n');p.replace(RUNS/'bc_online_suite_status.json')
        print(json.dumps(row),flush=True)
def execute(name,argv,gpu):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),XLA_PYTHON_CLIENT_MEM_FRACTION='.65',OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl',PYTHONUNBUFFERED='1',JAX_COMPILATION_CACHE_DIR=str(ROOT/'jax_cache'))
    with (RUNS/(name+'.log')).open('a') as log:
        p=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        record('started',name=name,pid=p.pid,gpu=gpu,command=argv);code=p.wait()
    record('finished' if code==0 else 'failed',name=name,exit_code=code)
    if code:raise RuntimeError(name+' failed')
def wait_done(folder):
    while not (folder/'DONE.json').exists():
        if (folder/'FAILED.json').exists():raise RuntimeError((folder/'FAILED.json').read_text())
        if 'failed' in json.loads((RUNS/'bc_pretrain_suite_status.json').read_text())['event']:
            raise RuntimeError('BC pretraining suite failed')
        time.sleep(5)
def cache(smoke,gpu):
    cache_dir=ROOT/'data'/('bc_smoke_base_cache_seed0' if smoke else 'bc_base_cache_seed0')
    checkpoint=RUNS/('smoke_bc_offline' if smoke else 'bc_offline')/'final.pkl'
    name='cache_bc_smoke' if smoke else 'cache_bc'
    argv=[sys.executable,'run_qam_replay.py','--stage','cache','--out',str(RUNS/name),
        '--offline-checkpoint',str(checkpoint),'--offline-steps','100' if smoke else '500000',
        '--cache-dir',str(cache_dir)]
    execute(name,argv,gpu)
def run(method,gpu,smoke=False):
    name=('smoke_bc_' if smoke else 'bc_')+method
    argv=[sys.executable,'run_bc_online.py','--method',method,'--out',str(RUNS/name)]
    if smoke:argv += ['--offline-checkpoint',str(RUNS/'smoke_bc_offline/final.pkl'),
        '--offline-steps','100','--online-steps','50','--warmup','10','--native-start','10','--prog-explore','30',
        '--cache-dir',str(ROOT/'data/bc_smoke_base_cache_seed0'),'--eval-episodes','1','--final-episodes','1']
    execute(name,argv,gpu)
def worker(gpu,methods):
    for method in methods:run(method,gpu)
def main():
    files=['run_bc_online.py','launch_bc_online.py','bc_pretrain.py','BC_PROTOCOL.md','run.py','dawn_agent.py',
        'run_qam_replay.py','qam_replay.py','dawn_replay_agent.py','run_policy_decorator.py','policy_decorator_agent.py']
    files += [str(p.relative_to(ROOT)) for p in (ROOT/'vendor/qam').rglob('*.py')]
    sources={p:digest(ROOT/p) for p in files};path=RUNS/'bc_online_source_manifest.json'
    if path.exists():assert json.loads(path.read_text())==sources
    else:path.write_text(json.dumps(sources,indent=2)+'\n')
    wait_done(RUNS/'smoke_bc_offline')
    cache(True,1)
    for method in METHODS:run(method,1,True)
    reference=json.loads((RUNS/'smoke_bc_offline/DONE.json').read_text())
    for method in METHODS:
        folder=RUNS/('smoke_bc_'+method)
        transfer=json.loads((folder/'BC_TRANSFER.json').read_text())
        assert transfer['critic_hash']==transfer['target_hash']==reference['q_hash']
        assert transfer['flow_hash']==reference['flow_hash']
        done=json.loads((folder/'DONE.json').read_text())
        assert done['steps']==50 and done['updates']==(41 if method=='native' else 10)
        assert done['critic_initialization']=='random'
        if method!='native':
            assert done['final_flow_hash']==reference['flow_hash']
            init=json.loads((folder/'initial_hashes.json').read_text())
            assert init['critic']==init['target']==reference['q_hash']
            assert done['final_critic_hash']!=init['critic']
        if method in ['dawn_qam','policy_decorator']:
            assert done['replay_stats']['replay_size']==1000050
    (RUNS/'BC_ONLINE_CHECKS_PASSED.json').write_text(json.dumps(dict(status='passed',sources=sources),indent=2)+'\n')
    record('bc_online_checks_passed_waiting_for_offline')
    wait_done(RUNS/'bc_offline')
    cache(False,0)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(worker,0,['native','dawn_qam']),pool.submit(worker,1,['policy_decorator','dawn_online'])]
        for f in futures:f.result()
    record('bc_all_training_complete')
if __name__=='__main__':
    try:main()
    except BaseException as e:record('bc_online_suite_failed',error=str(e));raise
