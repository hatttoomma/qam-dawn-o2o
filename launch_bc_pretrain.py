"""Checked BC pretraining job; online suite has its own launcher."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent;RUNS=ROOT/'runs'
lock=(RUNS/'bc_pretrain_suite.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='0',XLA_PYTHON_CLIENT_MEM_FRACTION='.65',OMP_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl',PYTHONUNBUFFERED='1',JAX_COMPILATION_CACHE_DIR=str(ROOT/'jax_cache'))

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def record(event,**data):
    row=dict(event=event,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**data)
    with (RUNS/'bc_pretrain_suite.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    p=RUNS/'bc_pretrain_suite_status.json.tmp';p.write_text(json.dumps(row,indent=2)+'\n');p.replace(RUNS/'bc_pretrain_suite_status.json')
    print(json.dumps(row),flush=True)
def execute(name,argv):
    with (RUNS/(name+'.log')).open('a') as log:
        p=subprocess.Popen(argv,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
        record('started',name=name,pid=p.pid,argv=argv);code=p.wait()
    record('finished' if code==0 else 'failed',name=name,exit_code=code)
    if code:raise RuntimeError(name+' failed')
def main():
    for name in ['source_manifest.json','replay_source_manifest.json','policy_decorator_source_manifest.json']:
        for p,h in json.loads((RUNS/name).read_text()).items():assert digest(ROOT/p)==h,p
    files=['bc_pretrain.py','test_bc_pretrain.py','launch_bc_pretrain.py','BC_PROTOCOL.md','run.py','dawn_agent.py']
    files += [str(p.relative_to(ROOT)) for p in (ROOT/'vendor/qam').rglob('*.py')]
    sources={p:digest(ROOT/p) for p in files};path=RUNS/'bc_pretrain_source_manifest.json'
    if path.exists():assert json.loads(path.read_text())==sources
    else:path.write_text(json.dumps(sources,indent=2)+'\n')
    assert '3 passed' in (RUNS/'bc_unit.log').read_text()
    execute('smoke_bc_offline',[sys.executable,'bc_pretrain.py','--out',str(RUNS/'smoke_bc_offline'),
        '--offline-steps','100','--eval-episodes','1'])
    done=json.loads((RUNS/'smoke_bc_offline/DONE.json').read_text())
    assert done['step']==100 and done['critic_trained']==False and done['q_hash']==done['target_q_hash']
    (RUNS/'BC_PRETRAIN_CHECKS_PASSED.json').write_text(json.dumps(dict(status='passed',sources=sources),indent=2)+'\n')
    execute('bc_offline',[sys.executable,'bc_pretrain.py','--out',str(RUNS/'bc_offline')])
    record('bc_pretraining_complete')
if __name__=='__main__':
    try:main()
    except BaseException as e:record('bc_pretrain_failed',error=str(e));raise
