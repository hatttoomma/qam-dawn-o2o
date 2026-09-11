"""Verify target-only intervention and execute four paired online runs."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
RUNS=ROOT/'runs/td_ablation'
ARMS=[(f'task{task}_{kind}',task,kind) for task in (1,2) for kind in ('soft','hard')]

def load(p): return json.loads(p.read_text())
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def record(event,**fields):
    value=dict(event=event,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**fields)
    with (RUNS/'suite.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
    p=RUNS/'suite_status.json.tmp';p.write_text(json.dumps(value,indent=2)+'\n');p.replace(RUNS/'suite_status.json')
    print(json.dumps(value),flush=True)

def execute(jobs):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',XLA_PYTHON_CLIENT_MEM_FRACTION='.21',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl',
        __EGL_VENDOR_LIBRARY_FILENAMES=str(RUNS/'egl_vendor.json'),
        PYTHONUNBUFFERED='1',JAX_COMPILATION_CACHE_DIR=str(ROOT/'jax_cache'))
    pending,errors=[],[]
    for name,cmd in jobs:
        if name!='report' and (RUNS/name/'DONE.json').exists():
            if name.startswith(('smoke_task','task')): assert (RUNS/name/'CHECKS_PASSED.json').exists()
            record('already_complete',name=name);continue
        output=(RUNS/(name+'.log')).open('a')
        p=subprocess.Popen([sys.executable,*cmd],cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
        record('started',name=name,pid=p.pid,command=cmd,memory_fraction=.21)
        pending.append((name,p,output))
    while pending:
        for name,p,output in pending[:]:
            code=p.poll()
            if code is not None:
                output.close();pending.remove((name,p,output))
                record('finished' if code==0 else 'failed',name=name,exit_code=code)
                if code:errors.append(name)
        if pending:time.sleep(5)
    if errors:raise RuntimeError('Failed jobs: '+', '.join(errors))

def command(name,task,kind,smoke=False):
    cmd=['run_td_ablation.py','--task',str(task),'--td-target',kind,'--stage','warm','--out',str(RUNS/name)]
    if smoke:cmd+=['--online-steps','60','--warmup','20','--eval-episodes','1','--final-episodes','1']
    return cmd

def check_smokes():
    for task in (1,2):
        ref=RUNS/'reference_task1' if task==1 else ROOT/'runs/task2/td_reference'
        soft=RUNS/f'smoke_task{task}_soft';hard=RUNS/f'smoke_task{task}_hard'
        assert load(ref/'DONE.json')['checkpoint_sha256']==load(soft/'DONE.json')['checkpoint_sha256']
        for f in ('eval_000000_001.json','eval_000060_001.json','eval_000060_001_mean_residual.json'):
            assert load(ref/f)['records']==load(soft/f)['records']
        assert load(soft/'initial_hashes.json')==load(hard/'initial_hashes.json')==load(ref/'initial_hashes.json')
        assert load(soft/'actual_agent_config.json')['full_initial_agent_hash']==load(hard/'actual_agent_config.json')['full_initial_agent_hash']
        assert load(soft/'warmup.json')==load(hard/'warmup.json')==load(ref/'warmup.json')
        assert load(soft/'DONE.json')['checkpoint_sha256']!=load(hard/'DONE.json')['checkpoint_sha256']
        for p in (soft,hard):
            assert load(p/'CHECKS_PASSED.json')['status']=='passed'
            assert load(p/'backup_math.json')['status']=='passed'
            assert load(p/'DONE.json')['updates']==10
            assert (p/'probe_000060.json').exists() and (p/'mc_000060_001.json').exists()
    result=dict(status='passed',tasks=[1,2],original_soft_full_checkpoint_parity=True,
        read_only_diagnostics_preserve_original_training=True,identical_initial_states_and_warmup=True,
        hard_changes_learned_state=True,backup_math_and_terminal_masks=True,
        concurrent_full_model_smokes=4,smoke_steps=60,smoke_updates=10)
    (RUNS/'CHECKS_PASSED.json').write_text(json.dumps(result,indent=2)+'\n');record('checks_passed',**result)

def main():
    assert load(RUNS/'runtime_setup.json')['probe_exit_code']==0
    old=load(ROOT/'runs/newdata_ablation/source_manifest.json')
    for n,h in old.items():assert digest(ROOT/n)==h,n
    sources=dict(old)
    for n in ['run_td_ablation.py','td_diagnostics.py','launch_td_ablation.py','TD_ABLATION_PROTOCOL.md']:sources[n]=digest(ROOT/n)
    p=RUNS/'source_manifest.json'
    if p.exists():assert load(p)==sources
    else:p.write_text(json.dumps(sources,indent=2)+'\n')
    for task in (1,2):
        p=ROOT/('runs/offline' if task==1 else 'runs/task2/offline')
        assert digest(p/'final.pkl')==load(p/'DONE.json')['checkpoint_sha256']
    if not (RUNS/'pip_freeze.txt').exists():
        with (RUNS/'pip_freeze.txt').open('w') as f:subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
    if not (RUNS/'CHECKS_PASSED.json').exists():
        tail=['--stage','warm','--online-steps','60','--warmup','20','--eval-episodes','1','--final-episodes','1']
        execute([('reference_task1',['run.py','--out',str(RUNS/'reference_task1'),*tail]),
            ('reference_task2',['run_task2.py','--out',str(ROOT/'runs/task2/td_reference'),*tail]),
            *[(f'smoke_task{t}_soft',command(f'smoke_task{t}_soft',t,'soft',True)) for t in (1,2)]])
        execute([(f'smoke_task{t}_hard',command(f'smoke_task{t}_hard',t,'hard',True)) for t in (1,2)])
        check_smokes()
    record('starting_formal_pairs',tasks=[1,2],td_targets=['soft','hard'])
    execute([(n,command(n,t,k)) for n,t,k in ARMS]);record('all_training_complete')
    if (ROOT/'export_td_ablation.py').exists():
        execute([('report',['export_td_ablation.py'])]);record('td_ablation_suite_complete')
    else:record('awaiting_report')

if __name__=='__main__':
    RUNS.mkdir(parents=True,exist_ok=True)
    lock=(RUNS/'suite.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:main()
    except BaseException as e:record('suite_failed',type=type(e).__name__,message=str(e));raise
