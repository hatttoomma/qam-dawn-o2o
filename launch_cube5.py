"""Audit, smoke, and train the three missing task pairs; reuse validated Task1/2."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RUNS = ROOT/'runs/cube5'


def load(p): return json.loads(p.read_text())
def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def record(event, **fields):
    value = dict(event=event,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**fields)
    with (RUNS/'suite.jsonl').open('a') as f: f.write(json.dumps(value)+'\n')
    p=RUNS/'suite_status.json.tmp';p.write_text(json.dumps(value,indent=2)+'\n')
    p.replace(RUNS/'suite_status.json')
    print(json.dumps(value),flush=True)


def execute(jobs):
    assert len(jobs) <= 3
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',XLA_PYTHON_CLIENT_MEM_FRACTION='.30',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl',
        __EGL_VENDOR_LIBRARY_FILENAMES=str(RUNS/'egl_vendor.json'),PYTHONUNBUFFERED='1',
        JAX_COMPILATION_CACHE_DIR=str(ROOT/'jax_cache'))
    pending, errors = [], []
    for name,cmd in jobs:
        if (RUNS/name/'DONE.json').exists():
            assert load(RUNS/name/'CHECKS_PASSED.json')['status']=='passed'
            assert digest(RUNS/name/'final.pkl')==load(RUNS/name/'DONE.json')['checkpoint_sha256']
            record('already_complete',name=name);continue
        out=(RUNS/(name+'.log')).open('a')
        p=subprocess.Popen([sys.executable,*cmd],cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT)
        record('started',name=name,pid=p.pid,command=cmd,memory_fraction=.30)
        pending.append((name,p,out))
    while pending:
        for name,p,out in pending[:]:
            code=p.poll()
            if code is not None:
                out.close();pending.remove((name,p,out))
                record('finished' if code==0 else 'failed',name=name,exit_code=code)
                if code:errors.append(name)
        if pending:time.sleep(5)
    if errors:raise RuntimeError('Failed jobs: '+', '.join(errors))


def command(task,stage,smoke=False):
    name=('smoke_' if smoke else '')+f'task{task}_{stage}'
    cmd=['run_cube5.py','--task',str(task),'--stage',stage,'--out',str(RUNS/name),'--dawn-batch','256']
    if smoke:
        cmd+=['--offline-steps','32','--online-steps','60','--warmup','20','--native-start','20',
              '--eval-episodes','1','--final-episodes','1','--block','16','--log-interval','16',
              '--offline-checkpoint',str(RUNS/f'smoke_task{task}_offline/final.pkl')]
    return name,cmd


def verify_reuse():
    entries=[]
    for task in (1,2):
        offline=ROOT/('runs/offline' if task==1 else 'runs/task2/offline')
        native=ROOT/('runs/native' if task==1 else 'runs/task2/native')
        dawn=ROOT/f'runs/actor_input_ablation/task{task}_base_action'
        oc=load(offline/'config.json')
        od=load(offline/'DONE.json')
        assert od['step']==500000 and oc['qam_config']['inv_temp']==1 and oc['qam_config']['edit_scale']==0
        assert digest(offline/'final.pkl')==od['checkpoint_sha256']
        for method,path in [('qam_edit',native),('dawn',dawn)]:
            cfg,done=load(path/'config.json'),load(path/'DONE.json')
            assert cfg['offline_sha256']==od['checkpoint_sha256']
            assert cfg['seed']==0 and cfg['offline_steps']==500000 and cfg['online_steps']==50000
            assert done['steps']==50000
            assert digest(path/'final.pkl')==done['checkpoint_sha256']
            if method=='dawn':
                actual=load(path/'actual_agent_config.json')
                assert actual['actor_input']=='base_action' and actual['td_target']=='hard'
                assert actual['utd']==.25 and cfg['dawn_batch']==256
                assert done['updates']==7500 and done['final_flow_hash']==od['flow_hash']
                assert done['initial_hashes']['critic']==od['q_hash']
                assert done['initial_hashes']['target']==od['target_q_hash']
                assert load(path/'CHECKS_PASSED.json')['status']=='passed'
            else:
                assert done['updates']==45001 and done['initial_flow_hash']==od['flow_hash']
            endpoint=load(path/'eval_050000_100.json')
            assert endpoint['episodes']==100 and endpoint['step']==50000
            entries.append(dict(task=task,method=method,path=str(path.relative_to(ROOT)),
                offline_path=str(offline.relative_to(ROOT)),checkpoint_sha256=done['checkpoint_sha256'],
                offline_sha256=od['checkpoint_sha256'],updates=done['updates'],
                final_evaluation_sha256=digest(path/'eval_050000_100.json'),
                source='historical native' if method=='qam_edit' else 'preceding actor-input experiment'))
    p=RUNS/'reused_runs.json'
    result=dict(status='passed',entries=entries)
    if p.exists():assert load(p)==result
    else:p.write_text(json.dumps(result,indent=2)+'\n')
    record('reuse_verified',runs=len(entries))


def smoke_checks():
    entries=[]
    for task in (3,4,5):
        off,native,warm=[RUNS/f'smoke_task{task}_{s}' for s in ('offline','native','warm')]
        for p in (off,native,warm):assert load(p/'CHECKS_PASSED.json')['status']=='passed'
        assert load(off/'DONE.json')['step']==32
        assert load(native/'DONE.json')['updates']==41
        assert load(warm/'DONE.json')['updates']==10
        a,b=load(native/'eval_000000_001.json'),load(warm/'eval_000000_001.json')
        assert a['records']==b['records']
        assert load(native/'task_manifest.json')==load(warm/'task_manifest.json')==load(off/'task_manifest.json')
        assert load(warm/'actor_input_checks.json')['actor_parameter_count']==160562
        entries.append(dict(task=task,offline_updates=32,native_updates=41,dawn_updates=10,
            matching_initial_evaluations=True,matching_task_manifest=True))
    result=dict(status='passed',tasks=entries)
    (RUNS/'CHECKS_PASSED.json').write_text(json.dumps(result,indent=2)+'\n')
    record('smokes_passed',**result)


def main():
    sources=load(ROOT/'runs/actor_input_ablation/source_manifest.json')
    for n,h in sources.items():assert digest(ROOT/n)==h,n
    for n in ['audit_cube5.py','run_cube5.py','launch_cube5.py','CUBE5_PROTOCOL.md']:
        sources[n]=digest(ROOT/n)
    p=RUNS/'source_manifest.json'
    if p.exists():assert load(p)==sources
    else:p.write_text(json.dumps(sources,indent=2)+'\n')
    if not (RUNS/'pip_freeze.txt').exists():
        with (RUNS/'pip_freeze.txt').open('w') as f:
            subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
    verify_reuse()
    if not (RUNS/'TASK_AUDIT_PASSED.json').exists():execute([('task_audit',['audit_cube5.py'])])
    assert load(RUNS/'TASK_AUDIT_PASSED.json')['status']=='passed'
    if not (RUNS/'CHECKS_PASSED.json').exists():
        for stage in ('offline','native','warm'):
            execute([command(t,stage,True) for t in (3,4,5)])
        smoke_checks()
    record('starting_formal_training',tasks=[3,4,5],offline_updates=500000,online_steps=50000)
    execute([command(t,'offline') for t in (3,4,5)])
    execute([command(t,'native') for t in (3,4,5)])
    execute([command(t,'warm') for t in (3,4,5)])
    record('all_training_complete')
    if (ROOT/'export_cube5.py').exists():execute([('report',['export_cube5.py'])])
    record('cube5_suite_complete')


if __name__ == '__main__':
    RUNS.mkdir(parents=True,exist_ok=True)
    lock=(RUNS/'suite.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:main()
    except BaseException as e:
        record('suite_failed',type=type(e).__name__,message=str(e));raise
