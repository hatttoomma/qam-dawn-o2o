"""Dispatch and compare all five UTD0.0625 tasks using matched 80k prefixes."""
from collections import deque
import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import launch_warmup80k as previous
from utd00625_support import ROOT, RUNS, UTD, reference_dir, collector_dir, schedule_source

previous.RUNS = RUNS
load, write, record = previous.load, previous.write, previous.record
TASKS = (1, 2, 3, 4, 5)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(name, rows):
    with (RUNS/name).open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prerequisites():
    ref = reference_dir(1).parent
    request = dict(load(ref/'suite_config.json'), tasks=list(TASKS), utd=UTD, gradient_updates=4375)
    path=RUNS/'suite_config.json'
    if path.exists():
        assert load(path)==request
    else:
        write(path,request)
    assert shutil.disk_usage(RUNS).free > 6*1024**3
    manifests = {}
    for folder in (reference_dir(1).parent,reference_dir(2).parent):
        assert load(folder/'suite_status.json')['event']=='all_training_complete'
        for name,expected in load(folder/'source_manifest.json').items():
            assert digest(ROOT/name)==expected,name
            if name in manifests:
                assert manifests[name]==expected
            manifests[name]=expected
    extension=load(ROOT/'utd00625_extension_diff.json')
    assert digest(ROOT/extension['source'])==extension['source_sha256']
    assert digest(ROOT/extension['extended'])==extension['extended_sha256']
    normalized=(ROOT/extension['extended']).read_text()
    for change in reversed(extension['changes']):
        assert normalized.count(change['new'])==change['count']
        normalized=normalized.replace(change['new'],change['old'])
    assert normalized==(ROOT/extension['source']).read_text()
    for name in ('run_utd00625.py','utd00625_support.py','launch_utd00625.py','utd00625_extension_diff.json','UTD00625_PROTOCOL.md'):
        manifests[name]=digest(ROOT/name)
    write(RUNS/'source_manifest.json',manifests)
    checks=[]
    for task in TASKS:
        old=reference_dir(task)
        done=load(old/'DONE.json')
        assert done['steps']==150000 and done['updates']==17500
        assert load(old/'CHECKS_PASSED.json')['status']=='passed'
        assert digest(old/'final.pkl')==done['checkpoint_sha256']
        config=load(old/'requested_config.json')
        assert digest(Path(config['offline_checkpoint']))==config['offline_sha256']
        source=collector_dir(task)
        shared=load(old/'shared_initialization.json')
        prefix=load(source/'WARMUP_COLLECTED.json')
        assert prefix['checkpoint_sha256']==digest(source/'latest.pkl')==shared['collector_checkpoint_sha256']
        assert prefix['updates']==shared['updates']==0
        assert prefix['replay_hash']==shared['replay_hash']
        assert 80000<=prefix['step']<80005
        assert load(old/'eval_000000_100.json')==load(source/'eval_000000_100.json')
        checks.append(dict(task=task, reference=str(old), collector=str(source),
            checkpoint_sha256=prefix['checkpoint_sha256'],replay_hash=prefix['replay_hash'],step=prefix['step']))
    library=Path('/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0').resolve(strict=True)
    write(RUNS/'egl_vendor.json',dict(file_format_version='1.0.0',ICD=dict(library_path=str(library))))
    compatibility=load(ref/'runtime_compatibility.json')
    assert str(library)==compatibility['egl_library']
    assert previous.environment()['XLA_FLAGS']==compatibility['xla_flags']
    write(RUNS/'runtime_compatibility.json',compatibility)
    frozen=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    assert frozen==(ref/'pip_freeze.txt').read_text()
    (RUNS/'pip_freeze.txt').write_text(frozen)
    write(RUNS/'PREREQUISITES_VERIFIED.json',dict(status='passed',matched_warmup_prefixes=checks,
        original_source_hashes_verified=True,wrapper_changes_reversible=True,runtime_unchanged=True))
    write(RUNS/'storage.json',dict(logical_directory=str(RUNS),physical_directory=str(RUNS.resolve()),
        free_bytes=shutil.disk_usage(RUNS).free))
    record('prerequisites_verified',**request)


def command(task, smoke=False):
    name=f'{"smoke_" if smoke else ""}task{task}_warm'
    args=['run_utd00625.py','--task',str(task),'--stage','warm','--out',str(RUNS/name),
        '--dawn-batch','256','--warmup','80' if smoke else '80000',
        '--online-steps','200' if smoke else '150000','--eval-episodes','2' if smoke else '100',
        '--final-episodes','2' if smoke else '100','--shared-warmup',str(collector_dir(task,smoke))]
    if smoke: args.append('--smoke')
    return name,args


def smoke_test():
    if (RUNS/'PREFLIGHT_PASSED.json').exists(): return
    prefix=collector_dir(1,True)
    if not (prefix/'WARMUP_COLLECTED.json').exists():
        _,collect_args=command(1,True)
        collect_args[collect_args.index('--out')+1]=str(prefix)
        index=collect_args.index('--shared-warmup')
        del collect_args[index:index+2]
        collect_args.append('--collect-warmup')
        record('smoke_collect_started',task=1,command=collect_args)
        with (RUNS/'smoke_task1_collector.log').open('a') as f:
            subprocess.run([sys.executable,*collect_args],cwd=ROOT,env=previous.environment(),
                stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,check=True)
    name,args=command(1,True)
    if not (RUNS/name/'CHECKS_PASSED.json').exists():
        record('smoke_started',task=1,command=args)
        with (RUNS/f'{name}.log').open('a') as f:
            subprocess.run([sys.executable,*args],cwd=ROOT,env=previous.environment(),
                stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,check=True)
    d=load(RUNS/name/'DONE.json')
    assert d['steps']==200 and d['updates']==7
    assert load(RUNS/name/'MATCHED_CONFIG_PASSED.json')['status']=='passed'
    assert load(RUNS/name/'utd_schedule_audit.json')['ast_single_constant_change_verified']
    restores=[json.loads(s) for s in (RUNS/name/'restore_checks.jsonl').read_text().splitlines()]
    assert all(r['observation_max_abs_error']==0 for r in restores)
    write(RUNS/'PREFLIGHT_PASSED.json',dict(status='passed',smoke_steps=200,smoke_updates=7,
        initial_agent_matches_reference=True,restored_simulator_exact=True,plain_TD_verified=True,
        schedule_only_one_constant_changed=True))
    record('preflight_passed')


def run_queue():
    waiting=deque(TASKS)
    active={}
    try:
        while waiting or active:
            while waiting and len(active)<2:
                task=waiting.popleft()
                name,args=command(task)
                if (RUNS/name/'CHECKS_PASSED.json').exists():
                    record('already_complete',task=task,name=name)
                    continue
                f=(RUNS/f'{name}.log').open('a')
                process=subprocess.Popen([sys.executable,*args],cwd=ROOT,env=previous.environment(),
                    stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
                active[task]=(name,process,f)
                record('started',task=task,name=name,pid=process.pid,command=args)
            for task,(name,process,f) in list(active.items()):
                code=process.poll()
                if code is None: continue
                f.close()
                del active[task]
                record('finished' if code==0 else 'failed',task=task,name=name,exit_code=code)
                assert code==0 and (RUNS/name/'CHECKS_PASSED.json').exists(),name
                summarize_partial()
            if active: time.sleep(5)
    except BaseException:
        for _,process,f in active.values():
            process.terminate()
            f.close()
        raise


def summarize_partial():
    rows,comparison,paired=[],[],[]
    for task in TASKS:
        current=RUNS/f'task{task}_warm'
        if not (current/'CHECKS_PASSED.json').exists(): continue
        done=load(current/'DONE.json')
        assert done['steps']==150000 and done['updates']==4375
        assert digest(current/'final.pkl')==done['checkpoint_sha256']
        before,after=load(reference_dir(task)/'actual_agent_config.json'),load(current/'actual_agent_config.json')
        assert after==dict(before,utd=UTD)
        assert load(current/'shared_initialization.json')==load(reference_dir(task)/'shared_initialization.json')
        assert load(current/'initial_hashes.json')==load(reference_dir(task)/'initial_hashes.json')
        for line in (current/'metrics.jsonl').read_text().splitlines():
            metric=json.loads(line)
            assert metric['updates']==max(0,(metric['step']-80000)//16)
            assert all(not isinstance(v,float) or math.isfinite(v) for v in metric.values())
        reference_evals={}
        for utd,folder in ((.25,reference_dir(task)),(UTD,current)):
            for path in sorted(folder.glob('eval_*_100*.json')):
                e=load(path)
                records=e['records']
                assert len(records)==e['episodes']==100
                assert abs(sum(r['success'] for r in records)/100-e['success'])<1e-10
                assert abs(sum(r['return_'] for r in records)/100-e['return_mean'])<1e-8
                mode='offline' if e['step']==0 else 'mean_residual' if e['deterministic_residual'] else 'sampled'
                updates=max(0,int((e['step']-80000)*utd))
                rows.append(dict(task=task,utd=utd,step=e['step'],updates=updates,mode=mode,
                    success=e['success'],return_mean=e['return_mean']))
                if e['step']==150000:
                    comparison.append(dict(task=task,utd=utd,mode=mode,online_steps=150000,updates=updates,
                        success=e['success'],return_mean=e['return_mean']))
                    if utd==.25: reference_evals[mode]=e
                    else:
                        old=reference_evals[mode]
                        gained=lost=0
                        for a,b in zip(old['records'],records):
                            assert all(a[k]==b[k] for k in ('episode','reset_seed','initial_hash'))
                            gained+=int(not a['success'] and b['success'])
                            lost+=int(a['success'] and not b['success'])
                        paired.append(dict(task=task,mode=mode,old_success=old['success'],new_success=e['success'],
                            delta_success_pp=100*(e['success']-old['success']),old_return=old['return_mean'],
                            new_return=e['return_mean'],delta_return=e['return_mean']-old['return_mean'],
                            gained_episodes=gained,lost_episodes=lost,episodes=100))
    if rows:
        write(RUNS/'all_results.json',rows)
        write_csv('all_results.csv',rows)
        write_csv('final_comparison.csv',comparison)
        write(RUNS/'paired_comparison.json',paired)
        write_csv('paired_comparison.csv',paired)
    return len({r['task'] for r in rows})


if __name__=='__main__':
    RUNS.mkdir(parents=True,exist_ok=True)
    lock=(RUNS/'suite.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        prerequisites()
        smoke_test()
        run_queue()
        assert summarize_partial()==5
        record('all_training_complete',tasks=list(TASKS),utd=UTD,updates=4375)
    except BaseException as exc:
        record('suite_failed',error=type(exc).__name__,message=str(exc))
        raise
