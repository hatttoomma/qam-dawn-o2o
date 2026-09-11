"""Check UTD parameterization, then run two UTD=1 online experiments."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys

import launch_td_batch_ablation as base

ROOT=Path(__file__).resolve().parent
RUNS=ROOT/'runs/utd_ablation'
base.RUNS=RUNS
load,digest,record,execute=base.load,base.digest,base.record,base.execute


def command(name, task, utd, smoke=False):
    cmd=['run_utd_ablation.py','--task',str(task),'--td-target','hard','--utd',str(utd),
         '--stage','warm','--dawn-batch','256','--out',str(RUNS/name)]
    if smoke:cmd+=['--online-steps','60','--warmup','20','--eval-episodes','1','--final-episodes','1']
    return cmd


def check_smokes():
    results=[]
    for task in (1,2):
        ref=(ROOT/'runs/td_ablation/utd_task1_reference' if task==1 else
             ROOT/f'runs/td_ablation/batch_comparison/smoke_task{task}_b256')
        low=RUNS/f'smoke_task{task}_utd025';high=RUNS/f'smoke_task{task}_utd1'
        assert load(ref/'DONE.json')['checkpoint_sha256']==load(low/'DONE.json')['checkpoint_sha256']
        for filename in ['initial_hashes.json','actual_agent_config.json','warmup.json']:
            assert load(low/filename)==load(high/filename)==load(ref/filename),(task,filename)
        for filename in ['eval_000000_001.json','eval_000060_001.json','eval_000060_001_mean_residual.json']:
            assert load(low/filename)['records']==load(ref/filename)['records']
        for folder,utd,updates in [(low,.25,10),(high,1.,40)]:
            assert load(folder/'CHECKS_PASSED.json')['status']=='passed'
            assert load(folder/'utd_intervention.json')['utd']==utd
            assert load(folder/'DONE.json')['updates']==updates
            assert load(folder/'config.json')['utd']==utd
        results.append(dict(task=task,reference=str(ref.relative_to(ROOT)),original_full_checkpoint_parity=True,
            identical_initial_state_and_warmup=True,utd025_updates=10,utd1_updates=40))
    result=dict(status='passed',tasks=results,only_loop_change_is_utd=True)
    (RUNS/'CHECKS_PASSED.json').write_text(json.dumps(result,indent=2)+'\n')
    record('checks_passed',**result)


def main():
    assert load(RUNS/'runtime_setup.json')['probe_exit_code']==0
    sources=load(ROOT/'runs/td_ablation/batch_comparison/source_manifest.json')
    for n,h in sources.items():assert digest(ROOT/n)==h,n
    for n in ['run_utd_ablation.py','launch_utd_ablation.py','UTD_ABLATION_PROTOCOL.md']:
        sources[n]=digest(ROOT/n)
    p=RUNS/'source_manifest.json'
    if p.exists():assert load(p)==sources
    else:p.write_text(json.dumps(sources,indent=2)+'\n')
    for task in (1,2):
        offline=ROOT/('runs/offline' if task==1 else 'runs/task2/offline')
        assert digest(offline/'final.pkl')==load(offline/'DONE.json')['checkpoint_sha256']
    if not (RUNS/'pip_freeze.txt').exists():
        with (RUNS/'pip_freeze.txt').open('w') as f:
            subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
    if not (RUNS/'CHECKS_PASSED.json').exists():
        execute([(f'smoke_task{t}_{label}',command(f'smoke_task{t}_{label}',t,u,True))
                 for t in (1,2) for label,u in [('utd025',.25),('utd1',1.)]])
        check_smokes()
    record('starting_formal_runs',tasks=[1,2],utd=1,batch_size=256,online_steps=50000,updates=30000)
    execute([(f'task{t}_utd1',command(f'task{t}_utd1',t,1.)) for t in (1,2)])
    record('all_training_complete')


if __name__=='__main__':
    RUNS.mkdir(parents=True,exist_ok=True)
    lock=(RUNS/'suite.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:main()
    except BaseException as e:record('suite_failed',type=type(e).__name__,message=str(e));raise
