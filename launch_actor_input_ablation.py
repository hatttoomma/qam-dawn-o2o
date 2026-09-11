"""Run matched-width masked/base-action actor pairs without changing the training schedule."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys

import launch_td_batch_ablation as base

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'runs/actor_input_ablation'
base.RUNS = RUNS
load, digest, record, execute = base.load, base.digest, base.record, base.execute


def command(name, task, mode, smoke=False):
    cmd = ['run_actor_input_ablation.py', '--task', str(task), '--td-target', 'hard',
           '--actor-input', mode, '--stage', 'warm', '--dawn-batch', '256', '--out', str(RUNS/name)]
    if smoke:
        cmd += ['--online-steps', '60', '--warmup', '20', '--eval-episodes', '1', '--final-episodes', '1']
    return cmd


def check_smokes():
    tasks = []
    for task in (1, 2):
        folders = [RUNS/f'smoke_task{task}_{mode}' for mode in ('masked', 'base_action')]
        a, b = folders
        assert load(a/'initial_hashes.json') == load(b/'initial_hashes.json')
        ac, bc = [load(f/'actual_agent_config.json') for f in folders]
        assert {k:v for k,v in ac.items() if k!='actor_input'} == {k:v for k,v in bc.items() if k!='actor_input'}
        ca, cb = [load(f/'config.json') for f in folders]
        assert {k:v for k,v in ca.items() if k not in ('out','actor_input')} == {
            k:v for k,v in cb.items() if k not in ('out','actor_input')}
        wa, wb = [load(f/'warmup.json') for f in folders]
        assert {k:v for k,v in wa.items() if k!='replay_hash'} == {k:v for k,v in wb.items() if k!='replay_hash'}
        for f in folders:
            assert load(f/'CHECKS_PASSED.json')['status']=='passed'
            assert load(f/'actor_input_checks.json')['actor_parameter_count']==160562
            assert load(f/'DONE.json')['updates']==10
            assert (f/'probe_000060.json').exists()
        tasks.append(dict(task=task, identical_initial_states=True, only_actor_input_config_differs=True,
                          ten_updates_each=True, warmup_steps=wa['steps']))
    result=dict(status='passed', tasks=tasks)
    (RUNS/'CHECKS_PASSED.json').write_text(json.dumps(result,indent=2)+'\n')
    record('checks_passed', **result)


def main():
    assert load(RUNS/'runtime_setup.json')['probe_exit_code']==0
    sources=load(ROOT/'runs/td_ablation/batch_comparison/source_manifest.json')
    for n,h in sources.items(): assert digest(ROOT/n)==h,n
    for n in ['actor_input_agent.py','actor_input_agent_patch.json','actor_input_diagnostics.py',
              'run_actor_input_ablation.py','launch_actor_input_ablation.py','ACTOR_INPUT_ABLATION_PROTOCOL.md']:
        sources[n]=digest(ROOT/n)
    p=RUNS/'source_manifest.json'
    if p.exists(): assert load(p)==sources
    else: p.write_text(json.dumps(sources,indent=2)+'\n')
    patch=load(ROOT/'actor_input_agent_patch.json')
    assert digest(ROOT/patch['original'])==patch['original_sha256']
    assert digest(ROOT/patch['adapted'])==patch['adapted_sha256']
    for task in (1,2):
        offline=ROOT/('runs/offline' if task==1 else 'runs/task2/offline')
        assert digest(offline/'final.pkl')==load(offline/'DONE.json')['checkpoint_sha256']
    if not (RUNS/'pip_freeze.txt').exists():
        with (RUNS/'pip_freeze.txt').open('w') as f:
            subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
    arms=[(f'task{t}_{m}',t,m) for t in (1,2) for m in ('masked','base_action')]
    if not (RUNS/'CHECKS_PASSED.json').exists():
        execute([('smoke_'+n,command('smoke_'+n,t,m,True)) for n,t,m in arms])
        check_smokes()
    record('starting_formal_pairs', tasks=[1,2], modes=['masked','base_action'],
           batch_size=256, utd=.25, online_steps=50000, updates=7500)
    execute([(n,command(n,t,m)) for n,t,m in arms])
    record('all_training_complete')


if __name__=='__main__':
    RUNS.mkdir(parents=True,exist_ok=True)
    lock=(RUNS/'suite.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try: main()
    except BaseException as e:
        record('suite_failed',type=type(e).__name__,message=str(e));raise
