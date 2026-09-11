"""Validate and launch paired Task5 500k-to-1M continuations without changing historical runs."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import launch_task5_long_critic as scheduler

ROOT=Path(__file__).resolve().parent
RUNS=ROOT/'runs/task5_continue_1m_20260910'
scheduler.RUNS=RUNS


def command(name,arm,smoke=False,pause=False):
    args=['run_task5_continue_1m.py','--stage',arm,'--out',str(RUNS/name),
          '--dawn-batch','256','--online-steps','500120' if smoke else '1000000',
          '--eval-episodes','2' if smoke else '100','--final-episodes','2' if smoke else '100']
    if smoke:args+=['--smoke']
    if pause:args+=['--stop-after-checkpoint','500040']
    return name,args


def check_origins():
    paths=[RUNS/arm/'origin_state_verified.json' for arm in ('warm','random')]
    if not all(p.exists() for p in paths):return
    origins=[scheduler.load(p) for p in paths]
    for arm,record in zip(('warm','random'),origins):
        assert record['arm']==arm and record['origin_steps']==500000 and record['origin_updates']==120000
        assert record['all_saved_fields_preserved'] and record['optimizer_and_alpha_states_preserved']
    scheduler.write(RUNS/'FORMAL_CONTINUATION_CHECKS_PASSED.json',dict(status='passed',
        both_arms_resume_corresponding_500k_checkpoint=True,new_warmup_steps=0,
        original_steps=500000,target_steps=1000000,original_updates=120000,target_updates=245000,
        sources={arm:record['source_resume_sha256'] for arm,record in zip(('warm','random'),origins)}))


def main():
    names=['run_task5_continue_1m.py','verify_task5_continue_1m.py','launch_task5_continue_1m.py',
           'TASK5_CONTINUE_1M_PROTOCOL.md','run.py','actor_input_agent.py','run_task5_long_critic.py',
           'launch_task5_long_critic.py','run_actor_input_ablation.py']
    manifest={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    path=RUNS/'source_manifest.json'
    if path.exists():assert scheduler.load(path)==manifest
    else:scheduler.write(path,manifest)
    assert shutil.disk_usage(ROOT).free>3*1024**3,'Insufficient disk space for validation and continuation'
    if not (RUNS/'PREFLIGHT_PASSED.json').exists():
        scheduler.execute([command('smoke_warm','warm',True),command('smoke_random','random',True)])
        if not (RUNS/'smoke_resume/SMOKE_PAUSED.json').exists():
            scheduler.execute([command('smoke_resume','warm',True,True)])
        scheduler.execute([command('smoke_resume','warm',True)])
        env=dict(os.environ,JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        subprocess.run([sys.executable,'verify_task5_continue_1m.py'],cwd=ROOT,env=env,check=True)
        scheduler.record('preflight_passed',checks=scheduler.load(RUNS/'PREFLIGHT_PASSED.json'))
        for folder in ('smoke_warm','smoke_random','smoke_resume'):
            for file in (RUNS/folder).glob('*.pkl'):file.unlink()
    scheduler.pair_check=check_origins
    scheduler.record('starting_1m_continuation',task=5,from_steps=500000,to_steps=1000000,
                     target_updates=245000,new_warmup_steps=0)
    scheduler.execute([command('warm','warm'),command('random','random')],formal=True)
    check_origins()
    scheduler.write(RUNS/'results.json',{arm:[scheduler.load(p) for p in sorted((RUNS/arm).glob('paired_*.json'))]
                                     for arm in ('warm','random')})
    scheduler.record('all_training_complete',task=5,steps_per_arm=1000000,updates_per_arm=245000)


if __name__=='__main__':
    RUNS.mkdir(parents=True,exist_ok=True)
    lock=(RUNS/'suite.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:main()
    except BaseException as exc:
        scheduler.record('suite_failed',type=type(exc).__name__,message=str(exc));raise
