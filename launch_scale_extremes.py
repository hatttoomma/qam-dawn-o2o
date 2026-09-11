"""Audit the two-task scale adapter and run the four requested extreme scales."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'runs/scale_extremes'
ARMS = [(f'task{task}_scale{label}', task, scale)
        for task in (1, 2) for label, scale in [('001', .01), ('100', 1.)]]


def load(p): return json.loads(p.read_text())
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def task_root(task): return ROOT / ('runs' if task == 1 else 'runs/task2')


def record(event, **fields):
    v = dict(event=event, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields)
    with (RUNS / 'suite.jsonl').open('a') as f: f.write(json.dumps(v) + '\n')
    p = RUNS / 'suite_status.json.tmp';p.write_text(json.dumps(v, indent=2) + '\n')
    p.replace(RUNS / 'suite_status.json')
    print(json.dumps(v), flush=True)


def command(name, task=1, scale=None, smoke=False):
    args = ['run.py' if scale is None else 'run_scale_extremes.py',
            '--stage', 'warm', '--out', str(RUNS / name)]
    if scale is not None: args += ['--task', str(task), '--residual-scale', str(scale)]
    if smoke: args += ['--online-steps', '60', '--warmup', '20', '--eval-episodes', '1', '--final-episodes', '1']
    return args


def execute_many(jobs):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', XLA_PYTHON_CLIENT_MEM_FRACTION='.21',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MUJOCO_GL='egl',
        PYTHONUNBUFFERED='1', JAX_COMPILATION_CACHE_DIR=str(ROOT / 'jax_cache'))
    pending, failures = [], []
    for name, args in jobs:
        log = (RUNS / (name + '.log')).open('a')
        p = subprocess.Popen([sys.executable, *args], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        record('started', name=name, pid=p.pid, command=args, gpu=0, memory_fraction=.21)
        pending.append((name, p, log))
    while pending:
        for item in pending[:]:
            name, p, log = item;code = p.poll()
            if code is not None:
                log.close();pending.remove(item)
                record('finished' if code == 0 else 'failed', name=name, exit_code=code)
                if code: failures.append(name)
        if pending: time.sleep(5)
    if failures: raise RuntimeError('Failed jobs: ' + ', '.join(failures))


def smoke_checks():
    references = {1: RUNS / 'smoke_reference_task1', 2: ROOT / 'runs/task2/scale_sweep/smoke_reference'}
    for task in (1, 2):
        ref = references[task];wrapped = RUNS / f'smoke_task{task}_scale010'
        assert load(ref / 'DONE.json')['checkpoint_sha256'] == load(wrapped / 'DONE.json')['checkpoint_sha256']
        for e in ['eval_000000_001.json', 'eval_000060_001.json', 'eval_000060_001_mean_residual.json']:
            assert load(ref / e)['records'] == load(wrapped / e)['records']
        init = load(ref / 'initial_hashes.json');warmup = load(ref / 'warmup.json')
        offline = load(task_root(task) / 'offline/DONE.json')
        assert init['critic'] == offline['q_hash'] and init['target'] == offline['target_q_hash']
        for name, scale in [(f'smoke_task{task}_scale010', .1),
                            *[('smoke_' + n, s) for n, t, s in ARMS if t == task]]:
            d = RUNS / name;done = load(d / 'DONE.json');actual = load(d / 'actual_agent_config.json')
            assert done['steps'] == 60 and done['updates'] == 10
            assert load(d / 'initial_hashes.json') == init and load(d / 'warmup.json') == warmup
            assert done['final_flow_hash'] == offline['flow_hash']
            assert done['final_critic_hash'] != init['critic'] and done['final_actor_hash'] != init['actor']
            assert actual['task'] == task and actual['residual_scale'] == scale and actual['action_dim'] == 25
            assert load(d / 'task_manifest.json')['reward_task_id'] == task
    result = dict(status='passed', scale010_checkpoint_parity_tasks=[1, 2],
        within_task_initial_and_warmup_states_identical=True, effective_scales=[.01, 1.],
        inherited_correct_task_Q_and_target=True, frozen_flow_unchanged=True,
        smoke_steps=60, smoke_updates=10, concurrent_smokes=4)
    (RUNS / 'CHECKS_PASSED.json').write_text(json.dumps(result, indent=2) + '\n')
    record('checks_passed', **result)


def main():
    historical = load(ROOT / 'runs/task2/scale_sweep/source_manifest.json')
    for n, h in historical.items(): assert digest(ROOT / n) == h, n
    sources = dict(historical)
    for n in ['run_scale_extremes.py', 'launch_scale_extremes.py', 'SCALE_EXTREMES_PROTOCOL.md']:
        sources[n] = digest(ROOT / n)
    p = RUNS / 'source_manifest.json'
    if p.exists(): assert load(p) == sources
    else: p.write_text(json.dumps(sources, indent=2) + '\n')
    assert load(ROOT / 'runs/task2/TASK_AUDIT_PASSED.json')['status'] == 'passed'
    for task in (1, 2):
        offline = task_root(task) / 'offline'
        assert digest(offline / 'final.pkl') == load(offline / 'DONE.json')['checkpoint_sha256']
    if not (RUNS / 'pip_freeze.txt').exists():
        with (RUNS / 'pip_freeze.txt').open('w') as f:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=f, check=True)
    if not (RUNS / 'CHECKS_PASSED.json').exists():
        execute_many([('smoke_reference_task1', command('smoke_reference_task1', smoke=True)),
            ('smoke_task1_scale010', command('smoke_task1_scale010', 1, .1, True)),
            ('smoke_task2_scale010', command('smoke_task2_scale010', 2, .1, True))])
        execute_many([('smoke_' + n, command('smoke_' + n, t, s, True)) for n, t, s in ARMS])
        smoke_checks()
    record('starting_formal_extremes', tasks=[1, 2], scales=[.01, 1.])
    execute_many([(n, command(n, t, s)) for n, t, s in ARMS])
    record('all_training_complete')
    if not (ROOT / 'export_scale_extremes.py').exists():
        record('awaiting_report_script');return
    execute_many([('report', ['export_scale_extremes.py'])])
    record('scale_extremes_suite_complete')


if __name__ == '__main__':
    RUNS.mkdir(parents=True, exist_ok=True)
    lock = (RUNS / 'suite.lock').open('w');fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try: main()
    except BaseException as e:
        record('suite_failed', type=type(e).__name__, message=str(e));raise
