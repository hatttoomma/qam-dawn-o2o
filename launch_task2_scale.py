"""Check the scale-only adapter, then run the three authorized task2 scales."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'runs/task2/scale_sweep'
SCALES = [('scale005', .05), ('scale020', .2), ('scale030', .3)]


def load(p): return json.loads(p.read_text())
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def record(event, **fields):
    value = dict(event=event, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields)
    with (RUNS / 'suite.jsonl').open('a') as f: f.write(json.dumps(value) + '\n')
    p = RUNS / 'suite_status.json.tmp'
    p.write_text(json.dumps(value, indent=2) + '\n')
    p.replace(RUNS / 'suite_status.json')
    print(json.dumps(value), flush=True)


def command(name, scale=None, smoke=False):
    args = ['run_task2.py' if scale is None else 'run_task2_scale.py',
            '--stage', 'warm', '--out', str(RUNS / name)]
    if scale is not None: args += ['--residual-scale', str(scale)]
    if smoke: args += ['--online-steps', '60', '--warmup', '20',
                       '--eval-episodes', '1', '--final-episodes', '1']
    return args


def execute_many(jobs):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', XLA_PYTHON_CLIENT_MEM_FRACTION='.28',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MUJOCO_GL='egl',
        PYTHONUNBUFFERED='1', JAX_COMPILATION_CACHE_DIR=str(ROOT / 'jax_cache'))
    pending = []
    for name, args in jobs:
        log = (RUNS / (name + '.log')).open('a')
        p = subprocess.Popen([sys.executable, *args], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        record('started', name=name, pid=p.pid, command=args, gpu=0, memory_fraction=.28)
        pending.append((name, p, log))
    failures = []
    while pending:
        for item in pending[:]:
            name, p, log = item
            code = p.poll()
            if code is not None:
                log.close()
                record('finished' if code == 0 else 'failed', name=name, exit_code=code)
                pending.remove(item)
                if code: failures.append(name)
        if pending: time.sleep(5)
    if failures: raise RuntimeError('Failed jobs: ' + ', '.join(failures))


def check_smokes():
    names = ['smoke_reference', 'smoke_scale010', *['smoke_' + n for n, _ in SCALES]]
    ref = load(RUNS / 'smoke_reference/DONE.json')
    same = load(RUNS / 'smoke_scale010/DONE.json')
    assert ref['checkpoint_sha256'] == same['checkpoint_sha256'], 'scale=.1 adapter changed training'
    initial = load(RUNS / 'smoke_reference/initial_hashes.json')
    warmup = load(RUNS / 'smoke_reference/warmup.json')
    offline = load(ROOT / 'runs/task2/offline/DONE.json')
    assert initial['critic'] == offline['q_hash'] and initial['target'] == offline['target_q_hash']
    for name in names:
        done = load(RUNS / name / 'DONE.json')
        assert done['steps'] == 60 and done['updates'] == 10
        assert load(RUNS / name / 'initial_hashes.json') == initial
        assert load(RUNS / name / 'warmup.json') == warmup
        assert done['final_flow_hash'] == offline['flow_hash']
        assert done['final_critic_hash'] != initial['critic'] and done['final_actor_hash'] != initial['actor']
        for e in ['eval_000000_001.json']:
            assert load(RUNS / name / e)['records'] == load(RUNS / 'smoke_reference' / e)['records']
    for e in ['eval_000060_001.json', 'eval_000060_001_mean_residual.json']:
        assert load(RUNS / 'smoke_reference' / e)['records'] == load(RUNS / 'smoke_scale010' / e)['records']
    for name, scale in [('smoke_scale010', .1), *[('smoke_' + n, s) for n, s in SCALES]]:
        actual = load(RUNS / name / 'actual_agent_config.json')
        assert actual['residual_scale'] == scale and actual['action_dim'] == 25 and actual['tau'] == .01
    result = dict(status='passed', scale010_matches_original_checkpoint=True,
        all_initial_states_and_warmup_identical=True, inherited_task2_Q_and_target=True,
        frozen_flow_unchanged=True, online_steps=60, updates=10,
        concurrent_scales=[s for _, s in SCALES])
    (RUNS / 'CHECKS_PASSED.json').write_text(json.dumps(result, indent=2) + '\n')
    record('checks_passed', **result)


def main():
    historical = load(ROOT / 'runs/task2/source_manifest.json')
    for n, h in historical.items(): assert digest(ROOT / n) == h, n
    sources = dict(historical)
    for n in ['run_task2_scale.py', 'launch_task2_scale.py', 'TASK2_SCALE_PROTOCOL.md']:
        sources[n] = digest(ROOT / n)
    p = RUNS / 'source_manifest.json'
    if p.exists(): assert load(p) == sources
    else: p.write_text(json.dumps(sources, indent=2) + '\n')
    assert load(ROOT / 'runs/task2/TASK_AUDIT_PASSED.json')['status'] == 'passed'
    offline = ROOT / 'runs/task2/offline'
    assert digest(offline / 'final.pkl') == load(offline / 'DONE.json')['checkpoint_sha256']
    if not (RUNS / 'pip_freeze.txt').exists():
        with (RUNS / 'pip_freeze.txt').open('w') as f:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=f, check=True)
    if not (RUNS / 'CHECKS_PASSED.json').exists():
        for name, scale in [('smoke_reference', None), ('smoke_scale010', .1)]:
            execute_many([(name, command(name, scale, True))])
        execute_many([('smoke_' + n, command('smoke_' + n, s, True)) for n, s in SCALES])
        check_smokes()
    record('starting_formal_scales', scales=[s for _, s in SCALES])
    execute_many([(n, command(n, s)) for n, s in SCALES])
    record('all_training_complete')
    if not (ROOT / 'export_task2_scale.py').exists():
        record('awaiting_report_script')
        return
    execute_many([('report', ['export_task2_scale.py'])])
    record('task2_scale_suite_complete')


if __name__ == '__main__':
    RUNS.mkdir(parents=True, exist_ok=True)
    lock = (RUNS / 'suite.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try: main()
    except BaseException as e:
        record('suite_failed', type=type(e).__name__, message=str(e))
        raise
