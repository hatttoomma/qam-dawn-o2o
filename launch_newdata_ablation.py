"""Verify the data-only intervention, then execute the two-task paired suite."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'runs/newdata_ablation'
ARMS = [(f'task{task}_{kind}', task, kind) for task in (1, 2) for kind in ('all', 'offline')]


def load(p): return json.loads(p.read_text())
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def record(event, **fields):
    value = dict(event=event, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields)
    with (RUNS / 'suite.jsonl').open('a') as f: f.write(json.dumps(value) + '\n')
    p = RUNS / 'suite_status.json.tmp';p.write_text(json.dumps(value, indent=2) + '\n')
    p.replace(RUNS / 'suite_status.json');print(json.dumps(value), flush=True)


def execute(jobs):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', XLA_PYTHON_CLIENT_MEM_FRACTION='.21',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MUJOCO_GL='egl',
        PYTHONUNBUFFERED='1', JAX_COMPILATION_CACHE_DIR=str(ROOT / 'jax_cache'))
    pending, errors = [], []
    for name, cmd in jobs:
        if name != 'report' and (RUNS / name / 'DONE.json').exists():
            if name.startswith(('smoke_task', 'task')): assert (RUNS / name / 'DATA_AUDIT_PASSED.json').exists()
            record('already_complete', name=name);continue
        out = (RUNS / (name + '.log')).open('a')
        p = subprocess.Popen([sys.executable, *cmd], cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
        record('started', name=name, pid=p.pid, command=cmd, memory_fraction=.21)
        pending.append((name, p, out))
    while pending:
        for name, p, out in pending[:]:
            code = p.poll()
            if code is not None:
                out.close();pending.remove((name, p, out))
                record('finished' if code == 0 else 'failed', name=name, exit_code=code)
                if code: errors.append(name)
        if pending: time.sleep(5)
    if errors: raise RuntimeError('Failed jobs: ' + ', '.join(errors))


def command(name, task, kind, smoke=False):
    cmd = ['run_newdata_ablation.py', '--task', str(task), '--train-data', kind,
           '--stage', 'native', '--out', str(RUNS / name)]
    if smoke: cmd += ['--online-steps', '60', '--native-start', '10', '--eval-episodes', '1', '--final-episodes', '1']
    return cmd


def check_smokes():
    for task in (1, 2):
        initial = None
        for kind, mode in [('all', 'native'), ('offline', 'pure_offline')]:
            d = RUNS / f'smoke_task{task}_{kind}';ref = RUNS / f'reference_task{task}_{mode}'
            assert load(d / 'DONE.json')['checkpoint_sha256'] == load(ref / 'DONE.json')['checkpoint_sha256']
            assert load(d / 'DONE.json')['updates'] == 51
            for filename in ['eval_000000_001.json', 'eval_000060_001.json']:
                assert load(d / filename)['records'] == load(ref / filename)['records']
            if initial is None: initial = load(d / 'initial_state.json')
            else: assert initial == load(d / 'initial_state.json')
            a = load(d / 'DATA_AUDIT_PASSED.json');assert a['status'] == 'passed'
            if kind == 'offline': assert a['sequences_touching_online'] == 0
            assert a['update_batches'] == 51 and a['collected_transitions'] == 60
    result = dict(status='passed', tasks=[1, 2], original_native_checkpoint_parity=True,
        zero_interaction_offline_checkpoint_parity=True, identical_initial_agents_and_optimizers=True,
        original_sampler_preserved=True, offline_new_data_leakage=0, concurrent_smokes=4,
        smoke_collection_steps=60, smoke_updates=51)
    (RUNS / 'CHECKS_PASSED.json').write_text(json.dumps(result, indent=2) + '\n')
    record('checks_passed', **result)


def main():
    historical = load(ROOT / 'runs/scale_extremes/source_manifest.json')
    for name, h in historical.items(): assert digest(ROOT / name) == h, name
    sources = dict(historical)
    for name in ['run_newdata_ablation.py', 'smoke_newdata_reference.py', 'launch_newdata_ablation.py', 'NEWDATA_ABLATION_PROTOCOL.md']:
        sources[name] = digest(ROOT / name)
    p = RUNS / 'source_manifest.json'
    if p.exists(): assert load(p) == sources
    else: p.write_text(json.dumps(sources, indent=2) + '\n')
    for task in (1, 2):
        root = ROOT / ('runs' if task == 1 else 'runs/task2')
        assert digest(root / 'offline/final.pkl') == load(root / 'offline/DONE.json')['checkpoint_sha256']
    if not (RUNS / 'pip_freeze.txt').exists():
        with (RUNS / 'pip_freeze.txt').open('w') as f: subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=f, check=True)
    if not (RUNS / 'CHECKS_PASSED.json').exists():
        jobs = []
        for task in (1, 2):
            for mode in ('native', 'pure_offline'):
                name = f'reference_task{task}_{mode}'
                jobs.append((name, ['smoke_newdata_reference.py', '--task', str(task), '--mode', mode,
                    '--stage', 'native', '--out', str(RUNS / name), '--online-steps', '60',
                    '--native-start', '10', '--eval-episodes', '1', '--final-episodes', '1']))
        execute(jobs)
        execute([('smoke_' + name, command('smoke_' + name, task, kind, True)) for name, task, kind in ARMS])
        check_smokes()
    record('starting_formal_pairs')
    execute([(name, command(name, task, kind)) for name, task, kind in ARMS])
    record('all_training_complete')
    if (ROOT / 'export_newdata_ablation.py').exists():
        execute([('report', ['export_newdata_ablation.py'])]);record('newdata_ablation_suite_complete')
    else: record('awaiting_report')


if __name__ == '__main__':
    RUNS.mkdir(parents=True, exist_ok=True)
    lock = (RUNS / 'suite.lock').open('w');fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try: main()
    except BaseException as e:
        record('suite_failed', type=type(e).__name__, message=str(e));raise
