"""Run the unchanged plain-TD trainer with two batch sizes on two tasks."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'runs/td_ablation/batch_comparison'
ARMS = [(f'task{t}_b{b}', t, b) for t in (1, 2) for b in (1024, 256)]


def load(p): return json.loads(p.read_text())
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def record(event, **fields):
    value = dict(event=event, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields)
    with (RUNS / 'suite.jsonl').open('a') as f: f.write(json.dumps(value) + '\n')
    p = RUNS / 'suite_status.json.tmp'; p.write_text(json.dumps(value, indent=2) + '\n')
    p.replace(RUNS / 'suite_status.json')
    print(json.dumps(value), flush=True)


def command(name, task, batch, smoke=False):
    cmd = ['run_td_ablation.py', '--task', str(task), '--td-target', 'hard',
           '--stage', 'warm', '--dawn-batch', str(batch), '--out', str(RUNS / name)]
    if smoke:
        cmd += ['--online-steps', '60', '--warmup', '20', '--eval-episodes', '1', '--final-episodes', '1']
    return cmd


def execute(jobs):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', XLA_PYTHON_CLIENT_MEM_FRACTION='.21',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               MUJOCO_GL='egl', __EGL_VENDOR_LIBRARY_FILENAMES=str(RUNS / 'egl_vendor.json'),
               PYTHONUNBUFFERED='1', JAX_COMPILATION_CACHE_DIR=str(ROOT / 'jax_cache'))
    pending, errors = [], []
    for name, cmd in jobs:
        if (RUNS / name / 'DONE.json').exists():
            assert load(RUNS / name / 'CHECKS_PASSED.json')['status'] == 'passed'
            record('already_complete', name=name); continue
        output = (RUNS / (name + '.log')).open('a')
        p = subprocess.Popen([sys.executable, *cmd], cwd=ROOT, env=env,
                             stdout=output, stderr=subprocess.STDOUT)
        record('started', name=name, pid=p.pid, command=cmd, memory_fraction=.21)
        pending.append((name, p, output))
    while pending:
        for name, p, output in pending[:]:
            code = p.poll()
            if code is not None:
                output.close(); pending.remove((name, p, output))
                record('finished' if code == 0 else 'failed', name=name, exit_code=code)
                if code: errors.append(name)
        if pending: time.sleep(5)
    if errors: raise RuntimeError('Failed jobs: ' + ', '.join(errors))


def check_smokes():
    records = {}
    for task in (1, 2):
        folders = [RUNS / f'smoke_task{task}_b{b}' for b in (1024, 256)]
        a, b = folders
        for name in ('initial_hashes.json', 'actual_agent_config.json', 'warmup.json'):
            assert load(a / name) == load(b / name), (task, name)
        ca, cb = [load(f / 'config.json') for f in folders]
        assert {k:v for k,v in ca.items() if k not in ('out','dawn_batch')} == {
            k:v for k,v in cb.items() if k not in ('out','dawn_batch')}
        assert load(a / 'eval_000000_001.json')['records'] == load(b / 'eval_000000_001.json')['records']
        for batch, folder in zip((1024, 256), folders):
            assert load(folder / 'config.json')['dawn_batch'] == batch
            assert load(folder / 'CHECKS_PASSED.json')['status'] == 'passed'
            assert load(folder / 'backup_math.json')['status'] == 'passed'
            assert load(folder / 'DONE.json')['updates'] == 10
            assert (folder / 'probe_000060.json').exists()
            records[folder.name] = dict(updates=10, batch=batch,
                checkpoint_sha256=load(folder / 'DONE.json')['checkpoint_sha256'])
        assert records[a.name]['checkpoint_sha256'] != records[b.name]['checkpoint_sha256']
    result = dict(status='passed', tasks=[1,2], identical_initial_states_and_warmup=True,
                  only_batch_config_differs=True, target_math_passed=True, runs=records)
    (RUNS / 'CHECKS_PASSED.json').write_text(json.dumps(result, indent=2) + '\n')
    record('checks_passed', **result)


def main():
    assert load(RUNS / 'runtime_setup.json')['probe_exit_code'] == 0
    sources = load(ROOT / 'runs/td_ablation/source_manifest.json')
    for n, h in sources.items(): assert digest(ROOT / n) == h, n
    for n in ['launch_td_batch_ablation.py', 'TD_BATCH_ABLATION_PROTOCOL.md']:
        sources[n] = digest(ROOT / n)
    p = RUNS / 'source_manifest.json'
    if p.exists(): assert load(p) == sources
    else: p.write_text(json.dumps(sources, indent=2) + '\n')
    for task in (1,2):
        p = ROOT / ('runs/offline' if task == 1 else 'runs/task2/offline')
        assert digest(p / 'final.pkl') == load(p / 'DONE.json')['checkpoint_sha256']
    if not (RUNS / 'pip_freeze.txt').exists():
        with (RUNS / 'pip_freeze.txt').open('w') as f:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=f, check=True)
    if not (RUNS / 'CHECKS_PASSED.json').exists():
        execute([('smoke_' + n, command('smoke_' + n, t, b, True)) for n,t,b in ARMS])
        check_smokes()
    record('starting_formal_pairs', tasks=[1,2], td_target='hard', batches=[1024,256])
    execute([(n, command(n,t,b)) for n,t,b in ARMS])
    record('all_training_complete')


if __name__ == '__main__':
    RUNS.mkdir(parents=True, exist_ok=True)
    lock = (RUNS / 'suite.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try: main()
    except BaseException as e:
        record('suite_failed', type=type(e).__name__, message=str(e)); raise
