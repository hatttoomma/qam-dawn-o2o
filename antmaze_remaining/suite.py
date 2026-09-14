"""One-GPU queue: preflight, task2 smoke, then native/DAWN pairs for tasks 2–5."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(os.environ.get('ANTMAZE_ROOT', Path(__file__).resolve().parent))
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
lock = (OUT / 'suite.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
TASKS = (2, 3, 4, 5)

def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)

def status(**kwargs):
    completed = [task for task in TASKS if (OUT / f'task{task}/dawn/CHECKS_PASSED.json').exists()]
    write(OUT / 'suite_status.json', dict(utc=utc(), suite_pid=os.getpid(),
          tasks=list(TASKS), completed_tasks=completed, **kwargs))

def verified_done(folder, stage, smoke):
    result = json.loads((folder / 'DONE.json').read_text())
    if stage == 'native':
        assert result['status'] == 'passed'
        assert (result['offline_updates'], result['online_env_steps'], result['online_updates']) == (
            (10, 12, 8) if smoke else (500000, 50000, 45001))
    else:
        checks = json.loads((folder / 'CHECKS_PASSED.json').read_text())
        assert checks['status'] == 'passed'
        assert (result['steps'], result['updates']) == ((100, 5) if smoke else (100000, 5000))

def run(task, stage, smoke=False):
    parent = OUT / ('smoke_task2' if smoke else f'task{task}')
    folder = parent / stage
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'DONE.json').exists():
        verified_done(folder, stage, smoke)
        return
    if (folder / 'progress.json').exists():
        raise RuntimeError(f'Existing incomplete run requires inspection: {folder}')
    env = dict(os.environ, ANTMAZE_ROOT=str(ROOT), ANTMAZE_OUT=str(folder),
               ANTMAZE_NATIVE=str(parent / 'native'), ANTMAZE_TASK=str(task),
               ANTMAZE_SMOKE='1' if smoke else '0', MUJOCO_GL='egl',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', XLA_PYTHON_CLIENT_PREALLOCATE='false')
    started = utc()
    tick = time.monotonic()
    with (folder / 'stdout.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT / f'{stage}.py')],
            cwd=ROOT / 'official', env=env, stdout=log, stderr=subprocess.STDOUT)
        status(task_id=task, stage=stage, smoke=smoke, state='running',
               child_pid=process.pid, started_utc=started)
        code = process.wait()
    write(folder / 'runtime.json', dict(started_utc=started, finished_utc=utc(),
          wall_seconds=time.monotonic()-tick, returncode=code))
    if code:
        status(task_id=task, stage=stage, smoke=smoke, state='failed', returncode=code)
        raise SystemExit(code)
    verified_done(folder, stage, smoke)

try:
    adapter_manifest = json.loads((ROOT / 'adapter_hashes.json').read_text())
    for name, expected in adapter_manifest.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    env = dict(os.environ, MUJOCO_GL='egl', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    status(stage='preflight', state='running')
    with (OUT / 'preflight.log').open('a') as log:
        subprocess.run([sys.executable, '-u', str(ROOT / 'preflight.py')],
                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    run(2, 'native', smoke=True)
    run(2, 'dawn', smoke=True)
    status(stage='smoke', state='passed')
    for task in TASKS:
        run(task, 'native')
        run(task, 'dawn')
    status(stage='complete', state='passed')
except BaseException as exc:
    write(OUT / 'FAILED.json', dict(utc=utc(), type=type(exc).__name__, message=str(exc)))
    raise
