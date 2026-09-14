"""Serial task1 pipeline: both smoke checks, official native QAM, then DAWN."""
import fcntl
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

def status(**kw):
    p = OUT / 'suite_status.json'
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        suite_pid=os.getpid(), **kw), indent=2))
    tmp.replace(p)

def run(stage, smoke=False):
    name = ('smoke_' if smoke else '') + stage
    folder = OUT / name
    folder.mkdir(exist_ok=True)
    if (folder / 'DONE.json').exists() and (stage == 'native' or (folder / 'CHECKS_PASSED.json').exists()):
        return
    env = dict(os.environ, ANTMAZE_ROOT=str(ROOT), ANTMAZE_OUT=str(folder),
        ANTMAZE_NATIVE=str(OUT / ('smoke_native' if smoke else 'native')),
        ANTMAZE_SMOKE='1' if smoke else '0', MUJOCO_GL='egl',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', XLA_PYTHON_CLIENT_PREALLOCATE='false')
    with (folder / 'stdout.log').open('a') as log:
        process = subprocess.Popen([sys.executable,'-u',str(ROOT/f'{stage}.py')],
            cwd=ROOT/'official', env=env, stdout=log, stderr=subprocess.STDOUT)
        status(stage=name, state='running', child_pid=process.pid)
        code = process.wait()
    if code:
        status(stage=name, state='failed', returncode=code)
        raise SystemExit(code)
    assert (folder/'DONE.json').exists()
    if stage == 'dawn': assert (folder/'CHECKS_PASSED.json').exists()

run('native', True)
run('dawn', True)
status(stage='smoke',state='passed')
run('native')
run('dawn')
status(stage='complete',state='passed')
