"""One-GPU queue: resume each50k training state, continue10k with online-only replay."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(os.environ.get('ANTMAZE_ROOT',Path(__file__).resolve().parent))
OUT=ROOT/'results'
OUT.mkdir(exist_ok=True)
lock=(OUT/'suite.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
INPUTS=json.loads((ROOT/'inputs.json').read_text())

def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())

def write(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False))
    temp.replace(path)

def status(**kwargs):
    completed=[task for task in range(1,6) if (OUT/f'task{task}/dawn/CHECKS_PASSED.json').exists()]
    write(OUT/'suite_status.json',dict(utc=utc(),suite_pid=os.getpid(),tasks=list(range(1,6)),
          completed_tasks=completed,**kwargs))

def verify(folder,smoke):
    done=json.loads((folder/'DONE.json').read_text())
    checks=json.loads((folder/'CHECKS_PASSED.json').read_text())
    assert checks['status'] == 'passed' and checks['online_only_replay_verified']
    assert (done['steps'],done['updates']) == ((50008,2502) if smoke else (60000,5000))
    assert checks['evaluation_steps_verified'] == ([50004,50008] if smoke else [55000,60000])

def run(task,smoke=False):
    folder=OUT/('smoke_task1' if smoke else f'task{task}')/'dawn'
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'CHECKS_PASSED.json').exists():
        verify(folder,smoke)
        return
    if (folder/'progress.json').exists() or (folder/'FAILED.json').exists():
        raise RuntimeError(f'Existing incomplete run requires inspection: {folder}')
    env=dict(os.environ,ANTMAZE_ROOT=str(ROOT),ANTMAZE_OUT=str(folder),
             ANTMAZE_NATIVE=INPUTS[str(task)]['native'],ANTMAZE_TASK=str(task),
             ANTMAZE_SMOKE='1' if smoke else '0',MUJOCO_GL='egl',
             OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',XLA_PYTHON_CLIENT_PREALLOCATE='false')
    started,tick=utc(),time.monotonic()
    with (folder/'stdout.log').open('a') as log:
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/'dawn.py')],cwd=ROOT/'official',
                                 env=env,stdout=log,stderr=subprocess.STDOUT)
        status(task_id=task,stage='dawn',smoke=smoke,state='running',child_pid=process.pid,started_utc=started)
        code=process.wait()
    write(folder/'runtime.json',dict(started_utc=started,finished_utc=utc(),
          wall_seconds=time.monotonic()-tick,returncode=code))
    if code:
        status(task_id=task,stage='dawn',smoke=smoke,state='failed',returncode=code)
        raise RuntimeError(f'Task {task} exited {code}; smoke={smoke}')
    verify(folder,smoke)

try:
    for name,expected in json.loads((ROOT/'adapter_hashes.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected,name
    status(stage='preflight',state='running')
    env=dict(os.environ,MUJOCO_GL='egl',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',XLA_PYTHON_CLIENT_PREALLOCATE='false')
    with (OUT/'preflight.log').open('a') as log:
        subprocess.run([sys.executable,'-u',str(ROOT/'preflight.py')],cwd=ROOT,env=env,
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    run(1,smoke=True)
    for task in range(1,6):
        run(task)
    status(stage='complete',state='passed')
except BaseException as exc:
    write(OUT/'FAILED.json',dict(utc=utc(),type=type(exc).__name__,message=str(exc)))
    raise
