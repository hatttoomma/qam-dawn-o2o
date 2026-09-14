import fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
lock=(OUT/'suite.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
gpu=open('/tmp/qam_scale100k_gpu.lock','w');fcntl.flock(gpu,fcntl.LOCK_EX)
def write(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def status(**kw):
 write(OUT/'suite_status.json',dict(utc=utc(),pid=os.getpid(),completed=[t for t in range(1,6) if (OUT/f'task{t}/dawn/CHECKS_PASSED.json').exists()],**kw))
def run(task,smoke=False):
 folder=OUT/('smoke' if smoke else f'task{task}')/'dawn';folder.mkdir(parents=True,exist_ok=True)
 assert not (folder/'DONE.json').exists() and not (folder/'latest.pkl').exists()
 tick=time.monotonic();start=utc()
 env=dict(os.environ,TASK=str(task),OUT=str(folder),SMOKE='1' if smoke else '0',MUJOCO_GL='egl',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',XLA_PYTHON_CLIENT_PREALLOCATE='false')
 with (folder/'stdout.log').open('w') as log:
  p=subprocess.Popen([sys.executable,'-u',str(ROOT/'dawn_resume.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  status(state='running',task=task,smoke=smoke,child_pid=p.pid);code=p.wait()
 write(folder/'runtime.json',dict(returncode=code,started_utc=start,finished_utc=utc(),wall_seconds=time.monotonic()-tick))
 assert code==0,f'Task{task} smoke={smoke} failed'
 checks=json.loads((folder/'CHECKS_PASSED.json').read_text());assert checks['status']=='passed'
 if not smoke:assert (checks['steps'],checks['updates'])==(100000,5625)
try:
 for name,digest in json.loads((ROOT/'adapter_hashes.json').read_text()).items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
 run(1,True)
 for task in range(1,6):run(task)
 status(state='complete')
except BaseException as e:
 write(OUT/'FAILED.json',dict(utc=utc(),type=type(e).__name__,message=str(e)));raise
