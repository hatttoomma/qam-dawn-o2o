import fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
lock=(OUT/'suite.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def write(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def status(**kw):
 completed=[f'task{t}/{m}' for t in range(1,6) for m in ['native','dawn'] if (OUT/f'task{t}/{m}/CHECKS_PASSED.json').exists()]
 write(OUT/'suite_status.json',dict(utc=utc(),pid=os.getpid(),completed=completed,**kw))
def run(task,method,smoke=False):
 folder=OUT/(f'smoke_{method}' if smoke else f'task{task}')/method;folder.mkdir(parents=True,exist_ok=True)
 assert not (folder/'latest.pkl').exists() and not (folder/'DONE.json').exists()
 env=dict(os.environ,TASK=str(task),METHOD=method,SMOKE='1' if smoke else '0',OUT=str(folder),
  MUJOCO_GL='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',
  XLA_FLAGS='--xla_gpu_enable_command_buffer=',__EGL_VENDOR_LIBRARY_FILENAMES=str(ROOT/'egl_vendor.json'))
 start=utc();tick=time.monotonic()
 with (folder/'stdout.log').open('w') as log:
  p=subprocess.Popen([sys.executable,'-u',str(ROOT/'run_online.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  status(state='running',task=task,method=method,smoke=smoke,child_pid=p.pid);code=p.wait()
 write(folder/'runtime.json',dict(returncode=code,started_utc=start,finished_utc=utc(),wall_seconds=time.monotonic()-tick))
 assert code==0,f'task{task}/{method} failed, smoke={smoke}'
 checks=json.loads((folder/'CHECKS_PASSED.json').read_text());assert checks['status']=='passed'
 if not smoke:assert (checks['steps'],checks['updates'])==(100000,95001 if method=='native' else 5625)
try:
 for name,digest in json.loads((ROOT/'adapter_hashes.json').read_text()).items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
 run(1,'native',True);run(1,'dawn',True)
 for task in range(1,6):
  run(task,'native');run(task,'dawn')
 status(state='complete')
except BaseException as e:write(OUT/'FAILED.json',dict(utc=utc(),type=type(e).__name__,message=str(e)));raise
