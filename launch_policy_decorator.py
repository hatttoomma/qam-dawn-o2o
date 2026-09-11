"""Validate and run one Policy Decorator pilot, preserving previous runs."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
RUNS=ROOT/'runs'
LOCK=(RUNS/'policy_decorator_suite.lock').open('w')
fcntl.flock(LOCK,fcntl.LOCK_EX|fcntl.LOCK_NB)
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='0',XLA_PYTHON_CLIENT_MEM_FRACTION='.65',
         OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl',
         PYTHONUNBUFFERED='1',JAX_COMPILATION_CACHE_DIR=str(ROOT/'jax_cache'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(event,**fields):
    value=dict(event=event,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**fields)
    with (RUNS/'policy_decorator_suite.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
    p=RUNS/'policy_decorator_suite_status.json.tmp';p.write_text(json.dumps(value,indent=2)+'\n')
    p.replace(RUNS/'policy_decorator_suite_status.json')
    print(json.dumps(value),flush=True)


def execute(name,argv):
    with (RUNS/(name+'.log')).open('a') as f:
        process=subprocess.Popen(argv,cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT)
        record('started',name=name,pid=process.pid,command=argv)
        code=process.wait()
    record('finished' if code==0 else 'failed',name=name,exit_code=code)
    if code:raise RuntimeError(f'{name} failed: {code}')


def main():
    for path in ['source_manifest.json','replay_source_manifest.json']:
        for p,h in json.loads((RUNS/path).read_text()).items():
            assert digest(ROOT/p)==h,p
    files=['run.py','dawn_agent.py','dawn_replay_agent.py','qam_replay.py',
           'policy_decorator_agent.py','run_policy_decorator.py','test_policy_decorator.py',
           'launch_policy_decorator.py','POLICY_DECORATOR_PROTOCOL.md']
    files += [str(p.relative_to(ROOT)) for p in (ROOT/'vendor/qam').rglob('*.py')]
    sources={p:digest(ROOT/p) for p in files}
    manifest=RUNS/'policy_decorator_source_manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())==sources
    else:manifest.write_text(json.dumps(sources,indent=2)+'\n')
    execute('policy_decorator_unit',[sys.executable,'-m','pytest','-q','test_policy_decorator.py'])
    execute('smoke_policy_decorator_qam_replay',[sys.executable,'run_policy_decorator.py','--stage','warm',
        '--out',str(RUNS/'smoke_policy_decorator_qam_replay'),'--online-steps','100','--warmup','20',
        '--prog-explore','60','--eval-episodes','1','--final-episodes','1'])
    smoke=RUNS/'smoke_policy_decorator_qam_replay'
    init=json.loads((smoke/'initial_hashes.json').read_text())
    ref=json.loads((RUNS/'warm_qam_replay/initial_hashes.json').read_text())
    for key in ['flow','critic','target','actor','actor_optimizer','critic_optimizer']:
        assert init[key]==ref[key],key
    assert init['alpha']==1.
    done=json.loads((smoke/'DONE.json').read_text())
    assert done['steps']==100 and done['updates']==20
    assert done['replay_stats']['replay_size']==1000100
    assert done['replay_stats']['sampled_rows']==20480
    assert done['gate_counts']['enabled_chunks']>0
    assert done['gate_counts']['learned_enabled_chunks']>0
    assert done['final_flow_hash']==init['flow']
    assert done['final_critic_hash']!=init['critic'] and done['final_actor_hash']!=init['actor']
    (RUNS/'POLICY_DECORATOR_CHECKS_PASSED.json').write_text(json.dumps(dict(status='passed',sources=sources),indent=2)+'\n')
    record('checks_passed')
    execute('policy_decorator_qam_replay',[sys.executable,'run_policy_decorator.py','--stage','warm',
        '--out',str(RUNS/'policy_decorator_qam_replay')])
    record('policy_decorator_all_training_complete')


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        record('policy_decorator_suite_failed',type=type(exc).__name__,message=str(exc))
        raise
