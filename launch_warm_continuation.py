"""Validate restoration, then run the predeclared DAWN continuation detached."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'runs'


def record(event, **fields):
    row = dict(event=event, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields)
    with (RUNS / 'warm_continuation_suite.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    p = RUNS / 'warm_continuation_suite_status.json.tmp'
    p.write_text(json.dumps(row, indent=2) + '\n')
    p.replace(RUNS / 'warm_continuation_suite_status.json')
    print(json.dumps(row), flush=True)


def run(name, target, smoke):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', XLA_PYTHON_CLIENT_MEM_FRACTION='.65',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        MUJOCO_GL='egl', PYTHONUNBUFFERED='1', JAX_COMPILATION_CACHE_DIR=str(ROOT/'jax_cache'))
    argv = [sys.executable, 'run_warm_continuation.py', '--out', str(RUNS/name),
            '--target-updates', str(target)] + (['--smoke'] if smoke else [])
    with (RUNS / (name + '.log')).open('a') as log:
        p = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        record('started', name=name, pid=p.pid, target_updates=target, command=argv)
        code = p.wait()
    record('finished' if code == 0 else 'failed', name=name, exit_code=code)
    if code:
        raise RuntimeError(name + ' failed')


def digest_tree(tree):
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, dict):
            h.update(b'dict')
            for k in sorted(x):
                h.update(k.encode())
                visit(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(type(x).__name__.encode())
            for v in x:
                visit(v)
        else:
            a = np.asarray(x)
            h.update(str((a.shape, a.dtype)).encode())
            h.update(a.tobytes())
    visit(tree)
    return h.hexdigest()


def main():
    run('smoke_warm_cont_split', 7504, True)
    with (RUNS/'smoke_warm_cont_split/latest.pkl').open('rb') as f:
        mid = pickle.load(f)
    assert int(mid['agent']['updates']) == 7504
    assert mid['scheduled_updates'] == 7505, 'Smoke must exercise pending updates at restore'
    run('smoke_warm_cont_split', 7508, True)
    run('smoke_warm_cont_direct', 7508, True)
    keys = ['agent', 'step', 'replay', 'episode', 'trace', 'ob', 'keys', 'numpy_rng', 'scheduled_updates']
    states = []
    for name in ['smoke_warm_cont_split', 'smoke_warm_cont_direct']:
        with (RUNS/name/'latest.pkl').open('rb') as f:
            state = pickle.load(f)
        states.append({key: digest_tree(state[key]) for key in keys})
    assert states[0] == states[1], {k: [s[k] for s in states] for k in keys if states[0][k] != states[1][k]}
    checks = dict(status='passed', restored_pending_updates=True,
        split_vs_uninterrupted_state_hashes=states[0], smoke_end_update=7508)
    (RUNS/'WARM_CONTINUATION_CHECKS_PASSED.json').write_text(json.dumps(checks, indent=2)+'\n')
    record('resume_checks_passed')
    run('warm_extended', 60000, False)
    record('warm_continuation_complete')


if __name__ == '__main__':
    lock = (RUNS/'warm_continuation_suite.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        main()
    except BaseException as e:
        record('warm_continuation_suite_failed', error=str(e))
        raise
