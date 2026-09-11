"""Detached, guarded Task5 critic-initialization experiment suite."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RUNS = ROOT/'runs/task5_long_critic_shared_20260910'


def load(p):
    return json.loads(p.read_text())


def write(p, value):
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(p)


def record(event, **fields):
    value = dict(event=event, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields)
    with (RUNS/'suite.jsonl').open('a') as f:
        f.write(json.dumps(value)+'\n')
    write(RUNS/'suite_status.json', value)
    print(json.dumps(value), flush=True)


def command(name, stage, smoke=False, pause=False, collect=False):
    args = ['run_task5_long_critic.py', '--stage', stage, '--out', str(RUNS/name),
            '--dawn-batch', '256', '--online-steps', '120' if smoke else '500000',
            '--warmup', '20' if smoke else '20000',
            '--eval-episodes', '2' if smoke else '100', '--final-episodes', '2' if smoke else '100']
    if smoke:
        args += ['--smoke']
    if pause:
        args += ['--stop-after-checkpoint', '40']
    if collect:
        args += ['--collect-warmup']
    else:
        args += ['--shared-warmup', str(RUNS/('smoke_collector' if smoke else 'shared_warmup'))]
    return name, args


def pair_check():
    folders = [RUNS/s for s in ('warm', 'random')]
    names = ('actual_agent_config.json', 'warmup.json', 'eval_000000_100.json')
    if not all((p/n).exists() for p in folders for n in names):
        return
    configs = [load(p/names[0]) for p in folders]
    prefixes = [load(p/names[1]) for p in folders]
    baselines = [load(p/names[2]) for p in folders]
    assert configs[0]['non_Q_initial_hash'] == configs[1]['non_Q_initial_hash']
    assert prefixes[0]['replay_hash'] == prefixes[1]['replay_hash']
    assert prefixes[0]['steps'] == prefixes[1]['steps']
    assert 20000 <= prefixes[0]['steps'] < 20005
    historical = load(ROOT/'runs/cube5/task5_warm/warmup.json')
    assert prefixes[0]['steps'] == historical['steps']
    shared = [load(p/'shared_initialization.json') for p in folders]
    assert shared[0]['collector_checkpoint_sha256'] == shared[1]['collector_checkpoint_sha256']
    assert baselines[0]['records'] == baselines[1]['records']
    write(RUNS/'FORMAL_PAIR_CHECKS_PASSED.json', dict(status='passed',
        identical_non_Q_initialization=True, identical_20k_warmup_replay=True,
        requested_warmup_steps=20000, actual_warmup_steps=prefixes[0]['steps'],
        shared_collector_checkpoint_sha256=shared[0]['collector_checkpoint_sha256'],
        warmup_matches_historical_task5=prefixes[0]['replay_hash'] == historical['replay_hash'],
        warmup_replay_hash=prefixes[0]['replay_hash'], identical_100_episode_baseline=True,
        baseline_success=baselines[0]['success'], baseline_return=baselines[0]['return_mean']))


def execute(jobs, formal=False):
    assert len(jobs) <= 2
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', XLA_PYTHON_CLIENT_MEM_FRACTION='.38',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MUJOCO_GL='egl',
               __EGL_VENDOR_LIBRARY_FILENAMES=str(ROOT/'runs/cube5/egl_vendor.json'),
               PYTHONUNBUFFERED='1', JAX_COMPILATION_CACHE_DIR=str(ROOT/'jax_cache'))
    pending, errors = [], []
    for name, cmd in jobs:
        if (RUNS/name/'DONE.json').exists():
            assert load(RUNS/name/'CHECKS_PASSED.json')['status'] == 'passed'
            continue
        handle = (RUNS/(name+'.log')).open('a')
        process = subprocess.Popen([sys.executable, *cmd], cwd=ROOT, env=env,
                                   stdout=handle, stderr=subprocess.STDOUT)
        record('started', name=name, pid=process.pid, command=cmd, formal=formal)
        pending.append((name, process, handle))
    try:
        while pending:
            if formal:
                pair_check()
            for name, process, handle in pending[:]:
                code = process.poll()
                if code is not None:
                    handle.close()
                    pending.remove((name, process, handle))
                    record('finished' if code == 0 else 'failed', name=name, exit_code=code, formal=formal)
                    if code:
                        errors.append(name)
            if errors:
                raise RuntimeError('Failed jobs: '+', '.join(errors))
            if pending:
                time.sleep(5)
    except BaseException:
        for _, process, handle in pending:
            process.terminate()
            handle.close()
        raise


def main():
    sources = ['run_task5_long_critic.py', 'verify_task5_long_critic.py', 'launch_task5_long_critic.py',
               'TASK5_LONG_CRITIC_PROTOCOL.md', 'run.py', 'actor_input_agent.py', 'run_cube5.py',
               'run_actor_input_ablation.py', 'audit_cube5.py', 'actor_input_diagnostics.py', 'dawn_agent.py',
               'vendor/qam/agents/qam.py', 'vendor/qam/utils/networks.py', 'vendor/qam/utils/datasets.py']
    manifest = {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sources}
    p = RUNS/'source_manifest.json'
    if p.exists():
        assert load(p) == manifest
    else:
        write(p, manifest)
    with (RUNS/'pip_freeze.txt').open('w') as f:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=f, check=True)
    assert shutil.disk_usage(ROOT).free > 4*1024**3, 'Insufficient space for retained checkpoints'
    if not (RUNS/'PREFLIGHT_PASSED.json').exists():
        if not (RUNS/'smoke_collector/WARMUP_COLLECTED.json').exists():
            execute([command('smoke_collector', 'warm', True, collect=True)])
        execute([command('smoke_warm', 'warm', True), command('smoke_random', 'random', True)])
        if not (RUNS/'smoke_resume/SMOKE_PAUSED.json').exists():
            execute([command('smoke_resume', 'warm', True, True)])
        execute([command('smoke_resume', 'warm', True)])
        env = dict(os.environ, JAX_PLATFORMS='cpu', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        subprocess.run([sys.executable, 'verify_task5_long_critic.py'], cwd=ROOT, env=env, check=True)
        record('preflight_passed', checks=load(RUNS/'PREFLIGHT_PASSED.json'))
        # Delete only temporary smoke binaries after recording exact state checksums.
        for folder in ('smoke_warm', 'smoke_random', 'smoke_resume', 'smoke_collector'):
            for path in (RUNS/folder).glob('*.pkl'):
                path.unlink()
    if not (RUNS/'shared_warmup/WARMUP_COLLECTED.json').exists():
        execute([command('shared_warmup', 'warm', collect=True)])
    record('starting_formal_training', task=5, arms=['warm', 'random'], primitive_steps=500000,
           warmup=20000, final_updates=120000, seed=0)
    execute([command('warm', 'warm'), command('random', 'random')], formal=True)
    pair_check()
    results = {}
    for name in ('warm', 'random'):
        assert load(RUNS/name/'CHECKS_PASSED.json')['status'] == 'passed'
        results[name] = [load(p) for p in sorted((RUNS/name).glob('paired_*.json'))]
    write(RUNS/'results.json', results)
    record('all_training_complete', task=5, steps_per_arm=500000, updates_per_arm=120000)


if __name__ == '__main__':
    RUNS.mkdir(parents=True, exist_ok=True)
    lock = (RUNS/'suite.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        main()
    except BaseException as exc:
        record('suite_failed', type=type(exc).__name__, message=str(exc))
        raise
