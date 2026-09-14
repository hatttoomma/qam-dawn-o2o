"""Validate and dispatch Task1/5 with 80k warmup and 150k total online steps."""
import argparse
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
RUNS = ROOT / 'runs/warmup80k_150k_20260911'


def load(path):
    return json.loads(path.read_text())


def write(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def record(event, **fields):
    data = dict(event=event, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields)
    with (RUNS / 'suite.jsonl').open('a') as stream:
        stream.write(json.dumps(data) + '\n')
    write(RUNS / 'suite_status.json', data)
    print(json.dumps(data), flush=True)


def environment():
    return dict(os.environ, CUDA_VISIBLE_DEVICES='0', XLA_PYTHON_CLIENT_MEM_FRACTION='.38',
                OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                MUJOCO_GL='egl', __EGL_VENDOR_LIBRARY_FILENAMES=str(RUNS / 'egl_vendor.json'),
                PYTHONUNBUFFERED='1', JAX_COMPILATION_CACHE_DIR=str(ROOT / 'jax_cache'),
                XLA_FLAGS=(os.environ.get('XLA_FLAGS', '') + ' --xla_gpu_enable_command_buffer=').strip())


def command(task, arm, *, smoke=False, collect=False, resume_test=False, pause=False):
    prefix = 'smoke_' if smoke else ''
    name = f'{prefix}task{task}_' + ('collector' if collect else ('resume' if resume_test else arm))
    args = ['run_warmup80k.py', '--task', str(task), '--stage', arm,
            '--out', str(RUNS / name), '--dawn-batch', '256',
            '--warmup', '80' if smoke else '80000',
            '--online-steps', '200' if smoke else '150000',
            '--eval-episodes', '2' if smoke else '100',
            '--final-episodes', '2' if smoke else '100']
    if smoke:
        args.append('--smoke')
    if collect:
        args.append('--collect-warmup')
    else:
        args += ['--shared-warmup', str(RUNS / f'{prefix}task{task}_collector')]
    if pause:
        args += ['--stop-after-checkpoint', '120']
    return name, args, 'WARMUP_COLLECTED.json' if collect else ('SMOKE_PAUSED.json' if pause else 'CHECKS_PASSED.json')


def execute(jobs, phase):
    jobs = list(jobs)
    while jobs:
        batch, jobs = jobs[:2], jobs[2:]
        pending = []
        for name, args, marker in batch:
            if (RUNS / name / marker).exists():
                record('already_complete', name=name, marker=marker, phase=phase)
                continue
            handle = (RUNS / f'{name}.log').open('a')
            process = subprocess.Popen([sys.executable, *args], cwd=ROOT, env=environment(),
                                       stdout=handle, stderr=subprocess.STDOUT)
            record('started', name=name, pid=process.pid, phase=phase, command=args)
            pending.append((name, process, handle, marker))
        try:
            while pending:
                for name, process, handle, marker in pending[:]:
                    code = process.poll()
                    if code is not None:
                        handle.close()
                        pending.remove((name, process, handle, marker))
                        record('finished' if code == 0 else 'failed', name=name, exit_code=code, phase=phase)
                        if code or not (RUNS / name / marker).exists():
                            raise RuntimeError(f'{name} failed or did not write {marker}')
                if pending:
                    time.sleep(5)
        except BaseException:
            for _, process, handle, _ in pending:
                process.terminate()
                handle.close()
            raise


def summarize(arms):
    results = []
    for task in (1, 5):
        for arm in arms:
            folder = RUNS / f'task{task}_{arm}'
            done = load(folder / 'DONE.json')
            assert done['steps'] == 150000 and done['updates'] == 17500
            assert load(folder / 'CHECKS_PASSED.json')['status'] == 'passed'
            initial = load(folder / 'eval_000000_100.json')
            results.append(dict(task=task, arm=arm, step=0, updates=0, mode='offline',
                                success=initial['success'], return_mean=initial['return_mean']))
            for path in sorted(folder.glob('eval_*_100*.json')):
                evaluation = load(path)
                if evaluation['step'] == 0:
                    continue
                rows = evaluation['records']
                assert len(rows) == evaluation['episodes'] == 100
                assert abs(sum(r['success'] for r in rows)/100 - evaluation['success']) < 1e-10
                assert abs(sum(r['return_'] for r in rows)/100 - evaluation['return_mean']) < 1e-8
                results.append(dict(task=task, arm=arm, step=evaluation['step'],
                                    updates=max(0, (evaluation['step']-80000)//4),
                                    mode='mean_residual' if evaluation['deterministic_residual'] else 'sampled',
                                    success=evaluation['success'], return_mean=evaluation['return_mean']))
    write(RUNS / 'results.json', results)
    import csv
    with (RUNS / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--critic-init', choices=('inherited', 'random', 'both'), default='inherited')
    args = parser.parse_args()
    arms = ['warm', 'random'] if args.critic_init == 'both' else ['warm' if args.critic_init == 'inherited' else 'random']
    request = dict(tasks=[1, 5], arms=arms, seed=0, offline_updates=500000,
                   warmup_steps=80000, online_steps=150000, gradient_updates=17500,
                   batch_size=256, utd=.25, td_target='hard', actor_input='state+base_action',
                   primary_evaluation='mean_residual', secondary_evaluation='sampled')
    path = RUNS / 'suite_config.json'
    if path.exists():
        assert load(path) == request, 'Existing suite has a different scope'
    else:
        write(path, request)
    assert shutil.disk_usage(ROOT).free > 3 * 1024**3, 'Less than 3 GiB free disk space'
    library = Path('/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0').resolve(strict=True)
    write(RUNS / 'egl_vendor.json', dict(file_format_version='1.0.0', ICD=dict(library_path=str(library))))
    write(RUNS / 'runtime_compatibility.json', dict(
        egl_library=str(library), xla_flags=environment()['XLA_FLAGS'],
        gpu_execution=True, cuda_graph_capture_disabled=True,
        reason='Default graph capture failed with CUDA_ERROR_SHARED_OBJECT_INIT_FAILED on this driver; use normal GPU kernel execution.'))
    sources = load(ROOT / 'runs/cube5/source_manifest.json')
    for name, expected in sources.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    for name in ('run_warmup80k.py', 'launch_warmup80k.py', 'verify_warmup80k.py', 'WARMUP80K_PROTOCOL.md'):
        sources[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    path = RUNS / 'source_manifest.json'
    if path.exists():
        assert load(path) == sources
    else:
        write(path, sources)
    if not (RUNS / 'pip_freeze.txt').exists():
        with (RUNS / 'pip_freeze.txt').open('w') as stream:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=stream, check=True)
    if not (RUNS / 'PREFLIGHT_PASSED.json').exists():
        execute([command(t, 'warm', smoke=True, collect=True) for t in (1, 5)], 'smoke_collect')
        execute([command(t, a, smoke=True) for t in (1, 5) for a in arms], 'smoke_train')
        execute([command(1, arms[0], smoke=True, resume_test=True, pause=True)], 'smoke_pause')
        execute([command(1, arms[0], smoke=True, resume_test=True)], 'smoke_resume')
        env = dict(environment(), JAX_PLATFORMS='cpu')
        subprocess.run([sys.executable, 'verify_warmup80k.py'], cwd=ROOT, env=env, check=True)
        record('preflight_passed', checks=load(RUNS / 'PREFLIGHT_PASSED.json'))
    record('starting_formal_warmup', **request)
    execute([command(t, 'warm', collect=True) for t in (1, 5)], 'formal_collect')
    record('starting_formal_updates', **request)
    execute([command(t, a) for t in (1, 5) for a in arms], 'formal_train')
    summarize(arms)
    record('all_training_complete', **request)


if __name__ == '__main__':
    RUNS.mkdir(parents=True, exist_ok=True)
    lock = (RUNS / 'suite.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        main()
    except BaseException as exc:
        record('suite_failed', error=type(exc).__name__, message=str(exc))
        raise
