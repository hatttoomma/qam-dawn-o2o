"""Run the unchanged 80k/150k experiment on Task2, Task3 and Task4."""
from collections import deque
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import launch_warmup80k as previous

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'runs/warmup80k_150k_task234_20260911'
REFERENCE = ROOT / 'runs/warmup80k_150k_20260911'
TASKS = (2, 3, 4)
OFFLINES = {
    2: ('runs/task2/offline', 'b117517f8e6d2227137a715de4c5045c0468b252949c0c782e94265dd489cd02'),
    3: ('runs/cube5/task3_offline', 'd955aeb235ca369996c54621421b29ba0cd933be61bc92909729cba0cc442ffd'),
    4: ('runs/cube5/task4_offline', 'aa98ada060943a3633979c5fbdf9cc1b77cda950a409a0e5e4f5fd2f79097670'),
}
previous.RUNS = RUNS
load, write, record = previous.load, previous.write, previous.record


def digest(path):
    with path.open('rb') as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def prerequisites():
    reference = load(REFERENCE / 'suite_config.json')
    request = dict(reference, tasks=list(TASKS))
    assert request['arms'] == ['warm'] and request['seed'] == 0
    assert load(REFERENCE / 'suite_status.json')['event'] == 'all_training_complete'
    assert load(REFERENCE / 'PREFLIGHT_PASSED.json')['status'] == 'passed'
    path = RUNS / 'suite_config.json'
    if path.exists():
        assert load(path) == request
    else:
        write(path, request)
    assert shutil.disk_usage(RUNS).free > 5 * 1024**3, 'Need 5 GiB for retained checkpoints'
    sources = load(REFERENCE / 'source_manifest.json')
    for name, expected in sources.items():
        assert digest(ROOT / name) == expected, name
    extension = load(ROOT / 'warmup80k_task234_extension_diff.json')
    assert digest(ROOT / 'run_warmup80k.py') == extension['source_sha256']
    assert digest(ROOT / 'run_warmup80k_task234.py') == extension['extended_sha256']
    normalized = (ROOT / 'run_warmup80k_task234.py').read_text()
    for change in reversed(extension['changes']):
        assert normalized.count(change['new']) == 1
        normalized = normalized.replace(change['new'], change['old'])
    assert normalized == (ROOT / 'run_warmup80k.py').read_text()
    for name in ('run_warmup80k_task234.py', 'launch_warmup80k_task234.py',
                 'warmup80k_task234_extension_diff.json', 'WARMUP80K_TASK234_PROTOCOL.md'):
        sources[name] = digest(ROOT / name)
    path = RUNS / 'source_manifest.json'
    if path.exists():
        assert load(path) == sources
    else:
        write(path, sources)
    task_checks = []
    audit = load(ROOT / 'runs/cube5/TASK_AUDIT_PASSED.json')
    assert audit['status'] == 'passed'
    for task, (folder, expected) in OFFLINES.items():
        offline = ROOT / folder
        done, config = load(offline / 'DONE.json'), load(offline / 'config.json')
        assert digest(offline / 'final.pkl') == done['checkpoint_sha256'] == expected
        assert done['step'] == config['offline_steps'] == 500000
        assert config['seed'] == 0
        assert config['qam_config']['edit_scale'] == 0
        assert config['qam_config']['inv_temp'] == 1
        assert load(offline / 'task_manifest.json') == audit['tasks'][task - 1]['manifest']
        old = ROOT / ('runs/actor_input_ablation/task2_base_action' if task == 2 else f'runs/cube5/task{task}_warm')
        actual = load(old / 'actual_agent_config.json')
        assert actual['inherited_Q_and_target'] and actual['actor_input_dim'] == 62
        assert actual['td_target'] == 'hard' and actual['utd'] == .25
        task_checks.append(dict(task=task, offline=folder, offline_sha256=expected,
                                expected_initial_agent_hash=actual['full_initial_agent_hash']))
    library = Path('/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0').resolve(strict=True)
    write(RUNS / 'egl_vendor.json', dict(file_format_version='1.0.0', ICD=dict(library_path=str(library))))
    compatibility = load(REFERENCE / 'runtime_compatibility.json')
    assert str(library) == compatibility['egl_library']
    assert previous.environment()['XLA_FLAGS'] == compatibility['xla_flags']
    write(RUNS / 'runtime_compatibility.json', compatibility)
    with (RUNS / 'pip_freeze.txt').open('w') as stream:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=stream, check=True)
    assert (RUNS / 'pip_freeze.txt').read_text() == (REFERENCE / 'pip_freeze.txt').read_text()
    write(RUNS / 'PREREQUISITES_VERIFIED.json', dict(status='passed', tasks=task_checks,
        reference_suite=str(REFERENCE), training_loop_unchanged=True,
        source_equivalence_verified=True, runtime_and_packages_unchanged=True,
        reused_runtime_preflight=str(REFERENCE / 'PREFLIGHT_PASSED.json'),
        note='Reuse completed Task1/5 runtime and restore checks. Each new process verifies its task dataset and exact inherited initial agent before collection.'))
    write(RUNS / 'storage.json', dict(logical_directory=str(RUNS),
        physical_directory=str(RUNS.resolve()), free_bytes=shutil.disk_usage(RUNS).free))
    record('prerequisites_verified', **request)


def start(task, collect):
    name, args, marker = previous.command(task, 'warm', collect=collect)
    args[0] = 'run_warmup80k_task234.py'
    if (RUNS / name / marker).exists():
        record('already_complete', task=task, name=name, marker=marker)
        return None
    handle = (RUNS / f'{name}.log').open('a')
    process = subprocess.Popen([sys.executable, *args], cwd=ROOT, env=previous.environment(),
                               stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
    record('started', task=task, name=name, pid=process.pid,
           phase='formal_collect' if collect else 'formal_train', command=args)
    return dict(task=task, collect=collect, name=name, process=process, handle=handle, marker=marker)


def run_queue():
    waiting = deque(TASKS)
    active = {}

    def advance(task, collect):
        job = start(task, collect)
        if job is not None:
            active[task] = job
        elif collect:
            advance(task, False)

    try:
        while waiting or active:
            while waiting and len(active) < 2:
                advance(waiting.popleft(), True)
            for task, job in list(active.items()):
                code = job['process'].poll()
                if code is None:
                    continue
                job['handle'].close()
                del active[task]
                record('finished' if code == 0 else 'failed', task=task, name=job['name'], exit_code=code)
                assert code == 0 and (RUNS / job['name'] / job['marker']).exists(), job['name']
                if job['collect']:
                    advance(task, False)
            if active:
                time.sleep(5)
    except BaseException:
        for job in active.values():
            job['process'].terminate()
            job['handle'].close()
        raise


def csv_file(name, rows):
    with (RUNS / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize():
    rows, comparison = [], []
    for task in range(1, 6):
        parent = RUNS if task in TASKS else REFERENCE
        folder = parent / f'task{task}_warm'
        done = load(folder / 'DONE.json')
        assert done['steps'] == 150000 and done['updates'] == 17500
        assert load(folder / 'CHECKS_PASSED.json')['status'] == 'passed'
        assert digest(folder / 'final.pkl') == done['checkpoint_sha256']
        baseline = load(folder / 'eval_000000_100.json')
        native_dir = ROOT / ('runs/native' if task == 1 else 'runs/task2/native' if task == 2 else f'runs/cube5/task{task}_native')
        native = load(native_dir / 'eval_050000_100.json')
        comparison.append(dict(task=task, method='QAM native', online_steps=50000,
            gradient_updates=load(native_dir / 'DONE.json')['updates'],
            success=native['success'], return_mean=native['return_mean']))
        for path in sorted(folder.glob('eval_*_100*.json')):
            e = load(path)
            assert len(e['records']) == e['episodes'] == 100
            assert abs(sum(r['success'] for r in e['records']) / 100 - e['success']) < 1e-10
            assert abs(sum(r['return_'] for r in e['records']) / 100 - e['return_mean']) < 1e-8
            for a, b, c in zip(baseline['records'], e['records'], native['records']):
                for k in ('episode', 'reset_seed', 'initial_hash'):
                    assert a[k] == b[k] == c[k]
            mode = 'offline' if e['step'] == 0 else 'mean_residual' if e['deterministic_residual'] else 'sampled'
            rows.append(dict(task=task, arm='warm', step=e['step'], updates=max(0, (e['step'] - 80000) // 4),
                             mode=mode, success=e['success'], return_mean=e['return_mean']))
            if e['step'] == 150000:
                comparison.append(dict(task=task, method=f'DAWN {mode}', online_steps=150000,
                                       gradient_updates=17500, success=e['success'], return_mean=e['return_mean']))
    new = [r for r in rows if r['task'] in TASKS]
    write(RUNS / 'results.json', new)
    csv_file('results.csv', new)
    write(RUNS / 'all5_results.json', rows)
    csv_file('all5_results.csv', rows)
    csv_file('all5_qam_native_comparison.csv', comparison)


if __name__ == '__main__':
    RUNS.mkdir(parents=True, exist_ok=True)
    lock = (RUNS / 'suite.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        prerequisites()
        run_queue()
        summarize()
        record('all_training_complete', tasks=list(TASKS), all_five_tasks_summarized=True)
    except BaseException as exc:
        record('suite_failed', error=type(exc).__name__, message=str(exc))
        raise
