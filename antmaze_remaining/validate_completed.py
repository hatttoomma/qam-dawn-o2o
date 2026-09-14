"""Validate and sync completed native/DAWN task pairs from a full snapshot."""
import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'output/antmaze_large_tasks2_5_20260913'
parser = argparse.ArgumentParser()
parser.add_argument('--snapshot', type=Path, default=OUT / 'full_snapshot.json')
args = parser.parse_args()
snapshot = json.loads(args.snapshot.read_text())
assert snapshot['full']
files = snapshot['files']
assert not any(k.endswith('/FAILED.json') for k in files)
prior = json.loads((ROOT / 'output/antmaze_large_task1_20260913/final_complete_snapshot.json').read_text())['files']

def finite(value):
    if isinstance(value, dict): return all(finite(v) for v in value.values())
    if isinstance(value, list): return all(finite(v) for v in value)
    if isinstance(value, (float, int)): return math.isfinite(value)
    return True

all_rows, completed = [], []
for task in (2, 3, 4, 5):
    prefix = f'results/task{task}/'
    if prefix + 'dawn/CHECKS_PASSED.json' not in files:
        continue
    get = lambda name: files[prefix + name]
    native, dawn, checks = get('native/DONE.json'), get('dawn/DONE.json'), get('dawn/CHECKS_PASSED.json')
    assert native['status'] == checks['status'] == 'passed'
    assert (native['offline_updates'], native['online_env_steps'], native['online_updates']) == (500000, 50000, 45001)
    assert native['official_main_byte_identical']
    assert (dawn['steps'], dawn['updates']) == (100000, 5000)
    assert (checks['steps'], checks['updates'], checks['task_id']) == (100000, 5000, task)
    assert checks['evaluation_steps_verified'] == [90000, 100000]
    for key in ('unchanged_historical_collection_and_update_loop', 'native_gate_passed',
                'offline_checkpoint_matched', 'hard_TD_verified', 'frozen_base_verified'):
        assert checks[key]
    native_cfg = get('native/actual_agent_config.json')
    assert native_cfg['vanilla_QAM'] and native_cfg['task_id'] == task and native_cfg['seed'] == 0
    assert native_cfg['agent_config'] == prior['results/native/actual_agent_config.json']['agent_config']
    assert get('native/source_verification.json')['status'] == 'passed'
    cfg, old_cfg = get('dawn/actual_agent_config.json'), prior['results/dawn/actual_agent_config.json']
    changed = {k for k in cfg.keys() | old_cfg.keys() if cfg.get(k) != old_cfg.get(k)}
    assert changed == {'task_id', 'checkpoint_steps', 'online_steps'}, changed
    assert cfg['task_id'] == task and cfg['online_steps'] == 100000 and cfg['checkpoint_steps'] == [0, 80000, 90000, 100000]
    data = get('native/dataset_manifest.json')
    assert data['task_id'] == task and data['environment'] == f'antmaze-large-navigate-singletask-task{task}-v0'
    assert data == get('dawn/dataset_manifest.json')
    offline = get('native/offline_checkpoint.json')
    initial = get('dawn/initial_hashes.json')
    assert get('dawn/config.json')['offline_sha256'] == offline['sha256']
    assert initial == dawn['initial_hashes']
    assert initial['flow'] == dawn['final_flow_hash'] == offline['flow_hash']
    assert initial['critic'] == offline['q_hash'] and initial['target'] == offline['target_q_hash']
    warm = get('dawn/warmup.json')
    assert warm['steps'] == warm['requested_steps'] == warm['chunks'] == 80000
    assert warm['actor_hash'] == initial['actor'] and warm['critic_hash'] == initial['critic']
    assert get('dawn/INITIAL_EVAL_MATCHED.json')['status'] == 'passed'
    for step in (90000, 100000):
        marker = get(f'dawn/eval_rng_preserved_{step:06d}.json')
        assert marker['status'] == 'passed' and marker['numpy_rng_unchanged'] and marker['step'] == step
    audits = {k: snapshot['checkpoint_audit'][k] for k in (
        prefix + 'native/offline_500k.pkl', prefix + 'native/native_final.pkl', prefix + 'dawn/final.pkl')}
    assert all(v['matched'] for v in audits.values())
    for stage in ('native', 'dawn'):
        assert get(f'{stage}/runtime.json')['returncode'] == 0
        assert finite(get(f'{stage}/metrics.jsonl'))
    for row in get('dawn/metrics.jsonl'):
        assert row['updates'] == max(0, row['step'] - 80000) // 4
    online_rows = [row for row in get('native/metrics.jsonl') if row['stage'] == 'native_online']
    assert online_rows and online_rows[-1]['online_env_steps'] == 50000
    for row in online_rows:
        assert row['online_updates'] == max(0, row['online_env_steps'] - 4999)
    base = get('native/fixed_eval/eval_000000_100.json')['records']
    assert base == get('dawn/eval_000000_100.json')['records']
    evaluate_names = [('Offline QAM', 'native/fixed_eval/eval_000000_100.json'),
                      ('QAM native', 'native/fixed_eval/eval_050000_100.json')]
    for step in (90000, 100000):
        for method, suffix in [('DAWN mean', '_mean_residual'), ('DAWN sampled', '')]:
            evaluate_names.append((method, f'dawn/eval_{step:06d}_100{suffix}.json'))
    rows = []
    for method, name in evaluate_names:
        evaluation = get(name)
        records = evaluation['records']
        assert evaluation['episodes'] == len(records) == 100
        assert [r['episode'] for r in records] == list(range(100))
        assert all((a['reset_seed'], a['initial_hash']) == (b['reset_seed'], b['initial_hash']) for a, b in zip(records, base))
        assert all(r['success'] in (0, 1) and r['return_'] == -r['length'] + r['success'] for r in records)
        success = sum(r['success'] for r in records) / 100
        returns = sum(r['return_'] for r in records) / 100
        assert math.isclose(success, evaluation['success'], abs_tol=1e-12)
        assert math.isclose(returns, evaluation['return_mean'], abs_tol=1e-9)
        steps = evaluation['step']
        updates = max(0, steps - 4999) if method == 'QAM native' else max(0, steps - 80000) // 4
        rows.append(dict(task=task, method=method, online_env_steps=steps, online_updates=updates,
                         success_rate=success, avg_return=returns,
                         gained_vs_offline=sum(a['success'] > b['success'] for a, b in zip(records, base)),
                         lost_vs_offline=sum(a['success'] < b['success'] for a, b in zip(records, base)),
                         source=prefix + name))
    for name, value in files.items():
        if not name.startswith(prefix): continue
        target = OUT / 'synced_results' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(''.join(json.dumps(r) + '\n' for r in value) if name.endswith('.jsonl') else json.dumps(value, indent=2))
    validation = dict(status='passed', task_id=task, snapshot_utc=snapshot['utc'],
                      paired_initial_states_verified=True, initial_records_exact_match=True,
                      evaluation_aggregates_recomputed=True, configuration_verified=True,
                      frozen_base_and_inherited_Q_verified=True, evaluation_rng_preserved=True,
                      update_counts_verified=True, finite_logged_metrics=True, checkpoint_audit=audits,
                      native_wall_seconds=get('native/runtime.json')['wall_seconds'],
                      dawn_wall_seconds=get('dawn/runtime.json')['wall_seconds'], summary=rows)
    (OUT / f'task{task}_completion_validation.json').write_text(json.dumps(validation, indent=2))
    (OUT / f'task{task}_result.json').write_text(json.dumps(rows, indent=2))
    all_rows.extend(rows)
    completed.append(task)
if all_rows:
    with (OUT / 'completed_tasks_comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
(OUT / 'synced_source_sha256.json').write_text(json.dumps(snapshot['file_sha256'], indent=2))
print(json.dumps(dict(completed_verified=completed, summary=all_rows), indent=2))
