"""Audit completed runs and produce measured endpoint comparisons only."""
import argparse, csv, json, math, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent / 'reports/100k_20260914'
parser = argparse.ArgumentParser()
parser.add_argument('--snapshot', type=Path, default=DEFAULT_OUT / 'final_complete_snapshot.json')
parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
args = parser.parse_args()
OUT = args.output_dir
OUT.mkdir(parents=True, exist_ok=True)
snapshot = json.loads(args.snapshot.read_text())
assert snapshot['full'] and not snapshot['errors']
rows, verified, errors, pairs = [], [], {}, {}

def finite(value):
    if isinstance(value, dict): return all(finite(v) for v in value.values())
    if isinstance(value, list): return all(finite(v) for v in value)
    return math.isfinite(value) if isinstance(value, (int, float)) else True

def check_eval(e, benchmark):
    records = e['records']
    assert e['step'] == 100000 and e['episodes'] == len(records) == 100
    assert [r['episode'] for r in records] == list(range(100))
    assert [r['reset_seed'] for r in records] == list(range(500000, 500100))
    assert all(r['success'] in (0, 1) for r in records)
    if benchmark == 'antmaze-large':
        assert all(r['return_'] == -r['length'] + r['success'] for r in records)
    else:
        # Live OGBench CubeEnv.compute_reward sums object successes minus2.
        # Each step can contribute -2, -1 or0; return is not just -episode_length.
        assert all(0 < r['length'] <= 500 and -2 * r['length'] <= r['return_'] <= 0
                   and float(r['return_']).is_integer() for r in records)
    assert finite(e)
    assert math.isclose(sum(r['success'] for r in records) / 100, e['success'], abs_tol=1e-12)
    assert math.isclose(sum(r['return_'] for r in records) / 100, e['return_mean'], abs_tol=1e-9)
    return [(r['episode'], r['reset_seed'], r['initial_hash']) for r in records]

for queue_name, queue in snapshot['queues'].items():
    files = queue['files']
    benchmark = 'cube-double' if queue_name == 'cube' else 'antmaze-large'
    methods = ['native', 'dawn'] if queue_name == 'cube' else [queue_name.split('_')[1]]
    local_dir = 'antmaze' if queue_name == 'antmaze_dawn' else queue_name
    assert queue['inputs.json'] == json.loads((HERE / local_dir / 'inputs.json').read_text())
    assert queue['adapter_hashes.json'] == json.loads((HERE / local_dir / 'adapter_hashes.json').read_text())
    assert all(v['matched'] for v in queue['source_audit'].values())
    for name, value in files.items():
        if name.endswith('/FAILED.json'): errors[queue_name + '/' + name] = value
    for task in range(1, 6):
        for method in methods:
            prefix = f'results/task{task}/{method}/'
            label = f'{benchmark}/task{task}/{method}'
            if prefix + 'CHECKS_PASSED.json' not in files or prefix + 'runtime.json' not in files: continue
            try:
                get = lambda name: files[prefix + name]
                done, cfg, checks = get('DONE.json'), get('actual_agent_config.json'), get('CHECKS_PASSED.json')
                updates = 5625 if method == 'dawn' else 95001
                assert (done['steps'], done['updates']) == (100000, updates)
                assert checks['status'] == 'passed' and checks['task'] == task
                assert (checks['steps'], checks['updates']) == (100000, updates)
                assert get('runtime.json')['returncode'] == 0
                assert queue['checkpoint_audit'][prefix + 'final.pkl']['matched']
                assert finite(get('metrics.jsonl'))
                if method == 'dawn':
                    for key, expected in dict(online_steps=100000, warmup=40000, batch_size=256,
                        td_target='naive', actor_input='state+base_action', residual_scale=.1,
                        critic_ensemble=10, target_aggregation='minimum', actor_Q_aggregation='minimum',
                        learning_rate=1e-4, discount=.99, target_tau=.01, actor_entropy_enabled=True,
                        automatic_alpha_enabled=True, inherited_Q_and_target=True).items():
                        assert cfg[key] == expected, (key, cfg.get(key))
                    assert done['initial_hashes']['flow'] == done['final_flow_hash']
                    if queue_name == 'cube':
                        assert (cfg['utd_before_switch'], cfg['utd_after_switch'], cfg['replay_switch']) == (.25, .0625, 50000)
                        assert cfg['horizon'] == 5 and cfg['actor_input_dim'] == 62
                        assert 40000 <= get('warmup.json')['steps'] <= 40004
                        mix, online = get('mixed_replay_audit.json'), get('online_replay_audit.json')
                        assert (mix['update'], mix['offline_per_batch'], mix['online_per_batch']) == (2500, 128, 128)
                        assert mix['offline_draws'] == mix['online_draws'] == 320000
                        assert (online['online_only_updates'], online['online_only_draws']) == (3125, 800000)
                        assert (online['offline_per_batch'], online['online_per_batch']) == (0, 256)
                        assert get('eval_rng_preserved_100000.json')['status'] == 'passed'
                        assert done['final_flow_hash'] == queue['inputs.json'][str(task)]['offline_metadata']['flow_hash']
                    else:
                        start = 50000 if task == 4 else 60000
                        assert cfg['utd'] == .0625 and cfg['horizon'] == 1 and cfg['actor_input_dim'] == 37
                        assert get('resume_source.json')['resume_step'] == start
                        assert get('RESTORE_PASSED.json')['status'] == 'passed'
                        assert get('environment_restored.json')['status'] == 'passed'
                        audit = get('replay_audit.json')
                        assert (audit['updates'], audit['online_buffer_size']) == (5625, 100000)
                        assert (audit['online_per_batch'], audit['offline_per_batch']) == (256, 0)
                        assert get('EVAL_INTEGRITY.json')['training_state_unchanged']
                    modes = [('mean', 'eval_100000_100_mean_residual.json'), ('sampled', 'eval_100000_100.json')]
                else:
                    assert cfg['utd'] == 1 and cfg['online_steps'] == 100000 and cfg['native_start'] == 5000
                    assert done['initial_flow_hash'] != done['final_flow_hash']
                    if queue_name == 'cube':
                        assert get('RESTORE_PASSED.json')['status'] == 'passed'
                        assert get('environment_restored.json')['status'] == 'passed'
                        assert cfg['qam_config']['edit_scale'] == 0 and cfg['qam_config']['horizon_length'] == 5
                        modes = [('native', 'eval_100000_100.json')]
                    else:
                        assert cfg['new_online_run'] and cfg['original_online50k_replay_missing']
                        assert cfg['agent_config']['edit_scale'] == 0 and cfg['agent_config']['horizon_length'] == 1
                        assert get('initial_state.json')['checkpoint_sha256'] == queue['inputs.json'][str(task)]['offline_sha256']
                        # Pinned official ReplayBuffer increments pointer modulo capacity,
                        # then size=max(pointer,size). At exact capacity size stays capacity-1.
                        # Training still runs100k updates/rollouts as logged; the saved slice
                        # excludes the physical last transition and is not a full resume state.
                        assert done['online_buffer_size'] == 99999
                        assert get('requested_flags.json')['balanced_sampling'] is False
                        assert get('metrics.jsonl')[-1]['step'] == 100000
                        assert get('metrics.jsonl')[-1]['updates'] == 95001
                        modes = [('native', 'fixed_eval/eval_100000_100.json')]
                run_rows, identities = [], []
                for mode, filename in modes:
                    evaluation = get(filename)
                    identity = check_eval(evaluation, benchmark)
                    identities.append(identity)
                    run_rows.append(dict(benchmark=benchmark, task=task, method=method, evaluation_mode=mode,
                        online_env_steps=100000, online_gradient_updates=updates, success_rate=evaluation['success'],
                        avg_return=evaluation['return_mean'], new_run_wall_seconds=get('runtime.json')['wall_seconds'],
                        source=queue['root'] + '/' + prefix + filename))
                assert all(i == identities[0] for i in identities)
                pairs.setdefault((benchmark, task), {})[method] = identities[0]
                rows.extend(run_rows); verified.append(label)
            except Exception as e: errors[label] = traceback.format_exc()

for pair, identity in pairs.items():
    if set(identity) == {'native', 'dawn'} and identity['native'] != identity['dawn']:
        errors['paired:' + str(pair)] = 'Native and DAWN initial states differ; do not label paired.'
OUT.mkdir(exist_ok=True)
if rows:
    with (OUT / 'completed_results.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
result = dict(snapshot_utc=snapshot['utc'], completed_verified=verified, count=len(verified),
              all20_complete=len(verified) == 20 and not errors, errors=errors, rows=rows)
(OUT / 'validation_summary.json').write_text(json.dumps(result, indent=2))
lines = ['# 100k online results', '', 'One training seed; 100 paired evaluation episodes per task.', '',
         '|Benchmark|Task|Method / evaluation|Online updates|Success|Avg return|', '|---|---:|---|---:|---:|---:|']
for r in rows:
    lines.append(f"|{r['benchmark']}|{r['task']}|{r['method']} / {r['evaluation_mode']}|{r['online_gradient_updates']}|{r['success_rate']:.0%}|{r['avg_return']:.2f}|")
lines += ['', 'DAWN mean residual is the primary comparison; sampled residual is secondary. Native uses the original QAM sampling policy.',
          'Antmaze native reuses offline500k and starts a new100k online run because the old50k replay was not saved. Cube native resumes its complete50k state.',
          'Antmaze native final replay export contains99999 online transitions: the pinned official replay size counter excludes its last slot at exact capacity. All100000 environment steps and95001 updates completed; this exported buffer plus partial environment state must not be treated as an exact resume checkpoint.',
          'New wall time excludes reused historical training. Gradient iteration counts are not equal compute budgets.']
if errors: lines += ['', 'Validation errors: ' + json.dumps(errors)]
(OUT / 'results.md').write_text('\n'.join(lines) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
if errors: raise SystemExit(1)
