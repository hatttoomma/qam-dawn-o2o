"""Compare recorded hyperparameters, source, initialization and evaluation evidence."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'output/warmup80k_150k_task234_20260911/settings_audit'


def load(name):
    return json.loads((ROOT / name).read_text())


def differences(records):
    keys = set().union(*(r.keys() for r in records.values()))
    return {k: {str(t): r.get(k, '<absent>') for t, r in records.items()}
            for k in sorted(keys) if len({json.dumps(r.get(k, '<absent>'), sort_keys=True)
                                        for r in records.values()}) != 1}


def records(filename, stage='warm'):
    return {t: load(f'task{t}_{stage}/{filename}') for t in range(1, 6)}


def main():
    actual = records('actual_agent_config.json')
    configs = records('config.json')
    requested = records('requested_config.json')
    offline = records('config.json', 'offline')
    initials = records('initial_hashes.json')
    warmups = records('warmup.json')
    manifests = records('task_manifest.json')
    actual_diff = differences(actual)
    assert set(actual_diff) == {'task', 'full_initial_agent_hash'}, actual_diff
    config_diff = differences(configs)
    assert set(config_diff) <= {'task', 'environment', 'out', 'offline_checkpoint', 'offline_sha256', 'shared_warmup'}, config_diff
    requested_diff = differences(requested)
    assert set(requested_diff) <= {'task', 'environment', 'out', 'offline_checkpoint', 'offline_sha256',
                                   'shared_warmup', 'shared_warmup_sha256', 'source_hashes'}, requested_diff
    common_sources = set.intersection(*(set(x['source_hashes']) for x in requested.values()))
    for name in common_sources:
        assert len({x['source_hashes'][name] for x in requested.values()}) == 1, name
    offline_qam = {t: x['qam_config'] for t, x in offline.items()}
    assert not differences(offline_qam)
    for x in offline.values():
        assert x['offline_steps'] == 500000 and x['seed'] == 0
    actor_hashes = {x['actor'] for x in initials.values()}
    assert len(actor_hashes) == 1
    for key in ('actor_optimizer', 'critic_optimizer', 'alpha'):
        assert len({json.dumps(x[key]) for x in initials.values()}) == 1, key
    assert len({x['non_Q_initial_hash'] for x in actual.values()}) == 1
    for name in ('pip_freeze.txt', 'runtime_compatibility.json'):
        assert (ROOT / 'task15_suite' / name).read_bytes() == (ROOT / 'task234_suite' / name).read_bytes()
    a, b = load('task15_suite/suite_config.json'), load('task234_suite/suite_config.json')
    assert set(differences({1: a, 2: b})) == {'tasks'}
    extension = load('source/warmup80k_task234_extension_diff.json')
    source = (ROOT / 'source/run_warmup80k_task234.py').read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == extension['extended_sha256']
    for change in reversed(extension['changes']):
        assert source.count(change['new']) == 1
        source = source.replace(change['new'], change['old'])
    assert source == (ROOT / 'source/run_warmup80k.py').read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == extension['source_sha256']
    assert len({x['max_episode_steps'] for x in manifests.values()}) == 1
    for k in ('dataset_size', 'observations_hash', 'actions_hash', 'observation_shape', 'action_shape'):
        assert len({json.dumps(x[k]) for x in manifests.values()}) == 1, k
    task_rows = []
    for t in range(1, 6):
        offline_done = load(f'task{t}_offline/DONE.json')
        assert initials[t]['flow'] == offline_done['flow_hash']
        assert initials[t]['critic'] == offline_done['q_hash']
        assert initials[t]['target'] == offline_done['target_q_hash']
        assert configs[t]['offline_sha256'] == offline_done['checkpoint_sha256']
        assert manifests[t]['reward_task_id'] == t
        assert warmups[t]['requested_steps'] == 80000
        assert 80000 <= warmups[t]['steps'] <= 80004
        shared = load(f'task{t}_warm/shared_initialization.json')
        assert shared['updates'] == 0 and shared['replay_hash'] == warmups[t]['replay_hash']
        assert load(f'task{t}_warm/backup_math.json')['hard_independent_of_entropy']
        assert load(f'task{t}_warm/actor_input_checks.json')['real_feature_gradients_nonzero']
        baseline = load(f'task{t}_warm/eval_000000_100.json')
        assert baseline['episodes'] == 100 and baseline['raw_record_aggregates_verified']
        baseline_ids_hash = baseline['record_ids_hash']
        evals = []
        for p in sorted((ROOT / f'task{t}_warm').glob('eval_*.json')):
            e = json.loads(p.read_text())
            assert e['episodes'] == 100 and e['raw_record_aggregates_verified']
            assert e['record_ids_hash'] == baseline_ids_hash
            evals.append(dict(step=e['step'], mean_residual=e['deterministic_residual']))
        metrics = load(f'task{t}_warm/metrics_verified.json')
        assert metrics['all_finite'] and metrics['update_schedule_verified']
        restoration = [json.loads(line) for line in (ROOT / f'task{t}_warm/restore_checks.jsonl').read_text().splitlines()]
        assert all(x['observation_max_abs_error'] == 0 for x in restoration)
        for line in (ROOT / f'task{t}_warm/evaluation_integrity.jsonl').read_text().splitlines():
            x = json.loads(line)
            assert x['agent_unchanged'] and x['training_numpy_rng_unchanged']
        done_path = ROOT / f'task{t}_warm/DONE.json'
        done = json.loads(done_path.read_text()) if done_path.exists() else None
        if done:
            assert done['steps'] == 150000 and done['updates'] == 17500
            assert done['final_flow_hash'] == initials[t]['flow']
        task_rows.append(dict(task=t, baseline_success=baseline['success'], baseline_return=baseline['return_mean'],
                              actual_warmup_steps=warmups[t]['steps'], replay_chunks=warmups[t]['chunks'],
                              offline_checkpoint_sha256=configs[t]['offline_sha256'], complete=bool(done),
                              evaluations=evals))
    result = dict(status='passed', actual_configuration_differences=actual_diff,
                  recorded_argument_differences=config_diff, requested_argument_differences=requested_diff,
                  shared_hyperparameters={k:v for k,v in actual[1].items() if k not in actual_diff},
                  offline_QAM_hyperparameters_identical=True, initial_residual_actor_bitwise_identical=True,
                  initial_non_Q_state_bitwise_identical=True, critic_inheritance_verified=True,
                  training_source_equivalence_verified=True, common_source_hashes_verified=sorted(common_sources),
                  runtime_and_package_versions_identical=True, evaluation_protocol_identical=True,
                  environment_horizon=manifests[1]['max_episode_steps'], tasks=task_rows,
                  operational_differences=['Different task goals/rewards/masks and task-specific pretrained base/Q weights.',
                    'Different collected replay/trajectories despite the same seed and sampling procedure.',
                    'Whole-chunk boundaries differ by at most four primitive steps; final update budget is fixed.',
                    'Task4 runs mostly alone; the other tasks ran with a second job on the same GPU.',
                    'Task2/3/4 checkpoint output is on the system disk, with the same serialization.',
                    'Independent GPU executions are not claimed bitwise deterministic.'])
    (ROOT / 'audit_result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('requested_argument_differences', 'shared_hyperparameters', 'recorded_argument_differences')}, indent=2))


if __name__ == '__main__':
    main()
