"""Audit launch smokes, including exact training-state equivalence across resume."""
import json
import pickle
import run as common

RUNS = common.ROOT/'runs/task5_long_critic_shared_20260910'


def read(folder, name):
    return json.loads((RUNS/folder/name).read_text())


def main():
    folders = ('smoke_warm', 'smoke_random', 'smoke_resume')
    for folder in folders:
        assert read(folder, 'CHECKS_PASSED.json')['status'] == 'passed'
        assert read(folder, 'DONE.json')['updates'] == 25
        assert read(folder, 'actor_input_checks.json')['real_feature_gradients_nonzero']
        assert read(folder, 'backup_math.json')['hard_independent_of_entropy']
    warm, random = [read(f, 'actual_agent_config.json') for f in folders[:2]]
    assert warm['non_Q_initial_hash'] == random['non_Q_initial_hash']
    a, b = [read(f, 'initial_hashes.json') for f in folders[:2]]
    for k in ('actor', 'actor_optimizer', 'critic_optimizer', 'alpha', 'flow'):
        assert a[k] == b[k], k
    assert a['critic'] != b['critic'] and b['critic'] == b['target']
    prefix = [read(f, 'warmup.json') for f in folders]
    assert len({x['replay_hash'] for x in prefix}) == 1
    assert len({x['steps'] for x in prefix}) == 1
    baseline = [read(f, 'eval_000000_002.json')['records'] for f in folders]
    assert baseline[0] == baseline[1] == baseline[2]
    shared = [read(f, 'shared_initialization.json') for f in folders]
    assert len({x['collector_checkpoint_sha256'] for x in shared}) == 1
    assert len({x['replay_hash'] for x in shared}) == 1
    states = []
    for folder in ('smoke_warm', 'smoke_resume'):
        with (RUNS/folder/'latest.pkl').open('rb') as f:
            states.append(pickle.load(f))
    fields = ('agent', 'step', 'replay', 'episode', 'trace', 'ob', 'keys', 'numpy_rng')
    hashes = {k:common.tree_hash(states[0][k]) for k in fields}
    for k in fields:
        assert hashes[k] == common.tree_hash(states[1][k]), 'Resume mismatch: '+k
    for name in ('eval_000120_002.json', 'eval_000120_002_mean_residual.json'):
        assert read('smoke_warm', name)['records'] == read('smoke_resume', name)['records']
    common.atomic_json(RUNS/'PREFLIGHT_PASSED.json', dict(status='passed',
        identical_non_Q_initialization=True, inherited_current_and_target_Q_verified=True,
        random_target_equals_current_Q=True, shared_warmup_replay_hash=prefix[0]['replay_hash'],
        identical_initial_evaluations=True, hard_TD_verified=True, action_input_verified=True,
        shared_warmup_initialization_verified=True,
        exact_resume_verified_fields=list(fields), resumed_final_hashes=hashes, smoke_updates=25))


if __name__ == '__main__':
    main()
