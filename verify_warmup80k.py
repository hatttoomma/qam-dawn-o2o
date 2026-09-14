"""Verify short warmup, learning, and uninterrupted/resumed state equivalence."""
import json
import pickle
import argparse
import flax
import jax.numpy as jnp
import run as common
from actor_input_agent import ActorInputAgent
from run_warmup80k import RUNS, OFFLINES

SOURCE = RUNS


def load(folder, name):
    return json.loads((SOURCE / folder / name).read_text())


def fingerprint(key, value):
    if key == 'replay':
        value = {k: common.np.asarray([r[k] for r in value]) for k in value[0]}
    return common.tree_hash(value)


def main():
    global SOURCE
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=__import__('pathlib').Path, default=RUNS)
    SOURCE = parser.parse_args().source
    arms = json.loads((SOURCE / 'suite_config.json').read_text())['arms']
    folders = [f'smoke_task{t}_{a}' for t in (1, 5) for a in arms] + ['smoke_task1_resume']
    for name in folders:
        done = load(name, 'DONE.json')
        assert done['steps'] == 200 and done['updates'] == 30
        assert load(name, 'CHECKS_PASSED.json')['status'] == 'passed'
        assert load(name, 'backup_math.json')['hard_independent_of_entropy']
        assert load(name, 'actor_input_checks.json')['real_feature_gradients_nonzero']
        warmup = load(name, 'warmup.json')
        assert warmup['requested_steps'] == 80 and 80 <= warmup['steps'] < 85
        assert load(name, 'shared_initialization.json')['updates'] == 0
        assert done['final_flow_hash'] == done['initial_hashes']['flow']
        for suffix in ('', '_mean_residual'):
            evaluation = load(name, f'eval_000200_002{suffix}.json')
            assert len(evaluation['records']) == 2 and evaluation['residual_enabled']
    names = [f'smoke_task1_{arms[0]}', 'smoke_task1_resume']
    states = []
    for name in names:
        with (SOURCE / name / 'latest.pkl').open('rb') as stream:
            states.append(pickle.load(stream))
    fields = ['agent', 'step', 'replay', 'episode', 'trace', 'ob', 'keys', 'numpy_rng']
    hashes = {k: fingerprint(k, states[0][k]) for k in fields}
    final_equal = {k: hashes[k] == fingerprint(k, states[1][k]) for k in fields}
    # Independent GPU executions may diverge before either run is paused. Establish
    # that fact explicitly rather than attributing any final difference to restore.
    assert load(names[0], 'initial_hashes.json') == load(names[1], 'initial_hashes.json')
    assert load(names[0], 'warmup.json')['replay_hash'] == load(names[1], 'warmup.json')['replay_hash']
    common_steps = set(p.name for p in (SOURCE/names[0]).glob('checkpoint_*.pkl')) & set(
        p.name for p in (SOURCE/names[1]).glob('checkpoint_*.pkl'))
    before_pause = sorted(n for n in common_steps if n != 'checkpoint_000200.pkl')[0]
    prefix = []
    for name in names:
        with (SOURCE/name/before_pause).open('rb') as stream:
            prefix.append(pickle.load(stream))
    prefix_equal = common.tree_hash(prefix[0]['agent']) == common.tree_hash(prefix[1]['agent'])
    if not all(final_equal.values()):
        assert not prefix_equal, 'Unexpected divergence appeared only after resume; investigate'
    roundtrips = []
    for name in folders:
        task = load(name, 'actual_agent_config.json')['task']
        with (SOURCE/name/'latest.pkl').open('rb') as stream:
            saved = pickle.load(stream)
        ob = jnp.asarray(saved['replay'][0]['observations'])
        qam = common.qam_template(ob, jnp.zeros(5), 0)
        offline, offline_sha = OFFLINES[task]
        assert common.file_hash(common.ROOT/offline) == offline_sha
        qam, _ = common.load_checkpoint(common.ROOT/offline, qam)
        inherited = load(name, 'actual_agent_config.json')['inherited_Q_and_target']
        template = ActorInputAgent.create(qam, ob, 0, inherited).replace(condition_on_base=True)
        restored, payload = common.load_checkpoint(SOURCE/name/'latest.pkl', template)
        assert common.tree_hash(flax.serialization.to_state_dict(restored)) == common.tree_hash(saved['agent'])
        for key in fields[1:]:
            assert fingerprint(key, payload[key]) == fingerprint(key, saved[key]), key
        common.np.random.set_state(payload['numpy_rng'])
        assert common.tree_hash(common.np.random.get_state()) == common.tree_hash(saved['numpy_rng'])
        env = common.ogbench.make_env_and_datasets(f'cube-double-play-singletask-task{task}-v0', env_only=True)
        rebuilt_ob = common.restore_env(env, payload['episode'], 0, payload['trace'], payload['ob'])
        error = float(common.np.max(common.np.abs(rebuilt_ob-payload['ob'])))
        assert error == 0.0
        env.close()
        roundtrips.append(dict(folder=name, agent_and_optimizer_exact=True,
            replay_and_rng_exact=True, simulator_observation_error=error,
            source_checkpoint_sha256=common.file_hash(SOURCE/name/'latest.pkl')))
        del template, restored, payload, saved, qam
    RUNS.mkdir(parents=True, exist_ok=True)
    common.atomic_json(RUNS / 'PREFLIGHT_PASSED.json', dict(status='passed', tasks=[1, 5], arms=arms,
        smoke_warmup=80, smoke_steps=200, smoke_updates=30, state_base_action_input_verified=True,
        plain_TD_verified=True, frozen_base_verified=True, checkpoint_roundtrips=roundtrips,
        independent_runs_final_exact_fields=final_equal, independent_runs_pre_pause_agent_equal=prefix_equal,
        compared_pre_pause_checkpoint=before_pause, source_directory=str(SOURCE),
        limitation='Independent GPU executions already differ before pause; final trajectories are not claimed bitwise identical.',
        primary_evaluation='mean_residual', final_updates_per_formal_run=17500))


if __name__ == '__main__':
    main()
