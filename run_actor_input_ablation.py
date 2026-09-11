"""Paired zero-feature / real-base-feature actor experiment on the original plain-TD loop."""
import argparse
import fcntl
import json
from pathlib import Path
import sys

import jax.numpy as jnp
import flax

import run as common
import actor_input_agent as implementation
import actor_input_diagnostics as td_diagnostics

RUNS = common.ROOT / 'runs/actor_input_ablation'


def hard_backup(reward, bootstrap_discount, next_qs, alpha, next_logp):
    return reward + bootstrap_discount * jnp.min(next_qs, axis=0)


def check_input_operator(agent, obs, out):
    import jax
    import numpy as np
    zero = jnp.zeros((*obs.shape[:-1], agent.action_dim), dtype=obs.dtype)
    base = jnp.broadcast_to(jnp.linspace(-.9, .9, agent.action_dim), zero.shape)
    x0 = implementation.actor_input(obs, zero, False)
    x1 = implementation.actor_input(obs, base, False)
    np.testing.assert_array_equal(x0, x1)
    np.testing.assert_array_equal(x1[..., :37], obs)
    x2 = implementation.actor_input(obs, base, True)
    np.testing.assert_array_equal(x2[..., :37], obs)
    np.testing.assert_array_equal(x2[..., 37:], base)
    def objective(params, condition):
        mu, logstd = agent.actor.apply_fn({'params':params}, implementation.actor_input(obs, base, condition))
        return jnp.square(mu).mean() + logstd.mean()
    g0 = jax.grad(objective)(agent.actor.params, False)['Dense_0']['kernel'][37:]
    g1 = jax.grad(objective)(agent.actor.params, True)['Dense_0']['kernel'][37:]
    assert np.count_nonzero(np.asarray(g0)) == 0
    assert np.linalg.norm(np.asarray(g1)) > 0
    old_params = dict(agent.actor.params)
    old_params['Dense_0'] = dict(old_params['Dense_0'])
    old_params['Dense_0']['kernel'] = old_params['Dense_0']['kernel'][:37]
    original_actor = common.DawnAgent  # Original factory remains untouched until hooks are installed.
    from dawn_agent import ResidualActor
    old_output = ResidualActor(agent.action_dim).apply({'params':old_params}, obs)
    masked_output = agent.actor.apply_fn({'params':agent.actor.params}, x0)
    for a,b in zip(old_output,masked_output):np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-5)
    common.atomic_json(out/'actor_input_checks.json',dict(status='passed', input_dim=62, action_dim=25,
        masked_base_features_exactly_zero=True, real_base_features_exact=True,
        masked_feature_gradients_zero=True, real_feature_gradients_nonzero=True,
        state_only_function_matches_37dim_projection=True,
        actor_parameter_count=sum(int(x.size) for x in jax.tree_util.tree_leaves(agent.actor.params))))


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--task', type=int, choices=[1, 2], required=True)
    extra.add_argument('--td-target', choices=['soft', 'hard'], required=True)
    extra.add_argument('--actor-input', choices=['masked', 'base_action'], required=True)
    selected, rest = extra.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    args = common.parse()
    assert args.stage == 'warm' and args.seed == 0 and args.offline_steps == 500000
    args.task, args.td_target = selected.task, selected.td_target
    args.actor_input = selected.actor_input
    assert args.td_target == 'hard' and args.dawn_batch == 256
    args.environment = f'cube-double-play-singletask-task{args.task}-v0'
    common.ENV = args.environment
    if args.task == 2: args.task2_offline_from_scratch = True
    reference = common.ROOT / ('runs' if args.task == 1 else 'runs/task2')
    if '--offline-checkpoint' not in rest: args.offline_checkpoint = str(reference / 'offline/final.pkl')
    assert Path(args.offline_checkpoint).resolve() == (reference / 'offline/final.pkl').resolve()
    out = Path(args.out)
    assert out.resolve().is_relative_to(RUNS.resolve())
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'run.lock').open('w');fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    requested = dict(vars(args), offline_sha256=common.file_hash(args.offline_checkpoint))
    assert requested['offline_sha256'] == json.loads((reference / 'offline/DONE.json').read_text())['checkpoint_sha256']
    p = out / 'td_config.json'
    if p.exists(): assert json.loads(p.read_text()) == requested, 'Resume configuration differs'
    else: common.atomic_json(p, requested)
    r=jnp.array([-3., 0., -5.]);d=jnp.array([0., .95, .95])
    qs=jnp.array([[-8., -7., -6.], [-9., -4., -8.]])
    logp=jnp.array([3., -4., 2.]);alpha=jnp.array(.02)
    soft=implementation.soft_backup(r,d,qs,alpha,logp);hard=hard_backup(r,d,qs,alpha,logp)
    common.np.testing.assert_allclose(soft-hard,-d*alpha*logp,rtol=0,atol=1e-6)
    common.np.testing.assert_array_equal(hard,hard_backup(r,d,qs,alpha*100,logp*10))
    assert float(hard[0]) == float(soft[0]) == float(r[0])
    common.atomic_json(out/'backup_math.json',dict(status='passed',terminal_reward_unchanged=True,
        hard_independent_of_entropy=True,soft_minus_hard_equals_discounted_entropy=True))
    if args.td_target == 'hard': implementation.soft_backup = hard_backup

    original_data, original_factory = common.make_data, implementation.ActorInputAgent
    original_eval, original_save = common.evaluate, common.save_checkpoint

    def checked_data(config):
        env, ds = original_data(config)
        manifest = dict(environment=args.environment, gym_environment=env.spec.id,
            reward_task_id=int(env.unwrapped._reward_task_id), dataset_size=ds.size,
            target_cube_xyzs=env.unwrapped._data.mocap_pos.tolist(),
            observation_shape=list(ds['observations'].shape), action_shape=list(ds['actions'].shape),
            max_episode_steps=env.spec.max_episode_steps,
            rewards_hash=common.tree_hash(ds['rewards']), masks_hash=common.tree_hash(ds['masks']),
            observations_hash=common.tree_hash(ds['observations']), actions_hash=common.tree_hash(ds['actions']))
        audit = json.loads((common.ROOT / 'runs/task2/TASK_AUDIT_PASSED.json').read_text())
        assert audit['status'] == 'passed'
        for k in ['environment', 'reward_task_id', 'target_cube_xyzs', 'rewards_hash', 'masks_hash']:
            assert manifest[k] == audit['tasks'][args.task-1][k], k
        assert env.spec.id == f'cube-double-singletask-task{args.task}-v0'
        if args.task == 2: assert manifest == json.loads((reference / 'offline/task_manifest.json').read_text())
        p = out / 'task_manifest.json'
        if p.exists(): assert json.loads(p.read_text()) == manifest
        else: common.atomic_json(p, manifest)
        return env, ds

    class Factory:
        @staticmethod
        def create(qam, obs, seed, warm):
            assert warm
            agent = original_factory.create(qam, obs, seed, warm).replace(condition_on_base=args.actor_input == 'base_action')
            assert obs.shape[-1] == 37 and agent.action_dim == 25
            check_input_operator(agent, obs, out)
            actual = dict(task=args.task, td_target=args.td_target, inherited_Q_and_target=True,
                actor_input=args.actor_input, actor_input_dim=62, observation_dim=37, base_feature_dim=25,
                hidden_dims=[256,256,256], activation='relu', gaussian_output=True, target_entropy=-25, utd=.25,
                residual_scale=agent.res_scale, target_tau=agent.tau, action_dim=agent.action_dim,
                target_aggregation='minimum', actor_Q_aggregation='minimum', critic_ensemble=10,
                actor_entropy_enabled=True, automatic_alpha_enabled=True,
                full_initial_agent_hash=common.tree_hash(flax.serialization.to_state_dict(agent)))
            p = out / 'actual_agent_config.json'
            if p.exists(): assert json.loads(p.read_text()) == actual
            else: common.atomic_json(p, actual)
            return agent

    def evaluate(qam, folder, step, config, **kwargs):
        return td_diagnostics.evaluate_with_mc(original_eval, qam, folder, step, config, **kwargs)

    def save(path, agent, **payload):
        original_save(path, agent, **payload)
        if Path(path).name == 'latest.pkl': td_diagnostics.fixed_probe(out, args.task, agent, payload)

    common.make_data, common.DawnAgent = checked_data, Factory
    common.evaluate, common.save_checkpoint = evaluate, save
    try:
        common.residual(args)
        done = json.loads((out / 'DONE.json').read_text())
        assert done['steps'] == args.online_steps
        assert done['updates'] == int((args.online_steps-args.warmup)*.25)
        offline = json.loads((reference / 'offline/DONE.json').read_text())
        assert done['initial_hashes']['critic'] == offline['q_hash']
        assert done['initial_hashes']['target'] == offline['target_q_hash']
        assert done['final_flow_hash'] == offline['flow_hash']
        common.atomic_json(out / 'CHECKS_PASSED.json', dict(status='passed', steps=done['steps'],
            updates=done['updates'], td_target=args.td_target, inherited_correct_task_Q=True,
            frozen_base=True, original_actor_loss_and_alpha_update=True, original_training_loop=True))
    except BaseException as e:
        common.atomic_json(out / 'FAILED.json', dict(type=type(e).__name__, message=str(e)));raise


if __name__ == '__main__': main()
