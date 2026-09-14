"""Antmaze adapter around the historical UTD .25 residual collection/update loop."""
import fcntl
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(os.environ.get('ANTMAZE_ROOT', Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / 'official'))
sys.path.insert(1, str(ROOT / 'legacy'))
import run as common
import paired_evaluation
import actor_input_agent as implementation
from agents.qam import QAMAgent, get_config
import jax
import jax.numpy as jnp
import numpy as np

SMOKE = os.environ.get('ANTMAZE_SMOKE') == '1'
OUT = Path(os.environ['ANTMAZE_OUT'])
NATIVE = Path(os.environ['ANTMAZE_NATIVE'])
OUT.mkdir(parents=True, exist_ok=True)
lock = (OUT / 'run.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
common.ENV = 'antmaze-large-navigate-singletask-task1-v0'
common.HORIZON, common.GAMMA = 1, .99
common.GRID = (0, 80, 120, 200) if SMOKE else (0, 80000, 100000, 120000, 150000)
args = SimpleNamespace(stage='warm', out=str(OUT), data='/root/.ogbench/data', seed=0,
    offline_steps=10 if SMOKE else 500000, online_steps=200 if SMOKE else 150000,
    offline_checkpoint=str(NATIVE / 'offline_500k.pkl'), eval_episodes=2 if SMOKE else 100,
    final_episodes=2 if SMOKE else 100, warmup=80 if SMOKE else 80000, dawn_batch=256,
    td_target='naive', actor_input='state+base_action', utd=.25)
native_done = json.loads((NATIVE / 'DONE.json').read_text())
assert native_done['status'] == 'passed'
assert native_done['offline_updates'] == args.offline_steps
assert native_done['online_env_steps'] == (12 if SMOKE else 50000)
assert native_done['online_updates'] == (8 if SMOKE else 45001)
offline = json.loads((NATIVE / 'offline_checkpoint.json').read_text())
assert common.file_hash(args.offline_checkpoint) == offline['sha256']
manifest = json.loads((ROOT / 'source_manifest.json').read_text())
for relative, expected in manifest['files'].items():
    assert common.file_hash(ROOT / relative) == expected, relative
common.atomic_json(OUT / 'requested_config.json', vars(args))


def template(obs, action, seed):
    cfg = get_config()
    cfg.horizon_length, cfg.action_chunking = 1, True
    cfg.inv_temp, cfg.fql_alpha, cfg.edit_scale = 10., 0., 0.
    return QAMAgent.create(seed, obs, action, cfg)


def hard_backup(reward, bootstrap_discount, next_qs, alpha, next_logp):
    return reward + bootstrap_discount * jnp.min(next_qs, axis=0)


original_data, original_eval, original_save = common.make_data, paired_evaluation.evaluate, common.save_checkpoint


def load_data(config):
    env, ds = original_data(config)
    assert env.unwrapped._reward_task_id == 1
    native_manifest = json.loads((NATIVE / 'dataset_manifest.json').read_text())
    assert ds.size == native_manifest['dataset_size']
    for name, spec in native_manifest['files'].items():
        assert common.file_hash(Path(config.data) / name) == spec['sha256']
    common.atomic_json(OUT / 'dataset_manifest.json', native_manifest)
    return env, ds


class Factory:
    @staticmethod
    def create(qam, obs, seed, warm):
        assert warm
        agent = implementation.ActorInputAgent.create(qam, obs, seed, warm).replace(condition_on_base=True)
        assert common.flow_hash(qam) == offline['flow_hash']
        assert common.tree_hash(agent.critic.params) == offline['q_hash']
        assert common.tree_hash(agent.target_params) == offline['target_q_hash']
        assert agent.action_dim == 8 and qam.config['horizon_length'] == 1
        features = implementation.actor_input(obs, jnp.ones(8), True)
        np.testing.assert_array_equal(features[:obs.shape[-1]], obs)
        np.testing.assert_array_equal(features[obs.shape[-1]:], np.ones(8))
        def objective(base):
            mu, std = agent.actor.apply_fn({'params':agent.actor.params},
                implementation.actor_input(obs, base, True))
            return mu.sum() + std.sum()
        assert float(jnp.linalg.norm(jax.grad(objective)(jnp.ones(8)))) > 0
        common.atomic_json(OUT / 'actual_agent_config.json', dict(
            inherited_Q_and_target=True, fresh_critic_optimizer=True, td_target='naive',
            actor_input='state+base_action', observation_dim=int(obs.shape[-1]),
            actor_input_dim=int(obs.shape[-1])+8, action_dim=8, horizon=1,
            residual_scale=.1, utd=.25, utd_unit='primitive_environment_step',
            batch_size=256, warmup=args.warmup, online_steps=args.online_steps,
            replay='online_only_uniform_growing_chunk_buffer', critic_ensemble=10,
            critic_hidden_dims=[512]*4, critic_activation='gelu', critic_layer_norm=True,
            actor_hidden_dims=[256]*3, actor_activation='relu', gaussian_output=True,
            target_aggregation='minimum', actor_Q_aggregation='minimum',
            learning_rate=1e-4, gradient_clip_norm=50., target_tau=.01, discount=.99,
            initial_alpha=.01, target_entropy=-8, actor_entropy_enabled=True,
            automatic_alpha_enabled=True, base_feature_gradient_nonzero=True,
            frozen_base_inv_temp=10., seed=0, source_files=manifest['files']))
        return agent


def evaluate(qam, folder, step, config, **kwargs):
    if step == args.warmup:
        # Same policy throughout warmup; retain its checkpoint without a duplicate rollout.
        return json.loads((OUT / f'eval_000000_{args.eval_episodes:03d}.json').read_text())
    rng_before = common.tree_hash(np.random.get_state())
    result = original_eval(qam, folder, step, config, **kwargs)
    if step == 0:
        native_base = json.loads((NATIVE / 'fixed_eval' / f'eval_000000_{args.eval_episodes:03d}.json').read_text())
        assert result['records'] == native_base['records'], 'Base policy or paired evaluation differs'
        common.atomic_json(OUT / 'INITIAL_EVAL_MATCHED.json', dict(status='passed',
            episodes=args.eval_episodes, native_offline_success=result['success']))
    elif not kwargs.get('deterministic', False):
        original_eval(qam, folder, step, config, **dict(kwargs, deterministic=True,
            suffix='_mean_residual', episodes=args.eval_episodes))
    assert common.tree_hash(np.random.get_state()) == rng_before
    return result


def save(path, agent, **payload):
    expected = max(0, payload['step'] - args.warmup) // 4
    assert int(agent.updates) == expected
    original_save(path, agent, **payload)
    if Path(path).name == 'latest.pkl':
        original_save(OUT / f'checkpoint_{payload["step"]:06d}.pkl', agent,
            step=payload['step'], updates=int(agent.updates))


r, d = jnp.array([-3., 0., -5.]), jnp.array([0., .99, .99])
qs, lp, alpha = jnp.array([[-8., -7., -6.], [-9., -4., -8.]]), jnp.array([3., -4., 2.]), .02
np.testing.assert_allclose(implementation.soft_backup(r,d,qs,alpha,lp)-hard_backup(r,d,qs,alpha,lp),
    -d*alpha*lp, atol=1e-6)
np.testing.assert_array_equal(hard_backup(r,d,qs,alpha,lp), hard_backup(r,d,qs,alpha*100,lp*10))
implementation.soft_backup = hard_backup
common.qam_template, common.DawnAgent = template, Factory
common.reset_train = lambda env, episode, seed: paired_evaluation.deterministic_reset(
    env, 100000 + seed*10000 + episode)[0]
common.make_data, common.evaluate, common.save_checkpoint = load_data, evaluate, save
try:
    common.residual(args)
    done = json.loads((OUT / 'DONE.json').read_text())
    assert done['steps'] == args.online_steps
    assert done['updates'] == (args.online_steps - args.warmup)//4
    assert done['final_flow_hash'] == offline['flow_hash']
    assert common.file_hash(OUT / 'final.pkl') == done['checkpoint_sha256']
    common.atomic_json(OUT / 'CHECKS_PASSED.json', dict(status='passed',
        unchanged_historical_collection_and_update_loop=True, native_gate_passed=True,
        offline_checkpoint_matched=True, hard_TD_verified=True, frozen_base_verified=True,
        steps=done['steps'], updates=done['updates']))
except BaseException as exc:
    common.atomic_json(OUT / 'FAILED.json', dict(type=type(exc).__name__,message=str(exc)))
    raise
