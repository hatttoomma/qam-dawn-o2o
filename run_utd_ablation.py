"""Expose only update-to-data ratio on the existing plain-TD residual loop."""
import argparse
import hashlib
import inspect
import difflib
import fcntl
import json
from pathlib import Path
import sys

import jax.numpy as jnp
import flax

import run as common
import dawn_agent as implementation
import td_diagnostics

RUNS = common.ROOT / 'runs/utd_ablation'


def hard_backup(reward, bootstrap_discount, next_qs, alpha, next_logp):
    return reward + bootstrap_discount * jnp.min(next_qs, axis=0)


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--task', type=int, choices=[1, 2], required=True)
    extra.add_argument('--td-target', choices=['soft', 'hard'], required=True)
    extra.add_argument('--utd', type=float, choices=[.25, 1.], required=True)
    selected, rest = extra.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    args = common.parse()
    assert args.stage == 'warm' and args.seed == 0 and args.offline_steps == 500000
    args.task, args.td_target, args.utd = selected.task, selected.td_target, selected.utd
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
    original_loop = inspect.getsource(common.residual)
    old = 'target_updates=int((step-args.warmup)*.25)'
    new = 'target_updates=int((step-args.warmup)*args.utd)'
    assert original_loop.count(old) == 1
    adapted_loop = original_loop.replace(old, new)
    assert adapted_loop.replace(new, old) == original_loop
    intervention = dict(status='passed', old=old, new=new, utd=args.utd,
        original_loop_sha256=hashlib.sha256(original_loop.encode()).hexdigest(),
        adapted_loop_sha256=hashlib.sha256(adapted_loop.encode()).hexdigest(),
        diff=''.join(difflib.unified_diff(original_loop.splitlines(True), adapted_loop.splitlines(True),
                                       fromfile='original residual loop', tofile='UTD-parameterized residual loop')))
    common.atomic_json(out/'utd_intervention.json', intervention)
    exec(compile(adapted_loop, '<verified UTD-only residual loop>', 'exec'), common.__dict__)

    original_data, original_factory = common.make_data, common.DawnAgent
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
            agent = original_factory.create(qam, obs, seed, warm)
            actual = dict(task=args.task, td_target=args.td_target, inherited_Q_and_target=True,
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
        assert done['updates'] == int((args.online_steps-args.warmup)*args.utd)
        offline = json.loads((reference / 'offline/DONE.json').read_text())
        assert done['initial_hashes']['critic'] == offline['q_hash']
        assert done['initial_hashes']['target'] == offline['target_q_hash']
        assert done['final_flow_hash'] == offline['flow_hash']
        common.atomic_json(out / 'CHECKS_PASSED.json', dict(status='passed', steps=done['steps'],
            updates=done['updates'], td_target=args.td_target, inherited_correct_task_Q=True,
            frozen_base=True, original_actor_loss_and_alpha_update=True, original_loop_except_utd=True, utd=args.utd))
    except BaseException as e:
        common.atomic_json(out / 'FAILED.json', dict(type=type(e).__name__, message=str(e)));raise


if __name__ == '__main__': main()
