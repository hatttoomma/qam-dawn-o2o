"""Select task1/task2 and change only residual scale in the original DAWN loop."""
import argparse
import fcntl
import json
from pathlib import Path
import sys

import run as common

RUNS = common.ROOT / 'runs/scale_extremes'


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--task', type=int, choices=[1, 2], required=True)
    extra.add_argument('--residual-scale', type=float, choices=[.01, .1, 1.], required=True)
    selected, rest = extra.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    args = common.parse()
    assert args.stage == 'warm'
    args.task = selected.task
    args.residual_scale = selected.residual_scale
    args.environment = f'cube-double-play-singletask-task{args.task}-v0'
    common.ENV = args.environment
    if args.task == 2: args.task2_offline_from_scratch = True
    reference = common.ROOT / ('runs' if args.task == 1 else 'runs/task2')
    if '--offline-checkpoint' not in rest:
        args.offline_checkpoint = str(reference / 'offline/final.pkl')
    assert Path(args.offline_checkpoint).resolve() == (reference / 'offline/final.pkl').resolve()
    out = Path(args.out)
    assert out.resolve().is_relative_to(RUNS.resolve())
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    requested = dict(vars(args), offline_sha256=common.file_hash(args.offline_checkpoint))
    assert requested['offline_sha256'] == json.loads((reference / 'offline/DONE.json').read_text())['checkpoint_sha256']
    path = out / 'scale_config.json'
    if path.exists(): assert json.loads(path.read_text()) == requested, 'Resume settings differ'
    else: common.atomic_json(path, requested)
    original_data = common.make_data

    def checked_data(config):
        env, ds = original_data(config)
        assert env.unwrapped._reward_task_id == args.task
        assert env.spec.id == f'cube-double-singletask-task{args.task}-v0'
        manifest = dict(environment=args.environment, gym_environment=env.spec.id,
            reward_task_id=int(env.unwrapped._reward_task_id), dataset_size=ds.size,
            target_cube_xyzs=env.unwrapped._data.mocap_pos.tolist(),
            observation_shape=list(ds['observations'].shape), action_shape=list(ds['actions'].shape),
            max_episode_steps=env.spec.max_episode_steps,
            rewards_hash=common.tree_hash(ds['rewards']), masks_hash=common.tree_hash(ds['masks']),
            observations_hash=common.tree_hash(ds['observations']), actions_hash=common.tree_hash(ds['actions']))
        audit = json.loads((common.ROOT / 'runs/task2/TASK_AUDIT_PASSED.json').read_text())
        expected = audit['tasks'][args.task - 1]
        assert audit['status'] == 'passed'
        for k in ['environment', 'reward_task_id', 'target_cube_xyzs', 'rewards_hash', 'masks_hash']:
            assert manifest[k] == expected[k], k
        if args.task == 2:
            assert manifest == json.loads((reference / 'offline/task_manifest.json').read_text())
        p = out / 'task_manifest.json'
        if p.exists(): assert json.loads(p.read_text()) == manifest
        else: common.atomic_json(p, manifest)
        return env, ds

    original_agent = common.DawnAgent

    class ScaleFactory:
        @staticmethod
        def create(qam, obs, seed, warm):
            assert warm
            agent = original_agent.create(qam, obs, seed, warm).replace(res_scale=args.residual_scale)
            actual = dict(task=args.task, residual_scale=agent.res_scale, tau=agent.tau,
                action_dim=agent.action_dim, inherited_critic=True,
                actor_hash=common.tree_hash(agent.actor.params),
                critic_hash=common.tree_hash(agent.critic.params),
                target_hash=common.tree_hash(agent.target_params))
            p = out / 'actual_agent_config.json'
            if p.exists(): assert json.loads(p.read_text()) == actual
            else: common.atomic_json(p, actual)
            return agent

    common.make_data = checked_data
    common.DawnAgent = ScaleFactory
    try: common.residual(args)
    except BaseException as e:
        common.atomic_json(out / 'FAILED.json', dict(type=type(e).__name__, message=str(e)))
        raise


if __name__ == '__main__': main()
