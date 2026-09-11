"""Select task2 while reusing the unchanged original QAM/DAWN training loops."""
import fcntl
import json
from pathlib import Path
import sys

import run as common

ENV = 'cube-double-play-singletask-task2-v0'
RUNS = common.ROOT / 'runs/task2'
common.ENV = ENV
original_make_data = common.make_data


def task2_data(args):
    env, ds = original_make_data(args)
    assert env.unwrapped._reward_task_id == 2
    assert env.spec.id == 'cube-double-singletask-task2-v0'
    manifest = dict(environment=ENV, gym_environment=env.spec.id,
        reward_task_id=int(env.unwrapped._reward_task_id), dataset_size=ds.size,
        target_cube_xyzs=env.unwrapped._data.mocap_pos.tolist(),
        observation_shape=list(ds['observations'].shape), action_shape=list(ds['actions'].shape),
        max_episode_steps=env.spec.max_episode_steps,
        rewards_hash=common.tree_hash(ds['rewards']), masks_hash=common.tree_hash(ds['masks']),
        observations_hash=common.tree_hash(ds['observations']), actions_hash=common.tree_hash(ds['actions']))
    out = Path(args.out)
    path = out / 'task_manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    else:
        common.atomic_json(path, manifest)
    if args.stage != 'offline':
        offline_manifest = json.loads((Path(args.offline_checkpoint).parent/'task_manifest.json').read_text())
        assert offline_manifest == manifest, 'Online task/data must match the task2 offline checkpoint'
    return env, ds


common.make_data = task2_data


if __name__ == '__main__':
    args = common.parse()
    assert args.stage in ('offline', 'native', 'warm')
    if '--offline-checkpoint' not in sys.argv:
        args.offline_checkpoint = str(RUNS/'offline/final.pkl')
    args.environment = ENV
    args.task2_offline_from_scratch = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    assert out.resolve().is_relative_to(RUNS.resolve()), 'Keep task2 separate from historical runs'
    lock = (out/'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        {'offline': common.offline, 'native': common.native, 'warm': common.residual}[args.stage](args)
    except BaseException as e:
        common.atomic_json(out/'FAILED.json', dict(type=type(e).__name__, message=str(e)))
        raise
