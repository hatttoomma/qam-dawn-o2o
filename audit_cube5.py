"""Verify all task-specific reward relabelings on the same cube-double play data."""
import argparse
import json
from pathlib import Path

import numpy as np
import run as common

RUNS = common.ROOT / 'runs/cube5'


def task_manifest(env, ds):
    return dict(environment=common.ENV, gym_environment=env.spec.id,
        reward_task_id=int(env.unwrapped._reward_task_id), dataset_size=ds.size,
        target_cube_xyzs=env.unwrapped._data.mocap_pos.tolist(),
        observation_shape=list(ds['observations'].shape), action_shape=list(ds['actions'].shape),
        max_episode_steps=env.spec.max_episode_steps,
        rewards_hash=common.tree_hash(ds['rewards']), masks_hash=common.tree_hash(ds['masks']),
        observations_hash=common.tree_hash(ds['observations']), actions_hash=common.tree_hash(ds['actions']))


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    tasks, shared = [], None
    expected = {
        'cube-double-play-v0.npz':'a73d1a33d029cedb8bc170ef94791ec585fa2d9450096f4f2a02b8cfbcf608c9',
        'cube-double-play-v0-val.npz':'b1fcdf4bd40750351a58d0d491d6be198366ce898f0c6a2e4cb5db331966013e'}
    hashes = {n:common.file_hash(common.ROOT/'data'/n) for n in expected}
    assert hashes == expected
    for task in range(1, 6):
        common.ENV = f'cube-double-play-singletask-task{task}-v0'
        env, ds = common.make_data(argparse.Namespace(data=str(common.ROOT/'data')))
        manifest = task_manifest(env, ds)
        assert manifest['reward_task_id'] == task
        assert manifest['gym_environment'] == f'cube-double-singletask-task{task}-v0'
        assert ds.size == 1000000 and ds['observations'].shape == (1000000,37)
        assert ds['actions'].shape == (1000000,5)
        assert set(np.unique(ds['rewards'])).issubset({-2.,-1.,0.})
        transition_hashes = {k:common.tree_hash(ds[k]) for k in ('observations','actions','next_observations','terminals')}
        if shared is None: shared = transition_hashes
        else: assert shared == transition_hashes
        tasks.append(dict(manifest=manifest, reward_counts={str(float(x)):int(n)
            for x,n in zip(*np.unique(ds['rewards'],return_counts=True))}))
        env.close()
        del ds
    assert len({json.dumps(t['manifest']['target_cube_xyzs']) for t in tasks}) == 5
    assert len({t['manifest']['rewards_hash'] for t in tasks}) == 5
    result = dict(status='passed', tasks=tasks, dataset_hashes=hashes,
        shared_transition_hashes=shared, distinct_task_rewards=True)
    common.atomic_json(RUNS/'TASK_AUDIT_PASSED.json', result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
