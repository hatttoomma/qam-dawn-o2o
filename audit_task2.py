"""Check task selection and official reward relabeling on identical play data."""
import argparse
import json
from pathlib import Path

import run as common
import numpy as np


def main():
    out = common.ROOT/'runs/task2'
    out.mkdir(parents=True, exist_ok=True)
    datasets, manifests = [], []
    for task in (1,2):
        common.ENV = f'cube-double-play-singletask-task{task}-v0'
        env, ds = common.make_data(argparse.Namespace(data=str(common.ROOT/'data')))
        assert env.unwrapped._reward_task_id == task
        assert env.spec.id == f'cube-double-singletask-task{task}-v0'
        assert ds.size == 1000000
        assert ds['observations'].shape == (1000000,37)
        assert ds['actions'].shape == (1000000,5)
        manifests.append(dict(environment=common.ENV, reward_task_id=task,
            target_cube_xyzs=env.unwrapped._data.mocap_pos.tolist(),
            rewards_hash=common.tree_hash(ds['rewards']), masks_hash=common.tree_hash(ds['masks']),
            reward_counts={str(float(x)):int(n) for x,n in zip(*np.unique(ds['rewards'],return_counts=True))}))
        datasets.append(ds)
        env.close()
    a,b=datasets
    for key in ('observations','actions','next_observations','terminals'):
        np.testing.assert_array_equal(a[key],b[key],err_msg=key)
    assert manifests[0]['target_cube_xyzs'] != manifests[1]['target_cube_xyzs']
    assert manifests[0]['rewards_hash'] != manifests[1]['rewards_hash']
    expected = {
        'cube-double-play-v0.npz':'a73d1a33d029cedb8bc170ef94791ec585fa2d9450096f4f2a02b8cfbcf608c9',
        'cube-double-play-v0-val.npz':'b1fcdf4bd40750351a58d0d491d6be198366ce898f0c6a2e4cb5db331966013e'}
    hashes={name:common.file_hash(common.ROOT/'data'/name) for name in expected}
    assert hashes == expected
    common.atomic_json(out/'TASK_AUDIT_PASSED.json', dict(status='passed',
        tasks=manifests, dataset_hashes=hashes, state_action_transitions_identical=True,
        different_reward_rows=int(np.count_nonzero(a['rewards']!=b['rewards'])),
        different_mask_rows=int(np.count_nonzero(a['masks']!=b['masks']))))
    print(json.dumps(manifests,indent=2))


if __name__=='__main__':
    main()
