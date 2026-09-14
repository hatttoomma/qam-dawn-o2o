"""Validate all inputs and the actual changed batch construction before training."""
import hashlib
import json
import os
from pathlib import Path
import sys

import jax
import numpy as np
import ogbench

import test_replay

ROOT = Path(os.environ.get('ANTMAZE_ROOT', Path(__file__).resolve().parent))
OUT = ROOT / 'results'
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()

manifest = json.loads((ROOT / 'source_manifest.json').read_text())
for relative, expected in manifest['files'].items():
    assert digest(ROOT / relative) == expected, relative
parent = json.loads((ROOT.parent / 'qam_antmaze/source_manifest.json').read_text())
for relative, expected in parent['files'].items():
    if relative != 'legacy/run.py':
        assert manifest['files'][relative] == expected, relative
assert manifest['parent_run_sha256'] == parent['files']['legacy/run.py']
inputs = json.loads((ROOT/'inputs.json').read_text())
tasks=[]
for task in range(1,6):
    spec=inputs[str(task)]
    native=Path(spec['native'])
    done=json.loads((native/'DONE.json').read_text())
    assert (done['offline_updates'],done['online_env_steps'],done['online_updates']) == (500000,50000,45001)
    meta=json.loads((native/'offline_checkpoint.json').read_text())
    assert digest(native/'offline_500k.pkl') == meta['sha256'] == spec['offline_sha256']
    ds=json.loads((native/'dataset_manifest.json').read_text())
    assert ds['task_id'] == task and ds['dataset_size'] == 1000000
    if task == 1:
        for name, info in ds['files'].items():
            assert digest(Path('/root/.ogbench/data')/name) == info['sha256']
    env=ogbench.make_env_and_datasets(ds['environment'],env_only=True)
    try:
        obs,_=env.reset(seed=0)
        assert env.unwrapped._reward_task_id == task
        assert obs.shape == (29,) and env.action_space.shape == (8,)
        assert env.spec.max_episode_steps == 1000 and np.isfinite(obs).all()
    finally:
        env.close()
    tasks.append(dict(task_id=task,offline_sha256=meta['sha256'],dataset_size=ds['dataset_size']))
result=dict(status='passed', source_checksums_verified=True, official_and_agent_unchanged=True,
            tasks=tasks,sampler_checks=test_replay.main(),devices=[str(x) for x in jax.devices()])
assert any(x.platform == 'gpu' for x in jax.devices())
(OUT/'preflight.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
