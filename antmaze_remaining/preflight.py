"""Verify source/data identity and the four requested task registrations."""
import hashlib
import json
from pathlib import Path
import ogbench
import numpy as np

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / 'source_manifest.json').read_text())
for relative, expected in manifest['files'].items():
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
prior = ROOT.parent / 'qam_antmaze/results/native/dataset_manifest.json'
dataset = json.loads(prior.read_text())
for name, spec in dataset['files'].items():
    assert hashlib.sha256((Path('/root/.ogbench/data') / name).read_bytes()).hexdigest() == spec['sha256']
tasks = []
for task in (2, 3, 4, 5):
    name = f'antmaze-large-navigate-singletask-task{task}-v0'
    env = ogbench.make_env_and_datasets(name, env_only=True)
    try:
        obs, _ = env.reset(seed=0)
        assert env.unwrapped._reward_task_id == task
        assert obs.shape == (29,) and env.action_space.shape == (8,)
        assert np.isfinite(obs).all() and env.spec.max_episode_steps == 1000
        tasks.append(dict(task_id=task, environment=name, gym_id=env.spec.id,
                          observation_dim=29, action_dim=8, max_episode_steps=1000))
    finally:
        env.close()
result = dict(status='passed', official_and_legacy_sources_verified=True,
              dataset_checksums_verified=True, tasks=tasks)
(ROOT / 'results/preflight.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
