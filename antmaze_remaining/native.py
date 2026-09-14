"""Run the unchanged official QAM main with local logging and evaluation hooks."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(os.environ.get('ANTMAZE_ROOT', Path(__file__).resolve().parent))
OUT = Path(os.environ['ANTMAZE_OUT'])
SMOKE = os.environ.get('ANTMAZE_SMOKE') == '1'
TASK = int(os.environ['ANTMAZE_TASK'])
assert TASK in (2, 3, 4, 5)
OUT.mkdir(parents=True, exist_ok=True)
lock = (OUT / 'run.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
sys.path.insert(0, str(ROOT / 'official'))
sys.path.insert(1, str(ROOT / 'legacy'))
os.environ.setdefault('MUJOCO_GL', 'egl')

import main as official
import run as common
import paired_evaluation
import numpy as np
import jax
from agents.qam import QAMAgent, get_config

START = time.time()
LAST_STEP = 0
OFFLINE = 10 if SMOKE else 500000
ONLINE = 12 if SMOKE else 50000
START_TRAINING = 5 if SMOKE else 5000
EVAL_EPISODES = 1 if SMOKE else 50
FIXED_EPISODES = 2 if SMOKE else 100
ENV = f'antmaze-large-navigate-singletask-task{TASK}-v0'
common.ENV, common.HORIZON, common.GAMMA = ENV, 1, .99
common_args = SimpleNamespace(seed=0, eval_episodes=FIXED_EPISODES)


def write(name, value):
    common.atomic_json(OUT / name, value)


def verify_sources():
    manifest = json.loads((ROOT / 'source_manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        assert common.file_hash(ROOT / relative) == expected, relative
    write('source_verification.json', dict(status='passed', **manifest))


class AgentFactory:
    @staticmethod
    def create(seed, obs, action, cfg):
        expected = get_config()
        expected.horizon_length = 1
        expected.action_chunking = True
        expected.inv_temp, expected.fql_alpha, expected.edit_scale = 10., 0., 0.
        assert cfg.to_dict() == expected.to_dict(), (cfg, expected)
        assert seed == 0
        agent = QAMAgent.create(seed, obs, action, cfg)
        names = list(agent.network.params)
        assert not any('edit' in k or 'one_step_actor' in k for k in names)
        write('actual_agent_config.json', dict(agent_config=dict(agent.config), modules=names,
            observation_dim=obs.shape[-1], action_dim=action.shape[-1], horizon=1,
            seed=seed, task_id=TASK, vanilla_QAM=True, source_commit='2726d767c9a0a7a46d49693f0391f73dc2cf58ac'))
        return agent


original_data = official.make_env_and_datasets


def load_data(*args, **kwargs):
    result = original_data(*args, **kwargs)
    env, _, ds, val = result
    assert env.unwrapped._reward_task_id == TASK
    assert ds['observations'].shape[-1] == env.observation_space.shape[-1]
    assert ds['actions'].shape[-1] == 8
    assert np.isfinite(ds['rewards']).all()
    write('dataset_manifest.json', dict(environment=ENV, gym_environment=env.spec.id,
        task_id=int(env.unwrapped._reward_task_id), max_episode_steps=env.spec.max_episode_steps,
        dataset_size=ds.size, validation_size=val.size,
        observation_shape=list(ds['observations'].shape), action_shape=list(ds['actions'].shape),
        reward_range=[float(ds['rewards'].min()), float(ds['rewards'].max())],
        masks_range=[float(ds['masks'].min()), float(ds['masks'].max())],
        files={p.name: dict(bytes=p.stat().st_size, sha256=common.file_hash(p))
            for p in Path('/root/.ogbench/data').glob('antmaze-large-navigate-v0*.npz')}))
    return result


original_log = official.LoggingHelper.log


def log(self, data, prefix, step):
    global LAST_STEP
    LAST_STEP = step
    if prefix in ('offline_agent', 'online_agent'):
        assert all(np.isfinite(np.asarray(v)).all() for v in data.values()), data
    original_log(self, data, prefix, step)
    if prefix != 'env' or step % 1000 == 0:
        env_step = max(0, step - OFFLINE)
        progress = dict(stage='offline' if step <= OFFLINE else 'native_online',
            offline_updates=min(step, OFFLINE), online_env_steps=env_step,
            online_updates=max(0, env_step - START_TRAINING + 1),
            total_step=step, elapsed_seconds=time.time()-START,
            utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        write('progress.json', progress)
        if prefix != 'env':
            common.log(OUT / 'metrics.jsonl', dict(prefix=prefix, **progress,
                metrics={k:float(np.asarray(v)) for k,v in data.items()}))


original_evaluate = official.evaluate


def evaluate(*args, **kwargs):
    agent = kwargs['agent']
    result = original_evaluate(*args, **kwargs)
    learned = int(agent.network.step) - 1
    step = LAST_STEP
    write(f'official_eval_{step:06d}.json', dict(step=step, updates=learned,
        episodes=kwargs['num_eval_episodes'], stats={k:float(v) for k,v in result[0].items()}))
    if step in (OFFLINE, OFFLINE + ONLINE):
        offline = step == OFFLINE
        expected = OFFLINE if offline else OFFLINE + ONLINE - START_TRAINING + 1
        assert learned == expected, (learned, expected)
        filename = 'offline_500k.pkl' if offline else 'native_final.pkl'
        common.save_checkpoint(OUT / filename, agent, step=OFFLINE if offline else ONLINE,
            qam_config=dict(agent.config), actual_gradient_updates=learned)
        # The independent evaluation leaves both training RNGs and official eval env intact.
        numpy_state, python_state = np.random.get_state(), random.getstate()
        before = common.tree_hash(agent.network.params)
        paired = paired_evaluation.evaluate(agent, OUT / 'fixed_eval', 0 if offline else ONLINE,
            common_args, episodes=FIXED_EPISODES, residual_enabled=False)
        assert common.tree_hash(np.random.get_state()) == common.tree_hash(numpy_state)
        assert random.getstate() == python_state
        assert common.tree_hash(agent.network.params) == before
        write('offline_checkpoint.json' if offline else 'native_checkpoint.json', dict(
            checkpoint=str(OUT / filename), sha256=common.file_hash(OUT / filename),
            actual_gradient_updates=learned, flow_hash=common.flow_hash(agent),
            q_hash=common.tree_hash(agent.network.params['modules_critic']),
            target_q_hash=common.tree_hash(agent.network.params['modules_target_critic']),
            fixed_success=paired['success'], fixed_return=paired['return_mean']))
    return result


def setup_local(**kwargs):
    flags = official.get_flag_dict()
    write('requested_flags.json', flags)
    # Same return fields used by main; no remote logging or authentication.
    return official.wandb.run


verify_sources()
official.agents['qam'] = AgentFactory
official.make_env_and_datasets = load_data
official.LoggingHelper.log = log
official.evaluate = evaluate
official.setup_wandb = setup_local
official.wandb = SimpleNamespace(run=SimpleNamespace(project='qam-reproduce', url='local-only'),
    log=lambda *a, **k: None, finish=lambda: None)

def entry(_):
    try:
        official.main(_)
        checkpoint = json.loads((OUT / 'native_checkpoint.json').read_text())
        expected = ONLINE - START_TRAINING + 1
        assert checkpoint['actual_gradient_updates'] == OFFLINE + expected
        write('DONE.json', dict(status='passed', smoke=SMOKE, offline_updates=OFFLINE,
            online_env_steps=ONLINE, online_updates=expected, elapsed_seconds=time.time()-START,
            native=checkpoint, official_main_byte_identical=True))
    except BaseException as exc:
        write('FAILED.json', dict(type=type(exc).__name__, message=str(exc), last_step=LAST_STEP))
        raise

if __name__ == '__main__':
    sys.argv = [sys.argv[0], '--agent=agents/qam.py', '--seed=0', f'--env_name={ENV}',
        '--tags=QAM', f'--run_group=antmaze_task{TASK}_500k_50k', f'--save_dir={OUT / "official_logs"}',
        f'--offline_steps={OFFLINE}', f'--online_steps={ONLINE}', '--sparse=False',
        '--horizon_length=1', '--agent.action_chunking=True', '--agent.inv_temp=10.0',
        '--agent.fql_alpha=0.0', '--agent.edit_scale=0.0', '--utd_ratio=1',
        f'--start_training={START_TRAINING}', f'--eval_episodes={EVAL_EPISODES}',
        f'--log_interval={5 if SMOKE else 5000}', f'--eval_interval={10 if SMOKE else 50000}',
        f'--save_interval={10 if SMOKE else 50000}', '--auto_cleanup=False']
    official.app.run(entry)
