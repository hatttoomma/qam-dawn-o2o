"""Render complete archived evaluations, with two independent task workers."""
import argparse
import csv
import json
import os
from pathlib import Path
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

import render_final_demos as demo
import run as common

FPS = 10
STRIDE = 2
COUNT = 100


def make_writer(path):
    return imageio.get_writer(str(path), format='FFMPEG', mode='I', fps=FPS,
                             codec='libx264', pixelformat='yuv420p', macro_block_size=16,
                             quality=None, output_params=['-crf', '28', '-preset', 'veryfast',
                                                          '-threads', '2', '-movflags', '+faststart'])


def rollout(env, qam, agent, archive, reference, task, method, episode, out):
    ob, _ = env.reset(seed=500000 + episode)
    assert abs(env.unwrapped.control_timestep() - .05) < 1e-12
    initial_hash = common.tree_hash(ob)
    expected = archive['records'][episode]
    input_matches = None
    if episode < len(reference):
        np.testing.assert_array_equal(ob.astype(np.float32), reference[episode])
        input_matches = True
    bk = jax.random.PRNGKey(800000 + episode)
    rk = jax.random.PRNGKey(900000 + episode)
    steps, decisions, ret, success = 0, 0, 0., 0.
    observations, actions, rewards = [np.asarray(ob).copy()], [], []

    def frame(status):
        return demo.decorate(env.render(), task, method, episode, COUNT, steps, ret,
                             status, archive['success'])

    frames = [frame('RUNNING')]
    finished = False
    while not finished:
        bk, key = jax.random.split(bk)
        base = qam.sample_actions(jnp.asarray(ob), key)
        if agent is None:
            action = base
        else:
            rk, rkey = jax.random.split(rk)
            action, _ = agent.sample(jnp.asarray(ob), base, rkey, deterministic=False)
        for a in np.asarray(action).reshape(5, -1):
            ob, reward, terminated, truncated, info = env.step(a)
            ret += float(reward); steps += 1
            success = max(success, float(info.get('success', 0.)))
            finished = bool(terminated or truncated)
            observations.append(np.asarray(ob).copy()); actions.append(a.copy()); rewards.append(float(reward))
            if steps % STRIDE == 0 or finished:
                status = ('SUCCESS' if success else 'TIMEOUT') if finished else 'RUNNING'
                frames.append(frame(status))
            if finished:
                break
        decisions += 1
    frames.extend([frames[-1]] * FPS)
    actual = dict(episode=episode, reset_seed=500000 + episode, initial_hash=initial_hash,
                  success=success, return_=ret, length=steps, decisions=decisions)
    differences = {k: dict(archived=expected[k], rendered=v) for k, v in actual.items()
                   if k != 'initial_hash' and expected[k] != v}
    folder = out / f'task{task}_{method}'
    folder.mkdir(exist_ok=True)
    np.savez_compressed(folder / f'episode_{episode:03d}_trajectory.npz',
                        observations=np.asarray(observations), actions=np.asarray(actions), rewards=np.asarray(rewards))
    record = dict(**actual, frames=len(frames), duration_seconds=len(frames)/FPS,
                  archived_initial_hash=expected['initial_hash'],
                  initial_float64_hash_matches=initial_hash == expected['initial_hash'],
                  archived_float32_input_match=input_matches,
                  original_outcomes_match=not differences, differences=differences)
    common.atomic_json(folder / f'episode_{episode:03d}.json', record)
    if episode in [0, 3, 49, 99] or not success:
        Image.fromarray(frames[-1]).save(folder / f'episode_{episode:03d}_final.jpg', quality=85)
    return frames, record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', type=int, choices=[1, 5], required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    task = args.task
    assert not (out / f'task{task}_manifest.json').exists()
    started = time.monotonic()
    models, envs, original_hashes, provenance = {}, {}, {}, {}
    for method in ['qam', 'dawn']:
        qam, agent, archive, origin = demo.load_models(task, method)
        models[method] = (qam, agent, archive)
        provenance[method] = origin
        original_hashes[method] = (common.tree_hash(qam.network.params),
                                   common.tree_hash(agent.actor.params) if agent else None)
        envs[method] = common.ogbench.make_env_and_datasets(
            f'cube-double-play-singletask-task{task}-v0', env_only=True, width=640, height=480)
    with np.load(demo.ROOT / demo.SOURCES[task, 'dawn'] / 'mc_000000_050.npz') as data:
        reference = data['observations'].copy()
    paths = {method: out / f'task{task}_{method}_all100.mp4' for method in ['qam', 'dawn']}
    paths['comparison'] = out / f'task{task}_comparison_all100.mp4'
    writers = {k: make_writer(v) for k, v in paths.items()}
    records = {'qam': [], 'dawn': []}
    frame_counts = dict(qam=0, dawn=0, comparison=0)
    chapters = []
    try:
        for episode in range(COUNT):
            paired = {}
            for method in ['qam', 'dawn']:
                qam, agent, archive = models[method]
                frames, record = rollout(envs[method], qam, agent, archive, reference,
                                         task, method, episode, out)
                record['video_start_seconds'] = frame_counts[method] / FPS
                for frame in frames: writers[method].append_data(frame)
                frame_counts[method] += len(frames)
                records[method].append(record)
                paired[method] = frames
            qa, da = records['qam'][-1], records['dawn'][-1]
            assert qa['initial_hash'] == da['initial_hash']
            count = max(len(paired['qam']), len(paired['dawn']))
            start = frame_counts['comparison']/FPS
            for k in range(count):
                frame = np.concatenate([paired[m][min(k, len(paired[m])-1)] for m in ['qam', 'dawn']], axis=1)
                writers['comparison'].append_data(frame)
                if episode in [0, 3, 49, 99] and k == count-1:
                    Image.fromarray(frame).save(out / f'task{task}_episode_{episode:03d}_preview.jpg', quality=85)
            frame_counts['comparison'] += count
            chapters.append(dict(episode=episode+1, reset_seed=500000+episode,
                                 start_seconds=start, end_seconds=frame_counts['comparison']/FPS,
                                 qam_success=qa['success'], dawn_success=da['success'],
                                 qam_steps=qa['length'], dawn_steps=da['length'],
                                 qam_return=qa['return_'], dawn_return=da['return_']))
            progress = dict(task=task, episodes_completed=episode+1, target=COUNT,
                            elapsed_seconds=time.monotonic()-started,
                            qam_successes=sum(x['success'] for x in records['qam']),
                            dawn_successes=sum(x['success'] for x in records['dawn']),
                            mismatches=sum(not x['original_outcomes_match'] for v in records.values() for x in v),
                            frame_counts=frame_counts)
            common.atomic_json(out / f'task{task}_progress.json', progress)
            if episode % 5 == 4 or episode == 0: demo.emit(**progress)
            del paired, frames, frame
    finally:
        for writer in writers.values(): writer.close()
        for env in envs.values():
            if getattr(env.unwrapped, '_renderer', None) is not None:
                env.unwrapped._renderer.close(); env.unwrapped._renderer = None
            env.close()
    for method in ['qam', 'dawn']:
        qam, agent, _ = models[method]
        assert original_hashes[method] == (common.tree_hash(qam.network.params),
                                            common.tree_hash(agent.actor.params) if agent else None)
    with (out / f'task{task}_chapters.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(chapters[0])); w.writeheader(); w.writerows(chapters)
    result = dict(task=task, selection='All 100 archived final-evaluation episodes, in original order',
                  training_seed=0, eval_seeds=[500000,500099], fps=FPS, frame_stride=STRIDE,
                  playback_speed=1., control_timestep=.05, end_hold_seconds=1.,
                  duration_rounding='Up to 0.1 s at episode end due to 10 fps sampling',
                  residual_sampling='stochastic; base and residual RNG match archived primary evaluation',
                  all_outcomes_match=all(x['original_outcomes_match'] for v in records.values() for x in v),
                  paired_initial_states_match=True, frozen_parameters_verified=True,
                  provenance=provenance, episodes=records, elapsed_seconds=time.monotonic()-started,
                  source_sha256=common.file_hash(Path(__file__)),
                  helper_sha256=common.file_hash(Path(demo.__file__)),
                  videos=[dict(name=path.name, bytes=path.stat().st_size, frames=frame_counts[k],
                               duration_seconds=frame_counts[k]/FPS, sha256=common.file_hash(path))
                          for k,path in paths.items()])
    common.atomic_json(out / f'task{task}_manifest.json', result)
    demo.emit(task=task, done=True, all_outcomes_match=result['all_outcomes_match'],
              elapsed_seconds=result['elapsed_seconds'], videos=result['videos'])


if __name__ == '__main__':
    main()
