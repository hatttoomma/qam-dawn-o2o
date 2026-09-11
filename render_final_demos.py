"""Render fixed final-evaluation episodes without modifying trained agents."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import imageio.v2 as imageio
import imageio_ffmpeg
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import run as common
from actor_input_agent import ActorInputAgent

ROOT = Path(__file__).resolve().parent
SOURCES = {
    (1, 'qam'): 'runs/native',
    (1, 'dawn'): 'runs/actor_input_ablation/task1_base_action',
    (5, 'qam'): 'runs/cube5/task5_native',
    (5, 'dawn'): 'runs/cube5/task5_warm',
}
EXPECTED = {
    (1, 'qam'): 'f28984db937d8df263da9b9004efa76311d90a31569acbe83e190949fb0e2364',
    (1, 'dawn'): '4ee4d1b5eb4eeb690f655463d300be313a1dcb1d24bf9264f2b5cd38b88d28bf',
    (5, 'qam'): '2e5b28279701912ceb5b556593e2ff2fcb9635a1bd53875c71da4584df4ed842',
    (5, 'dawn'): '427e6b11b3c75346860a4dd0f04ddab3bd2770bddcd249184dd156ffa9626590',
}
W, H, FPS = 640, 640, 20
FONT_PATH = '/usr/share/fonts/truetype/dejavu/'
FONT = {size: ImageFont.truetype(FONT_PATH + 'DejaVuSans.ttf', size)
        for size in (15, 16, 17, 22)}
BOLD = ImageFont.truetype(FONT_PATH + 'DejaVuSans-Bold.ttf', 25)


def emit(**kwargs):
    print(json.dumps(kwargs, ensure_ascii=False), flush=True)


def decorate(frame, task, method, episode, n, step, ret, status, eval_success):
    im = Image.new('RGB', (W, H), '#101827')
    im.paste(Image.fromarray(frame), (0, 96))
    d = ImageDraw.Draw(im)
    title = 'Single pick-and-place' if task == 1 else 'Two-cube stacking'
    d.text((18, 9), f'TASK {task}  /  {title}', fill='#d8e4f0', font=FONT[22])
    color = '#66c2ff' if method == 'qam' else '#ffbc69'
    name = 'QAM-EDIT (edit=0)' if method == 'qam' else 'DAWN residual'
    d.text((18, 37), name, fill=color, font=BOLD)
    d.text((455, 44), f'Prev eval100: {eval_success:.0%}', fill='#dae2ec', font=FONT[15])
    d.text((18, 75), 'Final checkpoint  |  500k offline + 50k online  |  sampled actions',
           fill='#a4b3c4', font=FONT[15])
    d.text((18, 584), f'Episode {episode + 1}/{n}  |  reset {500000 + episode}',
           fill='#e3eaf3', font=FONT[17])
    sc = '#79e7a6' if status == 'SUCCESS' else '#ff8c8c' if status == 'TIMEOUT' else '#d3dce8'
    d.text((480, 584), status, fill=sc, font=FONT[17])
    d.text((18, 612), f'Step {step}/500  |  {step / FPS:.2f} s  |  return {ret:.0f}  |  1x speed',
           fill='#afbdce', font=FONT[16])
    return np.asarray(im)


def writer(path, width=W):
    return imageio.get_writer(str(path), format='FFMPEG', mode='I', fps=FPS,
                              codec='libx264', pixelformat='yuv420p', macro_block_size=16,
                              quality=8, output_params=['-movflags', '+faststart', '-threads', '2'])


def load_models(task, method):
    folder = ROOT / SOURCES[task, method]
    checkpoint = folder / 'final.pkl'
    digest = common.file_hash(checkpoint)
    assert digest == EXPECTED[task, method]
    done = json.loads((folder / 'DONE.json').read_text())
    assert digest == done['checkpoint_sha256']
    config = json.loads((folder / 'config.json').read_text())
    obs = jnp.zeros((37,), dtype=jnp.float32)
    template = common.qam_template(obs, jnp.zeros((5,), dtype=jnp.float32), 0)
    if method == 'qam':
        qam, payload = common.load_checkpoint(checkpoint, template)
        agent = None
        base_sha = None
    else:
        base_path = Path(config['offline_checkpoint'])
        qam, base_payload = common.load_checkpoint(base_path, template)
        assert base_payload['step'] == 500000
        base_sha = common.file_hash(base_path)
        assert base_sha == json.loads((base_path.parent / 'DONE.json').read_text())['checkpoint_sha256']
        template_agent = ActorInputAgent.create(qam, obs, 0, warm=True).replace(condition_on_base=True)
        agent, payload = common.load_checkpoint(checkpoint, template_agent)
        assert agent.condition_on_base and agent.res_scale == .1
        assert int(agent.updates) == 7500
        assert common.flow_hash(qam) == done['final_flow_hash']
    assert payload['step'] == 50000
    archived = json.loads((folder / 'eval_050000_100.json').read_text())
    return qam, agent, archived, dict(source=str(folder), checkpoint_sha256=digest,
                                     base_checkpoint_sha256=base_sha)


def render_group(task, method, out, episodes):
    qam, agent, archived, provenance = load_models(task, method)
    policy_hash_before = common.tree_hash(qam.network.params)
    residual_hash_before = common.tree_hash(agent.actor.params) if agent else None
    folder = out / f'task{task}_{method}'
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    reference_folder = ROOT / SOURCES[task, 'dawn']
    with np.load(reference_folder / 'mc_000000_050.npz') as data:
        reference_initial_observations = data['observations'].copy()
    env = common.ogbench.make_env_and_datasets(
        f'cube-double-play-singletask-task{task}-v0', env_only=True, width=640, height=480)
    try:
        for episode in range(episodes):
            reset_seed = 500000 + episode
            ob, _ = env.reset(seed=reset_seed)
            assert abs(env.unwrapped.control_timestep() - 1 / FPS) < 1e-12
            initial_hash = common.tree_hash(ob)
            expected = archived['records'][episode]
            # Cross-host float64 IK can differ below policy-input precision.
            # Verify the actual float32 policy input against the archived observation.
            np.testing.assert_array_equal(ob.astype(np.float32), reference_initial_observations[episode])
            base_key = jax.random.PRNGKey(800000 + episode)
            residual_key = jax.random.PRNGKey(900000 + episode)
            ret, steps, success, decisions, nframes = 0., 0, 0., 0, 0
            actions, observations, rewards = [], [np.asarray(ob).copy()], []
            path = folder / f'episode_{episode:02d}.mp4'
            video = writer(path)
            try:
                image = decorate(env.render(), task, method, episode, episodes, 0, 0, 'RUNNING', archived['success'])
                video.append_data(image); nframes += 1
                if episode == 0:
                    Image.fromarray(image).save(folder / 'preview_start.png')
                finished = False
                while not finished:
                    base_key, key = jax.random.split(base_key)
                    base = qam.sample_actions(jnp.asarray(ob), key)
                    if agent is not None:
                        residual_key, rkey = jax.random.split(residual_key)
                        action, _ = agent.sample(jnp.asarray(ob), base, rkey, deterministic=False)
                    else:
                        action = base
                    for a in np.asarray(action).reshape(common.HORIZON, -1):
                        ob, reward, terminated, truncated, info = env.step(a)
                        ret += float(reward); steps += 1
                        success = max(success, float(info.get('success', 0.)))
                        finished = bool(terminated or truncated)
                        actions.append(a.copy()); observations.append(np.asarray(ob).copy()); rewards.append(float(reward))
                        status = ('SUCCESS' if success else 'TIMEOUT') if finished else 'RUNNING'
                        image = decorate(env.render(), task, method, episode, episodes, steps, ret, status, archived['success'])
                        video.append_data(image); nframes += 1
                        if steps == 50:
                            Image.fromarray(image).save(folder / f'preview_ep{episode:02d}_step050.png')
                        if finished:
                            break
                    decisions += 1
                for _ in range(FPS):
                    video.append_data(image); nframes += 1
                Image.fromarray(image).save(folder / f'preview_ep{episode:02d}_final.png')
            finally:
                video.close()
            record = dict(episode=episode, reset_seed=reset_seed, initial_hash=initial_hash,
                          success=success, return_=ret, length=steps, decisions=decisions)
            differences = {k: dict(expected=expected[k], rendered=record[k])
                           for k in record if k != 'initial_hash' and record[k] != expected[k]}
            np.savez_compressed(folder / f'episode_{episode:02d}_trajectory.npz',
                                observations=np.asarray(observations), actions=np.asarray(actions), rewards=np.asarray(rewards))
            metadata = dict(**record, video=str(path.relative_to(out)), frames=nframes,
                            seconds=nframes/FPS, original_evaluation_exact_match=not differences,
                            archived_initial_hash=expected['initial_hash'],
                            initial_float64_hash_matches=initial_hash == expected['initial_hash'],
                            initial_policy_input_float32_exact_match=True,
                            differences=differences, video_sha256=common.file_hash(path))
            common.atomic_json(folder / f'episode_{episode:02d}.json', metadata)
            records.append(metadata)
            emit(task=task, method=method, episode=episode, success=success, steps=steps,
                 return_=ret, original_exact_match=not differences)
    finally:
        if getattr(env.unwrapped, '_renderer', None) is not None:
            env.unwrapped._renderer.close()
            env.unwrapped._renderer = None
        env.close()
    assert policy_hash_before == common.tree_hash(qam.network.params)
    assert residual_hash_before == (common.tree_hash(agent.actor.params) if agent else None)
    result = dict(task=task, method=method, **provenance, episodes=records,
                  final100_success=archived['success'], final100_return=archived['return_mean'],
                  frozen_parameters_verified=True)
    common.atomic_json(folder / 'manifest.json', result)
    del qam, agent
    gc.collect()
    jax.clear_caches()
    return result


def concat_files(paths, output):
    listing = output.with_suffix('.concat.txt')
    listing.write_text(''.join(f"file '{p.resolve()}'\n" for p in paths))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-y', '-f', 'concat',
                    '-safe', '0', '-i', str(listing), '-c', 'copy', '-movflags', '+faststart', str(output)], check=True)


def compose_task(task, out, groups, episodes):
    for method in ['qam', 'dawn']:
        paths = [out / groups[task, method]['episodes'][ep]['video'] for ep in range(episodes)]
        concat_files(paths, out / f'task{task}_{method}_final.mp4')
    pair_paths = []
    for ep in range(episodes):
        paths = [out / groups[task, method]['episodes'][ep]['video'] for method in ['qam', 'dawn']]
        nums = [groups[task, method]['episodes'][ep]['frames'] for method in ['qam', 'dawn']]
        readers = [imageio.get_reader(str(p), format='FFMPEG') for p in paths]
        output = out / f'task{task}_paired_ep{ep:02d}.mp4'
        video = writer(output, width=2*W)
        last = [None, None]
        try:
            for k in range(max(nums)):
                for side in range(2):
                    if k < nums[side]: last[side] = readers[side].get_next_data()
                image = np.concatenate(last, axis=1)
                video.append_data(image)
                if k == 50 or k == max(nums)-1:
                    Image.fromarray(image).save(out / f'task{task}_paired_ep{ep:02d}_frame{k:04d}.jpg')
        finally:
            video.close()
            for reader in readers: reader.close()
        pair_paths.append(output)
    concat_files(pair_paths, out / f'task{task}_comparison.mp4')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--episodes', type=int, default=5)
    args = parser.parse_args()
    assert 1 <= args.episodes <= 100
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    groups = {}
    for task in [1, 5]:
        for method in ['qam', 'dawn']:
            manifest = out / f'task{task}_{method}/manifest.json'
            if manifest.exists():
                value = json.loads(manifest.read_text())
                assert len(value['episodes']) == args.episodes and value['checkpoint_sha256'] == EXPECTED[task, method]
                assert all(common.file_hash(out / x['video']) == x['video_sha256'] for x in value['episodes'])
                groups[task, method] = value
            else:
                groups[task, method] = render_group(task, method, out, args.episodes)
        compose_task(task, out, groups, args.episodes)
        emit(task=task, comparison_ready=True)
    manifest = dict(protocol='First five final evaluation episodes; no outcome-based selection',
                    playback_fps=FPS, simulation_control_timestep=.05, playback_speed=1.,
                    paired_layout='QAM left / DAWN right; same reset seeds and time; completed side freezes until other finishes',
                    end_hold_seconds=1., residual_sampling='stochastic, matching primary evaluation',
                    groups=list(groups.values()),
                    all_original_outcomes_match=all(e['original_evaluation_exact_match'] for g in groups.values() for e in g['episodes']),
                    source_sha256=common.file_hash(Path(__file__)),
                    videos=[dict(name=p.name, bytes=p.stat().st_size, sha256=common.file_hash(p))
                            for p in sorted(out.glob('*.mp4')) if '_ep' not in p.name])
    common.atomic_json(out / 'manifest.json', manifest)
    emit(done=True, all_original_outcomes_match=manifest['all_original_outcomes_match'], videos=manifest['videos'])


if __name__ == '__main__':
    main()
