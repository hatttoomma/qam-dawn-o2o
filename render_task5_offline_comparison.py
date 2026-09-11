"""Compare the exact frozen offline base with all 100 archived DAWN episodes."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import imageio.v2 as imageio
import imageio_ffmpeg
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw

import render_final_demos as demo
from render_all_evaluations import make_writer, FPS, STRIDE, COUNT
import run as common


def decorate(frame, episode, steps, ret, status):
    im = Image.new('RGB', (640, 640), '#101827')
    im.paste(Image.fromarray(frame), (0, 96))
    d = ImageDraw.Draw(im)
    d.text((18, 9), 'TASK 5  /  Two-cube stacking', fill='#d8e4f0', font=demo.FONT[22])
    d.text((18, 37), 'QAM OFFLINE ONLY', fill='#66c2ff', font=demo.BOLD)
    d.text((18, 75), '500k offline  |  no online updates  |  sampled actions',
           fill='#a4b3c4', font=demo.FONT[15])
    d.text((18, 584), f'Episode {episode+1}/{COUNT}  |  reset {500000+episode}',
           fill='#e3eaf3', font=demo.FONT[17])
    color = '#79e7a6' if status == 'SUCCESS' else '#ff8c8c' if status == 'TIMEOUT' else '#d3dce8'
    d.text((480, 584), status, fill=color, font=demo.FONT[17])
    d.text((18, 612), f'Step {steps}/500  |  {steps/20:.2f} s  |  return {ret:.0f}  |  1x speed',
           fill='#afbdce', font=demo.FONT[16])
    return np.asarray(im)


def rollout(env, qam, episode, out, archived, paired):
    ob, _ = env.reset(seed=500000+episode)
    initial_hash = common.tree_hash(ob)
    assert initial_hash == paired['initial_hash'], f'Initial-state mismatch: {episode}'
    key = jax.random.PRNGKey(800000+episode)
    ret, steps, decisions, success = 0., 0, 0, 0.
    observations, actions, rewards = [ob.copy()], [], []
    frames = [decorate(env.render(), episode, steps, ret, 'RUNNING')]
    done = False
    while not done:
        key, action_key = jax.random.split(key)
        action = qam.sample_actions(jnp.asarray(ob), action_key)
        for a in np.asarray(action).reshape(5, -1):
            ob, reward, terminated, truncated, info = env.step(a)
            steps += 1; ret += float(reward)
            success = max(success, float(info.get('success', 0.)))
            done = bool(terminated or truncated)
            observations.append(ob.copy()); actions.append(a.copy()); rewards.append(float(reward))
            if steps % STRIDE == 0 or done:
                status = ('SUCCESS' if success else 'TIMEOUT') if done else 'RUNNING'
                frames.append(decorate(env.render(), episode, steps, ret, status))
            if done:
                break
        decisions += 1
    frames.extend([frames[-1]]*FPS)
    record = dict(episode=episode, reset_seed=500000+episode, initial_hash=initial_hash,
                  success=success, return_=ret, length=steps, decisions=decisions,
                  frames=len(frames), duration_seconds=len(frames)/FPS)
    expected = archived['records'][episode] if episode < len(archived['records']) else None
    differences = {k: dict(archived=expected[k], rendered=record[k])
                   for k in ('success', 'return_', 'length', 'decisions')
                   if expected and expected[k] != record[k]}
    record.update(archived_reference_available=expected is not None,
                  archived_outcomes_match=not differences if expected else None,
                  differences=differences)
    folder = out/'offline_episodes'
    np.savez_compressed(folder/f'episode_{episode:03d}_trajectory.npz',
                        observations=np.asarray(observations), actions=np.asarray(actions),
                        rewards=np.asarray(rewards))
    common.atomic_json(folder/f'episode_{episode:03d}.json', record)
    return frames, record


def validate_and_package(out, manifest):
    checks = []
    for v in manifest['videos']:
        path = out/v['name']
        count, duration = imageio_ffmpeg.count_frames_and_secs(str(path))
        assert count == v['frames'] and abs(duration-v['duration_seconds']) < .11
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-threads', '2',
                        '-i', str(path), '-f', 'null', '-'], check=True)
        with imageio.get_reader(str(path), format='FFMPEG') as reader:
            assert reader.get_meta_data()['fps'] == FPS
            frame = reader.get_data(100)
            assert frame.shape == (640, 1280 if 'comparison' in path.name else 640, 3)
            Image.fromarray(frame).save(out/(path.stem+'_preview.jpg'), quality=90)
        checks.append(dict(name=v['name'], frames=count, duration_seconds=duration, full_decode_passed=True))
    common.atomic_json(out/'validation.json', dict(status='passed', checks=checks))
    files = [out/v['name'] for v in manifest['videos']]
    files += [out/'manifest.json', out/'chapters.csv', out/'validation.json']
    files += sorted(out.glob('*_preview.jpg'))
    with tarfile.open(out/'delivery.tar', 'w') as archive:
        for path in files:
            archive.add(path, arcname=path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--dawn-source', required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    assert not (out/'manifest.json').exists()
    (out/'offline_episodes').mkdir(exist_ok=True)
    source = Path(args.dawn_source).resolve()
    prior = json.loads((source/'task5_manifest.json').read_text())
    dawn_records = prior['episodes']['dawn']
    assert len(dawn_records) == COUNT and prior['all_outcomes_match']
    assert prior['fps'] == FPS and prior['frame_stride'] == STRIDE
    video = next(x for x in prior['videos'] if x['name'] == 'task5_dawn_all100.mp4')
    assert common.file_hash(source/video['name']) == video['sha256']
    checkpoint = demo.ROOT/'runs/cube5/task5_offline/final.pkl'
    digest = common.file_hash(checkpoint)
    assert digest == prior['provenance']['dawn']['base_checkpoint_sha256']
    done = json.loads((checkpoint.parent/'DONE.json').read_text())
    assert digest == done['checkpoint_sha256']
    qam, payload = common.load_checkpoint(checkpoint, common.qam_template(
        jnp.zeros((37,), jnp.float32), jnp.zeros((5,), jnp.float32), 0))
    assert payload['step'] == 500000 and common.flow_hash(qam) == done['flow_hash']
    initial_params = common.tree_hash(qam.network.params)
    archived = json.loads((checkpoint.parent/'eval_500000_050.json').read_text())
    env = common.ogbench.make_env_and_datasets(
        'cube-double-play-singletask-task5-v0', env_only=True, width=640, height=480)
    assert env.unwrapped.control_timestep() == .05
    paths = {'offline': out/'task5_offline_all100.mp4',
             'comparison': out/'task5_offline_vs_dawn_comparison_all100.mp4'}
    writers = {k: make_writer(v) for k, v in paths.items()}
    reader = imageio.get_reader(str(source/video['name']), format='FFMPEG')
    records, chapters = [], []
    counts = dict(offline=0, comparison=0, dawn=0)
    started = time.monotonic()
    try:
        for episode, da in enumerate(dawn_records):
            assert da['episode'] == episode and da['reset_seed'] == 500000+episode
            frames, record = rollout(env, qam, episode, out, archived, da)
            record['video_start_seconds'] = counts['offline']/FPS
            for frame in frames:
                writers['offline'].append_data(frame)
            counts['offline'] += len(frames)
            start = counts['comparison']/FPS
            assert counts['dawn'] == round(da['video_start_seconds']*FPS)
            for k in range(max(len(frames), da['frames'])):
                if k < da['frames']:
                    right = reader.get_next_data(); counts['dawn'] += 1
                pair = np.concatenate([frames[min(k, len(frames)-1)], right], axis=1)
                writers['comparison'].append_data(pair)
                counts['comparison'] += 1
                if episode in (0, 3, 49, 99) and k == 0:
                    Image.fromarray(pair).save(out/f'episode_{episode+1:03d}_start_preview.jpg', quality=85)
            records.append(record)
            chapters.append(dict(episode=episode+1, reset_seed=500000+episode,
                start_seconds=start, end_seconds=counts['comparison']/FPS,
                offline_success=record['success'], dawn_success=da['success'],
                offline_steps=record['length'], dawn_steps=da['length'],
                offline_return=record['return_'], dawn_return=da['return_']))
            progress = dict(episodes_completed=episode+1, total=COUNT,
                elapsed_seconds=time.monotonic()-started,
                offline_successes=sum(x['success'] for x in records),
                dawn_successes=sum(x['success'] for x in dawn_records[:episode+1]),
                archive_mismatches=sum(x['archived_outcomes_match'] is False for x in records))
            common.atomic_json(out/'progress.json', progress)
            if episode % 5 == 4 or episode == 0:
                demo.emit(**progress)
            del frames, frame, pair, right
        assert counts['dawn'] == video['frames']
        try:
            reader.get_next_data()
        except IndexError:
            pass
        else:
            raise AssertionError('Unexpected frames beyond archived DAWN episode 100')
    finally:
        reader.close()
        for writer in writers.values(): writer.close()
        if getattr(env.unwrapped, '_renderer', None) is not None:
            env.unwrapped._renderer.close(); env.unwrapped._renderer = None
        env.close()
    assert initial_params == common.tree_hash(qam.network.params)
    with (out/'chapters.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(chapters[0]))
        writer.writeheader(); writer.writerows(chapters)
    paired_results = {
        name: [i+1 for i, (a, b) in enumerate(zip(records, dawn_records))
               if bool(a['success']) == left and bool(b['success']) == right]
        for name, left, right in [('both_success', True, True), ('both_failure', False, False),
                                  ('residual_gain', False, True), ('residual_loss', True, False)]}
    metrics = {k: dict(success=sum(r['success'] for r in rows)/COUNT,
                       return_mean=float(np.mean([r['return_'] for r in rows])),
                       length_mean=float(np.mean([r['length'] for r in rows])))
               for k, rows in [('offline', records), ('dawn', dawn_records)]}
    manifest = dict(task=5, episode_count=COUNT, selection='All evaluation seeds 500000..500099 in order',
        training_seed=0, offline_updates=500000, dawn_online_env_steps=50000,
        offline_checkpoint=str(checkpoint), offline_checkpoint_sha256=digest,
        dawn_provenance=prior['provenance']['dawn'], reused_dawn_video=str(source/video['name']),
        reused_dawn_video_sha256=video['sha256'], frozen_parameters_verified=True,
        paired_initial_states_match=True, fps=FPS, frame_stride=STRIDE, playback_speed=1.,
        control_timestep=.05, end_hold_seconds=1., shorter_episode_padding='Hold final frame until paired episode ends',
        sampling='Stochastic; base RNG 800000+episode and residual RNG 900000+episode',
        archived_offline_reference_episodes=len(archived['records']),
        archived_offline_outcomes_match=all(x['archived_outcomes_match'] is not False for x in records),
        metrics=metrics, paired_results=paired_results, episodes=dict(offline=records, dawn=dawn_records),
        render_seconds=time.monotonic()-started, source_sha256=common.file_hash(Path(__file__)),
        videos=[dict(name=p.name, bytes=p.stat().st_size, sha256=common.file_hash(p), frames=counts[k],
                     duration_seconds=counts[k]/FPS) for k, p in paths.items()])
    common.atomic_json(out/'manifest.json', manifest)
    demo.emit(rendering_complete=True, metrics=metrics,
              paired_counts={k: len(v) for k, v in paired_results.items()},
              archived_offline_outcomes_match=manifest['archived_offline_outcomes_match'])
    validate_and_package(out, manifest)
    demo.emit(done=True, elapsed_seconds=time.monotonic()-started,
              delivery_bytes=(out/'delivery.tar').stat().st_size)


if __name__ == '__main__':
    main()
