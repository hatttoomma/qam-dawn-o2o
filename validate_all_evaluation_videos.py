"""Validate complete evaluation videos and prepare a compact delivery archive."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

import imageio.v2 as imageio
import imageio_ffmpeg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', type=int, required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out).resolve(); task = args.task
    m = json.loads((out / f'task{task}_manifest.json').read_text())
    assert m['frozen_parameters_verified'] and m['paired_initial_states_match']
    assert all(len(x) == 100 for x in m['episodes'].values())
    for method, records in m['episodes'].items():
        assert [x['episode'] for x in records] == list(range(100))
        assert [x['reset_seed'] for x in records] == list(range(500000,500100))
    checks = []
    for v in m['videos']:
        path = out / v['name']
        assert path.stat().st_size == v['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == v['sha256']
        frames, duration = imageio_ffmpeg.count_frames_and_secs(str(path))
        assert frames == v['frames'] and abs(duration-v['duration_seconds']) < .11
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-threads', '2',
                        '-i', str(path), '-f', 'null', '-'], check=True)
        reader = imageio.get_reader(str(path), format='FFMPEG')
        try:
            assert abs(reader.get_meta_data()['fps']-10) < 1e-6
            frame = reader.get_data(100)
            expected_width = 1280 if 'comparison' in path.name else 640
            assert frame.shape == (640, expected_width, 3)
            from PIL import Image
            Image.fromarray(frame).save(out / (path.stem+'_decoded_preview.jpg'), quality=90)
        finally:
            reader.close()
        checks.append(dict(name=v['name'], frames=frames, duration_seconds=duration,
                           full_decode_passed=True, sha256_passed=True))
    result = dict(status='passed', task=task, all_200_episodes_included=True,
                  original_outcomes_match=m['all_outcomes_match'], video_checks=checks,
                  rendered_successes={k:sum(int(x['success']) for x in v) for k,v in m['episodes'].items()},
                  mismatches={k:[x for x in v if not x['original_outcomes_match']] for k,v in m['episodes'].items()})
    validation = out / f'task{task}_validation.json'
    validation.write_text(json.dumps(result,indent=2)+'\n')
    files = [out/v['name'] for v in m['videos']]
    files.extend([out/f'task{task}_manifest.json',out/f'task{task}_chapters.csv',validation])
    files.extend(out.glob(f'task{task}_*_decoded_preview.jpg'))
    archive = out / f'task{task}_delivery.tar'
    with tarfile.open(archive, 'w') as tar:
        for f in files: tar.add(f, arcname=f.name)
    print(json.dumps(dict(**result, archive_bytes=archive.stat().st_size)),flush=True)


if __name__ == '__main__':
    main()
