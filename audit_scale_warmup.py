"""Capture and audit warmup replay without changing any training process."""
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'runs/scale_extremes'
OUT = RUNS / 'warmup_audit'


def main():
    from jax.tree_util import tree_flatten_with_path

    def tree_hash(tree):
        h = hashlib.sha256()
        for path, leaf in tree_flatten_with_path(tree)[0]:
            a = np.asarray(leaf)
            h.update(str(path).encode());h.update(str(a.shape).encode())
            h.update(str(a.dtype).encode());h.update(a.tobytes())
        return h.hexdigest()

    OUT.mkdir(exist_ok=True)
    result = dict(status='passed', raw_observation_atol=1e-10, training_float32_atol=1e-12,
                  extraction_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), arms={})
    for task in (1, 2):
        ref = ROOT / ('runs/warm' if task == 1 else 'runs/task2/warm')
        reference_meta = json.loads((ref / 'warmup.json').read_text())
        baseline = None
        for name, folder in [(f'task{task}_reference', ref),
                             (f'task{task}_scale001', RUNS / f'task{task}_scale001'),
                             (f'task{task}_scale100', RUNS / f'task{task}_scale100')]:
            meta = json.loads((folder / 'warmup.json').read_text())
            assert {k: v for k, v in meta.items() if k != 'replay_hash'} == {
                k: v for k, v in reference_meta.items() if k != 'replay_hash'}
            raw = (folder / 'latest.pkl').read_bytes()
            checkpoint_sha = hashlib.sha256(raw).hexdigest()
            checkpoint = pickle.loads(raw);del raw
            assert checkpoint['step'] >= meta['steps']
            prefix = checkpoint['replay'][:meta['chunks']]
            assert len(prefix) == meta['chunks'] and tree_hash(prefix) == meta['replay_hash']
            arrays = {k: np.asarray([r[k] for r in prefix]) for k in prefix[0]}
            if baseline is None: baseline = arrays
            diagnostics = {}
            for k, a in arrays.items():
                b = baseline[k];assert a.shape == b.shape and a.dtype == b.dtype
                assert np.isfinite(a).all() and np.isfinite(b).all()
                x, y = a.astype(np.float32), b.astype(np.float32)
                if k in ('observations', 'next_observations'):
                    np.testing.assert_allclose(a, b, rtol=0, atol=result['raw_observation_atol'])
                    np.testing.assert_allclose(x, y, rtol=0, atol=result['training_float32_atol'])
                else: np.testing.assert_array_equal(a, b)
                diagnostics[k] = dict(raw_changed_elements=int(np.sum(a != b)),
                    raw_max_abs_diff=float(np.max(np.abs(a - b))),
                    float32_changed_elements=int(np.sum(x != y)),
                    float32_max_abs_diff=float(np.max(np.abs(x - y))))
            target = OUT / (name + '.npz');np.savez_compressed(target, **arrays)
            result['arms'][name] = dict(source_checkpoint=str((folder / 'latest.pkl').relative_to(ROOT)),
                captured_checkpoint_sha256=checkpoint_sha, captured_checkpoint_step=checkpoint['step'],
                warmup_chunks=meta['chunks'], warmup_replay_hash=meta['replay_hash'],
                snapshot=str(target.relative_to(ROOT)), snapshot_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                comparison_with=f'task{task}_reference', diagnostics=diagnostics)
            del checkpoint, prefix
    (OUT / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
