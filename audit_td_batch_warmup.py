"""Audit paired warmup replay and retain portable array evidence."""
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from audit_td_warmup import tree_hash

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'runs/td_ablation/batch_comparison'
OUT = RUNS / 'warmup_audit'


def difference(a, b):
    assert a.keys() == b.keys()
    result = {}
    for k, x in a.items():
        y = b[k]
        assert x.shape == y.shape and x.dtype == y.dtype and np.isfinite(x).all()
        xf, yf = x.astype(np.float32), y.astype(np.float32)
        result[k] = dict(changed_elements=int(np.sum(x != y)),
                         raw_max_abs_diff=float(np.max(np.abs(x-y))),
                         float32_max_abs_diff=float(np.max(np.abs(xf-yf))))
    return result


def main():
    OUT.mkdir(exist_ok=True); rows = {}
    for task in (1, 2):
        base = None
        for batch in (1024, 256):
            name = f'task{task}_b{batch}'; folder = RUNS / name
            meta = json.loads((folder / 'warmup.json').read_text())
            raw = (folder / 'latest.pkl').read_bytes()
            source_sha = hashlib.sha256(raw).hexdigest()
            cp = pickle.loads(raw); del raw
            with (folder/'final.pkl').open('rb') as f: final = pickle.load(f)
            before, after = tree_hash(cp['agent']), tree_hash(final['agent'])
            assert cp['step'] == final['step'] == 50000 and before == after
            (folder/'evaluation_state_check.json').write_text(json.dumps(dict(task=task,batch_size=batch,
                agent_unchanged_during_final_evaluations=True,before_eval_hash=before,after_eval_hash=after),indent=2)+'\n')
            del final
            prefix = cp['replay'][:meta['chunks']]
            assert tree_hash(prefix) == meta['replay_hash']
            arrays = {k:np.asarray([r[k] for r in prefix]) for k in prefix[0]}
            if base is None: base = arrays
            diffs = difference(arrays, base)
            for k, a in arrays.items():
                if k in ('observations', 'next_observations'):
                    np.testing.assert_allclose(a, base[k], rtol=0, atol=1e-10)
                    np.testing.assert_allclose(a.astype(np.float32), base[k].astype(np.float32), rtol=0, atol=1e-12)
                else: np.testing.assert_array_equal(a, base[k])
            snap = OUT / (name + '.npz'); np.savez_compressed(snap, **arrays)
            rows[name] = dict(source_checkpoint=str((folder/'latest.pkl').relative_to(ROOT)),
                captured_checkpoint_sha256=source_sha, source_step=cp['step'], warmup_chunks=meta['chunks'],
                warmup_replay_hash=meta['replay_hash'], snapshot=str(snap.relative_to(ROOT)),
                snapshot_sha256=hashlib.sha256(snap.read_bytes()).hexdigest(), differences_from_current1024=diffs)
            historical = ROOT / f'runs/td_ablation/warmup_audit/task{task}_hard.npz'
            if historical.exists():
                with np.load(historical) as z: old = {k:z[k] for k in z.files}
                rows[name]['historical_snapshot'] = str(historical.relative_to(ROOT))
                rows[name]['historical_snapshot_sha256'] = hashlib.sha256(historical.read_bytes()).hexdigest()
                rows[name]['differences_from_historical1024'] = difference(arrays, old)
            del cp, prefix, arrays
    result = dict(status='passed', raw_observation_atol=1e-10, training_float32_atol=1e-12, arms=rows)
    (OUT/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Paired batch warmup audit passed', flush=True)


if __name__ == '__main__': main()
