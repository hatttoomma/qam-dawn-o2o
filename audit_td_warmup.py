"""Validate full base-only warmup replay after formal runs finish."""
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from jax.tree_util import tree_flatten_with_path

ROOT=Path(__file__).resolve().parent
RUNS=ROOT/'runs/td_ablation'
OUT=RUNS/'warmup_audit'

def tree_hash(tree):
    h=hashlib.sha256()
    for p,leaf in tree_flatten_with_path(tree)[0]:
        a=np.asarray(leaf);h.update(str(p).encode());h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()

def main():
    OUT.mkdir(exist_ok=True);rows={}
    for task in (1,2):
        base=None
        historical=ROOT/('runs/warm' if task==1 else 'runs/task2/warm')
        for kind,p in [('historical',historical),*[(k,RUNS/f'task{task}_{k}') for k in ('soft','hard')]]:
            meta=json.loads((p/'warmup.json').read_text())
            raw=(p/'latest.pkl').read_bytes();source_sha=hashlib.sha256(raw).hexdigest()
            checkpoint=pickle.loads(raw);del raw
            prefix=checkpoint['replay'][:meta['chunks']]
            assert tree_hash(prefix)==meta['replay_hash']
            arrays={k:np.asarray([r[k] for r in prefix]) for k in prefix[0]}
            if base is None:base=arrays
            assert arrays.keys()==base.keys()
            diffs={}
            for k,a in arrays.items():
                b=base[k];assert a.shape==b.shape and a.dtype==b.dtype and np.isfinite(a).all()
                x,y=a.astype(np.float32),b.astype(np.float32)
                if k in ('observations','next_observations'):
                    np.testing.assert_allclose(a,b,rtol=0,atol=1e-10)
                    np.testing.assert_allclose(x,y,rtol=0,atol=1e-12)
                else:np.testing.assert_array_equal(a,b)
                diffs[k]=dict(raw_changed_elements=int(np.sum(a!=b)),raw_max_abs_diff=float(np.max(np.abs(a-b))),
                    float32_max_abs_diff=float(np.max(np.abs(x-y))))
            name=f'task{task}_{kind}';snap=OUT/(name+'.npz');np.savez_compressed(snap,**arrays)
            rows[name]=dict(source_checkpoint=str((p/'latest.pkl').relative_to(ROOT)),
                captured_checkpoint_sha256=source_sha,source_step=checkpoint['step'],warmup_chunks=meta['chunks'],
                warmup_replay_hash=meta['replay_hash'],snapshot=str(snap.relative_to(ROOT)),
                snapshot_sha256=hashlib.sha256(snap.read_bytes()).hexdigest(),differences_from_historical=diffs)
            del checkpoint,prefix,arrays
    result=dict(status='passed',raw_observation_atol=1e-10,training_float32_atol=1e-12,arms=rows)
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print('Warmup audit passed',flush=True)

if __name__=='__main__':main()
