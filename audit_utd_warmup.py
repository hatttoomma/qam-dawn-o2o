"""Check warmup against the fixed UTD=.25 controls and final evaluation state."""
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from audit_td_warmup import tree_hash
from audit_td_batch_warmup import difference

ROOT=Path(__file__).resolve().parent
RUNS=ROOT/'runs/utd_ablation'


def main():
    out=RUNS/'warmup_audit';out.mkdir(exist_ok=True);records={}
    for task in (1,2):
        folder=RUNS/f'task{task}_utd1'
        baseline=ROOT/f'runs/td_ablation/batch_comparison/warmup_audit/task{task}_b256.npz'
        with np.load(baseline) as z:base={k:z[k] for k in z.files}
        meta=json.loads((folder/'warmup.json').read_text())
        raw=(folder/'latest.pkl').read_bytes();source_sha=hashlib.sha256(raw).hexdigest()
        cp=pickle.loads(raw);del raw
        with (folder/'final.pkl').open('rb') as f:final=pickle.load(f)
        before,after=tree_hash(cp['agent']),tree_hash(final['agent'])
        assert cp['step']==final['step']==50000 and before==after
        (folder/'evaluation_state_check.json').write_text(json.dumps(dict(task=task,utd=1,
            agent_unchanged_during_final_evaluations=True,before_eval_hash=before,after_eval_hash=after),indent=2)+'\n')
        del final
        prefix=cp['replay'][:meta['chunks']];assert tree_hash(prefix)==meta['replay_hash']
        arrays={k:np.asarray([r[k] for r in prefix]) for k in prefix[0]}
        diffs=difference(arrays,base)
        for k,a in arrays.items():
            if k in ('observations','next_observations'):
                np.testing.assert_allclose(a,base[k],rtol=0,atol=1e-10)
                np.testing.assert_allclose(a.astype(np.float32),base[k].astype(np.float32),rtol=0,atol=1e-12)
            else:np.testing.assert_array_equal(a,base[k])
        snap=out/f'task{task}_utd1.npz';np.savez_compressed(snap,**arrays)
        records[f'task{task}']=dict(source_checkpoint=str((folder/'latest.pkl').relative_to(ROOT)),
            captured_checkpoint_sha256=source_sha,source_step=cp['step'],warmup_chunks=meta['chunks'],
            snapshot=str(snap.relative_to(ROOT)),snapshot_sha256=hashlib.sha256(snap.read_bytes()).hexdigest(),
            baseline_snapshot=str(baseline.relative_to(ROOT)),baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
            differences_from_utd025=diffs)
        del cp,prefix,arrays
    (out/'audit.json').write_text(json.dumps(dict(status='passed',raw_observation_atol=1e-10,
        float32_observation_atol=1e-12,tasks=records),indent=2)+'\n')
    print('UTD warmup and final evaluation state audit passed',flush=True)


if __name__=='__main__':main()
