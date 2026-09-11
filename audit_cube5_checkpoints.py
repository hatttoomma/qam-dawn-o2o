"""Verify endpoint model state and complete warmup replay without new interaction."""
import json
from pathlib import Path
import pickle
import numpy as np
from audit_td_warmup import tree_hash
from launch_cube5 import digest, load

ROOT=Path(__file__).resolve().parent
RUNS=ROOT/'runs/cube5'


def locations(task):
    if task <= 2:
        old=ROOT/('runs' if task==1 else 'runs/task2')
        return old/'offline',old/'native',ROOT/f'runs/actor_input_ablation/task{task}_base_action'
    return tuple(RUNS/f'task{task}_{stage}' for stage in ('offline','native','warm'))


def main():
    out=RUNS/'checkpoint_audit';out.mkdir(exist_ok=True)
    rows=[]
    for task in range(1,6):
        offline,native,dawn=locations(task)
        od=load(offline/'DONE.json')
        assert digest(offline/'final.pkl')==od['checkpoint_sha256']
        with (offline/'final.pkl').open('rb') as f: cp=pickle.load(f)
        assert cp['step']==500000
        assert tree_hash(cp['agent']['network']['params']['modules_critic'])==od['q_hash']
        assert tree_hash(cp['agent']['network']['params']['modules_target_critic'])==od['target_q_hash']
        del cp
        for method,folder,updates in [('qam_edit',native,45001),('dawn',dawn,7500)]:
            done=load(folder/'DONE.json')
            assert digest(folder/'final.pkl')==done['checkpoint_sha256']
            with (folder/'latest.pkl').open('rb') as f: before=pickle.load(f)
            with (folder/'final.pkl').open('rb') as f: final=pickle.load(f)
            assert before['step']==final['step']==50000
            assert before['updates']==updates if method=='qam_edit' else int(before['agent']['updates'])==updates
            assert final['updates']==updates
            bh,fh=tree_hash(before['agent']),tree_hash(final['agent'])
            assert bh==fh
            row=dict(task=task,method=method,source_path=str(folder.relative_to(ROOT)),
                checkpoint_sha256=done['checkpoint_sha256'],latest_sha256=digest(folder/'latest.pkl'),
                offline_sha256=od['checkpoint_sha256'],updates=updates,
                before_final_evaluation_state_hash=bh,after_final_evaluation_state_hash=fh)
            if method=='dawn':
                a=final['agent']
                assert int(a['updates'])==int(a['actor']['step'])==int(a['critic']['step'])==7500
                assert a['actor']['params']['Dense_0']['kernel'].shape==(62,256)
                meta=load(folder/'warmup.json')
                prefix=before['replay'][:meta['chunks']]
                assert tree_hash(prefix)==meta['replay_hash']
                arrays={k:np.asarray([r[k] for r in prefix]) for k in prefix[0]}
                assert all(np.isfinite(a).all() for a in arrays.values())
                assert arrays['actions'].shape[1:]==arrays['base_actions'].shape[1:]==(25,)
                np.testing.assert_array_equal(arrays['actions'],arrays['base_actions'])
                snap=out/f'task{task}_warmup.npz';np.savez_compressed(snap,**arrays)
                row.update(warmup_replay_hash=meta['replay_hash'],warmup_chunks=meta['chunks'],
                    warmup_snapshot=str(snap.relative_to(ROOT)),warmup_snapshot_sha256=digest(snap),
                    actor_parameter_count=sum(int(x.size) for _,x in __import__('jax').tree_util.tree_flatten_with_path(a['actor']['params'])[0]))
                assert row['actor_parameter_count']==160562
                del prefix,arrays
            rows.append(row)
            del before,final
    result=dict(status='passed',entries=rows)
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('All five task pairs: model state and warmup audit passed',flush=True)


if __name__=='__main__':main()
