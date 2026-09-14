"""Verify full50k training states and online buffers for each continuation."""
import json
import os
from pathlib import Path
import pickle
import sys

ROOT=Path(os.environ.get('ANTMAZE_ROOT',Path(__file__).resolve().parent))
sys.path.insert(0,str(ROOT/'official'));sys.path.insert(1,str(ROOT/'legacy'))
import run as common
import jax
import numpy as np

manifest=json.loads((ROOT/'source_manifest.json').read_text())
for name,expected in manifest['files'].items():
    assert common.file_hash(ROOT/name)==expected,name
old=json.loads((ROOT.parent/'qam_antmaze/source_manifest.json').read_text())
assert manifest['files']==old['files'], 'Original online-only collection/update implementation must be unchanged'
inputs=json.loads((ROOT/'inputs.json').read_text())
tasks=[]
for task in range(1,6):
    parent=ROOT.parent/f'qam_antmaze_balanced40k/results/task{task}/dawn'
    done=json.loads((parent/'DONE.json').read_text())
    assert (done['steps'],done['updates'])==(50000,2500)
    assert json.loads((parent/'CHECKS_PASSED.json').read_text())['status']=='passed'
    assert common.file_hash(parent/'final.pkl')==done['checkpoint_sha256']
    assert common.file_hash(Path(inputs[str(task)]['native'])/'offline_500k.pkl')==inputs[str(task)]['offline_sha256']
    with (parent/'final.pkl').open('rb') as f:final=pickle.load(f)
    with (parent/'latest.pkl').open('rb') as f:resume=pickle.load(f)
    assert resume['step']==50000 and int(resume['agent']['updates'])==2500
    assert common.tree_hash(resume['agent'])==common.tree_hash(final['agent'])
    assert len(resume['replay'])==50000
    assert len(resume['keys'])==3 and len(resume['numpy_rng'])==5
    assert np.asarray(resume['ob']).shape==(29,)
    assert all(x['actions'].shape==(8,) and x['observations'].shape==(29,) for x in resume['replay'])
    required={'observations','actions','rewards','discounts','next_observations','base_actions','next_base_actions'}
    assert all(set(x)==required for x in resume['replay'])
    tasks.append(dict(task_id=task,parent=str(parent),resume_checkpoint_sha256=common.file_hash(parent/'latest.pkl'),
        published_final_sha256=done['checkpoint_sha256'],agent_state_hash=common.tree_hash(resume['agent']),
        online_replay_hash=common.tree_hash(resume['replay']),step=50000,updates=2500,online_buffer_size=50000,
        episode=int(resume['episode']),trace_length=len(resume['trace']),full_agent_matches_published_final=True))
    del resume,final
assert any(d.platform=='gpu' for d in jax.devices())
common.atomic_json(ROOT/'results/preflight.json',dict(status='passed',tasks=tasks,
    original_online_only_code_unchanged=True,all_five_full_resume_checkpoints_verified=True))
print(json.dumps(dict(status='passed',tasks=[x['task_id'] for x in tasks])))
