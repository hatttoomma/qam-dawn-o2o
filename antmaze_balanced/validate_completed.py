"""Local completion checks and comparisons against preserved prior experiments."""
import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'output/antmaze_large_balanced_20260914'
parser = argparse.ArgumentParser()
parser.add_argument('--snapshot', type=Path, default=OUT/'full_snapshot.json')
args = parser.parse_args()
snapshot = json.loads(args.snapshot.read_text())
assert snapshot['full']
files = snapshot['files']
assert files['results/preflight.json']['status'] == 'passed'
assert not any(k.endswith('/FAILED.json') for k in files)
old1 = json.loads((ROOT/'output/antmaze_large_task1_20260913/final_complete_snapshot.json').read_text())['files']
old_rest = json.loads((ROOT/'output/antmaze_large_tasks2_5_20260913/final_complete_snapshot.json').read_text())['files']

def finite(value):
    if isinstance(value,dict): return all(finite(x) for x in value.values())
    if isinstance(value,list): return all(finite(x) for x in value)
    if isinstance(value,(float,int)): return math.isfinite(value)
    return True

all_rows, completed = [], []
for task in range(1,6):
    prefix=f'results/task{task}/dawn/'
    if prefix+'CHECKS_PASSED.json' not in files or prefix+'runtime.json' not in files:
        continue
    get=lambda name:files[prefix+name]
    prior_files=old1 if task==1 else old_rest
    prior_prefix='results/' if task==1 else f'results/task{task}/'
    prior=lambda name:prior_files[prior_prefix+name]
    done,checks,cfg=get('DONE.json'),get('CHECKS_PASSED.json'),get('actual_agent_config.json')
    assert checks['status']=='passed' and checks['balanced_replay_verified']
    assert (done['steps'],done['updates'],checks['task_id'])==(50000,2500,task)
    assert checks['evaluation_steps_verified']==[45000,50000]
    for key in ('unchanged_collection_and_update_equations','native_gate_passed','offline_checkpoint_matched',
                'hard_TD_verified','frozen_base_verified'):
        assert checks[key]
    for key,value in dict(warmup=40000,online_steps=50000,batch_size=256,utd=.25,
                          offline_batch_size=128,online_batch_size=128,replay='balanced_offline_online',
                          seed=0,task_id=task,checkpoint_steps=[0,40000,45000,50000]).items():
        assert cfg[key]==value,(key,cfg[key])
    old_cfg=prior('dawn/actual_agent_config.json')
    allowed={'warmup','online_steps','replay','checkpoint_steps','source_files'}
    assert {k for k in old_cfg if cfg.get(k)!=old_cfg[k]} <= allowed
    assert get('parameter_comparison.json')['all_other_parameters_identical']
    assert get('dataset_manifest.json')==prior('native/dataset_manifest.json')
    initial,offline=get('initial_hashes.json'),prior('native/offline_checkpoint.json')
    assert get('config.json')['offline_sha256']==offline['sha256']
    assert initial==done['initial_hashes']
    assert initial['flow']==done['final_flow_hash']==offline['flow_hash']
    assert initial['critic']==offline['q_hash'] and initial['target']==offline['target_q_hash']
    warm=get('warmup.json')
    assert warm['steps']==warm['requested_steps']==warm['chunks']==40000
    assert warm['actor_hash']==initial['actor'] and warm['critic_hash']==initial['critic']
    assert get('INITIAL_EVAL_MATCHED.json')['status']=='passed'
    for step in (45000,50000):
        assert get(f'eval_rng_preserved_{step:06d}.json')['numpy_rng_unchanged']
    audit=get('replay_audit.json')
    assert audit['status']=='passed' and audit['update']==2500
    assert audit['offline_per_batch']==audit['online_per_batch']==128
    assert audit['offline_draws']==audit['online_draws']==320000
    assert audit['online_buffer_size']==50000 and audit['offline_buffer_size']==1000000
    model=snapshot['checkpoint_audit'][prefix+'final.pkl']
    assert model['matched'] and model['sha256']==done['checkpoint_sha256']
    assert get('runtime.json')['returncode']==0 and finite(get('metrics.jsonl'))
    for row in get('metrics.jsonl'):
        assert row['updates']==max(0,row['step']-40000)//4
    base=prior('native/fixed_eval/eval_000000_100.json')
    assert base['records']==get('eval_000000_100.json')['records']
    evaluation_sources=[('Offline QAM',base,0,'previous native offline'),
        ('QAM native',prior('native/fixed_eval/eval_050000_100.json'),45001,'previous native 50k'),
        ('DAWN online-only mean',prior('dawn/eval_100000_100_mean_residual.json'),5000,'previous DAWN 100k')]
    for step in (45000,50000):
        for method,suffix in [('DAWN balanced mean','_mean_residual'),('DAWN balanced sampled','')]:
            name=f'eval_{step:06d}_100{suffix}.json'
            evaluation_sources.append((method,get(name),(step-40000)//4,prefix+name))
    rows=[]
    for method,evaluation,updates,source in evaluation_sources:
        records=evaluation['records']
        assert len(records)==evaluation['episodes']==100
        assert [r['episode'] for r in records]==list(range(100))
        assert all((a['reset_seed'],a['initial_hash'])==(b['reset_seed'],b['initial_hash']) for a,b in zip(records,base['records']))
        assert all(r['success'] in (0,1) and r['return_']==-r['length']+r['success'] for r in records)
        success=sum(r['success'] for r in records)/100
        returns=sum(r['return_'] for r in records)/100
        assert math.isclose(success,evaluation['success'],abs_tol=1e-12)
        assert math.isclose(returns,evaluation['return_mean'],abs_tol=1e-9)
        rows.append(dict(task=task,method=method,online_env_steps=evaluation['step'],online_updates=updates,
                         success_rate=success,avg_return=returns,
                         gained_vs_offline=sum(a['success']>b['success'] for a,b in zip(records,base['records'])),
                         lost_vs_offline=sum(a['success']<b['success'] for a,b in zip(records,base['records'])),source=source))
    for name,value in files.items():
        if not name.startswith(prefix):continue
        target=OUT/'synced_results'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(''.join(json.dumps(r)+'\n' for r in value) if name.endswith('.jsonl') else json.dumps(value,indent=2))
    result=dict(status='passed',task=task,snapshot_utc=snapshot['utc'],configuration_verified=True,
        source_ratio_verified=True,checkpoint_audit=model,initial_records_exact_match=True,
        paired_evaluation_verified=True,evaluation_aggregates_recomputed=True,update_counts_verified=True,
        inherited_Q_and_frozen_base_verified=True,wall_seconds=get('runtime.json')['wall_seconds'],summary=rows)
    (OUT/f'task{task}_completion_validation.json').write_text(json.dumps(result,indent=2))
    lines=[f'Task {task} (seed0, 100 evaluation episodes)','',
        '|Method|Online env steps|Updates|Success|Avg return|','|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"|{r['method']}|{r['online_env_steps']}|{r['online_updates']}|{r['success_rate']:.0%}|{r['avg_return']:.2f}|")
    lines+=['','The new arm changes both warmup/training budget and replay; it does not isolate replay alone.']
    (OUT/f'task{task}_result.md').write_text('\n'.join(lines)+'\n')
    all_rows.extend(rows)
    completed.append(task)
if all_rows:
    with (OUT/'completed_tasks_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
result=dict(completed_verified=completed,summary=all_rows)
(OUT/'completed_validation_summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
