"""Validate completed continuations and compare with the exact50k parents."""
import argparse
import csv
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'output/antmaze_large_online60_utd00625_20260914'
parser=argparse.ArgumentParser()
parser.add_argument('--snapshot',type=Path,default=OUT/'full_snapshot.json')
args=parser.parse_args()
snapshot=json.loads(args.snapshot.read_text());assert snapshot['full']
files=snapshot['files']
assert files['results/preflight.json']['status']=='passed'
assert not any(k.endswith('/FAILED.json') for k in files)
parents=json.loads((ROOT/'output/antmaze_large_balanced_20260914/final_complete_snapshot.json').read_text())['files']
old1=json.loads((ROOT/'output/antmaze_large_task1_20260913/final_complete_snapshot.json').read_text())['files']
old_rest=json.loads((ROOT/'output/antmaze_large_tasks2_5_20260913/final_complete_snapshot.json').read_text())['files']

def finite(value):
    if isinstance(value,dict):return all(finite(x) for x in value.values())
    if isinstance(value,list):return all(finite(x) for x in value)
    if isinstance(value,(float,int)):return math.isfinite(value)
    return True

reference=json.loads((ROOT/'output/antmaze_large_online60_20260914/final_complete_snapshot.json').read_text())['files']
all_rows,completed=[],[]
for task in (1,2,3,5):
    prefix=f'results/task{task}/dawn/'
    if prefix+'CHECKS_PASSED.json' not in files or prefix+'runtime.json' not in files:continue
    get=lambda name:files[prefix+name]
    parent=lambda name:parents[prefix+name]
    old=old1 if task==1 else old_rest
    old_prefix='results/' if task==1 else f'results/task{task}/'
    prior=lambda name:old[old_prefix+name]
    done,checks,cfg=get('DONE.json'),get('CHECKS_PASSED.json'),get('actual_agent_config.json')
    assert checks['status']=='passed' and (done['steps'],done['updates'],checks['task_id'])==(60000,3125,task)
    assert (checks['additional_steps'],checks['additional_updates'])==(10000,625)
    assert checks['evaluation_steps_verified']==[60000]
    for key in ('exact50k_agent_and_optimizers_resumed','online_replay_restored','environment_restored',
                'frozen_base_verified','online_only_replay_verified','unchanged_update_equations'):
        assert checks[key]
    expected=dict(online_steps=60000,warmup=40000,batch_size=256,utd=.0625,seed=0,task_id=task,
        resume_env_step=50000,resume_updates=2500,additional_env_steps=10000,additional_warmup=0,
        online_batch_size=256,offline_batch_size=0,replay='online_only_uniform_growing_chunk_buffer',
        fresh_critic_optimizer=False,resumed_actor_critic_target_alpha_and_optimizers=True,
        checkpoint_steps=[50000,60000])
    for key,value in expected.items():assert cfg[key]==value,(key,cfg.get(key))
    allowed={'online_steps','checkpoint_steps','replay','offline_batch_size','online_batch_size',
             'offline_base_actions','fresh_critic_optimizer','source_files','utd'}
    assert {k for k in parent('actual_agent_config.json') if cfg.get(k)!=parent('actual_agent_config.json')[k]}<=allowed
    assert get('parameter_comparison.json')['all_other_hyperparameters_identical']
    assert get('dataset_manifest.json')==parent('dataset_manifest.json')
    resume=get('resume_audit.json');pre=next(x for x in files['results/preflight.json']['tasks'] if x['task_id']==task)
    assert pre['task_id']==task
    for key in ('agent_state_hash','online_replay_hash','resume_checkpoint_sha256','published_final_sha256'):
        assert resume[key]==pre[key]
    assert resume['published_final_sha256']==parent('DONE.json')['checkpoint_sha256']
    assert resume['all_agent_parameters_and_optimizers_exact'] and resume['online_buffer_size']==50000
    ref_config=reference[prefix+'actual_agent_config.json']
    assert {k for k in set(cfg)|set(ref_config) if cfg.get(k)!=ref_config.get(k)}=={'utd','checkpoint_steps','source_files'}
    assert get('utd_reference_comparison.json')['all_other_hyperparameters_identical']
    ref_resume=reference[prefix+'resume_audit.json']
    for key in ('agent_state_hash','online_replay_hash','resume_checkpoint_sha256','rollout_rng_hash',
                'numpy_rng_hash','actor_optimizer_hash','critic_optimizer_hash','alpha_optimizer_hash'):
        assert resume[key]==ref_resume[key],key
    assert not any('055000' in k for k in files if k.startswith(prefix))

    initial=get('initial_hashes.json')
    assert initial==done['initial_hashes']
    for a,b in [('actor','actor_hash'),('critic','critic_hash'),('actor_optimizer','actor_optimizer_hash'),
                ('critic_optimizer','critic_optimizer_hash'),('target','target_hash')]:
        if a in initial:assert initial[a]==resume[b]
    assert initial['alpha']==resume['alpha']
    assert initial['flow']==done['final_flow_hash']==parent('DONE.json')['final_flow_hash']
    assert get('environment_restored.json')['status']=='passed'
    assert get('environment_restored.json')['observation_matched']
    assert prefix+'warmup.json' not in files
    audit=get('replay_audit.json')
    assert audit['status']=='passed' and (audit['online_per_batch'],audit['offline_per_batch'])==(256,0)
    assert (audit['online_buffer_size'],audit['cumulative_updates'],audit['additional_updates'])==(60000,3125,625)
    assert (audit['additional_online_draws'],audit['additional_offline_draws'])==(160000,0)
    assert audit['warmup_and_prior_online_retained']
    assert get('runtime.json')['returncode']==0
    assert finite(get('metrics.jsonl'))
    for row in get('metrics.jsonl'):
        assert 50000<row['step']<=60000 and row['updates']==2500+(row['step']-50000)//16
    assert get('metrics.jsonl')[-1]['step']==60000
    for step in (60000,):
        marker=get(f'eval_rng_preserved_{step:06d}.json');assert marker['status']=='passed' and marker['numpy_rng_unchanged']
    model=snapshot['checkpoint_audit'][prefix+'final.pkl'];assert model['matched']
    for suffix in ('','_mean_residual'):
        name=f'eval_050000_100{suffix}.json'
        assert get(name)==parent(name)
    base=prior('native/fixed_eval/eval_000000_100.json')
    sources=[('Offline QAM',base,0,'original offline500k'),
        ('QAM native',prior('native/fixed_eval/eval_050000_100.json'),45001,'original QAM native50k'),
        ('Old DAWN online-only mean',prior('dawn/eval_100000_100_mean_residual.json'),5000,'old DAWN100k')]
    for step in (50000,60000):
        for mode,suffix in [('mean','_mean_residual'),('sampled','')]:
            name=f'eval_{step:06d}_100{suffix}.json'
            method=('Parent balanced ' if step==50000 else 'Continuation UTD0.0625 ')+mode
            sources.append((method,get(name),2500+(step-50000)//16,prefix+name))
    for mode,suffix in [('mean','_mean_residual'),('sampled','')]:
        name=f'eval_060000_100{suffix}.json'
        sources.append(('Reference UTD0.25 '+mode,reference[prefix+name],5000,'previous continuation/'+prefix+name))
    rows=[]
    for method,evaluation,updates,source in sources:
        records=evaluation['records'];assert len(records)==evaluation['episodes']==100
        assert [r['episode'] for r in records]==list(range(100))
        assert all((a['reset_seed'],a['initial_hash'])==(b['reset_seed'],b['initial_hash']) for a,b in zip(records,base['records']))
        assert all(r['success'] in (0,1) and r['return_']==-r['length']+r['success'] for r in records)
        success=sum(r['success'] for r in records)/100;returns=sum(r['return_'] for r in records)/100
        assert math.isclose(success,evaluation['success'],abs_tol=1e-12)
        assert math.isclose(returns,evaluation['return_mean'],abs_tol=1e-9)
        rows.append(dict(task=task,method=method,online_env_steps=evaluation['step'],online_updates=updates,
            success_rate=success,avg_return=returns,source=source))
    for name,value in files.items():
        if not name.startswith(prefix):continue
        target=OUT/'synced_results'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(''.join(json.dumps(r)+'\n' for r in value) if name.endswith('.jsonl') else json.dumps(value,indent=2))
    validation=dict(status='passed',task=task,snapshot_utc=snapshot['utc'],
        exact_resume_and_optimizer_state_verified=True,online_buffer_retained=True,online_only_updates_verified=True,
        frozen_base_verified=True,paired_evaluation_and_aggregates_verified=True,checkpoint_audit=model,
        wall_seconds=get('runtime.json')['wall_seconds'],summary=rows)
    (OUT/f'task{task}_completion_validation.json').write_text(json.dumps(validation,indent=2))
    lines=[f'Task {task} (seed0, 100 paired episodes)','',
        '|Method|Online env steps|Updates|Success|Avg return|','|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f"|{r['method']}|{r['online_env_steps']}|{r['online_updates']}|{r['success_rate']:.0%}|{r['avg_return']:.2f}|")
    lines+=['','Both continuation arms resume the same50k state. Only continuation UTD differs; the new arm omits55k evaluation. Report means across tasks1,2,3,5 only.']
    (OUT/f'task{task}_result.md').write_text('\n'.join(lines)+'\n')
    all_rows.extend(rows);completed.append(task)
if all_rows:
    with (OUT/'completed_tasks_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
result=dict(completed_verified=completed,summary=all_rows)
(OUT/'completed_validation_summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
