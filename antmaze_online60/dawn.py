"""Resume balanced50k completely; change only new training batches to online-only."""
import fcntl
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
from types import SimpleNamespace

ROOT=Path(os.environ.get('ANTMAZE_ROOT',Path(__file__).resolve().parent))
sys.path.insert(0,str(ROOT/'official'))
sys.path.insert(1,str(ROOT/'legacy'))
import run as common
import paired_evaluation
import actor_input_agent as implementation
from agents.qam import QAMAgent,get_config
import flax
import jax
import jax.numpy as jnp
import numpy as np

TASK=int(os.environ['ANTMAZE_TASK']);assert TASK in range(1,6)
SMOKE=os.environ.get('ANTMAZE_SMOKE')=='1'
OUT=Path(os.environ['ANTMAZE_OUT']);OUT.mkdir(parents=True,exist_ok=True)
NATIVE=Path(os.environ['ANTMAZE_NATIVE'])
PARENT=ROOT.parent/f'qam_antmaze_balanced40k/results/task{TASK}/dawn'
lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
common.ENV=f'antmaze-large-navigate-singletask-task{TASK}-v0'
common.HORIZON,common.GAMMA=1,.99
common.GRID=(50000,50004,50008) if SMOKE else (50000,55000,60000)
args=SimpleNamespace(stage='warm',out=str(OUT),data='/root/.ogbench/data',seed=0,
    offline_steps=500000,online_steps=50008 if SMOKE else 60000,
    offline_checkpoint=str(NATIVE/'offline_500k.pkl'),eval_episodes=2 if SMOKE else 100,
    final_episodes=2 if SMOKE else 100,warmup=40000,dawn_batch=256,
    td_target='naive',actor_input='state+base_action',utd=.25,task_id=TASK,
    environment=common.ENV,checkpoint_steps=list(common.GRID),resume_env_step=50000,
    resume_updates=2500,additional_env_steps=8 if SMOKE else 10000,
    replay='online_only_uniform_growing_chunk_buffer',offline_batch_size=0,online_batch_size=256)
common.atomic_json(OUT/'requested_config.json',vars(args))
parent_done=json.loads((PARENT/'DONE.json').read_text())
assert (parent_done['steps'],parent_done['updates'])==(50000,2500)
assert json.loads((PARENT/'CHECKS_PASSED.json').read_text())['status']=='passed'
assert common.file_hash(PARENT/'final.pkl')==parent_done['checkpoint_sha256']
offline=json.loads((NATIVE/'offline_checkpoint.json').read_text())
assert common.file_hash(args.offline_checkpoint)==offline['sha256']
manifest=json.loads((ROOT/'source_manifest.json').read_text())
for relative,expected in manifest['files'].items():
    assert common.file_hash(ROOT/relative)==expected,relative
source_resume=json.loads((ROOT/'results/preflight.json').read_text())['tasks'][TASK-1]
assert source_resume['task_id']==TASK
assert common.file_hash(PARENT/'latest.pkl')==source_resume['resume_checkpoint_sha256']
if not (OUT/'latest.pkl').exists():
    shutil.copy2(PARENT/'latest.pkl',OUT/'latest.pkl')
assert common.file_hash(OUT/'latest.pkl')==source_resume['resume_checkpoint_sha256'], 'Inspect any partial continuation before resuming'
for suffix in ('','_mean_residual'):
    if not SMOKE:
        name=f'eval_050000_100{suffix}.json'
        shutil.copy2(PARENT/name,OUT/name)

def template(obs,action,seed):
    cfg=get_config();cfg.horizon_length,cfg.action_chunking=1,True
    cfg.inv_temp,cfg.fql_alpha,cfg.edit_scale=10.,0.,0.
    return QAMAgent.create(seed,obs,action,cfg)

def hard_backup(reward,discount,next_qs,alpha,next_logp):
    return reward+discount*jnp.min(next_qs,axis=0)

original_data,original_load,original_save=common.make_data,common.load_checkpoint,common.save_checkpoint
original_restore,original_stack=common.restore_env,common.stack_replay
original_eval=paired_evaluation.evaluate

def load_data(config):
    env,ds=original_data(config)
    expected=json.loads((PARENT/'dataset_manifest.json').read_text())
    assert expected['task_id']==TASK and env.unwrapped._reward_task_id==TASK
    assert expected['dataset_size']==ds.size
    for name,spec in expected['files'].items():
        assert common.file_hash(Path(config.data)/name)==spec['sha256']
    common.atomic_json(OUT/'dataset_manifest.json',expected)
    return env,ds

class Factory:
    @staticmethod
    def create(qam,obs,seed,warm):
        assert warm and common.flow_hash(qam)==offline['flow_hash']==parent_done['final_flow_hash']
        template_agent=implementation.ActorInputAgent.create(qam,obs,seed,warm).replace(condition_on_base=True)
        agent,sv=original_load(PARENT/'latest.pkl',template_agent)
        final,_=original_load(PARENT/'final.pkl',template_agent)
        state=flax.serialization.to_state_dict(agent)
        assert common.tree_hash(state)==common.tree_hash(flax.serialization.to_state_dict(final))==source_resume['agent_state_hash']
        assert sv['step']==50000 and int(agent.updates)==2500 and len(sv['replay'])==50000
        old=json.loads((PARENT/'actual_agent_config.json').read_text())
        config=dict(old,online_steps=args.online_steps,checkpoint_steps=list(common.GRID),
            replay=args.replay,offline_batch_size=0,online_batch_size=256,offline_base_actions='unused',
            fresh_critic_optimizer=False,source_files=manifest['files'],
            resumed_actor_critic_target_alpha_and_optimizers=True,resume_env_step=50000,resume_updates=2500,
            additional_env_steps=args.additional_env_steps,additional_warmup=0)
        differences={k:[old[k],config.get(k)] for k in old if old[k]!=config.get(k)}
        allowed={'online_steps','checkpoint_steps','replay','offline_batch_size','online_batch_size',
                 'offline_base_actions','fresh_critic_optimizer','source_files'}
        assert set(differences)<=allowed,differences
        common.atomic_json(OUT/'actual_agent_config.json',config)
        common.atomic_json(OUT/'parameter_comparison.json',dict(status='passed',
            changed_existing_fields=differences,all_other_hyperparameters_identical=True,
            optimizer_state_carried_over=True))
        if SMOKE:
            for deterministic,suffix in ((False,''),(True,'_mean_residual')):
                result=original_eval(qam,OUT,50000,args,dawn=agent,deterministic=deterministic,suffix=suffix)
                parent=json.loads((PARENT/f'eval_050000_100{suffix}.json').read_text())
                assert result['records']==parent['records'][:2], 'Restored50k policy changed'
            common.atomic_json(OUT/'RESUME_EVAL_MATCHED.json',dict(status='passed',episodes=2,
                sampled_and_mean_records_exact=True))
        return agent

def load(path,agent):
    loaded,sv=original_load(path,agent)
    if Path(path)==OUT/'latest.pkl':
        assert sv['step']==50000 and int(loaded.updates)==2500
        assert common.tree_hash(flax.serialization.to_state_dict(loaded))==source_resume['agent_state_hash']
        replay_hash=common.tree_hash(sv['replay'])
        assert replay_hash==source_resume['online_replay_hash'] and len(sv['replay'])==50000
        audit=dict(source_resume,status='passed',resumed_env_step=50000,resumed_updates=2500,
            online_buffer_size=50000,additional_warmup=0,all_agent_parameters_and_optimizers_exact=True,
            rollout_rng_hash=common.tree_hash(sv['keys']),numpy_rng_hash=common.tree_hash(sv['numpy_rng']),
            alpha=float(jnp.exp(loaded.log_alpha)),actor_hash=common.tree_hash(loaded.actor.params),
            critic_hash=common.tree_hash(loaded.critic.params),target_hash=common.tree_hash(loaded.target_params),
            actor_optimizer_hash=common.tree_hash(loaded.actor.opt_state),
            critic_optimizer_hash=common.tree_hash(loaded.critic.opt_state),
            alpha_optimizer_hash=common.tree_hash(loaded.alpha_opt_state))
        common.atomic_json(OUT/'resume_audit.json',audit)
    return loaded,sv

def restore(env,episode,seed,trace,expected):
    obs=original_restore(env,episode,seed,trace,expected)
    common.atomic_json(OUT/'environment_restored.json',dict(status='passed',episode=episode,
        action_trace_length=len(trace),observation_hash=common.tree_hash(obs),
        expected_observation_hash=common.tree_hash(expected),observation_matched=True))
    return obs

def stack(records,indices):
    assert len(indices)==256 and len(records)>50000
    batch=original_stack(records,indices)
    assert all(v.shape[0]==256 and np.isfinite(v).all() for v in batch.values())
    additional_updates=(len(records)-50000)//4
    if additional_updates==1 or additional_updates%250==0 or len(records)==args.online_steps:
        common.atomic_json(OUT/'replay_audit.json',dict(status='passed',online_per_batch=256,
            offline_per_batch=0,online_buffer_size=len(records),warmup_and_prior_online_retained=True,
            cumulative_updates=2500+additional_updates,additional_updates=additional_updates,
            additional_online_draws=additional_updates*256,additional_offline_draws=0,
            sampled_prior50k=int((indices<50000).sum()),sampled_new10k=int((indices>=50000).sum())))
    return batch

def evaluate(qam,folder,step,config,**kwargs):
    before=common.tree_hash(np.random.get_state())
    result=original_eval(qam,folder,step,config,**kwargs)
    if not kwargs.get('deterministic',False):
        original_eval(qam,folder,step,config,**dict(kwargs,deterministic=True,
            suffix='_mean_residual',episodes=args.eval_episodes))
    assert common.tree_hash(np.random.get_state())==before
    common.atomic_json(OUT/f'eval_rng_preserved_{step:06d}.json',dict(status='passed',
        step=step,numpy_rng_unchanged=True))
    return result

def save(path,agent,**payload):
    assert int(agent.updates)==(payload['step']-40000)//4
    original_save(path,agent,**payload)
    if Path(path).name=='latest.pkl':
        original_save(OUT/f'checkpoint_{payload["step"]:06d}.pkl',agent,
            step=payload['step'],updates=int(agent.updates))

implementation.soft_backup=hard_backup
common.qam_template,common.DawnAgent=template,Factory
common.reset_train=lambda env,episode,seed:paired_evaluation.deterministic_reset(env,100000+seed*10000+episode)[0]
common.make_data,common.load_checkpoint,common.restore_env=load_data,load,restore
common.stack_replay,common.evaluate,common.save_checkpoint=stack,evaluate,save
try:
    common.residual(args)
    done=json.loads((OUT/'DONE.json').read_text());audit=json.loads((OUT/'replay_audit.json').read_text())
    assert done['steps']==args.online_steps and done['updates']==2500+args.additional_env_steps//4
    assert done['final_flow_hash']==offline['flow_hash']
    assert common.file_hash(OUT/'final.pkl')==done['checkpoint_sha256']
    assert audit['additional_updates']==args.additional_env_steps//4 and audit['additional_offline_draws']==0
    assert audit['online_buffer_size']==args.online_steps
    for step in common.GRID[1:]:
        for suffix in ('','_mean_residual'):
            e=json.loads((OUT/f'eval_{step:06d}_{args.eval_episodes:03d}{suffix}.json').read_text())
            assert e['step']==step and e['episodes']==args.eval_episodes
    common.atomic_json(OUT/'CHECKS_PASSED.json',dict(status='passed',task_id=TASK,
        exact50k_agent_and_optimizers_resumed=True,online_replay_restored=True,environment_restored=True,
        frozen_base_verified=True,online_only_replay_verified=True,unchanged_update_equations=True,
        steps=done['steps'],updates=done['updates'],additional_steps=args.additional_env_steps,
        additional_updates=args.additional_env_steps//4,evaluation_steps_verified=list(common.GRID[1:])))
except BaseException as exc:
    common.atomic_json(OUT/'FAILED.json',dict(type=type(exc).__name__,message=str(exc)))
    raise
