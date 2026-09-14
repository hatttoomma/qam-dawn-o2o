"""Continue exact existing DAWN states to100k with unchanged low-UTD core."""
import fcntl,json,os,pickle,shutil,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'official'),str(ROOT/'legacy')]
import run as common
import paired_evaluation
import actor_input_agent as impl
from agents.qam import QAMAgent,get_config
import flax,jax,jax.numpy as jnp,numpy as np
TASK=int(os.environ['TASK']);SMOKE=os.environ.get('SMOKE')=='1'
OUT=Path(os.environ['OUT']);OUT.mkdir(parents=True,exist_ok=True)
lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
spec=json.loads((ROOT/'inputs.json').read_text())[str(TASK)]
PARENT=Path(spec['parent']);NATIVE=Path(spec['native']);START=spec['resume_step'];UPDATES=2500+(START-50000)//16
TOTAL=START+32 if SMOKE else 100000
common.ENV=f'antmaze-large-navigate-singletask-task{TASK}-v0';common.HORIZON=1;common.GAMMA=.99;common.GRID=(START,TOTAL)
args=SimpleNamespace(stage='warm',out=str(OUT),data='/root/.ogbench/data',seed=0,offline_steps=500000,
 online_steps=TOTAL,offline_checkpoint=str(NATIVE/'offline_500k.pkl'),eval_episodes=2 if SMOKE else 100,
 final_episodes=2 if SMOKE else 100,warmup=40000,dawn_batch=256,utd=.0625,td_target='naive',actor_input='state+base_action')
manifest=json.loads((ROOT/'source_manifest.json').read_text())
for name,digest in manifest['files'].items():assert common.file_hash(ROOT/name)==digest,name
assert common.file_hash(args.offline_checkpoint)==spec['offline_sha256']
meta=json.loads((NATIVE/'offline_checkpoint.json').read_text());done=json.loads((PARENT/'DONE.json').read_text())
assert (done['steps'],done['updates'])==(START,UPDATES)
assert json.loads((PARENT/'CHECKS_PASSED.json').read_text())['status']=='passed'
assert common.file_hash(PARENT/'final.pkl')==done['checkpoint_sha256']
with (PARENT/'latest.pkl').open('rb') as f:payload=pickle.load(f)
with (PARENT/'final.pkl').open('rb') as f:published=pickle.load(f)
assert payload['step']==START and int(payload['agent']['updates'])==UPDATES and len(payload['replay'])==START
state_hash=common.tree_hash(payload['agent']);assert state_hash==common.tree_hash(published['agent'])
audit=dict(parent=str(PARENT),checkpoint_sha256=common.file_hash(PARENT/'latest.pkl'),agent_state_hash=state_hash,
 replay_hash=common.tree_hash(payload['replay']),numpy_rng_hash=common.tree_hash(payload['numpy_rng']),
 rollout_keys_hash=common.tree_hash(payload['keys']),resume_step=START,resume_updates=UPDATES,full_state_matches_published=True)
common.atomic_json(OUT/'resume_source.json',audit)
assert not (OUT/'latest.pkl').exists(),'Partial run requires inspection'
shutil.copy2(PARENT/'latest.pkl',OUT/'latest.pkl')
if not SMOKE:
 for p in PARENT.glob('eval_*_100*.json'):shutil.copy2(p,OUT/p.name)
common.atomic_json(OUT/'requested_config.json',dict(vars(args),task=TASK,resume_step=START,resume_updates=UPDATES,
 additional_env_steps=TOTAL-START,additional_updates=(TOTAL-START)//16,additional_warmup=0,evaluation_steps=[TOTAL]))
del payload,published
original_load,original_save,original_restore,original_data=common.load_checkpoint,common.save_checkpoint,common.restore_env,common.make_data
original_eval=paired_evaluation.evaluate

def template(obs,act,seed):
 cfg=get_config();cfg.horizon_length=1;cfg.action_chunking=True;cfg.inv_temp=10.;cfg.edit_scale=0.;cfg.fql_alpha=0.
 return QAMAgent.create(seed,obs,act,cfg)

class Factory:
 @staticmethod
 def create(qam,obs,seed,warm):
  assert warm and common.flow_hash(qam)==meta['flow_hash']
  agent=impl.ActorInputAgent.create(qam,obs,seed,True).replace(condition_on_base=True)
  agent,_=original_load(PARENT/'latest.pkl',agent)
  assert common.tree_hash(flax.serialization.to_state_dict(agent))==state_hash
  cfg=json.loads((PARENT/'actual_agent_config.json').read_text())
  cfg.update(online_steps=TOTAL,utd=.0625,replay='online_only_uniform_growing_chunk_buffer',offline_batch_size=0,
    online_batch_size=256,offline_base_actions='unused',fresh_critic_optimizer=False,checkpoint_steps=[START,TOTAL],
    resume_env_step=START,resume_updates=UPDATES,additional_env_steps=TOTAL-START,additional_warmup=0,
    resumed_actor_critic_target_alpha_and_optimizers=True,source_files=manifest['files'])
  common.atomic_json(OUT/'actual_agent_config.json',cfg)
  if SMOKE:
   for deterministic,suffix in [(False,''),(True,'_mean_residual')]:
    result=original_eval(qam,OUT,START,args,dawn=agent,deterministic=deterministic,suffix=suffix)
    ref=json.loads((PARENT/f'eval_{START:06d}_100{suffix}.json').read_text())
    assert result['records']==ref['records'][:2]
   common.atomic_json(OUT/'RESUME_EVAL_MATCHED.json',dict(status='passed'))
  return agent

def load(path,agent):
 agent,p=original_load(path,agent)
 if Path(path)!=OUT/'latest.pkl':return agent,p
 assert p['step']==START and int(agent.updates)==UPDATES
 assert common.tree_hash(flax.serialization.to_state_dict(agent))==state_hash
 assert common.tree_hash(p['replay'])==audit['replay_hash'] and common.tree_hash(p['numpy_rng'])==audit['numpy_rng_hash']
 assert common.tree_hash(p['keys'])==audit['rollout_keys_hash']
 common.atomic_json(OUT/'RESTORE_PASSED.json',dict(status='passed',**audit));return agent,p

def restore(env,episode,seed,trace,expected):
 ob=original_restore(env,episode,seed,trace,expected)
 common.atomic_json(OUT/'environment_restored.json',dict(status='passed',max_abs_error=float(np.max(np.abs(ob-expected)))))
 return ob

def data(config):
 env,ds=original_data(config);assert env.unwrapped._reward_task_id==TASK
 expected=json.loads((PARENT/'dataset_manifest.json').read_text());assert expected['dataset_size']==ds.size
 for name,info in expected['files'].items():assert common.file_hash(Path(config.data)/name)==info['sha256']
 common.atomic_json(OUT/'dataset_manifest.json',expected);return env,ds

original_stack=common.stack_replay

def stack(replay,idx):
 assert len(idx)==256 and len(replay)>START
 batch=original_stack(replay,idx);assert all(np.isfinite(v).all() for v in batch.values())
 step=len(replay);u=2500+(step-50000)//16
 if step%2000==0 or step in [START+16,TOTAL]:
  common.atomic_json(OUT/'replay_audit.json',dict(online_per_batch=256,offline_per_batch=0,online_buffer_size=step,
   updates=u,additional_updates=u-UPDATES,original_online_buffer_retained=True))
 return batch

def evaluate(qam,out,step,config,**kwargs):
 assert step==TOTAL
 rng=common.tree_hash(np.random.get_state());agent_hash=common.tree_hash(flax.serialization.to_state_dict(kwargs['dawn']))
 result=original_eval(qam,out,step,config,**kwargs)
 if not kwargs.get('deterministic',False):original_eval(qam,out,step,config,**dict(kwargs,deterministic=True,suffix='_mean_residual'))
 assert rng==common.tree_hash(np.random.get_state()) and agent_hash==common.tree_hash(flax.serialization.to_state_dict(kwargs['dawn']))
 common.atomic_json(OUT/'EVAL_INTEGRITY.json',dict(status='passed',step=step,training_state_unchanged=True));return result

def save(path,agent,**kw):
 assert int(agent.updates)==2500+(kw['step']-50000)//16
 original_save(path,agent,**kw)

impl.soft_backup=lambda r,d,q,a,lp:r+d*jnp.min(q,axis=0)
common.qam_template,common.DawnAgent=template,Factory
common.make_data,common.load_checkpoint,common.restore_env,common.stack_replay=data,load,restore,stack
common.reset_train=lambda env,ep,seed:paired_evaluation.deterministic_reset(env,100000+seed*10000+ep)[0]
common.evaluate,common.save_checkpoint=evaluate,save
try:
 common.residual(args)
 final=json.loads((OUT/'DONE.json').read_text())
 assert (final['steps'],final['updates'])==(TOTAL,2500+(TOTAL-50000)//16)
 assert final['final_flow_hash']==meta['flow_hash'] and common.file_hash(OUT/'final.pkl')==final['checkpoint_sha256']
 common.atomic_json(OUT/'CHECKS_PASSED.json',dict(status='passed',task=TASK,steps=TOTAL,updates=final['updates'],
  resume_step=START,resume_updates=UPDATES,additional_updates=final['updates']-UPDATES,
  complete_state_restored=True,online_only_replay=True,frozen_base=True,naive_TD=True,final_only_evaluation=True))
except BaseException as e:
 common.atomic_json(OUT/'FAILED.json',dict(type=type(e).__name__,message=str(e)));raise
