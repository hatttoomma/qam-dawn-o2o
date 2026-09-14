"""Cube native continuation and40k/mixed50k/low-UTD100k DAWN schedule."""
import fcntl,json,os,pickle,shutil,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent;OLD=Path('/root/autodl-tmp/qam_dawn_o2o')
sys.path[:0]=[str(OLD/'vendor/qam'),str(ROOT/'legacy'),str(ROOT)]
import run as common
import actor_input_agent as impl
from balanced_replay import BalancedReplay,test_sequence_conversion
import flax,jax,jax.numpy as jnp,numpy as np
TASK=int(os.environ['TASK']);METHOD=os.environ['METHOD'];SMOKE=os.environ.get('SMOKE')=='1'
OUT=Path(os.environ['OUT']);OUT.mkdir(parents=True,exist_ok=True)
lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
spec=json.loads((ROOT/'inputs.json').read_text())[str(TASK)];OFFLINE=Path(spec['offline']);meta=spec['offline_metadata'];NATIVE=Path(spec['native'])
assert common.file_hash(OFFLINE)==meta['checkpoint_sha256']
for name,digest in json.loads((ROOT/'source_manifest.json').read_text()).items():
 p=Path(name) if name.startswith('/') else ROOT/name
 assert common.file_hash(p)==digest,name
common.ENV=f'cube-double-play-singletask-task{TASK}-v0';common.HORIZON=5;common.GAMMA=.99
TOTAL=(50032 if METHOD=='native' else 240) if SMOKE else 100000
WARM=80 if SMOKE else 40000;SWITCH=160 if SMOKE else 50000
common.GRID=(50000,TOTAL) if METHOD=='native' else (0,WARM,SWITCH,TOTAL)
args=SimpleNamespace(stage='native' if METHOD=='native' else 'warm',out=str(OUT),data=str(OLD/'data'),seed=0,
 offline_steps=500000,online_steps=TOTAL,offline_checkpoint=str(OFFLINE),eval_episodes=2 if SMOKE else 100,
 final_episodes=2 if SMOKE else 100,warmup=WARM,replay_switch=SWITCH,dawn_batch=256,native_start=5000)
common.atomic_json(OUT/'requested_config.json',dict(vars(args),task=TASK,method=METHOD,smoke=SMOKE,
 utd_schedule=[dict(start_env=WARM,end_env=SWITCH,utd=.25,offline_batch=128,online_batch=128),
 dict(start_env=SWITCH,end_env=TOTAL,utd=.0625,offline_batch=0,online_batch=256)] if METHOD=='dawn' else [dict(utd=1)],
 horizon=5,naive_TD=True,state_plus_base_action=True))
original_data,original_load,original_save,original_eval,original_restore=common.make_data,common.load_checkpoint,common.save_checkpoint,common.evaluate,common.restore_env
if METHOD=='native':
 nd=json.loads((NATIVE/'DONE.json').read_text());assert (nd['steps'],nd['updates'])==(50000,45001)
 assert common.file_hash(NATIVE/'final.pkl')==nd['checkpoint_sha256']
 with (NATIVE/'latest.pkl').open('rb') as f:p=pickle.load(f)
 with (NATIVE/'final.pkl').open('rb') as f:final=pickle.load(f)
 assert p['step']==50000 and p['updates']==45001 and len(p['online'])==50000
 expected=common.tree_hash(p['agent']);assert expected==common.tree_hash(final['agent'])
 audit=dict(source=str(NATIVE/'latest.pkl'),checkpoint_sha256=common.file_hash(NATIVE/'latest.pkl'),agent_hash=expected,
  online_hash=common.tree_hash(p['online']),numpy_rng_hash=common.tree_hash(p['numpy_rng']),rng_hash=common.tree_hash(p['rng']),
  queue_hash=common.tree_hash(p['queue']),resume_step=50000,resume_updates=45001)
 common.atomic_json(OUT/'resume_source.json',audit)
 assert not (OUT/'latest.pkl').exists();shutil.copy2(NATIVE/'latest.pkl',OUT/'latest.pkl')
 if not SMOKE:
  for q in NATIVE.glob('eval_*_100.json'):shutil.copy2(q,OUT/q.name)
 del p,final

def data(config):
 env,ds=original_data(config);assert env.unwrapped._reward_task_id==TASK
 original=json.loads((OLD/'runs/cube5/TASK_AUDIT_PASSED.json').read_text())['tasks'][TASK-1]['manifest']
 # Dataset relabeling must match the audited task; checkpoint metadata pins the actor/Q.
 assert ds['observations'].shape[-1]==37 and ds['actions'].shape[-1]==5
 common.atomic_json(OUT/'dataset_manifest.json',dict(task=TASK,environment=common.ENV,dataset_size=ds.size,
  data_sha256=common.file_hash(Path(config.data)/'cube-double-play-v0.npz'),audited_task_manifest=original,
  observation_dim=37,primitive_action_dim=5,horizon=5))
 return env,ds

class Factory:
 @staticmethod
 def create(qam,obs,seed,warm):
  assert warm and qam.config['inv_temp']==1 and qam.config['edit_scale']==0 and qam.config['horizon_length']==5
  agent=impl.ActorInputAgent.create(qam,obs,seed,True).replace(condition_on_base=True)
  assert common.flow_hash(qam)==meta['flow_hash'] and common.tree_hash(agent.critic.params)==meta['q_hash']
  assert common.tree_hash(agent.target_params)==meta['target_q_hash']
  common.atomic_json(OUT/'actual_agent_config.json',dict(task=TASK,seed=0,inherited_Q_and_target=True,
    fresh_critic_optimizer=True,td_target='naive',actor_input='state+base_action',observation_dim=37,action_dim=25,
    actor_input_dim=62,residual_scale=.1,horizon=5,discount=.99,batch_size=256,warmup=WARM,replay_switch=SWITCH,
    online_steps=TOTAL,utd_before_switch=.25,utd_after_switch=.0625,offline_batch_before_switch=128,
    online_batch_before_switch=128,offline_batch_after_switch=0,online_batch_after_switch=256,
    actor_hidden_dims=[256]*3,actor_activation='relu',gaussian_output=True,critic_hidden_dims=[512]*4,
    critic_ensemble=10,critic_activation='gelu',critic_layer_norm=True,actor_Q_aggregation='minimum',target_aggregation='minimum',
    learning_rate=1e-4,target_tau=.01,gradient_clip_norm=50,actor_entropy_enabled=True,automatic_alpha_enabled=True,
    initial_alpha=.01,target_entropy=-25,full_initial_agent_hash=common.tree_hash(flax.serialization.to_state_dict(agent))))
  return agent

def load(path,agent):
 loaded,p=original_load(path,agent)
 if METHOD=='native' and Path(path)==OUT/'latest.pkl':
  assert common.tree_hash(flax.serialization.to_state_dict(loaded))==audit['agent_hash']
  for key,hash_key in [('online','online_hash'),('numpy_rng','numpy_rng_hash'),('rng','rng_hash'),('queue','queue_hash')]:assert common.tree_hash(p[key])==audit[hash_key]
  common.atomic_json(OUT/'RESTORE_PASSED.json',dict(status='passed',**audit))
  common.atomic_json(OUT/'actual_agent_config.json',dict(qam_config=dict(loaded.config),task=TASK,online_steps=TOTAL,
    utd=1,batch_size=256,native_start=5000,exact_native50k_agent_and_optimizer_resumed=True,offline_updates=500000))
  if SMOKE:
   e=original_eval(loaded,OUT,50000,args,episodes=2)
   ref=json.loads((NATIVE/'eval_050000_100.json').read_text());assert e['records']==ref['records'][:2]
 return loaded,p

def restore(env,ep,seed,trace,expected):
 ob=original_restore(env,ep,seed,trace,expected)
 common.atomic_json(OUT/'environment_restored.json',dict(status='passed',max_abs_error=float(np.max(np.abs(ob-expected)))))
 return ob

def evaluate(qam,out,step,config,**kw):
 if METHOD=='dawn' and WARM<=step<WARM+5:
  return json.loads((OUT/f'eval_000000_{args.eval_episodes:03d}.json').read_text())
 before=common.tree_hash(np.random.get_state())
 result=original_eval(qam,out,step,config,**kw)
 if METHOD=='dawn' and step>WARM and not kw.get('deterministic',False):
  original_eval(qam,out,step,config,**dict(kw,deterministic=True,suffix='_mean_residual'))
 assert before==common.tree_hash(np.random.get_state())
 common.atomic_json(OUT/f'eval_rng_preserved_{step:06d}.json',dict(status='passed',step=step));return result

def expected_updates(step):
 if METHOD=='native':return max(0,step-4999)
 return max(0,min(step,SWITCH)-WARM)//4+max(0,step-SWITCH)//16

def save(path,agent,**kw):
 u=kw['updates'] if METHOD=='native' else int(agent.updates)
 # During a warmup-crossing chunk the original collector still has zero updates.
 expect=expected_updates(kw['step'])
 if METHOD=='dawn' and kw['step']<WARM+5:expect=0
 assert u==expect,(kw['step'],u,expect)
 original_save(path,agent,**kw)
 if Path(path).name=='latest.pkl':
  original_save(OUT/f'checkpoint_{kw["step"]:06d}.pkl',agent,step=kw['step'],updates=u)
  common.atomic_json(OUT/'checkpoint_state.json',dict(step=kw['step'],updates=u,
   replay_size=len(kw['online'] if METHOD=='native' else kw['replay'])))

original_stack=common.stack_replay
def stack(replay,idx):
 batch=original_stack(replay,idx)
 assert len(idx) in (128,256) and all(np.isfinite(v).all() for v in batch.values())
 if METHOD=='dawn' and len(idx)==256:
  agent=sys._getframe(1).f_locals['agent'];update=int(agent.updates)+1;mixed_updates=(SWITCH-WARM)//4
  assert update>mixed_updates
  if update==mixed_updates+1 or update%250==0 or update==expected_updates(TOTAL):
   common.atomic_json(OUT/'online_replay_audit.json',dict(status='passed',update=update,offline_per_batch=0,
    online_per_batch=256,online_only_updates=update-mixed_updates,online_only_draws=(update-mixed_updates)*256,
    online_chunks=len(replay),warmup_and_mixed_phase_rollout_retained=True))
 return batch
common.stack_replay=stack
common.make_data,common.load_checkpoint,common.restore_env=data,load,restore
common.evaluate,common.save_checkpoint=evaluate,save
common.DawnAgent,common.BalancedReplay=Factory,BalancedReplay
impl.soft_backup=lambda r,d,q,a,lp:r+d*jnp.min(q,axis=0)
test_sequence_conversion();common.atomic_json(OUT/'sequence_boundary_check.json',dict(status='passed',terminal_and_truncation_handled=True,unexecuted_offline_action_suffix_zero_padded=True))
try:
 (common.native if METHOD=='native' else common.residual)(args)
 final=json.loads((OUT/'DONE.json').read_text());assert (final['steps'],final['updates'])==(TOTAL,expected_updates(TOTAL))
 assert common.file_hash(OUT/'final.pkl')==final['checkpoint_sha256']
 if METHOD=='dawn':
  assert final['final_flow_hash']==meta['flow_hash']
  mixed=json.loads((OUT/'mixed_replay_audit.json').read_text())
  assert mixed['update']==(SWITCH-WARM)//4 and mixed['offline_per_batch']==mixed['online_per_batch']==128
  online_audit=json.loads((OUT/'online_replay_audit.json').read_text());assert online_audit['online_only_updates']==(TOTAL-SWITCH)//16
 else:assert final['final_flow_hash']!=meta['flow_hash']
 common.atomic_json(OUT/'CHECKS_PASSED.json',dict(status='passed',task=TASK,method=METHOD,steps=TOTAL,updates=final['updates'],
  offline_checkpoint_verified=True,full_native_state_restored=METHOD=='native',mixed_then_online_only=METHOD=='dawn',
  frozen_base=METHOD=='dawn',original_native_update_loop=True,horizon=5,paired_evaluations=True))
except BaseException as e:
 common.atomic_json(OUT/'FAILED.json',dict(type=type(e).__name__,message=str(e)));raise
