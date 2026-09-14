"""Pinned official QAM online loop, initialized from the existing offline500k agent."""
import fcntl,json,os,shutil,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'official'),str(ROOT/'legacy')]
import main as official
import run as common
import paired_evaluation
import flax,jax,numpy as np
from agents.qam import QAMAgent,get_config
TASK=int(os.environ['TASK']);SMOKE=os.environ.get('SMOKE')=='1';OUT=Path(os.environ['OUT']);OUT.mkdir(parents=True,exist_ok=True)
lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
spec=json.loads((ROOT/'inputs.json').read_text())[str(TASK)];NATIVE=Path(spec['native']);TOTAL=24 if SMOKE else 100000
START_TRAIN=8 if SMOKE else 5000;EPISODES=2 if SMOKE else 100
LOG_INTERVAL=8 if SMOKE else 1000
ENV=f'antmaze-large-navigate-singletask-task{TASK}-v0';common.ENV=ENV;common.HORIZON=1
ckpt=NATIVE/'offline_500k.pkl';assert common.file_hash(ckpt)==spec['offline_sha256']
manifest=json.loads((ROOT/'source_manifest.json').read_text())
for name,digest in manifest['files'].items():assert common.file_hash(ROOT/name)==digest,name
args=SimpleNamespace(seed=0,eval_episodes=EPISODES)
meta=json.loads((NATIVE/'offline_checkpoint.json').read_text());tick=time.monotonic();LAST=0
for name in ['actual_agent_config.json','dataset_manifest.json']:assert (NATIVE/name).exists()

class Factory:
 @staticmethod
 def create(seed,obs,action,cfg):
  expected=get_config();expected.horizon_length=1;expected.action_chunking=True;expected.inv_temp=10.;expected.fql_alpha=0.;expected.edit_scale=0.
  assert cfg.to_dict()==expected.to_dict() and seed==0
  template=QAMAgent.create(seed,obs,action,cfg);agent,payload=common.load_checkpoint(ckpt,template)
  assert payload['step']==500000 and int(agent.network.step)-1==500000
  assert common.flow_hash(agent)==meta['flow_hash'] and common.tree_hash(agent.network.params['modules_critic'])==meta['q_hash']
  prior=json.loads((NATIVE/'actual_agent_config.json').read_text())
  assert json.loads(json.dumps(dict(agent.config)))==prior['agent_config']
  common.atomic_json(OUT/'actual_agent_config.json',dict(prior,offline_checkpoint_reused=True,
    online_steps=TOTAL,native_start=START_TRAIN,utd=1,offline_updates=500000,new_online_run=True,
    original_online50k_replay_missing=True,online_seed=0))
  common.atomic_json(OUT/'initial_state.json',dict(checkpoint_sha256=spec['offline_sha256'],
    agent_state_hash=common.tree_hash(flax.serialization.to_state_dict(agent)),flow_hash=common.flow_hash(agent),full_offline_optimizer_restored=True))
  if SMOKE:
   e=paired_evaluation.evaluate(agent,OUT/'fixed_eval',0,args,episodes=2,residual_enabled=False)
   ref=json.loads((NATIVE/'fixed_eval/eval_000000_100.json').read_text());assert e['records']==ref['records'][:2]
  else:
   (OUT/'fixed_eval').mkdir(exist_ok=True);shutil.copy2(NATIVE/'fixed_eval/eval_000000_100.json',OUT/'fixed_eval/eval_000000_100.json')
  return agent

old_data=official.make_env_and_datasets

def data(*a,**kw):
 result=old_data(*a,**kw);env,_,ds,_=result
 ref=json.loads((NATIVE/'dataset_manifest.json').read_text());assert env.unwrapped._reward_task_id==TASK and ds.size==ref['dataset_size']
 for name,item in ref['files'].items():assert common.file_hash(Path('/root/.ogbench/data')/name)==item['sha256']
 common.atomic_json(OUT/'dataset_manifest.json',ref);return result

old_log=official.LoggingHelper.log

def log(self,data,prefix,step):
 global LAST
 LAST=step
 if prefix in ('online_agent','offline_agent'):assert all(np.isfinite(np.asarray(v)).all() for v in data.values())
 # Official CsvLogger mutates data by adding step; capture metric fields first.
 metrics={k:float(np.asarray(v)) for k,v in data.items() if k not in ('step','updates')} if prefix=='online_agent' else None
 old_log(self,data,prefix,step)
 if step%LOG_INTERVAL==0 and prefix=='online_agent' or step==TOTAL:
  common.atomic_json(OUT/'progress.json',dict(stage='native',step=step,target=TOTAL,updates=max(0,step-START_TRAIN+1)))
  if prefix=='online_agent':common.log(OUT/'metrics.jsonl',dict(step=step,updates=max(0,step-START_TRAIN+1),**metrics))

def evaluate(*a,**kw):
 agent=kw['agent'];updates=int(agent.network.step)-1-500000
 assert updates==TOTAL-START_TRAIN+1
 before=common.tree_hash(flax.serialization.to_state_dict(agent));rng=common.tree_hash(np.random.get_state())
 result=paired_evaluation.evaluate(agent,OUT/'fixed_eval',TOTAL,args,episodes=EPISODES,residual_enabled=False)
 assert before==common.tree_hash(flax.serialization.to_state_dict(agent)) and rng==common.tree_hash(np.random.get_state())
 common.save_checkpoint(OUT/'final.pkl',agent,step=TOTAL,updates=updates,offline_updates=500000)
 # Capture final replay and RNG from the unchanged official loop for future extension.
 frame=sys._getframe(1);loc=frame.f_locals;replay=loc['replay_buffer'];env=loc['env']
 online={k:np.array(v[loc['train_dataset'].size:replay.size]) for k,v in replay.items()}
 sim=env.unwrapped
 common.save_checkpoint(OUT/'latest.pkl',agent,step=TOTAL,updates=updates,online=online,
    numpy_rng=np.random.get_state(),online_rng=np.asarray(loc['online_rng']),ob=np.asarray(loc['ob']),
    action_queue=loc['action_queue'],environment_state={'qpos':sim.data.qpos.copy(),'qvel':sim.data.qvel.copy()},
    resume_format='official_final_replay_and_partial_environment; verify before resuming')
 common.atomic_json(OUT/'DONE.json',dict(status='passed',steps=TOTAL,updates=updates,offline_updates=500000,
   checkpoint_sha256=common.file_hash(OUT/'final.pkl'),final_flow_hash=common.flow_hash(agent),
   initial_flow_hash=meta['flow_hash'],online_buffer_size=len(online['actions']),elapsed_seconds=time.monotonic()-tick))
 assert common.flow_hash(agent)!=meta['flow_hash']
 common.atomic_json(OUT/'CHECKS_PASSED.json',dict(status='passed',task=TASK,steps=TOTAL,updates=updates,
   pinned_official_main_unchanged=True,exact_offline_checkpoint=True,official_hyperparameters_unchanged=True,
   fresh_online_run_due_to_missing_prior_replay=True,paired_final_evaluation=True))
 return {'success':result['success'],'return':result['return_mean']},[],[]

def setup(**kw):
 common.atomic_json(OUT/'requested_flags.json',official.get_flag_dict());return official.wandb.run

official.agents['qam']=Factory;official.make_env_and_datasets=data;official.LoggingHelper.log=log;official.evaluate=evaluate
if SMOKE:official.tqdm.tqdm=lambda x:x
official.setup_wandb=setup;official.wandb=SimpleNamespace(run=SimpleNamespace(project='scale100k',url='local-only'),log=lambda *a,**k:None,finish=lambda:None)
sys.argv=[sys.argv[0],f'--env_name={ENV}','--seed=0','--offline_steps=0',f'--online_steps={TOTAL}',
 f'--start_training={START_TRAIN}','--utd_ratio=1','--horizon_length=1','--eval_interval=0','--save_interval=0',
 f'--log_interval={LOG_INTERVAL}',f'--eval_episodes={EPISODES}',f'--save_dir={OUT}/official',f'--agent={ROOT}/official/agents/qam.py',
 '--agent.action_chunking=True','--agent.inv_temp=10','--agent.edit_scale=0','--agent.fql_alpha=0','--auto_cleanup=False']
def entry(_):
 try:
  official.main(_)
  if SMOKE:
   records=[json.loads(line) for line in (OUT/'metrics.jsonl').read_text().splitlines()]
   assert [(r['step'],r['updates']) for r in records]==[(8,1),(16,9),(24,17)]
   common.atomic_json(OUT/'LOGGING_CHECK_PASSED.json',dict(status='passed',steps=[r['step'] for r in records],official_mutating_logger_exercised=True))
 except BaseException as e:common.atomic_json(OUT/'FAILED.json',dict(type=type(e).__name__,message=str(e)));raise
if __name__=='__main__':official.app.run(entry)
