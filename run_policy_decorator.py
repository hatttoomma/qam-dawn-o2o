"""Policy Decorator online: inherited QAM critic, QAM uniform replay."""
import argparse
import fcntl
import json
from pathlib import Path
import time
import run as common
from run import (ROOT, HORIZON, GAMMA, GRID, prepare_online, flow_hash, tree_hash, file_hash,
                 atomic_json, load_checkpoint, save_checkpoint, reset_train, restore_env,
                 evaluate, log, floats)
import jax
import jax.numpy as jnp
import numpy as np
from policy_decorator_agent import PolicyDecoratorAgent, collect_action
from qam_replay import QamReplay, cache_offline


def residual(args):
    if (Path(args.out)/'DONE.json').exists():
        return
    out, env, ds, ex, qam = prepare_online(args)
    config=json.loads((out/'config.json').read_text())
    config.update(method='Policy Decorator (QAM adaptation)',
                  policy_decorator_hyperparameters=dict(res_scale=.1,progressive_exploration=args.prog_explore,
                    progressive_unit='primitive steps at chunk start',learning_starts=args.warmup,
                    initial_alpha=1.,target_entropy=-25,autotune=True,lr_actor=1e-4,lr_critic=1e-4,lr_alpha=1e-4,
                    tau=.01,utd=.25,batch_size=args.dawn_batch,actor_dims=[256]*3,actor_input='obs',
                    log_std_min=-20,log_std_max=2,critic_input='clipped sum',max_global_grad_norm=50,
                    target_aggregation='minimum over all 10',actor_q_aggregation='minimum over all 10',
                    entropy_in_td=True,policy_frequency=1,target_frequency=1,
                    before_learning='gated uniform residual [-1,1]',after_learning='gated tanh Gaussian residual',
                    gate='independent Bernoulli per action chunk',base_policy_frozen=True),
                  replay_mode='qam_uniform_offline_online_sequences', critic_boundary_mask=True,
                  cache_manifest_sha256=file_hash(Path(args.cache_dir)/'manifest.json'))
    atomic_json(out/'config.json',config)
    buffer=QamReplay(ds,args.online_steps,qam,args.cache_dir,args.seed)
    online=[]
    del ds
    initial_flow = flow_hash(qam)
    agent = PolicyDecoratorAgent.create(qam,ex['observations'],args.seed,warm=True)
    assert tree_hash(agent.critic.params)==tree_hash(qam.network.params['modules_critic'])
    assert tree_hash(agent.target_params)==tree_hash(qam.network.params['modules_target_critic'])
    initial = dict(flow=initial_flow,critic=tree_hash(agent.critic.params),target=tree_hash(agent.target_params),
                   actor=tree_hash(agent.actor.params),actor_optimizer=tree_hash(agent.actor.opt_state),
                   critic_optimizer=tree_hash(agent.critic.opt_state),alpha=float(jnp.exp(agent.log_alpha)))
    atomic_json(out/'initial_hashes.json',initial)
    np.random.seed(31000+args.seed)
    gate_rng=np.random.RandomState(41000+args.seed)
    gate_counts=dict(chunks=0,enabled_chunks=0,random_enabled_chunks=0,learned_enabled_chunks=0)
    base_rng=jax.random.PRNGKey(1000+args.seed)
    next_rng=jax.random.PRNGKey(2000+args.seed)
    res_rng=jax.random.PRNGKey(3000+args.seed)
    ob, episode, trace, step, replay, elapsed = reset_train(env,0,args.seed),0,[],0,[],0.
    if (out/'latest.pkl').exists():
        agent,sv=load_checkpoint(out/'latest.pkl',agent)
        step,elapsed,replay,episode,trace=sv['step'],sv['elapsed'],sv['replay'],sv['episode'],sv['trace']
        online=sv['online']
        for transition in online:buffer.add(transition)
        buffer.online_base=sv['online_base_cache']
        buffer.sampled_rows,buffer.sampled_online,buffer.sampled_invalid=sv['sampling_counts']
        ob=restore_env(env,episode,args.seed,trace,sv['ob'])
        base_rng,next_rng,res_rng=map(jnp.asarray,sv['keys'])
        np.random.set_state(sv['numpy_rng'])
        gate_rng.set_state(sv['gate_rng'])
        gate_counts=sv['gate_counts']
    if step==0:
        evaluate(qam,out,0,args,dawn=agent,residual_enabled=False)
        evaluate(qam,out,0,args,dawn=agent,residual_enabled=True,deterministic=True,suffix='_mean_residual')
    tick=time.monotonic()
    info={}
    last_log=step//1000
    while step<args.online_steps:
        start_step=step
        start_ob=ob.copy()
        base_rng,key=jax.random.split(base_rng)
        base=qam.sample_actions(jnp.asarray(ob),key)
        learning_enabled=step>=args.warmup
        res_rng,rkey=jax.random.split(res_rng)
        action,enabled,random_phase,probability=collect_action(agent,jnp.asarray(ob),base,rkey,
            gate_rng.random_sample(),step,args.warmup,args.prog_explore)
        gate_counts['chunks']+=1
        gate_counts['enabled_chunks']+=int(enabled)
        gate_counts['random_enabled_chunks']+=int(enabled and random_phase)
        gate_counts['learned_enabled_chunks']+=int(enabled and not random_phase)
        chunk=np.asarray(action).reshape(HORIZON,-1)
        total_reward,term,trunc,length=0.,False,False,0
        for a in chunk:
            previous_ob=ob.copy()
            ob,reward,term,trunc,env_info=env.step(a)
            transition=dict(observations=previous_ob,actions=a.copy(),rewards=np.float32(reward),
                            terminals=np.float32(term or trunc),masks=np.float32(1.-term),next_observations=ob.copy())
            online.append(transition)
            buffer.add(transition)
            total_reward+=GAMMA**length*float(reward)
            length+=1
            step+=1
            trace.append(a.copy())
            if term or trunc or step==args.online_steps:
                break
        next_rng,nkey=jax.random.split(next_rng)
        next_base=np.asarray(qam.sample_actions(jnp.asarray(ob),nkey))
        buffer.set_actual_proposals(start_step,step-1,np.asarray(base),next_base)
        # A chunk that terminates early is a valid SMDP transition: its issued
        # action has an unexecuted suffix, but the bootstrap discount is zero.
        # Do not include an artificial partial chunk cut by the experiment budget.
        if length==HORIZON or term or trunc:
            replay.append(dict(observations=start_ob,actions=np.asarray(action),rewards=np.float32(total_reward),
                         discounts=np.float32((GAMMA**length)*(1.-term)),next_observations=ob.copy(),
                         base_actions=np.asarray(base),next_base_actions=next_base))
        if term or trunc:
            episode+=1
            ob,trace=reset_train(env,episode,args.seed),[]
        if start_step < args.warmup <= step:
            assert int(agent.updates)==0
            assert tree_hash(agent.actor.params)==initial['actor']
            assert tree_hash(agent.critic.params)==initial['critic']
            atomic_json(out/'warmup.json',dict(steps=step,requested_steps=args.warmup,chunks=len(replay),replay_hash=tree_hash(replay),
                                             actor_hash=tree_hash(agent.actor.params),critic_hash=tree_hash(agent.critic.params)))
        if learning_enabled:
            # Primitive-step UTD .25; accumulate fractional updates across chunks.
            target_updates=int((step-args.warmup)*.25)
            n=target_updates-int(agent.updates)
            for _ in range(n):
                agent,info=agent.update(buffer.sample(args.dawn_batch))
        if step//1000>last_log or step==args.online_steps:
            metrics=floats(info)
            elapsed+=time.monotonic()-tick
            log(out/'metrics.jsonl',dict(step=step,updates=int(agent.updates),elapsed_seconds=elapsed,replay_chunks=len(replay),**buffer.stats(),**gate_counts,residual_probability=probability,**metrics))
            atomic_json(out/'progress.json',dict(stage=args.stage,step=step,target=args.online_steps,updates=int(agent.updates)))
            last_log=step//1000
            tick=time.monotonic()
        if any(start_step < threshold <= step for threshold in GRID) or step==args.online_steps:
            elapsed+=time.monotonic()-tick
            assert flow_hash(qam)==initial_flow
            evaluate(qam,out,step,args,dawn=agent,residual_enabled=True)
            evaluate(qam,out,step,args,dawn=agent,residual_enabled=True,deterministic=True,suffix='_mean_residual')
            save_checkpoint(out/'latest.pkl',agent,step=step,elapsed=elapsed,replay=replay,episode=episode,trace=trace,
                            ob=ob,keys=[np.asarray(x) for x in (base_rng,next_rng,res_rng)],numpy_rng=np.random.get_state(),online=online,online_base_cache=buffer.online_base,
                            sampling_counts=[buffer.sampled_rows,buffer.sampled_online,buffer.sampled_invalid],
                            gate_rng=gate_rng.get_state(),gate_counts=gate_counts)
            tick=time.monotonic()
    evaluate(qam,out,args.online_steps,args,dawn=agent,episodes=args.final_episodes,residual_enabled=step>args.warmup)
    evaluate(qam,out,args.online_steps,args,dawn=agent,episodes=args.final_episodes,deterministic=True,
             residual_enabled=step>args.warmup,suffix='_mean_residual')
    save_checkpoint(out/'final.pkl',agent,step=step,updates=int(agent.updates))
    assert flow_hash(qam)==initial_flow
    atomic_json(out/'DONE.json',dict(arm='policy_decorator_qam_replay',gate_counts=gate_counts,steps=step,updates=int(agent.updates),initial_hashes=initial,
                replay_stats=buffer.stats(),
                final_flow_hash=flow_hash(qam),final_critic_hash=tree_hash(agent.critic.params),final_actor_hash=tree_hash(agent.actor.params),
                checkpoint_sha256=file_hash(out/'final.pkl'),elapsed_seconds=elapsed))
    env.close()


def parse():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['warm'],required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--data',default=str(ROOT/'data'))
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--offline-steps',type=int,default=500000)
    p.add_argument('--online-steps',type=int,default=50000)
    p.add_argument('--offline-checkpoint',default=str(ROOT/'runs/offline/final.pkl'))
    p.add_argument('--eval-episodes',type=int,default=50)
    p.add_argument('--final-episodes',type=int,default=100)
    p.add_argument('--block',type=int,default=100)
    p.add_argument('--log-interval',type=int,default=5000)
    p.add_argument('--warmup',type=int,default=8000)
    p.add_argument('--prog-explore',type=int,default=30000)
    p.add_argument('--native-start',type=int,default=5000)
    p.add_argument('--dawn-batch',type=int,default=1024)
    p.add_argument('--cache-dir',default=str(ROOT/'data/qam_base_cache_seed0'))
    p.add_argument('--cache-block',type=int,default=1024)
    return p.parse_args()


if __name__=='__main__':
    args=parse()
    Path(args.out).mkdir(parents=True,exist_ok=True)
    lock=(Path(args.out)/'run.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        residual(args)
    except BaseException as exc:
        atomic_json(Path(args.out)/'FAILED.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
