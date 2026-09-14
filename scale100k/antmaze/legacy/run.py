"""Reproducible one-seed QAM/DAWN experiment; no external logging service."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor/qam'))
os.environ.setdefault('MUJOCO_GL', 'egl')

import flax
import jax
import jax.numpy as jnp
import numpy as np
import ogbench

from agents.qam import QAMAgent, get_config
from utils.datasets import Dataset, ReplayBuffer
from dawn_agent import DawnAgent

ENV = 'cube-double-play-singletask-task1-v0'
HORIZON = 5
GAMMA = .99
GRID = (0, 5000, 10000, 20000, 30000, 40000, 50000)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def log(path, record):
    record = dict(record, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    with Path(path).open('a') as f:
        f.write(json.dumps(record, allow_nan=False) + '\n')
    print(json.dumps(record, allow_nan=False), flush=True)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def tree_hash(tree):
    h = hashlib.sha256()
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        arr = np.asarray(leaf)
        h.update(str(path).encode())
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def flow_hash(qam):
    return tree_hash({k: v for k, v in qam.network.params.items() if 'actor_' in k})


def save_checkpoint(path, agent, **extra):
    tmp = Path(str(path) + '.tmp')
    state = jax.device_get(flax.serialization.to_state_dict(agent))
    with tmp.open('wb') as f:
        pickle.dump({'agent': state, **extra}, f, protocol=5)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def load_checkpoint(path, agent):
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    return flax.serialization.from_state_dict(agent, payload['agent']), payload


def qam_template(obs, actions, seed):
    cfg = get_config()
    cfg.horizon_length = HORIZON
    cfg.action_chunking = True
    cfg.inv_temp = 1.
    cfg.edit_scale = 0.
    cfg.fql_alpha = 0.
    return QAMAgent.create(seed, obs, actions, cfg)


def make_data(args):
    env, data, _ = ogbench.make_env_and_datasets(ENV, dataset_dir=args.data, compact_dataset=False)
    data = {k: np.asarray(v) for k, v in data.items()
            if k in ('observations', 'actions', 'rewards', 'terminals', 'masks', 'next_observations')}
    data['actions'] = np.clip(data['actions'], -1. + 1e-5, 1. - 1e-5)
    return env, Dataset.create(**data)


def seq_batch(dataset, n, batch_size):
    b = dataset.sample_sequence(n * batch_size, HORIZON, GAMMA)
    return {k: np.asarray(v, np.float32).reshape((n, batch_size) + v.shape[1:]) for k, v in b.items()}


def floats(info):
    result = {k: float(np.asarray(v)) for k, v in info.items()}
    if not all(np.isfinite(list(result.values()))):
        raise FloatingPointError(result)
    return result


def reset_train(env, episode, seed):
    return env.reset(seed=100000 + seed * 10000 + episode)[0]


def restore_env(env, episode, seed, trace, expected):
    obs = reset_train(env, episode, seed)
    for action in trace:
        obs, _, terminated, truncated, _ = env.step(action)
        assert not (terminated or truncated), 'Checkpoint trace crosses episode boundary'
    np.testing.assert_allclose(obs, expected, atol=1e-7, rtol=1e-7)
    return obs


def evaluate(qam, out, step, args, dawn=None, residual_enabled=True, episodes=None, deterministic=False, suffix=''):
    episodes = args.eval_episodes if episodes is None else episodes
    path = Path(out) / f'eval_{step:06d}_{episodes:03d}{suffix}.json'
    if path.exists():
        return json.loads(path.read_text())
    np_state = np.random.get_state()
    env = ogbench.make_env_and_datasets(ENV, env_only=True)
    results = []
    start = time.monotonic()
    try:
        for episode in range(episodes):
            reset_seed = 500000 + args.seed*10000 + episode
            ob, _ = env.reset(seed=reset_seed)
            initial_hash = tree_hash(ob)
            base_key = jax.random.PRNGKey(800000 + args.seed*10000 + episode)
            residual_key = jax.random.PRNGKey(900000 + args.seed*10000 + episode)
            ret, steps, success, decision = 0., 0, 0., 0
            done = False
            while not done:
                base_key, key = jax.random.split(base_key)
                base = qam.sample_actions(jnp.asarray(ob), key)
                if dawn is not None and residual_enabled:
                    residual_key, rkey = jax.random.split(residual_key)
                    action, _ = dawn.sample(jnp.asarray(ob), base, rkey, deterministic=deterministic)
                else:
                    action = base
                chunk = np.asarray(action).reshape(HORIZON, -1)
                for a in chunk:
                    ob, reward, terminated, truncated, info = env.step(a)
                    ret += float(reward)
                    steps += 1
                    success = max(success, float(info.get('success', 0.)))
                    done = terminated or truncated
                    if done:
                        break
                decision += 1
            results.append(dict(episode=episode, reset_seed=reset_seed, initial_hash=initial_hash,
                                success=success, return_=ret, length=steps, decisions=decision))
        result = dict(step=step, episodes=episodes, success=float(np.mean([x['success'] for x in results])),
                      return_mean=float(np.mean([x['return_'] for x in results])),
                      duration_seconds=time.monotonic()-start, residual_enabled=residual_enabled,
                      deterministic_residual=deterministic, records=results)
        atomic_json(path, result)
        log(Path(out)/'evaluations.jsonl', {k:v for k,v in result.items() if k != 'records'})
        return result
    finally:
        env.close()
        np.random.set_state(np_state)


def offline(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'DONE.json').exists():
        return
    env, ds = make_data(args)
    ex = ds.get_subset(0)
    qam = qam_template(ex['observations'], ex['actions'], args.seed)
    np.random.seed(args.seed)
    step, elapsed = 0, 0.
    if (out/'latest.pkl').exists():
        qam, saved = load_checkpoint(out/'latest.pkl', qam)
        step, elapsed = saved['step'], saved['elapsed']
        np.random.set_state(saved['numpy_rng'])
    config = dict(vars(args), qam_config=dict(qam.config), actual_baseline='QAM_EDIT(edit_scale=0) = QAM',
                  dataset_size=ds.size, devices=[str(x) for x in jax.devices()])
    config['qam_config']['ob_dims'] = list(config['qam_config']['ob_dims'])
    atomic_json(out/'config.json', config)
    atomic_json(out/'dataset_manifest.json', {p.name:dict(bytes=p.stat().st_size, sha256=file_hash(p),
                source='http://rail.eecs.berkeley.edu/datasets/ogbench/'+p.name)
                for p in Path(args.data).glob('cube-double-play-v0*.npz')})
    env.close()
    if step == 0:
        evaluate(qam, out, 0, args)
    checkpoints = sorted(set([x for x in (100000,250000,args.offline_steps) if x <= args.offline_steps]))
    while step < args.offline_steps:
        n = min(args.block, args.offline_steps-step)
        future = [x for x in checkpoints if x > step]
        if future:
            n = min(n, min(future)-step)
        start = time.monotonic()
        qam, info = qam.batch_update(seq_batch(ds, n, 256))
        metrics = floats(info)
        elapsed += time.monotonic()-start
        step += n
        if step % args.log_interval == 0 or step == args.offline_steps:
            log(out/'metrics.jsonl', dict(step=step, elapsed_seconds=elapsed, updates_per_second=step/max(elapsed,1e-6), **metrics))
            atomic_json(out/'progress.json', dict(stage='offline', step=step, target=args.offline_steps, elapsed_seconds=elapsed))
        if step % 50000 == 0 or step in checkpoints:
            save_checkpoint(out/'latest.pkl', qam, step=step, elapsed=elapsed, numpy_rng=np.random.get_state())
        if step in checkpoints:
            evaluate(qam, out, step, args)
    save_checkpoint(out/'final.pkl', qam, step=step, elapsed=elapsed, numpy_rng=np.random.get_state())
    atomic_json(out/'DONE.json', dict(step=step, checkpoint_sha256=file_hash(out/'final.pkl'),
                                    flow_hash=flow_hash(qam), q_hash=tree_hash(qam.network.params['modules_critic']),
                                    target_q_hash=tree_hash(qam.network.params['modules_target_critic']), elapsed_seconds=elapsed))


def prepare_online(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env, ds = make_data(args)
    ex = ds.get_subset(0)
    qam = qam_template(ex['observations'], ex['actions'], args.seed)
    qam, saved = load_checkpoint(args.offline_checkpoint, qam)
    assert saved['step'] == args.offline_steps
    atomic_json(out/'config.json', dict(vars(args), offline_sha256=file_hash(args.offline_checkpoint),
                observation_dim=ex['observations'].shape[-1], action_dim=ex['actions'].shape[-1],
                qam_critic_dims=[512]*4, qam_critic_ensemble=10, discount=GAMMA, horizon=HORIZON))
    return out, env, ds, ex, qam


def native(args):
    if (Path(args.out)/'DONE.json').exists():
        return
    out, env, ds, ex, qam = prepare_online(args)
    initial_flow = flow_hash(qam)
    replay = ReplayBuffer.create_from_initial_dataset(dict(ds), ds.size+args.online_steps+HORIZON)
    online = []
    np.random.seed(31000+args.seed)
    rng = jax.random.PRNGKey(1000+args.seed)
    ob, episode, trace, queue, start_step, elapsed, updates = reset_train(env,0,args.seed), 0, [], [], 0, 0., 0
    if (out/'latest.pkl').exists():
        qam, sv = load_checkpoint(out/'latest.pkl', qam)
        start_step, elapsed, updates = sv['step'], sv['elapsed'], sv['updates']
        online, episode, trace, queue = sv['online'], sv['episode'], sv['trace'], sv['queue']
        for transition in online:
            replay.add_transition(transition)
        ob = restore_env(env,episode,args.seed,trace,sv['ob'])
        rng = jnp.asarray(sv['rng'])
        np.random.set_state(sv['numpy_rng'])
    if start_step == 0:
        evaluate(qam,out,0,args)
    info = {}
    tick = time.monotonic()
    for step in range(start_step+1,args.online_steps+1):
        rng, key = jax.random.split(rng)
        if not queue:
            queue = list(np.asarray(qam.sample_actions(jnp.asarray(ob),key)).reshape(HORIZON,-1))
        action = queue.pop(0)
        next_ob, reward, term, trunc, env_info = env.step(action)
        done = term or trunc
        transition = dict(observations=ob.copy(),actions=action.copy(),rewards=np.float32(reward),
                          terminals=np.float32(done),masks=np.float32(1.-term),next_observations=next_ob.copy())
        online.append(transition)
        replay.add_transition(transition)
        trace.append(action.copy())
        if done:
            episode += 1
            ob, trace, queue = reset_train(env,episode,args.seed), [], []
        else:
            ob = next_ob
        if step >= args.native_start:
            qam, info = qam.batch_update(seq_batch(replay,1,256))
            updates += 1
        if step % 1000 == 0 or step == args.online_steps:
            metrics = floats(info)
            elapsed += time.monotonic()-tick
            log(out/'metrics.jsonl',dict(step=step,updates=updates,elapsed_seconds=elapsed,replay_size=replay.size,**metrics))
            atomic_json(out/'progress.json',dict(stage='native',step=step,target=args.online_steps,updates=updates))
            tick=time.monotonic()
        if step in GRID or step == args.online_steps:
            elapsed += time.monotonic()-tick
            evaluate(qam,out,step,args)
            save_checkpoint(out/'latest.pkl',qam,step=step,elapsed=elapsed,updates=updates,online=online,
                            episode=episode,trace=trace,queue=queue,ob=ob,rng=np.asarray(rng),numpy_rng=np.random.get_state())
            tick=time.monotonic()
    evaluate(qam,out,args.online_steps,args,episodes=args.final_episodes)
    save_checkpoint(out/'final.pkl',qam,step=args.online_steps,updates=updates)
    atomic_json(out/'DONE.json',dict(arm='native',steps=args.online_steps,updates=updates,initial_flow_hash=initial_flow,
                final_flow_hash=flow_hash(qam),checkpoint_sha256=file_hash(out/'final.pkl'),elapsed_seconds=elapsed))
    env.close()


def stack_replay(records, indices):
    return {k:np.asarray([records[int(i)][k] for i in indices],np.float32) for k in records[0]}


def residual(args):
    if (Path(args.out)/'DONE.json').exists():
        return
    out, env, ds, ex, qam = prepare_online(args)
    del ds
    initial_flow = flow_hash(qam)
    agent = DawnAgent.create(qam,ex['observations'],args.seed,warm=args.stage=='warm')
    initial = dict(flow=initial_flow,critic=tree_hash(agent.critic.params),target=tree_hash(agent.target_params),
                   actor=tree_hash(agent.actor.params),actor_optimizer=tree_hash(agent.actor.opt_state),
                   critic_optimizer=tree_hash(agent.critic.opt_state),alpha=float(jnp.exp(agent.log_alpha)))
    atomic_json(out/'initial_hashes.json',initial)
    np.random.seed(31000+args.seed)
    base_rng=jax.random.PRNGKey(1000+args.seed)
    next_rng=jax.random.PRNGKey(2000+args.seed)
    res_rng=jax.random.PRNGKey(3000+args.seed)
    ob, episode, trace, step, replay, elapsed = reset_train(env,0,args.seed),0,[],0,[],0.
    if (out/'latest.pkl').exists():
        agent,sv=load_checkpoint(out/'latest.pkl',agent)
        step,elapsed,replay,episode,trace=sv['step'],sv['elapsed'],sv['replay'],sv['episode'],sv['trace']
        ob=restore_env(env,episode,args.seed,trace,sv['ob'])
        base_rng,next_rng,res_rng=map(jnp.asarray,sv['keys'])
        np.random.set_state(sv['numpy_rng'])
    if step==0:
        evaluate(qam,out,0,args,dawn=agent,residual_enabled=False)
    tick=time.monotonic()
    info={}
    last_log=step//1000
    while step<args.online_steps:
        start_step=step
        start_ob=ob.copy()
        base_rng,key=jax.random.split(base_rng)
        base=qam.sample_actions(jnp.asarray(ob),key)
        enabled=step>=args.warmup
        res_rng,rkey=jax.random.split(res_rng)
        if enabled:
            action,_=agent.sample(jnp.asarray(ob),base,rkey)
        else:
            action=base
        chunk=np.asarray(action).reshape(HORIZON,-1)
        total_reward,term,trunc,length=0.,False,False,0
        for a in chunk:
            ob,reward,term,trunc,env_info=env.step(a)
            total_reward+=GAMMA**length*float(reward)
            length+=1
            step+=1
            trace.append(a.copy())
            if term or trunc or step==args.online_steps:
                break
        next_rng,nkey=jax.random.split(next_rng)
        next_base=np.asarray(qam.sample_actions(jnp.asarray(ob),nkey))
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
        if enabled:
            # Resume at50k/2500 updates; one additional update per16 primitive steps.
            target_updates=2500+(step-50000)//16
            n=target_updates-int(agent.updates)
            for _ in range(n):
                idx=np.random.randint(len(replay),size=args.dawn_batch)
                agent,info=agent.update(stack_replay(replay,idx))
        if step//1000>last_log or step==args.online_steps:
            metrics=floats(info)
            elapsed+=time.monotonic()-tick
            log(out/'metrics.jsonl',dict(step=step,updates=int(agent.updates),elapsed_seconds=elapsed,replay_chunks=len(replay),**metrics))
            atomic_json(out/'progress.json',dict(stage=args.stage,step=step,target=args.online_steps,updates=int(agent.updates)))
            last_log=step//1000
            tick=time.monotonic()
        if any(start_step < threshold <= step for threshold in GRID) or step==args.online_steps:
            elapsed+=time.monotonic()-tick
            assert flow_hash(qam)==initial_flow
            evaluate(qam,out,step,args,dawn=agent,residual_enabled=int(agent.updates)>0)
            save_checkpoint(out/'latest.pkl',agent,step=step,elapsed=elapsed,replay=replay,episode=episode,trace=trace,
                            ob=ob,keys=[np.asarray(x) for x in (base_rng,next_rng,res_rng)],numpy_rng=np.random.get_state())
            tick=time.monotonic()
    evaluate(qam,out,args.online_steps,args,dawn=agent,episodes=args.final_episodes,residual_enabled=step>args.warmup)
    evaluate(qam,out,args.online_steps,args,dawn=agent,episodes=args.eval_episodes,deterministic=True,
             residual_enabled=step>args.warmup,suffix='_mean_residual')
    save_checkpoint(out/'final.pkl',agent,step=step,updates=int(agent.updates))
    assert flow_hash(qam)==initial_flow
    atomic_json(out/'DONE.json',dict(arm=args.stage,steps=step,updates=int(agent.updates),initial_hashes=initial,
                final_flow_hash=flow_hash(qam),final_critic_hash=tree_hash(agent.critic.params),final_actor_hash=tree_hash(agent.actor.params),
                checkpoint_sha256=file_hash(out/'final.pkl'),elapsed_seconds=elapsed))
    env.close()


def parse():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['offline','native','warm','random'],required=True)
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
    p.add_argument('--warmup',type=int,default=20000)
    p.add_argument('--native-start',type=int,default=5000)
    p.add_argument('--dawn-batch',type=int,default=1024)
    return p.parse_args()


if __name__=='__main__':
    args=parse()
    Path(args.out).mkdir(parents=True,exist_ok=True)
    run_lock=(Path(args.out)/'run.lock').open('w')
    fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        {'offline':offline,'native':native,'warm':residual,'random':residual}[args.stage](args)
    except BaseException as exc:
        atomic_json(Path(args.out)/'FAILED.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
