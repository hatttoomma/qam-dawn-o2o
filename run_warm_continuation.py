"""Continue the complete historical online-only DAWN state without retuning."""
import argparse
import fcntl
import json
import time
from pathlib import Path

import run as common
from run import (DawnAgent, GAMMA, HORIZON, ROOT, atomic_json, evaluate,
                 file_hash, floats, flow_hash, jax, jnp, load_checkpoint,
                 log, np, reset_train, restore_env, save_checkpoint,
                 stack_replay, tree_hash)

MILESTONES = (7500, 15000, 22500, 30000, 37500, 45001, 52500, 60000)
FINAL_SHA = 'd964ddc377febf020c81da30eb6efdb8c4681aa032580326ddbc8cbe3a7adbc1'


def scheduled_updates(step):
    return max(0, (int(step) - 20000) // 4)


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'DONE.json').exists():
        done = json.loads((out / 'DONE.json').read_text())
        if done['updates'] == args.target_updates:
            return
        assert args.smoke and done['updates'] < args.target_updates
    source = ROOT / 'runs/warm'
    config = json.loads((source / 'config.json').read_text())
    assert config['seed'] == 0 and config['warmup'] == 20000 and config['dawn_batch'] == 1024
    for name, expected in json.loads((ROOT / 'runs/source_manifest.json').read_text()).items():
        assert file_hash(ROOT / name) == expected, name
    assert file_hash(source / 'final.pkl') == FINAL_SHA
    sources = ['run_warm_continuation.py', 'WARM_CONTINUATION_PROTOCOL.md',
               'run.py', 'dawn_agent.py']
    manifest = {name: file_hash(ROOT / name) for name in sources}
    manifest_path = out / 'source_manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest
    else:
        atomic_json(manifest_path, manifest)
    args.seed = config['seed']
    args.eval_episodes = 50
    args.data = str(ROOT / 'data')
    env, ds = common.make_data(args)
    ex = ds.get_subset(0)
    del ds
    qam = common.qam_template(ex['observations'], ex['actions'], args.seed)
    offline_path = ROOT / 'runs/offline/final.pkl'
    assert file_hash(offline_path) == config['offline_sha256']
    qam, offline = load_checkpoint(offline_path, qam)
    assert offline['step'] == 500000
    frozen_hash = flow_hash(qam)
    template = DawnAgent.create(qam, ex['observations'], args.seed, warm=True)
    original, original_state = load_checkpoint(source / 'latest.pkl', template)
    original_final, _ = load_checkpoint(source / 'final.pkl', template)
    original_hash = tree_hash(original)
    assert original_hash == tree_hash(original_final)
    assert original_state['step'] == 50000 and int(original.updates) == 7500
    resume_path = out / 'latest.pkl'
    if resume_path.exists():
        agent, state = load_checkpoint(resume_path, template)
    else:
        agent, state = original, original_state
    step, episode, trace = state['step'], state['episode'], state['trace']
    replay, elapsed = state['replay'], state.get('continuation_elapsed', 0.)
    ob = restore_env(env, episode, args.seed, trace, state['ob'])
    restore_error = float(np.max(np.abs(ob - state['ob'])))
    base_rng, next_rng, res_rng = map(jnp.asarray, state['keys'])
    np.random.set_state(state['numpy_rng'])
    atomic_json(out / 'config.json', dict(vars(args), original_config=config,
        offline_sha256=config['offline_sha256'], origin_checkpoint_sha256=FINAL_SHA,
        origin_resume_sha256=file_hash(source / 'latest.pkl'),
        origin_agent_hash=original_hash, frozen_flow_hash=frozen_hash,
        update_milestones=list(MILESTONES), replay='uniform growing online-only',
        utd=.25, batch_size=1024, warmup_counted_from_original_start=20000,
        devices=[str(x) for x in jax.devices()],
        boundary_handling='Resume saved environment; resample missing action suffix at original 50k boundary'))
    log(out / 'events.jsonl', dict(event='restored', step=step, updates=int(agent.updates),
        replay_chunks=len(replay), episode=episode, trace_length=len(trace),
        environment_restore_max_error=restore_error, agent_hash=tree_hash(agent)))
    tick = time.monotonic()

    def checkpoint():
        save_checkpoint(out / 'latest.pkl', agent, step=step, elapsed=state['elapsed'],
            continuation_elapsed=elapsed, replay=replay, episode=episode, trace=trace,
            ob=ob, keys=[np.asarray(x) for x in (base_rng, next_rng, res_rng)],
            numpy_rng=np.random.get_state(), scheduled_updates=scheduled_updates(step))

    def milestone():
        nonlocal elapsed, tick
        update = int(agent.updates)
        if args.smoke or update not in MILESTONES:
            return
        elapsed += time.monotonic() - tick
        checkpoint()
        assert flow_hash(qam) == frozen_hash
        label = f'_u{update:06d}'
        weight_path = out / f'agent_u{update:06d}.pkl'
        if update != 7500 and not weight_path.exists():
            save_checkpoint(weight_path, agent, step=step, updates=update)
        result = evaluate(qam, out, step, args, dawn=agent, suffix=label)
        if update == 7500:
            reference = json.loads((source / 'eval_050000_050.json').read_text())
            # Hashes may vary with tiny MuJoCo observation roundoff; action outcomes
            # and return/length/success must reproduce the original checkpoint.
            fields = ('episode', 'reset_seed', 'success', 'return_', 'length', 'decisions')
            assert [[r[k] for k in fields] for r in result['records']] == [
                [r[k] for k in fields] for r in reference['records']], 'Restored evaluation mismatch'
        entries = [f'eval_{step:06d}_050{label}.json']
        if update in (45001, 60000):
            evaluate(qam, out, step, args, dawn=agent, episodes=100, suffix=label)
            evaluate(qam, out, step, args, dawn=agent, episodes=100,
                     deterministic=True, suffix=label + '_mean_residual')
            entries += [f'eval_{step:06d}_100{label}.json',
                        f'eval_{step:06d}_100{label}_mean_residual.json']
        atomic_json(out / f'milestone_u{update:06d}.json', dict(step=step, updates=update,
            agent_hash=tree_hash(agent), evaluation_files=entries,
            replay_chunks=len(replay), continuation_elapsed_seconds=elapsed))
        tick = time.monotonic()

    milestone()
    info = {}
    last_log = step // 1000
    last_save = step // 10000
    while int(agent.updates) < args.target_updates:
        debt = scheduled_updates(step) - int(agent.updates)
        assert debt >= 0
        if debt:
            # Also completes any updates pending at a checkpoint taken mid-batch.
            for _ in range(min(debt, args.target_updates - int(agent.updates))):
                idx = np.random.randint(len(replay), size=1024)
                agent, info = agent.update(stack_replay(replay, idx))
                milestone()
        else:
            start_ob = ob.copy()
            base_rng, key = jax.random.split(base_rng)
            base = qam.sample_actions(jnp.asarray(ob), key)
            res_rng, rkey = jax.random.split(res_rng)
            action, _ = agent.sample(jnp.asarray(ob), base, rkey)
            reward_sum, length, term, trunc = 0., 0, False, False
            for a in np.asarray(action).reshape(HORIZON, -1):
                ob, reward, term, trunc, _ = env.step(a)
                reward_sum += GAMMA ** length * float(reward)
                length += 1
                step += 1
                trace.append(a.copy())
                if term or trunc:
                    break
            next_rng, nkey = jax.random.split(next_rng)
            next_base = np.asarray(qam.sample_actions(jnp.asarray(ob), nkey))
            replay.append(dict(observations=start_ob, actions=np.asarray(action),
                rewards=np.float32(reward_sum), discounts=np.float32((GAMMA ** length) * (1. - term)),
                next_observations=ob.copy(), base_actions=np.asarray(base), next_base_actions=next_base))
            if term or trunc:
                episode += 1
                ob, trace = reset_train(env, episode, args.seed), []
        if step // 1000 > last_log and scheduled_updates(step) == int(agent.updates):
            elapsed += time.monotonic() - tick
            log(out / 'metrics.jsonl', dict(step=step, updates=int(agent.updates),
                continuation_elapsed_seconds=elapsed, replay_chunks=len(replay), **floats(info)))
            atomic_json(out / 'progress.json', dict(step=step, updates=int(agent.updates),
                target_updates=args.target_updates, replay_chunks=len(replay)))
            last_log = step // 1000
            tick = time.monotonic()
        if step // 10000 > last_save and scheduled_updates(step) == int(agent.updates):
            elapsed += time.monotonic() - tick
            checkpoint()
            last_save = step // 10000
            tick = time.monotonic()
    elapsed += time.monotonic() - tick
    checkpoint()
    assert flow_hash(qam) == frozen_hash
    assert int(agent.updates) == args.target_updates
    save_checkpoint(out / 'final.pkl', agent, step=step, updates=int(agent.updates))
    atomic_json(out / 'DONE.json', dict(arm='warm_extended', steps=step,
        updates=int(agent.updates), additional_updates=int(agent.updates)-7500,
        additional_environment_steps=step-50000, final_agent_hash=tree_hash(agent),
        final_flow_hash=frozen_hash, checkpoint_sha256=file_hash(out / 'final.pkl'),
        continuation_elapsed_seconds=elapsed, replay_chunks=len(replay)))
    env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=str(ROOT / 'runs/warm_extended'))
    p.add_argument('--target-updates', type=int, default=60000)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if not args.smoke:
        assert args.target_updates == 60000
    Path(args.out).mkdir(parents=True, exist_ok=True)
    lock = (Path(args.out) / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        main(args)
    except BaseException as e:
        atomic_json(Path(args.out) / 'FAILED.json', dict(type=type(e).__name__, message=str(e)))
        raise
