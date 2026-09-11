"""Read-only TD diagnostics; use existing evaluation rollouts and fixed RNG keys."""
from pathlib import Path
import json

import jax
import jax.numpy as jnp
import numpy as np

import run as common
from dawn_agent import sample_residual


@jax.jit
def probe(agent, batch, key):
    nk, ak = jax.random.split(key)
    alpha = jnp.exp(agent.log_alpha)
    u, logp, _ = sample_residual(agent.actor.apply_fn, agent.actor.params, batch['next_observations'], nk)
    na = jnp.clip(batch['next_base_actions'] + agent.res_scale * u, -1, 1)
    nq = agent.critic.apply_fn({'params': agent.target_params}, batch['next_observations'], na).min(axis=0)
    hard = batch['rewards'] + batch['discounts'] * nq
    entropy_term = -batch['discounts'] * alpha * logp
    qs = agent.critic.apply_fn({'params': agent.critic.params}, batch['observations'], batch['actions'])
    cu, _, _ = sample_residual(agent.actor.apply_fn, agent.actor.params, batch['observations'], ak)
    base = jnp.clip(batch['base_actions'], -1, 1)

    def value(uu):
        a = jnp.clip(batch['base_actions'] + agent.res_scale * uu, -1, 1)
        return agent.critic.apply_fn({'params': agent.critic.params}, batch['observations'], a).min(axis=0)

    residual_q = value(cu)
    base_q = agent.critic.apply_fn({'params': agent.critic.params}, batch['observations'], base).min(axis=0)
    grads = jax.grad(lambda uu: value(uu).sum())(cu)
    return dict(rewards=batch['rewards'], discounts=batch['discounts'], next_min_q=nq,
        next_logp=logp, entropy_target_term=entropy_term, hard_target=hard, soft_target=hard + entropy_term,
        current_qs=qs, base_q=base_q, residual_q=residual_q, residual_q_difference=residual_q-base_q,
        residual_gradient_norm=jnp.linalg.norm(grads, axis=-1))


def fixed_probe(out, task, agent, payload):
    """Called after the original latest checkpoint save; never touches training RNG."""
    out = Path(out)
    meta_path = out / 'warmup.json'
    if not meta_path.exists(): return
    meta = json.loads(meta_path.read_text())
    snap = out / 'fixed_probe.npz'
    if not snap.exists():
        prefix = payload['replay'][:meta['chunks']]
        assert common.tree_hash(prefix) == meta['replay_hash']
        ix = np.random.RandomState(44000 + task).randint(len(prefix), size=256)
        arrays = common.stack_replay(prefix, ix)
        np.savez_compressed(snap, **arrays)
        common.atomic_json(out / 'fixed_probe.json', dict(source='first base-only warmup replay',
            warmup_chunks=len(prefix), samples=256, indexes=ix.tolist(), sha256=common.file_hash(snap),
            independent_numpy_seed=44000+task, independent_jax_seed=55000+task))
    step = payload['step']
    path = out / f'probe_{step:06d}.json'
    if path.exists(): return
    with np.load(snap) as z: batch = {k: jnp.asarray(z[k]) for k in z.files}
    arrays = {k: np.asarray(v) for k, v in probe(agent, batch, jax.random.PRNGKey(55000+task)).items()}
    assert all(np.isfinite(a).all() for a in arrays.values())
    npz = out / f'probe_{step:06d}.npz';np.savez_compressed(npz, **arrays)
    q = arrays['current_qs']
    common.atomic_json(path, dict(step=step, updates=int(agent.updates), alpha=float(jnp.exp(agent.log_alpha)),
        samples=256, fixed_probe_sha256=common.file_hash(snap), array_sha256=common.file_hash(npz),
        entropy_target_mean=float(arrays['entropy_target_term'].mean()),
        entropy_target_abs_mean=float(np.abs(arrays['entropy_target_term']).mean()),
        reward_abs_mean=float(np.abs(arrays['rewards']).mean()),
        hard_td_mse=float(np.square(q-arrays['hard_target']).mean()),
        soft_td_mse=float(np.square(q-arrays['soft_target']).mean()),
        q_mean=float(q.mean()), base_q_mean=float(arrays['base_q'].mean()),
        residual_q_difference_mean=float(arrays['residual_q_difference'].mean()),
        residual_q_difference_abs_mean=float(np.abs(arrays['residual_q_difference']).mean()),
        residual_gradient_norm_mean=float(arrays['residual_gradient_norm'].mean())))


def evaluate_with_mc(original, qam, out, step, args, **kwargs):
    """Observe the exact original evaluator, adding no episodes or random draws."""
    episodes = kwargs.get('episodes', args.eval_episodes)
    suffix = kwargs.get('suffix', '')
    path = Path(out) / f'eval_{step:06d}_{episodes:03d}{suffix}.json'
    if episodes != args.eval_episodes or suffix or path.exists():
        return original(qam, out, step, args, **kwargs)
    agent = kwargs['dawn'];enabled = kwargs.get('residual_enabled', True)
    alpha = float(jnp.exp(agent.log_alpha))
    original_factory = common.ogbench.make_env_and_datasets
    traces, active = [], {}

    class EnvProxy:
        def __init__(self, env): self.env = env
        def __getattr__(self, name): return getattr(self.env, name)
        def reset(self, *a, **kw):
            result = self.env.reset(*a, **kw)
            active.clear();active.update(observation=np.asarray(result[0]).copy(), steps=0,
                return_=0., discounted_return=0., future_entropy_return=0., action=None)
            return result
        def step(self, action):
            result = self.env.step(action)
            ob, reward, term, trunc, info = result
            active['return_'] += float(reward)
            active['discounted_return'] += common.GAMMA**active['steps'] * float(reward)
            active['steps'] += 1
            if term or trunc:
                traces.append(dict(active, terminated=bool(term), truncated=bool(trunc)))
            return result

    class QamProxy:
        def sample_actions(self, *a, **kw):
            result = qam.sample_actions(*a, **kw)
            if active['steps'] == 0: active['action'] = np.asarray(result).copy()
            return result

    class AgentProxy:
        def sample(self, *a, **kw):
            result = agent.sample(*a, **kw)
            if active['steps'] == 0: active['action'] = np.asarray(result[0]).copy()
            else:
                active['future_entropy_return'] += common.GAMMA**active['steps'] * alpha * float(result[1]['residual_entropy'])
            return result

    def factory(*a, **kw):
        assert kw.get('env_only')
        return EnvProxy(original_factory(*a, **kw))

    common.ogbench.make_env_and_datasets = factory
    try:
        result = original(QamProxy(), out, step, args, **dict(kwargs, dawn=AgentProxy()))
    finally: common.ogbench.make_env_and_datasets = original_factory
    assert len(traces) == episodes
    obs = np.asarray([r['observation'] for r in traces], np.float32)
    actions = np.asarray([r['action'] for r in traces], np.float32)
    qs = np.asarray(agent.critic.apply_fn({'params': agent.critic.params}, jnp.asarray(obs), jnp.asarray(actions)))
    records = []
    for i, (tr, rec) in enumerate(zip(traces, result['records'])):
        assert tr['steps'] == rec['length'] and tr['return_'] == rec['return_']
        assert common.tree_hash(tr['observation']) == rec['initial_hash']
        records.append(dict(episode=i, reset_seed=rec['reset_seed'], initial_hash=rec['initial_hash'],
            success=rec['success'], length=tr['steps'], terminated=tr['terminated'], truncated=tr['truncated'],
            reward_mc=tr['discounted_return'], future_entropy_mc=tr['future_entropy_return'],
            soft_mc=tr['discounted_return']+tr['future_entropy_return'], q_mean=float(qs[:,i].mean()),
            q_min=float(qs[:,i].min())))
    target_kind = args.td_target
    target = np.asarray([r['soft_mc' if target_kind == 'soft' and enabled else 'reward_mc'] for r in records])
    error = qs.min(axis=0) - target
    completed = np.asarray([r['terminated'] for r in records])
    npz = Path(out) / f'mc_{step:06d}_{episodes:03d}.npz'
    np.savez_compressed(npz, observations=obs, actions=actions, qs=qs)
    common.atomic_json(Path(out) / f'mc_{step:06d}_{episodes:03d}.json', dict(step=step,
        episodes=episodes, td_target=target_kind, residual_enabled=enabled, alpha=alpha,
        source='same original evaluation episodes; no additional interaction',
        note='Truncated returns omit beyond-time-limit bootstrap; terminal subset is separately reported. Soft MC excludes first-action entropy. At residual-disabled points this is a base-policy reference.',
        all_horizon_limited_mae=float(np.abs(error).mean()),
        terminal_episodes=int(completed.sum()),
        terminal_mae=float(np.abs(error[completed]).mean()) if completed.any() else None,
        array_sha256=common.file_hash(npz), records=records))
    return result
