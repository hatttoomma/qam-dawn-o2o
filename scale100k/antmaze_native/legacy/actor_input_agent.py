"""Matched-width actor input ablation; all update equations retain the DAWN implementation."""
from functools import partial
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from utils.networks import Value


def actor_input(obs, base, condition_on_base):
    """Same input width in both arms; mask only the base-action feature block."""
    base = jnp.asarray(base, dtype=obs.dtype)
    features = base if condition_on_base else jnp.zeros_like(base)
    return jnp.concatenate([obs, features], axis=-1)


class ResidualActor(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, obs):
        x = obs
        for _ in range(3):
            x = nn.relu(nn.Dense(256)(x))
        mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(.01), name='mean')(x)
        raw = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(.01), name='logstd')(x)
        logstd = -20. + 11. * (jnp.tanh(raw) + 1.)
        return mean, logstd


def sample_residual(apply_fn, params, obs, key, deterministic=False):
    mean, logstd = apply_fn({'params': params}, obs)
    noise = jax.random.normal(key, mean.shape)
    raw = mean if deterministic else mean + jnp.exp(logstd) * noise
    action = jnp.tanh(raw)
    # DAWN entropy convention: unscaled tanh-Gaussian residual coordinates.
    logp = -.5 * ((raw - mean) / jnp.exp(logstd)) ** 2 - logstd - .5 * jnp.log(2. * jnp.pi)
    logp -= jnp.log(1. - action ** 2 + 1e-6)
    return action, logp.sum(axis=-1), logstd.mean()


def soft_backup(reward, bootstrap_discount, next_qs, alpha, next_logp):
    return reward + bootstrap_discount * (jnp.min(next_qs, axis=0) - alpha * next_logp)


@flax.struct.dataclass
class ActorInputAgent:
    critic: TrainState
    target_params: Any
    actor: TrainState
    log_alpha: Any
    alpha_opt_state: Any
    rng: Any
    updates: Any
    action_dim: int = flax.struct.field(pytree_node=False)
    condition_on_base: bool = flax.struct.field(pytree_node=False, default=False)
    res_scale: float = flax.struct.field(pytree_node=False, default=.1)
    tau: float = flax.struct.field(pytree_node=False, default=.01)

    @classmethod
    def create(cls, qam, obs, seed, warm):
        dim = qam.config['action_dim'] * qam.config['horizon_length']
        qdef = Value(hidden_dims=(512, 512, 512, 512), layer_norm=True, num_ensembles=10)
        adef = ResidualActor(dim)
        key = jax.random.PRNGKey(seed + 71000)
        qkey, akey, rng = jax.random.split(key, 3)
        qparams = qdef.init(qkey, obs, jnp.zeros((dim,), jnp.float32))['params']
        target = qparams
        if warm:
            qparams = qam.network.params['modules_critic']
            target = qam.network.params['modules_target_critic']
        tx = optax.chain(optax.clip_by_global_norm(50.), optax.adam(1e-4))
        critic = TrainState.create(apply_fn=qdef.apply, params=qparams, tx=tx)
        actor = TrainState.create(apply_fn=adef.apply, params=adef.init(akey, actor_input(obs, jnp.zeros((*obs.shape[:-1], dim), dtype=obs.dtype), False))['params'], tx=tx)
        log_alpha = jnp.asarray(jnp.log(.01), jnp.float32)
        return cls(critic, target, actor, log_alpha, optax.adam(1e-4).init(log_alpha), rng, jnp.int32(0), dim)

    @partial(jax.jit, static_argnames=('deterministic',))
    def sample(self, obs, base, key, deterministic=False):
        u, logp, _ = sample_residual(self.actor.apply_fn, self.actor.params, actor_input(obs, base, self.condition_on_base), key, deterministic)
        raw = base + self.res_scale * u
        return jnp.clip(raw, -1., 1.), {
            'residual_abs_mean': jnp.abs(self.res_scale * u).mean(),
            'action_clip_fraction': (jnp.abs(raw) > 1.).mean(),
            'residual_entropy': -logp.mean(),
        }

    @jax.jit
    def update(self, batch):
        rng, target_key, actor_key, alpha_key = jax.random.split(self.rng, 4)
        alpha = jnp.exp(self.log_alpha)
        u_next, next_logp, _ = sample_residual(self.actor.apply_fn, self.actor.params, actor_input(batch['next_observations'], batch['next_base_actions'], self.condition_on_base), target_key)
        next_actions = jnp.clip(batch['next_base_actions'] + self.res_scale * u_next, -1., 1.)
        next_qs = self.critic.apply_fn({'params': self.target_params}, batch['next_observations'], next_actions)
        target = jax.lax.stop_gradient(soft_backup(batch['rewards'], batch['discounts'], next_qs, alpha, next_logp))

        def critic_loss(params):
            qs = self.critic.apply_fn({'params': params}, batch['observations'], batch['actions'])
            loss = jnp.square(qs - target[None, :]).mean(axis=-1).sum()
            return loss, {'critic_loss': loss, 'q_mean': qs.mean(), 'q_min': qs.min(),
                          'q_max': qs.max(), 'target_mean': target.mean(),
                          'target_min': target.min(), 'target_max': target.max()}

        (_, info), qgrads = jax.value_and_grad(critic_loss, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=qgrads)

        def actor_loss(params):
            u, logp, logstd = sample_residual(self.actor.apply_fn, params, actor_input(batch['observations'], batch['base_actions'], self.condition_on_base), actor_key)
            actions = jnp.clip(batch['base_actions'] + self.res_scale * u, -1., 1.)
            qs = critic.apply_fn({'params': critic.params}, batch['observations'], actions)
            q = qs.min(axis=0)
            loss = (alpha * logp - q).mean()
            return loss, {'actor_loss': loss, 'entropy': -logp.mean(), 'logstd': logstd,
                          'residual_abs_mean': jnp.abs(self.res_scale * u).mean(),
                          'action_clip_fraction': (jnp.abs(batch['base_actions'] + self.res_scale*u) > 1.).mean()}

        (_, actor_info), agrads = jax.value_and_grad(actor_loss, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=agrads)
        _, new_logp, _ = sample_residual(actor.apply_fn, actor.params, actor_input(batch['observations'], batch['base_actions'], self.condition_on_base), alpha_key)
        alpha_grad = -jax.lax.stop_gradient((new_logp - self.action_dim).mean())
        alpha_updates, alpha_state = optax.adam(1e-4).update(alpha_grad, self.alpha_opt_state, self.log_alpha)
        log_alpha = optax.apply_updates(self.log_alpha, alpha_updates)
        target_params = jax.tree_util.tree_map(lambda t, p: (1. - self.tau)*t + self.tau*p,
                                              self.target_params, critic.params)
        info.update(actor_info)
        info.update(alpha=alpha, critic_grad_norm=optax.global_norm(qgrads), actor_grad_norm=optax.global_norm(agrads))
        return self.replace(critic=critic, actor=actor, target_params=target_params, log_alpha=log_alpha,
                            alpha_opt_state=alpha_state, rng=rng, updates=self.updates + 1), info

    @jax.jit
    def batch_update(self, batches):
        def body(agent, batch):
            return agent.update(batch)
        agent, info = jax.lax.scan(body, self, batches)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), info)
