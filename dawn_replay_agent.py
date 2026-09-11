"""Same DAWN update, adding the QAM replay boundary mask to critic MSE only."""
import flax
import jax
import jax.numpy as jnp
import optax
from dawn_agent import DawnAgent, sample_residual, soft_backup


@flax.struct.dataclass
class DawnQamReplayAgent(DawnAgent):
    @jax.jit
    def update(self, batch):
        rng, target_key, actor_key, alpha_key = jax.random.split(self.rng, 4)
        alpha = jnp.exp(self.log_alpha)
        u_next, next_logp, _ = sample_residual(self.actor.apply_fn, self.actor.params, batch['next_observations'], target_key)
        next_actions = jnp.clip(batch['next_base_actions'] + self.res_scale * u_next, -1., 1.)
        next_qs = self.critic.apply_fn({'params': self.target_params}, batch['next_observations'], next_actions)
        target = jax.lax.stop_gradient(soft_backup(batch['rewards'], batch['discounts'], next_qs, alpha, next_logp))

        def critic_loss(params):
            qs = self.critic.apply_fn({'params': params}, batch['observations'], batch['actions'])
            loss = (jnp.square(qs - target[None, :]) * batch['valid'][None, :]).mean(axis=-1).sum()
            return loss, {'critic_loss': loss, 'q_mean': qs.mean(), 'q_min': qs.min(),
                          'q_max': qs.max(), 'target_mean': target.mean(),
                          'target_min': target.min(), 'target_max': target.max()}

        (_, info), qgrads = jax.value_and_grad(critic_loss, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=qgrads)

        def actor_loss(params):
            u, logp, logstd = sample_residual(self.actor.apply_fn, params, batch['observations'], actor_key)
            actions = jnp.clip(batch['base_actions'] + self.res_scale * u, -1., 1.)
            qs = critic.apply_fn({'params': critic.params}, batch['observations'], actions)
            q = qs.min(axis=0)
            loss = (alpha * logp - q).mean()
            return loss, {'actor_loss': loss, 'entropy': -logp.mean(), 'logstd': logstd,
                          'residual_abs_mean': jnp.abs(self.res_scale * u).mean(),
                          'action_clip_fraction': (jnp.abs(batch['base_actions'] + self.res_scale*u) > 1.).mean()}

        (_, actor_info), agrads = jax.value_and_grad(actor_loss, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=agrads)
        _, new_logp, _ = sample_residual(actor.apply_fn, actor.params, batch['observations'], alpha_key)
        alpha_grad = -jax.lax.stop_gradient((new_logp - self.action_dim).mean())
        alpha_updates, alpha_state = optax.adam(1e-4).update(alpha_grad, self.alpha_opt_state, self.log_alpha)
        log_alpha = optax.apply_updates(self.log_alpha, alpha_updates)
        target_params = jax.tree_util.tree_map(lambda t, p: (1. - self.tau)*t + self.tau*p,
                                              self.target_params, critic.params)
        info.update(actor_info)
        info['valid_sequence_fraction'] = batch['valid'].mean()
        info.update(alpha=alpha, critic_grad_norm=optax.global_norm(qgrads), actor_grad_norm=optax.global_norm(agrads))
        return self.replace(critic=critic, actor=actor, target_params=target_params, log_alpha=log_alpha,
                            alpha_opt_state=alpha_state, rng=rng, updates=self.updates + 1), info

