"""Policy Decorator adapted to inherited QAM critics and QAM replay.

The SAC losses/actor parameterization are shared with the previous DAWN arm.
PD differences are alpha initialization and the collection schedule below.
Source: tongzhoumu/policy_decorator@92ba9ba442587ae5989c286355ace2200a0537fb,
online/pi_dec_diffusion_maniskill2.py. See POLICY_DECORATOR_PROTOCOL.md for
explicit deviations from the released ManiSkill implementation.
"""
import flax
import jax
import jax.numpy as jnp
import optax
from dawn_replay_agent import DawnQamReplayAgent


def residual_probability(primitive_step, horizon):
    if horizon <= 0:
        raise ValueError('Progressive exploration horizon must be positive')
    return min(max(primitive_step / horizon, 0.), 1.)


def collect_action(agent, observation, base, key, gate_draw, primitive_step,
                   learning_starts, progressive_horizon):
    """One Bernoulli gate for an entire chunk; never scales residual amplitude."""
    probability = residual_probability(primitive_step, progressive_horizon)
    enabled = gate_draw < probability
    random_phase = primitive_step < learning_starts
    if not enabled:
        return base, enabled, random_phase, probability
    if random_phase:
        residual = jax.random.uniform(key, base.shape, minval=-1., maxval=1.)
        action = jnp.clip(base + agent.res_scale * residual, -1., 1.)
    else:
        action, _ = agent.sample(observation, base, key)
    return action, enabled, random_phase, probability


@flax.struct.dataclass
class PolicyDecoratorAgent(DawnQamReplayAgent):
    @classmethod
    def create(cls, qam, obs, seed, warm=True):
        if not warm:
            raise ValueError('This experiment requires inherited critic weights')
        agent = super().create(qam, obs, seed, warm=True)
        # Official autotune=True initializes log_sac_alpha=0, so alpha=1.
        # The CLI sac_alpha=.2 is used only with autotune=False.
        log_alpha = jnp.asarray(0., jnp.float32)
        return agent.replace(log_alpha=log_alpha,
                             alpha_opt_state=optax.adam(1e-4).init(log_alpha))
