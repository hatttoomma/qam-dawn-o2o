"""Antmaze reset uses global NumPy noise as well as Gymnasium's local RNG."""
from types import SimpleNamespace
import gymnasium
import numpy as np
import run as common

_evaluate = common.evaluate

def deterministic_reset(env, seed):
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        env.action_space.seed(seed)
        return env.reset(seed=seed)
    finally:
        np.random.set_state(state)


class PairedReset(gymnasium.Wrapper):
    def reset(self, *, seed=None, **kwargs):
        assert seed is not None and not kwargs
        return deterministic_reset(self.env, seed)


def evaluate(*args, **kwargs):
    ogbench = common.ogbench
    def factory(*factory_args, **factory_kwargs):
        assert factory_kwargs.get('env_only') is True
        return PairedReset(ogbench.make_env_and_datasets(*factory_args, **factory_kwargs))
    # Only this serial evaluation call sees the wrapper. Never patch OGBench itself.
    common.ogbench = SimpleNamespace(make_env_and_datasets=factory)
    try:
        return _evaluate(*args, **kwargs)
    finally:
        common.ogbench = ogbench
