"""Check behavior actions, bootstrap masks, exact source counts and resume RNG."""
import json
from pathlib import Path
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from balanced_replay import BalancedReplay


class Dataset:
    size = 100
    def get_subset(self, indices):
        n = len(indices)
        return dict(observations=np.full((n,29), -2., np.float32),
            next_observations=np.full((n,29), -3., np.float32),
            actions=np.full((n,8), -.7, np.float32),
            rewards=np.full(n, -1., np.float32),
            masks=(indices % 2).astype(np.float32))


class QAM:
    config = {'horizon_length': 1}
    def sample_actions(self, obs, key):
        return jnp.tanh(obs[:, :8] * .05 + jax.random.normal(key, (obs.shape[0],8)) * .1)


def main():
    ds, qam = Dataset(), QAM()
    online = [dict(observations=np.full(29, 2.,np.float32), next_observations=np.full(29,3.,np.float32),
                   actions=np.full(8,.7,np.float32),rewards=np.float32(0.),discounts=np.float32(.99),
                   base_actions=np.full(8,.6,np.float32),next_base_actions=np.full(8,.5,np.float32))
              for _ in range(48)]
    with tempfile.TemporaryDirectory() as temp:
        sampler = BalancedReplay(ds,qam,0,.99,temp,48,40)
        np.random.seed(31)
        state = np.random.get_state()
        batch = sampler.sample(online,256,1)
        assert (batch['observations'][:128] == -2).all() and (batch['observations'][128:] == 2).all()
        np.testing.assert_array_equal(batch['actions'][:128],np.full((128,8),-.7,np.float32))
        assert not np.allclose(batch['base_actions'][:128],batch['actions'][:128])
        assert set(batch['discounts'][:128].tolist()) == {0.,float(np.float32(.99))}
        np.testing.assert_array_equal(batch['base_actions'][128:],np.full((128,8),.6,np.float32))
        audit = json.loads((Path(temp)/'replay_audit.json').read_text())
        assert audit['offline_per_batch'] == audit['online_per_batch'] == 128
        assert audit['offline_draws'] == audit['online_draws'] == 256
        np.random.set_state(state)
        restored = BalancedReplay(ds,qam,0,.99,temp,48,40).sample(online,256,1)
        for key in batch:
            np.testing.assert_array_equal(batch[key],restored[key])
    return dict(status='passed', exact_128_128=True, dataset_behavior_action_preserved=True,
                terminal_bootstrap_masks_preserved=True, base_features_not_behavior_actions=True,
                online_stored_base_preserved=True, checkpoint_resume_rng_reproducible=True)


if __name__ == '__main__':
    print(json.dumps(main()))
