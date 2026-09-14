"""Two independent uniform buffers, with exactly half of each update from each.

Offline critic actions remain dataset behavior actions. The residual actor's
base-action features and the TD bootstrap base actions come from frozen QAM,
sampled on demand with a separate reproducible JAX key stream.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


class BalancedReplay:
    def __init__(self, dataset, qam, seed, discount, out, online_steps, warmup):
        assert qam.config['horizon_length'] == 1
        self.dataset, self.qam = dataset, qam
        self.discount, self.out = discount, Path(out)
        self.key = jax.random.PRNGKey(41000 + seed)
        self.total_updates = (online_steps - warmup) // 4
        self.warmup = warmup

    def sample(self, online, batch_size, update_index):
        assert batch_size == 256 and len(online) > self.warmup
        half = batch_size // 2
        online_indices = np.random.randint(len(online), size=half)
        offline_indices = np.random.randint(self.dataset.size, size=half)
        on = {k: np.asarray([online[int(i)][k] for i in online_indices], np.float32)
              for k in online[0]}
        raw = self.dataset.get_subset(offline_indices)
        # Keys depend only on seed and update index: normal checkpoint resume
        # restores NumPy RNG and agent.updates, so no additional RNG state is needed.
        key = jax.random.fold_in(self.key, update_index)
        features = np.concatenate([raw['observations'], raw['next_observations']], axis=0)
        bases = np.asarray(self.qam.sample_actions(jnp.asarray(features), key), np.float32)
        assert bases.shape == (batch_size, 8)
        off = dict(observations=raw['observations'], actions=raw['actions'],
                   rewards=raw['rewards'], discounts=self.discount * raw['masks'],
                   next_observations=raw['next_observations'],
                   base_actions=bases[:half], next_base_actions=bases[half:])
        assert on.keys() == off.keys()
        batch = {k: np.concatenate([np.asarray(off[k], np.float32), on[k]], axis=0)
                 for k in on}
        assert all(v.shape[0] == batch_size and np.isfinite(v).all() for v in batch.values())
        np.testing.assert_array_equal(batch['actions'][:half], raw['actions'])
        np.testing.assert_allclose(batch['discounts'][:half], self.discount * raw['masks'])
        if update_index == 0 or (update_index + 1) % 250 == 0 or update_index + 1 == self.total_updates:
            audit = dict(status='passed', update=update_index + 1, batch_size=batch_size,
                offline_per_batch=half, online_per_batch=half,
                offline_draws=(update_index + 1)*half, online_draws=(update_index + 1)*half,
                offline_buffer_size=self.dataset.size, online_buffer_size=len(online),
                offline_sampling='uniform_with_replacement', online_sampling='uniform_with_replacement',
                warmup_retained=True, offline_critic_actions='dataset_behavior_actions',
                offline_base_actions='frozen_QAM_stochastic_resampled_per_update',
                offline_discounts='gamma_times_dataset_masks',
                offline_reward_mean=float(np.mean(raw['rewards'])),
                online_reward_mean=float(np.mean(on['rewards'])),
                offline_base_behavior_action_abs_diff=float(np.abs(bases[:half]-raw['actions']).mean()))
            path = self.out / 'replay_audit.json'
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(audit, indent=2, allow_nan=False))
            temp.replace(path)
        return batch
