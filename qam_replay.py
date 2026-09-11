"""Original QAM transition replay with frozen base proposals for DAWN."""
from pathlib import Path
import fcntl
import json
import time

import run as common
import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def indexed_proposals(qam, observations, indices, seed, kind):
    root = jax.random.fold_in(jax.random.PRNGKey(seed + 84000), kind)
    keys = jax.vmap(lambda i: jax.random.fold_in(root, i))(indices.astype(jnp.uint32))
    return jax.vmap(lambda ob, key: qam.sample_actions(ob, key))(observations, keys)


def original_sequences_with_indices(replay, batch_size):
    # The original sampler does not return indices. Replay is sampled on this
    # process's single training thread, so replaying this one RNG draw exposes
    # indices without changing its samples, RNG advancement, or boundary logic.
    state = np.random.get_state()
    indices = np.random.randint(replay.size - common.HORIZON + 1, size=batch_size)
    np.random.set_state(state)
    batch = replay.sample_sequence(batch_size, common.HORIZON, common.GAMMA)
    lengths = batch['valid'].sum(axis=-1).astype(np.int64)
    return batch, indices, indices + lengths - 1


def cache_offline(args):
    root = Path(args.cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    with (root/'build.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env, ds = common.make_data(args)
        env.close()
        ex = ds.get_subset(0)
        qam = common.qam_template(ex['observations'], ex['actions'], args.seed)
        qam, saved = common.load_checkpoint(args.offline_checkpoint, qam)
        assert saved['step'] == args.offline_steps
        expected = dict(offline_sha256=common.file_hash(args.offline_checkpoint),
                        dataset_size=ds.size, seed=args.seed, horizon=common.HORIZON,
                        dataset_sha256=common.file_hash(Path(args.data)/'cube-double-play-v0.npz'))
        manifest_path = root/'manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            assert all(manifest[k] == v for k, v in expected.items())
            for name, digest in manifest['files'].items():
                assert common.file_hash(root/name) == digest
            return
        start = time.monotonic()
        hashes = {}
        for kind, field in enumerate(['observations', 'next_observations']):
            name = 'base.npy' if kind == 0 else 'next_base.npy'
            tmp = root/(name+'.tmp')
            dest = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float32,
                                            shape=(ds.size, common.HORIZON * ex['actions'].shape[-1]))
            for offset in range(0, ds.size, args.cache_block):
                n = min(args.cache_block, ds.size-offset)
                indices = np.arange(offset, offset+args.cache_block, dtype=np.uint32)
                obs = np.asarray(ds[field][np.minimum(indices, ds.size-1)], np.float32)
                values = np.asarray(indexed_proposals(qam, obs, indices, args.seed, kind))
                dest[offset:offset+n] = values[:n]
                if offset % (args.cache_block*100) == 0:
                    common.atomic_json(root/'progress.json', dict(field=field, rows=offset+n,
                                       total=ds.size, elapsed_seconds=time.monotonic()-start))
            dest.flush()
            del dest
            tmp.replace(root/name)
            hashes[name] = common.file_hash(root/name)
        common.atomic_json(manifest_path, dict(expected, files=hashes,
                           elapsed_seconds=time.monotonic()-start,
                           proposal_seed='fold_in(PRNGKey(seed + 84000), kind), then row index'))


class QamReplay:
    def __init__(self, ds, capacity, qam, cache_dir, seed):
        self.offline_size = ds.size
        self.replay = common.ReplayBuffer.create_from_initial_dataset(dict(ds), ds.size+capacity+common.HORIZON)
        self.qam, self.seed = qam, seed
        self.offline_base = [np.load(Path(cache_dir)/name, mmap_mode='r')
                             for name in ['base.npy', 'next_base.npy']]
        dim = self.offline_base[0].shape[-1]
        self.online_base = [np.full((capacity, dim), np.nan, np.float32) for _ in range(2)]
        self.sampled_rows = self.sampled_online = self.sampled_invalid = 0

    def add(self, transition):
        self.replay.add_transition(transition)

    def set_actual_proposals(self, first_online_index, last_online_index, base, next_base):
        self.online_base[0][first_online_index] = base
        self.online_base[1][last_online_index] = next_base

    def proposals(self, indices, kind):
        result = np.empty((len(indices), self.offline_base[kind].shape[-1]), np.float32)
        offline = indices < self.offline_size
        result[offline] = self.offline_base[kind][indices[offline]]
        online_indices = indices[~offline] - self.offline_size
        if len(online_indices):
            missing = np.unique(online_indices[np.isnan(self.online_base[kind][online_indices, 0])])
            if len(missing):
                rows = missing+self.offline_size
                field = 'observations' if kind == 0 else 'next_observations'
                # Fixed-sized blocks avoid recompiling for each changing miss count.
                for offset in range(0, len(rows), 128):
                    real = rows[offset:offset+128]
                    padded = np.pad(real, (0, 128-len(real)), mode='edge')
                    obs = np.asarray(self.replay[field][padded], np.float32)
                    values = np.asarray(indexed_proposals(self.qam, obs, padded.astype(np.uint32), self.seed, kind))
                    self.online_base[kind][real-self.offline_size] = values[:len(real)]
            result[~offline] = self.online_base[kind][online_indices]
        assert np.isfinite(result).all()
        return result

    def sample(self, batch_size):
        seq, indices, next_indices = original_sequences_with_indices(self.replay, batch_size)
        valid = np.asarray(seq['valid'][:, -1], np.float32)
        self.sampled_rows += len(indices)
        self.sampled_online += int((indices >= self.offline_size).sum())
        self.sampled_invalid += int((valid == 0).sum())
        return dict(observations=np.asarray(seq['observations'], np.float32),
                    actions=np.asarray(seq['actions'], np.float32).reshape(batch_size, -1),
                    rewards=np.asarray(seq['rewards'][:, -1], np.float32),
                    discounts=np.asarray(common.GAMMA**common.HORIZON * seq['masks'][:, -1], np.float32),
                    next_observations=np.asarray(seq['next_observations'][:, -1], np.float32),
                    base_actions=self.proposals(indices, 0), next_base_actions=self.proposals(next_indices, 1),
                    valid=valid)

    def stats(self):
        return dict(replay_size=self.replay.size, offline_transitions=self.offline_size,
                    online_transitions=self.replay.size-self.offline_size,
                    sampled_rows=self.sampled_rows, sampled_online=self.sampled_online,
                    sampled_online_fraction=self.sampled_online/max(1, self.sampled_rows),
                    masked_sequence_fraction=self.sampled_invalid/max(1, self.sampled_rows))
