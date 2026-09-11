"""Ablate new-data eligibility while keeping the original native QAM loop."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import flax
import run as common

RUNS = common.ROOT / 'runs/newdata_ablation'


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--task', type=int, choices=[1, 2], required=True)
    parser.add_argument('--train-data', choices=['all', 'offline'], required=True)
    extra, rest = parser.parse_known_args();sys.argv = [sys.argv[0], *rest]
    args = common.parse();assert args.stage == 'native'
    args.task = extra.task;args.train_data = extra.train_data
    args.environment = f'cube-double-play-singletask-task{args.task}-v0'
    if args.task == 2: args.task2_offline_from_scratch = True
    common.ENV = args.environment
    reference = common.ROOT / ('runs' if args.task == 1 else 'runs/task2')
    if '--offline-checkpoint' not in rest: args.offline_checkpoint = str(reference / 'offline/final.pkl')
    assert Path(args.offline_checkpoint).resolve() == (reference / 'offline/final.pkl').resolve()
    assert args.offline_steps == 500000 and args.seed == 0
    out = Path(args.out);assert out.resolve().is_relative_to(RUNS.resolve())
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'run.lock').open('w');fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    requested = dict(vars(args), offline_sha256=common.file_hash(args.offline_checkpoint))
    assert requested['offline_sha256'] == json.loads((reference / 'offline/DONE.json').read_text())['checkpoint_sha256']

    def persist(name, value):
        p = out / name
        if p.exists(): assert json.loads(p.read_text()) == json.loads(json.dumps(value)), name
        else: common.atomic_json(p, value)

    persist('requested_config.json', requested)
    if (out / 'DONE.json').exists() and (out / 'DATA_AUDIT_PASSED.json').exists(): return
    state = dict(update_batches=0, batch_samples=0, online_start_samples=0,
        sequences_touching_online=0, largest_sampled_index=-1,
        index_chain_sha256='0' * 64, first_collection_prefix=None)
    box = {}
    original_data, original_prepare = common.make_data, common.prepare_online
    original_replay, original_save = common.ReplayBuffer, common.save_checkpoint
    original_load, original_eval = common.load_checkpoint, common.evaluate

    def checked_data(config):
        env, ds = original_data(config)
        assert env.unwrapped._reward_task_id == args.task
        assert env.spec.id == f'cube-double-singletask-task{args.task}-v0'
        manifest = dict(environment=args.environment, gym_environment=env.spec.id,
            reward_task_id=int(env.unwrapped._reward_task_id), dataset_size=ds.size,
            target_cube_xyzs=env.unwrapped._data.mocap_pos.tolist(),
            observation_shape=list(ds['observations'].shape), action_shape=list(ds['actions'].shape),
            max_episode_steps=env.spec.max_episode_steps,
            rewards_hash=common.tree_hash(ds['rewards']), masks_hash=common.tree_hash(ds['masks']),
            observations_hash=common.tree_hash(ds['observations']), actions_hash=common.tree_hash(ds['actions']))
        audit = json.loads((common.ROOT / 'runs/task2/TASK_AUDIT_PASSED.json').read_text())
        assert audit['status'] == 'passed'
        for k in ['environment', 'reward_task_id', 'target_cube_xyzs', 'rewards_hash', 'masks_hash']:
            assert manifest[k] == audit['tasks'][args.task - 1][k], k
        if args.task == 2: assert manifest == json.loads((reference / 'offline/task_manifest.json').read_text())
        persist('task_manifest.json', manifest)
        box['offline_hash'] = common.tree_hash(dict(ds))
        return env, ds

    def checked_prepare(config):
        result = original_prepare(config);qam = result[-1]
        persist('initial_state.json', dict(agent=common.tree_hash(flax.serialization.to_state_dict(qam)),
            params=common.tree_hash(qam.network.params), optimizer=common.tree_hash(qam.network.opt_state),
            rng=common.tree_hash(qam.rng), flow=common.flow_hash(qam), network_step=int(qam.network.step)))
        persist('effective_qam_config.json', dict(qam.config))
        return result

    class TrackedReplay:
        @classmethod
        def create_from_initial_dataset(cls, data, size):
            obj = cls();obj.live = original_replay.create_from_initial_dataset(data, size)
            obj.offline = common.Dataset.create(**data)
            obj.offline_size = obj.offline.size;box['replay'] = obj
            assert common.tree_hash(dict(obj.offline)) == box['offline_hash']
            return obj

        def __getattr__(self, name): return getattr(self.live, name)

        def add_transition(self, transition): self.live.add_transition(transition)

        def sample_sequence(self, batch_size, sequence_length, discount):
            assert batch_size == 256 and sequence_length == common.HORIZON and discount == common.GAMMA
            if state['first_collection_prefix'] is None:
                count = self.live.size - self.offline_size
                assert count == args.native_start
                prefix = {k: v[self.offline_size:self.live.size].copy() for k, v in self.live.items()}
                np.savez_compressed(out / 'preupdate_collection.npz', **prefix)
                state['first_collection_prefix'] = dict(transitions=count, hash=common.tree_hash(prefix),
                    snapshot_sha256=common.file_hash(out / 'preupdate_collection.npz'))
            source = self.live if args.train_data == 'all' else self.offline
            real_randint = np.random.randint;calls = []

            def tracked_randint(*a, **kw):
                indexes = real_randint(*a, **kw);calls.append(np.asarray(indexes).copy())
                assert a == (source.size - sequence_length + 1,) and kw == {'size': batch_size}
                return indexes

            # Observe the original sampler's draw without consuming another RNG value.
            np.random.randint = tracked_randint
            try: batch = source.sample_sequence(batch_size, sequence_length, discount)
            finally: np.random.randint = real_randint
            assert len(calls) == 1;indexes = calls[0]
            assert indexes.shape == (256,) and indexes.min() >= 0
            last = int(indexes.max()) + sequence_length - 1
            assert last < source.size
            if args.train_data == 'offline': assert last < self.offline_size
            state['update_batches'] += 1;state['batch_samples'] += batch_size
            state['online_start_samples'] += int((indexes >= self.offline_size).sum())
            state['sequences_touching_online'] += int((indexes + sequence_length - 1 >= self.offline_size).sum())
            state['largest_sampled_index'] = max(state['largest_sampled_index'], last)
            state['index_chain_sha256'] = hashlib.sha256(bytes.fromhex(state['index_chain_sha256']) +
                indexes.astype('<i8').tobytes()).hexdigest()
            return batch

    def snapshot():
        replay = box['replay']
        return dict(state, task=args.task, train_data=args.train_data, offline_size=replay.offline_size,
            collected_transitions=replay.live.size - replay.offline_size,
            training_pool_size=replay.live.size if args.train_data == 'all' else replay.offline_size)

    def save(path, agent, **payload):
        if Path(path).name == 'latest.pkl': payload['newdata_ablation_audit'] = snapshot()
        return original_save(path, agent, **payload)

    def load(path, agent):
        agent, payload = original_load(path, agent)
        if Path(path).parent == out and Path(path).name == 'latest.pkl':
            audit = payload['newdata_ablation_audit']
            assert audit['task'] == args.task and audit['train_data'] == args.train_data
            for k in state: state[k] = audit[k]
            assert state['update_batches'] == payload['updates']
        return agent, payload

    def evaluate(qam, folder, step, config, **kw):
        value = original_eval(qam, folder, step, config, **kw)
        common.atomic_json(out / f'data_audit_{step:06d}_{value["episodes"]:03d}.json', snapshot())
        return value

    common.make_data, common.prepare_online = checked_data, checked_prepare
    common.ReplayBuffer, common.save_checkpoint = TrackedReplay, save
    common.load_checkpoint, common.evaluate = load, evaluate
    try:
        common.native(args)
        final = snapshot();done = json.loads((out / 'DONE.json').read_text())
        assert done['steps'] == args.online_steps and done['updates'] == state['update_batches']
        assert done['updates'] == args.online_steps - args.native_start + 1
        assert final['collected_transitions'] == args.online_steps
        assert state['batch_samples'] == done['updates'] * 256
        assert common.tree_hash(dict(box['replay'].offline)) == box['offline_hash']
        if args.train_data == 'offline':
            assert state['online_start_samples'] == state['sequences_touching_online'] == 0
            assert final['training_pool_size'] == final['offline_size']
        common.atomic_json(out / 'DATA_AUDIT_PASSED.json', dict(status='passed', **final,
            offline_data_unchanged=True, original_native_loop=True, original_sequence_sampler=True))
    except BaseException as e:
        common.atomic_json(out / 'FAILED.json', dict(type=type(e).__name__, message=str(e)))
        raise


if __name__ == '__main__': main()
