"""Fresh, paired Task5 long-budget runs using the unchanged historical residual loop."""
import argparse
import fcntl
import json
import pickle
import shutil
from pathlib import Path
import sys

import run as common
import flax
import jax.numpy as jnp
import actor_input_agent as implementation
from audit_cube5 import task_manifest
from run_actor_input_ablation import check_input_operator, hard_backup

RUNS = common.ROOT / 'runs/task5_long_critic_shared_20260910'
OFFLINE = common.ROOT / 'runs/cube5/task5_offline/final.pkl'
OFFLINE_SHA = 'fc0ac82260d16262da8e0c1d7266fd19fc024dfc52c4b40b26e64b6463a1938a'
FROZEN = {
    'run.py': '28eb970ed70773d744a072460a4595b8fe26285812464b13a6a302bb25f043fd',
    'actor_input_agent.py': '1b00a08158cfbb2052656e332d7510dcab2b94d06e601812e25b544c9ef2db52',
    'run_cube5.py': 'd668ef30f404636ce2cfaee78996ec0ae598d88380662f41d752538ea97a0146',
    'run_actor_input_ablation.py': 'b3c63015441fe8340200a0eb1a03cc2369f249ac7cc08dbeba37bf12bcfcc950',
}


class SmokePause(Exception):
    pass


class WarmupCollected(Exception):
    pass


def paired_result(offline, current):
    assert offline['episodes'] == current['episodes']
    counts = dict(gained=0, lost=0, both_success=0, both_failure=0)
    gained, lost = [], []
    for a, b in zip(offline['records'], current['records']):
        for k in ('episode', 'reset_seed', 'initial_hash'):
            assert a[k] == b[k], (k, a[k], b[k])
        x, y = bool(a['success']), bool(b['success'])
        category = ('both_success' if x else 'gained') if y else ('lost' if x else 'both_failure')
        counts[category] += 1
        if category == 'gained': gained.append(a['episode'] + 1)
        if category == 'lost': lost.append(a['episode'] + 1)
    return dict(**counts, episodes=current['episodes'], offline_success=offline['success'],
                success=current['success'], success_delta=current['success']-offline['success'],
                offline_return_mean=offline['return_mean'], return_mean=current['return_mean'],
                return_delta=current['return_mean']-offline['return_mean'],
                gained_episodes_1based=gained, lost_episodes_1based=lost)


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--smoke', action='store_true')
    extra.add_argument('--stop-after-checkpoint', type=int, default=0)
    extra.add_argument('--collect-warmup', action='store_true')
    extra.add_argument('--shared-warmup')
    selected, rest = extra.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    args = common.parse()
    assert args.stage in ('warm', 'random') and args.seed == 0
    assert args.offline_steps == 500000 and args.dawn_batch == 256
    args.offline_checkpoint = str(OFFLINE)
    args.task, args.environment = 5, 'cube-double-play-singletask-task5-v0'
    args.td_target, args.actor_input = 'hard', 'base_action'
    common.ENV = args.environment
    if selected.smoke:
        assert args.online_steps == 120 and args.warmup == 20
        assert args.eval_episodes == args.final_episodes == 2
        common.GRID = (0, 40, 80, 120)
    else:
        assert args.online_steps == 500000 and args.warmup == 20000
        assert args.eval_episodes == args.final_episodes == 100
        assert selected.stop_after_checkpoint == 0
        common.GRID = (0, 50000, 100000, 200000, 500000)
    assert not (selected.collect_warmup and selected.shared_warmup)
    if selected.collect_warmup:
        assert args.stage == 'warm'
        common.GRID = (0, args.warmup)
    args.shared_warmup = selected.shared_warmup
    args.collect_warmup = selected.collect_warmup
    out = Path(args.out)
    assert out.resolve().is_relative_to(RUNS.resolve())
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name, sha in FROZEN.items():
        assert common.file_hash(common.ROOT/name) == sha, name
    assert common.file_hash(OFFLINE) == OFFLINE_SHA
    source_names = [*FROZEN, 'run_task5_long_critic.py', 'audit_cube5.py',
                    'actor_input_diagnostics.py', 'dawn_agent.py',
                    'vendor/qam/agents/qam.py', 'vendor/qam/utils/networks.py',
                    'vendor/qam/utils/datasets.py']
    requested = dict(vars(args), smoke=selected.smoke, grid=list(common.GRID),
                     offline_sha256=OFFLINE_SHA,
                     source_hashes={n:common.file_hash(common.ROOT/n) for n in source_names})
    if selected.shared_warmup:
        requested['shared_warmup_sha256'] = common.file_hash(Path(selected.shared_warmup)/'latest.pkl')
    config_path = out/'requested_config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == requested, 'Resume configuration or source differs'
    else:
        common.atomic_json(config_path, requested)
    if (out/'DONE.json').exists():
        assert common.file_hash(out/'final.pkl') == json.loads((out/'DONE.json').read_text())['checkpoint_sha256']
        return
    offline_done = json.loads((OFFLINE.parent/'DONE.json').read_text())
    audit = json.loads((common.ROOT/'runs/cube5/TASK_AUDIT_PASSED.json').read_text())
    assert audit['status'] == 'passed'
    original_data, original_eval, original_save = common.make_data, common.evaluate, common.save_checkpoint

    def checked_data(config):
        env, ds = original_data(config)
        manifest = task_manifest(env, ds)
        assert manifest == audit['tasks'][4]['manifest']
        assert manifest == json.loads((OFFLINE.parent/'task_manifest.json').read_text())
        common.atomic_json(out/'task_manifest.json', manifest)
        return env, ds

    class Factory:
        @staticmethod
        def create(qam, obs, seed, warm):
            assert qam.config['edit_scale'] == 0 and qam.config['inv_temp'] == 1
            agent = implementation.ActorInputAgent.create(qam, obs, seed, warm).replace(condition_on_base=True)
            assert obs.shape[-1] == 37 and agent.action_dim == 25
            check_input_operator(agent, obs, out)
            q_hash, target_hash = common.tree_hash(agent.critic.params), common.tree_hash(agent.target_params)
            if warm:
                assert q_hash == offline_done['q_hash'] and target_hash == offline_done['target_q_hash']
            else:
                assert q_hash == target_hash and q_hash != offline_done['q_hash']
            assert common.flow_hash(qam) == offline_done['flow_hash']
            initial_hash = common.tree_hash(flax.serialization.to_state_dict(agent))
            if warm:
                previous = json.loads((common.ROOT/'runs/cube5/task5_warm/actual_agent_config.json').read_text())
                assert initial_hash == previous['full_initial_agent_hash']
            non_q = dict(actor=flax.serialization.to_state_dict(agent.actor),
                         log_alpha=agent.log_alpha, alpha_opt_state=agent.alpha_opt_state,
                         rng=agent.rng, updates=agent.updates)
            common.atomic_json(out/'actual_agent_config.json', dict(task=5, td_target='hard',
                inherited_Q_and_target=warm, actor_input='state+base_action', actor_input_dim=62,
                observation_dim=37, base_feature_dim=25, hidden_dims=[256]*3, activation='relu',
                gaussian_output=True, target_entropy=-25, utd=.25, utd_unit='primitive_environment_step',
                batch_size=256, warmup=args.warmup, residual_scale=agent.res_scale,
                target_tau=agent.tau, action_dim=agent.action_dim, horizon=5, discount=.99,
                replay='online_only_growing_chunk_buffer', target_aggregation='minimum',
                actor_Q_aggregation='minimum', critic_ensemble=10, critic_hidden_dims=[512]*4,
                critic_activation='gelu', critic_layer_norm=True, learning_rate=1e-4,
                gradient_clip_norm=50, initial_alpha=.01, actor_entropy_enabled=True,
                automatic_alpha_enabled=True, critic_optimizer='fresh_adam',
                full_initial_agent_hash=initial_hash, non_Q_initial_hash=common.tree_hash(non_q)))
            if selected.shared_warmup and not (out/'latest.pkl').exists():
                source = Path(selected.shared_warmup)
                collected = json.loads((source/'WARMUP_COLLECTED.json').read_text())
                assert collected['checkpoint_sha256'] == common.file_hash(source/'latest.pkl')
                source_config = json.loads((source/'actual_agent_config.json').read_text())
                assert source_config['non_Q_initial_hash'] == common.tree_hash(non_q)
                with (source/'latest.pkl').open('rb') as f:
                    payload = pickle.load(f)
                assert int(payload['agent']['updates']) == 0
                del payload['agent']
                assert args.warmup <= payload['step'] < args.warmup + 5
                # Only the agent's freshly created Q/target differ between branches.
                original_save(out/'latest.pkl', agent, **payload)
                prefix = json.loads((source/'warmup.json').read_text())
                assert common.tree_hash(payload['replay']) == prefix['replay_hash']
                prefix.update(critic_hash=q_hash, shared_from=str(source))
                common.atomic_json(out/'warmup.json', prefix)
                shutil.copy2(source/f'eval_000000_{args.eval_episodes:03d}.json',
                             out/f'eval_000000_{args.eval_episodes:03d}.json')
                common.atomic_json(out/'shared_initialization.json', dict(
                    collector_checkpoint_sha256=collected['checkpoint_sha256'],
                    replay_hash=prefix['replay_hash'], step=payload['step'], updates=0,
                    full_initial_agent_hash=initial_hash, critic_hash=q_hash, target_hash=target_hash,
                    copied_fields=['replay','episode','trace','ob','keys','numpy_rng']))
            return agent

    def evaluate(qam, folder, step, config, **kwargs):
        if selected.collect_warmup and step > 0:
            return json.loads((out/f'eval_000000_{args.eval_episodes:03d}.json').read_text())
        agent = kwargs['dawn']
        before = common.tree_hash(flax.serialization.to_state_dict(agent))
        rng_before = common.tree_hash(common.np.random.get_state())
        result = original_eval(qam, folder, step, config, **kwargs)
        if step > 0 and not kwargs.get('deterministic', False):
            mean_kwargs = dict(kwargs, episodes=args.eval_episodes, deterministic=True, suffix='_mean_residual')
            mean = original_eval(qam, folder, step, config, **mean_kwargs)
            baseline = json.loads((out/f'eval_000000_{args.eval_episodes:03d}.json').read_text())
            common.atomic_json(out/f'paired_{step:06d}.json', dict(
                step=step, updates=int(agent.updates), sampled=paired_result(baseline, result),
                mean_residual=paired_result(baseline, mean)))
        assert before == common.tree_hash(flax.serialization.to_state_dict(agent))
        assert rng_before == common.tree_hash(common.np.random.get_state())
        common.log(out/'evaluation_integrity.jsonl', dict(step=step, agent_unchanged=True,
                   training_numpy_rng_unchanged=True, agent_hash=before))
        return result

    def save(path, agent, **payload):
        original_save(path, agent, **payload)
        if Path(path).name == 'latest.pkl':
            step = payload['step']
            assert int(agent.updates) == max(0, (step-args.warmup)//4)
            # Retain slim milestone weights; the latest file retains all replay and RNG state.
            original_save(out/f'checkpoint_{step:06d}.pkl', agent, step=step, updates=int(agent.updates))
            common.atomic_json(out/'checkpoint_state.json', dict(step=step, updates=int(agent.updates),
                               agent_hash=common.tree_hash(flax.serialization.to_state_dict(agent)),
                               replay_chunks=len(payload['replay'])))
            if selected.collect_warmup:
                assert int(agent.updates) == 0 and args.warmup <= step < args.warmup+5
                common.atomic_json(out/'WARMUP_COLLECTED.json', dict(step=step, updates=0,
                    checkpoint_sha256=common.file_hash(out/'latest.pkl'),
                    replay_hash=common.tree_hash(payload['replay']), frozen_flow_hash=offline_done['flow_hash']))
                raise WarmupCollected()
            if selected.stop_after_checkpoint and step >= selected.stop_after_checkpoint:
                common.atomic_json(out/'SMOKE_PAUSED.json', dict(step=step))
                raise SmokePause()

    r, d = jnp.array([-3.,0.,-5.]), jnp.array([0.,.95,.95])
    qs = jnp.array([[-8.,-7.,-6.],[-9.,-4.,-8.]])
    lp, alpha = jnp.array([3.,-4.,2.]), jnp.array(.02)
    soft, hard = implementation.soft_backup(r,d,qs,alpha,lp), hard_backup(r,d,qs,alpha,lp)
    common.np.testing.assert_allclose(soft-hard, -d*alpha*lp, rtol=0, atol=1e-6)
    common.np.testing.assert_array_equal(hard, hard_backup(r,d,qs,alpha*100,lp*10))
    common.atomic_json(out/'backup_math.json', dict(status='passed', hard_independent_of_entropy=True))
    implementation.soft_backup = hard_backup
    common.make_data, common.DawnAgent = checked_data, Factory
    common.evaluate, common.save_checkpoint = evaluate, save
    try:
        common.residual(args)
        done = json.loads((out/'DONE.json').read_text())
        assert done['steps'] == args.online_steps
        assert done['updates'] == (args.online_steps-args.warmup)//4
        assert done['final_flow_hash'] == offline_done['flow_hash']
        assert common.file_hash(out/'final.pkl') == done['checkpoint_sha256']
        common.atomic_json(out/'CHECKS_PASSED.json', dict(status='passed', task=5, stage=args.stage,
                           original_training_loop=True, task_data_verified=True,
                           frozen_base_verified=True, checkpoint_hash_verified=True, updates=done['updates']))
    except WarmupCollected:
        assert selected.collect_warmup
    except SmokePause:
        assert selected.smoke
    except BaseException as exc:
        common.atomic_json(out/'FAILED.json', dict(type=type(exc).__name__, message=str(exc)))
        raise


if __name__ == '__main__':
    main()
