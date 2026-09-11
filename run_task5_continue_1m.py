"""Continue each completed Task5 500k run to 1M with its full replay and optimizer state."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sys

import run as common
import flax
import jax.numpy as jnp
import actor_input_agent as implementation
from audit_cube5 import task_manifest
from run_actor_input_ablation import hard_backup
from run_task5_long_critic import OFFLINE, OFFLINE_SHA, paired_result

RUNS = common.ROOT/'runs/task5_continue_1m_20260910'
SOURCE = common.ROOT/'runs/task5_long_critic_shared_20260910'
FINAL_SHA = {
    'warm':'e7a849104ebfdf60f1dd9d90350479db45b7d8cae412fb80c8bd673d32fa1d2a',
    'random':'d8c6a28568bf9bd4f378488e1e08d39d3adbb6fc105a763d21191e6b0764b24e'}


class SmokePause(Exception):
    pass


def state_hash(agent):
    return common.tree_hash(flax.serialization.to_state_dict(agent))


def subset_evaluation(result, episodes):
    result = dict(result)
    result['records'] = result['records'][:episodes]
    result['episodes'] = episodes
    result['success'] = sum(r['success'] for r in result['records'])/episodes
    result['return_mean'] = sum(r['return_'] for r in result['records'])/episodes
    return result


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--smoke', action='store_true')
    extra.add_argument('--stop-after-checkpoint', type=int, default=0)
    selected, rest = extra.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    args = common.parse()
    assert args.stage in ('warm','random') and args.seed == 0
    assert args.offline_steps == 500000 and args.warmup == 20000 and args.dawn_batch == 256
    assert args.online_steps == (500120 if selected.smoke else 1000000)
    assert args.eval_episodes == args.final_episodes == (2 if selected.smoke else 100)
    assert selected.smoke or selected.stop_after_checkpoint == 0
    args.task, args.environment, args.td_target, args.actor_input = 5, 'cube-double-play-singletask-task5-v0', 'hard', 'base_action'
    args.offline_checkpoint = str(OFFLINE)
    common.ENV = args.environment
    common.GRID = (500040,500080,500120) if selected.smoke else (750000,1000000)
    source, out = SOURCE/args.stage, Path(args.out)
    assert out.resolve().is_relative_to(RUNS.resolve())
    out.mkdir(parents=True,exist_ok=True)
    lock = (out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old_sources = json.loads((SOURCE/'source_manifest.json').read_text())
    for name, sha in old_sources.items():
        assert common.file_hash(common.ROOT/name) == sha, name
    assert common.file_hash(OFFLINE) == OFFLINE_SHA
    assert common.file_hash(source/'final.pkl') == FINAL_SHA[args.stage]
    source_done = json.loads((source/'DONE.json').read_text())
    assert source_done['steps'] == 500000 and source_done['updates'] == 120000
    assert json.loads((source/'CHECKS_PASSED.json').read_text())['status']=='passed'
    actual = json.loads((source/'actual_agent_config.json').read_text())
    assert actual['td_target']=='hard' and actual['actor_input_dim']==62
    assert actual['batch_size']==256 and actual['utd']==.25 and actual['residual_scale']==.1
    source_latest_sha = common.file_hash(source/'latest.pkl')
    request = dict(vars(args), smoke=selected.smoke, grid=list(common.GRID),
        origin_resume_sha256=source_latest_sha, origin_final_sha256=FINAL_SHA[args.stage],
        origin_directory=str(source), source_hashes=dict(old_sources,
            **{'run_task5_continue_1m.py':common.file_hash(__file__)}))
    if (out/'requested_config.json').exists():
        assert json.loads((out/'requested_config.json').read_text()) == request
    else:common.atomic_json(out/'requested_config.json',request)
    if (out/'DONE.json').exists():
        assert common.file_hash(out/'final.pkl') == json.loads((out/'DONE.json').read_text())['checkpoint_sha256']
        return

    original_data, original_eval = common.make_data, common.evaluate
    original_save, original_restore = common.save_checkpoint, common.restore_env

    def checked_data(config):
        env,ds=original_data(config)
        manifest=task_manifest(env,ds)
        assert manifest==json.loads((source/'task_manifest.json').read_text())
        common.atomic_json(out/'task_manifest.json',manifest)
        return env,ds

    class Factory:
        @staticmethod
        def create(qam,obs,seed,warm):
            template=implementation.ActorInputAgent.create(qam,obs,seed,warm).replace(condition_on_base=True)
            agent,sv=common.load_checkpoint(source/'latest.pkl',template)
            final,_=common.load_checkpoint(source/'final.pkl',template)
            assert state_hash(agent)==state_hash(final)
            del final
            assert sv['step']==500000 and int(agent.updates)==120000
            assert common.flow_hash(qam)==source_done['final_flow_hash']
            assert common.tree_hash(agent.critic.params)==source_done['final_critic_hash']
            assert common.tree_hash(agent.actor.params)==source_done['final_actor_hash']
            required={'replay','episode','trace','ob','keys','numpy_rng','elapsed'}
            assert required.issubset(sv) and len(sv['keys'])==3 and len(sv['replay'])>0
            # Byte-identical copy preserves all saved arrays, optimizers, replay and RNG states.
            if not (out/'latest.pkl').exists():
                shutil.copy2(source/'latest.pkl',out/'latest.pkl')
                assert common.file_hash(out/'latest.pkl')==source_latest_sha
                common.atomic_json(out/'progress.json',dict(stage=args.stage,step=500000,
                    target=args.online_steps,updates=120000,status='restoring'))
            for step,suffix in [(0,''),(500000,''),(500000,'_mean_residual')]:
                origin=source/f'eval_{step:06d}_100{suffix}.json'
                destination=out/f'eval_{step:06d}_{args.eval_episodes:03d}{suffix}.json'
                if not destination.exists():
                    common.atomic_json(destination,subset_evaluation(json.loads(origin.read_text()),args.eval_episodes))
            common.atomic_json(out/'actual_agent_config.json',dict(actual,
                initialization_for_this_run='restored complete 500k agent including all optimizers',
                critic_optimizer='restored_500k_adam_state', additional_warmup_steps=0,
                full_initial_agent_hash=state_hash(agent),
                non_Q_initial_hash=common.tree_hash(dict(actor=flax.serialization.to_state_dict(agent.actor),
                    log_alpha=agent.log_alpha,alpha_opt_state=agent.alpha_opt_state,
                    rng=agent.rng,updates=agent.updates)),
                restored_alpha=float(jnp.exp(agent.log_alpha)),
                origin_steps=500000,origin_updates=120000,target_steps=args.online_steps,
                target_updates=(args.online_steps-20000)//4))
            last_matches=bool(common.np.array_equal(sv['replay'][-1]['next_observations'],sv['ob']))
            common.atomic_json(out/'origin_state_verified.json',dict(status='passed',arm=args.stage,
                origin_steps=sv['step'],origin_updates=int(agent.updates),
                source_resume_sha256=source_latest_sha,source_final_sha256=FINAL_SHA[args.stage],
                restored_agent_hash=state_hash(agent),latest_agent_equals_final_agent=True,
                replay_chunks=len(sv['replay']),episode=sv['episode'],trace_length=len(sv['trace']),
                all_saved_fields_preserved=True,optimizer_and_alpha_states_preserved=True,
                last_replay_next_observation_matches_current=last_matches,
                budget_boundary_handling='Replan one chunk at saved 500k observation; original checkpoint does not retain an unexecuted chunk suffix.'))
            return agent

    def restore(env,episode,seed,trace,expected):
        ob=original_restore(env,episode,seed,trace,expected)
        common.log(out/'restore_checks.jsonl',dict(episode=episode,trace_length=len(trace),
            observation_max_abs_error=float(common.np.max(common.np.abs(ob-expected)))))
        return ob

    def evaluate(qam,folder,step,config,**kwargs):
        agent=kwargs['dawn'];before=state_hash(agent)
        np_before=common.tree_hash(common.np.random.get_state())
        result=original_eval(qam,folder,step,config,**kwargs)
        if not kwargs.get('deterministic',False):
            mean=original_eval(qam,folder,step,config,**dict(kwargs,episodes=args.eval_episodes,
                deterministic=True,suffix='_mean_residual'))
            baseline=json.loads((out/f'eval_000000_{args.eval_episodes:03d}.json').read_text())
            origin_sampled=json.loads((out/f'eval_500000_{args.eval_episodes:03d}.json').read_text())
            origin_mean=json.loads((out/f'eval_500000_{args.eval_episodes:03d}_mean_residual.json').read_text())
            common.atomic_json(out/f'paired_{step:07d}.json',dict(step=step,updates=int(agent.updates),
                sampled=paired_result(baseline,result),mean_residual=paired_result(baseline,mean),
                versus_500k=dict(sampled=paired_result(origin_sampled,result),
                                mean_residual=paired_result(origin_mean,mean))))
        assert before==state_hash(agent)
        assert np_before==common.tree_hash(common.np.random.get_state())
        return result

    def save(path,agent,**payload):
        original_save(path,agent,**payload)
        if Path(path).name=='latest.pkl':
            step=payload['step'];assert int(agent.updates)==(step-20000)//4
            if not selected.smoke:
                original_save(out/f'checkpoint_{step:07d}.pkl',agent,step=step,updates=int(agent.updates))
            if selected.stop_after_checkpoint and step>=selected.stop_after_checkpoint:
                common.atomic_json(out/'SMOKE_PAUSED.json',dict(step=step,updates=int(agent.updates)))
                raise SmokePause()

    implementation.soft_backup=hard_backup
    common.make_data,common.DawnAgent=checked_data,Factory
    common.restore_env,common.evaluate,common.save_checkpoint=restore,evaluate,save
    try:
        common.residual(args)
        done=json.loads((out/'DONE.json').read_text())
        assert done['steps']==args.online_steps and done['updates']==(args.online_steps-20000)//4
        assert done['final_flow_hash']==source_done['final_flow_hash']
        assert common.file_hash(out/'final.pkl')==done['checkpoint_sha256']
        common.atomic_json(out/'CHECKS_PASSED.json',dict(status='passed',arm=args.stage,
            steps=done['steps'],updates=done['updates'],additional_steps=done['steps']-500000,
            additional_updates=done['updates']-120000,full_state_continuation=True,
            additional_warmup_steps=0,original_training_loop=True,frozen_base_verified=True))
    except SmokePause:
        assert selected.smoke
    except BaseException as exc:
        common.atomic_json(out/'FAILED.json',dict(type=type(exc).__name__,message=str(exc)))
        raise


if __name__=='__main__':main()
