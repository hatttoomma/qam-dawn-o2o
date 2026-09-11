"""Expand existing immutable QAM and conditioned residual loops to cube tasks 1–5."""
import argparse
import fcntl
import json
from pathlib import Path
import sys

import flax
import jax
import jax.numpy as jnp
import run as common
import actor_input_agent as implementation
import actor_input_diagnostics as diagnostics
from audit_cube5 import task_manifest
from run_actor_input_ablation import check_input_operator, hard_backup

RUNS = common.ROOT / 'runs/cube5'


def offline_directory(task):
    return common.ROOT / {1:'runs/offline', 2:'runs/task2/offline'}.get(task, f'runs/cube5/task{task}_offline')


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--task', type=int, choices=range(1,6), required=True)
    selected, rest = extra.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    args = common.parse()
    assert args.stage in ('offline','native','warm') and args.seed == 0
    args.task = selected.task
    args.environment = f'cube-double-play-singletask-task{args.task}-v0'
    args.td_target, args.actor_input = 'hard', 'base_action'
    if '--dawn-batch' not in rest: args.dawn_batch = 256
    assert args.dawn_batch == 256
    if '--offline-checkpoint' not in rest:
        args.offline_checkpoint = str(offline_directory(args.task)/'final.pkl')
    common.ENV = args.environment
    out = Path(args.out)
    assert out.resolve().is_relative_to(RUNS.resolve())
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'run.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    request = dict(vars(args))
    if args.stage != 'offline': request['offline_sha256'] = common.file_hash(args.offline_checkpoint)
    p = out/'requested_config.json'
    if p.exists(): assert json.loads(p.read_text()) == request, 'Resume configuration differs'
    else: common.atomic_json(p, request)
    audit = json.loads((RUNS/'TASK_AUDIT_PASSED.json').read_text())
    assert audit['status'] == 'passed'
    expected_manifest = audit['tasks'][args.task-1]['manifest']
    original_data, original_eval, original_save = common.make_data, common.evaluate, common.save_checkpoint

    def checked_data(config):
        env, ds = original_data(config)
        manifest = task_manifest(env, ds)
        assert manifest == expected_manifest
        common.atomic_json(out/'task_manifest.json', manifest)
        if args.stage != 'offline':
            directory = Path(args.offline_checkpoint).parent
            if args.task != 1 or directory != offline_directory(1):
                assert manifest == json.loads((directory/'task_manifest.json').read_text())
            assert common.file_hash(args.offline_checkpoint) == json.loads((directory/'DONE.json').read_text())['checkpoint_sha256']
        return env, ds

    def save(path, agent, **payload):
        original_save(path, agent, **payload)
        if args.stage == 'warm' and Path(path).name == 'latest.pkl':
            diagnostics.fixed_probe(out, args.task, agent, payload)

    def evaluate(qam, folder, step, config, **kwargs):
        before = common.tree_hash(flax.serialization.to_state_dict(kwargs.get('dawn', qam)))
        if args.stage == 'warm':
            result = diagnostics.evaluate_with_mc(original_eval, qam, folder, step, config, **kwargs)
        else: result = original_eval(qam, folder, step, config, **kwargs)
        after = common.tree_hash(flax.serialization.to_state_dict(kwargs.get('dawn', qam)))
        assert before == after
        common.log(out/'evaluation_integrity.jsonl', dict(step=step, episodes=result['episodes'],
            suffix=kwargs.get('suffix',''), before=before, after=after))
        return result

    original_factory = implementation.ActorInputAgent

    class Factory:
        @staticmethod
        def create(qam, obs, seed, warm):
            assert warm and qam.config['edit_scale'] == 0 and qam.config['inv_temp'] == 1
            agent = original_factory.create(qam, obs, seed, warm).replace(condition_on_base=True)
            check_input_operator(agent, obs, out)
            common.atomic_json(out/'actual_agent_config.json', dict(task=args.task, td_target='hard',
                inherited_Q_and_target=True, actor_input='base_action', actor_input_dim=62,
                observation_dim=37, base_feature_dim=25, hidden_dims=[256]*3, activation='relu',
                gaussian_output=True, target_entropy=-25, utd=.25, batch_size=256,
                residual_scale=agent.res_scale, target_tau=agent.tau, action_dim=agent.action_dim,
                target_aggregation='minimum', actor_Q_aggregation='minimum', critic_ensemble=10,
                actor_entropy_enabled=True, automatic_alpha_enabled=True,
                full_initial_agent_hash=common.tree_hash(flax.serialization.to_state_dict(agent))))
            return agent

    common.make_data, common.evaluate, common.save_checkpoint = checked_data, evaluate, save
    if args.stage == 'warm':
        r, d = jnp.array([-3.,0.,-5.]), jnp.array([0.,.95,.95])
        qs = jnp.array([[-8.,-7.,-6.],[-9.,-4.,-8.]])
        lp, alpha = jnp.array([3.,-4.,2.]), jnp.array(.02)
        soft, hard = implementation.soft_backup(r,d,qs,alpha,lp), hard_backup(r,d,qs,alpha,lp)
        common.np.testing.assert_allclose(soft-hard,-d*alpha*lp,rtol=0,atol=1e-6)
        common.np.testing.assert_array_equal(hard,hard_backup(r,d,qs,alpha*100,lp*10))
        common.atomic_json(out/'backup_math.json',dict(status='passed', hard_independent_of_entropy=True))
        implementation.soft_backup = hard_backup
        common.DawnAgent = Factory
    try:
        {'offline':common.offline, 'native':common.native, 'warm':common.residual}[args.stage](args)
        done = json.loads((out/'DONE.json').read_text())
        if args.stage == 'offline':
            assert done['step'] == args.offline_steps
            cfg = json.loads((out/'config.json').read_text())['qam_config']
            assert cfg['edit_scale'] == 0 and cfg['inv_temp'] == 1
        else:
            assert done['steps'] == args.online_steps
            offline = json.loads((Path(args.offline_checkpoint).parent/'DONE.json').read_text())
            if args.stage == 'warm':
                assert done['updates'] == int((args.online_steps-args.warmup)*.25)
                assert done['initial_hashes']['critic'] == offline['q_hash']
                assert done['initial_hashes']['target'] == offline['target_q_hash']
                assert done['final_flow_hash'] == offline['flow_hash']
            else:
                assert done['updates'] == max(0,args.online_steps-args.native_start+1)
                assert done['initial_flow_hash'] == offline['flow_hash']
                assert done['final_flow_hash'] != offline['flow_hash']
        assert common.file_hash(out/'final.pkl') == done['checkpoint_sha256']
        common.atomic_json(out/'CHECKS_PASSED.json', dict(status='passed', stage=args.stage,
            task=args.task, original_training_loop=True, task_data_verified=True,
            checkpoint_hash_verified=True, **({'updates':done['updates']} if args.stage!='offline' else {})))
    except BaseException as e:
        common.atomic_json(out/'FAILED.json',dict(type=type(e).__name__,message=str(e)))
        raise


if __name__ == '__main__': main()
