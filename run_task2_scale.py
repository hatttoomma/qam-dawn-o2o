"""Change only the static residual scale in the original task2 DAWN loop."""
import argparse
import fcntl
import json
from pathlib import Path
import sys

import run_task2 as task2

common = task2.common
RUNS = task2.RUNS / 'scale_sweep'


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--residual-scale', type=float, required=True)
    selected, rest = extra.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    args = common.parse()
    assert args.stage == 'warm', 'This sweep must inherit the task2 critic'
    assert selected.residual_scale in (.05, .1, .2, .3)
    if '--offline-checkpoint' not in rest:
        args.offline_checkpoint = str(task2.RUNS / 'offline/final.pkl')
    args.environment = task2.ENV
    args.task2_offline_from_scratch = True
    args.residual_scale = selected.residual_scale
    out = Path(args.out)
    assert out.resolve().is_relative_to(RUNS.resolve())
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Flax static fields are supplied by the restore template. Persist and check
    # scale before any load, so a checkpoint cannot be resumed with another scale.
    requested = dict(vars(args), offline_sha256=common.file_hash(args.offline_checkpoint))
    path = out / 'scale_config.json'
    if path.exists():
        assert json.loads(path.read_text()) == requested, 'Resume settings differ'
    else:
        common.atomic_json(path, requested)
    original_agent = common.DawnAgent

    class ScaleFactory:
        @staticmethod
        def create(qam, obs, seed, warm):
            assert warm
            agent = original_agent.create(qam, obs, seed, warm).replace(res_scale=args.residual_scale)
            actual = dict(residual_scale=agent.res_scale, tau=agent.tau,
                          action_dim=agent.action_dim, inherited_critic=True,
                          actor_hash=common.tree_hash(agent.actor.params),
                          critic_hash=common.tree_hash(agent.critic.params),
                          target_hash=common.tree_hash(agent.target_params))
            p = out / 'actual_agent_config.json'
            if p.exists():
                assert json.loads(p.read_text()) == actual
            else:
                common.atomic_json(p, actual)
            return agent

    common.DawnAgent = ScaleFactory
    try:
        common.residual(args)
    except BaseException as e:
        common.atomic_json(out / 'FAILED.json', dict(type=type(e).__name__, message=str(e)))
        raise


if __name__ == '__main__':
    main()
