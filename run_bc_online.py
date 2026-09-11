"""BC checkpoint adapter for the unchanged QAM/DAWN/PD online runners."""
import argparse
import fcntl
import json
from pathlib import Path
import run as common
import run_qam_replay as replay_runner
import run_policy_decorator as pd_runner
import flax
import jax.numpy as jnp
import optax
from dawn_agent import DawnAgent
from policy_decorator_agent import PolicyDecoratorAgent

ORIGINAL_PREPARE=common.prepare_online


@flax.struct.dataclass
class BCRandomPolicyDecorator(PolicyDecoratorAgent):
    @classmethod
    def create(cls,qam,obs,seed,warm=True):
        # Runner's legacy warm flag does not apply: explicitly create fresh Q.
        agent=DawnAgent.create.__func__(cls,qam,obs,seed,warm=False)
        log_alpha=jnp.asarray(0.,jnp.float32)
        return agent.replace(log_alpha=log_alpha,alpha_opt_state=optax.adam(1e-4).init(log_alpha))


def prepare_bc(args):
    out,env,ds,ex,qam=ORIGINAL_PREPARE(args)
    offline=json.loads((Path(args.offline_checkpoint).parent/'DONE.json').read_text())
    assert offline['critic_trained']==False
    assert offline['checkpoint_sha256']==common.file_hash(args.offline_checkpoint)
    assert offline['step']==args.offline_steps
    fresh=DawnAgent.create(qam,ex['observations'],args.seed,warm=False)
    qhash=common.tree_hash(fresh.critic.params)
    assert common.tree_hash(qam.network.params['modules_critic'])==qhash==offline['q_hash']
    assert common.tree_hash(qam.network.params['modules_target_critic'])==qhash
    for name in ['actor_slow','target_actor_slow','actor_fast','target_actor_fast']:
        assert common.tree_hash(qam.network.params['modules_'+name])==offline['bc_actor_hash']
    common.atomic_json(out/'BC_TRANSFER.json',dict(pretraining='pure BC',critic_initialization='random',
        shared_bc_checkpoint_sha256=offline['checkpoint_sha256'],critic_hash=qhash,target_hash=qhash,
        flow_hash=common.flow_hash(qam),bc_actor_hash=offline['bc_actor_hash'],
        native_optimizer_hash=common.tree_hash(qam.network.opt_state),online_optimizers='fresh',
        runtime_adapter='original online loops with BC checkpoint and explicit random PD critic factory'))
    cfg=json.loads((out/'config.json').read_text())
    cfg.update(pretraining='pure BC',critic_initialization='random',bc_arm=args.method,online_optimizers='fresh')
    common.atomic_json(out/'config.json',cfg)
    return out,env,ds,ex,qam


def main(args):
    common.prepare_online=prepare_bc
    replay_runner.prepare_online=prepare_bc
    pd_runner.prepare_online=prepare_bc
    pd_runner.PolicyDecoratorAgent=BCRandomPolicyDecorator
    if args.method=='native':common.native(args)
    elif args.method=='dawn_online':common.residual(args)
    elif args.method=='dawn_qam':replay_runner.residual(args)
    else:pd_runner.residual(args)
    done_path=Path(args.out)/'DONE.json';done=json.loads(done_path.read_text())
    done.update(arm='bc_'+args.method,pretraining='pure BC',critic_initialization='random')
    common.atomic_json(done_path,done)


def parse():
    p=argparse.ArgumentParser()
    p.add_argument('--method',choices=['native','dawn_online','dawn_qam','policy_decorator'],required=True)
    p.add_argument('--out',required=True);p.add_argument('--data',default=str(common.ROOT/'data'))
    p.add_argument('--seed',type=int,default=0);p.add_argument('--offline-steps',type=int,default=500000)
    p.add_argument('--online-steps',type=int,default=50000)
    p.add_argument('--offline-checkpoint',default=str(common.ROOT/'runs/bc_offline/final.pkl'))
    p.add_argument('--eval-episodes',type=int,default=50);p.add_argument('--final-episodes',type=int,default=100)
    p.add_argument('--block',type=int,default=100);p.add_argument('--log-interval',type=int,default=5000)
    p.add_argument('--warmup',type=int);p.add_argument('--native-start',type=int,default=5000)
    p.add_argument('--dawn-batch',type=int,default=1024);p.add_argument('--prog-explore',type=int,default=30000)
    p.add_argument('--cache-dir',default=str(common.ROOT/'data/bc_base_cache_seed0'))
    p.add_argument('--cache-block',type=int,default=1024)
    args=p.parse_args()
    args.stage='bc_'+args.method  # DAWN's warm=(stage=='warm') remains false.
    if args.warmup is None:args.warmup=8000 if args.method=='policy_decorator' else 20000
    return args


if __name__=='__main__':
    args=parse();Path(args.out).mkdir(parents=True,exist_ok=True)
    with (Path(args.out)/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main(args)
        except BaseException as exc:
            common.atomic_json(Path(args.out)/'FAILED.json',dict(type=type(exc).__name__,message=str(exc)))
            raise
