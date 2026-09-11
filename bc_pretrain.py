"""Pure flow-matching BC on QAM's exact policy architecture.

Only one actor is optimized offline. No critic or adjoint computation occurs.
Export initializes all QAM flow copies from BC, and all critics from the same
fresh random draw used by the previous random-critic DAWN experiments.
"""
import argparse
import fcntl
from pathlib import Path
import time
import run as common
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from utils.networks import ActorVectorField
from dawn_agent import DawnAgent


@flax.struct.dataclass
class FlowBC:
    actor: TrainState
    rng: object

    @classmethod
    def create(cls, observations, actions, seed):
        template=common.qam_template(observations,actions,seed)
        network=ActorVectorField(action_dim=25,hidden_dims=(512,)*4,layer_norm=False)
        tx=optax.chain(optax.clip_by_global_norm(1.),optax.adam(3e-4))
        state=TrainState.create(apply_fn=network.apply,params=template.network.params['modules_actor_slow'],tx=tx)
        return cls(state,template.rng)

    def loss(self,params,batch,key):
        xkey,tkey=jax.random.split(key)
        actions=batch['actions'].reshape((batch['actions'].shape[0],-1))
        noise=jax.random.normal(xkey,actions.shape)
        t=jax.random.uniform(tkey,(actions.shape[0],1))
        xt=(1-t)*noise+t*actions
        target=actions-noise
        prediction=self.actor.apply_fn({'params':params},batch['observations'],xt,t)
        return (jnp.square(prediction-target).mean(axis=-1)*batch['valid'][:,-1]).mean()

    @jax.jit
    def update(self,batch):
        rng,key=jax.random.split(self.rng)
        loss,grads=jax.value_and_grad(self.loss)(self.actor.params,batch,key)
        actor=self.actor.apply_gradients(grads=grads)
        return self.replace(actor=actor,rng=rng),dict(flow_loss=loss,actor_grad_norm=optax.global_norm(grads))

    @jax.jit
    def batch_update(self,batches):
        agent,info=jax.lax.scan(lambda a,b:a.update(b),self,batches)
        return agent,jax.tree_util.tree_map(lambda x:x.mean(),info)


def export_qam(bc,observations,actions,seed):
    qam=common.qam_template(observations,actions,seed)
    params=dict(qam.network.params)
    for name in ['actor_slow','target_actor_slow','actor_fast','target_actor_fast']:
        params['modules_'+name]=bc.actor.params
    fresh=DawnAgent.create(qam,observations,seed,warm=False)
    params['modules_critic']=fresh.critic.params
    params['modules_target_critic']=fresh.critic.params
    # No offline QAM optimizer exists. Start all online optimizers fresh.
    network=qam.network.replace(params=params,opt_state=qam.network.tx.init(params))
    return qam.replace(network=network,rng=bc.rng)


def main(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'DONE.json').exists():return
    env,ds=common.make_data(args);env.close();ex=ds.get_subset(0)
    bc=FlowBC.create(ex['observations'],ex['actions'],args.seed)
    initial_actor=common.tree_hash(bc.actor.params)
    initial_export=export_qam(bc,ex['observations'],ex['actions'],args.seed)
    qhash=common.tree_hash(initial_export.network.params['modules_critic'])
    common.atomic_json(out/'config.json',dict(vars(args),pretraining='pure flow-matching BC',dataset_size=ds.size,
        architecture='4x512 GELU, no LayerNorm',action_dim=5,horizon=5,flow_steps=10,batch_size=256,
        learning_rate=3e-4,gradient_clip=1.,critic_training=False,adjoint_matching=False,
        transfer='BC weights copied to slow/fast/targets; fresh random critics and all online optimizers'))
    common.atomic_json(out/'dataset_manifest.json',{p.name:dict(bytes=p.stat().st_size,sha256=common.file_hash(p))
        for p in Path(args.data).glob('cube-double-play-v0*.npz')})
    np.random.seed(args.seed)
    step,elapsed=0,0.
    if (out/'latest_bc.pkl').exists():
        bc,saved=common.load_checkpoint(out/'latest_bc.pkl',bc)
        step,elapsed=saved['step'],saved['elapsed'];np.random.set_state(saved['numpy_rng'])
    if step==0:common.evaluate(initial_export,out,0,args)
    checkpoints=sorted(set(x for x in [100000,250000,args.offline_steps] if x<=args.offline_steps))
    while step<args.offline_steps:
        n=min(args.block,args.offline_steps-step,min(x for x in checkpoints if x>step)-step)
        tick=time.monotonic()
        bc,info=bc.batch_update(common.seq_batch(ds,n,256));metrics=common.floats(info)
        step+=n;elapsed+=time.monotonic()-tick
        if step%args.log_interval==0 or step==args.offline_steps:
            common.log(out/'metrics.jsonl',dict(step=step,elapsed_seconds=elapsed,updates_per_second=step/max(elapsed,1e-6),**metrics))
            common.atomic_json(out/'progress.json',dict(stage='bc_offline',step=step,target=args.offline_steps,elapsed_seconds=elapsed))
        if step%50000==0 or step in checkpoints:
            common.save_checkpoint(out/'latest_bc.pkl',bc,step=step,elapsed=elapsed,numpy_rng=np.random.get_state())
        if step in checkpoints:
            policy=export_qam(bc,ex['observations'],ex['actions'],args.seed)
            assert common.tree_hash(policy.network.params['modules_critic'])==qhash
            common.evaluate(policy,out,step,args)
    policy=export_qam(bc,ex['observations'],ex['actions'],args.seed)
    common.save_checkpoint(out/'final_bc.pkl',bc,step=step,elapsed=elapsed)
    common.save_checkpoint(out/'final.pkl',policy,step=step,pretraining='pure BC',critic_trained=False)
    assert common.tree_hash(bc.actor.params)!=initial_actor
    assert common.tree_hash(policy.network.params['modules_critic'])==qhash
    common.atomic_json(out/'DONE.json',dict(step=step,checkpoint_sha256=common.file_hash(out/'final.pkl'),
        flow_hash=common.flow_hash(policy),bc_actor_hash=common.tree_hash(bc.actor.params),initial_actor_hash=initial_actor,
        q_hash=qhash,target_q_hash=common.tree_hash(policy.network.params['modules_target_critic']),
        critic_trained=False,elapsed_seconds=elapsed))


def parse():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    p.add_argument('--data',default=str(common.ROOT/'data'));p.add_argument('--seed',type=int,default=0)
    p.add_argument('--offline-steps',type=int,default=500000);p.add_argument('--block',type=int,default=100)
    p.add_argument('--log-interval',type=int,default=5000);p.add_argument('--eval-episodes',type=int,default=50)
    return p.parse_args()


if __name__=='__main__':
    args=parse();Path(args.out).mkdir(parents=True,exist_ok=True)
    with (Path(args.out)/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main(args)
        except BaseException as exc:
            common.atomic_json(Path(args.out)/'FAILED.json',dict(type=type(exc).__name__,message=str(exc)))
            raise
