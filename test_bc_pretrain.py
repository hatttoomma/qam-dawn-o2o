import run as common
import jax
import jax.numpy as jnp
import numpy as np
from bc_pretrain import FlowBC,export_qam
from dawn_agent import DawnAgent


def fixture():
    obs=jax.random.normal(jax.random.PRNGKey(4),(8,37))
    actions=jax.random.uniform(jax.random.PRNGKey(5),(8,5,5),minval=-1.,maxval=1.)
    batch=dict(observations=obs,actions=actions,valid=jnp.ones((8,5)),rewards=jnp.zeros((8,5)))
    return FlowBC.create(obs[0],actions[0,0],0),batch


def test_bc_is_reward_independent_and_masks_invalid_sequences():
    bc,batch=fixture();key=jax.random.PRNGKey(7)
    x,grads=jax.value_and_grad(bc.loss)(bc.actor.params,batch,key)
    y,other=jax.value_and_grad(bc.loss)(bc.actor.params,dict(batch,rewards=batch['rewards']+1000),key)
    assert float(x)==float(y) and common.tree_hash(grads)==common.tree_hash(other)
    zero,zgrads=jax.value_and_grad(bc.loss)(bc.actor.params,dict(batch,valid=jnp.zeros((8,5))),key)
    assert float(zero)==0 and all(np.all(np.asarray(v)==0) for v in jax.tree_util.tree_leaves(zgrads))


def test_export_preserves_bc_policy_and_creates_identical_untrained_critics():
    bc,batch=fixture();bc2,info=bc.update(batch);common.floats(info)
    qam=export_qam(bc2,batch['observations'][0],batch['actions'][0,0],0)
    original=export_qam(bc,batch['observations'][0],batch['actions'][0,0],0)
    assert common.tree_hash(original.network.params['modules_critic'])==common.tree_hash(qam.network.params['modules_critic'])
    fresh=DawnAgent.create(qam,batch['observations'][0],0,False)
    assert common.tree_hash(fresh.critic.params)==common.tree_hash(qam.network.params['modules_critic'])
    assert common.tree_hash(fresh.target_params)==common.tree_hash(qam.network.params['modules_target_critic'])
    for name in ['actor_slow','target_actor_slow','actor_fast','target_actor_fast']:
        assert common.tree_hash(qam.network.params['modules_'+name])==common.tree_hash(bc2.actor.params)
    key=jax.random.PRNGKey(10);flowkey,_=jax.random.split(key)
    noise=jax.random.normal(flowkey,(8,1,25));obs=batch['observations'][:,None,:]
    expected=qam.compute_flow_actions(obs,noise,model='slow')[:,0,:]
    actual=qam.sample_actions(batch['observations'],key)
    np.testing.assert_array_equal(actual,expected)
    altered=dict(qam.network.params)
    altered['modules_critic']=jax.tree_util.tree_map(lambda x:x+10,altered['modules_critic'])
    other=qam.replace(network=qam.network.replace(params=altered))
    np.testing.assert_array_equal(other.sample_actions(batch['observations'],key),actual)


def test_batch_update_matches_sequential_updates():
    bc,batch=fixture();batched={k:jnp.stack([v,v]) for k,v in batch.items()}
    block,_=bc.batch_update(batched)
    one,_=bc.update(batch);two,_=one.update(batch)
    for x,y in zip(jax.tree_util.tree_leaves(block),jax.tree_util.tree_leaves(two)):
        np.testing.assert_allclose(x,y,atol=1e-6,rtol=1e-5)
