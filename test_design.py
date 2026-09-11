import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor/qam'))

import jax
import jax.numpy as jnp
import numpy as np
from dawn_agent import DawnAgent, soft_backup
from run import qam_template, tree_hash, floats


def test_soft_target_uses_all_ten_minimum_and_terminal_mask():
    q=jnp.ones((10,2))*100.
    q=q.at[9].set(jnp.array([-100.,-2.]))
    actual=soft_backup(jnp.array([1.,2.]),jnp.array([0.,.9]),q,.2,jnp.array([10.,.5]))
    np.testing.assert_allclose(actual,np.array([1.,.11]),atol=1e-6)


def test_initialization_action_bounds_and_frozen_flow():
    qam=qam_template(jnp.zeros(39),jnp.zeros(5),0)
    warm=DawnAgent.create(qam,jnp.zeros(39),0,True)
    cold=DawnAgent.create(qam,jnp.zeros(39),0,False)
    original=tree_hash(qam.network.params)
    assert tree_hash(warm.critic.params)==tree_hash(qam.network.params['modules_critic'])
    assert tree_hash(warm.target_params)==tree_hash(qam.network.params['modules_target_critic'])
    assert tree_hash(cold.critic.params)!=tree_hash(warm.critic.params)
    assert tree_hash(cold.target_params)==tree_hash(cold.critic.params)
    assert tree_hash(warm.actor)==tree_hash(cold.actor)
    assert tree_hash(warm.critic.opt_state)==tree_hash(cold.critic.opt_state)
    obs=jax.random.normal(jax.random.PRNGKey(10),(4,39))
    base=jnp.ones((4,25))*.99
    action,info=warm.sample(obs,base,jax.random.PRNGKey(20))
    assert action.shape==(4,25)
    assert np.max(np.abs(np.asarray(action)))<=1.
    assert np.max(np.abs(np.asarray(action-base)))<=.100001
    qs=warm.critic.apply_fn({'params':warm.critic.params},obs,base)
    assert qs.shape==(10,4)
    batch=dict(observations=obs,actions=action,next_observations=obs+.01,
               rewards=jnp.ones(4)*-5.,discounts=jnp.array([0.,.99**5,.99**5,.99**5]),
               base_actions=base,next_base_actions=base)
    trained,info=warm.update(batch)
    floats(info)
    assert int(trained.updates)==1
    assert tree_hash(trained.critic.params)!=tree_hash(warm.critic.params)
    assert tree_hash(trained.actor.params)!=tree_hash(warm.actor.params)
    assert tree_hash(qam.network.params)==original


def test_original_sequence_mask_does_not_cross_episodes():
    from utils.datasets import Dataset
    ds=Dataset.create(observations=np.arange(8,dtype=np.float32)[:,None],actions=np.ones((8,1),np.float32),
                      next_observations=np.arange(1,9,dtype=np.float32)[:,None],rewards=np.ones(8,np.float32),
                      masks=np.array([1,0,1,1,1,1,1,1],np.float32),
                      terminals=np.array([0,1,0,0,0,0,0,1],np.float32))
    np.random.seed(0)
    b=ds.sample_sequence(100,5,.99)
    early=b['observations'][:,0]==0
    assert np.any(early)
    assert np.all(b['valid'][early,-1]==0)
    np.testing.assert_allclose(b['rewards'][early,-1],1.99)
    assert np.all(b['masks'][early,-1]==0)
