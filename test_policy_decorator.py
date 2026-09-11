import run as common
import jax
import jax.numpy as jnp
import numpy as np
from policy_decorator_agent import PolicyDecoratorAgent, collect_action, residual_probability
from dawn_replay_agent import DawnQamReplayAgent
from dawn_agent import soft_backup


def test_progressive_gating_and_uniform_initial_exploration():
    class FakeAgent:
        res_scale=.1
        def sample(self, obs, base, key):
            return jnp.clip(base+.07,-1,1), {}
    agent=FakeAgent();base=jnp.zeros(25);key=jax.random.PRNGKey(2)
    assert [residual_probability(t,30000) for t in [0,15000,30000,50000]]==[0,.5,1,1]
    action,on,random,p=collect_action(agent,None,base,key,.2,6000,8000,30000)
    assert not on and random and p==.2
    np.testing.assert_array_equal(action,base)
    action,on,random,p=collect_action(agent,None,base,key,.1,6000,8000,30000)
    expected=.1*jax.random.uniform(key,(25,),minval=-1.,maxval=1.)
    np.testing.assert_array_equal(action,expected)
    assert on and random and np.max(np.abs(action))<=.1
    action,on,random,p=collect_action(agent,None,base,key,.1,15000,8000,30000)
    assert on and not random and p==.5
    np.testing.assert_allclose(action,.07)  # gate does not halve the amplitude
    assert np.mean(np.random.RandomState(4).rand(100000)<p)>.495


def test_inheritance_alpha_soft_td_and_boundary_mask():
    obs=jax.random.normal(jax.random.PRNGKey(8),(8,37))
    qam=common.qam_template(obs[0],jnp.zeros(5),0)
    pd=PolicyDecoratorAgent.create(qam,obs[0],0)
    dawn=DawnQamReplayAgent.create(qam,obs[0],0,True)
    assert common.tree_hash(pd.critic.params)==common.tree_hash(qam.network.params['modules_critic'])
    assert common.tree_hash(pd.target_params)==common.tree_hash(qam.network.params['modules_target_critic'])
    assert common.tree_hash(pd.actor.params)==common.tree_hash(dawn.actor.params)
    assert float(jnp.exp(pd.log_alpha))==1 and int(pd.updates)==0
    qs=jnp.arange(20,dtype=jnp.float32).reshape(10,2)
    np.testing.assert_allclose(soft_backup(jnp.array([-1.,-2.]),jnp.array([0.,.9]),qs,1.,jnp.array([3.,4.])),[-1.,-4.7])
    batch=dict(observations=obs,actions=jnp.zeros((8,25)),next_observations=obs+.02,
               rewards=-jnp.ones(8),discounts=jnp.ones(8)*.99**5,
               base_actions=jnp.zeros((8,25)),next_base_actions=jnp.zeros((8,25)),valid=jnp.zeros(8))
    updated,info=pd.update(batch);common.floats(info)
    assert float(info['critic_loss'])==0 and float(info['critic_grad_norm'])==0
    assert common.tree_hash(updated.critic.params)==common.tree_hash(pd.critic.params)
    assert common.tree_hash(updated.actor.params)!=common.tree_hash(pd.actor.params)
    updated,info=pd.update(dict(batch,valid=jnp.ones(8)));common.floats(info)
    assert common.tree_hash(updated.critic.params)!=common.tree_hash(pd.critic.params)
    assert int(updated.updates)==1
    action,_=updated.sample(obs,jnp.ones((8,25))*.99,jax.random.PRNGKey(3))
    assert np.max(np.abs(action))<=1
