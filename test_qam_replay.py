from pathlib import Path
import run as common
import jax
import jax.numpy as jnp
import numpy as np
from dawn_agent import DawnAgent
from dawn_replay_agent import DawnQamReplayAgent
from qam_replay import QamReplay, original_sequences_with_indices, indexed_proposals


def dataset(n=16):
    return common.Dataset.create(observations=np.arange(n*37, dtype=np.float32).reshape(n,37),
        actions=np.arange(n*5,dtype=np.float32).reshape(n,5)/100,
        next_observations=np.arange(n*37,dtype=np.float32).reshape(n,37)+1,
        rewards=-np.ones(n,np.float32),terminals=np.isin(np.arange(n),[2,7,n-1]).astype(np.float32),
        masks=(np.arange(n)!=2).astype(np.float32))


def test_sampler_and_rng_exactly_match_original_qam():
    ds=dataset()
    np.random.seed(11)
    expected=ds.sample_sequence(512,5,.99)
    after=np.random.get_state()
    np.random.seed(11)
    actual,indices,last=original_sequences_with_indices(ds,512)
    after_new=np.random.get_state()
    for k in expected:np.testing.assert_array_equal(actual[k],expected[k])
    np.testing.assert_array_equal(after[1],after_new[1]);assert after[2:]==after_new[2:]
    np.testing.assert_array_equal(actual['observations'],ds['observations'][indices])
    np.testing.assert_array_equal(actual['next_observations'][:,-1],ds['next_observations'][last])
    assert np.any(actual['valid'][:,-1]==0)


def test_mixed_buffer_keeps_uniform_transition_sampling_and_cache_alignment(tmp_path):
    ds=dataset()
    for kind,name in enumerate(['base.npy','next_base.npy']):
        np.save(tmp_path/name,np.repeat((np.arange(ds.size,dtype=np.float32)/100+kind/2)[:,None],25,axis=1))
    replay=QamReplay(ds,10,None,tmp_path,0)
    for i in range(8):
        row={k:v[-1].copy() for k,v in ds.items()}
        row['terminals']=np.float32(0);row['masks']=np.float32(1)
        replay.add(row)
        replay.online_base[0][i]=.2+i/100
        replay.online_base[1][i]=.7+i/100
    np.random.seed(3)
    batch=replay.sample(10000)
    assert batch['actions'].shape==(10000,25)
    assert np.isfinite(batch['base_actions']).all() and np.isfinite(batch['next_base_actions']).all()
    # 24 rows -> 20 possible 5-step starts, four starts are online.
    assert abs(replay.stats()['sampled_online_fraction']-.2)<.015
    assert np.any(batch['valid']==0)
    assert np.all(batch['discounts']<=.99**5+1e-7)


def test_valid_batches_keep_dawn_update_and_invalid_rows_do_not_train_critic():
    obs=jax.random.normal(jax.random.PRNGKey(21),(8,37))
    qam=common.qam_template(obs[0],jnp.zeros(5),0)
    old=DawnAgent.create(qam,obs[0],0,True)
    new=DawnQamReplayAgent.create(qam,obs[0],0,True)
    assert common.tree_hash(old.critic.params)==common.tree_hash(new.critic.params)
    assert common.tree_hash(old.actor.params)==common.tree_hash(new.actor.params)
    batch=dict(observations=obs,actions=jnp.zeros((8,25)),next_observations=obs+.02,
               rewards=jnp.arange(8,dtype=jnp.float32)-8,discounts=jnp.ones(8)*.99**5,
               base_actions=jnp.zeros((8,25)),next_base_actions=jnp.zeros((8,25)),valid=jnp.ones(8))
    a,ai=old.update(batch);b,bi=new.update(batch)
    common.floats(ai);common.floats(bi)
    for left,right in [(a.critic.params,b.critic.params),(a.actor.params,b.actor.params),
                       (a.target_params,b.target_params),(a.log_alpha,b.log_alpha)]:
        for x,y in zip(jax.tree_util.tree_leaves(left),jax.tree_util.tree_leaves(right)):
            np.testing.assert_allclose(x,y,atol=2e-6,rtol=2e-5)
    zero,info=new.update(dict(batch,valid=jnp.zeros(8)))
    assert float(info['critic_grad_norm'])==0 and float(info['critic_loss'])==0
    assert common.tree_hash(zero.critic.params)==common.tree_hash(new.critic.params)
    assert common.tree_hash(zero.actor.params)!=common.tree_hash(new.actor.params)


def test_cache_keys_are_per_row_and_independent_of_row_order():
    qam=common.qam_template(jnp.zeros(37),jnp.zeros(5),0)
    obs=jax.random.normal(jax.random.PRNGKey(3),(4,37))
    ids=jnp.array([10,20,30,40],jnp.uint32)
    values=indexed_proposals(qam,obs,ids,0,0)
    permutation=np.array([3,0,2,1])
    other=indexed_proposals(qam,obs[permutation],ids[permutation],0,0)
    np.testing.assert_allclose(values[permutation],other,atol=1e-5,rtol=1e-5)
    assert np.max(np.abs(np.asarray(values)))<=1
