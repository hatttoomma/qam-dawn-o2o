"""Fixed128 offline sequences and128 online executed chunks per update."""
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
import run as common

def convert_sequence(seq,discount):
 valid=np.asarray(seq['valid']);length=valid.sum(axis=-1).astype(np.int64)
 assert np.all(length>=1)
 actions=np.where(valid[...,None]>0,seq['actions'],0).reshape(len(length),-1)
 # Terminal/truncated sequences stop at their actual final valid transition.
 return dict(observations=np.asarray(seq['observations'],np.float32),actions=np.asarray(actions,np.float32),
  rewards=np.asarray(seq['rewards'][:,-1],np.float32),
  discounts=np.asarray(discount**length*seq['masks'][:,-1],np.float32),
  next_observations=np.asarray(seq['next_observations'][:,-1],np.float32)),length

class BalancedReplay:
 def __init__(self,ds,qam,args):
  self.ds,self.qam,self.args=ds,qam,args;self.key=jax.random.PRNGKey(41000+args.seed)
 def sample(self,online,batch_size,update):
  assert batch_size==256 and online
  idx=np.random.randint(len(online),size=128)
  on=common.stack_replay(online,idx)
  seq=self.ds.sample_sequence(128,common.HORIZON,common.GAMMA)
  off,length=convert_sequence(seq,common.GAMMA)
  obs=np.concatenate([off['observations'],off['next_observations']],axis=0)
  bases=np.asarray(self.qam.sample_actions(jnp.asarray(obs),jax.random.fold_in(self.key,update)),np.float32)
  assert bases.shape==(256,25)
  off['base_actions'],off['next_base_actions']=bases[:128],bases[128:]
  batch={k:np.concatenate([off[k],on[k]],axis=0).astype(np.float32) for k in on}
  assert all(v.shape[0]==256 and np.isfinite(v).all() for v in batch.values())
  if update==0 or (update+1)%250==0 or update+1==(self.args.replay_switch-self.args.warmup)//4:
   common.atomic_json(Path(self.args.out)/'mixed_replay_audit.json',dict(status='passed',update=update+1,
    offline_per_batch=128,online_per_batch=128,offline_draws=128*(update+1),online_draws=128*(update+1),
    offline_size=self.ds.size,online_chunks=len(online),offline_critic_actions='dataset_sequences_zero_pad_unexecuted_suffix',
    bootstrap='gamma_to_actual_length_times_mask',partial_sequence_count=int((length<common.HORIZON).sum()),
    offline_base_features='frozen_QAM_sampled_per_update',offline_buffer_excludes_validation=True))
  return batch

def test_sequence_conversion():
 # One terminal, one time-limit truncation; invalid suffix must not enter actions.
 s=dict(valid=np.array([[1,1,0,0,0],[1,1,1,0,0]]),actions=np.ones((2,5,5))*7,
  observations=np.zeros((2,37)),rewards=np.array([[-1,-1.99,-1.99,-1.99,-1.99],[-1,-1.99,-2.9701,-2.9701,-2.9701]]),
  masks=np.array([[1,0,0,0,0],[1,1,1,1,1]]),next_observations=np.zeros((2,5,37)))
 b,length=convert_sequence(s,.99)
 np.testing.assert_array_equal(length,[2,3]);np.testing.assert_allclose(b['discounts'],[0,.99**3],rtol=1e-6)
 assert np.all(b['actions'].reshape(2,5,5)[0,2:]==0) and np.all(b['actions'].reshape(2,5,5)[1,3:]==0)
 np.testing.assert_allclose(b['rewards'],[-1.99,-2.9701],rtol=1e-6)
