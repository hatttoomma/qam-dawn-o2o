# Policy Decorator online pilot (predeclared before training)

One new run, seed 0, `cube-double-play-singletask-task1-v0`, shared 500k QAM offline checkpoint, 50k primitive online steps. Both current and target Q weights are inherited exactly; optimizers start fresh. The previous five runs remain untouched. QAM actor is frozen; official cube-double edit scale remains zero.

Algorithm reference: [official state Diffusion Policy implementation](https://github.com/tongzhoumu/policy_decorator/blob/92ba9ba442587ae5989c286355ace2200a0537fb/online/pi_dec_diffusion_maniskill2.py), commit `92ba9ba442587ae5989c286355ace2200a0537fb`; [paper](https://arxiv.org/abs/2412.13630). This is an OGBench/QAM adaptation, not a reproduction of the ManiSkill benchmark.

## Actual settings

| Setting | Value |
|---|---|
| Residual bound | `a=clip(a_QAM + 0.1*tanh(u), -1,1)` |
| Progressive exploration | `p(t)=min(t/30000,1)`, t = primitive steps at chunk start; independent Bernoulli per 5-action chunk |
| Before learning | Gate off: base only. Gate on: uniform residual in [-1,1], multiplied by 0.1 |
| Learning starts | 8,000 primitive steps; no parameter updates before then |
| After learning starts | Same gate, with tanh-Gaussian residual actor; after 30k every chunk enabled |
| Residual actor | State only (37 dimensions), 3×256 ReLU, 25 outputs (5 actions × 5 steps); log std range [-20,2]; output heads orthogonal gain 0.01 |
| Actor initialization | Same Flax initialization and seed stream as prior DAWN for a shared experimental control |
| Critics | Inherit all 10 QAM critics and their targets; each 4×512 GELU + LayerNorm; input state + clipped composite chunk |
| TD and actor Q aggregation | Minimum over all 10; SAC entropy bonus included in TD |
| Discount and reward | QAM gamma=0.99 per primitive step; discounted 5-step return; no extra reward offset |
| Entropy | Automatic; alpha starts at **1.0** (official `log_sac_alpha=0`), target entropy -25, log density in unscaled residual coordinates |
| Optimizers | Fresh Adam for actor, critic, log alpha; lr=1e-4, betas=(0.9,0.999), eps=1e-8 |
| Gradient clipping | Global norm 50 separately for actor and entire critic ensemble |
| Target update | Polyak tau=0.01 after each gradient update |
| Update ratio/batch | UTD=0.25 per primitive step, batch=1024; actor/alpha/target every critic update |
| Update timing | At complete QAM chunk boundaries; cumulative floor((step-8000)*0.25), final 10,500 updates |
| Replay | User-selected QAM uniform offline+online primitive sequence sampler; 1m offline + up to 50k online; horizon 5; QAM valid-sequence mask on critic MSE |
| Base proposals in replay | Same precomputed offline and actual/lazy online cache implementation as prior QAM-replay arms |
| Environments | One training environment; QAM horizon 5, execute all actions unless episode ends |
| Main evaluation | Previous paired seeds, stochastic residual, 50 episodes at 0/5k/10k/20k/30k/40k/50k, final 100; step 0 is the shared base-only reference |
| Official-style diagnostic | Mean residual, with stochastic frozen QAM, 50 episodes at every grid point (including decorated step 0), final 100; no exploration gate during evaluation |

## Explicit adaptations and interpretation

The official state script uses two newly initialized 3×256 critics, gamma 0.97, online-only replay, 16 environments, and 16 gradient updates after every 64 collected chunk transitions. It counts chunk decisions in `global_step`, not primitive actions. Here, inherited critic architecture, replay, reward/discount, action chunks, step unit and evaluation seeds follow the existing QAM experiment. The 30k exploration horizon is the official PegInsertion example's numeric choice, re-expressed in **primitive steps** for this 50k budget; it is not claimed to be a tuned cube-double value or the same physical horizon as the original script. Learning starts=8k is likewise primitive steps here.

For continuity with previous runs, Q inputs use clipped composite actions, clipping is global over the inherited critic ensemble, actor hidden-layer initialization uses Flax defaults (rather than PyTorch defaults), and updates accrue at complete chunk boundaries. The official script queries Q on the unclipped sum and clips gradients separately per critic. These are reported implementation adaptations, not official PD defaults.

Compared with the preceding DAWN+QAM replay arm, this run changes the progressive gate, pre-learning uniform residual exploration, learning start (20k→8k), and initial alpha (0.01→1). All shared SAC losses, actor initialization, inherited critics and replay code remain identical. It therefore compares a PD-style configuration, **not an isolated ablation of the progressive gate**. No hyperparameter selection will use this run's results.

Training RNG streams remain separate from evaluation. A separate gate RNG is checkpointed. Existing five-run source manifests, inherited weight hashes, frozen actor hashes, update counts, replay statistics, raw evaluation records and result hashes are checked before reporting. One seed supports a workflow/trend pilot only.
