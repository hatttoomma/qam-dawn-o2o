# Task2 residual-scale sweep

User confirmed new scales 0.05, 0.2 and 0.3 in this session. Reuse
the completed task2 scale=0.1 DAWN run and native QAM reference. All runs use
seed 0 and the identical existing task2 offline 500k checkpoint; no offline
training, additional seeds, task1 runs or longer online budgets are added.

Only residual scale changes. Preserve the original frozen stochastic QAM,
state-only 3x256 ReLU residual actor, inherited current/target 10-critic QAM
backbone, minimum over all 10 for actor and target, joint ensemble norm
clipping 50, SAC entropy TD, batch 1024, lr 1e-4, tau .01, alpha .01 with
automatic tuning and unscaled residual target entropy -25. No changes from
the preceding hyperparameter audit are implemented in this sweep.

Keep task2 OGBench rewards and termination semantics, chunk length 5,
primitive discount .99, 20k primitive-step base-only warmup without updates,
UTD .25 per primitive step, cached online-only replay and 50k primitive
online steps, yielding exactly 7,500 updates. Initial actor/Q/target/optimizer
states and all RNG seeds are identical across scales. Training diverges only
after enabling residual actions/updates; warmup replay must match scale .1.

Reuse unchanged run.py, dawn_agent.py and run_task2.py. The new wrapper changes
only DawnAgent.res_scale after normal agent creation. Persist and verify scale
on every restart because the Flax field is static. Record effective agent
settings and inherited weight hashes, in addition to the original configs.

Keep the original seven 50-episode learning-curve evaluations, final 100
paired episodes with sampled residuals, and the original 50-episode tanh(mean)
residual diagnostic. Base remains stochastic for both modes. No best-checkpoint
selection. Compare all four scales at the same budget and evaluation size.

Before formal runs, reproduce a short original scale=.1 run with the wrapper
and require identical final checkpoint/initial state/evaluation records. Run
short integration checks for all three requested scales and check inherited Q,
frozen flow, requested updates, effective scale and shared base-only prefix.

Use one RTX 4090. Independent processes may run concurrently after concurrent
smokes pass, each with XLA memory fraction .28 (allocator setting only). Keep
OMP/BLAS thread counts at 1, EGL and the existing JAX cache. This changes no
training hyperparameter. Run directories and process locks are isolated under
runs/task2/scale_sweep/. Historical runs and source files remain unchanged.

Validate raw episodes, paired seeds/initial states, budgets, hashes, warmup
identity, finite metrics and first50/final100 agreement before reporting.
Export tables, research PNG/PDF figures and a portable evidence bundle, keeping
weights remote. A single training seed supports a sensitivity pilot, not a
general optimum for residual scale.
