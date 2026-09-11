# Residual scales 0.01 and 1.0 on task1 and task2

The user requested extending the scale range to 0.01 and 1 on both tasks.
Run exactly four new formal arms: task1/0.01, task1/1.0, task2/0.01, task2/1.0.
Retain original scale .1 and native references, and existing task2 scales
.05/.2/.3 in the final report. Do not invent unrun task1 intermediate scales.

Every arm uses seed 0 and its OWN task's existing 500k offline QAM checkpoint,
with inherited current and target critics and fresh original online optimizers.
No new offline pretraining, extra seeds or online continuation beyond 50k
primitive steps. Shared base policy is frozen, stochastic, chunk length 5.

Only residual scale changes within a task. Preserve state-only 3x256 ReLU actor,
10 QAM critics (4x512 GELU+LN), min10 actor and target, joint ensemble gradient
clipping 50, SAC entropy TD, batch 1024, UTD .25 per primitive step, lr 1e-4,
tau .01, initial automatic alpha .01 and unscaled chunk target entropy -25.
Keep original online-only replay with cached base actions, 20k primitive-step
base-only warmup without updates, gamma .99 per primitive step, native OGBench
rewards, action clipping and termination/timeout handling. Exactly 7,500 updates.

The wrapper imports unchanged run.py/dawn_agent.py and changes only task selection
and the static DawnAgent.res_scale. Verify task reward/mask hashes against the
completed two-task audit and correct offline checkpoint hashes. Persist requested
and effective scale, checked before loading any resume checkpoint.

Preserve 50-episode sampled-action evaluations at 0/5k/10k/20k/30k/40k/50k
(intermediate completed chunk may be up to 4 primitive steps late), final 100
paired sampled-residual episodes, and the existing 50-episode tanh(mean) residual
diagnostic. Base remains stochastic. No best-checkpoint selection or early stop
based on results. Keep evaluation randomness separate from training.

Before the formal suite, reproduce short scale=.1 training through the new wrapper
against original task1 and task2 entries, requiring identical final checkpoints
and episode records. Then concurrently smoke-test all four requested arms using
the same batch size and full pretrained models. Verify exact initial states,
frozen flows, inherited Q/target, effective scale, finite metrics, 10 updates
over 60 primitive steps with a 20-step smoke warmup, and within-task warmup parity.

Four independent processes may share the existing RTX 4090 with memory fraction
.21 each after concurrent smokes pass. This allocator setting changes no training
hyperparameter. Preserve single-thread OMP/BLAS, EGL, JAX cache and package versions.
All new runs, locks and logs are isolated in runs/scale_extremes/. Historical
training scripts, source manifests, checkpoints and results remain untouched.

Before reporting, validate source hashes, per-task config parity, initial weights,
warmup replay parity against each task's original .1, exact budgets, episode means,
paired seeds/initial states, first50/final100 agreement and finite logs. Export
raw evidence, tables and research PNG/PDF figures; weights remain remote. A single
training seed supports only a sensitivity pilot, not a generally optimal scale.
