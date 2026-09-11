# Task2: QAM native versus DAWN pretrained-Q, seed 0

User confirmed two arms only. Change the task to
cube-double-play-singletask-task2-v0; keep the original 500,000 offline updates
and 50,000 primitive online environment steps. Do not extend online training.

Train one new QAM checkpoint from scratch with seed 0 and the official OGBench
task2 reward/mask relabeling of the same cube-double-play-v0 dataset. Do not
load task1 policy, critic or optimizer weights for this pretraining. Use the
last fixed 500k checkpoint, shared bit-for-bit by both online arms. The
underlying observations/actions remain identical across tasks; audit the
different goals and reward labels before training.

All training code and settings are reused from PROTOCOL.md through a small
environment-selection wrapper. Historical run.py, dawn_agent.py, vendor code
and run directories stay unchanged. Task2 data, goal and source manifests
are separate under runs/task2/.

Common: state observations, seed 0, batch 256 for offline QAM, edit_scale=0,
inv_temp=1, action chunk length 5 fully executed, discount 0.99, flow Euler
steps 10, one proposal, and the original reward/termination/timeout semantics.
Offline QAM uses the original losses and optimizer (lr 3e-4, norm clipping 1,
target tau .005), exactly as in task1.

QAM native: preserve the entire new offline agent, including optimizer and
target networks; uniform replay over all offline plus collected online data;
batch 256, UTD 1, learning starts at primitive step 5000 (45,001 online updates).

DAWN: inherit the new task2 current and target critic; freeze the QAM flow;
fresh residual actor and online optimizers; state-only 3x256 ReLU residual
actor, tanh-Gaussian residual scale .1, 10 QAM-backbone critics (4x512 with
LayerNorm), minimum over all 10 critics for both actor and target; SAC entropy
included in TD; lr 1e-4, norm clipping 50, tau .01, batch 1024, UTD .25;
automatic alpha initialized .01 and target entropy -25. Uniform growing
online-only replay, including the 20k base-only warmup, which remains inside
the 50k environment-step budget. No updates during warmup; 7,500 online updates.

Keep original evaluation settings: 50 paired episodes at online steps 0,
5k, 10k, 20k, 30k, 40k, 50k (intermediate DAWN evaluations may be delayed up
to four steps to finish a chunk); 100 paired episodes at the final checkpoint;
50-episode mean-residual diagnostic for DAWN only, with the base still sampled.
Offline evaluate at 0, 100k, 250k and 500k. No checkpoint selection by score.

Before formal runs, verify task2 environment/reward selection and execute
a 32-update offline / 50-step online smoke test for both methods. Verify
that warm inherits task2 Q/target, native and warm share their initial
evaluation, frozen flows remain unchanged, and the requested updates occur.
Reuse the previously checked, unchanged SAC/QAM core. Run the two online
arms sequentially on the available single RTX 4090, native first, preserving
the original .65 JAX memory allocation. No extra arms or seeds.

Report both task2 methods together and include the earlier task1 50k results
as clearly separated context. Do not treat task1's longer DAWN runs as the
task2 comparator. Validate episode aggregates, paired seeds, final100/first50
agreement, frozen-flow and checkpoint hashes, exact training budgets, and
finite training metrics. One training seed supports only a pilot comparison.
