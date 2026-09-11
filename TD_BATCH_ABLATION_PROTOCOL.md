# Plain-TD DAWN batch-size comparison

Run Task1 and Task2, seed 0, from their verified QAM offline 500,000-update
checkpoints. For each task run batch 256 and batch 1024 on the same current
machine. The latter is a fresh control; historical plain-TD results are context.
The current GPU driver is 580.76.05, versus 550.120 in the previous TD experiment.

Use the unchanged `run_td_ablation.py --td-target hard` entry point. The only
learning configuration difference within each pair is `--dawn-batch 256` versus
`--dawn-batch 1024`; output paths necessarily differ. Store all new runs in
`runs/td_ablation/batch_comparison`, preserving earlier runs and source files.

Both arms retain inherited critic and target weights, fresh DAWN Adam states,
frozen QAM base, residual scale 0.1, 10 critics with minimum for actor and target,
ordinary TD target R + discount * min Q, actor SAC entropy and automatic alpha,
alpha initial 0.01, residual target entropy -25, state-only 3x256 actor,
gamma 0.99, target tau 0.01, learning rates 1e-4, global gradient norm clip 50,
online-only replay of issued action chunks, and action horizon 5.

The total budget is 50,000 primitive online steps including the original 20,000
base-only warmup. UTD remains 0.25 per primitive step, so each run has exactly
7,500 updates. Do not multiply updates to offset a smaller batch: the experiment
changes batch size at fixed interaction and update budgets. Thus total sampled
training rows are 1,920,000 versus 7,680,000, with replacement from replay.
Training is restarted from offline, not continued from a tuned online checkpoint.

Before formal training, run all four configurations for 60 primitive steps,
20 warmup steps and 10 updates, with one evaluation episode. Verify full initial
agent states, inherited weights, identical pre-learning rollout, target math,
finite updates and output batch configuration. Original training/diagnostic
sources must match the 46-file previous manifest. Freeze this protocol and the
new launcher before launching formal training. No changes to algorithm code.

Retain the original seven evaluation points (50 episodes), final 100 paired
evaluation episodes using sampled residual, and the 50-episode mean-residual
diagnostic. Evaluation interactions do not enter replay or training budgets.
Retain the existing fixed 256-row warmup diagnostic probe in both arms; this
read-only probe size is independent of the training batch size. Check full warmup
arrays with the previously declared tolerances: raw observations atol 1e-10,
float32 observations atol 1e-12, all other fields exact. Report raw differences.

Report success and episode return against primitive online steps, final paired
outcomes, actual updates and total sampled rows, plus inherited/frozen-state and
finite-value checks. Compare current batch 1024 against historical batch 1024
and disclose any differences. Never select checkpoints by performance. One
training seed supports a trend pilot, not an across-seed superiority claim.
