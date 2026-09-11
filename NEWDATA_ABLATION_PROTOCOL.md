# QAM: ablate newly collected online training data

User request: test task1 and task2 and ensure the treatment is newly collected
data. Seed 0; reuse each task's OWN existing offline 500k checkpoint. Do not
retrain offline, reset any learned network/optimizer, add seeds or extend budgets.

Four paired formal runs on the new machine: task{1,2} x {all, offline}.
`all` reruns native QAM with uniform offline+online replay. `offline` runs the
same original native loop, collects and records transitions in a shadow buffer,
but samples every loss batch solely from the immutable original offline data.
The intervention is training eligibility of new transitions. No new observations,
actions, rewards or successors enter ANY training loss in the offline condition.
The shadow collection preserves rollout, action-RNG and evaluation call schedules;
it is not data-efficient deployment and must not be reported as zero env steps.
After policies diverge their collected trajectories may diverge; this is an
effect of treatment, not a second algorithmic change. Evaluation uses paired
fixed reset/action seeds and never provides training transitions.

Both conditions: 50,000 primitive collection steps (chunk 5), start updates at
step 5,000, exactly 45,001 updates, batch 256, UTD 1/primitive step. Same full
offline agent, Adam moments/count, agent RNG, current/target Q, current/target
flows. Preserve original losses: flow-matching BC prior + adjoint-matching actor
+ Q learning, lr 3e-4, 10 critics, actor mean Q, target mean-.5 std, target tau
.005, global network gradient clipping 1, inverse temperature 1, edit_scale=0,
fql_alpha=0, 10 flow steps, gamma .99, native rewards/masks/termination handling.
No DAWN components or entropy target switch. Initial NumPy training RNG seed
31000 and all QAM source/sequence sampler implementations remain unchanged.

The adapter calls run.native unchanged. Its replay facade only selects the live
union or original Dataset for sample_sequence. It observes the ORIGINAL sampler's
randint draws without extra random draws, audits every sampled index and entire
5-step window, stores a resumable chained index hash, and requires zero windows
touching new data in the offline condition. Offline array hashes must remain
unchanged. Snapshot and compare the first 5k collected transitions, before any
update (floating state comparisons may allow simulator roundoff, not action,
reward or mask changes). Persist complete initial agent and optimizer hashes.

Before formal training run full-checkpoint short integration checks for each task:
60 collection steps / start=10 / 51 updates / evaluation 1 episode. Require the
adapter with `all` to reproduce the ORIGINAL native loop final checkpoint exactly.
Require the adapter with `offline` to reproduce 51 pure offline gradient updates
with ZERO training environment interactions exactly. Compare initial/final
evaluations as well. Thus rollout and audit hooks are verified not to influence
the offline learner. Run all four adapters concurrently in smoke before launch;
use four processes at .21 XLA memory fraction on the existing RTX 4090 if it fits.
Administrative allocator differences change no optimizer/loss hyperparameters.

Record 50-episode evaluations at 0/5k/10k/20k/30k/40k/50k; final 100 episodes.
Report both update and collection-step counts, success, return, per-episode
paired outcomes, sampling provenance, source/checkpoint/data hashes. Compare
new-machine paired runs as PRIMARY; previous native results are a reproducibility
reference and never silently substituted or selected for higher score.
No early stopping or best-checkpoint selection. Preserve historical files.
