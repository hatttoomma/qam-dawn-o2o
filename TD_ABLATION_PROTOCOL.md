# Critic target entropy ablation: Task1 and Task2

User authorized the proposed online-only ablation of critic Soft TD versus plain
TD. Run task1 and task2 from each task's OWN existing QAM offline 500k checkpoint.
No offline retraining. Seed 0, residual scale .1, 50k primitive online steps,
20k base-only collection warmup, UTD .25, exactly 7,500 updates, batch 1024.
Keep online-only chunk replay, chunk=5, gamma=.99, critic ensemble=10 and minimum
aggregation for both target and actor. Freeze the QAM base throughout.

Four paired formal runs on the new machine: task{1,2} x {soft, hard}. Soft reruns
provide same-machine performance and diagnostic controls. In hard, change ONLY
the backup from R+d*(min(Q_next)-alpha*logp_next) to R+d*min(Q_next).
Actor SAC entropy term and automatic temperature updates remain enabled. This
is a deliberate target-only ablation, not standard SAC and not QAM's full target
(QAM uses mean-.5std rather than min). Do not simultaneously remove actor entropy,
change aggregation, replay, residual scale or optimizer settings.

Use original DawnAgent, its full update and sampling implementation, and the
unchanged run.residual loop. Replace only its module-level backup function in
the hard process before any JAX tracing. Preserve separate processes so compiled
functions cannot mix modes. Record target mode and reject mismatched resumes.
Actor/critic Adam are initialized exactly as in original DAWN (not inherited from
QAM); critic and target PARAMETERS are inherited. Initial full state hashes must
match within task. Actor/critic lr=1e-4, norm clip=50, target tau=.01, alpha init=.01,
automatic target entropy=-25 in unscaled tanh residual coordinates.

Verify 60-step / warmup20 / 10-update smoke Soft TD against the original entry
for each task, requiring exact full final checkpoint and evaluation parity.
Hard smokes must inherit identical initial state and warmup, keep frozen base,
and update actor/critic/alpha. Check analytical backup difference and terminal
mask behavior. Run four concurrent full-model smoke jobs before formal launch.

Diagnostics must not change training RNG or add training data. Reuse the existing
50-episode evaluation rollouts to capture raw gamma-discounted return and future
residual entropy return, excluding current-action entropy as in SAC Q semantics.
Report terminal episodes separately from time-limited truncated episodes; truncated
MC omits the beyond-horizon bootstrap and is not an unbiased continuing-value
reference. At residual-disabled evaluations, Q-vs-MC is a base-policy reference.
Evaluate Q on the exact first issued action chunk of each existing episode.

Capture a fixed 256-transition probe from the first base-only warmup replay using
independent NumPy seed 44000+task, and use fixed JAX seed 55000+task at every probe.
Record both counterfactual targets, actual entropy contribution, TD MSE, base and
residual Q difference, and gradient norm wrt unscaled residual u. Compare the
same fixed batch across modes (allow <=1e-12 float32 simulator roundoff only in
observations; actions/rewards/discounts/base proposals must match exactly).
Warmup replay arrays are checked raw <=1e-10 and training float32 <=1e-12 for
observations, exact for all other fields. Keep raw hashes and diagnostics.

Seven fixed curve points (0/5k/10k/20k/30k/40k/50k, chunk crossings may add <=4
steps), 50 paired episodes each. Final primary sampled-residual evaluation uses
100 episodes; mean-residual 50-episode evaluation is a separate diagnostic.
Do not select best checkpoints or infer multi-seed stability. Preserve all
historical sources/results; freeze new training and diagnostics source hashes
before launch. Export reproducible CSV, PNG/PDF curves and raw evidence.
