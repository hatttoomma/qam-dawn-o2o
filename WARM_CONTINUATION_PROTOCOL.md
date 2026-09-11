# DAWN pretrained-Q online-only continuation

Continue seed 0 on cube-double-play-singletask-task1-v0 from the existing warm
50,000-environment-step / 7,500-update checkpoint. The offline QAM checkpoint
still has 500,000 updates. Do not modify historical runs or training sources.

Restore the complete DAWN state (actor, critic, target, temperature, all Adam
states and training RNG), online-only replay, collection RNGs, NumPy sampling
RNG and the current environment episode by replaying its saved action trace.
Verify that latest.pkl and final.pkl contain identical agent states and that
the latter matches the historical SHA256. The frozen QAM proposal is unchanged.

All learning settings remain unchanged: 20,000 primitive-step warmup counted
from the original start, UTD 0.25, batch 1024, residual scale 0.1, state-only
3x256 ReLU actor, 10 critics with minimum for actor and target aggregation,
SAC entropy in the TD target, automatic temperature (initial value was 0.01),
target entropy -25, Adam 1e-4, gradient clipping 50, target tau 0.01, gamma
0.99, full action chunks of length 5, and growing uniform online-only replay.
The warmup and optimizer schedules are not restarted.

The old budget stopped at episode 281, trace length 26: only the first action
of the last issued chunk was executed. That artificial partial chunk was
excluded from replay by the old runner; its unexecuted suffix was not saved.
Resume at the saved environment state with a freshly sampled chunk. This is
a documented one-time boundary difference from an uninterrupted longer run.
Future checkpoints are taken at chunk boundaries, including any outstanding
update debt so restoration does not change the update schedule.

Predeclared cumulative update evaluations: 7,500 (restore check), 15,000,
22,500, 30,000, 37,500, **45,001**, 52,500, **60,000**. Evaluate the exact
agent update count, even when it is inside a group of updates after a chunk.
Use the same 50 paired evaluation seeds as the historical curves, stochastic
base and residual actions. At 45,001 and 60,000, additionally evaluate the
same 100 paired seeds and 100 episodes with mean residual actions (base still
stochastic). Preserve all milestone weights and a complete resume checkpoint.
Do not choose or stop training based on evaluation performance.

Expected environment steps are approximately 200,004 at 45,001 updates and
260,000 at 60,000, rounded up to a naturally completed chunk (or episode).
This is an update-count comparison, not an environment-step or compute matched
comparison: native QAM used 50,000 online steps, 45,001 updates and batch 256.
Both tables and plots must show actual environment steps and update counts.
Use one seed only; do not interpret episode confidence intervals as seed
variability or claim a general causal attribution from this experiment.
