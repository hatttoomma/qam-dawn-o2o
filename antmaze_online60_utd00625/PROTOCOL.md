# AntMaze continuation UTD ablation

Tasks1,2,3,5; seed0. Task4 is excluded. Reuse each original balanced50k checkpoint (40k base-only warmup plus10k balanced128offline/128online training, UTD0.25,2500 updates).

Resume the complete50k state: actor, critic, target, temperature, all optimizer states, online replay, rollout RNGs, NumPy RNG, current environment and episode trace. Do not resume the previous60k result. Do not restart warmup or reset alpha.

Continue50k to60k with uniform online-only replay,256 online /0 offline per batch. Retain all50k prior online transitions. Change continuation UTD0.25 to0.0625: cumulative update target2500+floor((env_step-50000)/16). This gives625 additional updates and3125 total updates. The parent50k training remains unchanged.

All other settings match the previous online-only60k continuation: naive TD, state+base action, residual scale0.1, frozen stochastic QAM base, inherited critic and existing optimizer state, unchanged actor entropy and learned temperature, min aggregation,10 critics, learning rates and architectures.

Only60k is newly evaluated:100 paired episodes each for mean and sampled residual. Copy existing50k evaluation records as reference; no55k evaluation/checkpoint. Evaluation isolates training RNG; omission of55k evaluation does not change the reference algorithm. The one-time implementation smoke resumes task1 for32 steps (2 updates), with2-episode restoration checks and a final50032 evaluation; it is isolated from production and excluded from performance summaries.

Preflight verifies all4 complete source checkpoints and frozen offline inputs. The isolated legacy/run.py differs from immutable original code in exactly one AST assignment (the cumulative update target); all actor/critic equations and collection code are unchanged. Compare every actual hyperparameter and every resumed state hash with the UTD0.25 continuation. Runtime scheduling checks enforce cumulative update counts, batch composition, frozen base and final checkpoint hashes.

Report each task at60k against its50k parent and UTD0.25/60k reference. Mean residual is primary; sampled residual secondary. Use the same100 paired reset episodes, and compute any macro mean over tasks1,2,3,5 consistently. Single training seed; do not treat episodes as independent training seeds.
