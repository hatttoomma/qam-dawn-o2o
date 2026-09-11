# Plain-TD DAWN: UTD 1 with batch 256

Run Task1 and Task2, seed 0, on the same machine as the just-completed batch-size
experiment (RTX 4090, driver 580.76.05, unchanged Python environment). Compare
each new run against `runs/td_ablation/batch_comparison/task{1,2}_b256`, whose
UTD was 0.25. Reuse each task's verified QAM offline 500,000-update checkpoint.
New runs begin from offline, not from the finished online models.

The only changed learning setting is UTD: 0.25 -> 1 gradient update per primitive
online step after warmup. Batch stays 256. Keep 20,000 base-only warmup steps,
counted within the total 50,000 online steps. At chunk boundaries accumulate
updates to floor((step - 20000) * utd), preserving the original loop timing.
Final updates are exactly 30,000 versus 7,500 for the previous controls. Total
sampled training rows become 7,680,000 versus 1,920,000 (with replacement).
QAM native still has a different 5k warmup and 45,001 updates; UTD alone is aligned.

Retain ordinary TD R + discount * min Q (no target entropy), actor SAC entropy
and automatic alpha, frozen QAM base, inherited critic/target parameters and
fresh DAWN Adam states, online-only replay of issued chunks, residual scale .1,
10 critics and minimum actor/target aggregation, state-only 3x256 actor,
gamma .99, horizon 5, lr 1e-4 for actor/critic/alpha, tau .01, gradient clip 50,
initial alpha .01 and target residual entropy -25. Actor, critic, alpha and
target continue to update once per gradient step. Do not rescale lr or tau.

Preserve all prior training sources and outputs. The new entry point checks
the exact original residual-loop source and replaces only its one UTD literal
with args.utd in memory. Save before/after source hashes and the exact one-line
diff in each run. The entry point also exposes UTD in config and completion
checks. Verify UTD=.25 short-run full checkpoint parity against the original
entry on both tasks before any formal run; UTD=1 short runs must have 40 updates
at 60 steps / 20 warmup, identical full initial state and warmup, and finite
training state. Short tests use the original full model, with one eval episode.

New results: `runs/utd_ablation/task{1,2}_utd1`. Formal evaluation schedule stays
at 0,5k,10k,20k,30k,40k,50k (50 episodes each); final sampled-residual evaluation
uses 100 paired reset seeds, and mean-residual diagnostic uses 50 episodes.
Evaluation interactions do not count toward training or enter replay. Keep
existing read-only MC and fixed 256-row warmup diagnostics unchanged.

Validate full initial states and warmup versus the existing UTD=.25 controls.
Raw observation tolerance remains 1e-10 and float32 tolerance 1e-12; all other
warmup fields must match exactly. Preserve any repeated endpoint evaluation
differences instead of selecting favorable repeats. The prior Task2 control
had one unstable repeated-evaluation episode; keep that observation visible.

Report raw final success/return and measured checkpoint curves, actual update
counts and sampled rows, with per-task paired final outcomes. One training seed
and the observed numerical/trajectory reproducibility variation limit conclusions
to this run. Do not add seeds, extend environment budget, or tune other settings.
