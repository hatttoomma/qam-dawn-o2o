# New-data ablation: scientific figure contract

Question: under matched native QAM training, how much improvement is attributable
to making newly collected transitions available for training? The final takeaway
must follow completed results; do not predetermine improvement or failure.

Use the established standalone research Matplotlib PNG/PDF delivery. A 2x2 line
figure has rows task1/task2 and columns success/mean return. Primary curves are
new-machine paired all-data versus offline-only training. Same seed 0, task-specific
offline500k checkpoint and 45,001 updates. Use the exact established seven 50-episode
points at 0/5k/10k/20k/30k/40k/50k. Increasing evaluation resolution would change
this protocol, so keep the discrete markers and connect measured points without
smoothing. No seed uncertainty band from a single training seed.

Primary x-axis: primitive collection steps. State visibly that the offline-only
condition discards all new data from training. A matching second figure can use
additional gradient updates; map step s to max(0,s-5000+1), dropping the duplicate
zero-update point if necessary. Report the exact 45,001 final update budget.

Use one blue root, solid filled-circle all-data and dashed open-square offline-only.
Neutral horizontal references show initial offline policy; historical native is
context in the table only, never substituted for a new-machine paired reference.
Use white background, quiet y-grid, success from zero, unnormalized return, readable
facet labels, shared legend and visible sample counts. Footprint about 13x9 inches.

Final table uses 100 paired episodes; do not mix with the curve's 50 episodes.
Retain task/mode/seed/step/update, success/return/episode length and sampling audit
fields, initial/checkpoint/config hashes, all per-episode records and paired outcome
counts. Validate source and data eligibility before plots. Inspect final PNGs for
legibility, clipping and honest labels. No publication to an external website.
