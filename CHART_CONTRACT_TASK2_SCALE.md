# Task2 residual-scale figures

Question: how does scale 0.05 / 0.1 / 0.2 / 0.3 affect DAWN online performance
at the same task2 checkpoint, seed, data protocol and interaction budget?
The takeaway must be derived from completed results, with no preselected winner.

Use research figures exported with Matplotlib to standalone PNG/PDF, continuing
the existing experimental report format. Figure 1 is two panels of measured
success and return versus primitive online steps (0--50k). Four DAWN scales
plus the existing native QAM reference; no smoothing or uncertainty bands.
Figure 2 compares the four final sampled-residual 100-episode success rates
and returns against residual scale, using measured dots without extrapolation.

The seven curve points are the fixed historical protocol. No denser evaluations
exist in the reference run; adding points would change evaluation cost and the
requested same setting. Keep discrete markers visible, disclose 50 episodes per
curve point, and use the separate four-category endpoint figure for exact scale
sensitivity. Curves and endpoint table must not mix 50 and 100 episodes.

Palette: single blue root with explicit tones for ordered scales, and dark grey
for native; distinguish four scales with circle/square/triangle/diamond markers,
line styles and open fills. Use white background, dark labels, restrained grid,
success axes 0--100%, return axes clearly labeled, and a 20k warmup boundary.
Keep a compact legend and sufficient margins; figure footprint about 13x6 inches.

Titles/captions identify cube-double task2, seed 0, shared 500k offline checkpoint,
50k primitive steps, 7,500 DAWN updates, inherited Q, online-only replay and
the evaluation action mode. Native is context (45,001 updates), not a fifth
residual-scale setting. Report mean-residual evaluations separately at 50 eps.

Retain per-episode success/return/length, scale, step, evaluation count, action
mode, initial-state hashes and training diagnostics in exported data. Validate
source hashes, configs, effective scale, paired evaluation identities, aggregate
means, warmup equality and final100/first50 agreement before plotting. Inspect
exported PNGs at readable size and correct clipping or overlapping labels.
