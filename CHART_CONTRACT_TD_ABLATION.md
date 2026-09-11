# Critic target entropy: scientific chart contract

Question: removing only the critic target entropy term changes online performance
and inherited Q adaptation by how much? Follow completed measurements; no assumed
benefit. Primary control is the same-machine Soft TD rerun, with historical Soft
TD and native QAM only as labeled context.

Use the established standalone Matplotlib PNG/PDF scientific artifact format.
Main 2x2 line figure: task rows, success/mean return columns, measured seven
predeclared curve points (50 evaluation episodes, chunk crossing <=4 steps).
Do not add evaluation points or smooth to satisfy a generic chart-density rule:
this would change the fixed protocol. Final 100-episode comparison uses a table.
Primitive online steps 0-50k; show 20k warmup and 7,500 total updates. Shade no
single-seed uncertainty band. Native contextual curve may be neutral grey dashed,
explicitly labeled 45,001 updates. Soft TD blue solid/filled circles; plain TD
blue lighter dashed/open squares. Initial policy can remain in raw start points.

Second figure: fixed-probe diagnostics at warmup and 30k/40k/50k, task rows,
three columns for mean entropy target contribution, mean residual-minus-base Q,
and gradient norm wrt unscaled residual u. These are four planned diagnostic
checkpoints; use explicit markers, label connecting lines as measured checkpoints
and avoid interpolated conclusions. Counterfactual entropy contribution is shown
also for hard mode but is NOT included in its trained target. Every panel uses
the same 256 warmup transitions and fixed probe RNG within task. MC diagnostics
are in rich CSV/JSON, separating terminated vs time-limited episodes; do not use
unqualified Q-to-truncated-return error as a calibration-performance plot.

White background, neutral axes and quiet y grid, one blue root with line/marker
distinctions, neutral reference, descriptive titles, 13x9 main / 16x9 diagnostic
footprints, shared legend. Success starts at zero; return and signed differences
use data-dependent axes with units and zero reference where meaningful. Keep
complete task/mode/seed/step/update/source metadata and episode-level evidence.
Inspect final PNGs for clipping, legibility, signs and correct denominators.
No website publication; retain reproduction scripts and audit provenance.
