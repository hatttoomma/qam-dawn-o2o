# Two-task scale range figures

Question: how do residual scales .01 and 1 affect each task, compared with .1
and all other already measured scales? Derive conclusions from completed data.
Continue the established standalone research PNG/PDF format with Matplotlib.

Figure 1: a 2x2 panel of measured learning curves, rows task1/task2, columns
success/return. Show .01/.1/1 and native context, with seven measured 50-episode
points, primitive steps 0--50k, and the 20k warmup boundary. The reference has
only seven fixed evaluations; keep that protocol, discrete markers and no
smoothing rather than inventing finer data. No seed uncertainty bands.

Figure 2: final 100-episode success/return versus scale, 2x2 panels by task.
Use a clearly labeled logarithmic scale axis from .01 to 1, measured dots only,
and native reference lines. Include task2 .05/.2/.3. Task1 has only .01/.1/1;
do not fill or interpolate the missing intermediate scales. Denominators and
action mode must be visible and distinct from 50-episode mean-residual checks.

Palette policy: one blue root with tones, distinct markers/line styles/open
fills for .01/.1/1, and dark grey native references. White background, dark
labels, quiet grids, no decorative borders. About 13x9 inches, readable facet
titles/legends/captions, success axes starting at zero. No cross-task return
normalization or averaging. Label native's different update count (45,001).

Validate source/config/scale/task/checkpoint/initialization/warmup consistency,
paired seeds and initial states, exact budgets, first50/final100 agreement,
finite metrics and raw means before plotting. Retain richer CSVs with task,
scale, step, episode count, mode, success, return, length and diagnostics.
Inspect both exported PNGs at readable resolution before delivering.
