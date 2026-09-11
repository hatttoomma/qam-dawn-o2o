# Five-task experimental figures

Audience: research collaborator comparing official cube-double QAM-EDIT and the fixed conditioned DAWN-derived online method. Source: the ten verified final100 episode files and seven curve50 checkpoints for each run. Show all five tasks; no selected best checkpoint, smoothed series, or extrapolation.

Endpoint figure: two horizontal dot-plot panels, one for success percentage (axis 0–100) and one for mean raw return (higher is better). Each task gets a pair of method markers and direct numeric labels. A sixth, clearly separated macro-average row gives equal weight to the five tasks. Use orange squares for QAM-EDIT and blue circles for DAWN, with a shared legend. Do not imply episode variation measures training-seed uncertainty. No confidence intervals with only one training seed.

Learning checkpoints figure: five rows (tasks) by two columns (success and return). Seven unconnected observed points per method, common primitive environment step axis 0–50k, success axis 0–100; label return axes and task panels. State that each point is 50 episodes, while the endpoint table/figure use 100. No lines between points that imply observed stability. Warmup budgets differ by the fixed method settings and are stated in the caption rather than visually conflated.

Both figures must state offline500k/online50k, seed0, QAM-EDIT's official selected edit_scale=0, and DAWN's plain TD/batch256/UTD.25/state+base input. Task1/2 reuse matching prior runs; Task3–5 are new. Keep title/subtitle/footer separate from panels and verify exported PNGs visually for readability and clipping. Export standalone PNG and PDF using Matplotlib.
