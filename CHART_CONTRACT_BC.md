# BC pretraining replacement: scientific comparison

- Question: how do the four approved online settings behave when initialized from pure BC instead of QAM pretraining?
- Standalone Matplotlib PNG/PDF. Four success-rate panels: native QAM, DAWN online-only, DAWN QAM replay, PD QAM replay. Each panel compares exactly two measured curves, QAM pretraining in blue and BC pretraining in gold, with distinct markers/line styles.
- Match DAWN against the previous random-Q arms. Native and PD panels explicitly state that critic initialization also changes; native additionally resets its combined optimizer. These two panels are complete transfer-configuration comparisons.
- Seven predeclared measured evaluation points, 50 episodes each, plotted at actual primitive step counts. No smoothing, synthetic points, or across-seed uncertainty bands. Final 100-episode scores and mean-residual diagnostics are separate tables.
- One shared 500k BC checkpoint, seed 0, 50k primitive online steps per new arm. Note the unchanged per-method 5k/20k/8k learning starts and PD 30k progressive exploration.
- Report all ten arms, four BC offline checkpoint evaluations, exact paired comparisons, source hashes, new BC cache provenance, and BC warmup equality. Inspect the final PNG for text overlap and clipping.
