# Policy Decorator scientific pilot chart

- Question: how does the new inherited-critic PD configuration compare with inherited-critic DAWN under the same QAM replay and with native QAM?
- Sources: actual seven-grid 50-episode evaluation JSON files. Final 100-episode results belong in the table, not appended to the curve. Step 0 is the shared frozen QAM reference; later points evaluate the full decorated policy without the training exploration gate.
- Standalone Matplotlib PNG/PDF, two panels (success rate, mean return), three methods: grey native QAM, blue DAWN+QAM replay, gold PD+QAM replay. Distinct markers and explicit legend. No smoothing, invented intermediate measurements, or across-seed confidence bands.
- All six experimental arms are retained in a separate CSV/table. Mean-residual PD evaluations are a separate diagnostic table with explicit denominators.
- Subtitle states seed 0, shared 500k offline checkpoint, 50 episodes per measured curve point. Footnote states PD learning starts=8k/progressive horizon=30k, DAWN base-only warmup=20k, and primitive-step units.
- Inspect final PNG for clipping and text overlap. Final claims are generated only after all 50k steps and source/hash/aggregate validations finish.
