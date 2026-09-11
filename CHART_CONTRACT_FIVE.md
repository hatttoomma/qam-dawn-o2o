# Five-arm scientific comparison

- Question: after the same QAM offline checkpoint, how does replacing DAWN online-only replay with the QAM replay mechanism affect pretrained versus random critic initialization?
- Takeaway is computed from completed evaluations only. The original three results are retained.
- Standalone Matplotlib scientific PNG/PDF, plus exact CSV and Chinese Markdown report. Follow the runtime requirement to use standard plotting tools for scientific figures intended for export.
- Two panels: success percentage and mean episode return against primitive online steps. Five named methods, seven pre-specified measured checkpoints per method; no smoothing or synthetic intermediate measurements. The fixed evaluation protocol supplies seven points, so retain marked lines and an exact table rather than retrospectively add evaluations.
- Hard two-root cap: blue for pretrained critic, gold for random critic, neutral grey for native QAM. Dashed/open markers for original DAWN online-only replay; solid/filled, different markers for QAM replay. Explicit legend identifies all five methods.
- 13.5×5.8 inch figure, 180 dpi PNG and vector PDF. Show all markers without clipping and inspect the final PNG.
- Subtitle states seed 0, shared 500k offline updates, and 50 evaluation episodes per curve point. Final 100-episode results and paired replay differences are reported separately in a table.
- Mark 20k base-only warmup within the 50k budget. No across-seed confidence bands.
- Sources: raw evaluation JSON, configuration/checkpoint hashes, replay sampling counts, and warmup comparison audits. No external publication or hosted dashboard.
