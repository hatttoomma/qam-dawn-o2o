# Dataset delivery

The original files were downloaded on the local host from the public official HTTP endpoint (the HTTPS endpoint currently redirects to an unrelated 404 page):

- http://rail.eecs.berkeley.edu/datasets/ogbench/cube-double-play-v0.npz
- http://rail.eecs.berkeley.edu/datasets/ogbench/cube-double-play-v0-val.npz

Remote transfer was accelerated using the same files mirrored at:

https://hf-mirror.com/datasets/zhouzypaul/ogbench_datasets/tree/8e7f4278a635a75338f0f1dc0e8f029d61503c4b

The full SHA-256 of each remote mirror download was checked against the locally downloaded ORIGINAL file before publication under its final filename. The bytes match exactly; this is not regenerated data.

| File | Bytes | SHA-256 |
|---|---:|---|
| cube-double-play-v0.npz | 297435656 | a73d1a33d029cedb8bc170ef94791ec585fa2d9450096f4f2a02b8cfbcf608c9 |
| cube-double-play-v0-val.npz | 29726524 | b1fcdf4bd40750351a58d0d491d6be198366ce898f0c6a2e4cb5db331966013e |
