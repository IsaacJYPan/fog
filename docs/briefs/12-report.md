# Brief 12: replay/live parity (in progress)

**Answer.** The 200-game baseline (smoke) aligns 24,149 rows and misses 3,195 live decisions. Own PP, own Tera fields and entity ids differ on every aligned row; history differs on 95.23%. The tool also exposes HP rounding, replay state blanking and a shorter training history window. It measures actual forward tensors against serialized training rows, with true own sheets and played order. No fixes or play comparison have been completed yet.

Not verified: stages 2 and 3. Baseline and per-row details: `12-artifacts/parity-before.json`; covering tests and rollout digests are alongside it. Both paths intentionally exclude the current turn at replacement requests; the live history overwrite affects 21,107 subsequent decision rows (smoke).
