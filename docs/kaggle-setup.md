# Kaggle setup checklist

See [KAGGLE_TRAINING.md](../KAGGLE_TRAINING.md) for installation and commands.

1. Install the local development environment and load the portable starting checkpoint.
2. Prepare team splits and replay data, retaining provenance and keeping holdout evidence out
   of the donor bank. Verify every path in the chosen recipe.
3. Copy the example account settings and authenticate with the official Kaggle client.
4. Freeze a small trial with `fog-experiment prepare --trial`, then submit its unique ID.
5. Check that CUDA is available, direct and replay audits pass, and replay shards validate.
6. Collect the result and inspect pipeline status, checkpoint/state pairs, logs, and hashes.
7. Run a local matched evaluation before deciding whether a larger experiment is justified.

For manual notebooks, attach a private dataset containing the bundle, enable a GPU and Internet
for dependency installation, and inspect setup before setting `RUN_TRAINING = True`. Save a
notebook version with outputs and retain its export archive. Small trials verify execution;
they do not establish stronger play or sufficient dataset coverage.
