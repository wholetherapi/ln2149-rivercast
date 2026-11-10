# ln2149-rivercast

Official implementation for the paper **RiverCast: Basin-Scale Streamflow Nowcasting with Gated Recurrent Mixtures**.

## Citation

> Jennifer Baker, Priya Raman, Meng Wu. "RiverCast: Basin-Scale Streamflow Nowcasting with Gated Recurrent Mixtures." Accepted at KDD 2026.

## What is included

- `src/model.py` — model definition
- `src/train.py` — training entry point
- `src/rollout.py` — evaluation / rollout utilities
- `src/dataset.py`, `src/metrics.py`, `src/utils.py` — data and metric helpers
- `configs/basins/` — per-basin configs used for the reported results
- `configs/` — experiment configuration
- `scripts/prepare_data.py`, `scripts/run_experiments.sh` — data prep and experiment driver
- `tests/` — smoke tests for the model and data pipeline

The repository contains the complete training and evaluation pipeline used for the
results reported in the paper. See `docs/setup.md` for environment setup and
`scripts/run_experiments.sh` for the full experiment driver.
