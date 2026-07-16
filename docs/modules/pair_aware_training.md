# `pair_aware_training.py`

Repository integration for pair-aware GINE training: attach shared identity/event tensors and task KPI targets, run per-task Optuna studies, and write deployment checkpoints.

## Used by

- `src/training.py` (`attach_pair_aware_targets`)
- `modules/voltage_training.py`, `modules/spower_training.py` (`run_task_training`)

## Main API

| Function | Description |
|----------|-------------|
| `attach_pair_aware_targets(graph_dataset, data_dir=..., epsilon=..., logger=...)` | Shared node/contingency vocabularies, event masks, log-KPI targets, structural flag masks |
| `run_task_training(task=..., train/val/test_scaled=..., ...)` | Independent Optuna study + test eval + deployment checkpoint |
| `normalize_op(value)` | Canonical `operating_point_<N>` name |

## Inputs (via `attach_pair_aware_targets`)

| Source | Content |
|--------|---------|
| Combined KPI CSVs | Finite log-KPI regression targets |
| Combined DISC CSVs | Structural flag-class masks |
| Graph dataset | Topology, labels, event location metadata |

## Outputs (via `run_task_training`)

| Path | Content |
|------|---------|
| `data/model/<task>_best_model.pt` | Deployment checkpoint (weights, vocabs, cuts, decode mode, …) |
| `data/model/<task>_best_hparams.json` | Same metadata without weights |
| `data/training/<task>/optuna_*.sqlite3`, `optuna_trials.csv` | Optuna study artifacts |

## Flow (per task)

1. Bind task-specific label / log-KPI / mask attributes; fit train-only log-KPI mean/std.
2. Sample Optuna hparams from `optuna.hparams` (capacity + optimizer only).
3. Train with fixed `training.pair_aware` loss weights; maximize validation selection score.
4. Evaluate the winning trial on the test set; save the deployment checkpoint.

Voltage and Spower use **separate** Optuna studies but a **shared** node/contingency vocabulary from attachment.

## Related modules

- [`pair_aware_gine`](pair_aware_gine.md), [`voltage_training`](voltage_training.md), [`spower_training`](spower_training.md)
- [`src/training.md`](../src/training.md)
