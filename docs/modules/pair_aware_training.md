# `pair_aware_training.py`

Repository integration for pair-aware GINE training: attach shared identity/event tensors and task KPI targets, run per-task Optuna studies, and write deployment checkpoints.

## Used by

- `src/training.py` (`attach_pair_aware_targets`)
- `modules/voltage_training.py`, `modules/spower_training.py` (`run_task_training`)

## Main API

| Function | Description |
|----------|-------------|
| `attach_pair_aware_targets(graph_dataset, data_dir=..., epsilon=..., logger=...)` | Shared node/contingency vocabularies, event masks, log-KPI targets |
| `run_task_training(task=..., train/val/test_scaled=..., ...)` | Independent Optuna study + test eval + deployment checkpoint |
| `normalize_op(value)` | Canonical `operating_point_<N>` name |

## Inputs (via `attach_pair_aware_targets`)

| Source | Content |
|--------|---------|
| Combined KPI CSVs | Finite log-KPI regression targets |
| Graph dataset | Topology, labels, event location metadata |

## Outputs (via `run_task_training`)

| Path | Content |
|------|---------|
| `data/model/<study_name>/<task>_best_model.pt` | Deployment checkpoint (weights, vocabs, cuts, decode mode, …) |
| `data/model/<study_name>/<task>_best_hparams.json` | Same metadata without weights |
| `data/model/<study_name>/x_scaler.pkl`, `edge_attr_scaler.pkl` | Train-fit feature scalers |
| `data/training/<study_name>/<task>/optuna_*.sqlite3`, `optuna_trials.csv` | Optuna study artifacts |
| `data/training/<study_name>/<task>/optuna_trials/trial_N/` | Per-trial `history.csv`, `model_state.pt`, `model_metadata.json` |
| `data/training/<study_name>/<task>/plots/` | Final diagnostic figures (incl. loss curve from best trial) |

## Flow (per task)

1. Read `num_classes` from `config["model"]["num_classes"]` (must be >= 2); validate `len(cuts) == num_classes - 2`.
2. Bind task-specific label / log-KPI / mask attributes; fit train-only log-KPI mean/std using activity classes only (labels `< num_classes - 1`).
3. Sample Optuna hparams from `optuna.hparams` (capacity + optimizer only).
4. Train with fixed `training.pair_aware` loss weights; maximize validation selection score.
5. Evaluate the winning trial on the test set; save the deployment checkpoint.

Voltage and Spower use **separate** Optuna studies but a **shared** node/contingency vocabulary from attachment.

## Related modules

- [`pair_aware_gine`](pair_aware_gine.md), [`voltage_training`](voltage_training.md), [`spower_training`](spower_training.md)
- [`src/training.md`](../src/training.md)
