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
| `data/training/<study_name>/<task>/optuna_trials/trial_N/` | Per-trial artifacts (see below) |
| `data/training/<study_name>/<task>/final_retrain/` | Train+val retrain of best hparams (`history.csv`, `model_state.pt`; mid-run `resume.pt` if interrupted) |
| `data/training/<study_name>/<task>/plots/` | Diagnostic figures from [`training_plots`](training_plots.md): multi-subplot `loss_curve.png` + `score_curve.png` from the best Optuna trial; confusion / distance / node examples from the final train+val test eval |

### Per-trial folder (`optuna_trials/trial_N/`)

| File | Role |
|------|------|
| `params.json` | Sampled Optuna hparams (written as soon as the trial starts) |
| `resume.pt` | Mid-trial checkpoint after every epoch (model, optimizer, scheduler, history, best_*) |
| `result.json` | Finished trial score/metadata (survives a crash between training end and Optuna `tell`) |
| `trial_done.json` | Written when the trial is reported complete or pruned; resume files are cleared |
| `history.csv`, `model_state.pt`, `model_metadata.json` | Final trial artifacts after training finishes |

## Flow (per task)

1. Read `num_classes` from `config["model"]["num_classes"]` (must be >= 2); validate `len(cuts) == num_classes - 2`.
2. Bind task-specific label / log-KPI / mask attributes; fit train-only log-KPI mean/std using activity classes only (labels `< num_classes - 1`).
3. Run Optuna via `study.optimize(..., n_trials=...)` (pruned counts). Prefer finishing any incomplete on-disk `trial_*` (mid-epoch `resume.pt`) before asking the remaining trials.
4. Sample Optuna hparams from `optuna.hparams` (capacity + optimizer only) for each new trial.
5. Train with fixed `training.pair_aware` loss weights; maximize validation selection score (`training.pair_aware.selection_score`).
6. Retrain the best hparams on **train+val** for the winning trial’s `best_epoch` epochs (no validation early stopping).
7. Evaluate that final model on the test set; save it as the deployment checkpoint. Train/val Optuna plots and study tables remain from the best trial.

Voltage and Spower use **separate** Optuna studies, SQLite DBs, and `optuna_trials/` trees, but a **shared** node/contingency vocabulary from attachment. Mid-trial resume is therefore **per task**: if Voltage finished and Spower died at trial 3 / epoch 23, a restart resumes only Spower from epoch 24.

## Mid-trial resume

DYNAGNN trains Optuna **serially** (no parallel workers). If the process is killed mid-trial:

1. Re-run the same training stage (`main.py` with `--from-step training`, or a full pipeline that reaches training).
2. The interrupted task loads `optuna_trials/trial_N/resume.pt` and continues from the next epoch.
3. Already-finished trials in that task’s SQLite study are kept; `optimize` is called with `remaining = n_trials - len(study.trials)`.

No extra CLI flag is required. Pipeline `--from-step` / `--to-step` still control **which stages** run; mid-trial resume applies **inside** the training stage per task.

## Related modules

- [`pair_aware_gine`](pair_aware_gine.md), [`training_plots`](training_plots.md), [`voltage_training`](voltage_training.md), [`spower_training`](spower_training.md)
- [`src/training.md`](../src/training.md)
