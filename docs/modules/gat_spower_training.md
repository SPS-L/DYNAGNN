# `gat_spower_training.py`

**GAT + CORAL** training loop for the **apparent-power (generator) severity** task.

## Used by

- `main.py` (via `src/training.py`, after spower labels are bound to `y_class`)

## Inputs

| Source | Content |
|--------|---------|
| PyG `DataLoader` | Train/val/test graphs with `y_spower` / `y_class` |
| `config.yaml` | `optuna.*`, `training.*`, `model.num_classes` |
| `training.high_class_threshold` | Optional weighted sampling and under-penalty |

## Outputs

| Path | Content |
|------|---------|
| `data/model/gat_spower_best_model.pt` | Best model weights |
| `data/model/gat_spower_best_hparams.json` | Best Optuna hyperparameters |
| `data/training/spower/optuna_trials.csv` | Full Optuna search log |
| `data/training/spower/loss_curves.png` | Train/val CORAL loss (best trial) |
| `data/training/spower/test_distance_hist.png` | Styled pred−true distance histogram |
| `data/training/spower/pred_true_examples/` | Class-9 UNDER/OVER scatter plots |
| `data/training/spower/checkpoints/` | Per-trial checkpoints |

## Main API

| Function | Description |
|----------|-------------|
| `run_gat_spower_training(...)` | Optuna search + test eval + figure export |
| `GAT_S` | Model class |
| `coral_predict` | Ordinal decoding from CORAL logits |

## Checkpoint / Optuna selection

Best epoch within each trial and the winning trial are chosen by maximizing validation composite score:

`high_recall + selection_f1_weight × high_f1 − selection_loss_weight × loss`

Weights are **required** in `config.yaml`: `training.selection_f1_weight`, `training.selection_loss_weight`.

## Config keys

- `optuna.n_trials`, `optuna.hparams.*` — search space
- `training.selection_f1_weight`, `training.selection_loss_weight` — required composite score weights (not Optuna-tuned)
- Shared with voltage task (separate study per task in `src/training.py`)
