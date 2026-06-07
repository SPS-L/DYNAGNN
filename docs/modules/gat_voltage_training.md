# `gat_voltage_training.py`

**GAT + CORAL** training loop for the **voltage (substation / voltage-level) severity** task.

## Used by

- `main.py` (via `src/training.py`, voltage pass: `y_voltage` → `y_class`)

## Inputs

| Source | Content |
|--------|---------|
| PyG `DataLoader` | Graphs with bus-level voltage severity labels |
| `config.yaml` | `training.*`, `optuna.*`, `model.num_classes` |

## Outputs

| Path | Content |
|------|---------|
| `data/model/gat_voltage_best_model.pt` | Best model weights |
| `data/model/gat_voltage_best_hparams.json` | Best Optuna hyperparameters |
| `data/training/voltage/optuna_trials.csv` | Full Optuna search log |
| `data/training/voltage/loss_curves.png` | Train/val CORAL loss (best trial) |
| `data/training/voltage/test_distance_hist.png` | Styled pred−true distance histogram |
| `data/training/voltage/pred_true_examples/` | Class-9 UNDER/OVER scatter plots |
| `data/training/voltage/checkpoints/` | Per-trial checkpoints |

## Main API

| Function | Description |
|----------|-------------|
| `run_gat_voltage_training(...)` | Optuna search + test eval + figure export |
| `GAT_V` | Model class |
| `coral_predict` | Thresholded ordinal class prediction |

## Checkpoint / Optuna selection

Best epoch within each trial and the winning trial are chosen by maximizing validation composite score:

`high_recall + selection_f1_weight × high_f1 − selection_loss_weight × loss`

Weights are **required** in `config.yaml`: `training.selection_f1_weight`, `training.selection_loss_weight`.

## Notes

Architecture mirrors `gat_spower_training.py` but targets bus-level labels from `Dataset_Voltage.csv`. Scalers (`x_scaler.pkl`, `edge_attr_scaler.pkl`) are **shared** between both tasks in `src/training.py`.
