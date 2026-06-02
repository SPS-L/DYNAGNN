# `gat_spower_training.py`

**GAT + CORAL** training loop for the **apparent-power (generator) severity** task.

## Used by

- `main.py` (via `src/training.py`, after spower labels are bound to `y_class`)

## Inputs

| Source | Content |
|--------|---------|
| PyG `DataLoader` | Train/val/test graphs with `y_spower` / `y_class` |
| `config.yaml` | `optuna.*`, `training.epochs`, `training.patience`, `model.num_classes` |
| `training.high_class_threshold` | Optional weighted sampling and under-penalty |

## Outputs

Under `data/model/`:

- `gat_spower_best_model.pt`
- `gat_spower_best_hparams.json`
- Optuna study artifacts under `data/training/` (via caller logger)

## Main API

| Function | Description |
|----------|-------------|
| `run_gat_spower_training(train_loader, val_loader, test_loader, training_dir, model_dir, config, ...)` | Full Optuna + retrain best |
| `GAT_S` | Model class |
| `coral_predict` | Ordinal decoding from CORAL logits |

## Model

Graph attention layers on bus/generator/load nodes with edge attributes; CORAL ordinal regression for `num_classes` severity levels.

## Config keys

- `optuna.n_trials`, `optuna.hparams.*` — search space
- Shared with voltage task (separate study per task in `src/training.py`)
