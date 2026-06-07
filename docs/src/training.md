# `src/training.py`

End-to-end **GAT training**: build a shared PyG dataset (voltage + spower labels), append electrical-distance features, train/val/test split, scaler fitting, Optuna hyperparameter search, and checkpoint export.

## Invoked by

- `main.py` (fifth pipeline stage)

## Inputs

| Source | Content |
|--------|---------|
| `data/Dataset/Dataset_Voltage.csv`, `Dataset_Spower.csv` | Class labels |
| `data/op_graphs/*.pt` | Graph structure and metadata |
| `data/op_electric_distance/*.csv` | `dz_fault` feature |
| `config.yaml` | `training.*`, `optuna.*`, `model.num_classes`, `network.country_filter` |

## Outputs

| Path | Content |
|------|---------|
| `data/Dataset/train_val_test_split.csv` | Created if missing |
| `data/model/x_scaler.pkl`, `edge_attr_scaler.pkl` | Feature scalers |
| `data/model/gat_voltage_best_model.pt`, `gat_spower_best_model.pt` | Checkpoints |
| `data/model/gat_*_best_hparams.json` | Best hyperparameters per task |
| `data/training/voltage/`, `data/training/spower/` | Optuna CSV, loss curves, test figures, checkpoints |

## Main entry point

| Function | Description |
|----------|-------------|
| `main()` | Full training flow (voltage then spower) |

## Flow (summary)

1. Ensure train/val/test split CSV exists (`dataset_split`).
2. Build shared `graph_dataset` with `y_voltage` and `y_spower` masks.
3. Append log electrical distance from fault to each node.
4. Fit scalers on train split; build weighted loaders if `high_class_threshold` is set.
5. `run_gat_voltage_training()` then `run_gat_spower_training()` (Optuna maximizes validation composite score `high_recall + selection_f1_weight·high_f1 − selection_loss_weight·loss`).

### Training selection score

Per-epoch checkpoints and the winning Optuna trial use:

`score = high_recall + selection_f1_weight × high_f1 − selection_loss_weight × loss`

Requires `training.selection_f1_weight` and `training.selection_loss_weight` in `config.yaml`.

## Related modules

- [`dataset_split`](../modules/dataset_split.md), [`gat_voltage_training`](../modules/gat_voltage_training.md), [`gat_spower_training`](../modules/gat_spower_training.md), [`graph_construction`](../modules/graph_construction.md), [`electric_distance`](../modules/electric_distance.md)
