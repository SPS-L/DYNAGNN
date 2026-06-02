# `gat_voltage_training.py`

**GAT + CORAL** training loop for the **voltage (substation / voltage-level) severity** task.

## Used by

- `main.py` (via `src/training.py`, voltage pass: `y_voltage` → `y_class`)

## Inputs

| Source | Content |
|--------|---------|
| PyG `DataLoader` | Graphs with bus-level voltage severity labels |
| `config.yaml` | Same structure as spower task |

## Outputs

Under `data/model/`:

- `gat_voltage_best_model.pt`
- `gat_voltage_best_hparams.json`

## Main API

| Function | Description |
|----------|-------------|
| `run_gat_voltage_training(...)` | Optuna search + final training |
| `GAT_V` | Model class |
| `coral_predict` | Thresholded ordinal class prediction |

## Notes

Architecture mirrors `gat_spower_training.py` but targets voltage-side mask columns from `Dataset_Voltage.csv`. Scalers (`x_scaler.pkl`, `edge_attr_scaler.pkl`) are **shared** between both tasks in `src/training.py`.
