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

The selection score is **not** backpropagated; gradients come from CORAL loss only. The composite score is evaluated on the validation set after each epoch and used solely to pick the best checkpoint within a trial and the best Optuna trial.

#### What `high_recall` and `high_f1` mean

Severity classes are ordinal integers `0 … num_classes - 1` (higher = more severe). For the selection score, each validation node is first converted to a binary label: **high** vs **not high**.

The cutoff comes from `training.high_class_threshold`:

| `high_class_threshold` | Which classes count as **high** | Example (`num_classes: 10`, classes 0–9) | Nordic smoke test (`num_classes: 4`, classes 0–3) |
|------------------------|----------------------------------|----------------------------------------|-----------------------------------------------------|
| `8` | All classes **≥ 8** | Classes **8 and 9** | — |
| `2` | All classes **≥ 2** | — | Classes **2 and 3** (default from `Nordic_test_setup.py`) |
| `null` | Top class only (`num_classes - 1`) | Class **9** only | Class **3** only (metrics only; no weighted sampling) |

So with `high_class_threshold: 8`, a bus/generator with true class 8 or 9 is a high-severity node; classes 0–7 are not. With `high_class_threshold: null`, only the top class counts as high for metrics—the same cutoff as `num_classes - 1`, but weighted sampling and the CORAL under-penalty stay disabled. The Nordic smoke test uses `high_class_threshold: 2` so both upper severity bins (2 and 3) count as high.

From validation predictions, each node gets two binary flags using the **same** cutoff:

- **True high** — ground-truth class ≥ cutoff
- **Predicted high** — predicted class ≥ cutoff

Example (`high_class_threshold: 8`): true class 9 and predicted class 7 → true high, not predicted high (miss → counts toward FN). True class 6 and predicted class 8 → not true high, predicted high (false alarm → counts toward FP).

Then:

- **`high_recall`** = TP / (TP + FN) — of all nodes that are truly high, how many did the model also flag as high?
- **`high_precision`** = TP / (TP + FP) — of all nodes the model flagged as high, how many are truly high?
- **`high_f1`** = harmonic mean of `high_recall` and `high_precision`

`high_f1` uses the **same** high/not-high split and the **same** `high_class_threshold` (or top-class fallback) as `high_recall`; it is not a separate class definition. Both metrics only change when you change `training.high_class_threshold` or `model.num_classes`.

#### Motivation

We weight high-class recall heavily because **missing an important component** (under-predicting severe instability) is typically more costly than **including extra components** (over-predicting severity). The composite score favors catching high-severity events while still penalizing excessive false alarms via `high_f1` and raw CORAL `loss`.

## Related modules

- [`dataset_split`](../modules/dataset_split.md), [`gat_training_exports`](../modules/gat_training_exports.md), [`gat_voltage_training`](../modules/gat_voltage_training.md), [`gat_spower_training`](../modules/gat_spower_training.md), [`graph_construction`](../modules/graph_construction.md), [`electric_distance`](../modules/electric_distance.md)
