# `gat_training_exports.py`

Shared **GAT training metrics, composite selection score, logging, and figure exports** used by both voltage and spower training loops.

## Used by

- `modules/gat_voltage_training.py`
- `modules/gat_spower_training.py`

## Inputs

| Source | Content |
|--------|---------|
| PyG `DataLoader` | Train / val / test graph batches |
| Trained `nn.Module` | GAT + CORAL head |
| `config.yaml` | `training.selection_f1_weight`, `training.selection_loss_weight` (required) |
| Task-specific callbacks | `forward_fn`, `get_labels_fn`, `coral_loss_fn`, `predict_fn` injected by each trainer |

## Outputs

Written under `data/training/voltage/` or `data/training/spower/` by the trainers:

| Path | Producer |
|------|----------|
| `optuna_trials.csv` | `export_optuna_trials_csv` |
| `loss_curves.png` | `plot_loss_curves` |
| `test_distance_hist.png` | `plot_distance_histogram` |
| `pred_true_examples/*.png` | `plot_pred_true_examples` |

Console / log output via `log_detailed_metrics` during test evaluation.

## Main API

| Function | Description |
|----------|-------------|
| `resolve_selection_weights(config)` | Read required `training.selection_f1_weight` and `training.selection_loss_weight` |
| `composite_selection_score(metrics, f1_weight=..., loss_weight=...)` | Validation objective for checkpoints and Optuna |
| `evaluate_detailed(...)` | One pass over a loader; returns full ordinal metrics dict |
| `log_detailed_metrics(logger, metrics, label=..., num_classes=...)` | Multi-line test logging (notebook-compatible) |
| `export_optuna_trials_csv(study, path)` | Flatten Optuna trials to CSV |
| `plot_loss_curves(train_history, val_history, path)` | Train/val CORAL loss curves |
| `plot_distance_histogram(metrics, num_classes, path)` | Styled pred−true distance bar chart |
| `plot_pred_true_examples(...)` | Up to 5 UNDER + 5 OVER class-9 pred−true scatter plots |
| `collect_pred_true_diffs(...)` | Flatten all test pred−true differences (helper) |
| `safe_div(numer, denom)` | Division with zero-denominator guard |

## Composite selection score

Best epoch within each trial and the winning Optuna trial maximize:

`high_recall + selection_f1_weight × high_f1 − selection_loss_weight × loss`

This score is **not backpropagated** — CORAL loss drives gradients; the composite score is used only for checkpointing and hyperparameter search.

## `evaluate_detailed` metrics

Key fields in the returned dict:

- **Loss / accuracy**: `loss`, `acc`, `correct`, `total`
- **Ordinal errors**: `false_under`, `false_over`, `under_by_k`, `over_by_k`, `mae`
- **Hard under-reporting**: `hard_under_gt3`, `hard_rate` (uses `CONSTRAINT_MAX_UNDER_CLASSES = 3` for logging only)
- **Per-class accuracy**: `class_correct`, `class_total`
- **High-class detection** (classes ≥ `high_class_threshold`, or top class when `null`): `high_recall`, `high_precision`, `high_f1`, `high_tp` / `high_fp` / `high_fn`

When `optimizer` is passed, the function runs in train mode (backward + step); otherwise eval mode.

## Notes

- `CONSTRAINT_MAX_UNDER_CLASSES` affects **reporting** in `log_detailed_metrics` only; it is not used in loss or selection.
- `plot_pred_true_examples` saves **pred−true scatter** plots only (not true-vs-pred class comparison plots from the TRAISIM notebooks).
- See [`src/training.md`](../src/training.md) for full selection-score motivation, threshold table, and config examples.
