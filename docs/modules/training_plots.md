# `training_plots.py`

Post-training diagnostic figures for pair-aware GINE Voltage / Spower runs.

## Used by

- `modules/pair_aware_gine.py` (`save_training_plots` after test evaluation)

## Output directory

All PNGs are written to:

```text
data/training/<study_name>/<task>/plots/
```

(`study_name` = `optuna.study_name`; `task` = `voltage` or `spower`).

## Figures

| File | Source | Contents |
|------|--------|----------|
| `loss_curve.png` | Winning Optuna trial `history.csv` | **One** PNG with **subplots** (2-column grid): total, classification, regression, gate, ordinal. Each panel shows **train** (blue) and **val** (orange). Empty grid cells are hidden. |
| `score_curve.png` | Same Optuna `history.csv` | Validation selection scores per epoch (`class` / `gated` / `logKPI`, plus `selected`) |
| `confusion_matrix.png` | Final train+val test predictions | Row-normalised confusion matrix |
| `distance_histogram.png` | Final test predictions | Histogram of signed offsets (pred − true) |
| `node_example_cls<N>_<UNDER\|OVER>_ex<k>_of_5.png` | Final test predictions | Up to 5 under- and 5 over-prediction examples |

Train/val curves always come from the **best Optuna trial**. Confusion / distance / node examples come from the **final train+val retrain** evaluated on the held-out test set.

## Loss subplot discovery

`plot_loss_curve` discovers columns named `train_<name>_loss` / `val_<name>_loss`. Preferred order:

`total` → `classification` → `regression` → `gate` → `inactive_gate` → `flag_gate` → `ordinal`

Main DYNAGNN training writes `gate` (not `inactive_gate`). Extra keys are only used if present in `history.csv`.

## Main API

| Function | Description |
|----------|-------------|
| `save_training_plots(...)` | Write all available diagnostics for one task |
| `plot_loss_curve(...)` | Multi-subplot train/val loss figure |
| `plot_score_curve(...)` | Validation selection-score curves |
| `plot_confusion_matrix(...)` | Test confusion matrix |
| `plot_distance_histogram(...)` | Test error-offset histogram |
| `plot_node_examples(...)` | Per-class under/over example panels |

## Related modules

- [`pair_aware_gine`](pair_aware_gine.md), [`pair_aware_training`](pair_aware_training.md)
- [`src/training.md`](../src/training.md)
