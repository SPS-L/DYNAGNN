# `src/dataset_split_step.py`

Builds **`train_val_test_split.csv`** from the combined KPI table using `training.*` split settings in `config.yaml`.

## Invoked by

- `main.py` (fourth pipeline stage)

## Pipeline

1. Read `KPI/KPI_voltage.csv` (`OP`, `Contingency` columns).
2. Apply `training.split_mode`, fractions, and `training.seed` from `config.yaml`.
3. Write `Dataset/train_val_test_split.csv`.

## Inputs

| Source | Content |
|--------|---------|
| `KPI/KPI_voltage.csv` | Scenario keys (must exist from `curve_process`) |
| `config.yaml` | `training.split_mode`, `training.training` / `validation` / `testing`, `training.seed` |

## Outputs

| Path | Role |
|------|------|
| `Dataset/train_val_test_split.csv` | Train / validation / test assignment per scenario |

## Notes

Use `main.py --from-step split --to-step split` to rebuild the split **without** re-running KPI extraction, actions/disconnections detection, or combined KPI table generation.

## Related modules

- [`dataset_split`](../modules/dataset_split.md)
