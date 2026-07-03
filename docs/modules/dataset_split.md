# `dataset_split.py`

Creates **`train_val_test_split.csv`** from a table with `OP` and `Contingency` columns using configurable fractions or operating-point grouping.

## Used by

- `src/curves_post_process.py` (split built from combined `KPI_voltage.csv`)

## Inputs

| Source | Content |
|--------|---------|
| `data/KPI/KPI_voltage.csv` | Example keys for splitting |
| `config.yaml` | `training.split_mode`, `seed`, `training` / `validation` / `testing` fractions |

## Outputs

- `data/Dataset/train_val_test_split.csv` — columns: `split`, `operating_point`, `contingency`

## Main API

| Function | Description |
|----------|-------------|
| `load_split_settings(config)` | Parse `SplitSettings` |
| `build_dataset_split(input_csv, output_csv=...)` | Write split CSV; returns `SplitSummary` |

## Split modes

- **`operating_point`**: entire OPs assigned to one split (reduces leakage)
- **`scenario`**: random split of individual scenarios

## Notes

Re-run `main.py --from-step curve_process` to regenerate the split when `training.*` split settings change.
