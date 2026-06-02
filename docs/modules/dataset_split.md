# `dataset_split.py`

Creates **`train_val_test_split.csv`** from a dataset table (`OP`, `Contingency`, …) using configurable fractions or operating-point grouping.

## Used by

- `main.py` (via `src/training.py`, if split file does not exist)

## Inputs

| Source | Content |
|--------|---------|
| `data/Dataset/Dataset_Voltage.csv` | Example keys for splitting |
| `config.yaml` | `training.split_mode`, `seed`, `training` / `validation` / `testing` fractions |

## Outputs

- `data/Dataset/train_val_test_split.csv` — columns include `OP`, `Contingency`, `split` (`train` / `val` / `test`)

## Main API

| Function | Description |
|----------|-------------|
| `load_split_settings(config)` | Parse `SplitSettings` |
| `build_dataset_split(input_csv, output_csv=...)` | Write split CSV; returns `SplitSummary` |

## Split modes

- **`operating_point`** (default): entire OPs assigned to one split (reduces leakage)
- Other modes as implemented in module (see source for `split_mode` handling)

## Notes

Deleting the split CSV forces regeneration on the next `main.py` run with the current config seed.
