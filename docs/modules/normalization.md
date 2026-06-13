# `normalization.py`

Log-transform, global z-score normalization, training-derived quantile class bins, and class-distribution plotting for KPI datasets.

## Used by

- `src/dataset_construction.py`

## Main API

| Function | Description |
|----------|-------------|
| `quantile_cuts_from_config(config, key)` | Parse `kpi.class_bins.<key>.cuts` quantile fractions |
| `load_split_lookup(split_csv)` | Build `(operating_point, contingency) → split` lookup |
| `build_class_dataset_for_type(...)` | Log → train-only scaler → z-cuts → class labels → CSV + scaler pkl |
| `save_normalization_report(rows, path)` | Write `KPI_normalization.csv` |
| `kpi_class_counts(csv_path, n_classes)` | Count class labels in a dataset CSV |
| `plot_voltage_spower_distribution(...)` | Save grouped bar chart PNG |

## Normalization flow

1. **`log1p`** on all finite KPI values with `x > -1` (whole dataset, before split-specific steps).
2. Fit one global **`StandardScaler`** (μ, σ) on all finite train cells pooled across components.
3. Transform every cell with that μ/σ.
4. Compute z-score cut thresholds from **training** z-values at the configured quantile fractions.
5. Assign classes on train/validation/test using those fixed thresholds.
6. Override action (and voltage disconnection) flag cells to class `len(cuts) + 1`.

Spower uses action flags only; voltage also applies disconnection flags.

## Outputs (under `data/normalization/`)

| File | Content |
|------|---------|
| `KPI_normalization.csv` | Per KPI type: μ, σ, z-cut columns, class counts |
| `kpi_scaler_voltage.pkl`, `kpi_scaler_spower.pkl` | `sklearn.preprocessing.StandardScaler` |

## Related

- [`dataset_construction`](../src/dataset_construction.md), [`paths`](paths.md)
