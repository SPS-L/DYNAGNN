# `modules/normalization.py`

Log10 transform (with zero replacement), global z-score normalization, training-derived range class bins, and class-distribution plotting for KPI datasets.

## Main functions

| Function | Description |
|----------|-------------|
| `activity_fraction_cuts_from_config(config, key)` | Parse `kpi.class_bins.<key>.cuts` activity fractions |
| `prepare_kpi_matrix(df, value_cols)` | Replace zeros with smallest positive value |
| `log10_transform_values(values)` | Apply `log10` to finite positive cells |
| `log_transform_dataframe(df, value_cols)` | Zero-replace + log10 on KPI columns |
| `build_class_dataset_for_type(...)` | Full normalize → discretize → save scaler/dataset |
| `save_normalization_report(rows, path)` | Write `KPI_normalization.csv` |
| `kpi_class_counts(csv_path, n_classes)` | Count class labels in a dataset CSV |
| `plot_voltage_spower_distribution(...)` | Grouped bar chart of voltage vs spower class counts |

## Pipeline (`build_class_dataset_for_type`)

1. **Zero replacement:** raw KPI zeros are set to the smallest positive value in the table (`log10(0)` is undefined).
2. **`log10`** on all finite positive KPI values (whole dataset, before split-specific steps).
3. Fit one global `StandardScaler` on **train** cells only (all components pooled).
4. Z-score every cell with train-fitted μ and σ.
5. Compute z-cut thresholds from **training** z-scores using **range cuts**: `z_min + fraction × (z_max − z_min)` for each configured activity fraction.
6. Assign integer KPI severity classes; override action/disconnection cells to the flag class.

## Related

- [`kpi_visualization`](kpi_visualization.md) — raw / log10 / z-score histograms with class-cut overlays
- [`dataset_construction`](../src/dataset_construction.md)
