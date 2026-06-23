# `modules/kpi_visualization.py`

Histograms of KPI values at each preprocessing stage during dataset construction.

## Outputs (per KPI type)

Written under `data/Dataset/KPI_visualization/`:

| File | Content |
|------|---------|
| `KPI_voltage_histogram.png` | Raw KPI values (log-scaled x and y axes) |
| `KPI_voltage_log10_histogram.png` | `log10(KPI)` after zero replacement |
| `KPI_voltage_zscore_histogram_class_cuts.png` | Train-fitted z-scores with red range-cut lines |
| `KPI_spower_*.png` | Same three plots for spower |

## Main functions

| Function | Description |
|----------|-------------|
| `plot_kpi_pipeline_histograms(...)` | Three histograms for one KPI type |
| `plot_all_kpi_pipeline_histograms(...)` | Voltage + spower pipeline plots |

Invoked automatically from `src/dataset_construction.py` after class datasets are built.
