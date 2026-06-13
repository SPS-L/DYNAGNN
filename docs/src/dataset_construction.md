# `src/dataset_construction.py`

Merges per-OP KPI and flag tables, builds the train/val/test split, applies **log-transform + z-score normalization** (train-only fit), discretizes severity into class labels, and writes normalization artifacts.

## Invoked by

- `main.py` (fourth pipeline stage)

## Pipeline (v1.1)

1. Merge per-OP KPI, action, and disconnection tables; mask flagged KPI cells with `NaN` in the raw combined KPI tables.
2. Write **`KPI_voltage.csv`** / **`KPI_spower.csv`** (raw KPI values, not normalized).
3. Build **`train_val_test_split.csv`** from the voltage KPI table (`OP`, `Contingency`) using `training.*` split settings.
4. For each KPI type: `log1p` on the full dataset → fit a global `StandardScaler` on **train** cells only → transform all splits → compute z-score quantile cuts from train → assign class labels → override action/disconnection cells to the flag class.
5. Save scalers, normalization report, class-label datasets, and a class-distribution plot.

## Inputs

| Source | Content |
|--------|---------|
| `data/KPI/KPI_*_operating_point_*.csv` | Raw KPI tables |
| `data/Actions/actions_*_operating_point_*.csv` | Action flags |
| `data/Disconnections/disconnections_*_operating_point_*.csv` | Disconnection flags |
| `data/op_graphs/operating_point_N.pt` | Graph component ids (filters unknown contingencies) |
| `config.yaml` | `kpi.class_bins.*.cuts` (quantile fractions), `training.*` (split) |

## Outputs

| Path | Role |
|------|------|
| `data/Actions/ACTIONS_voltage.csv`, `ACTIONS_spower.csv` | Combined action flags |
| `data/Disconnections/DISC_voltage.csv`, `DISC_spower.csv` | Combined disconnection flags |
| `data/KPI/KPI_voltage.csv`, `KPI_spower.csv` | Combined **raw** KPI tables (masked, not normalized) |
| `data/Dataset/train_val_test_split.csv` | Train / validation / test split |
| `data/normalization/KPI_normalization.csv` | Global μ, σ, and training z-cut thresholds per KPI type |
| `data/normalization/kpi_scaler_voltage.pkl`, `kpi_scaler_spower.pkl` | Fitted `StandardScaler` objects |
| `data/Dataset/Dataset_Voltage.csv` | Class labels (voltage task) |
| `data/Dataset/Dataset_Spower.csv` | Class labels (spower task) |
| `data/Dataset/dataset_class_distribution.png` | Grouped bar chart of class counts (voltage vs spower) |

Rows use **`OP`**, **`Contingency`**, plus one column per network component id.

## Class bins (`kpi.class_bins`)

`cuts` lists **quantile fractions in (0, 1)**, e.g. `[0.25, 0.5, 0.75]` → four KPI severity classes from pooled training z-scores, plus one action/disconnection flag class. Set `model.num_classes` to `len(cuts) + 2`.

## Main entry points

| Function | Description |
|----------|-------------|
| `build_datasets()` | Full merge → split → normalize → discretize flow |
| `main()` | Calls `build_datasets()` and logs output paths |

## Related modules

- [`normalization`](../modules/normalization.md), [`dataset_split`](../modules/dataset_split.md), [`paths`](../modules/paths.md)
