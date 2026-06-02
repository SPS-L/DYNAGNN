# `src/dataset_construction.py`

Merges per-OP KPI and flag tables, applies masks, **normalizes** KPIs, and **discretizes** severity into class labels for GAT training.

## Invoked by

- `main.py` (fourth pipeline stage)

## Inputs

| Source | Content |
|--------|---------|
| `data/KPI/KPI_*_operating_point_*.csv` | Raw KPI tables |
| `data/Actions/actions_*_operating_point_*.csv` | Action flags |
| `data/Disconnections/disconnections_*_operating_point_*.csv` | Disconnection flags |
| `data/op_graphs/operating_point_N.pt` | Graph component ids (filters unknown contingencies) |
| `config.yaml` | `kpi.class_bins` |

## Outputs

| Path | Role |
|------|------|
| `data/Actions/ACTIONS_voltage.csv`, `ACTIONS_spower.csv` | Combined action flags |
| `data/Disconnections/DISC_voltage.csv`, `DISC_spower.csv` | Combined disconnection flags |
| `data/KPI/KPI_voltage.csv`, `KPI_spower.csv` | Normalized KPI tables |
| `data/KPI/KPI_normalization_minmax.csv` | Min/max used per table |
| `data/Dataset/Dataset_Voltage.csv` | Class labels (voltage task) |
| `data/Dataset/Dataset_Spower.csv` | Class labels (spower task) |

## Main entry points

| Function | Description |
|----------|-------------|
| `build_datasets()` | Core merge, mask, normalize, discretize logic |
| `main()` | Calls `build_datasets()` and logs output paths |

Rows use **`OP`**, **`Contingency`**, plus one column per network component id.

## Related modules

- [`paths`](../modules/paths.md) (directory constants)
