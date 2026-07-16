# `src/dataset_construction.py`

Assigns class labels from **fixed raw KPI cut thresholds** and writes dataset artifacts. Expects combined KPI and flag tables from `curve_process`.

## Invoked by

- `main.py` (fourth pipeline stage)

## Pipeline (v1.11 labeling, retained in v1.2)

1. Read combined KPI and flag tables written by `curve_process`.
2. For each KPI type: assign class labels from fixed raw cuts in `config.yaml` → override action/disconnection cells to the flag class.
3. Save class-bins report, class-label datasets, and class-distribution plot.

With $K$ cuts, labeling produces $K+1$ KPI severity classes (indices $0,\ldots,K$) plus one flag class ($K+1$). Set `model.num_classes` to **`len(cuts) + 2`**.

## Inputs

| Source | Content |
|--------|---------|
| `data/KPI/KPI_voltage.csv`, `KPI_spower.csv` | Combined raw KPI tables (masked) |
| `data/Actions/ACTIONS_voltage.csv`, `ACTIONS_spower.csv` | Combined action flags |
| `data/Disconnections/DISC_voltage.csv`, `DISC_spower.csv` | Combined disconnection flags |
| `config.yaml` | `kpi.class_bins.*.cuts` (raw KPI thresholds) |

## Outputs

| Path | Role |
|------|------|
| `data/Dataset/KPI_class_bins.csv` | Applied raw cut thresholds and class metadata per KPI type |
| `data/Dataset/Dataset_Voltage.csv` | Class labels (voltage task) |
| `data/Dataset/Dataset_Spower.csv` | Class labels (spower task) |
| `data/Dataset/dataset_class_distribution.png` | Grouped bar chart of class counts (voltage vs spower) |

Rows use **`OP`**, **`Contingency`**, plus one column per network component id.

## Class bins (`kpi.class_bins`)

`cuts` lists **strictly increasing raw KPI thresholds** $\tau_1 < \cdots < \tau_K$:

| Class | Rule |
|-------|------|
| $0$ | KPI $\le \tau_1$ |
| $1,\ldots,K-1$ | $\tau_j <$ KPI $\le \tau_{j+1}$ |
| $K$ | KPI $> \tau_K$ |
| $K+1$ | Action / disconnection (voltage and spower: actions + DISC) |

**Example (Nordic):** `cuts: [1e-6, 2.25e-5, 3e-4, 5.625e-4]` → classes 0–4 by KPI magnitude, flag class 5, `model.num_classes: 6`.

Set `model.num_classes` to **`len(cuts) + 2`**.

## Main entry points

| Function | Description |
|----------|-------------|
| `build_datasets()` | Read combined tables → discretize → write dataset artifacts |
| `main()` | Calls `build_datasets()` and logs output paths |

## Related modules

- [`paths`](../modules/paths.md)

## Notes

Combined tables and `train_val_test_split.csv` are produced by [`curves_post_process.py`](curves_post_process.md). Re-run `main.py --from-step curve_process` when split settings change; re-run `main.py --from-step dataset` when KPI cuts change.
