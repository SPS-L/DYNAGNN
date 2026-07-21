# `src/curves_post_process.py`

Post-processes Dynawo **curve results** into KPI tables and binary **action** / **disconnection** flags, and merges per-OP tables into combined CSVs.

## Invoked by

- `main.py` (third pipeline stage)

## Pipeline

1. Extract per-OP KPI tables (`run_kpi`).
2. Detect per-OP action flags (`run_actions_detection`).
3. Detect per-OP disconnection flags (`run_disconnections_detection`).
4. Merge per-OP KPI, action, and disconnection tables; mask flagged KPI cells with `NaN` in the combined KPI tables.
5. Write combined **`KPI_voltage.csv`** / **`KPI_spower.csv`** and combined action/disconnection CSVs.

## Inputs

| Source | Content |
|--------|---------|
| `Simulations_Scenarios/` | Per-contingency `outputs/curves/curves.xml` |
| `data/generator_Snom/` | Spower KPI normalization |
| `data/inputs/contingencies.csv` | Fault labels |
| `data/op_graphs/operating_point_N.pt` | Graph component ids (filters unknown contingencies during merge) |
| `config.yaml` | `kpi.*`, `simulation.event_time`, `network.country_filter` |

## Outputs

Per operating point under `data/`:

| Directory | Per-OP files |
|-----------|--------------|
| `KPI/` | `KPI_voltage_operating_point_N.csv`, `KPI_spower_operating_point_N.csv` |
| `Actions/` | `actions_voltage_operating_point_N.csv`, `actions_spower_...` |
| `Disconnections/` | `disconnections_voltage_operating_point_N.csv`, `disconnections_spower_...` |

Combined artifacts (used for KPI cut analysis and downstream dataset construction):

| Path | Role |
|------|------|
| `KPI/KPI_voltage.csv`, `KPI_spower.csv` | Combined **raw** KPI tables (masked) |
| `Actions/ACTIONS_voltage.csv`, `ACTIONS_spower.csv` | Combined action flags |
| `Disconnections/DISC_voltage.csv`, `DISC_spower.csv` | Combined disconnection flags |

## Main entry points

| Function | Description |
|----------|-------------|
| `main()` | Full post-process flow (per-OP extraction → combine) |
| `build_combined_tables()` | Merge per-OP tables and write combined CSVs |

## Related modules

- [`kpi`](../modules/kpi.md), [`actions_detection`](../modules/actions_detection.md), [`disconnections_detection`](../modules/disconnections_detection.md)

## Notes

Use `main.py --to-step curve_process` when you want combined KPI tables **without** building the split or class-label datasets — for example before choosing `kpi.class_bins.*.cuts` (see [`HowTo.md`](../HowTo.md#kpi-cut-thresholds--recommendations)). To rebuild the split after changing `training.*` split settings, run `main.py --from-step split --to-step split`.
