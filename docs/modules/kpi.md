# `kpi.py`

Extracts **sliding-window KPIs** from Dynawo **curves.xml** per contingency: voltage severity at voltage levels / buses, apparent-power stress at generators.

## Used by

- `main.py` (via `src/curves_post_process.py`)

## Inputs

| Source | Content |
|--------|---------|
| `Simulations_Scenarios/operating_point_N/contingency_*/outputs/curves/curves.xml` | Time series |
| `.dyd`, IIDM | Curve id → component mapping |
| `data/generator_Snom/operating_point_N.csv` | Spower normalization |
| `data/inputs/contingencies.csv` | Human-readable contingency labels |
| `config.yaml` | `kpi.window_sec`, `kpi.step_sec`, `simulation.event_time`, `network.country_filter` |

## Outputs

Per OP under `data/KPI/`:

- `KPI_voltage_operating_point_N.csv` — `Contingency` + component columns (float scores)
- `KPI_spower_operating_point_N.csv` — same for generators

## Main API

| Function | Description |
|----------|-------------|
| `process_kpi_operating_point` / `process_operating_point` | One OP |
| `run_kpi()` | All operating points under `Simulations_Scenarios/` |

## KPI timing

- Windows start at **`max(0, event_time - 1)`** seconds (see `resolve_kpi_start_time`).
- Window length and step from `kpi.window_sec` and `kpi.step_sec`.

## Curve variables

- Voltage: `*_Upu_value`
- Spower: `generator_PGen`, `generator_QGen` combined per generator

## Notes

Contingencies without `curves.xml` are skipped with a warning. Downstream [`curves_post_process.py`](../src/curves_post_process.md) merges per-OP tables; [`dataset_construction.py`](../src/dataset_construction.md) discretizes combined KPIs into class labels using `kpi.class_bins`.
