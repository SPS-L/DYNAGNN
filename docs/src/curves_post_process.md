# `src/curves_post_process.py`

Post-processes Dynawo **curve results** into KPI tables and binary **action** / **disconnection** flags.

## Invoked by

- `main.py` (third pipeline stage)

## Inputs

| Source | Content |
|--------|---------|
| `Simulations_Scenarios/` | Per-contingency `outputs/curves/curves.xml` |
| `data/generator_Snom/` | Spower KPI normalization |
| `data/inputs/contingencies.csv` | Fault labels |
| `config.yaml` | `kpi.*`, `simulation.event_time`, `network.country_filter` |

## Outputs

Per operating point under `data/`:

| Directory | Files |
|-----------|--------|
| `KPI/` | `KPI_voltage_operating_point_N.csv`, `KPI_spower_operating_point_N.csv` |
| `Actions/` | `actions_voltage_operating_point_N.csv`, `actions_spower_...` |
| `Disconnections/` | `disconnections_voltage_operating_point_N.csv`, `disconnections_spower_...` |

## Main entry point

| Function | Description |
|----------|-------------|
| `main()` | Runs `run_kpi()`, `run_actions_detection()`, `run_disconnections_detection()` in sequence |

## Related modules

- [`kpi`](../modules/kpi.md), [`actions_detection`](../modules/actions_detection.md), [`disconnections_detection`](../modules/disconnections_detection.md)
