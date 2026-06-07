# `src/simulate.py`

Dynawo **contingency simulation** driver: optional OP initialization, curve export setup, scenario creation, and Dynawo runs for every operating point × contingency.

## Invoked by

- `main.py` (first pipeline stage)

## Inputs

| Source | Content |
|--------|---------|
| `data/inputs/operating_point_*` | Pre-fault Dynawo cases |
| `data/inputs/contingencies.csv` | Contingency definitions |
| `config.yaml` | `dynawo.path`, `simulation.initialization_duration`, `simulation.event_time` |

## Outputs

| Path | Content |
|------|---------|
| `Simulations_Scenarios/operating_point_N/contingency_<id>/` | Case copy + `Events.dyd` / `Events.par` + Dynawo `outputs/` |
| `Simulations_Scenarios/simulation_results.csv` | Resume/skip status (`Operating Point`, `Contingency`, `Status`) |
| `data/dynagnn.log` | Pipeline log (via `modules.pipeline_logging`) |

## Main entry point

| Function | Description |
|----------|-------------|
| `main()` | Run full simulation pass for all operating points in `data/inputs/` |

## Behaviour

1. Optional **initialization** per OP when `simulation.initialization_duration` > 0.
2. **`generate_curves()`** — writes `fic_CRV.xml` per OP.
3. For each OP × contingency: create scenario folder, run Dynawo; **Success** rows are skipped on re-run (read from `simulation_results.csv`).

## Related modules

- [`event_files`](../modules/event_files.md), [`initialization`](../modules/initialization.md), [`dynawo_runner`](../modules/dynawo_runner.md), [`curve_generation`](../modules/curve_generation.md), [`simulation_results`](../modules/simulation_results.md)
