# `simulation_results.py`

Helpers for reading **`simulation_results.csv`** and filtering contingency folders to **successful** Dynawo runs.

## Used by

- `src/simulate.py` (read existing results for resume/skip)
- `src/curves_post_process.py` (`load_successful_runs`)
- `modules/kpi.py`
- `modules/actions_detection.py`
- `modules/disconnections_detection.py`

## Inputs

| Source | Content |
|--------|---------|
| `<data.path>/Simulations_Scenarios/simulation_results.csv` | Per-scenario status log |

CSV columns: `Operating Point`, `Contingency`, `Status` (and any others written by `simulate`).

## Outputs

No files written by this module — it only **reads** the results CSV and returns in-memory structures.

Downstream stages use the filtered contingency list to skip failed or incomplete simulations.

## Main API

| Function | Description |
|----------|-------------|
| `resolve_results_csv(results_dir)` | Path to `simulation_results.csv` under `Simulations_Scenarios/` |
| `read_simulation_results(results_csv)` | `(operating_point, contingency) → status` dict |
| `load_successful_runs(results_csv)` | Set of pairs where `Status == "Success"` |
| `list_successful_contingency_dirs(op_dir, successful_runs)` | Sorted `contingency_*` subdirs for one OP that succeeded |

## Constants

| Name | Value |
|------|-------|
| `SIMULATION_RESULTS_CSV` | `"simulation_results.csv"` |
| `SUCCESS_STATUS` | `"Success"` |

## Notes

- Missing CSV returns an empty dict / set (no error).
- `list_successful_contingency_dirs` matches folder names like `contingency_*` against `(op_dir.name, folder.name)` in the successful-runs set.
- See [`src/simulate.md`](../src/simulate.md) for how rows are appended during simulation and how re-runs skip successful scenarios.
