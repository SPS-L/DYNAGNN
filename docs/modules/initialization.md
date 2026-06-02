# `initialization.py`

Runs a **steady-state Dynawo initialization** per operating point (no contingency events) and optionally updates the IIDM in place.

## Used by

- `main.py` (via `src/simulate.py`, batch init before contingencies)
- `DYNAGNN.py` (optional pre-prediction init)

## Inputs

| Source | Content |
|--------|---------|
| `data/inputs/operating_point_N/` or `--case-dir` | `.jobs`, IIDM reference in jobs |
| `config.yaml` | `simulation.initialization_duration` or `inference.initialization_duration` |
| `dynawo.path` | Via `dynawo_runner` |

## Outputs

- Updated IIDM in case folder (when successful)
- Log lines in `data/dynagnn.log` (training pipeline) or discarded (`DYNAGNN.py`)
- `InitResult` per OP (success, messages, run time)

## Main API

| Function | Description |
|----------|-------------|
| `discover_case(op_dir)` | Find jobs + IIDM → `OperatingPointCase` |
| `initialize_one_operating_point(case, execution_path, run_time, log_file, ...)` | Single OP |
| `initialize_operating_points(op_paths, ...)` | Batch |
| `resolve_initialization_duration(config)` | Read duration; `None` or `<= 0` skips |
| `write_initialization_status_log(log_file, result)` | Append init summary |

## Jobs file behaviour

Comments out `Events.dyd` in the jobs modeler section when that line exists (so initialization runs without fault events), then restores the original jobs file. If `Events.dyd` is not listed in jobs, only `stopTime` is patched for the init run.

## Notes

Matches deliverable preprocessing: short Dynawo run so IIDM reflects a consistent initial state before curve generation and contingencies.
