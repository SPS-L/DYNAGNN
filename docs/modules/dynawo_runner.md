# `dynawo_runner.py`

Low-level **Dynawo execution** and **simulation log formatting** (dynamo.log-style).

## Used by

- `main.py` (via `src/simulate.py`)
- `modules/initialization.py`
- `DYNAGNN.py` (initialization only; log discarded)

## Inputs

| Source | Content |
|--------|---------|
| `config.yaml` | `dynawo.path` (via callers) |
| Scenario `*.jobs` | Dynawo job definition |

## Outputs

- Appends to `data/dynagnn.log` during `main.py` simulation
- Dynawo writes under scenario `output/` (caller-dependent)

## Main API

| Function | Description |
|----------|-------------|
| `run_dynawo_job(execution_path, jobs_file, operating_point, contingency, log_file)` | Shell out to Dynawo env; returns success bool |
| `write_simulation_log_header(log_file, dynagnn_version)` | Version banner once per log |
| `append_simulation_log(log_file, *entries)` | Structured INFO/ERROR lines |
| `default_dynawo_execution_path(case_dir, config_path)` | Resolve env script for a case folder |

## Log format

Lines mirror Dynawo style, e.g. `| INFO | ...`, with `DYNAGNN VERSION` at file start.

## Notes

Failures are recorded in the simulation CSV (`Failed`) and in the log; successful runs are skipped on resume.
