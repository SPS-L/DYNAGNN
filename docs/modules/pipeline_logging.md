# `pipeline_logging.py`

Unified logging for **`main.py`** and all `src/` pipeline stages.

## Used by

- `main.py`, `src/simulate.py`, `src/build_op_assets.py`, `src/curves_post_process.py`, `src/dataset_construction.py`, `src/training.py`

## Outputs

| Path | Content |
|------|---------|
| `data/dynagnn.log` | Single log file per `main.py` run (recreated each run) |

## Main API

| Function | Description |
|----------|-------------|
| `configure_pipeline_logging(log_path=None, ...)` | Set up file + console handlers and optional stdout tee |
| `get_logger()` | Return the `dynagnn` logger (auto-configures if needed) |
| `get_pipeline_log_path()` | Resolved log file path |
| `log_step_banner(step_name)` | Write a `STEP: …` section header |

## Notes

- `DYNAGNN.py` uses its own console logger and does **not** write to `data/dynagnn.log`.
- Dynawo simulation messages from `simulate` are appended to `data/dynagnn.log` via `dynawo_runner.append_simulation_log`.
- When `tee_stdout=True` (default), any `print()` output is redirected into the pipeline logger so it remains timestamped and consistently formatted in `data/dynagnn.log`.
