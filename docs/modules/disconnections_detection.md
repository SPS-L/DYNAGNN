# `disconnections_detection.py`

Labels components as **disconnected / de-energized** after contingencies using timeline events (spower) and voltage curve analysis (voltage).

## Used by

- `main.py` (via `src/curves_post_process.py`)
- `dataset_construction.py` (KPI masking)

## Inputs

| Source | Content |
|--------|---------|
| `Simulations_Scenarios/operating_point_N/contingency_*/` | Timeline XML, curve outputs |
| `.dyd` / IIDM | Component mapping, energization checks |
| `config.yaml` | `simulation.event_time`, `network.country_filter` |

## Outputs

Per OP under `data/Disconnections/`:

- `disconnections_voltage_operating_point_N.csv`
- `disconnections_spower_operating_point_N.csv`

Same schema as action tables: `OP`, `Contingency`, component columns with `0`/`1`.

## Main API

| Function | Description |
|----------|-------------|
| `process_disconnections_operating_point(op_dir, ...)` | One OP |
| `run_disconnections_detection(op_start=..., op_end=..., op_numbers=...)` | Batch |

## Notes

- **Spower:** generator disconnections from timeline + DYD static ids.
- **Voltage:** de-energization from processed voltage traces (with all-zero curve filtering).
- Combined `DISC_voltage.csv` / `DISC_spower.csv` written by `dataset_construction.py`.
