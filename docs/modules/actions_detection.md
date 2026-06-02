# `actions_detection.py`

Detects **control and topology actions** after the fault time from Dynawo **timeline** XML, maps dynamic model ids to static IIDM components, and writes binary flag tables per operating point.

## Used by

- `main.py` (via `src/curves_post_process.py`)
- `dataset_construction.py` (masks KPI cells where action = 1)

## Inputs

| Source | Content |
|--------|---------|
| `Simulations_Scenarios/operating_point_N/contingency_*/` | `outputs/timeLine/timeline.xml` (or `outputs/timeline/`) |
| Contingency `.dyd` | `build_dyd_id_to_staticid_map` for DM_* → staticId |
| `config.yaml` | `simulation.event_time` (actions with `t >= event_time`) |
| `config.yaml` | `network.country_filter` (optional FR-only components) |

## Outputs

Per operating point under `data/Actions/`:

- `actions_voltage_operating_point_N.csv` — columns: `OP`, `Contingency`, one per voltage-level component
- `actions_spower_operating_point_N.csv` — columns: `OP`, `Contingency`, one per generator

Cell value `1` means an action affected that component; `0` otherwise.

## Main API

| Function | Description |
|----------|-------------|
| `process_actions_operating_point(op_dir, ...)` | Process one OP folder |
| `run_actions_detection(op_start=..., op_end=..., op_numbers=...)` | Batch over `Simulations_Scenarios` |

## Notes

- Timeline events are linked through DYD connections to buses, lines, generators, and voltage levels.
- Combined tables `ACTIONS_voltage.csv` / `ACTIONS_spower.csv` are produced by `dataset_construction.py`.
