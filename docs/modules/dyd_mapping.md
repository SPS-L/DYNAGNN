# `dyd_mapping.py`

Parses Dynawo **`.dyd`** files to map **blackBoxModel** dynamic ids to **static IIDM ids** (and reverse).

## Used by

- `kpi.py`, `actions_detection.py`, `disconnections_detection.py`
- `graph_construction.py` (generator dynamic ↔ static)
- `generator_snom.py`

## Inputs

- Path to a contingency or OP `.dyd` file (XML, Dynawo namespace)

## Outputs

- In-memory dicts: `dynamic_id → static_id`, `static_id → dynamic_id`

## Main API

| Function | Description |
|----------|-------------|
| `build_dyd_id_to_staticid_map(dyd_path)` | Forward map |
| `build_static_id_to_dynamic_id_map(dyd_path)` | Reverse map |

## Notes

Required for interpreting timeline `DM_*` events and curve names as substation/voltage-level/generator labels used in KPI and flag CSV columns.
