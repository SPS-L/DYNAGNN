# `generator_snom.py`

Extracts generator **rated apparent power (SNom)** from `.par` / `.dyd` for each operating point.

## Used by

- `main.py` (via `src/build_op_assets.py`)
- `kpi.py` (normalize generator spower KPIs by SNom)

## Inputs

| Source | Content |
|--------|---------|
| `data/inputs/operating_point_N/` | IIDM + `.dyd` in the OP folder |
| Generator parameter sets in `.par` | `SNom` values linked via DYD |

## Outputs

- `data/generator_Snom/operating_point_N.csv` — static generator id, SNom (MVA), metadata columns

## Main API

| Function | Description |
|----------|-------------|
| `build_generator_snom_for_operating_point(op_dir, output_dir, dyd_path=...)` | One OP CSV |
| `build_generator_snom_tables(inputs_dir, snom_dir)` | Batch all OPs |
| `build_generator_snom_matrix(snom_dir)` | Optional matrix export (multiple OPs) |

## Notes

If SNom is missing for an OP, spower KPI computation still runs but may warn and use empty ratings.
