# `curve_generation.py`

Builds **`fic_CRV.xml`** (Dynawo curve export list) and updates **`.jobs`** files so simulations export voltage, generator P/Q, and synchronous-machine speed curves.

## Used by

- `main.py` (via `src/simulate.py`, before contingency runs)
- `generator_snom.py` (IIDM/DYD discovery helpers)

## Inputs

Per `data/inputs/operating_point_N/`:

| File | Role |
|------|------|
| `*.iidm` / `*.xiidm` | Bus/section ids for voltage curves; generator list |
| `*.dyd` | Dynamic model types (sync vs async generators) |

## Outputs

- `fic_CRV.xml` in each OP folder
- In-place edit of `*.jobs` → `<curves inputFile="fic_CRV.xml" exportMode="XML"/>` (unprefixed jobs) or `<dyn:curves inputFile="fic_CRV.xml" exportMode="XML"/>` (prefixed `dyn:` jobs)

## Main API

| Function | Description |
|----------|-------------|
| `generate_curves_for_operating_point(op_dir)` | One OP |
| `generate_curves(op_start=..., op_end=..., op_numbers=...)` | Batch under `data/inputs/` |
| `find_iidm_file`, `find_dyd_file` | Shared discovery in the OP folder |

## Notes

Without this step, Dynawo may not write `outputs/curves/curves.xml`, and KPI extraction will skip contingencies.
