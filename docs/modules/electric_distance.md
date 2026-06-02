# `electric_distance.py`

Computes **electrical distance** between network elements from the IIDM Y-bus (voltage-level or substation aggregation).

## Used by

- `main.py` (via `src/build_op_assets.py`)
- `main.py` (via `src/training.py`, appends `dz_fault` feature)
- `DYNAGNN.py`
- `DYNAGNN.py`

## Inputs

| Source | Content |
|--------|---------|
| `data/inputs/operating_point_N/` or `.iidm` path | Network case |
| Optional precomputed tables | — |

## Outputs

- `data/op_electric_distance/operating_point_N.csv` — pairs `(VLi, VLj, dij)` or substation columns
- Inference: `dynagnn_output/electrical_distance.csv`

## Main API

| Function | Description |
|----------|-------------|
| `write_electrical_distance_csv_from_iidm(iidm_path, output_csv)` | From file or OP directory |
| `write_electrical_distance_csv_from_network(...)` | From in-memory network |

## Dependencies

- **pypowsybl** — load IIDM
- **scipy** — sparse LU / dense solve for impedance-based distance

## Notes

Distances are joined to graph nodes during training/inference to encode fault location relative to each bus/generator.
