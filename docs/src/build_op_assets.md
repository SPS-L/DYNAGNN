# `src/build_op_assets.py`

Builds **static per-operating-point assets** from pre-fault inputs (no Dynawo simulation required).

## Invoked by

- `main.py` (second pipeline stage)

## Inputs

| Source | Content |
|--------|---------|
| `data/inputs/operating_point_*/` | IIDM and `.dyd` |

## Outputs

| Path | Content |
|------|---------|
| `data/op_graphs/operating_point_N.pt` | PyG graph + metadata bundle |
| `data/op_electric_distance/operating_point_N.csv` | Pairwise electrical distances |
| `data/generator_Snom/operating_point_N.csv` | Generator nominal power table |

## Main entry point

| Function | Description |
|----------|-------------|
| `main()` | Build assets for all `operating_point_*` folders under `--inputs` (default `data/inputs/`) |

## CLI flags (when called programmatically with `argparse`)

| Flag | Purpose |
|------|---------|
| `--skip-electrical-distance` | Skip distance CSVs |
| `--skip-generator-snom` | Skip SNom CSVs |
| `--skip-existing-graphs` | Skip existing `.pt` files |
| `--skip-existing-electric-distance` | Skip existing distance CSVs |
| `--skip-existing-generator-snom` | Skip existing SNom CSVs |
| `--verbose` | Print node/edge type counts |
| `--show-examples` | Print one example metadata record per type |

## Related modules

- [`graph_construction`](../modules/graph_construction.md), [`electric_distance`](../modules/electric_distance.md), [`generator_snom`](../modules/generator_snom.md)
