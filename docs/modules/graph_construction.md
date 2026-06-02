# `graph_construction.py`

Builds **PyTorch Geometric** graphs from IIDM cases: buses, generators, loads as nodes; lines, transformers, connections as edges.

## Used by

- `main.py` (via `src/build_op_assets.py`, `src/training.py`)
- `DYNAGNN.py`

## Inputs

| Source | Content |
|--------|---------|
| OP directory or `.iidm` | Network state (pre-fault) |
| Optional `.dyd` | Generator dynamic model linkage |

## Outputs

Saved bundle (via caller):

```python
{"data": Data, "metadata": dict}
```

- `data.x` — node features (type, V, angle, P, Q, fault flag, …)
- `data.edge_index`, `data.edge_attr` — topology
- `metadata["node_metadata"]`, `metadata["edge_metadata"]` — ids, types, countries, names for masks and inference event lookup

Default path: `data/op_graphs/operating_point_N.pt`

## Main API

| Function | Description |
|----------|-------------|
| `build_graph(op_dir, dyd_path=..., compact=True)` | Main builder |
| `build_graph_from_iidm`, `build_pyg_graph_from_iidm` | Lower-level |
| `build_graph_from_directory` | Directory discovery |

## Dependencies

- **pypowsybl** — IIDM import
- **torch_geometric** — `Data` objects

## Node / edge types

Nodes: `bus`, `generator`, `load`. Edges: `line`, `transformer`, `connection` (and HVDC where present).

## Notes

`dataset_construction.py` drops KPI rows whose `Contingency` is not represented on the graph for that OP.
