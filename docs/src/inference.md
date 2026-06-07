# `DYNAGNN.py`

**Inference** on a single operating point: optional Dynawo initialization, graph build, per-scenario fault injection, and GAT voltage / spower predictions.

Not part of `main.py`; run standalone after training.

## Command

```bash
python3 DYNAGNN.py --case-dir /path/to/operating_point --events-csv /path/to/events.csv
```

## Inputs

| Source | Content |
|--------|---------|
| `--case-dir` | One OP folder (IIDM, `.dyd`, `.jobs`, …) |
| `--events-csv` | One row per scenario (see below) |
| `config.yaml` | `data.path`, `model.num_classes`, `network.country_filter`, `inference.initialization_duration`, `dynawo.path` |
| `<data.path>/model/` | `gat_voltage_best_model.pt`, `gat_spower_best_model.pt`, scalers, hparams JSON |

## `events.csv`

| Column | Description |
|--------|-------------|
| `scenario_id` | Integer label (output subfolder `scenario_<id>/`) |
| `Event` | Fault component id on the graph |

Accepted column names for the event field (case-insensitive): `Event`, `event`, `component`, `component_name`, `contingency`, `event_id`.

Use the **same id namespace** as training: `contingencies.csv` **Fault name** and column 2 of `Dataset_Voltage.csv` / `Dataset_Spower.csv`.

Example:

| scenario_id | Event |
|-------------|-------|
| `1` | `L1011-1013a` |
| `2` | `1011_131` |

## Outputs

Under `<case-dir>/dynagnn_output/`:

| Path | Content |
|------|---------|
| `electrical_distance.csv` | Pairwise electrical distances for this case |
| `scenario_<id>/prediction_voltage.csv` | Bus-level predicted severity class |
| `scenario_<id>/prediction_spower.csv` | Generator-level predicted severity class |

## Flow (summary)

1. Optional **initialization** when `inference.initialization_duration` > 0 (updates IIDM in place).
2. Compute **electrical distance** CSV from the case IIDM.
3. **Build graph** (`graph_construction.build_graph`, compact).
4. Load **scalers** and **GAT models** from `<data.path>/model/`.
5. For each `events.csv` row: clone the base graph, resolve the event, set `fault_on`, append `dz_fault`, scale features, run voltage and spower forward passes.

## Event lookup and `fault_on` placement

Each scenario’s **Event** id is resolved on the graph via `event_lookup` (built from graph metadata). `_find_event_location` returns a location; `_inject_single_event_fault` sets `fault_on = 1.0`:

| Event id matches | Location | `fault_on` set on |
|----------------|----------|-------------------|
| Node `id` (voltage level) | node | `data.x[..., fault_on]` |
| `busbarSectionIds` entry (NODE_BREAKER) | node | `data.x[..., fault_on]` |
| `busIds` entry (BUS_BREAKER) | node | `data.x[..., fault_on]` |
| Edge `id` (line, transformer, HVDC, connection, …) | edge | `data.edge_attr[..., fault_on]` on **both** directed half-edges |

Edge endpoint fields (`bus1`, `bus2` in edge metadata) are **not** used for event resolution — only equipment **edge `id`**.

Ids are matched exactly when possible, then via canonical normalization and safe substring fallbacks. If the id is not found, inference raises an error for that scenario.

After `fault_on` is set, **log electrical distance** (`dz_fault`) is appended from the fault anchors to every node, then the same scalers as training are applied before the model forward pass.

Training uses the same lookup and fault rules — see [`training.md`](training.md#event-lookup-and-fault_on-placement).

## Related modules

- [`graph_construction`](../modules/graph_construction.md), [`electric_distance`](../modules/electric_distance.md), [`initialization`](../modules/initialization.md), [`dynawo_runner`](../modules/dynawo_runner.md), [`gat_voltage_training`](../modules/gat_voltage_training.md), [`gat_spower_training`](../modules/gat_spower_training.md)
