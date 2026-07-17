# AMS — Adaptive Model Selection (model reduction)

This folder implements **node-breaker model reduction** for TwinEU DSL scenarios: it loads deployment checkpoints, predicts where dynamic activity is expected, and sets IIDM switch `retained` flags so only relevant substations keep full node-breaker detail.

It does not run the training pipeline and does not read `config.yaml`.

**DYNAGNN** (`DYNAGNN.py`) answers: *what dynamic activity class does each component have under this contingency?*

**AMS** (this folder) answers: *given a TwinEU scenario, which node-breaker switches must stay detailed in the IIDM before we run the simulator?*

Both use the same pair-aware GINE checkpoints; the inputs and outputs differ.

## Relationship to DYNAGNN

| Entry point | Role | Typical inputs | Typical outputs |
|-------------|------|----------------|-----------------|
| [`main.py`](../main.py) | Training pipeline | `config.yaml`, `<data.path>/inputs/` | Checkpoints, scalers, KPIs |
| [`DYNAGNN.py`](../DYNAGNN.py) | Inference | OP folder, `events.csv` | `prediction_voltage.csv`, `prediction_spower.csv` |
| **`AMS/main.py`** | Model reduction | TwinEU `.dsl`, IIDM, DYD | IIDM updated in place |

Dynamic activity prediction is the core of the DYNAGNN repository. **`AMS/`** is an optional companion for one AMS workflow — switch retention driven by those predictions.

## When to use AMS

Use this module when you:

- have a TwinEU **DSL** scenario (not just a flat `events.csv`);
- need to **reduce node-breaker complexity** in the IIDM via switch `retained` attributes;
- want substations with high predicted activity (or explicit DSL actions) to keep full detail.

For per-component class CSVs on arbitrary operating points, use [`DYNAGNN.py`](../DYNAGNN.py) instead ([`docs/src/inference.md`](../docs/src/inference.md)).

## Layout

```
AMS/
├── main.py
├── modules/
│   ├── DSL_reader.py
│   ├── base_graph_construction.py
│   ├── electric_distance.py
│   ├── event_graph_construction.py
│   ├── pair_aware_gine.py
│   ├── pair_aware_models.py
│   └── node_breaker_simplification.py
└── models/                          # One subfolder per network
    └── Nordic/
        ├── voltage_best_model.pt
        ├── spower_best_model.pt
        ├── x_scaler.pkl
        └── edge_attr_scaler.pkl
```

Each network name (e.g. `Nordic`) maps to `models/<network>/`. Select with `--network` on the CLI.

## Bundled Nordic models

Ready-to-use Nordic deployment checkpoints are shipped under **`models/Nordic/`** (pair-aware GINE models trained on the bundled [`examples/Nordic`](../examples/Nordic) case):

| File | Description |
|------|-------------|
| `voltage_best_model.pt` | Voltage deployment checkpoint |
| `spower_best_model.pt` | Spower deployment checkpoint |
| `x_scaler.pkl` | Node feature scaler |
| `edge_attr_scaler.pkl` | Edge feature scaler |

For other networks, copy the same four files from `<data.path>/model/<study_name>/` after [`main.py`](../main.py) training.

Checkpoints must be DYNAGNN v1.2 pair-aware GINE bundles (`model_type="pair_aware_gine"`). Legacy GAT weights are not supported. The optional `*_best_hparams.json` files are not required.

```bash
NETWORK=MyCase
SRC=/path/to/<data.path>/model/<study_name>
DST=/path/to/DYNAGNN/AMS/models/$NETWORK
mkdir -p "$DST"
cp "$SRC/voltage_best_model.pt" "$SRC/spower_best_model.pt" "$SRC/x_scaler.pkl" "$SRC/edge_attr_scaler.pkl" "$DST/"
```

Use IIDM, DYD, and DSL for the **same network family** as the checkpoints.

## CLI

```bash
cd AMS
python3 main.py <scenario.dsl> <network.xiidm> <network.dyd> --network Nordic --epsilon 1
python3 main.py <scenario.dsl> <network.xiidm> <network.dyd> -n MyCase --json
```

| Argument | Description |
|----------|-------------|
| `dsl_path` | TwinEU scenario `.dsl` |
| `iidm_path` | Network `.iidm` / `.xiidm` (**modified in place**) |
| `dyd_path` | Dynamic models `.dyd` |
| `--network`, `-n` | Subfolder under `models/` (e.g. `Nordic`) |
| `--epsilon` | Retain switches where predicted class ≥ ε (default `1.0`) |
| `--json [PATH]` | Optional export of DSL location lists |

## Programmatic API

```python
import sys
from pathlib import Path

AMS_DIR = Path("/path/to/DYNAGNN/AMS")
sys.path.insert(0, str(AMS_DIR))

from main import run

action_locations, events_list, substation_predictions = run(
    "scenario.dsl",
    "network.xiidm",
    "network.dyd",
    network="Nordic",
    epsilon=1.0,
)
```

## Pipeline

1. **DSL reader** — `action_locations` (all referenced components) and `events_list` (open switch / line ids only).
2. **Base graph** — PyG graph from IIDM + DYD (same feature schema as DYNAGNN).
3. **Electric distance** — voltage-level distances for `log1p(dZ_fault)`.
4. **Event graphs** — one graph per DSL event; fault masks and pair-aware identity tensors.
5. **Pair-aware GINE** — voltage (buses) and spower (generators) predictions; max-aggregate to substation level.
6. **Node-breaker simplification** — all switches `retained="false"`, then `retained="true"` where the substation contains a component in `action_locations` or max predicted class ≥ `--epsilon`.

The IIDM is updated **in place**. Work on a copy if you need the original.

## Assumptions

- Runs **before** dynamic simulation, on the **initial OP** IIDM topology.
- DSL timing (`at`, `when` thresholds, etc.) is not evaluated over time — only static location extraction.
- Inference masks target country **FR** (`event_graph_construction._attach_inference_masks`). Adapt the module if you train with a different region filter.
- Component and contingency ids in the DSL/IIDM must appear in the checkpoint vocabularies.

## Related docs

- [`docs/src/training.md`](../docs/src/training.md) — how checkpoints are produced
- [`docs/src/inference.md`](../docs/src/inference.md) — `DYNAGNN.py` prediction flow
- [`docs/modules/pair_aware_inference.md`](../docs/modules/pair_aware_inference.md) — checkpoint format and decode paths
