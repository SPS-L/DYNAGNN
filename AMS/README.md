# AMS — Adaptive Model Selection (model reduction)

Pip-installable companion to [DYNAGNN](https://github.com/SPS-L/DYNAGNN) for **node-breaker model reduction** from a scenario `.dsl` file plus network IIDM/DYD.

It loads deployment checkpoints, predicts where dynamic activity is expected, and sets IIDM switch `retained` flags so only relevant substations keep full node-breaker detail.

## Install

```bash
# From GitHub (subdirectory package)
python3 -m pip install "dynagnn-ams @ git+https://github.com/SPS-L/DYNAGNN.git#subdirectory=AMS"

# Or editable from a local clone
python3 -m pip install -e /path/to/DYNAGNN/AMS
```

Dependencies (`torch`, `torch-geometric`, `pypowsybl`, …) are installed automatically. First install can take a few minutes.

**Python 3.10 is required** (same as DYNAGNN). Python 3.11+ / 3.13 often segfault with these native stacks.

Bundled Nordic checkpoints ship inside the package (`dynagnn_ams/models/Nordic/`).

## CLI

```bash
dynagnn-ams <scenario.dsl> <network.xiidm> <network.dyd> \
  --network Nordic --epsilon 1 --device mps
```

| Argument | Description |
|----------|-------------|
| `dsl_path` | Scenario `.dsl` |
| `iidm_path` | Network `.iidm` / `.xiidm` (**modified in place**) |
| `dyd_path` | Dynamic models `.dyd` |
| `--network`, `-n` | Subfolder under packaged `models/` (e.g. `Nordic`) |
| `--models-dir` | Optional override root containing `<network>/` checkpoints |
| `--epsilon` | Retain switches where predicted class ≥ ε (default `1.0`) |
| `--device` | `auto` (default), `cpu`, `mps`, `cuda`, or `cuda:N` |
| `--json [PATH]` | Optional export of DSL location lists |

On Apple Silicon, use `--device mps` (default PyPI torch already includes MPS). For NVIDIA, install a CUDA torch build (TwinEU `ams.sh` does this when `--device cuda`).

## Programmatic API

```python
from dynagnn_ams import run

action_locations, events_list, substation_predictions = run(
    "scenario.dsl",
    "network.xiidm",
    "network.dyd",
    network="Nordic",
    epsilon=1.0,
    device="mps",
)
```

## Pipeline

1. **DSL reader** — `action_locations` and `events_list`
2. **Base graph** — PyG graph from IIDM + DYD
3. **Electric distance** — `log1p(dZ_fault)`
4. **Event graphs** — one graph per DSL event
5. **Pair-aware GINE** — voltage / spower → substation max aggregate
6. **Node-breaker simplification** — `retained` flags on switches

## Layout

```
AMS/
├── pyproject.toml          # pip package dynagnn-ams → CLI `dynagnn-ams`
├── README.md
└── dynagnn_ams/
    ├── cli.py              # sole entry (console script)
    ├── modules/
    └── models/
        └── Nordic/
            ├── voltage_best_model.pt
            ├── spower_best_model.pt
            ├── x_scaler.pkl
            └── edge_attr_scaler.pkl
```

## Related docs

- [`docs/HowTo.md`](../docs/HowTo.md) — AMS section in DYNAGNN
- [`docs/src/training.md`](../docs/src/training.md) — how checkpoints are produced
