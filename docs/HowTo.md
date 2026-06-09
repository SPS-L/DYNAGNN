# How-to (inputs, config, Nordic example details)

This page keeps the longer, technical setup notes so the project-root [`README.md`](../README.md) can stay short.

---

## Data folder and `config.yaml`

1. In `config.yaml`, set **`data.path`** to the folder where DYNAGNN should keep all data (inputs and everything produced by `main.py`).

2. Create **`inputs/`** under that folder and add your cases:

```
<data.path>/
└── inputs/
    ├── contingencies.csv
    ├── operating_point_1/
    ├── operating_point_2/
    └── …
```

Each `operating_point_<N>/` folder must contain the Dynawo case (`*.iidm` or `*.xiidm`, `*.dyd`, `*.jobs`, `*.par`, and optionally `*.crt`).

3. Fill in **`config.yaml`** completely before running the pipeline (all sections below).

---

## `contingencies.csv`

Path: `<data.path>/inputs/contingencies.csv`

| Column | Description |
|--------|-------------|
| Contingency ID | e.g. `1l`, `2b`, `1g`, `1t`, `1lo` (type letter: `l` line, `b` bus, `g` generator, `t` transformer, `lo` load) |
| Fault name | Equipment id (see notes below per type) |
| Type | `line`, `bus`, `generator`, `transformer`, or `load` |
| Operating point | Optional; comma-separated OP indices (e.g. `38,39`), or empty for all OPs |

**Fault name by type**

| Type | Fault name | Dynawo event |
|------|------------|--------------|
| `line` | IIDM line id | `EventQuadripoleDisconnection` |
| `transformer` | IIDM two-winding transformer id | `EventQuadripoleDisconnection` (same as line) |
| `bus` | IIDM `bus` id (bus-breaker) or `busbarSection` id (node-breaker) | `EventConnectedStatus` |
| `load` | IIDM load id | `EventConnectedStatus` (same as bus) |
| `generator` | **Dynamic model id** from the case `.dyd`, not the IIDM static id | `EventSetPointBoolean` |

Example (for **bus**, use **Type** `bus` in both cases; **Fault name** is a `bus` id in bus-breaker models or a `busbarSection` id in node-breaker models):

| Contingency ID | Fault name | Type | Operating point |
|----------------|------------|------|-----------------|
| `1l` | `LINE_EXAMPLE_1` | `line` | |
| `1b` | `BUS_EXAMPLE_1` | `bus` | |
| `2b` | `BBS_EXAMPLE_1` | `bus` | |
| `1g` | `GEN_DM_EXAMPLE_1` | `generator` | |
| `1t` | `TRAFO_EXAMPLE_1` | `transformer` | |
| `1lo` | `LOAD_EXAMPLE_1` | `load` | "34, 38"|

`1b` / `BUS_EXAMPLE_1` — bus-breaker (`bus` id). `2b` / `BBS_EXAMPLE_1` — node-breaker (`busbarSection` id).

---

## `config.yaml` reference

| Section | Key | Options | Purpose |
|---------|-----|---------|---------|
| **dynagnn** | `version` | string | Log header version |
| **dynawo** | `path` | path | Dynawo env script or install path |
| **data** | `path` | path | Data root (`inputs/` and all pipeline outputs) |
| **simulation** | `event_time` | float (s) | Fault time |
| | `initialization_duration` | float (s), or `0` / omit | Steady-state run before contingencies |
| **network** | `country_filter` | list, or `[]` | Country codes to keep; empty = no filter |
| **kpi** | `window_sec` | float (s) | KPI window length |
| | `step_sec` | float (s) | KPI window step |
| | `class_bins.voltage.cuts` | list of floats | Voltage class boundaries |
| | `class_bins.spower.cuts` | list of floats | Spower class boundaries |
| **model** | `num_classes` | integer ≥ 2 | Severity levels |
| **training** | `epochs` | integer | Max epochs per trial |
| | `patience` | integer | Early stopping |
| | `batch_size` | integer | Batch size |
| | `split_mode` | `scenario`, `operating_point` | How train/val/test split is built |
| | `seed` | integer | Random seed |
| | `training` | float | Train fraction or OP count |
| | `validation` | float | Validation fraction or OP count |
| | `testing` | float | Test fraction or OP count |
| | `high_class_threshold` | integer or `null` | Minimum class for **high** severity (sampling, under-penalty, composite score); `null` = top class only for metrics, no weighted sampling |
| | `selection_f1_weight` | float (**required**) | Weight on `high_f1` in checkpoint / Optuna composite score |
| | `selection_loss_weight` | float (**required**) | Weight on `loss` (subtracted) in composite score |
| **optuna** | `n_trials` | integer | Hyperparameter trials |
| | `hparams.*` | see `config.yaml` | Optuna search spaces (`categorical`, `int`, `float`) |
| **inference** | `initialization_duration` | float (s), or `0` / omit | Steady-state run for `DYNAGNN.py` |

---

## Resuming the pipeline (`--from-step`)

`main.py` runs five stages in order:

1. **`simulate`** — optional per-OP **initialization**, curve export setup, and Dynawo **contingency runs** (`src/simulate.py`)
2. **`build_op_assets`** — graphs, electrical distance, generator SNom (`src/build_op_assets.py`)
3. **`curve_process`** — KPI tables and action/disconnection flags (`src/curves_post_process.py`)
4. **`dataset`** — merged datasets and class labels (`src/dataset_construction.py`)
5. **`training`** — GAT training and model export (`src/training.py`)

Omit `--from-step` to run the **full pipeline** (including initialization and contingency simulations). To **rerun from a later stage** (for example after fixing config or code downstream of simulations), pass `--from-step`:

```bash
# Full run (default)
python3 main.py

# Skip initialization and contingency simulations; rebuild graph assets and continue
python3 main.py --from-step build_op_assets

# Skip simulations and graph assets; start at curve/KPI post-processing
python3 main.py --from-step curve_process

# Rebuild datasets only (KPI/Actions/Disconnections must already exist)
python3 main.py --from-step dataset

# Retrain models only (dataset CSVs and graph assets must already exist)
python3 main.py --from-step training
```

**Prerequisites:** each start point assumes the **outputs of all earlier stages** are already present under `<data.path>/`. If required files are missing, the step will fail.

| `--from-step` | Skips | You must already have |
|---------------|-------|------------------------|
| *(omit — full run)* | — | `inputs/` cases and `contingencies.csv` |
| `build_op_assets` | `simulate` (initialization + contingency runs) | `inputs/operating_point_*` (IIDM, `.dyd`, …) |
| `curve_process` | `simulate` (initialization + contingency runs), `build_op_assets` | `Simulations_Scenarios/` with Dynawo `outputs/curves/`, `generator_Snom/`, `inputs/contingencies.csv` |
| `dataset` | through `curve_process` | `KPI/`, `Actions/`, `Disconnections/`, `op_graphs/` |
| `training` | through `dataset` | `Dataset/Dataset_Voltage.csv`, `Dataset_Spower.csv`, `op_graphs/`, `op_electric_distance/` |

See the per-stage input tables in [`src/simulate.md`](src/simulate.md), [`src/build_op_assets.md`](src/build_op_assets.md), [`src/curves_post_process.md`](src/curves_post_process.md), [`src/dataset_construction.md`](src/dataset_construction.md), and [`src/training.md`](src/training.md).

Each `main.py` run still recreates `<data.path>/dynagnn.log` from scratch.

---

## Nordic example details

### Operating points in the Nordic example

| Operating point | Description |
|----------------|-------------|
| **1** | Original (from Dynawo `DynaWaltz` repo) |
| **2** | 10% decrease |
| **3** | 20% decrease |
| **4** | 30% decrease |
| **5** | 35% decrease |
| **6** | 40% decrease |
| **7** | 50% decrease |
| **8** | 55% decrease |
| **9** | 60% decrease |
| **10** | 70% decrease |

### Nordic smoke-test `config.yaml` (written by `Nordic_test_setup.py`)

Run:

```bash
cd "/absolute/path/to/DYNAGNN"
python3 Nordic_test_setup.py --dynawo-env "/absolute/path/to/myEnvDynawo.sh" --force
```

The following is **exactly what the script writes** to `config.yaml` (shown here so you can review the values used):

```yaml
# Configuration for DYNAGNN scripts.
#
# Quick smoke test defaults for the bundled Nordic example (examples/Nordic/data).
# Before running main.py, set dynawo.path and data.path to absolute paths on your machine.

dynagnn:
  version: 1

dynawo:
  path: "/absolute/path/to/myEnvDynawo.sh"

data:
  path: "/absolute/path/to/DYNAGNN/examples/Nordic/data"

simulation:
  event_time: 10.0
  initialization_duration: 10.0  # steady-state init per OP before contingencies; use 0 to skip

network:
  country_filter: []

kpi:
  window_sec: 5.0
  step_sec: 1.0
  class_bins:
    voltage:
      cuts: [0.33, 0.66]  # 3 severity bins on [0, 1] + 1 flag class => model.num_classes: 4
    spower:
      cuts: [0.33, 0.66]

model:
  num_classes: 4

training:
  epochs: 30          # keep low for a quick smoke test
  patience: 8
  batch_size: 16
  split_mode: operating_point
  seed: 42
  training: 0.8
  validation: 0.1
  testing: 0.1
  high_class_threshold: 2  # classes >= 2 (2 and 3) are "high" with num_classes: 4
  selection_f1_weight: 0.5
  selection_loss_weight: 0.1

optuna:
  n_trials: 5
  hparams:
    hidden_dim:
      type: categorical
      choices: [64, 128, 256]
    num_layers:
      type: int
      low: 2
      high: 4
    hidden_channels:
      type: categorical
      choices: [16, 32, 64]
    num_heads:
      type: categorical
      choices: [1, 2, 4, 8]
    dropout:
      type: float
      low: 0.1
      high: 0.5
    num_gnn_layers:
      type: int
      low: 2
      high: 4
    lr:
      type: float
      low: 1.0e-4
      high: 5.0e-3
      log: true
    weight_decay:
      type: float
      low: 1.0e-6
      high: 1.0e-3
      log: true
    under_penalty_lambda:
      type: float
      low: 0.0
      high: 2.0
    coral_prediction_threshold:
      type: float
      low: 0.3
      high: 0.7

inference:
  initialization_duration: 10.0  # steady-state run before graph build; use 0 to skip
```

With `model.num_classes: 4` (classes 0–3), `high_class_threshold: 2` treats classes **2 and 3** as high severity. That enables weighted train sampling and the CORAL under-penalty (when Optuna picks `under_penalty_lambda > 0`), and sets the cutoff for `high_recall` / `high_f1` in the validation composite score. See [`src/training.md`](src/training.md) for details.

