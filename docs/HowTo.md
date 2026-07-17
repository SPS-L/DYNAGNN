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

**Solver precision:** When you create or duplicate operating points, ensure every `*.jobs` file uses the **same solver precision** across all OPs (e.g. identical `precision` / `iidmImport/export` settings in the Dynawo job XML). Mixed precision between OPs can change numerical KPI values enough to shift class labels at the raw cut thresholds.

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
| **dynagnn** | `version` | string | Log header version (current: `1.2`) |
| **dynawo** | `path` | path | Dynawo env script or install path |
| **data** | `path` | path | Data root (`inputs/` and all pipeline outputs) |
| **simulation** | `event_time` | float (s) | Fault time |
| | `initialization_duration` | float (s), or `0` / omit | Steady-state run before contingencies |
| **network** | `country_filter` | list, or `[]` | Country codes to keep; empty = no filter |
| **kpi** | `window_sec` | float (s) | KPI window length |
| | `step_sec` | float (s) | KPI window step |
| | `class_bins.voltage.cuts` | list of positive floats (ascending) | Raw KPI cut thresholds for voltage class labels |
| | `class_bins.spower.cuts` | list of positive floats (ascending) | Raw KPI cut thresholds for spower class labels |
| **model** | `num_classes` | integer ≥ 2 | Must equal `len(cuts) + 2` (KPI bins + flag class) |
| **training** | `epochs` | integer | Max epochs per Optuna trial |
| | `patience` | integer | Early stopping |
| | `batch_size` | integer | Batch size |
| | `split_mode` | `scenario`, `operating_point` | How train/val/test split is built |
| | `seed` | integer | Random seed |
| | `training` | float | Train fraction or OP count |
| | `validation` | float | Validation fraction or OP count |
| | `testing` | float | Test fraction or OP count |
| | `pair_aware.*` | see below | Fixed loss / decoding settings (not Optuna-tuned) |
| **optuna** | `n_trials` | integer | Hyperparameter trials per task (Voltage and Spower are independent) |
| | `study_name` | string (**required**) | Folder name for `data/training/<study_name>/` and `data/model/<study_name>/` |
| | `hparams.*` | see `config.yaml` | Search spaces for model capacity + optimizer |
| **inference** | `initialization_duration` | float (s), or `0` / omit | Steady-state run for `DYNAGNN.py` |

### `training.pair_aware` (fixed)

These keys are **not** Optuna-tuned. They fix how the training objective is built and how predictions are decoded / selected.

Each forward pass produces three heads: class logits, an inactive (class-0) gate logit, and a standardized log-KPI prediction. The scalar loss minimized by SGD is

$$
\mathcal{L}
=
w_{\mathrm{cls}}\,\mathcal{L}_{\mathrm{CE}}
+
w_{\mathrm{reg}}\,\mathcal{L}_{\mathrm{Huber}}
+
w_{\mathrm{gate}}\,\mathcal{L}_{\mathrm{BCE}}
+
w_{\mathrm{ord}}\,\mathcal{L}_{\mathrm{CDF}}
$$

where the config keys map as:

| Config key | Symbol | Term |
|------------|--------|------|
| `classification_weight` | $w_{\mathrm{cls}}$ | Multi-class cross-entropy $\mathcal{L}_{\mathrm{CE}}$ on all labels $0,\ldots,C-1$ ($C =$ `model.num_classes`) |
| `regression_weight` | $w_{\mathrm{reg}}$ | Smooth L1 (Huber) $\mathcal{L}_{\mathrm{Huber}}$ on standardized $\log_{10}(\mathrm{KPI}+\varepsilon)$; finite KPI targets only (flag class has none) |
| `inactive_gate_weight` | $w_{\mathrm{gate}}$ | Binary cross-entropy $\mathcal{L}_{\mathrm{BCE}}$ of the gate vs “true class is 0” |
| `ordinal_weight` | $w_{\mathrm{ord}}$ | Ordinal CDF consistency $\mathcal{L}_{\mathrm{CDF}}$ on the class logits |

The other `pair_aware` keys do **not** enter $\mathcal{L}$ as scalar multipliers:

| Key | Where it goes |
|-----|----------------|
| `class_weight_mode` | Builds per-class weights inside $\mathcal{L}_{\mathrm{CE}}$ (e.g. `inverse`, `sqrt_inverse`) |
| `gate_pos_weight_mode` | Builds the positive-class weight inside $\mathcal{L}_{\mathrm{BCE}}$ (e.g. `balanced`) |
| `epsilon` | $\varepsilon$ in $\log_{10}(\mathrm{KPI}+\varepsilon)$ when forming regression targets (and when inverting log-KPI decode) |
| `gate_threshold` | Decode only: treat as class 0 when $\sigma(\mathrm{gate})\ge$ threshold (`gated` path) |
| `selection_output` | Which decode path is scored for checkpoint / Optuna selection: `auto`, `class`, `gated`, or `log_kpi` |

**Example (Nordic defaults from `Nordic_test_setup.py`):** $w_{\mathrm{cls}}=1.0$, $w_{\mathrm{reg}}=0.30$, $w_{\mathrm{gate}}=0.15$, $w_{\mathrm{ord}}=0.15$, `class_weight_mode: inverse`, `epsilon: 1.0e-10`, `selection_output: class`.

### `optuna.hparams` (tuned)

`hidden_dim`, `node_id_dim`, `contingency_id_dim`, `type_dim`, `pair_dim`, `num_gnn_layers`, `decoder_hidden_dim`, `dropout`, `lr`, `weight_decay`.

---

## KPI cut thresholds — recommendations

DYNAGNN turns **continuous** KPI values (sliding-window variance of post-contingency curves) into **ordinal** severity classes. Each band should represent **increasing dynamic behavior**: class 0 is the least active; higher classes capture progressively stronger responses.

### Choose cuts on the training set only

Cut thresholds define what “mild” vs “severe” means for your problem. **Perform this analysis only on training scenarios** — never on validation or test rows. Using val/test KPIs to pick bins leaks split information into the labels and invalidates evaluation.

Suggested workflow:

1. Run the pipeline through **`curve_process`** and stop there:

```bash
python3 main.py --to-step curve_process
```

2. Use these files under `<data.path>/` (all produced by `curve_process`):

| File | Role |
|------|------|
| `KPI/KPI_voltage.csv` | Combined raw voltage KPI values (flagged cells masked with `NaN`) |
| `KPI/KPI_spower.csv` | Combined raw spower KPI values (flagged cells masked with `NaN`) |
| `Dataset/train_val_test_split.csv` | Train / validation / test assignment per scenario (`split`, `operating_point`, `contingency`) |

3. Join the KPI tables with `train_val_test_split.csv` and **keep only rows where `split` is `train`**. Inspect the training KPI distribution (histograms, percentiles, class counts after tentative cuts).
4. Set `kpi.class_bins.voltage.cuts` and `kpi.class_bins.spower.cuts` in `config.yaml`, then run the remaining stages:

```bash
python3 main.py --from-step dataset
```

(or `python3 main.py --from-step dataset --to-step training` if you only want dataset + training).

### Recommended first cut (class 0 → class 1)

**Class 0** should be the **inactive / numerical-noise** band: components whose post-contingency KPI is effectively at the solver noise floor.

A practical lower bound for that floor is the **maximum variance of numerical noise**, which scales as **k²**, where **k** is the **solver precision** declared in your `*.jobs` files (see [Solver precision](#data-folder-and-configyaml) above). Use the **first cut** τ₁ so that class 0 is KPI ≤ τ₁ and class 1 starts above τ₁:

| Class | Meaning |
|-------|---------|
| 0 | KPI ≤ τ₁ — inactive / noise-dominated |
| 1 | τ₁ < KPI ≤ τ₂ |
| 2+ | progressively stronger dynamics |

Set τ₁ to **at least k²** so class 0 is not filled with numerical artifacts. You may raise τ₁ slightly after training-set analysis if a strict k² threshold leaves class 0 too sparse or too large.

**Example (Nordic):** solver precision is **k = 10⁻⁴**, so **k² = 10⁻⁸**. On the Nordic **training** set, τ₁ = 10⁻⁶ for both voltage and spower (`cuts: [1e-6, 2.25e-5, 3e-4, 5.625e-4]`) — above k² so class 0 remains noise-dominated while higher cuts follow the training KPI distribution.

Higher cuts (τ₂, τ₃, …) have no universal default: choose them from **training-set** percentiles or domain reasoning so each class reflects meaningfully stronger dynamics than the previous one.

---

## Pipeline control (`--from-step`, `--to-step`)

`main.py` runs five stages in order:

1. **`simulate`** — optional per-OP **initialization**, curve export setup, and Dynawo **contingency runs** (`src/simulate.py`)
2. **`build_op_assets`** — graphs, electrical distance, generator SNom (`src/build_op_assets.py`)
3. **`curve_process`** — per-OP KPI/flag tables, combined KPI CSVs, and train/val/test split (`src/curves_post_process.py`)
4. **`dataset`** — class labels from configured KPI cuts (`src/dataset_construction.py`)
5. **`training`** — pair-aware GINE Optuna training and model export (`src/training.py`)

Omit both flags for a **full run**. Use **`--from-step`** to resume from a later stage; use **`--to-step`** to stop after a stage (inclusive).

```bash
# Full run (default)
python3 main.py

# Run through curve_process only (for KPI cut analysis)
python3 main.py --to-step curve_process

# Skip initialization and contingency simulations; rebuild graph assets and continue
python3 main.py --from-step build_op_assets

# Skip simulations and graph assets; start at curve/KPI post-processing
python3 main.py --from-step curve_process

# Rebuild class-label datasets only (combined KPI/split CSVs must already exist)
python3 main.py --from-step dataset

# Retrain models only (dataset CSVs and graph assets must already exist)
python3 main.py --from-step training

# Dataset + training only (after setting KPI cuts)
python3 main.py --from-step dataset --to-step training
```

**Prerequisites:** each start point assumes the **outputs of all earlier stages** are already present under `<data.path>/`. If required files are missing, the step will fail.

| `--from-step` | Skips | You must already have |
|---------------|-------|------------------------|
| *(omit — full run)* | — | `inputs/` cases and `contingencies.csv` |
| `build_op_assets` | `simulate` (initialization + contingency runs) | `inputs/operating_point_*` (IIDM, `.dyd`, …) |
| `curve_process` | `simulate` (initialization + contingency runs), `build_op_assets` | `Simulations_Scenarios/` with Dynawo `outputs/curves/`, `generator_Snom/`, `inputs/contingencies.csv`, `op_graphs/` |
| `dataset` | through `curve_process` | `KPI/KPI_voltage.csv`, `KPI/KPI_spower.csv`, `Dataset/train_val_test_split.csv`, `Actions/ACTIONS_*.csv`, `Disconnections/DISC_*.csv` |
| `training` | through `dataset` | `Dataset/Dataset_Voltage.csv`, `Dataset_Spower.csv`, `Dataset/train_val_test_split.csv`, `op_graphs/`, `op_electric_distance/` |

| `--to-step` | Stops after | Key outputs for cut analysis / next stage |
|-------------|-------------|-------------------------------------------|
| `curve_process` | Combined KPI tables + split | `KPI/KPI_voltage.csv`, `KPI/KPI_spower.csv`, `Dataset/train_val_test_split.csv` |
| `dataset` | Class-label datasets | `Dataset/Dataset_Voltage.csv`, `Dataset/Dataset_Spower.csv`, `Dataset/KPI_class_bins.csv` |
| `training` | Trained models | `model/<study_name>/voltage_best_model.pt`, `model/<study_name>/spower_best_model.pt`, `training/<study_name>/<task>/plots/` |

See the per-stage input tables in [`src/simulate.md`](src/simulate.md), [`src/build_op_assets.md`](src/build_op_assets.md), [`src/curves_post_process.md`](src/curves_post_process.md), [`src/dataset_construction.md`](src/dataset_construction.md), and [`src/training.md`](src/training.md).

Each `main.py` run still recreates `<data.path>/dynagnn.log` from scratch.

---

## Nordic example details

### Operating points in the Nordic example

| Operating point | Total generator Power (MW) |
|----------------|-----------------------------|
| **1** | 11506.0 |
| **2** | 10398.3 |
| **3** | 9359.5 |
| **4** | 8309.6 |
| **5** | 7783.4 |
| **6** | 7241.1 |
| **7** | 6169.1 |
| **8** | 5658.8 |
| **9** | 5207.7 |

### Nordic `config.yaml` (written by `Nordic_test_setup.py`)

Run:

```bash
cd "/absolute/path/to/DYNAGNN"
python3 Nordic_test_setup.py --dynawo-env "/absolute/path/to/myEnvDynawo.sh" --force
```

The following is **exactly what the script writes** to `config.yaml` (shown here so you can review the values used):

```yaml
# Configuration for DYNAGNN scripts.
#
# Nordic example defaults (examples/Nordic/data).
# Before running main.py, set dynawo.path and data.path to absolute paths on your machine.

dynagnn:
  version: 1.2

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
      cuts: [1e-6, 2.25e-5, 3e-4, 5.625e-4]  # 5 KPI classes (0-4) + 1 flag class => model.num_classes: 6
    spower:
      cuts: [1e-6, 2.25e-5, 3e-4, 5.625e-4]

model:
  num_classes: 6

training:
  epochs: 150
  patience: 20
  batch_size: 16
  split_mode: operating_point
  seed: 42
  training: 0.7
  validation: 0.1
  testing: 0.2

  # Fixed loss construction and output-decoding settings.
  pair_aware:
    classification_weight: 1.0
    regression_weight: 0.30
    inactive_gate_weight: 0.15
    ordinal_weight: 0.15
    class_weight_mode: inverse
    gate_pos_weight_mode: balanced
    gate_threshold: 0.50
    epsilon: 1.0e-10
    selection_output: class

optuna:
  n_trials: 50
  study_name: nordic_v1
  hparams:
    hidden_dim:
      type: categorical
      choices: [128, 256]
    node_id_dim:
      type: categorical
      choices: [16, 24, 32]
    contingency_id_dim:
      type: categorical
      choices: [16, 32, 64]
    type_dim:
      type: categorical
      choices: [8, 16]
    pair_dim:
      type: categorical
      choices: [8, 16, 32]
    num_gnn_layers:
      type: int
      low: 3
      high: 5
    decoder_hidden_dim:
      type: categorical
      choices: [128, 256, 512]
    dropout:
      type: float
      low: 0.02
      high: 0.25
    lr:
      type: float
      low: 0.00001
      high: 0.0015
      log: true
    weight_decay:
      type: float
      low: 0.0000001
      high: 0.001
      log: true

inference:
  initialization_duration: 10.0  # steady-state run before graph build; use 0 to skip
```

With `model.num_classes = len(cuts) + 2`, the highest class index is the action/disconnection **flag** class. Voltage and Spower are tuned independently with Optuna; validation checkpoints maximize a balanced multi-class selection score (see [`src/training.md`](src/training.md)). Deployment checkpoints are written under `model/<study_name>/` as `voltage_best_model.pt` and `spower_best_model.pt`.

**Example (Nordic):** four cuts → classes 0–4 by KPI magnitude, flag class 5, `num_classes: 6`.

### KPI class bins (v1.11 labeling, used by v1.2)

See [KPI cut thresholds — recommendations](#kpi-cut-thresholds--recommendations) for how to choose cuts (training set only, noise floor at k²).

`kpi.class_bins.<type>.cuts` lists **strictly increasing raw KPI thresholds**. During dataset construction, DYNAGNN:

1. Uses raw KPI values from the combined KPI tables (no log transform or scaling).
2. Assigns class labels from the fixed cuts (KPI severity classes $0,\ldots,K$).
3. Overrides action / disconnection cells to the flag class $K+1$ (voltage and spower: actions + DISC).

Set `model.num_classes` to **`len(cuts) + 2`**. Applied cuts are recorded in `<data.path>/Dataset/KPI_class_bins.csv`.

**Example (Nordic):** `cuts: [1e-6, 2.25e-5, 3e-4, 5.625e-4]` → `num_classes: 6`.

---

## AMS — model reduction (`AMS/`)

The **`AMS/`** folder is a **standalone application** for **Adaptive Model Selection**: it uses trained DYNAGNN checkpoints to simplify node-breaker models before simulation. It is **not** invoked by `main.py` and does not use `config.yaml`.

### How it fits the repository

| Entry point | Role |
|-------------|------|
| `main.py` | Training pipeline (simulations → KPIs → checkpoints) |
| `DYNAGNN.py` | Inference on new operating points and events |
| `AMS/main.py` | Optional model reduction from TwinEU DSL (IIDM switch retention) |

Dynamic activity prediction is the core of DYNAGNN. **`AMS/`** is an optional companion module for one AMS use case; it reuses trained checkpoints but is not part of the training or standard inference scripts.

### Bundled Nordic models

Ready-to-use deployment checkpoints for the Nordic case are in **`AMS/models/Nordic/`** (voltage/spower `.pt` files and scalers).

### Checkpoints (other networks)

```bash
NETWORK=MyCase
mkdir -p "AMS/models/$NETWORK"
cp "<data.path>/model/<study_name>/voltage_best_model.pt" "AMS/models/$NETWORK/"
cp "<data.path>/model/<study_name>/spower_best_model.pt" "AMS/models/$NETWORK/"
cp "<data.path>/model/<study_name>/x_scaler.pkl" "AMS/models/$NETWORK/"
cp "<data.path>/model/<study_name>/edge_attr_scaler.pkl" "AMS/models/$NETWORK/"
```

Multiple networks can coexist (`AMS/models/Nordic/`, `AMS/models/MyCase/`, …). Select with `--network` on the CLI.

### Run

```bash
cd AMS
python3 main.py <scenario.dsl> <network.xiidm> <network.dyd> --network Nordic --epsilon 1
```

- **`--epsilon`** — retain node-breaker switches in substations whose max predicted class is ≥ this value (default `1.0`).
- **`--json`** — optionally export parsed DSL locations to JSON under `AMS/`.

The IIDM is **modified in place**. Use a copy if you need the original file.

Full reference: [`AMS/README.md`](../AMS/README.md).
