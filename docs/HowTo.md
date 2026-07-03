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
| **dynagnn** | `version` | string | Log header version |
| **dynawo** | `path` | path | Dynawo env script or install path |
| **data** | `path` | path | Data root (`inputs/` and all pipeline outputs) |
| **simulation** | `event_time` | float (s) | Fault time |
| | `initialization_duration` | float (s), or `0` / omit | Steady-state run before contingencies |
| **network** | `country_filter` | list, or `[]` | Country codes to keep; empty = no filter |
| **kpi** | `window_sec` | float (s) | KPI window length |
| | `step_sec` | float (s) | KPI window step |
| | `class_bins.voltage.cuts` | list of positive floats (ascending) | Raw KPI cut thresholds for voltage class labels |
| | `class_bins.spower.cuts` | list of positive floats (ascending) | Raw KPI cut thresholds for spower class labels |
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

| Class | Meaning (example with four cuts) |
|-------|----------------------------------|
| 0 | KPI ≤ τ₁ — inactive / noise-dominated |
| 1 | τ₁ < KPI ≤ τ₂ |
| 2+ | progressively stronger dynamics |

Set τ₁ to **at least k²** so class 0 is not filled with numerical artifacts. You may raise τ₁ slightly after training-set analysis if a strict k² threshold leaves class 0 too sparse or too large.

**Nordic example:** solver precision is **k = 10⁻⁴**, so **k² = 10⁻⁸**. On the Nordic **training** set, we used **τ₁ = 10⁻⁷** for both voltage and spower (`cuts: [1e-7, 7.5e-7, 7.5e-6, 1.5e-5]`) — slightly above k² to obtain a more balanced class distribution while still treating the lowest band as noise-dominated.

Higher cuts (τ₂, τ₃, …) have no universal default: choose them from **training-set** percentiles or domain reasoning so each class reflects meaningfully stronger dynamics than the previous one.

---

## Pipeline control (`--from-step`, `--to-step`)

`main.py` runs five stages in order:

1. **`simulate`** — optional per-OP **initialization**, curve export setup, and Dynawo **contingency runs** (`src/simulate.py`)
2. **`build_op_assets`** — graphs, electrical distance, generator SNom (`src/build_op_assets.py`)
3. **`curve_process`** — per-OP KPI/flag tables, combined KPI CSVs, and train/val/test split (`src/curves_post_process.py`)
4. **`dataset`** — class labels from configured KPI cuts (`src/dataset_construction.py`)
5. **`training`** — GAT training and model export (`src/training.py`)

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
| `training` | Trained models | `model/gat_voltage_best_model.pt`, `model/gat_spower_best_model.pt` |

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
| **10** | 4609.8 |

Operating point **1** is the unmodified Nordic case from the Dynawo `DynaWaltz` repository. Points **2–10** were produced with **Dynawo load-area variation** events run as **dynamic simulations**: a ramped load reduction was applied to **all loads**, and the percentage in the table is the **reduction per load** (e.g. 10% decrease → each load ends at **90%** of its initial value). Generators rebalanced through **primary frequency control** during those runs.

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
  version: 1.11

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
      cuts: [1e-7, 7.5e-7, 7.5e-6, 1.5e-5]  # 5 KPI classes (0-4) + 1 flag class => model.num_classes: 6
    spower:
      cuts: [1e-7, 7.5e-7, 7.5e-6, 1.5e-5]

model:
  num_classes: 6

training:
  epochs: 30          # keep low for a quick smoke test
  patience: 8
  batch_size: 16
  split_mode: operating_point
  seed: 42
  training: 0.8
  validation: 0.1
  testing: 0.1
  high_class_threshold: 4  # classes >= 4 (4 and 5) are "high" with num_classes: 6
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

With `model.num_classes: 6` (classes 0–5), `high_class_threshold: 4` treats classes **4 and 5** as high severity. Class **5** is the action/disconnection flag class. That enables weighted train sampling and the CORAL under-penalty (when Optuna picks `under_penalty_lambda > 0`), and sets the cutoff for `high_recall` / `high_f1` in the validation composite score. See [`src/training.md`](src/training.md) for details.

### KPI class bins (v1.11)

See [KPI cut thresholds — recommendations](#kpi-cut-thresholds--recommendations) for how to choose cuts (training set only, noise floor at k², Nordic example).

`kpi.class_bins.<type>.cuts` lists **strictly increasing raw KPI thresholds** (e.g. `[1e-7, 7.5e-7, 7.5e-6, 1.5e-5]`). During dataset construction, DYNAGNN:

1. Uses raw KPI values from the combined KPI tables (no log transform or scaling).
2. Assigns class labels from the fixed cuts (classes 0–4 by KPI magnitude).
3. Overrides action / disconnection cells to class 5 (voltage and spower: actions + DISC).

Set `model.num_classes` to **`len(cuts) + 2`**. Applied cuts are recorded in `<data.path>/Dataset/KPI_class_bins.csv`.

