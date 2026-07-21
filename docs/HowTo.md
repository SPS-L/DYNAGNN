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

$$ \Large \mathcal{L} = w_{\mathrm{cls}} \cdot \mathcal{L}_{\mathrm{CE}} + w_{\mathrm{reg}} \cdot \mathcal{L}_{\mathrm{Huber}} + w_{\mathrm{gate}} \cdot \mathcal{L}_{\mathrm{BCE}} + w_{\mathrm{ord}} \cdot \mathcal{L}_{\mathrm{CDF}} $$

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

1. Run the pipeline through **`split`** and stop there:

```bash
python3 main.py --to-step split
```

2. Use these files under `<data.path>/`:

| File | Role |
|------|------|
| `KPI/KPI_voltage.csv` | Combined raw voltage KPI values (flagged cells masked with `NaN`) — from `curve_process` |
| `KPI/KPI_spower.csv` | Combined raw spower KPI values (flagged cells masked with `NaN`) — from `curve_process` |
| `Dataset/train_val_test_split.csv` | Train / validation / test assignment per scenario (`split`, `operating_point`, `contingency`) — from `split` |

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

`main.py` runs six stages in order:

1. **`simulate`** — optional per-OP **initialization**, curve export setup, and Dynawo **contingency runs** (`src/simulate.py`)
2. **`build_op_assets`** — graphs, electrical distance, generator SNom (`src/build_op_assets.py`)
3. **`curve_process`** — per-OP KPI/flag tables and combined KPI CSVs (`src/curves_post_process.py`)
4. **`split`** — train/val/test split CSV (`src/dataset_split_step.py`)
5. **`dataset`** — class labels from configured KPI cuts (`src/dataset_construction.py`)
6. **`training`** — pair-aware GINE Optuna training and model export (`src/training.py`)

Omit both flags for a **full run**. Use **`--from-step`** to resume from a later stage; use **`--to-step`** to stop after a stage (inclusive).

```bash
# Full run (default)
python3 main.py

# Run through curve_process only (for KPI cut analysis)
python3 main.py --to-step curve_process

# Rebuild split only (requires combined KPI tables)
python3 main.py --from-step split --to-step split

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
| `split` | through `curve_process` | `KPI/KPI_voltage.csv`, `KPI/KPI_spower.csv`, `Actions/ACTIONS_*.csv`, `Disconnections/DISC_*.csv` |
| `dataset` | through `split` | `KPI/KPI_voltage.csv`, `KPI/KPI_spower.csv`, `Dataset/train_val_test_split.csv`, `Actions/ACTIONS_*.csv`, `Disconnections/DISC_*.csv` |
| `training` | through `dataset` | `Dataset/Dataset_Voltage.csv`, `Dataset_Spower.csv`, `Dataset/train_val_test_split.csv`, `op_graphs/`, `op_electric_distance/` |

| `--to-step` | Stops after | Key outputs for cut analysis / next stage |
|-------------|-------------|-------------------------------------------|
| `curve_process` | Combined KPI tables | `KPI/KPI_voltage.csv`, `KPI/KPI_spower.csv` |
| `split` | Train/val/test split | `Dataset/train_val_test_split.csv` |
| `dataset` | Class-label datasets | `Dataset/Dataset_Voltage.csv`, `Dataset/Dataset_Spower.csv`, `Dataset/KPI_class_bins.csv` |
| `training` | Trained models | `model/<study_name>/voltage_best_model.pt`, `model/<study_name>/spower_best_model.pt`, `training/<study_name>/<task>/plots/` |

See the per-stage input tables in [`src/simulate.md`](src/simulate.md), [`src/build_op_assets.md`](src/build_op_assets.md), [`src/curves_post_process.md`](src/curves_post_process.md), [`src/dataset_split_step.md`](src/dataset_split_step.md), [`src/dataset_construction.md`](src/dataset_construction.md), and [`src/training.md`](src/training.md).

Each `main.py` run still recreates `<data.path>/dynagnn.log` from scratch.

---

## Nordic example details

### Operating points in the Nordic example

The Nordic example contains **35 operating points** (`operating_point_1` … `operating_point_35`). Each folder holds a **distinct steady-state Nordic32 case**: the same network topology and dynamic models, but with different total loading and regional load distribution.

The table below summarizes each OP. **Target load (MW)** is the total active load after applying the scaling described in [Load-scaling method](#load-scaling-method) and running a Dynawo equilibrium; the reference case has total active load **≈ 11060.6 MW**.

| OP | Split | Band | Pattern | Target load (MW) |
|---:|---|---|---|---:|
| 1 | test | low | uniform | 9016.9 |
| 2 | train | medium | central_up | 9780.3 |
| 3 | train | low | north_up | 8964.4 |
| 4 | train | high | south_up | 10634.7 |
| 5 | train | high | north_central_stress | 10841.2 |
| 6 | test | medium | mixed | 10038.0 |
| 7 | validation | low | uniform | 8986.2 |
| 8 | validation | medium | central_up | 9655.3 |
| 9 | train | high | north_up | 11140.3 |
| 10 | train | high | south_up | 10675.0 |
| 11 | train | high | north_central_stress | 10750.3 |
| 12 | train | low | mixed | 8962.7 |
| 13 | train | medium | uniform | 9851.1 |
| 14 | train | high | central_up | 10392.0 |
| 15 | train | low | north_up | 9091.6 |
| 16 | validation | high | south_up | 10239.0 |
| 17 | test | high | north_central_stress | 10719.6 |
| 18 | train | low | mixed | 8510.6 |
| 19 | test | low | uniform | 9249.8 |
| 20 | train | high | central_up | 10701.0 |
| 21 | train | low | north_up | 8716.4 |
| 22 | train | low | south_up | 9380.5 |
| 23 | test | medium | north_central_stress | 10034.1 |
| 24 | train | medium | mixed | 9766.9 |
| 25 | train | low | uniform | 9251.0 |
| 26 | train | high | central_up | 10616.9 |
| 27 | validation | high | north_up | 10376.1 |
| 28 | train | medium | south_up | 9827.7 |
| 29 | train | low | north_central_stress | 9004.1 |
| 30 | train | high | mixed | 10829.0 |
| 31 | train | low | uniform | 9161.9 |
| 32 | validation | medium | central_up | 10316.3 |
| 33 | train | medium | north_up | 10294.1 |
| 34 | train | medium | south_up | 10595.8 |
| 35 | train | low | north_central_stress | 9888.0 |

**Split assignment:** 25 train / 5 validation / 5 test operating points (fixed before load sampling). DYNAGNN’s default Nordic config uses `split_mode: operating_point` with fractions **5/7 · 1/7 · 1/7** (≈ 25 / 5 / 5 OPs for 35 OPs):

```yaml
training: 0.7142857143   # 5/7
validation: 0.1428571429 # 1/7
testing: 0.1428571429    # 1/7
```

#### Load-scaling method

Operating points are **sampled load scenarios** (not OPF solutions). All 35 cases start from the **same reference Nordic32 steady state**, then apply randomized load scaling in three layers — global, regional, and per-load — following Hagmar et al. (IET Smart Grid, 2020).

**1. Reference loads**

Let $P^0_i$ and $Q^0_i$ denote the active and reactive power of load $i$ in the reference case ($i = 1,\ldots,22$). Loads are grouped into three areas on Nordic32:

| Area | Load IDs |
|---|---|
| North | 11_1, 12_1, 13_1, 22_1, 31_1 |
| Central | 04_1, 05_1, 41_1, 42_1, 43_1, 46_1, 47_1 |
| South | 01_1, 02_1, 03_1, 32_1, 51_1, 61_1, 62_1, 63_1, 71_1, 72_1 |

**2. Loading band (global factor)**

Each OP is assigned a band that sets the overall system loading:

| Band | Range for $g$ |
|---|---|
| low | $[0.80,\; 0.85]$ |
| medium | $[0.86,\; 0.90]$ |
| high | $[0.91,\; 0.95]$ |

Sample $g$ uniformly within the OP’s band.

**3. Area stress pattern**

Each OP also gets a pattern that modulates loads by area. Let $a_N$, $a_C$, $a_S$ be multipliers for North, Central, and South:

| Pattern | Sampling rule |
|---|---|
| uniform | $a_N = a_C = a_S = 1$ |
| central_up | $a_C \sim \mathcal{U}(1.05,\, 1.12)$; $a_N = a_S = 1$ |
| north_up | $a_N \sim \mathcal{U}(1.05,\, 1.10)$; $a_C = a_S = 1$ |
| south_up | $a_S \sim \mathcal{U}(1.05,\, 1.10)$; $a_N = a_C = 1$ |
| north_central_stress | $a_N \sim \mathcal{U}(0.94,\, 0.98)$, $a_C \sim \mathcal{U}(1.04,\, 1.10)$, $a_S = 1$ |
| mixed | $a_N, a_C, a_S$ each $\sim \mathcal{U}(0.92,\, 1.08)$ |

**4. Per-load factor**

Independently for every load, sample

$$f_i \sim \mathcal{U}(0.80,\, 1.20).$$

Power factor is preserved by applying the same scale to $P$ and $Q$.

**5. Combined scale and target load**

For load $i$ in area $A(i) \in \{N, C, S\}$:

$$ s_i = g \cdot a_{A(i)} \cdot f_i, \qquad \Delta P_i = \Delta Q_i = s_i - 1. $$

Dynawo applies the variation as

$$ P_i = P^0_i\,(1 + \Delta P_i) = P^0_i\, s_i, \qquad Q_i = Q^0_i\,(1 + \Delta Q_i) = Q^0_i\, s_i. $$

The **target total active load** before the equilibrium run is

$$P_{\mathrm{target}} = \sum_{i=1}^{22} P^0_i\, s_i.$$

This is the value in the table above (rounded to one decimal).

**6. Steady-state finalization**

Load deltas are ramped in Dynawo (via area variation models) over a short window, then the simulation is run to equilibrium. The resulting network state replaces the case IIDM. Each OP folder therefore contains a **converged steady-state snapshot** at a different loading level and regional stress, ready for contingency simulation.

Each split (train / validation / test) includes low-, medium-, and high-loading OPs so that no subset is dominated by a single loading level.

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
  epochs: 100
  patience: 12
  batch_size: 16
  split_mode: operating_point
  seed: 42
  training: 0.7142857143
  validation: 0.1428571429
  testing: 0.1428571429

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
  n_trials: 15
  study_name: nordic
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

With `model.num_classes = len(cuts) + 2`, the highest class index is the action/disconnection **flag** class. Voltage and Spower are tuned independently with Optuna; validation checkpoints maximize a balanced multi-class selection score (see [`src/training.md`](src/training.md)). After the search, best hparams are **retrained on train+val** for the winning trial’s `best_epoch` epochs; that model is written under `model/<study_name>/` as `voltage_best_model.pt` and `spower_best_model.pt` and evaluated on test.

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

## AMS — model reduction (`AMS/` → pip package `dynagnn-ams`)

The **`AMS/`** folder is a **standalone pip package** (`dynagnn-ams`) for **Adaptive Model Selection**: it uses trained DYNAGNN checkpoints to simplify node-breaker models before simulation. It is **not** invoked by DYNAGNN `main.py` and does not use `config.yaml`.

### Install

```bash
python3 -m pip install "dynagnn-ams @ git+https://github.com/SPS-L/DYNAGNN.git#subdirectory=AMS"
# or from a local clone:
python3 -m pip install -e /path/to/DYNAGNN/AMS
```

Dependencies (`torch`, `torch-geometric`, `pypowsybl`, …) are pulled by pip. Nordic checkpoints are bundled in the package.

### How it fits the repository

| Entry point | Role |
|-------------|------|
| `main.py` | Training pipeline (simulations → KPIs → checkpoints) |
| `DYNAGNN.py` | Inference on new operating points and events |
| `dynagnn-ams` | Optional model reduction from a scenario `.dsl` (IIDM switch retention) |

### Bundled Nordic models

Ready-to-use deployment checkpoints: **`AMS/dynagnn_ams/models/Nordic/`**.

### Checkpoints (other networks)

```bash
NETWORK=MyCase
mkdir -p "AMS/dynagnn_ams/models/$NETWORK"
cp "<data.path>/model/<study_name>/voltage_best_model.pt" "AMS/dynagnn_ams/models/$NETWORK/"
cp "<data.path>/model/<study_name>/spower_best_model.pt" "AMS/dynagnn_ams/models/$NETWORK/"
cp "<data.path>/model/<study_name>/x_scaler.pkl" "AMS/dynagnn_ams/models/$NETWORK/"
cp "<data.path>/model/<study_name>/edge_attr_scaler.pkl" "AMS/dynagnn_ams/models/$NETWORK/"
```

### Run

```bash
# Bundled Nordic checkpoints (default)
dynagnn-ams <scenario.dsl> <network.xiidm> <network.dyd> --network Nordic --epsilon 1

# Your own trained checkpoints (no need to copy into the pip package)
dynagnn-ams <scenario.dsl> <network.xiidm> <network.dyd> \
  --network MyCase --models-dir /path/to/my_models --epsilon 1
```

Pass `--models-dir` as either the models **root** (`…/my_models/MyCase/…`) or the **checkpoint folder** directly if it already contains `voltage_best_model.pt` and `spower_best_model.pt`.

The IIDM is **modified in place**.

Full reference: [`AMS/README.md`](../AMS/README.md).
