# DYNAGNN version history

This note documents KPI labeling methodology and the training/inference model stack across **v1.0**, **v1.1**, **v1.11**, and **v1.2** (current). Notation: $x$ denotes a raw KPI value for one component in one scenario; flagged or invalid cells are excluded from the labeling steps below.

---

## Pipeline order

| Stage | Version 1.0 | Version 1.1 | Version 1.11 | Version 1.2 |
|-------|-------------|-------------|--------------|-------------|
| Combined KPI tables | Raw values, then min–max normalized in place | Raw values only | Raw values only | Same as v1.11 |
| Train / validation / test split | Built **after** class labeling, during training | Built **before** labeling, from raw voltage scenarios | Same as v1.1 | Same as v1.11 |
| Class datasets | Produced in dataset construction | Same stage; log10 + z-score + range cuts | Same stage; **fixed raw KPI cuts** | Same labeling as v1.11 |
| Labeling metadata | Min / max per KPI table | $\mu$, $\sigma$, training z-cut thresholds | Raw cut thresholds in `KPI_class_bins.csv` | Same as v1.11 |
| Activity model | — | GAT + CORAL ordinal head | GAT + CORAL ordinal head | **Pair-aware residual GINE** (direct multi-class) |

---

## Version 1.0 — min–max normalization and fixed unit-interval bins

### Normalization

For each KPI type (voltage or spower), all finite raw values in the **full combined table** were used to define one global range:

$$x_{\min} = \min\{ x \},\qquad x_{\max} = \max\{ x \}$$

Each finite value was mapped linearly to the unit interval:

$$\tilde{x} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

(with the degenerate case $\tilde{x} = 0$ when $x_{\min} = x_{\max}$). No logarithmic transform was applied. The fitted bounds were stored as a min–max normalization record. **Validation and test data contributed to $x_{\min}$ and $x_{\max}$.**

The combined KPI files stored these **normalized** values $\tilde{x}$, not the raw KPIs.

### Class labeling

Configuration supplied a sorted list of cut points on the normalized scale:

$$0 < c_1 < c_2 < \cdots < c_K < 1$$

For each finite $\tilde{x}$, the KPI severity class was

$$y = \bigl|\{\, j : \tilde{x} > c_j \,\}\bigr|$$

i.e. the number of cut thresholds strictly exceeded (equivalently, binning $[0,1]$ into $K+1$ intervals defined by $\{c_j\}$). Cells with action or disconnection flags were assigned a dedicated **flag class**

$$y_{\mathrm{flag}} = K + 1$$

Voltage used both action and disconnection flags; spower used action flags only.

**Total number of classes:** $K + 2$ ($K+1$ KPI bins + one flag class), with $K = |\{c_j\}|$.

The cut values $c_j$ were **fixed in configuration** (e.g. $0.33$, $0.66$): they did not adapt to the empirical distribution of the data beyond the initial min–max scaling.

---

## Version 1.1 — log10 transform, train-only z-score, and range-based bins

### Pre-processing

Raw KPI zeros are replaced with the **smallest positive value** in the table (because $\log_{10}(0)$ is undefined). All finite positive values are then log-transformed **before** any split-specific step:

$$x' = \log_{10}(x)$$

This is applied to the **entire dataset** (train, validation, and test).

#### Why log10?

KPI values are derived from **maximum windowed variance** of post-contingency curves. They are non-negative, strongly **right-skewed**, and often span many orders of magnitude: most components stay near zero, while a few near the fault can be much larger.

Applying $\log_{10}$ before z-scoring:

- **Handles zeros:** zeros are mapped to the smallest positive value so $\log_{10}$ is defined everywhere finite values exist.
- **Compresses multiplicative spread:** large spikes contribute less disproportionately to $\mu$ and $\sigma$, so a single global z-score is meaningful across the full training pool.
- **Stabilises range bins:** without log, most training $z$-values would cluster below a few outliers.

Only $\mu$, $\sigma$, and $\{\tau_j\}$ are fit on training data; the log step itself is not split-specific.

### Normalization (z-score)

For each KPI type separately, only **training** scenarios contribute to the location and scale. All finite $x'$ from training rows and all component columns are pooled into one set $\mathcal{T}$:

$$\mu = \frac{1}{|\mathcal{T}|} \sum_{x' \in \mathcal{T}} x', \qquad \sigma = \sqrt{\frac{1}{|\mathcal{T}|} \sum_{x' \in \mathcal{T}} (x' - \mu)^2}$$

Every finite cell (all splits) is standardized with these **train-fitted** parameters:

$$z = \frac{x' - \mu}{\sigma}$$

Validation and test cells are transformed with the same $\mu$ and $\sigma$; they do **not** influence $\mu$ or $\sigma$.

### Class labeling (range cuts on training $z$)

Configuration supplies activity fractions

$$0 < f_1 < f_2 < \cdots < f_K < 1$$

From **training** z-scores only, define the empirical range $z_{\min}$ and $z_{\max}$ on finite train cells. Cut thresholds are placed along that range:

$$\tau_j = z_{\min} + f_j \,(z_{\max} - z_{\min}), \quad j = 1,\ldots,K$$

For each finite $z$ (any split), the KPI severity class is

$$y = \bigl|\{\, j : z > \tau_j \,\}\bigr|$$

So training defines both $(\mu, \sigma)$ and $\{\tau_j\}$; validation and test receive labels via the same mapping. Action and disconnection flags again override KPI bins with

$$y_{\mathrm{flag}} = M, \qquad M = K + 1$$

(the index of the first flag class equals the number of KPI severity classes). Voltage applies disconnection flags; spower applies action flags only.

**Total number of classes:** $K + 2$, with $K = |\{f_j\}|$.

Unlike v1.0, the effective bin boundaries $\{\tau_j\}$ are **data-driven on the training set** (via the training z range), while the **fractions** $\{f_j\}$ remain user-specified in configuration.

### Artifacts and diagnostics (v1.1)

Version 1.1 stored $\mu$, $\sigma$, the $\tau_j$, and fitted scalers under a dedicated **normalization** folder. Combined KPI tables retained **raw** (masked) values; class labels lived only in the dataset tables. Dataset construction also wrote histograms under `Dataset/KPI_visualization/`: raw KPI (log-scaled axes), $\log_{10}(\mathrm{KPI})$, and z-score with class-cut overlays.

---

## Version 1.11 — fixed raw KPI class cuts

### Rationale

KPI values are already a **variance-based measure of dynamic activity** on the physical scale. Discretizing them into ordered severity classes **already encodes** which events are mild vs severe: higher classes correspond to larger post-contingency dynamics without an extra rescaling step.

Because the downstream task is **classification** (not regression on a normalized continuous target), additional log transforms, z-scoring, or train-derived bin placement add complexity without improving the supervision signal. Fixed **raw KPI cut thresholds** in configuration define severity bands directly on the same scale produced by `src/curves_post_process.py`.

There is no need for normalization, scaling, or distribution-fitting artifacts between KPI extraction and class labeling.

### Class labeling

Configuration supplies a sorted list of **raw KPI cut thresholds** (e.g. $10^{-6}$, $2.25\times10^{-5}$, $3\times10^{-4}$, $5.625\times10^{-4}$). For each finite raw KPI value $x$:

| Class | Rule (example with four cuts) |
|-------|-------------------------------|
| 0 | $x \le \tau_1$ |
| 1 | $\tau_1 < x \le \tau_2$ |
| 2 | $\tau_2 < x \le \tau_3$ |
| 3 | $\tau_3 < x \le \tau_4$ |
| 4 | $x > \tau_4$ |

where $\tau_1,\ldots,\tau_K$ are the configured cuts. Cells with action or disconnection flags are assigned the **flag class** $K+1$ (voltage and spower: actions + disconnection).

**Total number of classes:** $K + 2$ with $K = |\{\tau_j\}|$.

Cuts are **fixed in configuration** on the raw KPI scale and do not depend on the training split or empirical distribution.

### Artifacts (v1.11)

| Path | Role |
|------|------|
| `data/KPI/KPI_voltage.csv`, `KPI_spower.csv` | Combined raw KPI tables (masked) |
| `data/Dataset/Dataset_Voltage.csv`, `Dataset_Spower.csv` | Class labels from raw cuts |
| `data/Dataset/KPI_class_bins.csv` | Applied raw cut thresholds and class metadata |
| `data/Dataset/dataset_class_distribution.png` | Class-count bar chart |

**Removed:** `modules/normalization.py`, `modules/kpi_visualization.py`, `<data.path>/normalization/`, and `Dataset/KPI_visualization/`.

### Model stack in v1.11

Training still used a **GAT** encoder with a **CORAL** ordinal classification head, Optuna-tuned GAT capacity, and high-class composite selection (`high_recall` / `high_f1` / loss). Deployment checkpoints were named `gat_voltage_best_model.pt` and `gat_spower_best_model.pt`.

---

## Version 1.2 — pair-aware residual GINE (current)

### What changed

v1.2 keeps the **v1.11 KPI labeling** (fixed raw cuts → KPI severity classes plus one flag class) and the same Dynawo → graph → KPI → dataset pipeline stages. The upgrade replaces the **activity model and training method**:

| Aspect | v1.11 | v1.2 |
|--------|-------|------|
| Encoder | GAT | Residual edge-aware **GINE** |
| Head | CORAL ordinal | **Direct multi-class** logits + inactive gate + log-KPI regression |
| Conditioning | Graph features + `fault_on` / `dz_fault` | Same electrical features **plus** target-id / contingency-id embeddings and explicit **target–contingency** pair interactions |
| Architecture choice | Single GAT-CORAL path | Pair-aware GINE only (no GAT-CORAL fallback) |
| Optuna studies | Shared-style GAT hparams (heads, CORAL threshold, under-penalty, …) | **Independent** Voltage and Spower studies; capacity + optimizer only |
| Selection score | `high_recall + w_f1·high_f1 − w_loss·loss` | `0.40·balanced_acc + 0.30·macro_f1 + 0.20·acc + 0.10·within_one` |
| Checkpoints | `gat_voltage_best_model.pt`, `gat_spower_best_model.pt` | `voltage_best_model.pt`, `spower_best_model.pt` |
| External inference I/O | One class per component | **Unchanged** (`prediction_voltage.csv` / `prediction_spower.csv`) |

### Method (pair-aware GINE)

`PairAwareGINE` predicts configured activity classes **directly** (`0 … num_classes−1`, with `num_classes = len(cuts) + 2`). The GNN is the primary predictor. It receives:

- graph topology and physical node/edge features (including `fault_on` and electrical distance to the fault);
- explicit **target-component** identity embeddings;
- **contingency** identity / location encoding and event context;
- graph-level mean and max pooling.

A residual GINE stack with jumping knowledge feeds a shared decoder that produces:

1. **class logits** over all configured classes (KPI activity levels plus the flag class for disconnected / controlled components);
2. an **inactive (class-0) gate**;
3. an auxiliary **log-KPI** regression head (supervised where a finite KPI exists).

The flag class is **learned** — there is no deterministic post-hoc override from disconnection flags at evaluation/inference. Structural disconnection masks are retained for target construction and audit only. No historical KPI/class prior is used.

A separate operating-point context encoder is **not** used (`op_context_embedding_dim` is not an Optuna parameter). OP information remains available through node/edge electrical features and graph mean/max pooling.

### Configuration

- **Fixed** under `training.pair_aware`: loss weights (`classification_weight`, `regression_weight`, `inactive_gate_weight`, `ordinal_weight`), `class_weight_mode`, `gate_pos_weight_mode`, `gate_threshold`, `epsilon`, `selection_output`.
- **Tuned** under `optuna.hparams` only: `hidden_dim`, `node_id_dim`, `contingency_id_dim`, `type_dim`, `pair_dim`, `num_gnn_layers`, `decoder_hidden_dim`, `dropout`, `lr`, `weight_decay`.
- Set `model.num_classes` to **`len(cuts) + 2`** so it matches the KPI cuts and one flag class.

Pipeline entry points are unchanged: `main.py` → `src/training.py` → `modules/voltage_training.py` / `modules/spower_training.py`. Inference remains `DYNAGNN.py` with the same CLI and CSV outputs.

### Nordic example

Operating point **10** was removed from the bundled Nordic example (folder and docs). The example now ships **nine** operating points (`operating_point_1` … `operating_point_9`). The smoke-test config uses four KPI cuts → `num_classes: 6`.

### Artifacts (v1.2)

| Path | Role |
|------|------|
| `data/model/voltage_best_model.pt`, `spower_best_model.pt` | Deployment checkpoints (vocabularies, cuts, selected decode path, weights) |
| `data/model/voltage_best_hparams.json`, `spower_best_hparams.json` | Checkpoint metadata without weights |
| `data/model/x_scaler.pkl`, `edge_attr_scaler.pkl` | Train-fit feature scalers |
| `data/training/<task>/optuna_*.sqlite3`, `optuna_trials.csv` | Per-task Optuna studies |

**Removed from the active stack:** GAT-CORAL training/decoding modules, `gat_*_best_model.pt` naming, CORAL Optuna knobs (`under_penalty_lambda`, `coral_prediction_threshold`, …), and high-class selection weights (`high_class_threshold`, `selection_f1_weight`, `selection_loss_weight`).

---

## Side-by-side summary (labeling)

| Aspect | v1.0 | v1.1 | v1.11 / v1.2 labeling |
|--------|------|------|------------------------|
| Transform before labeling | None | Zero replace, then $x' = \log_{10}(x)$ | None |
| Scaling | Min–max on **all** data → $\tilde{x} \in [0,1]$ | Train-only z-score on $\log_{10}(x)$ | None |
| Quantity used for bins | $\tilde{x}$ | $z$ | Raw $x$ |
| Config `cuts` meaning | Interval edges on $[0,1]$ | Activity fractions along training $z$ range | Raw KPI thresholds (ascending) |
| Bin boundaries | Fixed $c_j$ on $\tilde{x}$ | $\tau_j = z_{\min} + f_j\,(z_{\max} - z_{\min})$ from **train** | Fixed $\tau_j$ on raw KPI scale |
| Split timing | After labeling (training stage) | Before labeling (dataset stage) | Before labeling (dataset stage) |
| KPI table content | Normalized values | Raw KPI values | Raw KPI values |
| Extra artifacts | Min–max bounds CSV | Scalers, `KPI_normalization.csv`, KPI histograms | `KPI_class_bins.csv` only |

---

## Example (illustrative)

**Configuration:** three interior cuts ($K = 3$) → four KPI severity classes plus one flag class → **total classes** $= 5$.

**v1.0** with $c = (0.25,\, 0.5,\, 0.75)$: after min–max, class 0 is $\tilde{x} \le 0.25$, class 1 is $(0.25, 0.5]$, etc., regardless of how many training points fall in each bin.

**v1.1** with $f = (0.5,\, 0.8,\, 0.9)$: on training $z$, take $z_{\min}$ and $z_{\max}$; $\tau_j = z_{\min} + f_j (z_{\max} - z_{\min})$. Class 0 is $z \le \tau_1$, class 1 is $(\tau_1, \tau_2]$, etc. The same $\tau_j$ are applied to validation and test.

**v1.11 / v1.2** with raw cuts $\tau = (10^{-6},\, 2.25\times10^{-5},\, 3\times10^{-4})$ (three cuts, four KPI classes): class 0 is $x \le 10^{-6}$, class 1 is $(10^{-6},\, 2.25\times10^{-5}]$, etc., on the **raw** KPI from curve post-processing. With a fourth cut $5.625\times10^{-4}$ (as in the Nordic example), class 4 is $x > 5.625\times10^{-4}$ and **total classes** $= 6$ including the flag class.

In all versions, flag cells receive class $K+1$.

---

## Migration notes

**v1.0 → v1.1:** Replace interval-style cut values (e.g. $0.33$, $0.66$) with activity fractions along the training z range (e.g. $0.5$, $0.8$, $0.9$). Set `model.num_classes` to $|\{f_j\}| + 2$. Re-run dataset construction so splits, scalers, cuts, and labels are regenerated consistently.

**v1.1 → v1.11:** Replace activity fractions in `kpi.class_bins.*.cuts` with **raw KPI thresholds** (strictly increasing positive values). Set `dynagnn.version` to `1.11` and `model.num_classes` to `len(cuts) + 2`. Re-run from `dataset` (or the full pipeline). Remove obsolete `normalization/` and `Dataset/KPI_visualization/` folders if present.

**v1.11 → v1.2:** Keep `kpi.class_bins.*.cuts` and set `model.num_classes` to `len(cuts) + 2`. Replace GAT/CORAL training keys with `training.pair_aware` and the pair-aware `optuna.hparams` list. Set `dynagnn.version` to `1.2`. Retrain from `training` (or full pipeline). Load new checkpoints `voltage_best_model.pt` / `spower_best_model.pt` — old `gat_*` weights are not compatible. Inference CLI and prediction CSV layout are unchanged.

---

## Reason for the updates

### v1.0 → v1.1

Version 1.0 mapped every raw KPI onto $[0,1]$ using a **single global** minimum and maximum taken over the full dataset. When the corpus contains even a few scenarios with very severe dynamics, those outliers set $x_{\max}$. Every other value is then compressed toward zero:

$$\tilde{x} = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \approx 0 \quad \text{for most cells}$$

Components that still exhibit **meaningful** dynamic activity can end up with $\tilde{x}$ indistinguishably close to inactive ones, simply because they are small **relative to the most extreme case**, not because they are physically unimportant. Severity labels therefore become sensitive to which rare contingencies happen to appear in the table.

Version 1.1 addressed this by describing dynamics through **statistical behavior** rather than absolute rescaled magnitude. Log10 transform (with zero replacement) and train-only z-scoring summarize how active a component is **relative to the training distribution**; range cuts on $z$ define severity bins from the training z span instead of fixed edges on $[0,1]$.

### v1.1 → v1.11

Version 1.1 still introduced an intermediate continuous representation ($z$-scores) and train-dependent cut placement before discretization. For a **classification** pipeline, that step is redundant: the KPI itself already ranks dynamic severity on a physically meaningful scale, and the class index is the quantity the model is trained to predict.

Version 1.11 assigns labels **directly from raw KPI values** using fixed thresholds chosen for the problem domain. Severity is encoded once—at labeling time—without log transforms, scalers, or split-specific bin fitting. The supervision signal is easier to interpret, reproducible across datasets, and independent of which contingencies happen to dominate the training distribution.

### v1.11 → v1.2

Version 1.11 still predicted activity with a GAT + CORAL stack that treated ordinal ranks as cumulative thresholds and selected checkpoints mainly for high-severity recall. That design did not explicitly condition each target component on the **identity of the contingency** it is paired with, and CORAL decoding plus high-class heuristics added moving parts that were hard to align with a fixed multi-class labeling scheme.

Version 1.2 replaces that stack with a **pair-aware residual GINE**: residual edge-aware message passing, target and contingency embeddings, and a direct multi-class head (plus gate and log-KPI auxiliaries). Voltage and Spower are tuned in **separate Optuna studies** with capacity/optimizer search only; loss construction stays fixed under `training.pair_aware`. The external inference contract stays one predicted class per component, while the learned representation becomes explicitly event- and target-conditioned.
