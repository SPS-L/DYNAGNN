# DYNAGNN version 1.1

This note summarizes methodological changes in **v1.1**, with emphasis on KPI normalization and class labeling. Notation: $x$ denotes a raw KPI value for one component in one scenario; flagged or invalid cells are excluded from the steps below.

---

## Pipeline order

| Stage | Version 1.0 | Version 1.1 |
|-------|-------------|-------------|
| Combined KPI tables | Raw values, then min–max normalized in place | Raw values only (no normalization in KPI tables) |
| Train / validation / test split | Built **after** class labeling, during training | Built **before** normalization, from raw voltage scenarios |
| Class datasets | Produced in dataset construction | Same stage, but with a different normalization and labeling rule |
| Normalization metadata | Min / max per KPI table | $\mu$, $\sigma$, and training z-score cut thresholds |

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

### Artifacts and diagnostics

Version 1.1 stores $\mu$, $\sigma$, the $\tau_j$, and fitted scalers under a dedicated **normalization** folder. Combined KPI tables retain **raw** (masked) values; class labels live only in the dataset tables. Dataset construction also writes histograms under `Dataset/KPI_visualization/`: raw KPI (log-scaled axes), $\log_{10}(\mathrm{KPI})$, and z-score with class-cut overlays.

---

## Side-by-side summary

| Aspect | v1.0 | v1.1 |
|--------|------|------|
| Transform before scaling | None | Zero replace, then $x' = \log_{10}(x)$ |
| Scale parameters | $x_{\min}$, $x_{\max}$ from **all** data | $\mu$, $\sigma$ from **train** only |
| Normalized quantity | $\tilde{x} \in [0,1]$ | $z \in \mathbb{R}$ |
| Bin boundaries | Fixed $c_j \in (0,1)$ on $\tilde{x}$ | $\tau_j = z_{\min} + f_j\,(z_{\max} - z_{\min})$ |
| Config `cuts` meaning | Interval edges on $[0,1]$ | Activity fractions along training $z$ range |
| Split timing | After labeling (training stage) | Before normalization (dataset stage) |
| KPI table content | Normalized values | Raw KPI values |

---

## Example (illustrative)

**Configuration:** three cuts ($K = 3$).

**v1.0** with $c = (0.25,\, 0.5,\, 0.75)$: after min–max, class 0 is $\tilde{x} \le 0.25$, class 1 is $(0.25, 0.5]$, etc., regardless of how many training points fall in each bin.

**v1.1** with $f = (0.5,\, 0.8,\, 0.9)$: on training $z$, take $z_{\min}$ and $z_{\max}$; $\tau_j = z_{\min} + f_j (z_{\max} - z_{\min})$. Class 0 is $z \le \tau_1$, class 1 is $(\tau_1, \tau_2]$, etc. The same $\tau_j$ are applied to validation and test.

In both versions, flag cells receive class $K+1$ and **total classes** $= K + 2$.

---

## Migration note

When upgrading from v1.0 to v1.1, replace interval-style cut values (e.g. $0.33$, $0.66$) with activity fractions along the training z range (e.g. $0.5$, $0.8$, $0.9$) and set the model class count to $|\{f_j\}| + 2$. Re-run dataset construction so splits, scalers, cuts, and labels are regenerated consistently.

---

## Reason for the update

Version 1.0 mapped every raw KPI onto $[0,1]$ using a **single global** minimum and maximum taken over the full dataset. When the corpus contains even a few scenarios with very severe dynamics, those outliers set $x_{\max}$. Every other value is then compressed toward zero:

$$\tilde{x} = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \approx 0 \quad \text{for most cells}$$

Components that still exhibit **meaningful** dynamic activity can end up with $\tilde{x}$ indistinguishably close to inactive ones, simply because they are small **relative to the most extreme case**, not because they are physically unimportant. Severity labels therefore become sensitive to which rare contingencies happen to appear in the table. The pipeline **depends heavily on the composition of the dataset**, and the same physical response can receive different classes if the global range shifts—making it harder for the model to **generalize** to new operating points or fault patterns.

Version 1.1 addresses this by describing dynamics through **statistical behavior** rather than absolute rescaled magnitude. Log10 transform (with zero replacement) and train-only z-scoring summarize how active a component is **relative to the training distribution**; range cuts on $z$ define severity bins from the training z span instead of fixed edges on $[0,1]$. Validation and test data are labeled with thresholds learned from training only, which reduces leakage and stabilizes the meaning of each class across splits.

With a representative training set, this approach captures the **statistical structure of dynamic activity** in the grid. The model can learn **patterns of behavior**—how disturbance severity ranks across components and scenarios—rather than memorizing label boundaries that collapse whenever one contingency dominates the min–max range. Supervision is aligned with **relative severity**, so the GAT learns the statistical behavior of dynamics, not arbitrary rescaled labels tied to a few extreme scenarios.
