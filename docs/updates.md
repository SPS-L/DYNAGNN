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

## Version 1.1 — log transform, train-only z-score, and quantile-based bins

### Pre-processing

All finite raw values with $x > -1$ are log-transformed **before** any split-specific step:

$$x' = \ln(1 + x)$$

This is applied to the **entire dataset** (train, validation, and test).

#### Why log-transform?

KPI values are derived from **maximum windowed variance** of post-contingency curves. They are non-negative, strongly **right-skewed**, and often **heavy-tailed**: most components stay near zero, while a few near the fault can be orders of magnitude larger.

Applying $\ln(1+x)$ before z-scoring:

- **Handles zeros cleanly:** $\ln(1+0)=0$, so inactive components are unchanged; plain $\ln(x)$ would be undefined at $x=0$.
- **Compresses the upper tail:** large spikes contribute less disproportionately to $\mu$ and $\sigma$, so a single global z-score is meaningful across the full training pool.
- **Stabilises quantile bins:** without log, most training $z$-values would cluster below a few outliers, and cuts such as $Q_{0.25}(z)$ would separate only the lowest-variance cells rather than spread severity levels more evenly.

In short, log-transform maps wide **multiplicative** spreads in raw variance onto a scale where **additive** z-scores and quantile thresholds better reflect relative dynamic activity. Only $\mu$, $\sigma$, and $\{\tau_j\}$ are fit on training data; the log step itself is not split-specific.

### Normalization (z-score)

For each KPI type separately, only **training** scenarios contribute to the location and scale. All finite $x'$ from training rows and all component columns are pooled into one set $\mathcal{T}$:

$$\mu = \frac{1}{|\mathcal{T}|} \sum_{x' \in \mathcal{T}} x', \qquad \sigma = \sqrt{\frac{1}{|\mathcal{T}|} \sum_{x' \in \mathcal{T}} (x' - \mu)^2}$$

Every finite cell (all splits) is standardized with these **train-fitted** parameters:

$$z = \frac{x' - \mu}{\sigma}$$

Validation and test cells are transformed with the same $\mu$ and $\sigma$; they do **not** influence $\mu$ or $\sigma$.

### Class labeling (quantile cuts on training $z$)

Configuration supplies quantile fractions

$$0 < q_1 < q_2 < \cdots < q_K < 1$$

From the **training** z-scores only, cut thresholds are estimated as empirical quantiles:

$$\tau_j = Q_{q_j}(z \mid \text{train}), \quad j = 1,\ldots,K$$

For each finite $z$ (any split), the KPI severity class is

$$y = \bigl|\{\, j : z > \tau_j \,\}\bigr|$$

So training defines both $(\mu, \sigma)$ and $\{\tau_j\}$; validation and test receive labels via the same mapping. Action and disconnection flags again override KPI bins with

$$y_{\mathrm{flag}} = M, \qquad M = K + 1$$

(the index of the first flag class equals the number of KPI severity classes). Voltage applies disconnection flags; spower applies action flags only.

**Total number of classes:** $K + 2$, with $K = |\{q_j\}|$.

Unlike v1.0, the effective bin boundaries $\{\tau_j\}$ are **data-driven on the training set** (via quantiles of $z$), while the **fractions** $\{q_j\}$ remain user-specified in configuration.

### Artifacts

Version 1.1 stores $\mu$, $\sigma$, the $\tau_j$, and fitted scalers under a dedicated **normalization** folder. Combined KPI tables retain **raw** (masked) values; class labels live only in the dataset tables.

---

## Side-by-side summary

| Aspect | v1.0 | v1.1 |
|--------|------|------|
| Transform before scaling | None | $x' = \ln(1+x)$ |
| Scale parameters | $x_{\min}$, $x_{\max}$ from **all** data | $\mu$, $\sigma$ from **train** only |
| Normalized quantity | $\tilde{x} \in [0,1]$ | $z \in \mathbb{R}$ |
| Bin boundaries | Fixed $c_j \in (0,1)$ on $\tilde{x}$ | $\tau_j = Q_{q_j}(z \mid \text{train})$ |
| Config `cuts` meaning | Interval edges on $[0,1]$ | Quantile levels for training $z$ |
| Split timing | After labeling (training stage) | Before normalization (dataset stage) |
| KPI table content | Normalized values | Raw KPI values |

---

## Example (illustrative)

**Configuration:** three cuts ($K = 3$).

**v1.0** with $c = (0.25,\, 0.5,\, 0.75)$: after min–max, class 0 is $\tilde{x} \le 0.25$, class 1 is $(0.25, 0.5]$, etc., regardless of how many training points fall in each bin.

**v1.1** with $q = (0.25,\, 0.5,\, 0.75)$: $\tau_1$, $\tau_2$, $\tau_3$ are the 25th, 50th, and 75th percentiles of **training** $z$; on training data, KPI classes 0–3 each contain approximately 25% of finite cells (up to ties and boundary effects). The same $\tau_j$ are applied to validation and test.

In both versions, flag cells receive class $K+1$ and **total classes** $= K + 2$.

---

## Migration note

When upgrading from v1.0 to v1.1, replace interval-style cut values (e.g. $0.33$, $0.66$) with quantile fractions (e.g. $0.25$, $0.5$, $0.75$) and set the model class count to $|\{q_j\}| + 2$. Re-run dataset construction so splits, scalers, cuts, and labels are regenerated consistently.

---

## Reason for the update

Version 1.0 mapped every raw KPI onto $[0,1]$ using a **single global** minimum and maximum taken over the full dataset. When the corpus contains even a few scenarios with very severe dynamics, those outliers set $x_{\max}$. Every other value is then compressed toward zero:

$$\tilde{x} = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \approx 0 \quad \text{for most cells}$$

Components that still exhibit **meaningful** dynamic activity can end up with $\tilde{x}$ indistinguishably close to inactive ones, simply because they are small **relative to the most extreme case**, not because they are physically unimportant. Severity labels therefore become sensitive to which rare contingencies happen to appear in the table. The pipeline **depends heavily on the composition of the dataset**, and the same physical response can receive different classes if the global range shifts—making it harder for the model to **generalize** to new operating points or fault patterns.

Version 1.1 addresses this by describing dynamics through **statistical behavior** rather than absolute rescaled magnitude. Log-transform and train-only z-scoring summarize how active a component is **relative to the training distribution**; quantile cuts on $z$ define severity bins from that distribution instead of fixed edges on $[0,1]$. Validation and test data are labeled with thresholds learned from training only, which reduces leakage and stabilizes the meaning of each class across splits.

With a representative training set, this approach captures the **statistical structure of dynamic activity** in the grid. The model can learn **patterns of behavior**—how disturbance severity ranks across components and scenarios—rather than memorizing label boundaries that collapse whenever one contingency dominates the min–max range. Supervision is aligned with **relative severity**, so the GAT learns the statistical behavior of dynamics, not arbitrary rescaled labels tied to a few extreme scenarios.
