# `pair_aware_gine.py`

Shared **pair-aware residual GINE** model, losses, metrics, and training/evaluation loops used by Voltage and Spower.

## Used by

- `modules/pair_aware_training.py`
- `modules/pair_aware_inference.py` (model class + hparams)

## Main types

| Name | Role |
|------|------|
| `PairAwareHParams` | Model capacity and optimizer settings (Optuna-tuned) |
| `PairAwareLossWeights` | Fixed classification / regression / inactive-gate / flag-gate / ordinal weights |
| `SelectionScoreWeights` | Configurable validation protection-selection score weights |
| `PairAwareGINE` | Residual GINE encoder + severity / flag-gate / inactive-gate / log-KPI heads |
| `ResidualGINEBlock` | One residual edge-aware GINE layer |

## Model behavior

`PairAwareGINE` uses a hierarchical output (`K = model.num_classes`):

- residual edge-aware GINE message passing with jumping knowledge;
- target-component and contingency identity embeddings;
- event encoding and explicit target–contingency pair interactions;
- graph mean/max context;
- **severity head** for KPI activity classes `0 … K−2`;
- **flag gate** (binary) for class `K−1` (disconnected / controlled);
- **inactive gate** (binary) for class 0 inside the non-flag branch;
- auxiliary log-KPI regression (severity samples with finite KPI only).

The flag class is learned by the separate flag gate (thresholded at decode). No structural DISC/ACTIONS masks are used at training time. No historical KPI/class prior is used.

Forward output keys: `class_logits` (severity, size `K−1`), `flag_logit`, `inactive_logit`, `log_kpi_std`.

## Training helpers

| Function | Description |
|----------|-------------|
| `run_pair_aware_training(...)` | Epoch loop, early stopping, checkpoint selection |
| `evaluate_saved_pair_aware_model(...)` | Reload best weights and evaluate on a loader |
| `classification_metrics(...)` | Confusion-matrix metrics and ordinal offsets |
| `protection_selection_score(...)` | Validation composite used by Optuna / early stopping |
| `selection_score(...)` | Legacy reporting score (not used for Optuna) |
| `compute_flag_pos_weight(...)` | Balanced pos-weight for the flag-gate BCE |

## Decode paths

All paths apply the flag gate first when $\sigma(\mathrm{flag}) \ge$ `flag_gate_threshold` → predict class `K−1`.

| Mode | Non-flag rule |
|------|---------------|
| `class` | `argmax` on severity logits |
| `gated` | Inactive-gate threshold (`inactive_gate_threshold`) → class 0, else argmax over active severity classes |
| `log_kpi` | Invert standardized log-KPI and map through configured cuts |

## Training artifacts

After evaluating the final best model on the test set, diagnostic figures are written to `data/training/<study_name>/<task>/plots/` by `modules/training_plots.py`:

| File | Contents |
|------|----------|
| `loss_curve.png` | Train and validation loss curves per epoch — total plus classification / regression / inactive-gate / flag-gate / ordinal components (from the winning Optuna trial’s `optuna_trials/trial_N/history.csv`; the deployment model itself is a later train+val retrain) |
| `score_curve.png` | Validation protection selection scores per epoch (`class` / `gated` / `logKPI`, plus the selected score) from the same Optuna `history.csv` |
| `confusion_matrix.png` | Row-normalised confusion matrix on the test set (final train+val model) |
| `distance_histogram.png` | Histogram of signed prediction offsets (pred − true) |
| `node_example_cls<N>_<UNDER|OVER>_ex<k>_of_5.png` | Up to 5 under- and 5 over-prediction examples |

## Notes

Operating-point context is carried by electrical node/edge features and graph pooling; there is no separate OP-context encoder.
