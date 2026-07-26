# `pair_aware_gine.py`

Shared **pair-aware residual GINE** model, losses, metrics, and training/evaluation loops used by Voltage and Spower.

## Used by

- `modules/pair_aware_training.py`
- `modules/pair_aware_inference.py` (model class + hparams)

## Main types

| Name | Role |
|------|------|
| `PairAwareHParams` | Model capacity and optimizer settings (Optuna-tuned) |
| `PairAwareLossWeights` | Fixed classification / regression / gate / ordinal weights |
| `PairAwareGINE` | Residual GINE encoder + multi-class / gate / log-KPI heads |
| `ResidualGINEBlock` | One residual edge-aware GINE layer |

## Model behavior

`PairAwareGINE` predicts configured activity classes directly (`0 … num_classes−1`):

- residual edge-aware GINE message passing with jumping knowledge;
- target-component and contingency identity embeddings;
- event encoding and explicit target–contingency pair interactions;
- graph mean/max context;
- multi-class logits, inactive (class-0) gate, and auxiliary log-KPI regression.

The flag class (action / disconnection) is learned — not overwritten deterministically at evaluation time. No structural disconnection masks are used.

Forward output keys: `class_logits`, `inactive_logit`, `log_kpi_std`.

## Training helpers

| Function | Description |
|----------|-------------|
| `run_pair_aware_training(...)` | Epoch loop, early stopping, checkpoint selection |
| `evaluate_saved_pair_aware_model(...)` | Reload best weights and evaluate on a loader |
| `classification_metrics(...)` | Confusion-matrix metrics and ordinal offsets |
| `selection_score(...)` | Validation composite used by Optuna |

## Decode paths

| Mode | Rule |
|------|------|
| `class` | `argmax` on class logits |
| `gated` | Class-0 gate threshold, else argmax over active classes |
| `log_kpi` | Invert standardized log-KPI and map through configured cuts (flag class still from class head) |

## Training artifacts

After evaluating the final best model on the test set, diagnostic figures are written to `data/training/<study_name>/<task>/plots/` by `modules/training_plots.py`:

| File | Contents |
|------|----------|
| `loss_curve.png` | Train and validation loss curves per epoch — total plus classification / regression / gate / ordinal components (from the winning Optuna trial’s `optuna_trials/trial_N/history.csv`; the deployment model itself is a later train+val retrain) |
| `score_curve.png` | Validation selection scores per epoch (`class` / `gated` / `logKPI`, plus the selected score) from the same Optuna `history.csv` |
| `confusion_matrix.png` | Row-normalised confusion matrix on the test set (final train+val model) |
| `distance_histogram.png` | Histogram of signed prediction offsets (pred − true) |
| `node_example_cls<N>_<UNDER|OVER>_ex<k>_of_5.png` | Up to 5 under- and 5 over-prediction examples |

## Notes

Operating-point context is carried by electrical node/edge features and graph pooling; there is no separate OP-context encoder.
