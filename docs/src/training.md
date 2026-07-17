# `src/training.py`

End-to-end **pair-aware GINE training**: build a shared PyG dataset (voltage + spower labels), append electrical-distance features, attach pair-aware targets, train/val/test split, scaler fitting, independent Optuna studies per task, and deployment checkpoint export.

## Invoked by

- `main.py` (fifth pipeline stage)

## Inputs

| Source | Content |
|--------|---------|
| `data/Dataset/Dataset_Voltage.csv`, `Dataset_Spower.csv` | Class labels (`0 … num_classes−1`) |
| `data/Dataset/train_val_test_split.csv` | Train/validation/test split (built by `curve_process`) |
| `data/KPI/KPI_voltage.csv`, `KPI_spower.csv` | Raw KPI values (for log-KPI regression targets) |
| `data/op_graphs/*.pt` | Graph structure and metadata |
| `data/op_electric_distance/*.csv` | `dz_fault` feature |
| `config.yaml` | `training.*` (incl. `pair_aware`), `optuna.*`, `model.num_classes`, `kpi.class_bins.*.cuts`, `network.country_filter` |

## Outputs

| Path | Content |
|------|---------|
| `data/model/x_scaler.pkl`, `edge_attr_scaler.pkl` | Feature scalers (train-fit only) |
| `data/model/voltage_best_model.pt`, `spower_best_model.pt` | Deployment checkpoints |
| `data/model/voltage_best_hparams.json`, `spower_best_hparams.json` | Checkpoint metadata (hparams, vocabs, cuts, …) |
| `data/model/training_summary.json` | Best-trial summary for both tasks |
| `data/training/voltage/`, `data/training/spower/` | Optuna SQLite/CSV, trial folders, test evaluation artifacts |

## Main entry point

| Function | Description |
|----------|-------------|
| `main()` | Full training flow (shared graph build → voltage Optuna → spower Optuna) |

## Flow (summary)

1. Load class-label datasets and the split CSV (must already exist from dataset construction).
2. Build shared `graph_dataset` with `y_voltage` / `y_spower` masks; resolve each row’s **Contingency** on the graph and set `fault_on` (see [Event lookup](#event-lookup-and-fault_on-placement)).
3. Append log electrical distance from fault to each node (`dz_fault`).
4. Attach pair-aware tensors via `attach_pair_aware_targets()`: shared node/contingency vocabularies, event masks, log-KPI targets.
5. Fit feature scalers on the **train** split only; scale all splits.
6. `run_voltage_training()` then `run_spower_training()` — each runs its own Optuna study maximizing the validation [selection score](#training-selection-score).

Set `model.num_classes` to **`len(cuts) + 2`** (must be >= 2): `len(cuts)` KPI activity classes plus one flag class (disconnected/controlled). The pipeline validates `len(kpi.class_bins.<task>.cuts) == num_classes - 2`.

## Model (pair-aware residual GINE)

DYNAGNN uses one model family: **`PairAwareGINE`** (`modules/pair_aware_gine.py`). There is no architecture selector and no legacy GAT-CORAL path.

The model predicts activity classes **directly** (`0 … num_classes−1`) and uses:

- residual edge-aware **GINE** message passing;
- concatenation of the initial node representation and all GINE-layer outputs (jumping knowledge);
- **target-component** and **contingency** identity embeddings;
- event encoding and explicit target–contingency pair interactions;
- graph mean/max context;
- a multi-class head, a class-0 (inactive) gate, and an auxiliary log-KPI regression head.

Operating-point information enters through node/edge electrical features and graph-level pooling. A separate OP-context encoder is **not** used.

Task entry points:

- [`voltage_training`](../modules/voltage_training.md) → buses (`bus_node_mask`)
- [`spower_training`](../modules/spower_training.md) → generators (`gen_node_mask`)

Both call [`pair_aware_training.run_task_training()`](../modules/pair_aware_training.md).

### Loss (fixed under `training.pair_aware`)

| Weight key | Role |
|------------|------|
| `classification_weight` | Cross-entropy over all configured classes |
| `regression_weight` | Smooth L1 on standardized log-KPI (finite KPI targets only; flag class has none) |
| `inactive_gate_weight` | BCE gate for class 0 vs active |
| `ordinal_weight` | Ordinal CDF consistency on class logits |

Also fixed: `class_weight_mode`, `gate_pos_weight_mode`, `gate_threshold`, `epsilon`, `selection_output` (`auto` / `class` / `gated` / `log_kpi`).

### Optuna (`optuna.hparams`)

Tunable capacity and optimizer settings only:

`hidden_dim`, `node_id_dim`, `contingency_id_dim`, `type_dim`, `pair_dim`, `num_gnn_layers`, `decoder_hidden_dim`, `dropout`, `lr`, `weight_decay`.

Voltage and Spower each get an independent study (`pair_aware_voltage`, `pair_aware_spower`).

## Event lookup and `fault_on` placement

For each **Contingency** (dataset column 2), `_find_event_location` resolves a graph location and sets `fault_on = 1.0`. Inference uses the same rules — see [`inference.md`](inference.md#event-lookup-and-fault_on-placement).

| Event id matches | Location | `fault_on` set on |
|----------------|----------|-------------------|
| Node `id` (voltage level) | node | `data.x[..., fault_on]` |
| `busbarSectionIds` entry (NODE_BREAKER) | node | `data.x[..., fault_on]` |
| `busIds` entry (BUS_BREAKER) | node | `data.x[..., fault_on]` |
| Edge `id` (line, transformer, HVDC, connection, …) | edge | `data.edge_attr[..., fault_on]` on **both** directed half-edges |

Edge endpoint fields (`bus1`, `bus2` in edge metadata) are **not** used for event resolution — only equipment **edge `id`**.

Ids are matched exactly when possible, then via canonical normalization and safe substring fallbacks.

## Training selection score

Per-epoch checkpoints and the winning Optuna trial maximize:

```text
score = 0.40·balanced_accuracy + 0.30·macro_f1 + 0.20·accuracy + 0.10·within_one_accuracy
```

computed on validation predictions for the configured multi-class task. The score is **not** backpropagated; gradients come from the multi-term pair-aware loss only.

`selection_output` chooses which decoding path is scored when set to `auto` (best among `class` / `gated` / `log_kpi` on validation) or a fixed mode.

## Related modules

- [`pair_aware_gine`](../modules/pair_aware_gine.md), [`pair_aware_training`](../modules/pair_aware_training.md), [`voltage_training`](../modules/voltage_training.md), [`spower_training`](../modules/spower_training.md)
- [`graph_construction`](../modules/graph_construction.md), [`electric_distance`](../modules/electric_distance.md)
