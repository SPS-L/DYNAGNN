# Pair-aware six-class GINE

DYNAGNN supports two model architectures through `config.yaml`:

```yaml
model:
  architecture: pair_aware_gine  # or gat_coral
  num_classes: 6
```

`pair_aware_gine` predicts classes 0–5 directly. The model uses a residual
edge-aware GINE encoder, layer concatenation, target–contingency interaction
features, graph mean/max context, a class-0 gate, and an auxiliary log-KPI
regression head.

## Training

Use the normal pipeline:

```bash
python3 main.py --from-step training
```

The standard graph construction, split CSV, electrical-distance feature, and
training-only feature scaling are reused. The new checkpoints are written to:

```text
<data.path>/model/pair_aware_voltage_best_model.pt
<data.path>/model/pair_aware_spower_best_model.pt
```

Each deployment checkpoint contains the model state, hyperparameters, selected
validation decoding strategy, KPI thresholds/statistics, and the shared node
and contingency vocabularies.

## Inference

Use the existing interface:

```bash
python3 DYNAGNN.py --case-dir /path/to/operating_point --events-csv /path/to/events.csv
```

The output format is unchanged: one predicted activity class per component in
`prediction_voltage.csv` and `prediction_spower.csv`.

## Configuration

The default pair-aware settings are:

```yaml
model:
  pair_aware:
    hidden_dim: 128
    node_id_dim: 24
    contingency_id_dim: 32
    type_dim: 8
    pair_dim: 32
    op_context_embedding_dim: 32
    num_gnn_layers: 3
    decoder_hidden_dim: 256
    dropout: 0.15

training:
  pair_aware:
    op_context_mode: none
    lr: 0.0002
    weight_decay: 0.00001
    classification_weight: 1.0
    regression_weight: 0.30
    inactive_gate_weight: 0.20
    ordinal_weight: 0.10
    class_weight_mode: sqrt_inverse
    gate_pos_weight_mode: balanced
    gate_threshold: 0.50
    epsilon: 1.0e-10
    selection_output: auto
```

The repository integration currently requires `op_context_mode: none`. The
operating condition still enters through node/edge electrical features and
mean/max graph pooling.

Set `model.architecture: gat_coral` to use the legacy implementation and its
existing checkpoint names.
