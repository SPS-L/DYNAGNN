# `pair_aware_inference.py`

Helpers to load pair-aware deployment checkpoints and run a single-scenario forward pass for Voltage or Spower.

## Used by

- `DYNAGNN.py`

## Main API

| Function | Description |
|----------|-------------|
| `load_pair_aware_checkpoint(path, expected_task=...)` | Load and validate a deployment `.pt` dict |
| `load_pair_aware_model(checkpoint, device)` | Build `PairAwareGINE` and load weights |
| `predict_pair_aware(model=..., sample_cpu=..., checkpoint=..., device=...)` | One predicted class per target component |

## Checkpoint requirements

Expects `model_type == "pair_aware_gine"` and fields including: `model_state_dict`, `hparams`, vocab sizes/maps, `selected_output`, `cuts`, log-KPI mean/std, `epsilon`, `gate_threshold`.

## Per-scenario prep

Before the forward pass, inference attaches:

- node tokens from the checkpoint vocabulary;
- contingency token for the scenario event id;
- event node/edge masks from the graph’s event location (same rules as training).

Decoding follows `selected_output` (`class`, `gated`, or `log_kpi`). External output remains one integer class per target component.

## Related modules

- [`pair_aware_gine`](pair_aware_gine.md)
- [`src/inference.md`](../src/inference.md)
