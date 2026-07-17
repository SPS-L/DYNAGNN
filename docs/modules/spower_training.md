# `spower_training.py`

Thin entry point for **Spower** (generator) pair-aware GINE Optuna training.

## Used by

- `src/training.py`

## Main API

| Function | Description |
|----------|-------------|
| `run_spower_training(...)` | Delegates to `pair_aware_training.run_task_training(task="spower", ...)` |

Targets generator nodes via `gen_node_mask`. Deployment checkpoint: `data/model/<study_name>/spower_best_model.pt`.

## Related modules

- [`pair_aware_training`](pair_aware_training.md), [`pair_aware_gine`](pair_aware_gine.md), [`voltage_training`](voltage_training.md)
- [`src/training.md`](../src/training.md)
