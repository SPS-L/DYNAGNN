# `voltage_training.py`

Thin entry point for **Voltage** (bus) pair-aware GINE Optuna training.

## Used by

- `src/training.py`

## Main API

| Function | Description |
|----------|-------------|
| `run_voltage_training(...)` | Delegates to `pair_aware_training.run_task_training(task="voltage", ...)` |

Targets bus nodes via `bus_node_mask`. Deployment checkpoint: `data/model/<study_name>/voltage_best_model.pt`.

## Related modules

- [`pair_aware_training`](pair_aware_training.md), [`pair_aware_gine`](pair_aware_gine.md), [`spower_training`](spower_training.md)
- [`src/training.md`](../src/training.md)
