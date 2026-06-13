# `paths.py`

Central **path constants** for the DYNAGNN project tree. All paths under the data root are resolved from **`data.path`** in `config.yaml` (absolute path, or relative to the repository root).

## Used by

- All `src/` pipeline scripts and most `modules/` packages

## Constants

| Name | Path |
|------|------|
| `PROJECT_ROOT` | Repository root |
| `CONFIG_PATH` | `config.yaml` |
| `DATA_DIR` | `<data.path>/` |
| `INPUTS_DIR` | `<data.path>/inputs/` |
| `KPI_DIR` | `<data.path>/KPI/` |
| `ACTIONS_DIR` | `<data.path>/Actions/` |
| `DISCONNECTIONS_DIR` | `<data.path>/Disconnections/` |
| `DATASET_DIR` | `<data.path>/Dataset/` |
| `NORMALIZATION_DIR` | `<data.path>/normalization/` |
| `OP_GRAPHS_DIR` | `<data.path>/op_graphs/` |
| `OP_ELECTRIC_DISTANCE_DIR` | `<data.path>/op_electric_distance/` |
| `SNOM_DIR` | `<data.path>/generator_Snom/` |
| `CONTINGENCIES_CSV` | `<data.path>/inputs/contingencies.csv` |
| `SIMULATIONS_DIR` | `<data.path>/Simulations_Scenarios/` |

## Functions

| Function | Description |
|----------|-------------|
| `load_config(config_path)` | Load `config.yaml` |
| `resolve_data_dir(config)` | Resolve `data.path` with `~` expansion |
| `snom_csv_for_operating_point(op_name)` | `<data.path>/generator_Snom/<op_name>.csv` |
