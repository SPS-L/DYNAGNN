# `event_files.py`

Writes Dynawo **`Events.dyd`** and **`Events.par`** for a single contingency (line or transformer trip, bus/load open, or generator trip).

## Used by

- `main.py` (via `src/simulate.py`, `create_scenario`)

## Inputs

| Parameter | Source |
|-----------|--------|
| `contingency_id`, `fault_name`, `fault_type` | `contingencies.csv` row (type normalized on read) |
| `event_time` | `config.yaml` → `simulation.event_time` |

## Outputs

In each `Simulations_Scenarios/.../contingency_<id>/`:

- `Events.dyd` — blackBoxModel + connect to `NETWORK` or generator signal
- `Events.par` — disconnect time and parameters
- In-place edit of `*.jobs` → `<dynModels dydFile="Events.dyd"/>` (unprefixed jobs) or `<dyn:dynModels dydFile="Events.dyd"/>` (prefixed `dyn:` jobs)
- `*.jobs` — `<dyn:dynModels dydFile="Events.dyd"/>` (or unprefixed equivalent) inserted in the modeler block when missing

## Main API

| Function | Description |
|----------|-------------|
| `normalize_fault_type(fault_type)` | Validate CSV **Type** (`line`, `bus`, `generator`, `transformer`, `load`; `bus` → `busbarsection` internally) |
| `write_event_files(scenario_dir, contingency_id, fault_name, fault_type)` | Write both files and patch jobs |
| `patch_jobs_events_reference`, `update_jobs_events_reference` | Jobs XML helpers (prefix-aware, same style as curve/init patching) |
| `build_dyd_single`, `build_par_single` | XML string builders |
| `lib_from_fault_type(fault_type)` | Dynawo event library name |
| `default_event_time(config_path)` | Read event time from config |

## Type → Dynawo mapping

| CSV **Type** | Normalized | Fault name | Dynawo library | Parameter |
|--------------|------------|------------|----------------|-----------|
| `line` | `line` | IIDM line id | `EventQuadripoleDisconnection` | `event_disconnectOrigin` / `Extremity` |
| `transformer` | `transformer` | IIDM two-winding transformer id | `EventQuadripoleDisconnection` | same as line |
| `bus` | `busbarsection` | `bus` id (bus-breaker) or `busbarSection` id (node-breaker) | `EventConnectedStatus` | `event_open` |
| `load` | `load` | IIDM load id | `EventConnectedStatus` | `event_open` |
| `generator` | `generator` | Dynamic model id in `.dyd` (not IIDM static id) | `EventSetPointBoolean` | `event_stateEvent1` |

**Contingency ID** suffix letters (convention): `l` line, `b` bus, `g` generator, `t` transformer, `lo` load (e.g. `1lo`).
