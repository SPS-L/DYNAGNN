# Tools (outside the pipeline)

Optional utilities that are **not** stages of `main.py`. Use them when you need extra analysis; the core training pipeline does not call these scripts.

## Physically aware KPI cut optimization

**Script:** [`physically_aware_kpi_cut_optimization.py`](physically_aware_kpi_cut_optimization.py)

Recommends **raw KPI class cuts** (`kpi.class_bins.*.cuts` in `config.yaml`) from **training** scenarios only. Candidates are scored for class-balance usability and post-event curve morphology (physical severity), then ranked on a Pareto front.

This is **one optional method**. You may instead:

- choose cuts manually from training KPI histograms / percentiles, or
- use any other cut-selection procedure you prefer.

Always keep the first cut at or above the solver noise floor guidance in [`docs/HowTo.md`](../docs/HowTo.md#kpi-cut-thresholds--recommendations).

### Prerequisites

Run the pipeline at least through **`split`** so the case data directory contains:

- `KPI/KPI_voltage.csv`, `KPI/KPI_spower.csv`
- `Dataset/train_val_test_split.csv`
- `Simulations_Scenarios/` (with `curves.xml` under successful scenarios)
- `generator_Snom/`, `inputs/contingencies.csv`

```bash
python3 main.py --to-step split
```

### Usage

From the DYNAGNN repository root (with your usual env activated):

```bash
# Default: bundled Nordic example (examples/Nordic/data)
python3 tools/physically_aware_kpi_cut_optimization.py

# Your own network (same layout as config data.path)
python3 tools/physically_aware_kpi_cut_optimization.py \
  --data-path /absolute/path/to/my_network/data
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `--data-path` | Network `data/` directory (default: `examples/Nordic/data`) |
| `--output-dir` | Where results are written (default: `<data-path>/kpi_cut_optimization`) |
| `--shared-cuts` / `--no-shared-cuts` | One absolute cut set for voltage and spower, or independent |
| `--noise-floor` | Class-0 floor ε used in the candidate grid |
| `--help` | Full CLI |

### Outputs

Under the output directory (default `<data-path>/kpi_cut_optimization/`):

- `recommendation.json` — recommended voltage / spower (and optional shared) cuts
- `metrics_*.csv`, `pareto_*.csv`, `candidates.csv`
- `plots/` — usability vs physical trade-off figures
- `run_settings.json` — resolved CLI settings

Copy the recommended thresholds into `config.yaml`:

```yaml
kpi:
  class_bins:
    voltage:
      cuts: [...]   # from recommendation.json
    spower:
      cuts: [...]
```

Then continue the pipeline:

```bash
python3 main.py --from-step dataset
```
