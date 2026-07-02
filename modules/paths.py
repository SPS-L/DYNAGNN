# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: Project data path constants

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load ``config.yaml``; return an empty dict if the file is missing."""
    if not config_path.exists():
        return {}
    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_data_dir(
    config: Mapping[str, Any] | None = None,
    *,
    config_path: Path = CONFIG_PATH,
) -> Path:
    """Return the configured data root from ``data.path`` in ``config.yaml``."""
    cfg = dict(config) if config is not None else load_config(config_path)
    raw = (cfg.get("data") or {}).get("path")
    if not raw:
        raise RuntimeError(
            "Data directory path is missing. Set data.path in config.yaml "
            "(absolute path or path relative to the project root)."
        )

    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    else:
        path = path.resolve()
    return path


DATA_DIR = resolve_data_dir()
INPUTS_DIR = DATA_DIR / "inputs"
KPI_DIR = DATA_DIR / "KPI"
ACTIONS_DIR = DATA_DIR / "Actions"
DISCONNECTIONS_DIR = DATA_DIR / "Disconnections"
DATASET_DIR = DATA_DIR / "Dataset"
OP_GRAPHS_DIR = DATA_DIR / "op_graphs"
OP_ELECTRIC_DISTANCE_DIR = DATA_DIR / "op_electric_distance"
SNOM_DIR = DATA_DIR / "generator_Snom"
SIMULATIONS_DIR = DATA_DIR / "Simulations_Scenarios"
CONTINGENCIES_CSV = INPUTS_DIR / "contingencies.csv"


def snom_csv_for_operating_point(op_name: str) -> Path:
    """Per-OP generator SNom table under ``<data.path>/generator_Snom/``."""
    return SNOM_DIR / f"{op_name}.csv"
