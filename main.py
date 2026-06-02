# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: End-to-end pipeline entry point

from __future__ import annotations

import sys
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: PyYAML. Install it with: pip install pyyaml") from exc

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.dynawo_runner import write_simulation_log_header
from modules.paths import CONFIG_PATH, DATA_DIR
from modules.pipeline_logging import configure_pipeline_logging, get_logger, get_pipeline_log_path, log_step_banner
from src import build_op_assets, curves_post_process, dataset_construction, simulate, training

MODEL_DIR = DATA_DIR / "model"
VOLTAGE_MODEL = MODEL_DIR / "gat_voltage_best_model.pt"
SPOWER_MODEL = MODEL_DIR / "gat_spower_best_model.pt"


def _dynagnn_version() -> str:
    if not CONFIG_PATH.exists():
        return "1"
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    version = (config.get("dynagnn") or {}).get("version")
    return str(version) if version is not None else "1"


def main() -> None:
    log_path = DATA_DIR / "dynagnn.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    configure_pipeline_logging()
    write_simulation_log_header(get_pipeline_log_path(), _dynagnn_version())
    logger = get_logger()

    logger.info("DYNAGNN pipeline started.")
    logger.info("Expected outputs when complete: %s, %s", VOLTAGE_MODEL, SPOWER_MODEL)

    steps = (
        simulate.main,
        build_op_assets.main,
        curves_post_process.main,
        dataset_construction.main,
        training.main,
    )

    for step in steps:
        step()

    if not VOLTAGE_MODEL.is_file() or not SPOWER_MODEL.is_file():
        raise SystemExit(
            f"Pipeline finished but trained models are missing. Expected:\n"
            f"  {VOLTAGE_MODEL}\n"
            f"  {SPOWER_MODEL}\n"
            f"See log: {get_pipeline_log_path()}"
        )

    logger.info("DYNAGNN pipeline completed successfully.")
    logger.info("Trained models: %s", VOLTAGE_MODEL)
    logger.info("Trained models: %s", SPOWER_MODEL)


if __name__ == "__main__":
    main()
