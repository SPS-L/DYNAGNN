# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: Train/validation/test split generation stage

from __future__ import annotations

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules import dataset_split
from modules.paths import CONFIG_PATH, DATASET_DIR, KPI_DIR
from modules.pipeline_logging import get_logger, log_step_banner


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> None:
    log_step_banner("split")
    logger = get_logger()

    kpi_voltage_path = KPI_DIR / "KPI_voltage.csv"
    if not kpi_voltage_path.is_file():
        raise FileNotFoundError(
            f"Missing combined KPI table: {kpi_voltage_path}. "
            "Run the curve_process stage first (main.py --to-step curve_process)."
        )

    split_csv = DATASET_DIR / "train_val_test_split.csv"
    config = load_config()

    logger.info("Building train/validation/test split from %s", kpi_voltage_path.name)
    split_summary = dataset_split.build_dataset_split(
        kpi_voltage_path,
        output_csv=split_csv,
        config=config,
    )

    logger.info(
        "Split built. total=%d train=%d val=%d test=%d mode=%s seed=%d",
        split_summary.total_examples,
        split_summary.train_examples,
        split_summary.validation_examples,
        split_summary.test_examples,
        split_summary.split_mode,
        split_summary.seed,
    )
    logger.info("Split CSV: %s", split_csv)
    logger.info("split completed.")
