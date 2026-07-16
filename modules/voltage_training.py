# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: Voltage pair-aware GINE Optuna training entry point
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from modules.pair_aware_training import run_task_training


def run_voltage_training(
    *,
    train_scaled: list,
    val_scaled: list,
    test_scaled: list,
    training_dir: Path,
    model_dir: Path,
    config: dict,
    attachment: dict[str, Any],
    logger: logging.Logger,
) -> dict:
    return run_task_training(
        task="voltage",
        train_scaled=train_scaled,
        val_scaled=val_scaled,
        test_scaled=test_scaled,
        training_dir=training_dir,
        model_dir=model_dir,
        config=config,
        attachment=attachment,
        logger=logger,
    )
