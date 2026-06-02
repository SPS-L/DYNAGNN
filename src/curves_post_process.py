# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: KPI, actions, and disconnections post-processing

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.actions_detection import run_actions_detection
from modules.disconnections_detection import run_disconnections_detection
from modules.kpi import run_kpi
from modules.paths import SIMULATIONS_DIR
from modules.pipeline_logging import get_logger, log_step_banner
from modules.simulation_results import load_successful_runs, resolve_results_csv


def main() -> None:
    log_step_banner("curves_post_process")
    logger = get_logger()

    results_csv = resolve_results_csv(SIMULATIONS_DIR)
    successful_runs = load_successful_runs(results_csv)
    if not successful_runs:
        raise RuntimeError(
            f"No successful simulations found in {results_csv}. "
            "Run simulate.py first and ensure at least one scenario succeeds."
        )
    logger.info(
        "Post-processing %d successful simulation(s) from %s",
        len(successful_runs),
        results_csv.name,
    )

    logger.info("Step 1/3: KPI extraction")
    run_kpi(successful_runs=successful_runs)

    logger.info("Step 2/3: Actions detection")
    run_actions_detection(successful_runs=successful_runs)

    logger.info("Step 3/3: Disconnections detection")
    run_disconnections_detection(successful_runs=successful_runs)

    logger.info("curves_post_process completed.")
