# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: End-to-end pipeline entry point

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: PyYAML. Install it with: pip install pyyaml") from exc

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.dynawo_runner import write_simulation_log_header
from modules.paths import CONFIG_PATH, DATA_DIR
from modules.pipeline_logging import configure_pipeline_logging, get_logger, get_pipeline_log_path
from src import build_op_assets, curves_post_process, dataset_construction, simulate, training

MODEL_DIR = DATA_DIR / "model"

with CONFIG_PATH.open("r", encoding="utf-8") as handle:
    _CONFIG = yaml.safe_load(handle) or {}

MODEL_ARCHITECTURE = str(
    (_CONFIG.get("model") or {}).get("architecture", "gat_coral")
).strip().lower()

if MODEL_ARCHITECTURE == "pair_aware_gine":
    VOLTAGE_MODEL = MODEL_DIR / "pair_aware_voltage_best_model.pt"
    SPOWER_MODEL = MODEL_DIR / "pair_aware_spower_best_model.pt"
elif MODEL_ARCHITECTURE == "gat_coral":
    VOLTAGE_MODEL = MODEL_DIR / "gat_voltage_best_model.pt"
    SPOWER_MODEL = MODEL_DIR / "gat_spower_best_model.pt"
else:
    raise SystemExit(
        "Unsupported model.architecture: "
        f"{MODEL_ARCHITECTURE!r}. Expected 'gat_coral' or 'pair_aware_gine'."
    )

PIPELINE_STEPS: tuple[tuple[str, Callable[[], None]], ...] = (
    ("simulate", simulate.main),
    ("build_op_assets", build_op_assets.main),
    ("curve_process", curves_post_process.main),
    ("dataset", dataset_construction.main),
    ("training", training.main),
)

STEP_INDEX = {name: index for index, (name, _) in enumerate(PIPELINE_STEPS)}
RESUME_FROM_STEP_CHOICES = tuple(name for name, _ in PIPELINE_STEPS[1:])
TO_STEP_CHOICES = tuple(name for name, _ in PIPELINE_STEPS)


def _parse_args() -> argparse.Namespace:
    step_names = ", ".join(name for name, _ in PIPELINE_STEPS)
    parser = argparse.ArgumentParser(
        description="Run the DYNAGNN training pipeline end to end or resume from a later stage.",
    )
    parser.add_argument(
        "--from-step",
        choices=RESUME_FROM_STEP_CHOICES,
        default=None,
        metavar="STEP",
        help=(
            "Resume from this stage instead of running the full pipeline. "
            f"Stages in order: {step_names}. "
            "Choices: build_op_assets, curve_process, dataset, training "
            "(requires outputs from earlier stages; see docs/HowTo.md)."
        ),
    )
    parser.add_argument(
        "--to-step",
        choices=TO_STEP_CHOICES,
        default=None,
        metavar="STEP",
        help=(
            "Stop after this stage (inclusive). "
            f"Stages in order: {step_names}. "
            "Useful to run through curve_process for KPI cut analysis before dataset/training."
        ),
    )
    return parser.parse_args()


def _dynagnn_version() -> str:
    if not CONFIG_PATH.exists():
        return "1"
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    version = (config.get("dynagnn") or {}).get("version")
    return str(version) if version is not None else "1"


def main() -> None:
    args = _parse_args()
    start_index = 0 if args.from_step is None else STEP_INDEX[args.from_step]
    end_index = len(PIPELINE_STEPS) - 1 if args.to_step is None else STEP_INDEX[args.to_step]

    if start_index > end_index:
        raise SystemExit(
            f"--from-step {args.from_step!r} is after --to-step {args.to_step!r}. "
            "Choose a start stage that comes before or equal to the stop stage."
        )

    log_path = DATA_DIR / "dynagnn.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    configure_pipeline_logging()
    write_simulation_log_header(get_pipeline_log_path(), _dynagnn_version())
    logger = get_logger()

    logger.info("DYNAGNN pipeline started.")
    if start_index > 0:
        skipped = ", ".join(name for name, _ in PIPELINE_STEPS[:start_index])
        logger.info("Resuming from step %s (skipping: %s).", args.from_step, skipped)
    if args.to_step is not None:
        logger.info("Stopping after step %s.", args.to_step)
    if end_index == STEP_INDEX["training"]:
        logger.info("Expected outputs when complete: %s, %s", VOLTAGE_MODEL, SPOWER_MODEL)

    for step_name, step in PIPELINE_STEPS[start_index : end_index + 1]:
        step()

    if end_index < STEP_INDEX["training"]:
        logger.info("DYNAGNN pipeline stopped after %s.", PIPELINE_STEPS[end_index][0])
        return

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
