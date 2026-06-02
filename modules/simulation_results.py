# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: Simulation results CSV helpers

from __future__ import annotations

import csv
from pathlib import Path

SIMULATION_RESULTS_CSV = "simulation_results.csv"
SUCCESS_STATUS = "Success"


def resolve_results_csv(results_dir: Path) -> Path:
    """Return the simulation results CSV under ``Simulations_Scenarios/``."""
    return results_dir / SIMULATION_RESULTS_CSV


def read_simulation_results(results_csv: Path) -> dict[tuple[str, str], str]:
    """Load ``(operating_point, contingency) -> status`` from the results CSV."""
    if not results_csv.exists():
        return {}

    results: dict[tuple[str, str], str] = {}
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            operating_point = row.get("Operating Point", "").strip()
            contingency = row.get("Contingency", "").strip()
            status = row.get("Status", "").strip()
            if operating_point and contingency:
                results[(operating_point, contingency)] = status
    return results


def load_successful_runs(results_csv: Path) -> set[tuple[str, str]]:
    """Return successful ``(operating_point, contingency)`` pairs from the results CSV."""
    return {
        key
        for key, status in read_simulation_results(results_csv).items()
        if status == SUCCESS_STATUS
    }


def list_successful_contingency_dirs(
    op_dir: Path,
    successful_runs: set[tuple[str, str]],
) -> list[Path]:
    """Return contingency folders for successful simulations in one operating point."""
    op_name = op_dir.name
    return sorted(
        [
            path
            for path in op_dir.iterdir()
            if path.is_dir()
            and path.name.startswith("contingency_")
            and (op_name, path.name) in successful_runs
        ],
        key=lambda path: path.name,
    )
