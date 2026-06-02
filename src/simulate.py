# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: Dynawo contingency simulation driver

from __future__ import annotations

import csv
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: PyYAML. Install it with: pip install pyyaml") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.curve_generation import generate_curves, op_sort_key
from modules.dynawo_runner import append_simulation_log, find_scenario_log_file, run_dynawo_job
from modules.event_files import write_event_files
from modules.initialization import (
    InitResult,
    initialize_operating_points,
    resolve_initialization_duration,
    write_initialization_status_log,
)
from modules.paths import CONFIG_PATH, INPUTS_DIR, SIMULATIONS_DIR
from modules.pipeline_logging import get_logger, get_pipeline_log_path, log_step_banner
from modules.simulation_results import read_simulation_results, resolve_results_csv

DEFAULT_CONFIG = CONFIG_PATH
DYNAMO_NONLINEAR_ITERATION_WARN = "the maximum number of nonlinear iterations has been reached"


def resolve_results_csv_path(results_dir: Path) -> Path:
    """Return the single simulation results CSV used for resume/skip across all OPs."""
    results_dir.mkdir(parents=True, exist_ok=True)
    return resolve_results_csv(results_dir)


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_dynagnn_version(config: dict) -> str:
    version = config.get("dynagnn", {}).get("version")
    if version is None:
        return "1"
    return str(version)


def resolve_dynawo_path(config: dict) -> str:
    dynawo_path = config.get("dynawo", {}).get("path")
    if not dynawo_path:
        raise RuntimeError("Dynawo path is missing. Set dynawo.path in config.yaml.")

    path = Path(dynawo_path).expanduser()
    if path.is_dir():
        candidates = [
            path / "myEnvDynawoRTE.sh",
            path / "dynawo.sh",
            path.parent / "dynawo-rte" / "myEnvDynawoRTE.sh",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

    return str(path)


def read_contingencies(csv_path: Path) -> list[tuple[str, str, str, Optional[set[str]]]]:
    rows: list[tuple[str, str, str, Optional[set[str]]]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue

            contingency_id = row[0].strip()
            fault_name = row[1].strip()
            fault_type = row[2].strip()
            op_raw = row[3].strip() if len(row) >= 4 else ""

            if not contingency_id or not fault_name or not fault_type:
                continue

            op_keys = None
            if op_raw:
                op_keys = {part.strip() for part in op_raw.split(",") if part.strip()}

            rows.append((contingency_id, fault_name, fault_type, op_keys))

    return rows


def find_contingencies_csv(inputs_dir: Path) -> Path:
    preferred = inputs_dir / "contingencies.csv"
    if preferred.exists():
        return preferred
    raise RuntimeError(f"No contingency CSV found at {preferred}")


def extract_op_key(folder_name: str) -> str:
    prefix = "operating_point_"
    return folder_name[len(prefix) :] if folder_name.startswith(prefix) else folder_name


def should_contingency_exist(op_key: str, op_keys: Optional[set[str]]) -> bool:
    return op_keys is None or op_key in op_keys


def discover_operating_points(inputs_dir: Path) -> list[Path]:
    return sorted(
        [path for path in inputs_dir.iterdir() if path.is_dir() and path.name.startswith("operating_point_")],
        key=op_sort_key,
    )


def copy_operating_point_contents(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        if item.suffix.lower() == ".png":
            continue
        dst_item = dst_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst_item, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst_item)


def find_jobs_file(scenario_dir: Path) -> Optional[Path]:
    jobs_files = sorted(scenario_dir.glob("*.jobs"))
    return jobs_files[0] if jobs_files else None


def create_scenario(
    op_path: Path,
    output_dir: Path,
    contingency_id: str,
    fault_name: str,
    fault_type: str,
) -> Path:
    scenario_dir = output_dir / op_path.name / f"contingency_{contingency_id}"
    copy_operating_point_contents(op_path, scenario_dir)
    write_event_files(scenario_dir, contingency_id, fault_name, fault_type)
    return scenario_dir


def write_result(results_csv: Path, operating_point: str, contingency: str, status: str) -> None:
    rows = []
    headers = ["Operating Point", "Contingency", "Status"]
    updated = False

    if results_csv.exists():
        with results_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if (
                    row.get("Operating Point", "").strip() == operating_point
                    and row.get("Contingency", "").strip() == contingency
                ):
                    row["Status"] = status
                    updated = True
                rows.append(
                    {
                        "Operating Point": row.get("Operating Point", "").strip(),
                        "Contingency": row.get("Contingency", "").strip(),
                        "Status": row.get("Status", "").strip(),
                    }
                )

    if not updated:
        rows.append(
            {
                "Operating Point": operating_point,
                "Contingency": contingency,
                "Status": status,
            }
        )

    tmp_path = results_csv.with_name(results_csv.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([row["Operating Point"], row["Contingency"], row["Status"]])
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, results_csv)


def write_skipped_log(log_file: Path, operating_point: str, contingency: str) -> None:
    append_simulation_log(
        log_file,
        ("INFO", f"Skip simulation {operating_point} {contingency} (already successful)"),
    )


def find_nonlinear_iteration_warn_lines(scenario_dir: Path) -> list[str]:
    scenario_log = find_scenario_log_file(scenario_dir)
    if scenario_log is None:
        return []

    matches: list[str] = []
    with scenario_log.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if DYNAMO_NONLINEAR_ITERATION_WARN in line and "| WARN |" in line:
                matches.append(line.rstrip("\n"))
    return matches


def write_nonlinear_iteration_warnings(log_file: Path, scenario_dir: Path) -> None:
    warn_lines = find_nonlinear_iteration_warn_lines(scenario_dir)
    if not warn_lines:
        return

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        for line in warn_lines:
            handle.write(line + "\n")
    append_simulation_log(log_file)


def write_jobs_missing_log(log_file: Path, operating_point: str, contingency: str, scenario_dir: Path) -> None:
    append_simulation_log(
        log_file,
        ("INFO", f"Run simulation {operating_point} {contingency}"),
        ("ERROR", f"No .jobs file found in {scenario_dir}"),
    )


def main() -> None:
    log_step_banner("simulate")
    logger = get_logger()

    config = load_config(DEFAULT_CONFIG)
    dynagnn_version = resolve_dynagnn_version(config)
    execution_path = resolve_dynawo_path(config)
    log_file = get_pipeline_log_path()

    inputs_dir = INPUTS_DIR.resolve()
    contingencies_csv = find_contingencies_csv(inputs_dir)
    contingencies = read_contingencies(contingencies_csv)
    if not contingencies:
        raise RuntimeError(f"No contingencies found in {contingencies_csv}")

    operating_points = discover_operating_points(inputs_dir)
    if not operating_points:
        raise RuntimeError(f"No operating point folders found in {inputs_dir}")

    output_dir = SIMULATIONS_DIR.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir

    initialization_duration = resolve_initialization_duration(config)
    init_results: dict[str, InitResult] = {}
    if initialization_duration is not None:
        logger.info("Initializing operating points (initialization_duration=%gs)...", initialization_duration)
        init_results = initialize_operating_points(
            operating_points,
            execution_path=execution_path,
            initialization_duration=initialization_duration,
            log_file=log_file,
            dynagnn_version=dynagnn_version,
        )
        failed_inits = [(name, result) for name, result in init_results.items() if not result.success]
        if failed_inits:
            for name, result in failed_inits:
                write_initialization_status_log(log_file, result)
            failed_names = ", ".join(sorted(name for name, _ in failed_inits))
            logger.error("Operating point initialization failed for: %s", failed_names)
            raise SystemExit(1)
    else:
        logger.info("Skipping operating point initialization (initialization_duration not set or <= 0).")

    logger.info("Generating curve files for all operating points...")
    generate_curves()

    logger.info("Inputs: %s", inputs_dir)
    logger.info("Contingencies: %s", contingencies_csv)
    logger.info("Output: %s", output_dir)
    logger.info("Dynawo execution path: %s", execution_path)

    results_csv = resolve_results_csv_path(results_dir)
    existing_results = read_simulation_results(results_csv)
    if existing_results:
        logger.info(
            "Loaded %d existing result(s) from %s; successful rows will be skipped.",
            len(existing_results),
            results_csv.name,
        )

    total_created = 0
    total_success = 0
    total_failed = 0

    for op_idx, op_path in enumerate(operating_points, start=1):
        op_key = extract_op_key(op_path.name)
        contingencies_for_op = [row for row in contingencies if should_contingency_exist(op_key, row[3])]

        write_initialization_status_log(log_file, init_results.get(op_path.name))

        logger.info(
            "%d/%d %s: %d contingencies",
            op_idx,
            len(operating_points),
            op_path.name,
            len(contingencies_for_op),
        )

        for contingency_id, fault_name, fault_type, _ in contingencies_for_op:
            contingency_name = f"contingency_{contingency_id}"
            result_key = (op_path.name, contingency_name)

            if existing_results.get(result_key) == "Success":
                logger.info("  Skipping %s (already successful)", contingency_name)
                write_skipped_log(log_file, op_path.name, contingency_name)
                continue

            if existing_results.get(result_key) == "Failed":
                logger.info("  Retrying %s (previous run failed)", contingency_name)
            else:
                logger.info("  Creating %s", contingency_name)

            scenario_dir = output_dir / op_path.name / contingency_name
            if scenario_dir.exists():
                logger.info("    Scenario folder already exists")
            else:
                scenario_dir = create_scenario(op_path, output_dir, contingency_id, fault_name, fault_type)
                total_created += 1

            jobs_file = find_jobs_file(scenario_dir)
            if jobs_file is None:
                logger.error("    No .jobs file found in %s", scenario_dir)
                write_jobs_missing_log(log_file, op_path.name, contingency_name, scenario_dir)
                write_result(results_csv, op_path.name, contingency_name, "Failed")
                existing_results[result_key] = "Failed"
                total_failed += 1
                continue

            logger.info("    Running Dynawo")
            success = run_dynawo_job(
                execution_path=execution_path,
                jobs_file=jobs_file,
                operating_point=op_path.name,
                contingency=contingency_name,
                log_file=log_file,
            )
            status = "Success" if success else "Failed"
            write_result(results_csv, op_path.name, contingency_name, status)
            existing_results[result_key] = status
            write_nonlinear_iteration_warnings(log_file, scenario_dir)
            if success:
                total_success += 1
            else:
                total_failed += 1

    logger.info("Simulation pipeline completed.")
    logger.info("Scenarios created: %d", total_created)
    logger.info("Successful runs: %d", total_success)
    logger.info("Failed runs: %d", total_failed)
    logger.info("Simulation results CSV: %s", results_csv)
