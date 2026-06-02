# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: Dynawo execution and simulation logging

from datetime import datetime
import os
from pathlib import Path
from typing import Literal, Optional, Tuple, Union
import subprocess

LogLevel = Literal["INFO", "WARN", "ERROR"]

# Matches Dynawo dynamo.log section separators.
LOG_SECTION_SEPARATOR = " ============================================================ "

SCENARIO_LOG_DIR = Path("output") / "log"
SCENARIO_LOG_PATTERNS = ("*.log", "*.txt", "*.xml")


def scenario_log_dir(scenario_dir: Path) -> Path:
    """Return ``<scenario_dir>/output/log``."""
    return scenario_dir / SCENARIO_LOG_DIR


def find_scenario_log_file(scenario_dir: Path) -> Optional[Path]:
    """Return the Dynawo log file under ``output/log`` (``.log``, ``.txt``, or ``.xml``)."""
    log_dir = scenario_log_dir(scenario_dir)
    if not log_dir.is_dir():
        return None

    matches: list[Path] = []
    for pattern in SCENARIO_LOG_PATTERNS:
        matches.extend(log_dir.glob(pattern))
    matches = sorted(path for path in matches if path.is_file())
    return matches[0] if matches else None


def dynawo_execution_path_from_config(config_path: Union[str, Path]) -> Optional[Path]:
    """Return the Dynawo execution path from a project YAML config, if present."""
    config_path = Path(config_path)
    if not config_path.exists():
        return None

    try:
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        dynawo_path = (config.get("dynawo") or {}).get("path")
        return Path(dynawo_path).expanduser() if dynawo_path else None
    except ModuleNotFoundError:
        # Lightweight fallback for this project's simple config shape.
        in_dynawo = False
        for raw_line in config_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.rstrip()
            if line.strip() == "dynawo:":
                in_dynawo = True
                continue
            if in_dynawo and line and not line.startswith((" ", "\t")):
                in_dynawo = False
            if in_dynawo and line.strip().startswith("path:"):
                value = line.split(":", 1)[1].strip().strip('"').strip("'")
                return Path(value).expanduser() if value else None

    return None


def default_dynawo_execution_path(
    script_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Return the Dynawo launcher path from env, config, or the legacy Deliverable layout."""
    env_path = os.environ.get("DYNAWO_EXECUTION_PATH")
    if env_path:
        return Path(env_path).expanduser()

    if config_path is not None:
        config_dynawo_path = dynawo_execution_path_from_config(config_path)
        if config_dynawo_path is not None:
            return config_dynawo_path

    start = Path(script_path).resolve()
    local_config = start.parent / "config.yaml"
    config_dynawo_path = dynawo_execution_path_from_config(local_config)
    if config_dynawo_path is not None:
        return config_dynawo_path

    start_dir = start if start.is_dir() else start.parent
    for parent in (start_dir, *start_dir.parents):
        if parent.name == "Deliverable":
            return parent.parent / "dynawo-rte" / "myEnvDynawoRTE.sh"

    raise ValueError(
        f"Cannot derive Dynawo execution path from {script_path!s}: no Deliverable ancestor found."
    )


def format_simulation_log_line(
    message: str,
    level: LogLevel = "INFO",
    timestamp: Optional[str] = None,
) -> str:
    """Format one line like the pipeline logger: ``YYYY-MM-DD HH:MM:SS.mmm | LEVEL | message``."""
    if timestamp is None:
        now = datetime.now()
        ts = now.strftime("%Y-%m-%d %H:%M:%S") + f".{int(now.microsecond / 1000):03d}"
    else:
        ts = timestamp
    return f"{ts} | {level} | {message}"


def append_simulation_log(
    log_file: Union[str, Path],
    *entries: Tuple[LogLevel, str],
    section_break: bool = True,
) -> None:
    """Append simulation log lines in dynamo.log format."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        for level, message in entries:
            handle.write(format_simulation_log_line(message, level) + "\n")
        if section_break:
            handle.write(format_simulation_log_line(LOG_SECTION_SEPARATOR, "INFO") + "\n")


def write_simulation_log_header(log_file: Union[str, Path], dynagnn_version: str) -> None:
    """Write DYNAGNN version banner at the top of a new simulations log (dynamo.log-style)."""
    log_path = Path(log_file)
    if log_path.exists() and log_path.stat().st_size > 0:
        return

    append_simulation_log(
        log_path,
        ("INFO", LOG_SECTION_SEPARATOR),
        ("INFO", f"DYNAGNN VERSION     :     {dynagnn_version}"),
        ("INFO", LOG_SECTION_SEPARATOR),
        section_break=False,
    )


def run_dynawo_job(
    execution_path: str,
    jobs_file: Union[str, Path],
    operating_point: str,
    contingency: str,
    log_file: Union[str, Path],
    run_label: Optional[str] = None,
) -> bool:
    """Execute a Dynawo job and append a standardized log entry.

    Returns True when the command output indicates success.
    """
    cmd = f"{execution_path} jobs {jobs_file}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output = result.stdout + result.stderr

    phase_suffix = f" ({run_label})" if run_label else ""
    run_message = f"Run simulation {operating_point} {contingency}{phase_suffix}"
    success = "succeeded" in output

    entries: list[Tuple[LogLevel, str]] = [("INFO", run_message)]
    if success:
        entries.append(("INFO", "Dynawo job successful"))
    else:
        entries.append(("ERROR", "Dynawo job failed"))
        stderr_text = (result.stderr or result.stdout or "").strip()
        if stderr_text:
            for line in stderr_text.splitlines():
                stripped = line.strip()
                if stripped:
                    entries.append(("ERROR", stripped))

    append_simulation_log(log_file, *entries)
    return success
