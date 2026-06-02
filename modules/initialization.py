# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: Operating-point steady-state initialization

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from modules.dynawo_runner import (
    LogLevel,
    append_simulation_log,
    find_scenario_log_file,
    run_dynawo_job,
    write_simulation_log_header,
)

# Dynawo jobs XML may use unprefixed tags or namespace prefixes (e.g. dyn:network).
_XML_TAG_PREFIX = r"(?:[\w-]+:)?"

EVENTS_DYD_LINE_RE = re.compile(
    rf'^(\s*)(<{_XML_TAG_PREFIX}dynModels\s+dydFile="Events\.dyd"\s*/>)\s*$',
    re.MULTILINE,
)
SIMULATION_TAG_RE = re.compile(
    rf"<({_XML_TAG_PREFIX}simulation)\s+([^>]+)>",
    re.MULTILINE,
)
NETWORK_IIDM_RE = re.compile(
    rf'<{_XML_TAG_PREFIX}network\s+[^>]*\biidmFile="([^"]+)"',
    re.MULTILINE,
)


@dataclass(frozen=True)
class OperatingPointCase:
    op_dir: Path
    jobs_path: Path
    iidm_path: Path


@dataclass
class InitResult:
    operating_point: str
    success: bool
    run_time: float = 0.0
    skipped: bool = False
    dynawo_failed: bool = False
    messages: list[str] = field(default_factory=list)


def resolve_initialization_duration(config: dict) -> Optional[float]:
    """Return total initialization Dynawo duration (seconds), or ``None`` to skip init."""
    raw = config.get("simulation", {}).get("initialization_duration")
    if raw is None:
        return None
    run_time = float(raw)
    if run_time <= 0:
        return None
    return run_time


def find_single_jobs_file(op_dir: Path) -> Path:
    jobs_files = sorted(op_dir.glob("*.jobs"))
    if len(jobs_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .jobs in {op_dir}, found {len(jobs_files)}: "
            f"{[path.name for path in jobs_files]}"
        )
    return jobs_files[0]


def network_iidm_path(jobs_path: Path, op_dir: Path) -> Path:
    content = jobs_path.read_text(encoding="utf-8")
    match = NETWORK_IIDM_RE.search(content)
    if not match:
        raise RuntimeError(f"Could not find network iidmFile in {jobs_path}")
    return op_dir / match.group(1)


def patch_jobs_for_init(content: str, stop_time: float) -> str:
    """Comment Events.dyd when present and set stopTime on the first ``<simulation>`` tag."""
    if EVENTS_DYD_LINE_RE.search(content):
        content = EVENTS_DYD_LINE_RE.sub(
            r"\1<!-- \2 -->",
            content,
            count=1,
        )

    stop_time_str = f"{float(stop_time):g}"

    def repl(match: re.Match[str]) -> str:
        tag = match.group(1)
        attrs = match.group(2)
        if re.search(r'\bstopTime\s*=\s*"[^"]*"', attrs):
            attrs = re.sub(
                r'\bstopTime\s*=\s*"[^"]*"',
                f'stopTime="{stop_time_str}"',
                attrs,
            )
        else:
            attrs = f'{attrs.strip()} stopTime="{stop_time_str}"'
        return f"<{tag} {attrs.strip()}>"

    new_content, count = SIMULATION_TAG_RE.subn(repl, content, count=1)
    if count == 0:
        raise RuntimeError("Could not find <simulation ...> tag")
    return new_content


def discover_case(op_dir: Path) -> OperatingPointCase:
    jobs_path = find_single_jobs_file(op_dir)
    iidm_path = network_iidm_path(jobs_path, op_dir)
    if not iidm_path.exists():
        raise FileNotFoundError(f"Missing network IIDM for {op_dir.name}: {iidm_path}")
    return OperatingPointCase(op_dir=op_dir, jobs_path=jobs_path, iidm_path=iidm_path)


def final_state_iidm(op_dir: Path) -> Path:
    path = op_dir / "outputs" / "finalState" / "outputIIDM.xml"
    if not path.exists():
        raise FileNotFoundError(f"Dynawo final state IIDM not found: {path}")
    return path


def initialize_one_operating_point(
    case: OperatingPointCase,
    *,
    execution_path: str,
    run_time: float,
    log_file: Path,
    backup_iidm: bool = False,
    clean_outputs: bool = True,
) -> InitResult:
    """Run initialization for one operating point. Returns success/failure details."""
    label = case.op_dir.name
    outputs_dir = case.op_dir / "outputs"

    try:
        original_jobs = case.jobs_path.read_text(encoding="utf-8")
        patched_jobs = patch_jobs_for_init(original_jobs, stop_time=run_time)
    except Exception as exc:
        return InitResult(
            operating_point=label,
            success=False,
            run_time=run_time,
            messages=[f"patch_jobs: {exc}"],
        )

    init_succeeded = False
    case.jobs_path.write_text(patched_jobs, encoding="utf-8")
    prev_cwd = os.getcwd()
    try:
        if clean_outputs and outputs_dir.exists():
            shutil.rmtree(outputs_dir)

        os.chdir(case.op_dir)
        ok = run_dynawo_job(
            execution_path=execution_path,
            jobs_file=case.jobs_path.name,
            operating_point=label,
            contingency="initialization",
            log_file=log_file,
            run_label=f"initialization_duration={run_time:g}s",
        )
        if not ok:
            return InitResult(
                operating_point=label,
                success=False,
                run_time=run_time,
                dynawo_failed=True,
                messages=["Dynawo job failed"],
            )

        try:
            source_iidm = final_state_iidm(case.op_dir)
        except FileNotFoundError as exc:
            scenario_log = find_scenario_log_file(case.op_dir)
            msg = str(exc)
            if scenario_log is not None:
                msg = f"{msg} (see {scenario_log})"
            return InitResult(
                operating_point=label,
                success=False,
                run_time=run_time,
                messages=[f"final_state: {msg}"],
            )

        if backup_iidm:
            backup_path = case.iidm_path.with_suffix(case.iidm_path.suffix + ".pre_init.bak")
            shutil.copy2(case.iidm_path, backup_path)

        shutil.copy2(source_iidm, case.iidm_path)
        init_succeeded = True
        return InitResult(
            operating_point=label,
            success=True,
            run_time=run_time,
        )
    except Exception as exc:
        return InitResult(
            operating_point=label,
            success=False,
            run_time=run_time,
            messages=[f"unexpected: {exc}"],
        )
    finally:
        try:
            os.chdir(prev_cwd)
        except OSError:
            pass
        case.jobs_path.write_text(original_jobs, encoding="utf-8")
        if clean_outputs and init_succeeded and outputs_dir.exists():
            shutil.rmtree(outputs_dir)


def initialize_operating_points(
    operating_points: Sequence[Path],
    *,
    execution_path: str,
    initialization_duration: float,
    log_file: Path,
    dynagnn_version: str,
    backup_iidm: bool = False,
    clean_outputs: bool = True,
) -> dict[str, InitResult]:
    """Initialize each operating point folder. Returns results keyed by folder name."""
    results: dict[str, InitResult] = {}
    for op_path in operating_points:
        print(f"\nInitializing {op_path.name} (duration={initialization_duration:g}s)...")
        try:
            case = discover_case(op_path)
        except (FileNotFoundError, RuntimeError) as exc:
            result = InitResult(
                operating_point=op_path.name,
                success=False,
                run_time=initialization_duration,
                messages=[str(exc)],
            )
            results[op_path.name] = result
            print(f"  ✗ {exc}")
            break

        result = initialize_one_operating_point(
            case,
            execution_path=execution_path,
            run_time=initialization_duration,
            log_file=log_file,
            backup_iidm=backup_iidm,
            clean_outputs=clean_outputs,
        )
        results[op_path.name] = result
        if result.success:
            print(f"  ✓ {op_path.name} initialized")
        else:
            print(f"  ✗ {op_path.name} initialization failed")
            for message in result.messages:
                print(f"    {message}")
            break

    return results


def write_initialization_status_log(log_file: Path, result: Optional[InitResult]) -> None:
    """Record initialization outcome at the top of the per-OP simulations log."""
    if result is None or result.skipped:
        return

    if result.success:
        append_simulation_log(
            log_file,
            (
                "INFO",
                f"Operating point initialization successful "
                f"(initialization_duration={result.run_time:g}s)",
            ),
            section_break=False,
        )
        return

    entries: list[tuple[LogLevel, str]] = [("ERROR", "Operating point initialization failed")]
    for message in result.messages:
        entries.append(("ERROR", message))
    append_simulation_log(log_file, *entries, section_break=False)
