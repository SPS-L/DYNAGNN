# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: Disconnection flag detection

from __future__ import annotations

import csv
import math
from pathlib import Path
import re
from typing import Optional
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import yaml

try:
    from .dyd_mapping import build_dyd_id_to_staticid_map
    from .event_files import normalize_fault_type
except ImportError:  # pragma: no cover
    from dyd_mapping import build_dyd_id_to_staticid_map
    from event_files import normalize_fault_type

try:
    from .kpi import (
        VOLTAGE_SUFFIXES,
        build_country_component_sets,
        build_voltage_curve_to_voltage_level_map,
        contingency_label,
        find_curves_file,
        find_dyd_file,
        find_iidm_file,
        make_label,
        matching_curves,
        op_sort_key,
        parse_curves_xml,
    )
except ImportError:  # pragma: no cover
    from kpi import (
        VOLTAGE_SUFFIXES,
        build_country_component_sets,
        build_voltage_curve_to_voltage_level_map,
        contingency_label,
        find_curves_file,
        find_dyd_file,
        find_iidm_file,
        make_label,
        matching_curves,
        op_sort_key,
        parse_curves_xml,
    )

try:
    from .paths import CONFIG_PATH, CONTINGENCIES_CSV, DISCONNECTIONS_DIR, SIMULATIONS_DIR
    from .simulation_results import (
        list_successful_contingency_dirs,
        load_successful_runs,
        resolve_results_csv,
    )
except ImportError:  # pragma: no cover
    from paths import CONFIG_PATH, CONTINGENCIES_CSV, DISCONNECTIONS_DIR, SIMULATIONS_DIR
    from simulation_results import (
        list_successful_contingency_dirs,
        load_successful_runs,
        resolve_results_csv,
    )

GENERATOR_DISCONNECT_MSG = "GENERATOR : disconnecting"
VOLTAGE_ZERO_THRESHOLD = 0.05
VOLTAGE_ENERGIZED_THRESHOLD = 0.2
VOLTAGE_HOLD_SECONDS = 3.0


def local_tag(tag: str) -> str:
    return tag.split("}", 1)[-1] if tag.startswith("{") else tag


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get_country_filter(config: dict) -> Optional[set[str]]:
    raw = (config.get("network") or {}).get("country_filter", [])
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        countries = [part.strip().upper() for part in raw.split(",") if part.strip()]
    else:
        countries = [str(part).strip().upper() for part in raw if str(part).strip()]
    return set(countries) if countries else None


def build_contingency_fault_map() -> dict[str, tuple[str, str]]:
    if not CONTINGENCIES_CSV.exists():
        return {}
    mapping: dict[str, tuple[str, str]] = {}
    with CONTINGENCIES_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                contingency_id = str(row[0]).strip()
                fault_name = str(row[1]).strip()
                fault_type_raw = str(row[2]).strip() if len(row) >= 3 else ""
                fault_type = normalize_fault_type(fault_type_raw) if fault_type_raw else ""
                mapping[contingency_id] = (fault_name, fault_type)
    return mapping


def hold_seconds_to_samples(time_values: list[float], hold_seconds: float) -> int:
    if hold_seconds <= 0:
        return 1
    if len(time_values) < 2:
        return max(1, int(math.ceil(hold_seconds)))
    time = np.asarray(time_values, dtype=float)
    dt = np.diff(time[np.isfinite(time)])
    positive_dt = dt[dt > 0]
    if positive_dt.size == 0:
        return max(1, int(math.ceil(hold_seconds)))
    time_step = float(np.median(positive_dt))
    if time_step <= 0:
        return max(1, int(math.ceil(hold_seconds)))
    return max(1, int(math.ceil(hold_seconds / time_step)))


def first_true_run_start(mask: np.ndarray, run_len: int) -> Optional[int]:
    run_len = max(1, int(run_len))
    if mask.size < run_len:
        return None
    for idx in range(0, mask.size - run_len + 1):
        if bool(mask[idx : idx + run_len].all()):
            return idx
    return None


def is_voltage_disconnection(
    time_values: list[float],
    values: list[float],
    *,
    zero_threshold: float = VOLTAGE_ZERO_THRESHOLD,
    energized_threshold: float = VOLTAGE_ENERGIZED_THRESHOLD,
    hold_seconds: float = VOLTAGE_HOLD_SECONDS,
) -> tuple[bool, bool]:
    """Return (is_disconnected, is_considered) for one busbar/bus voltage curve."""
    min_len = min(len(time_values), len(values))
    if min_len == 0:
        return False, False

    time = np.asarray(time_values[:min_len], dtype=float)
    series = np.asarray(values[:min_len], dtype=float)
    finite = np.isfinite(time) & np.isfinite(series)
    time = time[finite]
    series = series[finite]
    if series.size == 0:
        return False, False

    if math.isclose(float(series.min()), 0.0, abs_tol=zero_threshold) and math.isclose(
        float(series.max()), 0.0, abs_tol=zero_threshold
    ):
        return False, False

    start_v = float(series[0])
    if not np.isfinite(start_v) or start_v <= energized_threshold:
        return False, False

    hold_samples = hold_seconds_to_samples(time.tolist(), hold_seconds)
    zero_mask = series <= zero_threshold
    return first_true_run_start(zero_mask, hold_samples) is not None, True


def expand_component_disconnections_to_voltage_levels(
    disconnected_components: set[str],
    component_to_voltage_level: dict[str, str],
    voltage_level_network: set[str],
) -> set[str]:
    """Promote busbar/bus disconnections to voltage levels (any section -> VL=1)."""
    vl_tripped: set[str] = set()
    for component_id in disconnected_components:
        vl_id = component_to_voltage_level.get(component_id)
        if vl_id:
            vl_tripped.add(vl_id)
        elif component_id in voltage_level_network:
            vl_tripped.add(component_id)
        else:
            vl_tripped.add(component_id)
    return vl_tripped


def voltage_level_ids_from_iidm(iidm_path: Optional[Path], allowed_voltage: Optional[set[str]]) -> set[str]:
    if allowed_voltage is not None:
        return set(allowed_voltage)
    if not iidm_path or not iidm_path.exists():
        return set()
    try:
        root = ET.parse(iidm_path).getroot()
    except Exception:
        return set()
    return {
        (element.get("id") or "").strip()
        for element in root.iter()
        if local_tag(element.tag) == "voltageLevel" and (element.get("id") or "").strip()
    }


def voltage_disconnections_from_curves(
    curves_data: dict[str, dict[str, list[float]]],
    id_to_staticid: dict[str, str],
    component_to_voltage_level: dict[str, str],
    allowed_voltage: Optional[set[str]],
    voltage_level_network: set[str],
) -> tuple[set[str], set[str]]:
    disconnected_components: set[str] = set()
    considered_voltage_levels: set[str] = set()

    for curve_name, data in matching_curves(curves_data, VOLTAGE_SUFFIXES).items():
        component_id = make_label(curve_name, VOLTAGE_SUFFIXES, id_to_staticid)
        vl_id = component_to_voltage_level.get(component_id, component_id)
        if allowed_voltage is not None and vl_id not in allowed_voltage:
            continue

        disconnected, considered = is_voltage_disconnection(data["time"], data["value"])
        if considered:
            considered_voltage_levels.add(vl_id)
        if disconnected:
            disconnected_components.add(component_id)

    vl_tripped = expand_component_disconnections_to_voltage_levels(
        disconnected_components,
        component_to_voltage_level,
        voltage_level_network,
    )
    return vl_tripped, considered_voltage_levels


def find_timeline_file(contingency_dir: Path) -> Optional[Path]:
    candidates = [
        contingency_dir / "outputs" / "timeLine" / "timeline.xml",
        contingency_dir / "outputs" / "timeline" / "timeline.xml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(contingency_dir.rglob("timeline.xml"))
    return matches[0] if matches else None


def parse_generator_disconnect_models(timeline_path: Path) -> set[str]:
    models: set[str] = set()
    if not timeline_path.exists():
        return models
    try:
        root = ET.parse(timeline_path).getroot()
    except Exception as exc:
        print(f"    Warning: failed to parse timeline XML {timeline_path}: {exc}")
        return models

    for event in root.iter():
        if local_tag(event.tag) != "event":
            continue
        message = (event.get("message") or "").strip()
        if message != GENERATOR_DISCONNECT_MSG:
            continue
        model = (event.get("modelName") or "").strip()
        if model:
            models.add(model)
    return models


def generator_ids_from_iidm(iidm_path: Optional[Path], allowed_generators: Optional[set[str]]) -> set[str]:
    if allowed_generators is not None:
        return set(allowed_generators)
    if not iidm_path or not iidm_path.exists():
        return set()
    try:
        root = ET.parse(iidm_path).getroot()
    except Exception:
        return set()
    return {
        (element.get("id") or "").strip()
        for element in root.iter()
        if local_tag(element.tag) == "generator" and (element.get("id") or "").strip()
    }


def spower_disconnections_from_timeline(
    contingency_dir: Path,
    id_to_staticid: dict[str, str],
    allowed_generators: Optional[set[str]],
    generator_network: set[str],
) -> tuple[set[str], set[str]]:
    timeline_path = find_timeline_file(contingency_dir)
    if timeline_path is None:
        return set(), set()

    disconnected: set[str] = set()
    for model in parse_generator_disconnect_models(timeline_path):
        static_id = id_to_staticid.get(model, model)
        if allowed_generators is not None and static_id not in allowed_generators:
            continue
        if static_id in generator_network or allowed_generators is None:
            disconnected.add(static_id)
    return disconnected, generator_network


def write_disconnection_csv(
    output_path: Path,
    op_name: str,
    case_order: list[str],
    flags_by_case: dict[str, set[str]],
    network_by_case: dict[str, set[str]],
    contingency_labels: dict[str, str],
) -> None:
    components = sorted({component for values in flags_by_case.values() for component in values})
    rows = []
    for contingency_name in case_order:
        row = {
            "OP": op_name,
            "Contingency": contingency_labels.get(contingency_name, contingency_name),
        }
        flags = flags_by_case.get(contingency_name, set())
        network = network_by_case.get(contingency_name, set())
        for component in components:
            if component in flags:
                row[component] = 1
            elif component in network:
                row[component] = 0
            else:
                row[component] = None
        rows.append(row)
    pd.DataFrame(rows, columns=["OP", "Contingency", *components]).to_csv(output_path, index=False)


def process_operating_point(
    op_dir: Path,
    fault_map: dict[str, tuple[str, str]],
    *,
    output_dir: Path,
    country_filter: Optional[set[str]],
    successful_runs: set[tuple[str, str]],
) -> tuple[Path, Path]:
    op_name = op_dir.name
    contingency_dirs = list_successful_contingency_dirs(op_dir, successful_runs)
    if not contingency_dirs:
        raise RuntimeError(f"No successful contingency folders found in {op_dir}")

    case_order: list[str] = []
    voltage_flags_by_case: dict[str, set[str]] = {}
    voltage_network_by_case: dict[str, set[str]] = {}
    spower_flags_by_case: dict[str, set[str]] = {}
    spower_network_by_case: dict[str, set[str]] = {}
    contingency_labels: dict[str, str] = {}

    for idx, contingency_dir in enumerate(contingency_dirs, start=1):
        print(f"  {op_name}: {contingency_dir.name} ({idx}/{len(contingency_dirs)})")
        curves_file = find_curves_file(contingency_dir)
        if curves_file is None:
            print("    Warning: curves.xml not found, skipping voltage disconnections")
            continue

        dyd_path = find_dyd_file(contingency_dir) or find_dyd_file(op_dir)
        iidm_path = find_iidm_file(contingency_dir, op_dir)
        id_to_staticid = build_dyd_id_to_staticid_map(dyd_path)
        contingency_labels[contingency_dir.name] = contingency_label(
            contingency_dir.name,
            fault_map,
            id_to_staticid,
        )
        component_to_vl = build_voltage_curve_to_voltage_level_map(iidm_path)
        allowed_voltage, allowed_generators = build_country_component_sets(iidm_path, country_filter)
        voltage_network = voltage_level_ids_from_iidm(iidm_path, allowed_voltage)
        generator_network = generator_ids_from_iidm(iidm_path, allowed_generators)

        curves_data = parse_curves_xml(curves_file)
        vl_tripped, vl_considered = voltage_disconnections_from_curves(
            curves_data,
            id_to_staticid,
            component_to_vl,
            allowed_voltage,
            voltage_network,
        )
        gen_tripped, _ = spower_disconnections_from_timeline(
            contingency_dir,
            id_to_staticid,
            allowed_generators,
            generator_network,
        )

        case_order.append(contingency_dir.name)
        voltage_flags_by_case[contingency_dir.name] = vl_tripped
        voltage_network_by_case[contingency_dir.name] = vl_considered or voltage_network
        spower_flags_by_case[contingency_dir.name] = gen_tripped
        spower_network_by_case[contingency_dir.name] = generator_network

    output_dir.mkdir(parents=True, exist_ok=True)
    voltage_path = output_dir / f"disconnections_voltage_{op_name}.csv"
    spower_path = output_dir / f"disconnections_spower_{op_name}.csv"
    write_disconnection_csv(
        voltage_path,
        op_name,
        case_order,
        voltage_flags_by_case,
        voltage_network_by_case,
        contingency_labels,
    )
    write_disconnection_csv(
        spower_path,
        op_name,
        case_order,
        spower_flags_by_case,
        spower_network_by_case,
        contingency_labels,
    )
    return voltage_path, spower_path


process_disconnections_operating_point = process_operating_point


def run_disconnections_detection(*, successful_runs: Optional[set[tuple[str, str]]] = None) -> list[tuple[Path, Path]]:
    """Build disconnection CSVs for successful simulations under Simulations_Scenarios/."""
    if not SIMULATIONS_DIR.exists():
        raise FileNotFoundError(f"Missing simulations folder: {SIMULATIONS_DIR}")

    results_csv = resolve_results_csv(SIMULATIONS_DIR)
    if successful_runs is None:
        successful_runs = load_successful_runs(results_csv)
    if not successful_runs:
        raise RuntimeError(
            f"No successful simulations found in {results_csv}. "
            "Run simulate.py first and ensure at least one scenario succeeds."
        )

    config = load_config()
    country_filter = get_country_filter(config)
    fault_map = build_contingency_fault_map()
    print(f"Processing {len(successful_runs)} successful simulation(s) from {results_csv.name}")
    op_dirs = sorted(
        [path for path in SIMULATIONS_DIR.iterdir() if path.is_dir() and path.name.startswith("operating_point_")],
        key=op_sort_key,
    )
    if not op_dirs:
        raise RuntimeError(f"No operating point folders found in {SIMULATIONS_DIR}")

    outputs: list[tuple[Path, Path]] = []
    for op_dir in op_dirs:
        contingency_dirs = list_successful_contingency_dirs(op_dir, successful_runs)
        if not contingency_dirs:
            print(f"Skipping {op_dir.name} (no successful simulations)")
            continue

        print(f"Processing {op_dir.name} ({len(contingency_dirs)} successful contingencies)")
        paths = process_operating_point(
            op_dir,
            fault_map,
            output_dir=DISCONNECTIONS_DIR,
            country_filter=country_filter,
            successful_runs=successful_runs,
        )
        for path in paths:
            print(f"  Wrote {path}")
        outputs.append(paths)
    if not outputs:
        raise RuntimeError("No disconnection outputs were written for any operating point.")
    return outputs
