# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: KPI extraction from Dynawo curves

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

try:  # Support both `python modules/kpi.py` and `python -m modules.kpi`.
    from .dyd_mapping import build_dyd_id_to_staticid_map
    from .event_files import normalize_fault_type
except ImportError:  # pragma: no cover
    from dyd_mapping import build_dyd_id_to_staticid_map
    from event_files import normalize_fault_type


try:
    from .paths import (
        CONFIG_PATH,
        CONTINGENCIES_CSV,
        KPI_DIR,
        SIMULATIONS_DIR,
        SNOM_DIR,
        snom_csv_for_operating_point,
    )
    from .simulation_results import (
        list_successful_contingency_dirs,
        load_successful_runs,
        resolve_results_csv,
    )
except ImportError:  # pragma: no cover
    from paths import (
        CONFIG_PATH,
        CONTINGENCIES_CSV,
        KPI_DIR,
        SIMULATIONS_DIR,
        SNOM_DIR,
        snom_csv_for_operating_point,
    )
    from simulation_results import (
        list_successful_contingency_dirs,
        load_successful_runs,
        resolve_results_csv,
    )

DYNAWO_NS = {"dynawo": "http://www.rte-france.com/dynawo"}
ZERO_CURVE_ABS_TOL = 1e-12
T_MAX = 100.0

VOLTAGE_SUFFIXES = ["Upu_value"]
GEN_P_SUFFIXES = ["generator_PGen"]
GEN_Q_SUFFIXES = ["generator_QGen"]

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


def resolve_event_time(config: dict) -> float:
    return float((config.get("simulation") or {}).get("event_time", 10.0))


def resolve_kpi_start_time(config: dict) -> float:
    """Earliest time (s) for KPI sliding windows: ``max(0, event_time - 1)``."""
    return max(0.0, resolve_event_time(config) - 1.0)


def get_kpi_window_settings(config: dict) -> tuple[float, float]:
    kpi_config = config.get("kpi") or {}
    window_sec = float(kpi_config.get("window_sec", 5.0))
    step_sec = float(kpi_config.get("step_sec", 1.0))
    if window_sec <= 0.0:
        raise ValueError("kpi.window_sec must be positive.")
    if step_sec <= 0.0:
        raise ValueError("kpi.step_sec must be positive.")
    return window_sec, step_sec


def parse_float(value: Optional[str], default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_xml_namespace(root: ET.Element) -> Optional[str]:
    if root.tag.startswith("{") and "}" in root.tag:
        return root.tag.split("}")[0][1:]
    return None


def op_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def find_dyd_file(search_dir: Path) -> Optional[Path]:
    dyd_files = sorted(search_dir.rglob("*.dyd"), key=lambda p: (0 if "snapshot" in p.name.lower() else 1, len(p.parts), p.name))
    return dyd_files[0] if dyd_files else None


def find_iidm_file(contingency_dir: Path, op_dir: Path) -> Optional[Path]:
    def candidates(search_dir: Path) -> list[Path]:
        out = []
        for path in search_dir.rglob("*"):
            name = path.name.lower()
            if path.is_file() and (
                name.endswith(".iidm")
                or name.endswith(".xiidm")
                or name == "outputiidm.xml"
            ):
                out.append(path)
        return out

    files = candidates(contingency_dir) or candidates(op_dir)
    files.sort(
        key=lambda p: (
            0 if "snapshot" in p.name.lower() else 1,
            0 if p.name.lower().endswith((".iidm", ".xiidm")) else 1,
            len(p.parts),
            p.name,
        )
    )
    return files[0] if files else None


def build_voltage_curve_to_voltage_level_map(iidm_path: Optional[Path]) -> dict[str, str]:
    """Map voltage curve component IDs to voltage level IDs.

    In node-breaker voltage levels, Dynawo voltage curves are busbar-section IDs.
    In bus-breaker voltage levels, Dynawo voltage curves are bus IDs.
    Mixed IIDM files are handled by inspecting each voltage level independently.
    """
    if not iidm_path or not iidm_path.exists():
        return {}

    try:
        root = ET.parse(iidm_path).getroot()
        ns_uri = get_xml_namespace(root)
        ns = {"iidm": ns_uri} if ns_uri else {}
        vl_path = ".//iidm:voltageLevel" if ns_uri else ".//voltageLevel"
        bb_path = "iidm:nodeBreakerTopology/iidm:busbarSection" if ns_uri else "nodeBreakerTopology/busbarSection"
        bus_path = "iidm:busBreakerTopology/iidm:bus" if ns_uri else "busBreakerTopology/bus"

        mapping: dict[str, str] = {}
        for voltage_level in root.findall(vl_path, ns):
            voltage_level_id = voltage_level.get("id")
            if not voltage_level_id:
                continue
            for busbar in voltage_level.findall(bb_path, ns):
                busbar_id = busbar.get("id")
                if busbar_id:
                    mapping[busbar_id] = voltage_level_id
            for bus in voltage_level.findall(bus_path, ns):
                bus_id = bus.get("id")
                if bus_id:
                    mapping[bus_id] = voltage_level_id
        return mapping
    except Exception as exc:
        print(f"      Warning: failed to parse IIDM voltage mapping from {iidm_path}: {exc}")
        return {}


def aggregate_component_metrics_to_voltage_levels(
    component_metrics: dict[str, float],
    component_to_voltage_level: dict[str, str],
) -> dict[str, float]:
    """Map busbar/bus metrics to voltage levels using max per VL."""
    vl_metrics: dict[str, float] = {}
    for component_id, value in component_metrics.items():
        vl_id = component_to_voltage_level.get(component_id, component_id)
        current = vl_metrics.get(vl_id)
        if current is None or value > current:
            vl_metrics[vl_id] = value
    return vl_metrics


def build_country_component_sets(
    iidm_path: Optional[Path],
    country_filter: Optional[set[str]],
) -> tuple[Optional[set[str]], Optional[set[str]]]:
    """Return allowed voltage labels and generator IDs for the configured country filter."""
    if country_filter is None:
        return None, None
    if not iidm_path or not iidm_path.exists():
        return set(), set()

    try:
        root = ET.parse(iidm_path).getroot()
    except Exception as exc:
        print(f"      Warning: failed to parse IIDM country filter from {iidm_path}: {exc}")
        return set(), set()

    substation_country: dict[str, str] = {}
    voltage_level_substation: dict[str, str] = {}
    voltage_level_buses: dict[str, set[str]] = {}
    generator_voltage_level: dict[str, str] = {}

    for element in root.iter():
        tag = element.tag.split("}", 1)[-1] if element.tag.startswith("{") else element.tag
        if tag == "substation":
            sub_id = (element.get("id") or "").strip()
            country = (element.get("country") or "").strip().upper()
            if sub_id:
                substation_country[sub_id] = country
        elif tag == "voltageLevel":
            vl_id = (element.get("id") or "").strip()
            sub_id = (element.get("substationId") or "").strip()
            if vl_id:
                voltage_level_substation[vl_id] = sub_id
        elif tag == "bus":
            bus_id = (element.get("id") or "").strip()
            vl_id = (element.get("voltageLevelId") or "").strip()
            if bus_id and vl_id:
                voltage_level_buses.setdefault(vl_id, set()).add(bus_id)
        elif tag == "generator":
            gen_id = (element.get("id") or "").strip()
            vl_id = (element.get("voltageLevelId") or "").strip()
            if gen_id and vl_id:
                generator_voltage_level[gen_id] = vl_id

    allowed_vls = {
        vl_id
        for vl_id, sub_id in voltage_level_substation.items()
        if substation_country.get(sub_id, "").upper() in country_filter
    }
    allowed_voltage = allowed_vls
    allowed_generators = {
        gen_id
        for gen_id, vl_id in generator_voltage_level.items()
        if vl_id in allowed_vls
    }
    return allowed_voltage, allowed_generators


try:
    from .generator_snom import load_generator_snom_by_operating_point, load_generator_snom_for_operating_point
except ImportError:  # pragma: no cover
    from generator_snom import load_generator_snom_by_operating_point, load_generator_snom_for_operating_point


def make_label(curve_name: str, suffixes: list[str], id_to_staticid: dict[str, str]) -> str:
    label = curve_name
    for suffix in suffixes:
        suffix_token = f"_{suffix}"
        if label.endswith(suffix_token):
            label = label[: -len(suffix_token)]
            break

    if label.startswith("NETWORK_"):
        label = label[len("NETWORK_") :]
    else:
        label = id_to_staticid.get(label, label)
    return label


def curve_min_max_both_zero(values: object) -> bool:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return True
    return math.isclose(float(arr.min()), 0.0, abs_tol=ZERO_CURVE_ABS_TOL) and math.isclose(
        float(arr.max()), 0.0, abs_tol=ZERO_CURVE_ABS_TOL
    )


def parse_curves_xml(curves_file: Path) -> dict[str, dict[str, list[float]]]:
    root = ET.parse(curves_file).getroot()
    curves = root.findall(".//dynawo:curve", DYNAWO_NS)
    if not curves:
        curves = root.findall(".//curve")

    curves_data: dict[str, dict[str, list[float]]] = {}
    for curve in curves:
        model = curve.get("model")
        variable = curve.get("variable")
        if not model or not variable:
            continue
        curve_name = f"{model}_{variable}"
        times: list[float] = []
        values: list[float] = []
        points = curve.findall("dynawo:point", DYNAWO_NS) or curve.findall("point")
        for point in points:
            t = parse_float(point.get("time"))
            if t > T_MAX + 1e-9:
                break
            times.append(t)
            values.append(parse_float(point.get("value")))
        if times and values:
            curves_data[curve_name] = {"time": times, "value": values}
    return curves_data


def matching_curves(curves_data: dict[str, dict[str, list[float]]], suffixes: list[str]) -> dict[str, dict[str, list[float]]]:
    return {
        curve_name: data
        for curve_name, data in curves_data.items()
        if any(curve_name.endswith(suffix) for suffix in suffixes)
    }


def max_variance_windowed(
    time_values: list[float],
    values: list[float],
    window_sec: float,
    step_sec: float,
    *,
    kpi_start_sec: float = 0.0,
) -> Optional[float]:
    if not time_values or not values:
        return None
    min_len = min(len(time_values), len(values))
    time = np.asarray(time_values[:min_len], dtype=float)
    series = np.asarray(values[:min_len], dtype=float)
    finite = np.isfinite(time) & np.isfinite(series)
    time = time[finite]
    series = series[finite]
    if time.size == 0 or curve_min_max_both_zero(series):
        return None

    t_start = max(float(time.min()), float(kpi_start_sec))
    t_end = float(time.max())
    if t_start + window_sec > t_end:
        return None
    max_var: Optional[float] = None
    t = t_start
    while t + window_sec <= t_end:
        mask = (time >= t) & (time < t + window_sec)
        if mask.any():
            value = float(np.var(series[mask]))
            max_var = value if max_var is None else max(max_var, value)
        t += step_sec
    return max_var


def voltage_kpis_from_curves(
    curves_data: dict[str, dict[str, list[float]]],
    id_to_staticid: dict[str, str],
    voltage_curve_to_voltage_level: dict[str, str],
    allowed_voltage: Optional[set[str]],
    *,
    window_sec: float,
    step_sec: float,
    kpi_start_sec: float,
) -> tuple[dict[str, float], set[str]]:
    """Compute per-busbar/bus KPIs, then aggregate with max per voltage level."""
    component_scores: dict[str, float] = {}

    for curve_name, data in matching_curves(curves_data, VOLTAGE_SUFFIXES).items():
        component_id = make_label(curve_name, VOLTAGE_SUFFIXES, id_to_staticid)
        vl_id = voltage_curve_to_voltage_level.get(component_id, component_id)
        if allowed_voltage is not None and vl_id not in allowed_voltage:
            continue

        score = max_variance_windowed(
            data["time"],
            data["value"],
            window_sec,
            step_sec,
            kpi_start_sec=kpi_start_sec,
        )
        if score is None:
            continue

        current = component_scores.get(component_id)
        if current is None or score > current:
            component_scores[component_id] = score

    scores = aggregate_component_metrics_to_voltage_levels(
        component_scores,
        voltage_curve_to_voltage_level,
    )
    return scores, set(scores)


def resolve_snom(label: str, snom_by_generator: dict[str, float]) -> Optional[float]:
    candidates = [label, label.split("__", 1)[0]]
    for candidate in candidates:
        value = snom_by_generator.get(candidate)
        if value and value > 0.0:
            return float(value)
    return None


def spower_kpis_from_curves(
    curves_data: dict[str, dict[str, list[float]]],
    id_to_staticid: dict[str, str],
    snom_by_generator: dict[str, float],
    allowed_generators: Optional[set[str]],
    *,
    window_sec: float,
    step_sec: float,
    kpi_start_sec: float,
) -> dict[str, float]:
    p_curves: dict[str, dict[str, list[float]]] = {}
    q_curves: dict[str, dict[str, list[float]]] = {}

    for curve_name, data in matching_curves(curves_data, GEN_P_SUFFIXES).items():
        label = make_label(curve_name, GEN_P_SUFFIXES, id_to_staticid)
        p_curves[label] = data
    for curve_name, data in matching_curves(curves_data, GEN_Q_SUFFIXES).items():
        label = make_label(curve_name, GEN_Q_SUFFIXES, id_to_staticid)
        q_curves[label] = data

    scores: dict[str, float] = {}
    for label in sorted(set(p_curves) | set(q_curves)):
        base_label = label.split("__", 1)[0]
        if allowed_generators is not None and base_label not in allowed_generators:
            continue
        snom = resolve_snom(label, snom_by_generator)
        if not snom:
            continue

        p_data = p_curves.get(label)
        q_data = q_curves.get(label)
        if p_data is None and q_data is None:
            continue

        time_values = (p_data or q_data)["time"]
        min_len = len(time_values)
        if p_data is not None:
            min_len = min(min_len, len(p_data["value"]))
        if q_data is not None:
            min_len = min(min_len, len(q_data["value"]))
        if min_len == 0:
            continue

        p_values = p_data["value"][:min_len] if p_data is not None else [0.0] * min_len
        q_values = q_data["value"][:min_len] if q_data is not None else [0.0] * min_len
        s_norm = [math.hypot(float(p), float(q)) / snom for p, q in zip(p_values, q_values)]

        score = max_variance_windowed(
            time_values[:min_len],
            s_norm,
            window_sec,
            step_sec,
            kpi_start_sec=kpi_start_sec,
        )
        if score is not None:
            scores[label] = score
    return scores


def normalize_contingency_id(value: str) -> str:
    text = str(value).strip()
    return text[len("contingency_") :] if text.startswith("contingency_") else text


def build_contingency_fault_map() -> dict[str, tuple[str, str]]:
    csv_path = CONTINGENCIES_CSV
    if not csv_path.exists():
        return {}
    mapping: dict[str, tuple[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
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


def contingency_label(
    contingency_name: str,
    fault_map: dict[str, tuple[str, str]],
    id_to_staticid: dict[str, str],
) -> str:
    contingency_id = normalize_contingency_id(contingency_name)
    fault_name, fault_type = fault_map.get(contingency_id, (contingency_name, ""))
    if fault_type == "generator":
        return id_to_staticid.get(fault_name, fault_name)
    return fault_name


def find_curves_file(contingency_dir: Path) -> Optional[Path]:
    preferred = contingency_dir / "outputs" / "curves" / "curves.xml"
    if preferred.exists():
        return preferred
    matches = sorted(contingency_dir.rglob("curves.xml"))
    return matches[0] if matches else None


def write_kpi_csv(
    output_path: Path,
    op_name: str,
    rows_by_contingency: dict[str, dict[str, float]],
    contingency_labels: dict[str, str],
) -> None:
    components = sorted({component for scores in rows_by_contingency.values() for component in scores})
    rows = []
    for contingency_name in sorted(rows_by_contingency):
        row = {
            "OP": op_name,
            "Contingency": contingency_labels.get(contingency_name, contingency_name),
        }
        scores = rows_by_contingency.get(contingency_name, {})
        for component in components:
            row[component] = scores.get(component, None)
        rows.append(row)

    df = pd.DataFrame(rows, columns=["OP", "Contingency", *components])
    df.to_csv(output_path, index=False)


def write_voltage_kpi_csv(
    output_path: Path,
    rows_by_contingency: dict[str, dict[str, float]],
    components_by_contingency: dict[str, set[str]],
    contingency_labels: dict[str, str],
) -> None:
    components = sorted(
        {
            component
            for components_for_contingency in components_by_contingency.values()
            for component in components_for_contingency
        }
    )
    rows = []
    for contingency_name in sorted(set(rows_by_contingency) | set(components_by_contingency)):
        row = {"Contingency": contingency_labels.get(contingency_name, contingency_name)}
        scores = rows_by_contingency.get(contingency_name, {})
        for component in components:
            row[component] = scores.get(component, None)
        rows.append(row)

    df = pd.DataFrame(rows, columns=["Contingency", *components])
    df.to_csv(output_path, index=False)


def process_operating_point(
    op_dir: Path,
    snom_by_op: dict[str, dict[str, float]],
    fault_map: dict[str, tuple[str, str]],
    *,
    output_dir: Path,
    country_filter: Optional[set[str]],
    window_sec: float,
    step_sec: float,
    kpi_start_sec: float,
    successful_runs: set[tuple[str, str]],
) -> tuple[Path, Path]:
    op_name = op_dir.name
    contingency_dirs = list_successful_contingency_dirs(op_dir, successful_runs)
    if not contingency_dirs:
        raise RuntimeError(f"No successful contingency folders found in {op_dir}")

    voltage_rows: dict[str, dict[str, float]] = {}
    voltage_components_by_contingency: dict[str, set[str]] = {}
    spower_rows: dict[str, dict[str, float]] = {}
    contingency_labels: dict[str, str] = {}
    snom_by_generator = snom_by_op.get(op_name)
    if snom_by_generator is None:
        snom_csv = snom_csv_for_operating_point(op_name)
        if snom_csv.exists():
            snom_by_generator = load_generator_snom_for_operating_point(snom_csv)
        else:
            snom_by_generator = {}
    if not snom_by_generator:
        print(f"  Warning: no SNom values found for {op_name} ({snom_csv_for_operating_point(op_name)})")

    for idx, contingency_dir in enumerate(contingency_dirs, start=1):
        print(f"  {op_name}: {contingency_dir.name} ({idx}/{len(contingency_dirs)})")
        curves_file = find_curves_file(contingency_dir)
        if curves_file is None:
            print("    Warning: curves.xml not found, skipping")
            continue

        dyd_path = find_dyd_file(contingency_dir) or find_dyd_file(op_dir)
        iidm_path = find_iidm_file(contingency_dir, op_dir)
        id_to_staticid = build_dyd_id_to_staticid_map(dyd_path)
        contingency_labels[contingency_dir.name] = contingency_label(
            contingency_dir.name,
            fault_map,
            id_to_staticid,
        )
        voltage_curve_to_voltage_level = build_voltage_curve_to_voltage_level_map(iidm_path)
        allowed_voltage, allowed_generators = build_country_component_sets(
            iidm_path,
            country_filter,
        )
        curves_data = parse_curves_xml(curves_file)

        voltage_scores, voltage_components = voltage_kpis_from_curves(
            curves_data,
            id_to_staticid,
            voltage_curve_to_voltage_level,
            allowed_voltage,
            window_sec=window_sec,
            step_sec=step_sec,
            kpi_start_sec=kpi_start_sec,
        )
        voltage_rows[contingency_dir.name] = voltage_scores
        voltage_components_by_contingency[contingency_dir.name] = voltage_components
        spower_rows[contingency_dir.name] = spower_kpis_from_curves(
            curves_data,
            id_to_staticid,
            snom_by_generator,
            allowed_generators,
            window_sec=window_sec,
            step_sec=step_sec,
            kpi_start_sec=kpi_start_sec,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    voltage_path = output_dir / f"KPI_voltage_{op_name}.csv"
    spower_path = output_dir / f"KPI_spower_{op_name}.csv"
    write_voltage_kpi_csv(
        voltage_path,
        voltage_rows,
        voltage_components_by_contingency,
        contingency_labels,
    )
    write_kpi_csv(spower_path, op_name, spower_rows, contingency_labels)
    return voltage_path, spower_path


process_kpi_operating_point = process_operating_point


def run_kpi(*, successful_runs: Optional[set[tuple[str, str]]] = None) -> list[tuple[Path, Path]]:
    """Build KPI CSVs for successful simulations under Simulations_Scenarios/."""
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
    window_sec, step_sec = get_kpi_window_settings(config)
    kpi_start_sec = resolve_kpi_start_time(config)
    try:
        snom_by_op = load_generator_snom_by_operating_point(SNOM_DIR)
    except FileNotFoundError:
        snom_by_op = {}
    print(f"KPI window start: t >= {kpi_start_sec:g}s (event_time - 1)")
    print(f"Processing {len(successful_runs)} successful simulation(s) from {results_csv.name}")
    fault_map = build_contingency_fault_map()
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
            snom_by_op,
            fault_map,
            output_dir=KPI_DIR,
            country_filter=country_filter,
            window_sec=window_sec,
            step_sec=step_sec,
            kpi_start_sec=kpi_start_sec,
            successful_runs=successful_runs,
        )
        voltage_path, spower_path = paths
        print(f"  Wrote {voltage_path}")
        print(f"  Wrote {spower_path}")
        outputs.append(paths)
    if not outputs:
        raise RuntimeError("No KPI outputs were written for any operating point.")
    return outputs
