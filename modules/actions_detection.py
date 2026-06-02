# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: Timeline action flag detection

from __future__ import annotations

import csv
from collections import defaultdict, deque
from pathlib import Path
import re
from typing import Optional
import xml.etree.ElementTree as ET

import pandas as pd
import yaml

try:
    from .dynawo_runner import find_scenario_log_file
    from .dyd_mapping import build_dyd_id_to_staticid_map
    from .event_files import normalize_fault_type
except ImportError:  # pragma: no cover
    from dynawo_runner import find_scenario_log_file
    from dyd_mapping import build_dyd_id_to_staticid_map
    from event_files import normalize_fault_type


try:
    from .paths import (
        ACTIONS_DIR,
        CONFIG_PATH,
        CONTINGENCIES_CSV,
        SIMULATIONS_DIR,
    )
    from .simulation_results import (
        list_successful_contingency_dirs,
        load_successful_runs,
        resolve_results_csv,
    )
except ImportError:  # pragma: no cover
    from paths import (
        ACTIONS_DIR,
        CONFIG_PATH,
        CONTINGENCIES_CSV,
        SIMULATIONS_DIR,
    )
    from simulation_results import (
        list_successful_contingency_dirs,
        load_successful_runs,
        resolve_results_csv,
    )

def local_tag(tag: str) -> str:
    return tag.split("}", 1)[-1] if tag.startswith("{") else tag


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_event_time(config: dict) -> float:
    """Return fault/event time (s); timeline actions use ``t >= event_time``."""
    return float((config.get("simulation") or {}).get("event_time", 10.0))


def get_country_filter(config: dict) -> Optional[set[str]]:
    raw = (config.get("network") or {}).get("country_filter", [])
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        countries = [part.strip().upper() for part in raw.split(",") if part.strip()]
    else:
        countries = [str(part).strip().upper() for part in raw if str(part).strip()]
    return set(countries) if countries else None


def op_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def normalize_contingency_id(value: str) -> str:
    text = str(value).strip()
    return text[len("contingency_") :] if text.startswith("contingency_") else text


def build_fault_map() -> dict[str, tuple[str, str]]:
    if not CONTINGENCIES_CSV.exists():
        return {}
    out: dict[str, tuple[str, str]] = {}
    with CONTINGENCIES_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                key = str(row[0]).strip()
                value = str(row[1]).strip()
                fault_type_raw = str(row[2]).strip() if len(row) >= 3 else ""
                fault_type = normalize_fault_type(fault_type_raw) if fault_type_raw else ""
                if key:
                    out[key] = (value, fault_type)
    return out


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


def find_iidm_file(contingency_dir: Path, op_dir: Path) -> Optional[Path]:
    def candidates(search_dir: Path) -> list[Path]:
        out = []
        for path in search_dir.rglob("*"):
            name = path.name.lower()
            if path.is_file() and (name.endswith((".iidm", ".xiidm")) or name == "outputiidm.xml"):
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


def find_dyd_files(contingency_dir: Path, op_dir: Path) -> list[Path]:
    files = sorted(contingency_dir.rglob("*.dyd"), key=lambda p: (0 if "snapshot" in p.name.lower() else 1, len(p.parts), p.name))
    if files:
        return files
    return sorted(op_dir.rglob("*.dyd"), key=lambda p: (0 if "snapshot" in p.name.lower() else 1, len(p.parts), p.name))


def parse_timeline_xml(timeline_path: Path, *, event_time: float) -> set[str]:
    """Return timeline model names with events at or after event_time."""
    action_models: set[str] = set()
    try:
        root = ET.parse(timeline_path).getroot()
    except Exception as exc:
        print(f"    Warning: failed to parse timeline XML {timeline_path}: {exc}")
        return action_models

    for event in root.iter():
        if local_tag(event.tag) != "event":
            continue
        model = (event.get("modelName") or "").strip()
        if not model:
            continue
        try:
            event_t = float((event.get("time") or "").strip())
        except ValueError:
            event_t = event_time
        if event_t >= event_time:
            action_models.add(model)
    return action_models


def parse_timeline_log(log_path: Path, *, event_time: float) -> set[str]:
    """Best-effort fallback for text logs containing timeline-like event lines."""
    action_models: set[str] = set()
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return action_models

    for line in text.splitlines():
        if "modelName" in line:
            model_match = re.search(r'modelName=["\']?([^"\'>\s]+)', line)
        else:
            model_match = re.search(r"\bmodel(?:Name)?\s*[:=]\s*([^\s,;]+)", line)
        model = model_match.group(1).strip() if model_match else ""
        if not model:
            continue
        if "time" in line:
            time_match = re.search(r'time=["\']?([0-9.+-eE]+)', line) or re.search(
                r"\btime\s*[:=]\s*([0-9.+-eE]+)", line
            )
            try:
                event_t = float(time_match.group(1)) if time_match else event_time
            except ValueError:
                event_t = event_time
        else:
            event_t = event_time
        if event_t >= event_time:
            action_models.add(model)
    return action_models


def parse_timeline(contingency_dir: Path, *, event_time: float) -> set[str]:
    timeline_path = find_timeline_file(contingency_dir)
    if timeline_path is not None:
        return parse_timeline_xml(timeline_path, event_time=event_time)
    log_path = find_scenario_log_file(contingency_dir)
    if log_path is not None:
        return parse_timeline_log(log_path, event_time=event_time)
    return set()


def parse_dyd_relations(dyd_paths: list[Path]) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
    """Return (adjacency, direct_static, network_static) for DYD dynamic models."""
    adjacency: dict[str, set[str]] = defaultdict(set)
    direct_static: dict[str, set[str]] = defaultdict(set)
    network_static: dict[str, set[str]] = defaultdict(set)
    dynamic_ids: set[str] = set()
    roots = []

    for dyd_path in dyd_paths:
        try:
            root = ET.parse(dyd_path).getroot()
        except Exception as exc:
            print(f"    Warning: failed to parse DYD {dyd_path}: {exc}")
            continue
        roots.append(root)
        for element in root.iter():
            if local_tag(element.tag) in {"blackBoxModel", "modelicaModel", "modelTemplateExpansion"}:
                dynamic_id = (element.get("id") or "").strip()
                if not dynamic_id:
                    continue
                dynamic_ids.add(dynamic_id)
                static_id = (element.get("staticId") or "").strip()
                if static_id:
                    direct_static[dynamic_id].add(static_id)

    for root in roots:
        for element in root.iter():
            if local_tag(element.tag) not in {"connect", "macroConnect"}:
                continue
            id1 = (element.get("id1") or "").strip()
            id2 = (element.get("id2") or "").strip()
            name1 = (element.get("name1") or "").strip()
            name2 = (element.get("name2") or "").strip()
            if id1 == "NETWORK" and id2 != "NETWORK":
                if name1:
                    network_static[id2].add(name1)
                continue
            if id2 == "NETWORK" and id1 != "NETWORK":
                if name2:
                    network_static[id1].add(name2)
                continue
            if id1 in dynamic_ids and id2 in dynamic_ids:
                adjacency[id1].add(id2)
                adjacency[id2].add(id1)
    return adjacency, direct_static, network_static


def resolve_impacted_components(
    model_name: str,
    *,
    adjacency: dict[str, set[str]],
    direct_static: dict[str, set[str]],
    network_static: dict[str, set[str]],
    iidm_component_ids: set[str],
) -> set[str]:
    """
    Resolve impacted IIDM component IDs for one timeline model:

    1. If modelName is already an IIDM component ID, use it directly.
    2. If modelName is calculatedBus_{voltageLevel}_0, use that voltage level.
    3. Else if modelName is a DYD model with staticId(s), use those staticId(s).
    4. Else treat it as a controller and use direct DYD neighbors that have
       staticId(s), plus direct NETWORK links where name1/name2 is an IIDM ID.

    No controller-to-controller cascade.
    """
    if not model_name:
        return set()
    if model_name in iidm_component_ids:
        return {model_name}
    if model_name.startswith("calculatedBus_") and model_name.endswith("_0"):
        voltage_level_id = model_name[len("calculatedBus_") : -len("_0")]
        if voltage_level_id in iidm_component_ids:
            return {voltage_level_id}
    direct = direct_static.get(model_name, set())
    if direct:
        impacted_direct = {static_id for static_id in direct if static_id in iidm_component_ids}
        if impacted_direct:
            return impacted_direct

    impacted: set[str] = set()
    for neighbor in adjacency.get(model_name, set()):
        static_ids = direct_static.get(neighbor, set())
        impacted |= {static_id for static_id in static_ids if static_id in iidm_component_ids}
    impacted |= {static_id for static_id in network_static.get(model_name, set()) if static_id in iidm_component_ids}
    return impacted


def build_iidm_maps(
    iidm_path: Path,
    country_filter: Optional[set[str]],
) -> tuple[dict[str, set[str]], dict[str, set[str]], set[str], set[str], set[str]]:
    """Return component->voltage-levels, voltage-level->generators, voltage levels, generators, component IDs."""
    component_to_voltage: dict[str, set[str]] = defaultdict(set)
    voltage_to_generators: dict[str, set[str]] = defaultdict(set)
    voltage_names: set[str] = set()
    generators: set[str] = set()
    component_ids: set[str] = set()
    bus_to_voltage_level: dict[str, str] = {}
    converter_station_to_voltage: dict[str, set[str]] = defaultdict(set)

    try:
        root = ET.parse(iidm_path).getroot()
    except Exception as exc:
        print(f"    Warning: failed to parse IIDM {iidm_path}: {exc}")
        return component_to_voltage, voltage_to_generators, voltage_names, generators, component_ids

    substation_country: dict[str, str] = {}
    voltage_level_substation: dict[str, str] = {}
    voltage_level_country: dict[str, str] = {}
    for element in root.iter():
        tag = local_tag(element.tag)
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
    for vl_id, sub_id in voltage_level_substation.items():
        voltage_level_country[vl_id] = substation_country.get(sub_id, "")

    def voltage_allowed(voltage_level_id: str) -> bool:
        if country_filter is None:
            return True
        return voltage_level_country.get(voltage_level_id, "").upper() in country_filter

    def walk(node: ET.Element, current_vl: Optional[str] = None) -> None:
        tag = local_tag(node.tag)
        next_vl = current_vl
        if tag == "voltageLevel":
            voltage_level_id = (node.get("id") or "").strip()
            if voltage_level_id:
                next_vl = voltage_level_id
                if voltage_allowed(voltage_level_id):
                    voltage_names.add(voltage_level_id)
        if tag == "bus":
            bus_id = (node.get("id") or "").strip()
            if bus_id:
                if next_vl:
                    bus_to_voltage_level[bus_id] = next_vl
        for child in list(node):
            walk(child, next_vl)

    walk(root)

    def component_voltage_names(element: ET.Element, current_vl: Optional[str]) -> set[str]:
        out: set[str] = set()
        for attr in ("voltageLevelId", "voltageLevelId1", "voltageLevelId2", "voltageLevelId3"):
            value = (element.get(attr) or "").strip()
            if value:
                out.add(value)
        if current_vl:
            out.add(current_vl)
        for attr in ("bus", "connectableBus", "bus1", "connectableBus1", "bus2", "connectableBus2", "bus3", "connectableBus3"):
            bus = (element.get(attr) or "").strip()
            if bus and bus in bus_to_voltage_level:
                out.add(bus_to_voltage_level[bus])
        return out

    def collect_components(node: ET.Element, current_vl: Optional[str] = None) -> None:
        tag = local_tag(node.tag)
        next_vl = current_vl
        if tag == "voltageLevel":
            voltage_level_id = (node.get("id") or "").strip()
            if voltage_level_id:
                next_vl = voltage_level_id
        component_id = (node.get("id") or "").strip()
        if component_id:
            component_ids.add(component_id)
            names = component_voltage_names(node, next_vl)
            names = {name for name in names if voltage_allowed(name)}
            if names:
                component_to_voltage[component_id] |= names
                if tag in {"vscConverterStation", "lccConverterStation"}:
                    converter_station_to_voltage[component_id] |= names
            if tag == "generator":
                generators.add(component_id)
                for name in names:
                    voltage_to_generators[name].add(component_id)
        for child in list(node):
            collect_components(child, next_vl)

    collect_components(root)

    for element in root.iter():
        if local_tag(element.tag) != "hvdcLine":
            continue
        hvdc_id = (element.get("id") or "").strip()
        if not hvdc_id:
            continue
        component_ids.add(hvdc_id)
        for attr in ("converterStation1", "converterStation2"):
            converter_station_id = (element.get(attr) or "").strip()
            if converter_station_id:
                component_to_voltage[hvdc_id] |= {
                    name
                    for name in converter_station_to_voltage.get(converter_station_id, set())
                    if voltage_allowed(name)
                }
    return component_to_voltage, voltage_to_generators, voltage_names, generators, component_ids


def process_case(
    contingency_dir: Path,
    op_dir: Path,
    country_filter: Optional[set[str]],
    *,
    event_time: float,
) -> Optional[tuple[set[str], set[str], set[str], set[str]]]:
    action_models = parse_timeline(contingency_dir, event_time=event_time)
    iidm_path = find_iidm_file(contingency_dir, op_dir)
    dyd_paths = find_dyd_files(contingency_dir, op_dir)
    if iidm_path is None or not dyd_paths:
        print(f"    Warning: missing IIDM/DYD for {contingency_dir.name}")
        return None

    adjacency, direct_static, network_static = parse_dyd_relations(dyd_paths)
    component_to_voltage, voltage_to_generators, voltage_network, gen_network, component_ids = build_iidm_maps(
        iidm_path,
        country_filter,
    )
    dynamic_gens = {
        static_id
        for static_ids in direct_static.values()
        for static_id in static_ids
        if static_id in gen_network
    }

    impacted_components: set[str] = set()
    for model in action_models:
        impacted_components |= resolve_impacted_components(
            model,
            adjacency=adjacency,
            direct_static=direct_static,
            network_static=network_static,
            iidm_component_ids=component_ids,
        )

    action_voltage: set[str] = set()
    for component in impacted_components:
        action_voltage |= component_to_voltage.get(component, set())

    action_generators: set[str] = set()
    for voltage_name in action_voltage:
        action_generators |= {gen for gen in voltage_to_generators.get(voltage_name, set()) if gen in dynamic_gens}
    action_generators |= impacted_components & dynamic_gens

    return action_voltage, action_generators, voltage_network, dynamic_gens


def write_flag_csv(
    output_path: Path,
    case_order: list[str],
    flags_by_case: dict[str, set[str]],
    network_by_case: dict[str, set[str]],
    contingency_labels: dict[str, str],
    op_name: str,
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
    country_filter: Optional[set[str]],
    fault_map: dict[str, tuple[str, str]],
    output_dir: Path,
    *,
    event_time: float,
    successful_runs: set[tuple[str, str]],
) -> tuple[Path, Path]:
    contingency_dirs = list_successful_contingency_dirs(op_dir, successful_runs)
    if not contingency_dirs:
        raise RuntimeError(f"No successful contingency folders found in {op_dir}")

    case_order: list[str] = []
    action_voltage_by_case: dict[str, set[str]] = {}
    action_gen_by_case: dict[str, set[str]] = {}
    voltage_network_by_case: dict[str, set[str]] = {}
    gen_network_by_case: dict[str, set[str]] = {}
    contingency_labels: dict[str, str] = {}

    for idx, contingency_dir in enumerate(contingency_dirs, start=1):
        print(f"  {op_dir.name}: {contingency_dir.name} ({idx}/{len(contingency_dirs)})")
        result = process_case(contingency_dir, op_dir, country_filter, event_time=event_time)
        if result is None:
            continue
        dyd_paths = find_dyd_files(contingency_dir, op_dir)
        id_to_staticid = {}
        for dyd_path in dyd_paths:
            id_to_staticid.update(build_dyd_id_to_staticid_map(dyd_path))
        contingency_labels[contingency_dir.name] = contingency_label(
            contingency_dir.name,
            fault_map,
            id_to_staticid,
        )
        action_voltage, action_gen, voltage_network, gen_network = result
        case_order.append(contingency_dir.name)
        action_voltage_by_case[contingency_dir.name] = action_voltage
        action_gen_by_case[contingency_dir.name] = action_gen
        voltage_network_by_case[contingency_dir.name] = voltage_network
        gen_network_by_case[contingency_dir.name] = gen_network

    output_dir.mkdir(parents=True, exist_ok=True)
    action_voltage_path = output_dir / f"actions_voltage_{op_dir.name}.csv"
    action_gen_path = output_dir / f"actions_spower_{op_dir.name}.csv"

    write_flag_csv(action_voltage_path, case_order, action_voltage_by_case, voltage_network_by_case, contingency_labels, op_dir.name)
    write_flag_csv(action_gen_path, case_order, action_gen_by_case, gen_network_by_case, contingency_labels, op_dir.name)
    return action_voltage_path, action_gen_path


process_actions_operating_point = process_operating_point


def run_actions_detection(*, successful_runs: Optional[set[tuple[str, str]]] = None) -> list[tuple[Path, Path]]:
    """Build action CSVs for successful simulations under Simulations_Scenarios/."""
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
    event_time = resolve_event_time(config)
    country_filter = get_country_filter(config)
    fault_map = build_fault_map()
    print(f"Timeline action filter: t >= event_time ({event_time:g}s)")
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
            country_filter,
            fault_map,
            ACTIONS_DIR,
            event_time=event_time,
            successful_runs=successful_runs,
        )
        for path in paths:
            print(f"  Wrote {path}")
        outputs.append(paths)
    if not outputs:
        raise RuntimeError("No action outputs were written for any operating point.")
    return outputs
