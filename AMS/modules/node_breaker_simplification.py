"""IIDM switch retention from DSL action locations and GAT substation predictions."""

from __future__ import annotations

import lzma
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, Set, Tuple, Union


def _local_tag(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _iter_by_local_tag(root: ET.Element, local_name: str):
    for element in root.iter():
        if _local_tag(element.tag) == local_name:
            yield element


def _find_children_by_local_tag(parent: ET.Element, local_name: str):
    for child in parent:
        if _local_tag(child.tag) == local_name:
            yield child


def _load_iidm_tree(path: Path) -> ET.ElementTree:
    raw = path.read_bytes()
    if path.suffix.lower() == ".xz" or raw.startswith(b"\xfd7zXZ\x00"):
        raw = lzma.decompress(raw)
    return ET.ElementTree(ET.fromstring(raw))


def _write_iidm_tree(tree: ET.ElementTree, path: Path) -> None:
    root = tree.getroot()
    if root.tag.startswith("{"):
        ns_uri = root.tag.split("}", 1)[0][1:]
        ET.register_namespace("iidm", ns_uri)

    if path.suffix.lower() == ".xz":
        buffer = ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True)
        path.write_bytes(lzma.compress(buffer))
    else:
        tree.write(path, encoding="utf-8", xml_declaration=True)


def substation_ids_for_components(
    root: ET.Element,
    component_ids: Set[str],
) -> Tuple[Set[str], Set[str]]:
    """Map component ids (lines, loads, voltage levels, …) to substation ids."""
    if not component_ids:
        return set(), set()

    wanted = {value.strip() for value in component_ids if str(value).strip()}
    matched: Set[str] = set()
    retained_substations: Set[str] = set()

    voltage_level_to_substation: dict[str, str] = {}
    for substation in _iter_by_local_tag(root, "substation"):
        substation_id = substation.attrib.get("id", "").strip()
        for voltage_level in _find_children_by_local_tag(substation, "voltageLevel"):
            voltage_level_id = voltage_level.attrib.get("id", "").strip()
            if voltage_level_id and substation_id:
                voltage_level_to_substation[voltage_level_id] = substation_id

        for element in substation.iter():
            element_ids = {
                element.attrib.get("id", "").strip(),
                element.attrib.get("name", "").strip(),
            }
            for element_id in element_ids.intersection(wanted):
                matched.add(element_id)
                if substation_id:
                    retained_substations.add(substation_id)

    for element in root.iter():
        element_ids = {
            element.attrib.get("id", "").strip(),
            element.attrib.get("name", "").strip(),
        }
        matches = element_ids.intersection(wanted)
        if not matches:
            continue

        matched.update(matches)
        for attr_name in ("voltageLevelId", "voltageLevelId1", "voltageLevelId2"):
            voltage_level_id = element.attrib.get(attr_name, "").strip()
            substation_id = voltage_level_to_substation.get(voltage_level_id)
            if substation_id:
                retained_substations.add(substation_id)

    return retained_substations, wanted.difference(matched)


def _retained_substations_from_predictions(
    substation_predictions: Mapping[str, int],
    epsilon: float,
) -> Set[str]:
    retained: Set[str] = set()
    threshold = float(epsilon)
    for substation_id, value in substation_predictions.items():
        sid = str(substation_id).strip()
        if not sid:
            continue
        try:
            cls = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid prediction for substation {sid!r}: {value!r}") from exc
        if cls >= threshold:
            retained.add(sid)
    return retained


def _patch_iidm_switches(
    root: ET.Element,
    retained_substations: Set[str],
) -> Tuple[int, int, int]:
    total_switches = 0
    for switch in _iter_by_local_tag(root, "switch"):
        switch.set("retained", "false")
        total_switches += 1

    retained_switches = 0
    matched_substations = 0
    for substation in _iter_by_local_tag(root, "substation"):
        substation_id = substation.attrib.get("id", "").strip()
        if substation_id not in retained_substations:
            continue

        matched_substations += 1
        for switch in _iter_by_local_tag(substation, "switch"):
            switch.set("retained", "true")
            retained_switches += 1

    return total_switches, matched_substations, retained_switches


@dataclass(frozen=True)
class NodeBreakerSimplificationResult:
    total_switches: int
    matched_substations: int
    retained_switches: int
    retained_substation_count: int
    action_location_substations: Tuple[str, ...]
    prediction_substations: Tuple[str, ...]
    unmatched_action_locations: Tuple[str, ...]


def apply_node_breaker_simplification(
    iidm_path: Union[str, Path],
    epsilon: Union[int, float],
    action_locations: Sequence[str],
    substation_predictions: Mapping[str, int],
) -> NodeBreakerSimplificationResult:
    """Patch switch retention in an IIDM file in place.

  1. Set every switch to ``retained="false"``.
  2. Retain all switches in substations that contain any ``action_locations`` component.
  3. Retain all switches in substations whose prediction is ``>= epsilon``.
    """
    path = Path(iidm_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"IIDM file not found: {path}")

    tree = _load_iidm_tree(path)
    root = tree.getroot()

    action_component_ids = {str(value).strip() for value in action_locations if str(value).strip()}
    from_actions, unmatched = substation_ids_for_components(root, action_component_ids)
    from_predictions = _retained_substations_from_predictions(substation_predictions, epsilon)

    retained_substations = set(from_actions)
    retained_substations.update(from_predictions)

    total, matched_subs, retained_sw = _patch_iidm_switches(root, retained_substations)
    _write_iidm_tree(tree, path)

    return NodeBreakerSimplificationResult(
        total_switches=total,
        matched_substations=matched_subs,
        retained_switches=retained_sw,
        retained_substation_count=len(retained_substations),
        action_location_substations=tuple(sorted(from_actions)),
        prediction_substations=tuple(sorted(from_predictions)),
        unmatched_action_locations=tuple(sorted(unmatched)),
    )
