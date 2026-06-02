# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: DYD dynamic-to-static identifier mapping

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, Tuple


GeneratorDynamicModel = Tuple[str, str]


def build_dyd_id_to_staticid_map(dyd_path: Optional[Path]) -> Dict[str, str]:
    """Parse a DYD file and return ``{blackBoxModel id: staticId}`` for all models that define both."""
    if not dyd_path or not dyd_path.exists():
        return {}

    id_to_staticid: Dict[str, str] = {}
    try:
        tree = ET.parse(dyd_path)
        root = tree.getroot()

        for bb in root.findall(".//blackBoxModel"):
            bb_id = bb.attrib.get("id")
            static_id = bb.attrib.get("staticId")
            if bb_id and static_id:
                id_to_staticid[bb_id] = static_id

        ns = {"dynawo": "http://www.rte-france.com/dynawo"}
        for bb in root.findall(".//dynawo:blackBoxModel", ns):
            bb_id = bb.attrib.get("id")
            static_id = bb.attrib.get("staticId")
            if bb_id and static_id:
                id_to_staticid[bb_id] = static_id
    except Exception as exc:
        print(f"  ⚠ Failed to parse DYD for staticId mapping: {exc}")
        return {}

    return id_to_staticid


def build_static_id_to_dynamic_id_map(dyd_path: Optional[Path]) -> Dict[str, str]:
    """Map IIDM ``staticId`` -> Dynawo ``blackBoxModel`` ``id`` (first ``id`` wins if duplicates)."""
    out: Dict[str, str] = {}
    for dyn_id, static_id in build_dyd_id_to_staticid_map(dyd_path).items():
        if static_id not in out:
            out[static_id] = dyn_id
    return out


def _iter_blackbox_models(root: ET.Element):
    for bb in root.findall(".//blackBoxModel"):
        yield bb
    ns = {"dynawo": "http://www.rte-france.com/dynawo"}
    for bb in root.findall(".//dynawo:blackBoxModel", ns):
        yield bb


def build_generator_static_to_dynamic_map(
    dyd_path: Optional[Path],
) -> Dict[str, GeneratorDynamicModel]:
    """Map IIDM generator ``staticId`` -> ``(blackBoxModel id, lib)`` for generator models."""
    if not dyd_path or not dyd_path.exists():
        return {}

    static_to_dynamic: Dict[str, GeneratorDynamicModel] = {}
    try:
        root = ET.parse(dyd_path).getroot()
        for bb in _iter_blackbox_models(root):
            lib = bb.attrib.get("lib") or ""
            if "Generator" not in lib:
                continue
            static_id = bb.attrib.get("staticId")
            dyn_id = bb.attrib.get("id")
            if not static_id or not dyn_id or static_id in static_to_dynamic:
                continue
            static_to_dynamic[static_id] = (dyn_id, lib)
    except Exception as exc:
        print(f"  ⚠ Failed to parse DYD for generator mapping: {exc}")
        return {}

    return static_to_dynamic
