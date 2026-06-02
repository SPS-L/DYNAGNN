# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: Dynawo curve export file generation

from __future__ import annotations

from pathlib import Path
import re
from typing import Optional
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Dynawo jobs XML may use unprefixed tags or namespace prefixes (e.g. dyn:curves).
_XML_TAG_PREFIX = r"(?:[\w-]+:)?"
CURVES_LINE_RE = re.compile(
    rf'^(\s*)<{_XML_TAG_PREFIX}curves\s+[^>]*/>\s*$',
    re.MULTILINE,
)
LOGS_TAG_RE = re.compile(
    rf'^(\s*)(<{_XML_TAG_PREFIX}logs\b)',
    re.MULTILINE,
)
OUTPUTS_CLOSE_RE = re.compile(
    rf'^(\s*)(</{_XML_TAG_PREFIX}outputs>)',
    re.MULTILINE,
)

from .dyd_mapping import build_generator_static_to_dynamic_map
from .paths import INPUTS_DIR
DYNAWO_NS = "http://www.rte-france.com/dynawo"

GENERATOR_POWER_VARIABLES = ("generator_PGen", "generator_QGen")
OMEGA_VARIABLE = "generator_omegaPu"
SYNC_GENERATOR_LIB_MARKER = "GeneratorSynchronous"


def is_generator_dynamic_model(lib: str) -> bool:
    return "Generator" in lib


def is_synchronous_generator_lib(lib: str) -> bool:
    return SYNC_GENERATOR_LIB_MARKER in lib


def local_tag(tag: str) -> str:
    return tag.split("}", 1)[-1] if tag.startswith("{") else tag


def op_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def find_iidm_file(op_dir: Path) -> Optional[Path]:
    candidates = []
    for path in op_dir.iterdir():
        name = path.name.lower()
        if path.is_file() and (
            name.endswith(".iidm")
            or name.endswith(".xiidm")
            or name == "outputiidm.xml"
        ):
            candidates.append(path)
    candidates.sort(
        key=lambda p: (
            0 if "snapshot" in p.name.lower() else 1,
            0 if p.name.lower().endswith((".iidm", ".xiidm")) else 1,
            p.name.lower(),
        )
    )
    return candidates[0] if candidates else None


def find_dyd_file(op_dir: Path) -> Optional[Path]:
    candidates = sorted(
        [path for path in op_dir.iterdir() if path.is_file() and path.name.lower().endswith(".dyd")],
        key=lambda p: (0 if "snapshot" in p.name.lower() else 1, p.name.lower()),
    )
    return candidates[0] if candidates else None


def find_jobs_files(op_dir: Path) -> list[Path]:
    return sorted(path for path in op_dir.iterdir() if path.is_file() and path.name.lower().endswith(".jobs"))


def extract_voltage_curve_ids(iidm_path: Path) -> list[str]:
    """Return voltage curve component IDs according to each voltage level topology."""
    root = ET.parse(iidm_path).getroot()
    curve_ids: list[str] = []

    for voltage_level in root.iter():
        if local_tag(voltage_level.tag) != "voltageLevel":
            continue
        topology_kind = (voltage_level.get("topologyKind") or "").strip().upper()

        if topology_kind == "NODE_BREAKER":
            for child in voltage_level.iter():
                if local_tag(child.tag) == "busbarSection":
                    busbar_id = child.get("id")
                    if busbar_id:
                        curve_ids.append(busbar_id)
        elif topology_kind == "BUS_BREAKER":
            for child in voltage_level.iter():
                if local_tag(child.tag) == "bus":
                    bus_id = child.get("id")
                    if bus_id:
                        curve_ids.append(bus_id)

    return dedup_preserve_order(curve_ids)


def extract_iidm_generator_ids(iidm_path: Path) -> list[str]:
    """Return all generator IDs defined in the IIDM/XIIDM file."""
    root = ET.parse(iidm_path).getroot()
    generator_ids: list[str] = []

    for element in root.iter():
        if local_tag(element.tag) != "generator":
            continue
        generator_id = element.get("id")
        if generator_id:
            generator_ids.append(generator_id)

    return dedup_preserve_order(generator_ids)


def extract_generator_curve_models(
    iidm_path: Path,
    dyd_path: Path,
) -> tuple[list[str], list[str]]:
    """Return dynamic model IDs for IIDM generators that have a DYD mapping."""
    iidm_generator_ids = extract_iidm_generator_ids(iidm_path)
    static_to_dynamic = build_generator_static_to_dynamic_map(dyd_path)

    generator_models: list[str] = []
    sync_generator_models: list[str] = []

    for static_id in iidm_generator_ids:
        dynamic_info = static_to_dynamic.get(static_id)
        if dynamic_info is None:
            continue
        dynamic_model_id, lib = dynamic_info
        if not is_generator_dynamic_model(lib):
            continue

        generator_models.append(dynamic_model_id)
        if is_synchronous_generator_lib(lib):
            sync_generator_models.append(dynamic_model_id)

    return (
        dedup_preserve_order(generator_models),
        dedup_preserve_order(sync_generator_models),
    )


def dedup_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def generate_curve_xml(
    output_path: Path,
    voltage_curve_ids: list[str],
    generator_models: list[str],
    sync_generator_models: list[str],
) -> None:
    root = ET.Element("curvesInput")
    root.set("xmlns", DYNAWO_NS)

    for voltage_id in voltage_curve_ids:
        ET.SubElement(root, "curve", model="NETWORK", variable=f"{voltage_id}_Upu_value")

    for generator_model in generator_models:
        for variable in GENERATOR_POWER_VARIABLES:
            ET.SubElement(root, "curve", model=generator_model, variable=variable)

    for generator_model in sync_generator_models:
        ET.SubElement(root, "curve", model=generator_model, variable=OMEGA_VARIABLE)

    xml_text = ET.tostring(root, encoding="unicode")
    pretty_xml = minidom.parseString(xml_text).toprettyxml(indent="  ", encoding="utf-8")
    output_path.write_bytes(pretty_xml)


def jobs_tag_prefix(content: str) -> str:
    """Return ``dyn:`` when the jobs file uses prefixed Dynawo tags, else ``''``."""
    if re.search(r"<dyn:", content) or re.search(r'xmlns:dyn\s*=', content):
        return "dyn:"
    return ""


def patch_jobs_curve_reference(content: str, curve_file: str) -> str:
    """Add or update the curves export tag, preserving jobs-file formatting."""
    prefix = jobs_tag_prefix(content)
    curves_tag = f'{prefix}curves inputFile="{curve_file}" exportMode="XML"'

    curves_match = CURVES_LINE_RE.search(content)
    if curves_match:
        indent = curves_match.group(1)
        return CURVES_LINE_RE.sub(f"{indent}<{curves_tag}/>", content, count=1)

    logs_match = LOGS_TAG_RE.search(content)
    if logs_match:
        indent = logs_match.group(1)
        insertion = f"{indent}<{curves_tag}/>\n"
        return content[: logs_match.start()] + insertion + content[logs_match.start() :]

    outputs_close = OUTPUTS_CLOSE_RE.search(content)
    if outputs_close:
        indent = outputs_close.group(1)
        insertion = f"{indent}<{curves_tag}/>\n"
        return content[: outputs_close.start()] + insertion + content[outputs_close.start() :]

    raise RuntimeError("No <curves>, <logs>, or </outputs> element found in jobs file")


def update_jobs_curve_reference(jobs_path: Path, curve_file: str = "fic_CRV.xml") -> None:
    """Ensure jobs file references the generated curve export file."""
    original = jobs_path.read_text(encoding="utf-8")
    patched = patch_jobs_curve_reference(original, curve_file=curve_file)
    jobs_path.write_text(patched, encoding="utf-8")


def generate_curves_for_operating_point(op_dir: Path) -> Path:
    iidm_path = find_iidm_file(op_dir)
    dyd_path = find_dyd_file(op_dir)
    if iidm_path is None:
        raise FileNotFoundError(f"Missing IIDM/XIIDM in {op_dir}")
    if dyd_path is None:
        raise FileNotFoundError(f"Missing DYD in {op_dir}")

    voltage_curve_ids = extract_voltage_curve_ids(iidm_path)
    generator_models, sync_generator_models = extract_generator_curve_models(
        iidm_path,
        dyd_path,
    )
    output_path = op_dir / "fic_CRV.xml"
    generate_curve_xml(
        output_path,
        voltage_curve_ids,
        generator_models,
        sync_generator_models,
    )

    for jobs_path in find_jobs_files(op_dir):
        update_jobs_curve_reference(jobs_path, curve_file=output_path.name)

    print(
        f"{op_dir.name}: wrote {output_path.name} "
        f"({len(voltage_curve_ids)} voltage, {len(generator_models)} generator P/Q, "
        f"{len(sync_generator_models)} synchronous omegaPu)"
    )
    return output_path


def generate_curves() -> list[Path]:
    """Generate curve files for all operating-point folders under data/inputs."""
    if not INPUTS_DIR.exists():
        raise FileNotFoundError(f"Missing inputs directory: {INPUTS_DIR}")
    op_dirs = sorted(
        [path for path in INPUTS_DIR.iterdir() if path.is_dir() and path.name.startswith("operating_point_")],
        key=op_sort_key,
    )
    if not op_dirs:
        raise RuntimeError(f"No operating-point folders found in {INPUTS_DIR}")
    return [generate_curves_for_operating_point(op_dir) for op_dir in op_dirs]
