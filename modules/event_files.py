# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: Dynawo contingency event file generation

import re
from pathlib import Path
from typing import Optional, Union

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


try:
    from .curve_generation import jobs_tag_prefix
    from .paths import CONFIG_PATH
except ImportError:  # pragma: no cover
    from curve_generation import jobs_tag_prefix
    from paths import CONFIG_PATH

# Dynawo jobs XML may use unprefixed tags or namespace prefixes (e.g. dyn:dynModels).
_XML_TAG_PREFIX = r"(?:[\w-]+:)?"
EVENTS_DYD_ACTIVE_RE = re.compile(
    rf'^(\s*)<{_XML_TAG_PREFIX}dynModels\s+dydFile="Events\.dyd"\s*/>\s*$',
    re.MULTILINE,
)
EVENTS_DYD_COMMENTED_RE = re.compile(
    rf'^(\s*)<!--\s*(<{_XML_TAG_PREFIX}dynModels\s+dydFile="Events\.dyd"\s*/>)\s*-->\s*$',
    re.MULTILINE,
)
FIRST_DYD_MODELS_RE = re.compile(
    rf'^(\s*)<{_XML_TAG_PREFIX}dynModels\s+dydFile="[^"]+"\s*/>\s*$',
    re.MULTILINE,
)


def default_event_time(config_path: Path = CONFIG_PATH) -> float:
    """Read the default event time from config.yaml."""
    if yaml is None or not config_path.exists():
        return 10.0
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raw = (config.get("simulation") or {}).get("event_time", 10.0)
    if raw is None:
        return 10.0
    return float(raw)


_ALLOWED_FAULT_TYPES = frozenset({
    "line",
    "bus",
    "busbarsection",
    "generator",
    "transformer",
    "load",
})


def normalize_fault_type(fault_type: str) -> str:
    """Validate CSV Type values (case-insensitive).

    ``bus`` is the contingencies.csv label for bus-breaker (``bus`` id) and
    node-breaker (``busbarSection`` id) faults; both use the same Dynawo event.
    """
    t = fault_type.strip().lower()
    if t == "bus":
        t = "busbarsection"
    if t not in _ALLOWED_FAULT_TYPES:
        raise ValueError(
            f"Unsupported contingency type {fault_type!r}. "
            "Use exactly one of: line, bus, busbarsection, generator, "
            "transformer, load."
        )
    return t


def lib_from_fault_type(fault_type: str) -> str:
    """Return the Dynawo event library name for a normalized fault type."""
    t = normalize_fault_type(fault_type)
    if t in ("line", "transformer"):
        return "EventQuadripoleDisconnection"
    if t in ("busbarsection", "load"):
        return "EventConnectedStatus"
    if t == "generator":
        return "EventSetPointBoolean"
    return "UnknownType"


def build_dyd_single(fault_name: str, fault_type: str) -> str:
    """Build an Events.dyd payload for a single contingency."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<dynamicModelsArchitecture xmlns="http://www.rte-france.com/dynawo">',
    ]
    t = normalize_fault_type(fault_type)
    if t == "generator":
        lines.append(
            f'  <blackBoxModel id="GeneratorDisconnection_{fault_name}" '
            f'lib="EventSetPointBoolean" parFile="Events.par" '
            f'parId="GeneratorDisconnection_{fault_name}"/>'
        )
        lines.append(
            f'  <connect id1="GeneratorDisconnection_{fault_name}" '
            f'var1="event_state1_value" id2="{fault_name}" '
            f'var2="generator_switchOffSignal2_value"/>'
        )
    else:
        lib_name = lib_from_fault_type(fault_type)
        lines.append(
            f'  <blackBoxModel id="Disconnect_{fault_name}" lib="{lib_name}" '
            f'parFile="Events.par" parId="Disconnect_{fault_name}"/>'
        )
        lines.append(
            f'  <connect id1="Disconnect_{fault_name}" var1="event_state1_value" '
            f'id2="NETWORK" var2="{fault_name}_state_value"/>'
        )
    lines.append("</dynamicModelsArchitecture>")
    return "\n".join(lines)


def build_par_single(
    contingency_id: Union[int, str],
    fault_name: str,
    fault_type: str,
    event_time: Optional[Union[int, float]] = None,
) -> str:
    """Build an Events.par payload for a single contingency."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<parametersSet xmlns="http://www.rte-france.com/dynawo">',
    ]
    t = normalize_fault_type(fault_type)
    if event_time is None:
        event_time = default_event_time()
    event_time_str = f"{float(event_time):g}"
    if t == "generator":
        lines.append(f"  <!-- {contingency_id} Generator disconnection {fault_name}-->")
        lines.append(f'  <set id="GeneratorDisconnection_{fault_name}">')
        lines.append(f'        <par type="DOUBLE" name="event_tEvent" value="{event_time_str}"/>')
        lines.append('        <par type="BOOL" name="event_stateEvent1" value="true"/>')
        lines.append("  </set>")
    else:
        lines.append(f"  <!-- {contingency_id} Disconnection of {t} {fault_name}-->")
        lines.append(f'  <set id="Disconnect_{fault_name}">')
        lines.append(f'        <par type="DOUBLE" name="event_tEvent" value="{event_time_str}"/>')
        if t in ("line", "transformer"):
            lines.append('        <par type="BOOL" name="event_disconnectOrigin" value="true"/>')
            lines.append('        <par type="BOOL" name="event_disconnectExtremity" value="true"/>')
        elif t in ("busbarsection", "load"):
            lines.append('        <par type="BOOL" name="event_open" value="true"/>')
        lines.append("  </set>")
    lines.append("</parametersSet>")
    return "\n".join(lines)


def find_jobs_files(case_dir: Path) -> list[Path]:
    return sorted(path for path in case_dir.iterdir() if path.is_file() and path.name.lower().endswith(".jobs"))


def patch_jobs_events_reference(content: str) -> str:
    """Ensure jobs file references ``Events.dyd`` inside ``<modeler>``, preserving formatting."""
    prefix = jobs_tag_prefix(content)
    events_tag = f'{prefix}dynModels dydFile="Events.dyd"'

    active_match = EVENTS_DYD_ACTIVE_RE.search(content)
    if active_match:
        indent = active_match.group(1)
        return EVENTS_DYD_ACTIVE_RE.sub(f"{indent}<{events_tag}/>", content, count=1)

    commented_match = EVENTS_DYD_COMMENTED_RE.search(content)
    if commented_match:
        indent = commented_match.group(1)
        return EVENTS_DYD_COMMENTED_RE.sub(f"{indent}<{events_tag}/>", content, count=1)

    first_dyd_match = FIRST_DYD_MODELS_RE.search(content)
    if first_dyd_match is None:
        raise RuntimeError('No <dynModels dydFile="..."/> element found in jobs file')

    line_end = content.find("\n", first_dyd_match.start())
    insert_at = len(content) if line_end == -1 else line_end + 1
    indent = first_dyd_match.group(1)
    insertion = f"{indent}<{events_tag}/>\n"
    return content[:insert_at] + insertion + content[insert_at:]


def update_jobs_events_reference(jobs_path: Path) -> None:
    """Add or update the Events.dyd reference in a jobs file."""
    original = jobs_path.read_text(encoding="utf-8")
    patched = patch_jobs_events_reference(original)
    if patched != original:
        jobs_path.write_text(patched, encoding="utf-8")


def write_event_files(
    scenario_dir: Path,
    contingency_id: Union[int, str],
    fault_name: str,
    fault_type: str,
    event_time: Optional[Union[int, float]] = None,
) -> None:
    """Write Events.dyd and Events.par for a single contingency and patch jobs files."""
    scenario_dir = Path(scenario_dir)
    (scenario_dir / "Events.dyd").write_text(
        build_dyd_single(fault_name, fault_type),
        encoding="utf-8",
    )
    (scenario_dir / "Events.par").write_text(
        build_par_single(contingency_id, fault_name, fault_type, event_time=event_time),
        encoding="utf-8",
    )
    for jobs_path in find_jobs_files(scenario_dir):
        update_jobs_events_reference(jobs_path)
