"""
Parse TwinEU-style DSL scenario files and map actions to IIDM component locations.

Outputs:
  - action_locations: all resolved location names for supported actions, plus
    IIDM ids referenced in ``when`` condition monitors (``<VARIABLE> at
    <COMPONENT_TYPE> "<ID>"`` per DSL_Description.pdf §3.5)
  - events_list: subset for open switch / open line only (model prediction targets)

Supported dynamic clusters (static, known at scenario authoring time):
  - regex "..." on load|generator|...
  - group by "attr" "value" on load|generator|...
  - where "prop" <op> <value> [unit] on load|generator|... (IIDM attributes)

Skipped (runtime — not resolved here):
  - ``when`` / ``after`` trigger logic (thresholds, timing); actions inside
    ``when`` blocks are still extracted; condition monitor ids go to
    ``action_locations`` only
"""

from __future__ import annotations

import lzma
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

# ``when <VARIABLE> at <COMPONENT_TYPE> "<COMPONENT_ID>" ...`` (DSL_Description.pdf §3.5)
_CONDITION_COMPONENT_TYPES = (
    "bus|line|asset|voltageLevel|generator|load|switch|substation|busbarSection|"
    "twoWindingsTransformer|threeWindingsTransformer|battery|shunt|"
    "staticVarCompensator|hvdcLine|2_windings_transformer|3_windings_transformer"
)
_CONDITION_VARIABLES = "voltage|frequency|P|Q|U|state"
_CONDITION_AT_COMPONENT = re.compile(
    rf"\b(?:{_CONDITION_VARIABLES})\s+at\s+(?:{_CONDITION_COMPONENT_TYPES})\s+\"([^\"]*)\"",
    re.IGNORECASE,
)
# Legacy/alternate TwinEU form: ``power flow on line "…"``
_CONDITION_ON_LINE = re.compile(
    r"(?:power\s+flow|active_flow|reactive_flow)\s+on\s+line\s+\"([^\"]*)\"",
    re.IGNORECASE,
)
_WHEN_STATEMENT = re.compile(r"\bwhen\b", re.IGNORECASE)

ActionKind = Literal[
    "open_switch",
    "close_switch",
    "open_line",
    "close_line",
    "increase_load",
    "decrease_load",
    "set_load",
    "increase_generator",
    "decrease_generator",
    "set_generator",
    "apply_fault",
    "clear_fault",
]

EVENT_KINDS = frozenset({"open_switch", "open_line"})

# DSL componentType -> IIDM element bucket used for dynamic clusters / faults
COMPONENT_TYPE_BUCKETS: dict[str, str] = {
    "line": "lines",
    "generator": "generators",
    "load": "loads",
    "switch": "switches",
    "twoWindingsTransformer": "two_windings_transformers",
    "threeWindingsTransformer": "three_windings_transformers",
    "2_windings_transformer": "two_windings_transformers",
    "3_windings_transformer": "three_windings_transformers",
    "substation": "substations",
    "voltageLevel": "voltage_levels",
    "busbarSection": "busbar_sections",
    "battery": "batteries",
    "shunt": "shunt_compensators",
    "staticVarCompensator": "static_var_compensators",
    "hvdcLine": "hvdc_lines",
    "bus": "buses",
}


@dataclass
class FunctionDef:
    name: str
    params: list[str]
    body: str


@dataclass
class ParsedAction:
    kind: ActionKind
    switch_id: str | None = None
    line_id: str | None = None
    target_ids: list[str] = field(default_factory=list)
    dynamic_cluster: dict | None = None
    fault_component_type: str | None = None
    fault_component_id: str | None = None


@dataclass
class ResolvedAction:
    kind: ActionKind
    locations: list[str]


@dataclass
class SwitchInfo:
    switch_id: str
    voltage_level_id: str
    node1: int
    node2: int


@dataclass
class IidmNetwork:
    """Lightweight IIDM index built from XML (.iidm, .xiidm, .xml, .xz)."""

    switches: dict[str, SwitchInfo] = field(default_factory=dict)
    vl_injections: dict[str, dict[int, list[str]]] = field(default_factory=dict)
    components: dict[str, list[dict]] = field(default_factory=dict)

    def resolve_switch(self, switch_id: str) -> list[str]:
        info = self.switches.get(switch_id)
        if info is None:
            return [switch_id]

        node_map = self.vl_injections.get(info.voltage_level_id, {})
        matched: list[str] = []
        for node in (info.node1, info.node2):
            matched.extend(node_map.get(node, []))

        if matched:
            seen: set[str] = set()
            out: list[str] = []
            for name in matched:
                if name not in seen:
                    seen.add(name)
                    out.append(name)
            return out
        return [info.voltage_level_id]

    def resolve_line(self, line_id: str) -> list[str]:
        return [line_id]

    def resolve_fault(self, component_type: str, component_id: str) -> list[str]:
        return [component_id]

    def resolve_targets(
        self,
        target_ids: list[str],
        dynamic_cluster: dict | None,
    ) -> list[str]:
        if dynamic_cluster is not None:
            return self._resolve_dynamic_cluster(dynamic_cluster)
        return list(target_ids)

    def _resolve_dynamic_cluster(self, spec: dict) -> list[str]:
        # Works for any component type in the DSL "on <type>" clause (load, generator, …).
        # Types like bus/asset have no IIDM component bucket for cluster expansion → [].
        bucket = COMPONENT_TYPE_BUCKETS.get(spec["component_type"], spec["component_type"])
        components = self.components.get(bucket, [])
        filter_type = spec["filter_type"]

        if filter_type == "regex":
            pattern = spec["pattern"]
            try:
                regex = re.compile(pattern, re.IGNORECASE)
            except re.error as exc:
                raise ValueError(f"Invalid regex in DSL: {pattern}") from exc
            matched = [c["id"] for c in components if c.get("id") and regex.search(c["id"])]
            if not matched and pattern == r"^Y.*(T|Y|TR).[12].$":
                fallback = re.compile(r"^Y.*(T|Y|TR).[12]$", re.IGNORECASE)
                matched = [
                    c["id"] for c in components if c.get("id") and fallback.match(c["id"])
                ]
            return matched

        if filter_type == "group_by":
            prop = spec["property"]
            value = spec["value"]
            out: list[str] = []
            for comp in components:
                cid = comp.get("id")
                if not cid:
                    continue
                if prop == "tags":
                    if value in (comp.get("tags") or []):
                        out.append(cid)
                elif str(_get_attr(comp, prop, "")) == str(value):
                    out.append(cid)
            return out

        if filter_type == "property":
            return _resolve_property_filter(components, spec)

        raise ValueError(f"Unsupported dynamic cluster filter: {filter_type}")


def _resolve_property_filter(components: list[dict], spec: dict) -> list[str]:
    """Match components by numeric/string comparison on an IIDM attribute."""
    property_name = str(spec["property"]).strip()
    operator = str(spec["operator"])
    threshold = float(spec["value"])
    matched: list[str] = []

    for comp in components:
        cid = comp.get("id")
        if not cid:
            continue
        prop_value = _get_attr(comp, property_name, None)
        if prop_value is None:
            continue
        try:
            numeric_prop = float(prop_value)
            numeric_threshold = float(threshold)
            if operator == "<" and numeric_prop < numeric_threshold:
                matched.append(cid)
            elif operator == ">" and numeric_prop > numeric_threshold:
                matched.append(cid)
            elif operator == "<=" and numeric_prop <= numeric_threshold:
                matched.append(cid)
            elif operator == ">=" and numeric_prop >= numeric_threshold:
                matched.append(cid)
            elif operator == "==" and abs(numeric_prop - numeric_threshold) < 1e-6:
                matched.append(cid)
            elif operator == "!=" and abs(numeric_prop - numeric_threshold) >= 1e-6:
                matched.append(cid)
        except (TypeError, ValueError):
            if operator == "==" and str(prop_value) == str(threshold):
                matched.append(cid)
            elif operator == "!=" and str(prop_value) != str(threshold):
                matched.append(cid)

    return matched


def _snake_to_camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def _get_attr(record: dict, name: str, default=None):
    if name in record:
        return record[name]
    camel = _snake_to_camel(name)
    if camel in record:
        return record[camel]
    return default


def _local_tag(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _load_network_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    if path.suffix.lower() == ".xz" or raw.startswith(b"\xfd7zXZ\x00"):
        return lzma.decompress(raw)
    return raw


def load_iidm(path: str | Path) -> IidmNetwork:
    path = Path(path)
    data = _load_network_bytes(path)
    root = ET.fromstring(data)
    network = IidmNetwork()

    bucket_names = {
        "line": "lines",
        "generator": "generators",
        "load": "loads",
        "switch": "switches",
        "twoWindingsTransformer": "two_windings_transformers",
        "threeWindingsTransformer": "three_windings_transformers",
        "substation": "substations",
        "voltageLevel": "voltage_levels",
        "battery": "batteries",
        "staticVarCompensator": "static_var_compensators",
        "shunt": "shunt_compensators",
        "hvdcLine": "hvdc_lines",
        "busbarSection": "busbar_sections",
    }

    substation_by_vl: dict[str, str] = {}
    for substation in root.iter():
        if _local_tag(substation.tag) != "substation":
            continue
        sub_id = substation.attrib.get("id")
        if not sub_id:
            continue
        for vl in substation:
            if _local_tag(vl.tag) == "voltageLevel":
                vl_id = vl.attrib.get("id")
                if vl_id:
                    substation_by_vl[vl_id] = sub_id

    for elem in root.iter():
        tag = _local_tag(elem.tag)
        if tag not in bucket_names:
            continue
        bucket = bucket_names[tag]
        attrs = dict(elem.attrib)
        if "id" not in attrs:
            continue
        network.components.setdefault(bucket, []).append(attrs)

    for vl in root.iter():
        if _local_tag(vl.tag) != "voltageLevel":
            continue
        vl_id = vl.attrib.get("id")
        if not vl_id:
            continue

        node_map: dict[int, list[str]] = {}
        network.vl_injections[vl_id] = node_map
        sub_id = substation_by_vl.get(vl_id)

        for child in vl.iter():
            child_tag = _local_tag(child.tag)
            if child_tag == "switch":
                sid = child.attrib.get("id")
                if not sid:
                    continue
                try:
                    n1 = int(child.attrib["node1"])
                    n2 = int(child.attrib["node2"])
                except (KeyError, ValueError):
                    continue
                network.switches[sid] = SwitchInfo(sid, vl_id, n1, n2)

            elif child_tag in {"generator", "load"}:
                cid = child.attrib.get("id")
                node_raw = child.attrib.get("node")
                if not cid or node_raw is None:
                    continue
                try:
                    node = int(node_raw)
                except ValueError:
                    continue
                node_map.setdefault(node, []).append(cid)
                bucket = "generators" if child_tag == "generator" else "loads"
                for comp in network.components.get(bucket, []):
                    if comp.get("id") == cid:
                        comp["voltage_level_id"] = vl_id
                        if sub_id:
                            comp["substation_id"] = sub_id

    return network


def _strip_comments(text: str) -> str:
    return re.sub(r"//[^\n]*", "", text)


def _find_matching_delimiter(text: str, open_pos: int, open_ch: str, close_ch: str) -> int:
    depth = 0
    in_string = False
    i = open_pos
    while i < len(text):
        ch = text[i]
        if ch == '"' and (i == 0 or text[i - 1] != "\\"):
            in_string = not in_string
        elif not in_string:
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    raise ValueError(f"Unbalanced {open_ch}{close_ch} in DSL text")


def _split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    in_string = False
    for ch in text:
        if ch == '"':
            in_string = not in_string
            current.append(ch)
        elif not in_string:
            if ch in "([{":
                depth += 1
                current.append(ch)
            elif ch in ")]}":
                depth -= 1
                current.append(ch)
            elif ch == "," and depth == 0:
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts


def _parse_function_definitions(text: str) -> tuple[dict[str, FunctionDef], str]:
    functions: dict[str, FunctionDef] = {}
    parts: list[str] = []
    cursor = 0
    while cursor < len(text):
        match = re.search(r"\bfunc\s+(\w+)\s*\(", text[cursor:])
        if not match:
            parts.append(text[cursor:])
            break
        start = cursor + match.start()
        parts.append(text[cursor:start])
        name = match.group(1)
        paren_open = cursor + match.end() - 1
        paren_close = _find_matching_delimiter(text, paren_open, "(", ")")
        params_raw = text[paren_open + 1 : paren_close]
        params = [p.strip() for p in params_raw.split(",") if p.strip()]
        brace_open = text.find("{", paren_close)
        if brace_open == -1:
            parts.append(text[start:])
            break
        brace_close = _find_matching_delimiter(text, brace_open, "{", "}")
        body = text[brace_open + 1 : brace_close]
        functions[name] = FunctionDef(name=name, params=params, body=body)
        cursor = brace_close + 1
    return functions, "".join(parts)


def _parse_call_bindings(arg_str: str, param_names: list[str]) -> dict[str, str]:
    bindings: dict[str, str] = {}
    positional = 0
    for piece in _split_top_level_commas(arg_str):
        if not piece:
            continue
        named = re.match(r"(\w+)\s*:\s*(.+)", piece, re.DOTALL)
        if named:
            bindings[named.group(1)] = named.group(2).strip()
        elif positional < len(param_names):
            bindings[param_names[positional]] = piece.strip()
            positional += 1
    return bindings


def _substitute_params(body: str, bindings: dict[str, str]) -> str:
    result = body
    for param, value in sorted(bindings.items(), key=lambda item: len(item[0]), reverse=True):
        result = re.sub(rf"\b{re.escape(param)}\b", value, result)
    return result


def _unwrap_check_and_apply(text: str) -> str:
    pattern = re.compile(r"checkAndApply\s*\(", re.IGNORECASE)
    while True:
        match = pattern.search(text)
        if not match:
            break
        paren_open = match.end() - 1
        paren_close = _find_matching_delimiter(text, paren_open, "(", ")")
        inner = text[match.end() : paren_close].strip()
        text = text[: match.start()] + inner + text[paren_close + 1 :]
    return text


def _expand_function_calls(text: str, functions: dict[str, FunctionDef]) -> str:
    if not functions:
        return text

    names = sorted(functions.keys(), key=len, reverse=True)
    call_head = re.compile(rf"\b({'|'.join(re.escape(n) for n in names)})\s*\(")

    while True:
        matches = list(call_head.finditer(text))
        if not matches:
            break

        chosen = None
        for match in matches:
            name = match.group(1)
            paren_open = match.end() - 1
            paren_close = _find_matching_delimiter(text, paren_open, "(", ")")
            arg_str = text[match.end() : paren_close]
            if not call_head.search(arg_str):
                chosen = (match.start(), paren_close + 1, name, arg_str)
                break

        if chosen is None:
            match = matches[0]
            name = match.group(1)
            paren_open = match.end() - 1
            paren_close = _find_matching_delimiter(text, paren_open, "(", ")")
            arg_str = text[match.end() : paren_close]
            chosen = (match.start(), paren_close + 1, name, arg_str)

        start, end, name, arg_str = chosen
        fdef = functions[name]
        bindings = _parse_call_bindings(arg_str, fdef.params)
        expanded = _substitute_params(fdef.body, bindings)
        text = text[:start] + expanded + text[end:]

    return text


def _prepare_dsl_text(raw: str) -> str:
    text = _strip_comments(raw)
    functions, program = _parse_function_definitions(text)
    program = _unwrap_check_and_apply(program)
    program = _expand_function_calls(program, functions)
    return program


def _normalize_id(value: str) -> str:
    return value.strip()


def _extract_quoted_strings(chunk: str) -> list[str]:
    return [
        normalized
        for item in re.findall(r'"([^"]*)"', chunk)
        if (normalized := _normalize_id(item))
    ]


def _split_when_condition_executable(clause: str) -> tuple[str, str]:
    """Split ``<condition> , <executable>`` at the top-level comma."""
    depth = 0
    in_string = False
    for index, char in enumerate(clause):
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(depth - 1, 0)
        elif char == "," and depth == 0:
            return clause[:index].strip(), clause[index + 1 :].strip()
    return clause.strip(), ""


def _extract_ids_from_condition_text(condition: str) -> list[str]:
    locations: list[str] = []
    for pattern in (_CONDITION_AT_COMPONENT, _CONDITION_ON_LINE):
        for match in pattern.finditer(condition):
            component_id = match.group(1).strip()
            if component_id:
                locations.append(component_id)
    return locations


def extract_condition_locations(text: str) -> list[str]:
    """Extract IIDM ids monitored in ``when`` conditions (action_locations only)."""
    prepared = _prepare_dsl_text(text)
    locations: list[str] = []

    for match in _WHEN_STATEMENT.finditer(prepared):
        start = match.end()
        end = prepared.find(";", start)
        if end == -1:
            continue
        clause = prepared[start:end].strip()
        condition, _executable = _split_when_condition_executable(clause)
        locations.extend(_extract_ids_from_condition_text(condition))

    return _unique_preserve_order(locations)


def parse_target(chunk: str) -> tuple[list[str], dict | None]:
    chunk = chunk.strip()
    if chunk.startswith("dynamic cluster"):
        m = re.search(
            r'dynamic\s+cluster\s+where\s+(?:"([^"]*)"|(\w+))\s*'
            r"(<=|>=|==|!=|<|>)\s*([-+]?[0-9.]+)"
            r"(?:\s*(?:pu|Hz|MW|MVar|kV))?\s+on\s+(\w+)",
            chunk,
            re.IGNORECASE,
        )
        if m:
            property_name = (m.group(1) or m.group(2) or "").strip()
            return [], {
                "filter_type": "property",
                "property": property_name,
                "operator": m.group(3),
                "value": float(m.group(4)),
                "component_type": m.group(5),
            }

        m = re.search(
            r'dynamic\s+cluster\s+regex\s+"([^"]*)"\s+on\s+(\w+)',
            chunk,
            re.IGNORECASE,
        )
        if m:
            return [], {
                "filter_type": "regex",
                "pattern": m.group(1),
                "component_type": m.group(2),
            }

        m = re.search(
            r'dynamic\s+cluster\s+group\s+by\s+"([^"]*)"\s+"([^"]*)"\s+on\s+(\w+)',
            chunk,
            re.IGNORECASE,
        )
        if m:
            return [], {
                "filter_type": "group_by",
                "property": _normalize_id(m.group(1)),
                "value": _normalize_id(m.group(2)),
                "component_type": m.group(3),
            }

        raise ValueError(f"Could not parse dynamic cluster target: {chunk}")

    m = re.search(r'cluster\s+(?:"[^"]*"\s+)?\[(.*?)\]', chunk, re.IGNORECASE | re.DOTALL)
    if m:
        return _extract_quoted_strings(m.group(1)), None

    if chunk.startswith("["):
        return _extract_quoted_strings(chunk), None

    m = re.match(r'"([^"]*)"', chunk)
    if m:
        normalized = _normalize_id(m.group(1))
        return ([normalized] if normalized else []), None

    m = re.match(r"(\S+)", chunk)
    if m:
        return [_normalize_id(m.group(1))], None

    return [], None


def parse_dsl(text: str) -> list[ParsedAction]:
    text = _prepare_dsl_text(text)
    ordered: list[tuple[int, ParsedAction]] = []

    def add(pos: int, action: ParsedAction) -> None:
        ordered.append((pos, action))

    for m in re.finditer(
        r"(open|close)\s+switch\s+\"([^\"]+)\"",
        text,
        re.IGNORECASE,
    ):
        verb = m.group(1).lower()
        add(
            m.start(),
            ParsedAction(
                kind="open_switch" if verb == "open" else "close_switch",
                switch_id=_normalize_id(m.group(2)),
            ),
        )

    for m in re.finditer(
        r"(open|close)\s+line\s+\"([^\"]+)\"",
        text,
        re.IGNORECASE,
    ):
        verb = m.group(1).lower()
        add(
            m.start(),
            ParsedAction(
                kind="open_line" if verb == "open" else "close_line",
                line_id=_normalize_id(m.group(2)),
            ),
        )

    for m in re.finditer(
        r"(increase|decrease)\s+load\s+(.+?)\s+by\s+[-+0-9.]+\s*%",
        text,
        re.IGNORECASE | re.DOTALL,
    ):
        target_chunk = m.group(2).strip()
        verb = m.group(1).lower()
        ids, dyn = parse_target(target_chunk)
        add(
            m.start(),
            ParsedAction(
                kind="increase_load" if verb == "increase" else "decrease_load",
                target_ids=ids,
                dynamic_cluster=dyn,
            ),
        )

    for m in re.finditer(
        r"(increase|decrease)\s+generator\s+(.+?)\s+by\s+[-+0-9.]+\s*%",
        text,
        re.IGNORECASE | re.DOTALL,
    ):
        target_chunk = m.group(2).strip()
        verb = m.group(1).lower()
        ids, dyn = parse_target(target_chunk)
        add(
            m.start(),
            ParsedAction(
                kind="increase_generator" if verb == "increase" else "decrease_generator",
                target_ids=ids,
                dynamic_cluster=dyn,
            ),
        )

    for m in re.finditer(
        r"set\s+load\s+(.+?)\s+(?:from\s+[-+0-9.]+\s+)?to\s+[-+0-9.]",
        text,
        re.IGNORECASE | re.DOTALL,
    ):
        target_chunk = m.group(1).strip()
        ids, dyn = parse_target(target_chunk)
        add(m.start(), ParsedAction(kind="set_load", target_ids=ids, dynamic_cluster=dyn))

    for m in re.finditer(
        r"set\s+generator\s+(.+?)\s+(?:from\s+[-+0-9.]+\s+)?to\s+[-+0-9.]",
        text,
        re.IGNORECASE | re.DOTALL,
    ):
        target_chunk = m.group(1).strip()
        ids, dyn = parse_target(target_chunk)
        add(
            m.start(),
            ParsedAction(kind="set_generator", target_ids=ids, dynamic_cluster=dyn),
        )

    for m in re.finditer(
        r"(apply|clear)\s+fault\s+at\s+(\w+)\s+\"([^\"]+)\"",
        text,
        re.IGNORECASE,
    ):
        verb = m.group(1).lower()
        add(
            m.start(),
            ParsedAction(
                kind="apply_fault" if verb == "apply" else "clear_fault",
                fault_component_type=m.group(2),
                fault_component_id=_normalize_id(m.group(3)),
            ),
        )

    ordered.sort(key=lambda item: item[0])
    return [action for _, action in ordered]


def resolve_actions(
    actions: Iterable[ParsedAction], network: IidmNetwork
) -> list[ResolvedAction]:
    resolved: list[ResolvedAction] = []

    for action in actions:
        locations: list[str] = []

        if action.kind in {"open_switch", "close_switch"}:
            assert action.switch_id is not None
            locations = network.resolve_switch(action.switch_id)
        elif action.kind in {"open_line", "close_line"}:
            assert action.line_id is not None
            locations = network.resolve_line(action.line_id)
        elif action.kind in {
            "increase_load",
            "decrease_load",
            "set_load",
            "increase_generator",
            "decrease_generator",
            "set_generator",
        }:
            locations = network.resolve_targets(action.target_ids, action.dynamic_cluster)
        elif action.kind in {"apply_fault", "clear_fault"}:
            assert action.fault_component_type and action.fault_component_id
            locations = network.resolve_fault(
                action.fault_component_type, action.fault_component_id
            )
        else:
            continue

        resolved.append(ResolvedAction(kind=action.kind, locations=locations))

    return resolved


def _unique_preserve_order(names: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def read_dsl(dsl_path: str | Path, iidm_path: str | Path) -> tuple[list[str], list[str]]:
    """
    Read a DSL file and IIDM network; return (action_locations, events_list).
    """
    dsl_text = Path(dsl_path).read_text(encoding="utf-8")
    network = load_iidm(iidm_path)
    actions = parse_dsl(dsl_text)
    resolved = resolve_actions(actions, network)
    condition_locations = extract_condition_locations(dsl_text)

    action_locations: list[str] = []
    events_list: list[str] = []

    for item in resolved:
        action_locations.extend(item.locations)
        if item.kind in EVENT_KINDS:
            events_list.extend(item.locations)

    action_locations.extend(condition_locations)

    return _unique_preserve_order(action_locations), _unique_preserve_order(events_list)
