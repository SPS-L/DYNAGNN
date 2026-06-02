# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: Generator SNom extraction

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import xml.etree.ElementTree as ET
import yaml

try:
    from .curve_generation import op_sort_key
except ImportError:  # pragma: no cover
    from curve_generation import op_sort_key

try:
    from .paths import CONFIG_PATH, INPUTS_DIR, SNOM_DIR, snom_csv_for_operating_point
except ImportError:  # pragma: no cover
    from paths import CONFIG_PATH, INPUTS_DIR, SNOM_DIR, snom_csv_for_operating_point


@dataclass
class GeneratorSnomRecord:
    static_id: str
    op_name: str
    snom_mva: float
    source: str
    dynamic_model_id: str
    par_id: str
    notes: str


@dataclass(frozen=True)
class GeneratorParRef:
    par_id: str
    model_id: str
    par_file: str


EVENTS_DYD_BASENAME = "events.dyd"


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


def local_tag(tag: str) -> str:
    return tag.split("}", 1)[-1] if tag.startswith("{") else tag


def _safe_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def find_iidm_file(op_dir: Path) -> Path:
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
    if not candidates:
        raise FileNotFoundError(f"No IIDM/XIIDM file found in {op_dir}")
    return candidates[0]


def find_dyd_file(op_dir: Path) -> Path:
    """Return the preferred single case DYD (legacy helper)."""
    return find_case_dyd_files(op_dir)[0]


def find_case_dyd_files(op_dir: Path) -> list[Path]:
    """Return case DYD files in ``op_dir``, excluding ``Events.dyd``."""
    candidates = sorted(
        [
            path
            for path in op_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".dyd"
            and path.name.lower() != EVENTS_DYD_BASENAME
        ],
        key=lambda p: (0 if "snapshot" in p.name.lower() else 1, p.name.lower()),
    )
    if not candidates:
        raise FileNotFoundError(f"No case DYD file found in {op_dir}")
    return candidates


def resolve_par_path(op_dir: Path, par_file: str) -> Path:
    path = Path(par_file)
    if path.is_absolute():
        return path
    return (op_dir / path).resolve()


def parse_iidm_generators(
    iidm_path: Path,
    *,
    country_filter: Optional[set[str]] = None,
) -> dict[str, dict[str, Optional[float]]]:
    """Return IIDM generator static IDs and limit-based proxy Snom inputs."""
    root = ET.parse(iidm_path).getroot()
    generators: dict[str, dict[str, Optional[float]]] = {}

    if country_filter is None:
        gen_elems = [element for element in root.iter() if local_tag(element.tag) == "generator"]
    else:
        gen_elems = []
        for substation in root.iter():
            if local_tag(substation.tag) != "substation":
                continue
            country = (substation.get("country") or "").strip().upper()
            if country not in country_filter:
                continue
            for element in substation.iter():
                if local_tag(element.tag) == "generator":
                    gen_elems.append(element)

    for gen in gen_elems:
        static_id = gen.get("id")
        if not static_id:
            continue

        max_p = _safe_float(gen.get("maxP"))
        max_q_at_max_p = 0.0
        for element in gen:
            if local_tag(element.tag) != "reactiveCapabilityCurve":
                continue
            for point in element:
                if local_tag(point.tag) != "point":
                    continue
                point_p = _safe_float(point.get("p"))
                max_q = _safe_float(point.get("maxQ"))
                if (
                    max_p is not None
                    and point_p is not None
                    and math.isclose(point_p, max_p, rel_tol=0.0, abs_tol=1e-9)
                ):
                    max_q_at_max_p = max_q if max_q is not None else 0.0
                    break
            break

        limit_proxy = math.hypot(max_p, max_q_at_max_p) if max_p is not None else None
        generators[static_id] = {"limit_proxy_mva": limit_proxy}

    return generators


def parse_dyd_generator_par_refs(dyd_path: Path) -> dict[str, GeneratorParRef]:
    """Map IIDM ``staticId`` -> PAR reference for generator blackBoxModels."""
    root = ET.parse(dyd_path).getroot()
    mapping: dict[str, GeneratorParRef] = {}

    for element in root.iter():
        if local_tag(element.tag) != "blackBoxModel":
            continue
        lib = element.get("lib") or ""
        static_id = element.get("staticId") or ""
        par_id = element.get("parId") or ""
        par_file = element.get("parFile") or ""
        model_id = element.get("id") or ""
        if not static_id or not par_id or not par_file or "Generator" not in lib:
            continue
        if static_id in mapping:
            continue
        mapping[static_id] = GeneratorParRef(
            par_id=par_id,
            model_id=model_id,
            par_file=par_file,
        )

    return mapping


def merge_dyd_generator_par_refs(dyd_paths: Sequence[Path]) -> dict[str, GeneratorParRef]:
    merged: dict[str, GeneratorParRef] = {}
    for dyd_path in dyd_paths:
        for static_id, ref in parse_dyd_generator_par_refs(dyd_path).items():
            if static_id not in merged:
                merged[static_id] = ref
    return merged


def lookup_generator_snom(
    op_dir: Path,
    par_cache: dict[Path, dict[str, float]],
    *,
    par_file: str,
    par_id: str,
) -> Optional[float]:
    par_path = resolve_par_path(op_dir, par_file)
    if par_path not in par_cache:
        if not par_path.is_file():
            return None
        par_cache[par_path] = parse_par_generator_snom(par_path)
    return par_cache[par_path].get(par_id)


def parse_par_generator_snom(par_path: Path) -> dict[str, float]:
    root = ET.parse(par_path).getroot()
    by_set_id: dict[str, float] = {}

    for element in root.iter():
        if local_tag(element.tag) != "set":
            continue
        set_id = element.get("id") or ""
        if not set_id:
            continue
        for child in element:
            if local_tag(child.tag) != "par":
                continue
            if child.get("name") != "generator_SNom":
                continue
            value = _safe_float(child.get("value"))
            if value is not None:
                by_set_id[set_id] = value
            break

    return by_set_id


def build_records_for_operating_point(
    op_dir: Path,
    *,
    country_filter: Optional[set[str]] = None,
    dyd_path: Optional[Path] = None,
    dyd_paths: Optional[Sequence[Path]] = None,
) -> list[GeneratorSnomRecord]:
    iidm_path = find_iidm_file(op_dir)
    if dyd_paths is not None:
        case_dyd_paths = list(dyd_paths)
    elif dyd_path is not None:
        case_dyd_paths = [dyd_path]
    else:
        case_dyd_paths = find_case_dyd_files(op_dir)

    iidm_gens = parse_iidm_generators(iidm_path, country_filter=country_filter)
    static_to_par = merge_dyd_generator_par_refs(case_dyd_paths)
    par_cache: dict[Path, dict[str, float]] = {}

    records: list[GeneratorSnomRecord] = []
    for static_id, proxies in iidm_gens.items():
        par_id = ""
        model_id = ""
        exact_snom = None
        source = "proxy_curve"
        notes = ""

        par_ref = static_to_par.get(static_id)
        if par_ref is not None:
            par_id = par_ref.par_id
            model_id = par_ref.model_id
            exact_snom = lookup_generator_snom(
                op_dir,
                par_cache,
                par_file=par_ref.par_file,
                par_id=par_ref.par_id,
            )
            if exact_snom is None and not resolve_par_path(op_dir, par_ref.par_file).is_file():
                notes = f"PAR file not found: {par_ref.par_file}"

        limit_proxy = proxies["limit_proxy_mva"]

        if exact_snom is not None:
            chosen = exact_snom
            source = "dynawo_par_generator_SNom"
        elif limit_proxy is not None:
            chosen = limit_proxy
            source = "proxy_sqrt_maxp2_maxq_at_maxp2"
            notes = "No dynamic SNom; used sqrt(maxP^2 + maxQ_at_maxP^2) from IIDM."
        else:
            chosen = 0.0
            source = "proxy_zero_fallback"
            notes = "No dynamic SNom and no IIDM maxP; fallback to 0."

        records.append(
            GeneratorSnomRecord(
                static_id=static_id,
                op_name=op_dir.name,
                snom_mva=float(chosen),
                source=source,
                dynamic_model_id=model_id,
                par_id=par_id,
                notes=notes,
            )
        )

    return records


def write_snom_csv_for_operating_point(
    records: list[GeneratorSnomRecord],
    output_csv: Path,
) -> Path:
    """Write one OP SNom table: columns ``static_id``, ``snom_mva``."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["static_id", "snom_mva"])
        writer.writeheader()
        for record in sorted(records, key=lambda item: item.static_id):
            writer.writerow(
                {
                    "static_id": record.static_id,
                    "snom_mva": f"{record.snom_mva:.6f}",
                }
            )
    return output_csv


def load_generator_snom_for_operating_point(csv_path: Path) -> dict[str, float]:
    """Load static_id -> SNom for one operating point CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing generator SNom table: {csv_path}")

    values: dict[str, float] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Invalid generator SNom header in {csv_path}")

        if "static_id" in reader.fieldnames and "snom_mva" in reader.fieldnames:
            for row in reader:
                static_id = str(row.get("static_id", "")).strip()
                if not static_id:
                    continue
                try:
                    values[static_id] = float(str(row.get("snom_mva", "")).strip())
                except ValueError:
                    continue
            return values

        # Legacy wide matrix row: operating_point + generator columns.
        static_ids = [name for name in reader.fieldnames if name != "operating_point"]
        for row in reader:
            for static_id, cell in zip(static_ids, (row.get(name) for name in static_ids)):
                text = "" if cell is None else str(cell).strip()
                if not text:
                    continue
                try:
                    values[str(static_id).strip()] = float(text)
                except ValueError:
                    continue
            break
    return values


def load_generator_snom_by_operating_point(
    snom_dir: Path = SNOM_DIR,
) -> dict[str, dict[str, float]]:
    """Load all per-OP SNom tables from ``data/generator_Snom/``."""
    if not snom_dir.exists():
        raise FileNotFoundError(f"Missing generator SNom directory: {snom_dir}")

    snom_by_op: dict[str, dict[str, float]] = {}
    for csv_path in sorted(snom_dir.glob("operating_point_*.csv"), key=op_sort_key):
        op_name = csv_path.stem
        snom_by_op[op_name] = load_generator_snom_for_operating_point(csv_path)
    if not snom_by_op:
        raise FileNotFoundError(f"No operating_point_*.csv files found in {snom_dir}")
    return snom_by_op


def build_generator_snom_for_operating_point(
    op_dir: Path,
    *,
    output_dir: Path = SNOM_DIR,
    country_filter: Optional[set[str]] = None,
    dyd_path: Optional[Path] = None,
) -> tuple[Path, int]:
    records = build_records_for_operating_point(
        op_dir,
        country_filter=country_filter,
        dyd_path=dyd_path,
    )
    if not records:
        raise RuntimeError(f"No generator SNom records were produced for {op_dir.name}")
    output_csv = output_dir / f"{op_dir.name}.csv"
    write_snom_csv_for_operating_point(records, output_csv)
    return output_csv, len(records)


def build_generator_snom_tables(
    *,
    inputs_dir: Path = INPUTS_DIR,
    output_dir: Path = SNOM_DIR,
    country_filter: Optional[set[str]] = None,
    config: Optional[dict] = None,
) -> list[Path]:
    """Build ``data/generator_Snom/operating_point_*.csv`` from ``data/inputs/``."""
    cfg = config if config is not None else load_config()
    if country_filter is None:
        country_filter = get_country_filter(cfg)

    if not inputs_dir.exists():
        raise FileNotFoundError(f"Missing inputs directory: {inputs_dir}")

    op_dirs = sorted(
        [path for path in inputs_dir.iterdir() if path.is_dir() and path.name.startswith("operating_point_")],
        key=op_sort_key,
    )
    if not op_dirs:
        raise RuntimeError(f"No operating_point_* folders found in {inputs_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for op_dir in op_dirs:
        path, _ = build_generator_snom_for_operating_point(
            op_dir,
            output_dir=output_dir,
            country_filter=country_filter,
        )
        outputs.append(path)
    return outputs


# Backwards-compatible alias.
build_generator_snom_matrix = build_generator_snom_tables
