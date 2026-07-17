# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: pair-aware GINE training pipeline

from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
import joblib
from sklearn.preprocessing import StandardScaler

from modules.pair_aware_training import attach_pair_aware_targets
from modules.spower_training import run_spower_training
from modules.voltage_training import run_voltage_training
from modules.paths import (
    CONFIG_PATH,
    DATASET_DIR,
    DATA_DIR,
    OP_ELECTRIC_DISTANCE_DIR,
    OP_GRAPHS_DIR,
    PROJECT_ROOT,
)
from modules.pipeline_logging import get_logger, log_step_banner

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device() -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def _canonical_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(value).strip()).upper()


def _event_id_candidates(event_id: str) -> list[str]:
    raw = str(event_id).strip()
    cands: list[str] = [raw]
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'"}:
        cands.append(raw[1:-1])
    if "." in raw:
        parts = [p for p in raw.split(".") if p]
        if parts:
            cands.append(parts[-1])
            cands.append("".join(parts))
    out: list[str] = []
    seen: set[str] = set()
    for c in cands:
        cs = str(c).strip()
        if cs and cs not in seen:
            seen.add(cs)
            out.append(cs)
    return out


def _resolve_graph_path(op_value: Any, graph_dir: Path) -> Path:
    op_raw = str(op_value).strip()
    candidates = [op_raw, f"{op_raw}.pt"]
    if op_raw.isdigit():
        candidates.append(f"operating_point_{op_raw}.pt")
    elif not op_raw.endswith(".pt") and not op_raw.startswith("operating_point_"):
        candidates.append(f"operating_point_{op_raw}.pt")
    for name in candidates:
        p = graph_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No graph file found for operating point '{op_raw}'. Tried: {candidates}")


def _build_event_lookup(metadata: dict) -> dict:
    exact: dict[str, tuple[str, int, str]] = {}
    canonical: dict[str, list[tuple[str, int, str]]] = {}

    def _register(identifier: Any, loc_type: str, idx: int) -> None:
        sid = str(identifier).strip()
        if not sid:
            return
        payload = (loc_type, int(idx), sid)
        if sid not in exact:
            exact[sid] = payload
        cid = _canonical_id(sid)
        canonical.setdefault(cid, []).append(payload)

    for node_key, node_meta in (metadata.get("node_metadata", {}) or {}).items():
        node_idx = int(node_meta.get("index"))
        node_id = str(node_meta.get("id", node_key)).strip()
        _register(node_id, "node", node_idx)
        for bus_id in (node_meta.get("busbarSectionIds", []) or []):
            _register(bus_id, "node", node_idx)
        for bus_id in (node_meta.get("busIds", []) or []):
            _register(bus_id, "node", node_idx)

    for edge_idx, edge_meta in enumerate(metadata.get("edge_metadata", []) or []):
        edge_id = str(edge_meta.get("id", "")).strip()
        _register(edge_id, "edge", edge_idx)

    return {"exact": exact, "canonical": canonical}


def _find_event_location(event_id: Any, metadata: dict, event_lookup: dict) -> tuple[str, int, str]:
    candidates = _event_id_candidates(str(event_id))
    for cand in candidates:
        hit = event_lookup["exact"].get(cand)
        if hit is not None:
            return hit

    for cand in candidates:
        cid = _canonical_id(cand)
        hits = event_lookup["canonical"].get(cid, [])
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise KeyError(f"Event '{event_id}' matched multiple locations for canonical id '{cid}'")

    for cand in candidates:
        cc = _canonical_id(cand)
        if not cc:
            continue
        contains_hits: list[tuple[str, int, str]] = []
        for key, vals in event_lookup["canonical"].items():
            if cc in key or key in cc:
                contains_hits.extend(vals)
        uniq = {(t, i, s) for t, i, s in contains_hits}
        if len(uniq) == 1:
            return next(iter(uniq))

    raise KeyError(f"Event '{event_id}' not found in node metadata or edge metadata")


def _preprocess_column_map(df_columns: Iterable[Any]) -> dict[str, Any]:
    col_map: dict[str, Any] = {}
    for col in df_columns:
        col_str = str(col)
        col_map[col_str] = col
        col_map[col_str.strip()] = col
    return col_map


@dataclass(frozen=True)
class _PreparedMetaVoltage:
    bus_nodes: list[tuple[int, str]]
    event_lookup: dict


@dataclass(frozen=True)
class _PreparedMetaSpower:
    gen_nodes: list[tuple[int, str]]
    event_lookup: dict


def _prepare_metadata_cache_voltage(metadata: dict, *, country_filter: Sequence[str]) -> _PreparedMetaVoltage:
    node_list: list[tuple[int, str, str, bool, str]] = []
    for node_key, node_meta in (metadata.get("node_metadata", {}) or {}).items():
        idx = int(node_meta["index"])
        ntype = str(node_meta.get("type", "")).lower()
        nid = str(node_meta.get("id", node_key)).strip()
        has_freq = bool(node_meta.get("has_frequency", False))
        country = str(node_meta.get("country", "")).upper()
        node_list.append((idx, ntype, nid, has_freq, country))
    node_list = sorted(node_list, key=lambda item: item[0])

    allow = {str(c).upper() for c in (country_filter or []) if str(c).strip()}
    def _keep_country(c: str) -> bool:
        return True if not allow else str(c).upper() in allow

    bus_nodes = [
        (idx, nid)
        for idx, ntype, nid, _, country in node_list
        if ntype == "bus" and _keep_country(country)
    ]
    return _PreparedMetaVoltage(bus_nodes=bus_nodes, event_lookup=_build_event_lookup(metadata))


def _prepare_metadata_cache_spower(metadata: dict, *, country_filter: Sequence[str]) -> _PreparedMetaSpower:
    node_list: list[tuple[int, str, str, str, bool]] = []
    for node_key, node_meta in (metadata.get("node_metadata", {}) or {}).items():
        idx = int(node_meta["index"])
        ntype = str(node_meta.get("type", "")).lower()
        nid = str(node_meta.get("id", node_key)).strip()
        country = str(node_meta.get("country", "")).upper()
        has_dynamic_model = bool(node_meta.get("hasDynamicModel", False))
        node_list.append((idx, ntype, nid, country, has_dynamic_model))
    node_list = sorted(node_list, key=lambda item: item[0])

    allow = {str(c).upper() for c in (country_filter or []) if str(c).strip()}
    def _keep_country(c: str) -> bool:
        return True if not allow else str(c).upper() in allow

    gen_nodes = [
        (idx, nid)
        for idx, ntype, nid, country, has_dynamic_model in node_list
        if ntype == "generator" and _keep_country(country) and has_dynamic_model
    ]
    return _PreparedMetaSpower(gen_nodes=gen_nodes, event_lookup=_build_event_lookup(metadata))


def _build_y_class(row: pd.Series, nodes: Sequence[tuple[int, str]], col_map: dict, num_classes: int) -> torch.Tensor:
    values: list[int] = []
    for _, nid in nodes:
        col_name = str(nid).strip()
        if col_name not in col_map:
            raise KeyError(f"Missing class column for id: {col_name!r}")
        value = row[col_map[col_name]]
        if pd.isna(value):
            raise ValueError(f"NaN class for id {nid}")
        v = int(round(float(value)))
        if v < 0 or v >= num_classes:
            raise ValueError(f"Class {v} out of range [0, {num_classes - 1}] for id {nid}")
        values.append(v)
    return torch.tensor(values, dtype=torch.long)


def build_graph_dataset_voltage(
    dataset_df: pd.DataFrame,
    graph_dir: Path,
    *,
    num_classes: int,
    country_filter: Sequence[str],
    logger: logging.Logger,
) -> tuple[list, list[tuple[int, str, str, str]]]:
    graph_cache: dict[str, dict] = {}
    meta_cache: dict[str, _PreparedMetaVoltage] = {}
    graph_dataset: list = []
    skipped: list[tuple[int, str, str, str]] = []
    col_map = _preprocess_column_map(dataset_df.columns)

    logger.info("Voltage: building graph_dataset from %d rows", len(dataset_df))
    for row_idx, row in dataset_df.iterrows():
        op_value = row.iloc[0]
        event_id = row.iloc[1]
        try:
            graph_path = _resolve_graph_path(op_value, graph_dir)
            graph_key = str(graph_path)
            if graph_key not in graph_cache:
                loaded = torch.load(graph_path, weights_only=False)
                graph_cache[graph_key] = loaded
                meta_cache[graph_key] = _prepare_metadata_cache_voltage(
                    loaded["metadata"],
                    country_filter=country_filter,
                )

            base = graph_cache[graph_key]
            pre_meta = meta_cache[graph_key]
            data_i = base["data"].clone()
            meta_i = base["metadata"]
            data_i.metadata = meta_i
            data_i.node_metadata = meta_i.get("node_metadata", {}) or {}
            data_i.edge_metadata = meta_i.get("edge_metadata", []) or []
            bus_nodes_all = pre_meta.bus_nodes

            bus_nodes: list[tuple[int, str]] = []
            for idx, nid in bus_nodes_all:
                col_name = str(nid).strip()
                if col_name not in col_map:
                    continue
                value = row[col_map[col_name]]
                if pd.isna(value):
                    continue
                bus_nodes.append((idx, nid))

            if not bus_nodes:
                skipped.append((int(row_idx), str(op_value), str(event_id), "No usable bus labels for this row"))
                continue

            try:
                loc_type, loc_idx, loc_id = _find_event_location(event_id, meta_i, pre_meta.event_lookup)
            except KeyError:
                skipped.append((int(row_idx), str(op_value), str(event_id), "Event disconnected (isolated by open switches)"))
                continue

            if loc_type == "node":
                node_schema = list(meta_i.get("node_feature_schema", []) or [])
                if "fault_on" not in node_schema:
                    raise KeyError("'fault_on' not found in node_feature_schema")
                fault_col = node_schema.index("fault_on")
                data_i.x[int(loc_idx), fault_col] = 1.0
            else:
                edge_schema = list(meta_i.get("edge_feature_schema", []) or [])
                if "fault_on" not in edge_schema:
                    raise KeyError("'fault_on' not found in edge_feature_schema")
                fault_col = edge_schema.index("fault_on")
                data_i.edge_attr[int(loc_idx), fault_col] = 1.0
                reverse_idx = int(loc_idx) + 1
                if reverse_idx < int(data_i.edge_attr.shape[0]):
                    data_i.edge_attr[reverse_idx, fault_col] = 1.0

            bus_node_indices = torch.tensor([idx for idx, _ in bus_nodes], dtype=torch.long)
            bus_node_mask = torch.zeros(data_i.x.shape[0], dtype=torch.bool)
            bus_node_mask[bus_node_indices] = True

            y_fr = _build_y_class(row, bus_nodes, col_map, num_classes)
            y_node = torch.full((data_i.x.shape[0],), -1, dtype=torch.long)
            y_node[bus_node_indices] = y_fr

            data_i.y_class = y_node
            data_i.bus_node_indices = bus_node_indices
            data_i.bus_node_mask = bus_node_mask
            data_i.bus_node_names = [nid for _, nid in bus_nodes]
            data_i.op_name = str(op_value)
            data_i.event_id = str(event_id)
            data_i.event_location_type = loc_type
            data_i.event_location_index = int(loc_idx)
            data_i.event_location_id = str(loc_id)
            graph_dataset.append(data_i)
        except Exception as e:
            skipped.append((int(row_idx), str(op_value), str(event_id), f"Error: {str(e)}"))

    return graph_dataset, skipped


def build_graph_dataset_spower(
    dataset_df: pd.DataFrame,
    graph_dir: Path,
    *,
    num_classes: int,
    country_filter: Sequence[str],
    logger: logging.Logger,
) -> tuple[list, list[tuple[int, str, str, str]]]:
    graph_cache: dict[str, dict] = {}
    meta_cache: dict[str, _PreparedMetaSpower] = {}
    graph_dataset: list = []
    skipped: list[tuple[int, str, str, str]] = []
    col_map = _preprocess_column_map(dataset_df.columns)

    logger.info("Spower: building graph_dataset from %d rows", len(dataset_df))
    for row_idx, row in dataset_df.iterrows():
        op_value = row.iloc[0]
        event_id = row.iloc[1]
        try:
            graph_path = _resolve_graph_path(op_value, graph_dir)
            graph_key = str(graph_path)
            if graph_key not in graph_cache:
                loaded = torch.load(graph_path, weights_only=False)
                graph_cache[graph_key] = loaded
                meta_cache[graph_key] = _prepare_metadata_cache_spower(
                    loaded["metadata"],
                    country_filter=country_filter,
                )

            base = graph_cache[graph_key]
            pre_meta = meta_cache[graph_key]
            data_i = base["data"].clone()
            meta_i = base["metadata"]
            data_i.metadata = meta_i
            data_i.node_metadata = meta_i.get("node_metadata", {}) or {}
            data_i.edge_metadata = meta_i.get("edge_metadata", []) or []
            gen_nodes_all = pre_meta.gen_nodes

            gen_nodes: list[tuple[int, str]] = []
            for idx, nid in gen_nodes_all:
                col_name = str(nid).strip()
                if col_name not in col_map:
                    continue
                value = row[col_map[col_name]]
                if pd.isna(value):
                    continue
                gen_nodes.append((idx, nid))

            if not gen_nodes:
                skipped.append((int(row_idx), str(op_value), str(event_id), "No usable generator labels for this row"))
                continue

            try:
                loc_type, loc_idx, loc_id = _find_event_location(event_id, meta_i, pre_meta.event_lookup)
            except KeyError:
                skipped.append((int(row_idx), str(op_value), str(event_id), "Event disconnected or missing"))
                continue

            if loc_type == "node":
                node_schema = list(meta_i.get("node_feature_schema", []) or [])
                if "fault_on" not in node_schema:
                    raise KeyError("'fault_on' not found in node_feature_schema")
                fault_col = node_schema.index("fault_on")
                data_i.x[int(loc_idx), fault_col] = 1.0
            else:
                edge_schema = list(meta_i.get("edge_feature_schema", []) or [])
                if "fault_on" not in edge_schema:
                    raise KeyError("'fault_on' not found in edge_feature_schema")
                fault_col = edge_schema.index("fault_on")
                data_i.edge_attr[int(loc_idx), fault_col] = 1.0
                reverse_idx = int(loc_idx) + 1
                if reverse_idx < int(data_i.edge_attr.shape[0]):
                    data_i.edge_attr[reverse_idx, fault_col] = 1.0

            gen_node_indices = torch.tensor([idx for idx, _ in gen_nodes], dtype=torch.long)
            gen_node_mask = torch.zeros(data_i.x.shape[0], dtype=torch.bool)
            gen_node_mask[gen_node_indices] = True

            y_fr = _build_y_class(row, gen_nodes, col_map, num_classes)
            y_node = torch.full((data_i.x.shape[0],), -1, dtype=torch.long)
            y_node[gen_node_indices] = y_fr

            data_i.y_class = y_node
            data_i.gen_node_indices = gen_node_indices
            data_i.gen_node_mask = gen_node_mask
            data_i.gen_node_names = [nid for _, nid in gen_nodes]
            data_i.op_name = str(op_value)
            data_i.event_id = str(event_id)
            data_i.event_location_type = loc_type
            data_i.event_location_index = int(loc_idx)
            data_i.event_location_id = str(loc_id)
            graph_dataset.append(data_i)
        except Exception as e:
            skipped.append((int(row_idx), str(op_value), str(event_id), f"Error: {str(e)}"))

    return graph_dataset, skipped


def build_graph_dataset_multi(
    *,
    dataset_voltage: pd.DataFrame,
    dataset_spower: pd.DataFrame,
    graph_dir: Path,
    num_classes: int,
    country_filter: Sequence[str],
    logger: logging.Logger,
) -> tuple[list, list[tuple[int, str, str, str]]]:
    """
    Build ONE PyG dataset list (shared graphs) that contains BOTH target/mask sets:
    - voltage: bus_node_mask + y_voltage (stored in data.y_voltage)
    - spower:  gen_node_mask + y_spower (stored in data.y_spower)

    Later, you can switch tasks by setting data.y_class = data.y_voltage or data.y_spower.
    """
    graph_cache: dict[str, dict] = {}
    meta_cache_v: dict[str, _PreparedMetaVoltage] = {}
    meta_cache_s: dict[str, _PreparedMetaSpower] = {}

    out: list = []
    skipped: list[tuple[int, str, str, str]] = []

    col_map_v = _preprocess_column_map(dataset_voltage.columns)
    col_map_s = _preprocess_column_map(dataset_spower.columns)

    # Build lookup for spower rows by (OP, Contingency)
    spower_lookup: dict[tuple[str, str], pd.Series] = {}
    for _, row_s in dataset_spower.iterrows():
        key = (str(row_s.iloc[0]).strip(), str(row_s.iloc[1]).strip())
        spower_lookup[key] = row_s

    logger.info("Multi: building shared graph_dataset from voltage rows=%d", len(dataset_voltage))
    for row_idx, row_v in dataset_voltage.iterrows():
        op_value = row_v.iloc[0]
        event_id = row_v.iloc[1]
        key = (str(op_value).strip(), str(event_id).strip())
        row_s = spower_lookup.get(key)
        if row_s is None:
            skipped.append((int(row_idx), str(op_value), str(event_id), "Missing matching row in Dataset_Spower.csv"))
            continue

        try:
            graph_path = _resolve_graph_path(op_value, graph_dir)
            graph_key = str(graph_path)
            if graph_key not in graph_cache:
                loaded = torch.load(graph_path, weights_only=False)
                graph_cache[graph_key] = loaded
                meta_cache_v[graph_key] = _prepare_metadata_cache_voltage(
                    loaded["metadata"],
                    country_filter=country_filter,
                )
                meta_cache_s[graph_key] = _prepare_metadata_cache_spower(
                    loaded["metadata"],
                    country_filter=country_filter,
                )

            base = graph_cache[graph_key]
            pre_v = meta_cache_v[graph_key]
            pre_s = meta_cache_s[graph_key]

            data_i = base["data"].clone()
            meta_i = base["metadata"]
            data_i.metadata = meta_i
            data_i.node_metadata = meta_i.get("node_metadata", {}) or {}
            data_i.edge_metadata = meta_i.get("edge_metadata", []) or []

            # Resolve event location once (shared)
            try:
                loc_type, loc_idx, loc_id = _find_event_location(event_id, meta_i, pre_v.event_lookup)
            except KeyError:
                skipped.append((int(row_idx), str(op_value), str(event_id), "Event disconnected or missing in metadata"))
                continue

            # Inject fault_on
            if loc_type == "node":
                node_schema = list(meta_i.get("node_feature_schema", []) or [])
                if "fault_on" not in node_schema:
                    raise KeyError("'fault_on' not found in node_feature_schema")
                fault_col = node_schema.index("fault_on")
                data_i.x[int(loc_idx), fault_col] = 1.0
            else:
                edge_schema = list(meta_i.get("edge_feature_schema", []) or [])
                if "fault_on" not in edge_schema:
                    raise KeyError("'fault_on' not found in edge_feature_schema")
                fault_col = edge_schema.index("fault_on")
                data_i.edge_attr[int(loc_idx), fault_col] = 1.0
                reverse_idx = int(loc_idx) + 1
                if reverse_idx < int(data_i.edge_attr.shape[0]):
                    data_i.edge_attr[reverse_idx, fault_col] = 1.0

            # Voltage labels/mask (buses)
            bus_nodes_all = pre_v.bus_nodes
            bus_nodes: list[tuple[int, str]] = []
            for idx, nid in bus_nodes_all:
                col_name = str(nid).strip()
                if col_name not in col_map_v:
                    continue
                value = row_v[col_map_v[col_name]]
                if pd.isna(value):
                    continue
                bus_nodes.append((idx, nid))

            bus_node_indices = torch.tensor([idx for idx, _ in bus_nodes], dtype=torch.long) if bus_nodes else torch.empty((0,), dtype=torch.long)
            bus_node_mask = torch.zeros(data_i.x.shape[0], dtype=torch.bool)
            if bus_nodes:
                bus_node_mask[bus_node_indices] = True

            y_voltage = torch.full((data_i.x.shape[0],), -1, dtype=torch.long)
            if bus_nodes:
                y_fr_v = _build_y_class(row_v, bus_nodes, col_map_v, num_classes)
                y_voltage[bus_node_indices] = y_fr_v

            # Spower labels/mask (generators with dynamic model)
            gen_nodes_all = pre_s.gen_nodes
            gen_nodes: list[tuple[int, str]] = []
            for idx, nid in gen_nodes_all:
                col_name = str(nid).strip()
                if col_name not in col_map_s:
                    continue
                value = row_s[col_map_s[col_name]]
                if pd.isna(value):
                    continue
                gen_nodes.append((idx, nid))

            gen_node_indices = torch.tensor([idx for idx, _ in gen_nodes], dtype=torch.long) if gen_nodes else torch.empty((0,), dtype=torch.long)
            gen_node_mask = torch.zeros(data_i.x.shape[0], dtype=torch.bool)
            if gen_nodes:
                gen_node_mask[gen_node_indices] = True

            y_spower = torch.full((data_i.x.shape[0],), -1, dtype=torch.long)
            if gen_nodes:
                y_fr_s = _build_y_class(row_s, gen_nodes, col_map_s, num_classes)
                y_spower[gen_node_indices] = y_fr_s

            # Store both targets + masks
            data_i.y_voltage = y_voltage
            data_i.y_spower = y_spower
            data_i.bus_node_indices = bus_node_indices
            data_i.bus_node_mask = bus_node_mask
            data_i.gen_node_indices = gen_node_indices
            data_i.gen_node_mask = gen_node_mask
            data_i.op_name = str(op_value)
            data_i.event_id = str(event_id)
            data_i.event_location_type = loc_type
            data_i.event_location_index = int(loc_idx)
            data_i.event_location_id = str(loc_id)

            # Default y_class is voltage (will be switched before spower training)
            data_i.y_class = y_voltage
            out.append(data_i)
        except Exception as e:
            skipped.append((int(row_idx), str(op_value), str(event_id), f"Error: {str(e)}"))

    return out, skipped


_OP_ELECTRIC_CACHE: dict[str, tuple[dict, list[str]]] = {}


def _resolve_electric_distance_path(op_value: Any, electric_distance_dir: Path) -> Path:
    op_raw = str(op_value).strip()
    candidates = [op_raw, f"{op_raw}.csv"]
    if op_raw.isdigit():
        candidates.append(f"operating_point_{op_raw}.csv")
    elif not op_raw.endswith(".csv") and not op_raw.startswith("operating_point_"):
        candidates.append(f"operating_point_{op_raw}.csv")
    for name in candidates:
        p = electric_distance_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No electric-distance CSV found for operating point '{op_raw}'. Tried: {candidates}")


def _node_index_to_bus_vl_id(metadata: dict) -> list[str]:
    num_nodes = 0
    for node_meta in (metadata.get("node_metadata", {}) or {}).values():
        num_nodes = max(num_nodes, int(node_meta["index"]) + 1)
    idx_to_bus_vl = [""] * int(num_nodes)
    for node_meta in (metadata.get("node_metadata", {}) or {}).values():
        idx = int(node_meta["index"])
        idx_to_bus_vl[idx] = str(node_meta.get("voltageLevelId", "")).strip()
    return idx_to_bus_vl


def _load_op_electric_distances(csv_path: Path) -> dict[str, pd.Series]:
    df = pd.read_csv(csv_path, usecols=["VLi", "VLj", "dij"])
    return {str(vli): grp.set_index("VLj")["dij"] for vli, grp in df.groupby("VLi", sort=False)}


def _get_op_electric_context(op_name: str, graph_dir: Path, electric_distance_dir: Path) -> tuple[dict, list[str]]:
    cache_key = str(op_name)
    if cache_key in _OP_ELECTRIC_CACHE:
        return _OP_ELECTRIC_CACHE[cache_key]

    graph_path = _resolve_graph_path(op_name, graph_dir)
    ed_path = _resolve_electric_distance_path(op_name, electric_distance_dir)
    loaded = torch.load(graph_path, weights_only=False)
    idx_to_bus_vl = _node_index_to_bus_vl_id(loaded["metadata"])
    dist_by_vli = _load_op_electric_distances(ed_path)
    _OP_ELECTRIC_CACHE[cache_key] = (dist_by_vli, idx_to_bus_vl)
    return dist_by_vli, idx_to_bus_vl


def _event_anchor_nodes(data_i, loc_type: str, loc_idx: int) -> list[int]:
    num_nodes = int(data_i.x.shape[0])
    if loc_type == "node":
        idx = int(loc_idx)
        return [idx] if 0 <= idx < num_nodes else []
    if loc_type == "edge":
        if data_i.edge_index.numel() == 0:
            return []
        eidx = int(loc_idx)
        if eidx < 0 or eidx >= int(data_i.edge_index.shape[1]):
            return []
        s = int(data_i.edge_index[0, eidx].item())
        d = int(data_i.edge_index[1, eidx].item())
        out: list[int] = []
        if 0 <= s < num_nodes:
            out.append(s)
        if 0 <= d < num_nodes and d != s:
            out.append(d)
        return out
    return []


def _distances_from_precomputed(dist_by_vli: dict, source_vls: list[str], target_vls: list[str]) -> np.ndarray:
    out = np.full(len(target_vls), np.inf, dtype=np.float64)
    for vli in source_vls:
        series = dist_by_vli.get(vli)
        if series is None:
            continue
        d = series.reindex(target_vls).to_numpy(dtype=np.float64, copy=False)
        out = np.minimum(out, d)
    return out


def append_electrical_distance_feature(
    graph_dataset: list,
    *,
    graph_dir: Path,
    electric_distance_dir: Path,
    logger: logging.Logger,
) -> dict[str, int]:
    stats = {
        "total": len(graph_dataset),
        "appended": 0,
        "missing_event": 0,
        "empty_graph": 0,
        "all_disconnected": 0,
        "missing_op_csv": 0,
    }

    for data_i in graph_dataset:
        num_nodes = int(data_i.x.shape[0])
        if num_nodes <= 0:
            stats["empty_graph"] += 1
            continue

        loc_type = str(getattr(data_i, "event_location_type", "missing"))
        loc_idx = int(getattr(data_i, "event_location_index", -1))
        anchors = _event_anchor_nodes(data_i, loc_type, loc_idx)

        if not anchors:
            dz = np.zeros(num_nodes, dtype=np.float64)
            stats["missing_event"] += 1
        else:
            try:
                dist_by_vli, idx_to_bus_vl = _get_op_electric_context(
                    str(getattr(data_i, "op_name", "")),
                    graph_dir,
                    electric_distance_dir,
                )
            except FileNotFoundError:
                dz = np.zeros(num_nodes, dtype=np.float64)
                stats["missing_op_csv"] += 1
            else:
                source_vls: list[str] = []
                for a in anchors:
                    ai = int(a)
                    if 0 <= ai < num_nodes:
                        vl = idx_to_bus_vl[ai]
                        if vl:
                            source_vls.append(vl)
                source_vls = list(dict.fromkeys(source_vls))

                target_vls = sorted({vl for vl in idx_to_bus_vl if vl})
                if not source_vls or not target_vls:
                    dz = np.zeros(num_nodes, dtype=np.float64)
                    stats["missing_event"] += 1
                else:
                    min_dist = _distances_from_precomputed(dist_by_vli, source_vls, target_vls)
                    finite = np.isfinite(min_dist)
                    if finite.any():
                        max_finite = float(min_dist[finite].max())
                        min_dist[~finite] = max_finite + 1.0
                        dz_by_vl = {vl: float(np.log1p(d)) for vl, d in zip(target_vls, min_dist)}
                    else:
                        dz_by_vl = {}
                        stats["all_disconnected"] += 1
                    dz = np.array(
                        [dz_by_vl.get(idx_to_bus_vl[i], 0.0) for i in range(num_nodes)],
                        dtype=np.float64,
                    )

        dz_tensor = torch.tensor(dz, dtype=data_i.x.dtype, device=data_i.x.device)
        data_i.x = torch.cat([data_i.x, dz_tensor.unsqueeze(1)], dim=1)
        data_i.dz_fault = dz_tensor
        stats["appended"] += 1

    logger.info("Electrical distance appended. stats=%s", stats)
    return stats


EDGE_CONT_FEATURE_NAMES = ("r", "x", "b1", "g1", "b2", "g2")


def _edge_cont_cols_from_metadata(data) -> list[int]:
    metadata = getattr(data, "metadata", {}) or {}
    edge_schema = list(metadata.get("edge_feature_schema", []) or [])
    if not edge_schema:
        raise RuntimeError("Missing edge_feature_schema on graph sample; regenerate op_graphs.")
    missing = [name for name in EDGE_CONT_FEATURE_NAMES if name not in edge_schema]
    if missing:
        raise RuntimeError(f"Graph edge_feature_schema missing expected continuous features: {missing}")
    return [edge_schema.index(name) for name in EDGE_CONT_FEATURE_NAMES]


def _scale_split(
    split: Sequence,
    *,
    x_scaler: StandardScaler,
    edge_attr_scaler: StandardScaler,
    node_cont_cols: Sequence[int],
    edge_cont_cols: Sequence[int],
) -> list:
    scaled: list = []
    for d in split:
        d2 = d.clone()
        x_part = d2.x[:, list(node_cont_cols)].cpu().numpy()
        d2.x[:, list(node_cont_cols)] = torch.tensor(x_scaler.transform(x_part), dtype=d2.x.dtype)
        if d2.edge_attr.shape[0] > 0:
            edge_part = d2.edge_attr[:, list(edge_cont_cols)].cpu().numpy()
            d2.edge_attr[:, list(edge_cont_cols)] = torch.tensor(
                edge_attr_scaler.transform(edge_part),
                dtype=d2.edge_attr.dtype,
            )
        scaled.append(d2)
    return scaled


def _split_graph_dataset_by_csv(graph_dataset: Sequence, split_csv: Path) -> tuple[list, list, list]:
    split_df = pd.read_csv(split_csv)
    required = {"split", "operating_point", "contingency"}
    missing = required.difference(split_df.columns)
    if missing:
        raise ValueError(f"Split CSV missing columns: {sorted(missing)}")

    split_lookup = {
        (str(row.operating_point).strip(), str(row.contingency).strip()): str(row.split).strip().lower()
        for row in split_df.itertuples(index=False)
    }
    valid = {"train", "validation", "test"}
    invalid_splits = sorted(set(split_lookup.values()) - valid)
    if invalid_splits:
        raise ValueError(f"Unexpected split names in {split_csv}: {invalid_splits}")

    train_data: list = []
    val_data: list = []
    test_data: list = []
    missing_examples: list[tuple[str, str]] = []

    for d in graph_dataset:
        key = (str(getattr(d, "op_name", "")).strip(), str(getattr(d, "event_id", "")).strip())
        split_name = split_lookup.get(key)
        if split_name is None:
            missing_examples.append(key)
        elif split_name == "train":
            train_data.append(d)
        elif split_name == "validation":
            val_data.append(d)
        elif split_name == "test":
            test_data.append(d)

    if missing_examples:
        preview = missing_examples[:10]
        raise RuntimeError(
            f"{len(missing_examples)} graph_dataset examples were not found in {split_csv}. "
            f"First missing examples: {preview}"
        )
    if not train_data or not val_data or not test_data:
        raise RuntimeError(
            f"Invalid split sizes from {split_csv}: train={len(train_data)}, validation={len(val_data)}, test={len(test_data)}"
        )
    return train_data, val_data, test_data


def main() -> None:
    cfg = _load_config()
    training_cfg = cfg.get("training", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    network_cfg = cfg.get("network", {}) or {}
    country_filter = network_cfg.get("country_filter", []) or []

    log_step_banner("training")
    logger = get_logger()

    training_dir = DATA_DIR / "training"
    model_dir = DATA_DIR / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)

    seed = int(training_cfg.get("seed", 42))
    _set_seed(seed)
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Using device: %s", _device())
    logger.info("Seed: %d", seed)
    logger.info("network.country_filter: %s", country_filter)

    if "num_classes" not in model_cfg:
        raise KeyError("Missing required config key: model.num_classes (in config.yaml)")
    num_classes = int(model_cfg["num_classes"])
    if num_classes < 2:
        raise ValueError(f"model.num_classes must be >= 2, got {num_classes}")
    logger.info("Model: pair-aware residual GINE")
    logger.info("num_classes: %d", num_classes)

    # Paths expected by notebooks (but rooted in this project)
    dataset_voltage_csv = DATASET_DIR / "Dataset_Voltage.csv"
    dataset_spower_csv = DATASET_DIR / "Dataset_Spower.csv"
    split_csv = DATASET_DIR / "train_val_test_split.csv"

    if not split_csv.exists():
        raise FileNotFoundError(
            f"Missing split CSV: {split_csv}. "
            "Run the curve_process stage first (main.py --to-step curve_process)."
        )
    logger.info("Split CSV: %s", split_csv)

    # 1) Load tables
    if not dataset_voltage_csv.exists():
        raise FileNotFoundError(f"Missing voltage dataset CSV: {dataset_voltage_csv}")
    if not dataset_spower_csv.exists():
        raise FileNotFoundError(f"Missing spower dataset CSV: {dataset_spower_csv}")

    dataset_voltage = pd.read_csv(dataset_voltage_csv)
    dataset_spower = pd.read_csv(dataset_spower_csv)
    logger.info("Loaded voltage dataset: %s shape=%s", dataset_voltage_csv, dataset_voltage.shape)
    logger.info("Loaded spower dataset: %s shape=%s", dataset_spower_csv, dataset_spower.shape)
    logger.info("Graph dir: %s", OP_GRAPHS_DIR)
    logger.info("Electric distance dir: %s", OP_ELECTRIC_DISTANCE_DIR)

    # 2) Build ONE shared PyG dataset (both targets/masks) + electrical distance
    graph_dataset, skipped = build_graph_dataset_multi(
        dataset_voltage=dataset_voltage,
        dataset_spower=dataset_spower,
        graph_dir=OP_GRAPHS_DIR,
        num_classes=num_classes,
        country_filter=country_filter,
        logger=logger,
    )
    logger.info("Shared graph_dataset size=%d skipped=%d", len(graph_dataset), len(skipped))
    if skipped:
        logger.info("Skipped preview: %s", skipped[:5])

    append_electrical_distance_feature(
        graph_dataset,
        graph_dir=OP_GRAPHS_DIR,
        electric_distance_dir=OP_ELECTRIC_DISTANCE_DIR,
        logger=logger,
    )

    pair_cfg = training_cfg.get("pair_aware", {}) or {}
    pair_attachment = attach_pair_aware_targets(
        graph_dataset,
        data_dir=DATA_DIR,
        epsilon=float(pair_cfg.get("epsilon", 1.0e-10)),
        logger=logger,
    )

    # 3) Apply shared split and fit feature scalers on the training set only.
    node_cont_cols = [1, 2, 3, 4, 6]
    batch_size = int(training_cfg.get("batch_size", 16))
    train_all, val_all, test_all = _split_graph_dataset_by_csv(graph_dataset, split_csv)
    logger.info("Shared split sizes: train=%d val=%d test=%d", len(train_all), len(val_all), len(test_all))

    # Fit ONE set of scalers (same graph feature space for both tasks)
    x_scaler = StandardScaler()
    x_scaler.fit(np.vstack([d.x[:, node_cont_cols].cpu().numpy() for d in train_all]))
    edge_cont_cols = _edge_cont_cols_from_metadata(train_all[0])
    edge_rows = [d.edge_attr[:, edge_cont_cols].cpu().numpy() for d in train_all if d.edge_attr.shape[0] > 0]
    if not edge_rows:
        raise RuntimeError("No edges in train split; cannot fit edge scaler.")
    edge_attr_scaler = StandardScaler()
    edge_attr_scaler.fit(np.vstack(edge_rows))

    joblib.dump(x_scaler, model_dir / "x_scaler.pkl")
    joblib.dump(edge_attr_scaler, model_dir / "edge_attr_scaler.pkl")
    logger.info("Saved shared scalers to %s", model_dir)

    train_scaled = _scale_split(
        train_all,
        x_scaler=x_scaler,
        edge_attr_scaler=edge_attr_scaler,
        node_cont_cols=node_cont_cols,
        edge_cont_cols=edge_cont_cols,
    )
    val_scaled = _scale_split(
        val_all,
        x_scaler=x_scaler,
        edge_attr_scaler=edge_attr_scaler,
        node_cont_cols=node_cont_cols,
        edge_cont_cols=edge_cont_cols,
    )
    test_scaled = _scale_split(
        test_all,
        x_scaler=x_scaler,
        edge_attr_scaler=edge_attr_scaler,
        node_cont_cols=node_cont_cols,
        edge_cont_cols=edge_cont_cols,
    )

    logger.info("Scaled shared dataset ready. batch_size=%d", batch_size)

    logger.info("Starting Voltage pair-aware GINE Optuna training")
    voltage_result = run_voltage_training(
        train_scaled=train_scaled,
        val_scaled=val_scaled,
        test_scaled=test_scaled,
        training_dir=training_dir,
        model_dir=model_dir,
        config=cfg,
        attachment=pair_attachment,
        logger=logger,
    )

    logger.info("Starting Spower pair-aware GINE Optuna training")
    spower_result = run_spower_training(
        train_scaled=train_scaled,
        val_scaled=val_scaled,
        test_scaled=test_scaled,
        training_dir=training_dir,
        model_dir=model_dir,
        config=cfg,
        attachment=pair_attachment,
        logger=logger,
    )

    (model_dir / "training_summary.json").write_text(
        json.dumps({"voltage": voltage_result, "spower": spower_result}, indent=2),
        encoding="utf-8",
    )
    logger.info("All pair-aware GINE Optuna training flows completed.")
