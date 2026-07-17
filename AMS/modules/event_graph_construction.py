# Build per-event inference graphs from a base graph and electrical distances.

from __future__ import annotations

import re
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

NODE_CONT_COLS = [1, 2, 3, 4, 6]
EDGE_CONT_FEATURE_NAMES = ("r", "x", "b1", "g1", "b2", "g2")


def _canonical_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(value).strip()).upper()


def _event_id_candidates(event_id: str) -> List[str]:
    raw = str(event_id).strip()
    cands = [raw]
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'"}:
        cands.append(raw[1:-1])
    if "." in raw:
        parts = [p for p in raw.split(".") if p]
        if parts:
            cands.append(parts[-1])
            cands.append("".join(parts))
    out: List[str] = []
    seen: set[str] = set()
    for cand in cands:
        text = str(cand).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def build_event_lookup(metadata: dict) -> Dict[str, dict]:
    exact: Dict[str, Tuple[str, int, str]] = {}
    canonical: Dict[str, List[Tuple[str, int, str]]] = {}

    def _register(identifier: str, loc_type: str, idx: int) -> None:
        sid = str(identifier).strip()
        if not sid:
            return
        payload = (loc_type, int(idx), sid)
        if sid not in exact:
            exact[sid] = payload
        cid = _canonical_id(sid)
        canonical.setdefault(cid, []).append(payload)

    for node_key, node_meta in metadata.get("node_metadata", {}).items():
        node_idx = int(node_meta.get("index", node_key))
        node_id = str(node_meta.get("id", node_key)).strip()
        _register(node_id, "node", node_idx)
        for bus_id in node_meta.get("busbarSectionIds", []) or []:
            _register(str(bus_id), "node", node_idx)

    for edge_idx, edge_meta in enumerate(metadata.get("edge_metadata", [])):
        edge_id = str(edge_meta.get("id", "")).strip()
        _register(edge_id, "edge", edge_idx)

    return {"exact": exact, "canonical": canonical}


def find_event_location(event_id: str, event_lookup: dict) -> Tuple[str, int, str]:
    candidates = _event_id_candidates(event_id)
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
            raise KeyError(
                f"Event '{event_id}' matched multiple locations for canonical id '{cid}'"
            )

    for cand in candidates:
        cc = _canonical_id(cand)
        if not cc:
            continue
        contains_hits = []
        for key, vals in event_lookup["canonical"].items():
            if cc in key or key in cc:
                contains_hits.extend(vals)
        uniq = {(t, i, s) for t, i, s in contains_hits}
        if len(uniq) == 1:
            return next(iter(uniq))

    raise KeyError(f"Event '{event_id}' not found in graph metadata")


def _node_index_to_voltage_level_id(metadata: dict) -> List[str]:
    num_nodes = 0
    for node_meta in metadata.get("node_metadata", {}).values():
        num_nodes = max(num_nodes, int(node_meta["index"]) + 1)
    idx_to_vl = [""] * int(num_nodes)
    for node_meta in metadata.get("node_metadata", {}).values():
        idx = int(node_meta["index"])
        idx_to_vl[idx] = str(node_meta.get("voltageLevelId", "")).strip()
    return idx_to_vl


def _distance_lookup_from_table(electric_distance_table: pd.DataFrame) -> Dict[str, pd.Series]:
    required = {"VLi", "VLj", "dij"}
    missing = required.difference(electric_distance_table.columns)
    if missing:
        raise ValueError(f"Electric distance table missing columns: {sorted(missing)}")
    return {
        str(vli): grp.set_index("VLj")["dij"]
        for vli, grp in electric_distance_table.groupby("VLi", sort=False)
    }


def _event_anchor_nodes(data: Data, loc_type: str, loc_idx: int) -> List[int]:
    num_nodes = int(data.x.shape[0])
    if loc_type == "node":
        idx = int(loc_idx)
        return [idx] if 0 <= idx < num_nodes else []
    if loc_type == "edge":
        if data.edge_index.numel() == 0:
            return []
        eidx = int(loc_idx)
        if eidx < 0 or eidx >= int(data.edge_index.shape[1]):
            return []
        source = int(data.edge_index[0, eidx].item())
        target = int(data.edge_index[1, eidx].item())
        out: List[int] = []
        if 0 <= source < num_nodes:
            out.append(source)
        if 0 <= target < num_nodes and target != source:
            out.append(target)
        return out
    return []


def _distances_from_precomputed(
    dist_by_vli: Dict[str, pd.Series],
    source_vls: List[str],
    target_vls: List[str],
) -> np.ndarray:
    out = np.full(len(target_vls), np.inf, dtype=np.float64)
    for vli in source_vls:
        series = dist_by_vli.get(vli)
        if series is None:
            continue
        distances = series.reindex(target_vls).to_numpy(dtype=np.float64, copy=False)
        out = np.minimum(out, distances)
    return out


def _append_dz_fault(
    data: Data,
    metadata: dict,
    dist_by_vli: Dict[str, pd.Series],
    loc_type: str,
    loc_idx: int,
) -> None:
    num_nodes = int(data.x.shape[0])
    if num_nodes <= 0:
        return

    idx_to_vl = _node_index_to_voltage_level_id(metadata)
    anchors = _event_anchor_nodes(data, loc_type, loc_idx)
    if not anchors:
        dz = np.zeros(num_nodes, dtype=np.float64)
    else:
        source_vls = []
        for anchor in anchors:
            if 0 <= anchor < num_nodes:
                vl = idx_to_vl[anchor]
                if vl:
                    source_vls.append(vl)
        source_vls = list(dict.fromkeys(source_vls))
        target_vls = sorted({vl for vl in idx_to_vl if vl})
        if not source_vls or not target_vls:
            dz = np.zeros(num_nodes, dtype=np.float64)
        else:
            min_dist = _distances_from_precomputed(dist_by_vli, source_vls, target_vls)
            finite = np.isfinite(min_dist)
            if finite.any():
                max_finite = float(min_dist[finite].max())
                min_dist[~finite] = max_finite + 1.0
                dz_by_vl = {vl: float(np.log1p(d)) for vl, d in zip(target_vls, min_dist)}
            else:
                dz_by_vl = {}
            dz = np.array(
                [dz_by_vl.get(idx_to_vl[i], 0.0) for i in range(num_nodes)],
                dtype=np.float64,
            )

    dz_tensor = torch.tensor(dz, dtype=data.x.dtype, device=data.x.device)
    data.x = torch.cat([data.x, dz_tensor.unsqueeze(1)], dim=1)
    data.dz_fault = dz_tensor


def _set_fault_flag(data: Data, metadata: dict, loc_type: str, loc_idx: int) -> None:
    if loc_type == "node":
        node_schema = metadata.get("node_feature_schema", [])
        fault_col = node_schema.index("fault_on")
        data.x[loc_idx, fault_col] = 1.0
    elif loc_type == "edge":
        edge_schema = metadata.get("edge_feature_schema", [])
        fault_col = edge_schema.index("fault_on")
        data.edge_attr[loc_idx, fault_col] = 1.0
        reverse_idx = loc_idx + 1
        if reverse_idx < data.edge_attr.shape[0]:
            data.edge_attr[reverse_idx, fault_col] = 1.0


def _attach_inference_masks(data: Data, metadata: dict) -> None:
    num_nodes = int(data.x.shape[0])
    bus_mask = torch.zeros(num_nodes, dtype=torch.bool)
    gen_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for node_meta in metadata.get("node_metadata", {}).values():
        idx = int(node_meta["index"])
        if idx < 0 or idx >= num_nodes:
            continue
        node_type = str(node_meta.get("type", "")).lower()
        country = str(node_meta.get("country", "")).upper()
        if country != "FR":
            continue
        if node_type == "bus":
            bus_mask[idx] = True
        elif node_type == "generator" and bool(node_meta.get("hasDynamicModel", False)):
            gen_mask[idx] = True

    data.bus_node_mask = bus_mask
    data.gen_node_mask = gen_mask


def build_event_graphs(
    base_data: Data,
    metadata: dict,
    events_list: List[str],
    electric_distance_table: pd.DataFrame,
) -> List[Data]:
    """Clone the base graph for each event, set fault flags, and append log1p(dZ_fault)."""
    if not events_list:
        return []

    event_lookup = build_event_lookup(metadata)
    dist_by_vli = _distance_lookup_from_table(electric_distance_table)
    graphs: List[Data] = []

    for event_id in events_list:
        data = base_data.clone()
        meta = deepcopy(metadata)
        data.metadata = meta
        data.node_metadata = meta.get("node_metadata", {})
        data.edge_metadata = meta.get("edge_metadata", [])

        loc_type, loc_idx, loc_id = find_event_location(event_id, event_lookup)
        _set_fault_flag(data, meta, loc_type, loc_idx)
        _append_dz_fault(data, meta, dist_by_vli, loc_type, loc_idx)
        _attach_inference_masks(data, meta)

        data.event_id = str(event_id)
        data.event_location_type = loc_type
        data.event_location_index = int(loc_idx)
        data.event_location_id = str(loc_id)
        graphs.append(data)

    return graphs
