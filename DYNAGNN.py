# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: Component dynamic activity prediction from steady-state operating points and events

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from modules import electric_distance, graph_construction, initialization
from modules.dynawo_runner import default_dynawo_execution_path
from modules.paths import CONFIG_PATH, DATA_DIR
from modules.pair_aware_inference import (
    load_pair_aware_checkpoint,
    load_pair_aware_model,
    predict_pair_aware,
)
from modules.pipeline_logging import build_pipeline_formatter


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("dynagnn_inference")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = build_pipeline_formatter()
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def _load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _run_step(
    logger: logging.Logger,
    step: str,
    fn: Callable[[], Any],
    *,
    on_success: Optional[Callable[[Any], Sequence[str]]] = None,
) -> Any:
    try:
        result = fn()
        if on_success:
            for detail in on_success(result):
                logger.info("%s — %s", step, detail)
        return result
    except Exception as exc:
        logger.error("%s — failed: %s", step, exc)
        raise


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

    raise KeyError(
        f"Component '{event_id}' not found in graph node/edge metadata "
        "(use the IIDM component id/name as in training datasets)"
    )


def _edge_cont_cols_from_metadata(metadata: dict) -> list[int]:
    edge_schema = list((metadata.get("edge_feature_schema", []) or []))
    if not edge_schema:
        raise RuntimeError("Missing edge_feature_schema on graph; cannot scale edges.")
    names = ("r", "x", "b1", "g1", "b2", "g2")
    missing = [n for n in names if n not in edge_schema]
    if missing:
        raise RuntimeError(f"edge_feature_schema missing {missing}")
    return [edge_schema.index(n) for n in names]


def _apply_scaling(data, *, x_scaler, edge_attr_scaler, node_cont_cols: Sequence[int], edge_cont_cols: Sequence[int]):
    d2 = data.clone()
    x_part = d2.x[:, list(node_cont_cols)].cpu().numpy()
    d2.x[:, list(node_cont_cols)] = torch.tensor(x_scaler.transform(x_part), dtype=d2.x.dtype)
    if d2.edge_attr.shape[0] > 0:
        edge_part = d2.edge_attr[:, list(edge_cont_cols)].cpu().numpy()
        d2.edge_attr[:, list(edge_cont_cols)] = torch.tensor(
            edge_attr_scaler.transform(edge_part),
            dtype=d2.edge_attr.dtype,
        )
    return d2


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


def _append_dz_fault(data_i, *, metadata: dict, dist_by_vli: dict, idx_to_bus_vl: list[str]) -> None:
    num_nodes = int(data_i.x.shape[0])
    loc_type = str(getattr(data_i, "event_location_type", "missing"))
    loc_idx = int(getattr(data_i, "event_location_index", -1))
    anchors = _event_anchor_nodes(data_i, loc_type, loc_idx)

    if not anchors:
        dz = np.zeros(num_nodes, dtype=np.float64)
    else:
        source_vls: list[str] = []
        for a in anchors:
            if 0 <= int(a) < num_nodes:
                vl = idx_to_bus_vl[int(a)]
                if vl:
                    source_vls.append(vl)
        source_vls = list(dict.fromkeys(source_vls))
        target_vls = sorted({vl for vl in idx_to_bus_vl if vl})
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
            dz = np.array([dz_by_vl.get(idx_to_bus_vl[i], 0.0) for i in range(num_nodes)], dtype=np.float64)

    dz_tensor = torch.tensor(dz, dtype=data_i.x.dtype, device=data_i.x.device)
    data_i.x = torch.cat([data_i.x, dz_tensor.unsqueeze(1)], dim=1)
    data_i.dz_fault = dz_tensor


def _allowed_countries(cfg: dict) -> set[str]:
    network_cfg = cfg.get("network", {}) or {}
    raw = network_cfg.get("country_filter", []) or []
    return {str(x).upper() for x in raw if str(x).strip()}


def _keep_country(country: str, allow: set[str]) -> bool:
    return True if not allow else str(country).upper() in allow


def _build_masks_and_names(metadata: dict, *, allow_countries: set[str]) -> tuple[torch.Tensor, list[str], torch.Tensor, list[str]]:
    node_meta = metadata.get("node_metadata", {}) or {}
    # Build index->meta list
    nodes = []
    for node_key, m in node_meta.items():
        idx = int(m["index"])
        ntype = str(m.get("type", "")).lower()
        nid = str(m.get("id", node_key)).strip()
        country = str(m.get("country", "")).upper()
        has_dyn = bool(m.get("hasDynamicModel", False))
        nodes.append((idx, ntype, nid, country, has_dyn))
    nodes.sort(key=lambda t: t[0])

    bus = [(idx, nid) for idx, ntype, nid, country, _ in nodes if ntype == "bus" and _keep_country(country, allow_countries)]
    gen = [
        (idx, nid)
        for idx, ntype, nid, country, has_dyn in nodes
        if ntype == "generator" and _keep_country(country, allow_countries) and has_dyn
    ]

    num_nodes = max((int(m["index"]) for m in node_meta.values()), default=-1) + 1
    bus_mask = torch.zeros(num_nodes, dtype=torch.bool)
    gen_mask = torch.zeros(num_nodes, dtype=torch.bool)
    if bus:
        bus_mask[torch.tensor([i for i, _ in bus], dtype=torch.long)] = True
    if gen:
        gen_mask[torch.tensor([i for i, _ in gen], dtype=torch.long)] = True
    return bus_mask, [n for _, n in bus], gen_mask, [n for _, n in gen]


def _read_scenarios_csv(path: Path) -> list[tuple[int, str]]:
    """Read scenarios CSV: scenario_id (int) + Event (IIDM component name/id for fault_on)."""
    df = pd.read_csv(path)
    col_map = {str(c).strip().lower(): c for c in df.columns}

    if "scenario_id" not in col_map:
        raise ValueError(f"{path} must contain a 'scenario_id' column")
    sid_col = col_map["scenario_id"]

    component_col = None
    for key in ("event", "component", "component_name", "contingency", "event_id"):
        if key in col_map:
            component_col = col_map[key]
            break
    if component_col is None:
        raise ValueError(
            f"{path} must contain an 'Event' column with the faulted component name "
            "(IIDM id, same as column 2 in Dataset_Voltage/Dataset_Spower)"
        )

    scenarios: list[tuple[int, str]] = []
    for _, row in df.iterrows():
        scenario_id = int(row[sid_col])
        component_name = str(row[component_col]).strip()
        if not component_name:
            continue
        scenarios.append((scenario_id, component_name))
    if not scenarios:
        raise ValueError(f"No scenarios found in {path}")
    return scenarios


def _inject_single_event_fault(
    sample,
    *,
    component_name: str,
    metadata: dict,
    event_lookup: dict,
) -> None:
    loc_type, loc_idx, loc_id = _find_event_location(component_name, metadata, event_lookup)
    sample.event_location_type = loc_type
    sample.event_location_index = int(loc_idx)
    sample.event_location_id = str(loc_id)

    node_schema = list((metadata.get("node_feature_schema", []) or []))
    edge_schema = list((metadata.get("edge_feature_schema", []) or []))
    if loc_type == "node":
        fault_col = node_schema.index("fault_on")
        sample.x[int(loc_idx), fault_col] = 1.0
    else:
        fault_col = edge_schema.index("fault_on")
        sample.edge_attr[int(loc_idx), fault_col] = 1.0
        reverse_idx = int(loc_idx) + 1
        if reverse_idx < int(sample.edge_attr.shape[0]):
            sample.edge_attr[reverse_idx, fault_col] = 1.0


def _build_scenario_sample(
    data_base,
    *,
    metadata: dict,
    event_lookup: dict,
    scenario_id: int,
    component_name: str,
    op_name: str,
    bus_mask_all: torch.Tensor,
    bus_names: list[str],
    gen_mask_all: torch.Tensor,
    gen_names: list[str],
    dist_by_vli: dict,
    idx_to_bus_vl: list[str],
    x_scaler,
    edge_attr_scaler,
    node_cont_cols: Sequence[int],
    edge_cont_cols: Sequence[int],
):
    sample = data_base.clone()
    sample.metadata = metadata
    sample.node_metadata = metadata.get("node_metadata", {}) or {}
    sample.edge_metadata = metadata.get("edge_metadata", []) or []
    sample.op_name = op_name
    sample.scenario_id = int(scenario_id)
    sample.event_id = str(component_name)

    _inject_single_event_fault(
        sample, component_name=component_name, metadata=metadata, event_lookup=event_lookup
    )

    sample.bus_node_mask = bus_mask_all.clone()
    sample.bus_node_names = list(bus_names)
    sample.gen_node_mask = gen_mask_all.clone()
    sample.gen_node_names = list(gen_names)

    _append_dz_fault(sample, metadata=metadata, dist_by_vli=dist_by_vli, idx_to_bus_vl=idx_to_bus_vl)
    return _apply_scaling(
        sample,
        x_scaler=x_scaler,
        edge_attr_scaler=edge_attr_scaler,
        node_cont_cols=node_cont_cols,
        edge_cont_cols=edge_cont_cols,
    )


def _run_scenario_inference(
    *,
    scenario_id: int,
    sample_cpu,
    scenario_dir: Path,
    model_v,
    model_s,
    device: torch.device,
    v_hp: dict,
    s_hp: dict,
    logger: logging.Logger,
) -> tuple[int, int]:
    """Run voltage and spower forward passes in parallel; write CSVs under scenario_dir."""

    def _predict_voltage() -> int:
        rows: list[dict] = []
        with torch.no_grad():
            pred_v = predict_pair_aware(
                model=model_v,
                sample_cpu=sample_cpu,
                checkpoint=v_hp,
                device=device,
            )
            names_v = list(sample_cpu.bus_node_names)
            for comp_name, yp in zip(names_v, pred_v.detach().cpu().numpy().astype(int).tolist(), strict=False):
                rows.append({"component_name": str(comp_name), "predicted_class": int(yp)})
        out_v = scenario_dir / "prediction_voltage.csv"
        pd.DataFrame(rows, columns=["component_name", "predicted_class"]).to_csv(out_v, index=False)
        logger.info("  scenario_%s: saved %s (rows=%d)", scenario_id, out_v.name, len(rows))
        return len(rows)

    def _predict_spower() -> int:
        rows: list[dict] = []
        with torch.no_grad():
            pred_s = predict_pair_aware(
                model=model_s,
                sample_cpu=sample_cpu,
                checkpoint=s_hp,
                device=device,
            )
            names_s = list(sample_cpu.gen_node_names)
            for comp_name, yp in zip(names_s, pred_s.detach().cpu().numpy().astype(int).tolist(), strict=False):
                rows.append({"component_name": str(comp_name), "predicted_class": int(yp)})
        out_s = scenario_dir / "prediction_spower.csv"
        pd.DataFrame(rows, columns=["component_name", "predicted_class"]).to_csv(out_s, index=False)
        logger.info("  scenario_%s: saved %s (rows=%d)", scenario_id, out_s.name, len(rows))
        return len(rows)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        fut_v = ex.submit(_predict_voltage)
        fut_s = ex.submit(_predict_spower)
        return fut_v.result(), fut_s.result()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DYNAGNN — component dynamic activity prediction from steady-state operating points and events."
    )
    parser.add_argument(
        "--case-dir",
        required=True,
        type=Path,
        help="Folder containing IIDM/DYD/jobs/etc for one operating point.",
    )
    parser.add_argument(
        "--events-csv",
        required=True,
        type=Path,
        help=(
            "CSV with scenario_id (int) and Event (IIDM component name/id to fault on the graph); "
            "one row per scenario."
        ),
    )
    args = parser.parse_args()

    logger = _setup_logger()
    cfg = _load_config()
    device = _device()

    model_cfg = cfg.get("model", {}) or {}
    if "num_classes" not in model_cfg:
        raise KeyError("Missing required config key: model.num_classes (in config.yaml)")
    num_classes = int(model_cfg["num_classes"])
    if num_classes < 2:
        raise ValueError(f"model.num_classes must be >= 2, got {num_classes}")

    # Determine initialization duration
    infer_cfg = cfg.get("inference", {}) or {}
    init_duration = infer_cfg.get("initialization_duration", 0)
    init_duration = float(init_duration)

    case_dir = Path(args.case_dir)
    events_csv = Path(args.events_csv)
    if not case_dir.exists():
        raise FileNotFoundError(f"Missing case-dir: {case_dir}")
    if not events_csv.exists():
        raise FileNotFoundError(f"Missing events CSV: {events_csv}")

    out_dir = case_dir / "dynagnn_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Starting prediction: case_dir=%s, events_csv=%s, device=%s",
        case_dir.name,
        events_csv.name,
        device,
    )

    # 1) Optional Dynawo initialization
    if init_duration is not None and init_duration > 0:
        logger.info("Running Dynawo initialization for %.3fs in %s", init_duration, case_dir)
        exec_path = default_dynawo_execution_path(case_dir, CONFIG_PATH)
        case = initialization.discover_case(case_dir)
        init_result = initialization.initialize_one_operating_point(
            case,
            execution_path=str(exec_path),
            run_time=float(init_duration),
            log_file=Path(os.devnull),
            backup_iidm=False,
            clean_outputs=False,
        )
        if init_result.success:
            logger.info(
                "initialization — duration=%gs, IIDM updated: %s",
                init_result.run_time,
                case.iidm_path.name,
            )
        else:
            for message in init_result.messages:
                logger.error("Initialization failed: %s", message)
            raise RuntimeError(f"Operating point initialization failed for {case_dir.name}")
    else:
        logger.info("Initialization skipped (inference.initialization_duration not set or <= 0).")

    # 2) Electrical distance CSV
    iidm_path = graph_construction._resolve_iidm_path(case_dir)
    electrical_csv = out_dir / "electrical_distance.csv"
    logger.info("Computing electrical distances from %s", iidm_path)

    def _compute_electrical_distance() -> int:
        return electric_distance.write_electrical_distance_csv_from_iidm(iidm_path, electrical_csv)

    n_dist_rows = _run_step(
        logger,
        "electrical_distance",
        _compute_electrical_distance,
        on_success=lambda n: [f"Wrote {electrical_csv.name} ({n} rows) from {iidm_path.name}"],
    )
    logger.info("Wrote %s (%d rows)", electrical_csv, n_dist_rows)

    # 3) Build base graph (PyG)
    def _build_base_graph():
        return graph_construction.build_graph(case_dir, compact=True)

    data_base, metadata = _run_step(
        logger,
        "graph_construction",
        _build_base_graph,
        on_success=lambda pair: [
            f"nodes={int(pair[0].x.shape[0])}, edges={int(pair[0].edge_attr.shape[0])}",
        ],
    )
    allow_countries = _allowed_countries(cfg)
    bus_mask_all, bus_names, gen_mask_all, gen_names = _build_masks_and_names(metadata, allow_countries=allow_countries)
    logger.info(
        "inference_masks — country_filter=%s, bus_nodes=%d, generator_nodes=%d",
        sorted(allow_countries) if allow_countries else "all",
        int(bus_mask_all.sum().item()),
        int(gen_mask_all.sum().item()),
    )

    # 4) Load scalers and models
    model_dir = DATA_DIR / "model"

    def _load_scalers():
        return (
            joblib.load(model_dir / "x_scaler.pkl"),
            joblib.load(model_dir / "edge_attr_scaler.pkl"),
        )

    x_scaler, edge_attr_scaler = _run_step(
        logger,
        "load_scalers",
        _load_scalers,
        on_success=lambda _: [f"Loaded from {model_dir}"],
    )
    node_cont_cols = [1, 2, 3, 4, 6]  # matches training (includes dz_fault)
    edge_cont_cols = _edge_cont_cols_from_metadata(metadata)

    v_weights = model_dir / "voltage_best_model.pt"
    s_weights = model_dir / "spower_best_model.pt"
    v_hp = load_pair_aware_checkpoint(v_weights, expected_task="voltage")
    s_hp = load_pair_aware_checkpoint(s_weights, expected_task="spower")

    def _load_models():
        return (
            load_pair_aware_model(v_hp, device),
            load_pair_aware_model(s_hp, device),
        )

    model_v, model_s = _run_step(
        logger,
        "load_models",
        _load_models,
        on_success=lambda _: [
            f"voltage={v_weights.name}, spower={s_weights.name}, num_classes={num_classes}"
        ],
    )

    # 5) Prepare electrical-distance lookup for dz_fault
    dist_by_vli = _load_op_electric_distances(electrical_csv)
    idx_to_bus_vl = _node_index_to_bus_vl_id(metadata)

    # 6) Read scenarios (one row per event)
    scenarios = _read_scenarios_csv(events_csv)
    logger.info("read_scenarios — loaded %d scenario(s) from %s", len(scenarios), events_csv.name)
    op_name = case_dir.name
    event_lookup = _build_event_lookup(metadata)

    logger.info("Running inference for %d scenarios", len(scenarios))
    for scenario_id, component_name in scenarios:
        logger.info("Scenario %s | component=%s", scenario_id, component_name)
        scenario_dir = out_dir / f"scenario_{scenario_id}"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        try:
            sample_cpu = _build_scenario_sample(
                data_base,
                metadata=metadata,
                event_lookup=event_lookup,
                scenario_id=scenario_id,
                component_name=component_name,
                op_name=op_name,
                bus_mask_all=bus_mask_all,
                bus_names=bus_names,
                gen_mask_all=gen_mask_all,
                gen_names=gen_names,
                dist_by_vli=dist_by_vli,
                idx_to_bus_vl=idx_to_bus_vl,
                x_scaler=x_scaler,
                edge_attr_scaler=edge_attr_scaler,
                node_cont_cols=node_cont_cols,
                edge_cont_cols=edge_cont_cols,
            )

            _run_scenario_inference(
                scenario_id=scenario_id,
                sample_cpu=sample_cpu,
                scenario_dir=scenario_dir,
                model_v=model_v,
                model_s=model_s,
                device=device,
                v_hp=v_hp,
                s_hp=s_hp,
                logger=logger,
            )
        except Exception as exc:
            logger.error("scenario_%s — failed (component=%s): %s", scenario_id, component_name, exc)
            raise

    logger.info("Prediction complete. Outputs under %s", out_dir)


if __name__ == "__main__":
    main()

