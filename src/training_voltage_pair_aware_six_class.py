#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Train a direct pair-aware six-class Voltage GNN.

The model learns classes 0..5 directly. Class 5 is the disconnected/controlled
class and is included in the classification loss and test predictions. No
historical KPI/class prior and no deterministic class-5 override are used.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from training import (  # noqa: E402
    _edge_cont_cols_from_metadata,
    _scale_split,
    append_electrical_distance_feature,
    build_graph_dataset_multi,
)
from modules.gat_voltage_pair_aware_six_class import (  # noqa: E402
    ACTIVITY_CLASSES,
    NUM_CLASSES,
    PairAwareHParams,
    PairAwareLossWeights,
    run_pair_aware_training,
)
from modules.op_context_aggregate import (  # noqa: E402
    OPContextTransformer,
    normalize_op,
    normalize_ops,
)

DEFAULT_CUTS = np.asarray([1.0e-6, 2.25e-5, 3.0e-4, 5.625e-4], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct pair-aware six-class Voltage GNN")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, default=None)
    parser.add_argument("--train-ops", nargs="+", required=True)
    parser.add_argument("--validation-ops", nargs="*", default=[])
    parser.add_argument("--test-ops", nargs="*", default=[])
    parser.add_argument("--scenario-name", default="Pair-aware direct six-class Voltage GNN")
    parser.add_argument("--op-context-mode", choices=["none", "aggregate"], default="aggregate")

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--fixed-epochs", type=int, default=None)
    parser.add_argument("--selection-output", choices=["class", "gated", "log_kpi"], default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--node-id-dim", type=int, default=24)
    parser.add_argument("--contingency-id-dim", type=int, default=32)
    parser.add_argument("--type-dim", type=int, default=8)
    parser.add_argument("--pair-dim", type=int, default=32)
    parser.add_argument("--op-context-embedding-dim", type=int, default=32)
    parser.add_argument("--gnn-layers", type=int, default=3)
    parser.add_argument("--decoder-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)

    parser.add_argument("--classification-weight", type=float, default=1.0)
    parser.add_argument("--regression-weight", type=float, default=0.30)
    parser.add_argument("--inactive-gate-weight", type=float, default=0.20)
    parser.add_argument("--ordinal-weight", type=float, default=0.10)
    parser.add_argument("--class-weight-mode", choices=["none", "sqrt_inverse"], default="sqrt_inverse")
    parser.add_argument("--gate-pos-weight-mode", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--gate-threshold", type=float, default=0.50)
    parser.add_argument("--epsilon", type=float, default=1.0e-10)
    parser.add_argument("--cuts", type=float, nargs=4, default=DEFAULT_CUTS.tolist())
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def configure_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("dynagnn_pair_aware_six_class")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(output_dir / "training.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def first_existing(paths: Sequence[Path], label: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing {label}. Tried: {[str(p) for p in paths]}")


def resolve_paths(data_path: Path, graph_dir: Path | None) -> dict[str, Path]:
    data_path = data_path.expanduser().resolve()
    graphs = graph_dir.expanduser().resolve() if graph_dir is not None else data_path / "op_graphs"
    return {
        "voltage": first_existing(
            [data_path / "Dataset" / "Dataset_Voltage.csv", data_path / "Dataset_Voltage.csv"],
            "Dataset_Voltage.csv",
        ),
        "spower": first_existing(
            [data_path / "Dataset" / "Dataset_Spower.csv", data_path / "Dataset_Spower.csv"],
            "Dataset_Spower.csv",
        ),
        "kpi_voltage": first_existing(
            [data_path / "KPI" / "KPI_voltage.csv", data_path / "KPI_voltage.csv"],
            "KPI_voltage.csv",
        ),
        "disc_voltage": first_existing(
            [
                data_path / "Disconnections" / "DISC_voltage.csv",
                data_path / "DISC" / "DISC_voltage.csv",
                data_path / "Dataset" / "DISC_voltage.csv",
                data_path / "DISC_voltage.csv",
            ],
            "DISC_voltage.csv",
        ),
        "graphs": first_existing([graphs], "op_graphs"),
        "electric": first_existing([data_path / "op_electric_distance"], "op_electric_distance"),
    }


def load_config(path: Path) -> dict:
    with path.expanduser().open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_output(path: Path, overwrite: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}. Use --overwrite.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _op_column(frame: pd.DataFrame) -> str:
    for candidate in ("OP", "operating_point", "OperatingPoint"):
        if candidate in frame.columns:
            return candidate
    return str(frame.columns[0])


def filter_frame(frame: pd.DataFrame, allowed_ops: set[str]) -> pd.DataFrame:
    column = _op_column(frame)
    normalized = frame[column].astype("string").map(normalize_op)
    out = frame.loc[normalized.isin(allowed_ops)].copy()
    out[column] = normalized.loc[out.index]
    return out.reset_index(drop=True)


def _node_id(meta_key: str, meta: dict) -> str:
    return str(meta.get("id", meta_key)).strip()


def attach_pair_aware_tensors(
    graph_dataset: list,
    *,
    kpi_voltage: pd.DataFrame,
    disc_voltage: pd.DataFrame,
    epsilon: float,
) -> dict:
    """Attach component/contingency IDs, direct log-KPI targets, and event anchors."""
    kpi = kpi_voltage.copy()
    kpi.columns = [str(col).strip() for col in kpi.columns]
    for col in ("OP", "Contingency"):
        if col not in kpi.columns:
            raise KeyError(f"KPI_voltage.csv is missing {col}")
        kpi[col] = kpi[col].astype("string").str.strip()
    kpi["OP"] = kpi["OP"].map(normalize_op)
    kpi_lookup = {
        (str(row["OP"]).strip(), str(row["Contingency"]).strip()): row
        for _, row in kpi.iterrows()
    }

    disc = disc_voltage.copy()
    disc.columns = [str(col).strip() for col in disc.columns]
    for col in ("OP", "Contingency"):
        if col not in disc.columns:
            raise KeyError(f"DISC_voltage.csv is missing {col}")
        disc[col] = disc[col].astype("string").str.strip()
    disc["OP"] = disc["OP"].map(normalize_op)
    disc_lookup = {
        (str(row["OP"]).strip(), str(row["Contingency"]).strip()): row
        for _, row in disc.iterrows()
    }

    # IDs are input metadata, not labels. Build the vocabulary over the allowed network samples
    # so validation/test IDs are represented rather than collapsed to an unknown token.
    node_ids: set[str] = set()
    contingencies: set[str] = set()
    for data in graph_dataset:
        contingencies.add(str(data.event_id).strip())
        for key, meta in (getattr(data, "metadata", {}) or {}).get("node_metadata", {}).items():
            node_ids.add(_node_id(str(key), meta))
    node_vocab = {node_id: idx + 1 for idx, node_id in enumerate(sorted(node_ids))}
    contingency_vocab = {event: idx + 1 for idx, event in enumerate(sorted(contingencies))}

    stats = {
        "graphs": len(graph_dataset),
        "finite_kpi_targets": 0,
        "missing_kpi_targets": 0,
        "structural_class5_targets": 0,
        "class5_label_mask_mismatches": 0,
        "node_event_graphs": 0,
        "edge_event_graphs": 0,
        "node_vocab_size": len(node_vocab) + 1,
        "contingency_vocab_size": len(contingency_vocab) + 1,
    }

    for data in graph_dataset:
        data.op_name = normalize_op(data.op_name)
        event_id = str(data.event_id).strip()
        row = kpi_lookup.get((data.op_name, event_id))
        disc_row = disc_lookup.get((data.op_name, event_id))
        if row is None:
            raise KeyError(f"Missing KPI row for {(data.op_name, event_id)}")
        if disc_row is None:
            raise KeyError(f"Missing DISC row for {(data.op_name, event_id)}")

        n_nodes = int(data.x.shape[0])
        n_edges = int(data.edge_attr.shape[0])
        node_token = torch.zeros(n_nodes, dtype=torch.long)
        y_log = torch.full((n_nodes,), float("nan"), dtype=torch.float32)
        event_node_mask = torch.zeros(n_nodes, dtype=torch.bool)
        event_edge_mask = torch.zeros(n_edges, dtype=torch.bool)
        structural_class5_mask = torch.zeros(n_nodes, dtype=torch.bool)

        metadata = getattr(data, "metadata", {}) or {}
        for key, meta in (metadata.get("node_metadata", {}) or {}).items():
            idx = int(meta["index"])
            node_id = _node_id(str(key), meta)
            node_token[idx] = int(node_vocab.get(node_id, 0))
            if str(meta.get("type", "")).lower() == "bus":
                if node_id in kpi.columns:
                    value = pd.to_numeric(pd.Series([row[node_id]]), errors="coerce").iloc[0]
                    if pd.notna(value) and np.isfinite(float(value)):
                        y_log[idx] = float(np.log10(max(float(value), 0.0) + float(epsilon)))
                        stats["finite_kpi_targets"] += 1
                    else:
                        stats["missing_kpi_targets"] += 1
                if node_id in disc.columns:
                    disc_value = pd.to_numeric(pd.Series([disc_row[node_id]]), errors="coerce").iloc[0]
                    structural_class5_mask[idx] = bool(pd.notna(disc_value) and float(disc_value) > 0.5)

        y_bus = data.y_class[data.bus_node_mask]
        structural_bus = structural_class5_mask[data.bus_node_mask]
        stats["structural_class5_targets"] += int(structural_bus.sum().item())
        stats["class5_label_mask_mismatches"] += int(((y_bus == 5) != structural_bus).sum().item())

        location_type = str(data.event_location_type)
        location_index = int(data.event_location_index)
        if location_type == "node":
            if not (0 <= location_index < n_nodes):
                raise IndexError(f"Bad node event index {location_index}")
            event_node_mask[location_index] = True
            event_graph_type = 0
            stats["node_event_graphs"] += 1
        elif location_type == "edge":
            edge_schema = list(metadata.get("edge_feature_schema", []) or [])
            if "fault_on" not in edge_schema:
                raise KeyError("fault_on missing from edge_feature_schema")
            fault_col = edge_schema.index("fault_on")
            event_edge_mask = data.edge_attr[:, fault_col] > 0.5
            if not bool(event_edge_mask.any()):
                if not (0 <= location_index < n_edges):
                    raise IndexError(f"Bad edge event index {location_index}")
                event_edge_mask[location_index] = True
            endpoints = torch.unique(data.edge_index[:, event_edge_mask].reshape(-1))
            event_node_mask[endpoints] = True
            event_graph_type = 1
            stats["edge_event_graphs"] += 1
        else:
            raise ValueError(f"Unsupported event location type: {location_type}")

        data.node_token = node_token
        data.contingency_token = torch.tensor([int(contingency_vocab.get(event_id, 0))], dtype=torch.long)
        data.y_log_kpi = y_log
        data.event_node_mask = event_node_mask
        data.event_edge_mask = event_edge_mask
        data.event_graph_type = torch.tensor([event_graph_type], dtype=torch.long)
        data.structural_class5_mask = structural_class5_mask

    return {**stats, "node_vocab": node_vocab, "contingency_vocab": contingency_vocab}


def split_graphs(graphs: list, train_ops: set[str], validation_ops: set[str], test_ops: set[str]):
    train: list = []
    validation: list = []
    test: list = []
    unexpected: set[str] = set()
    for data in graphs:
        op = normalize_op(data.op_name)
        data.op_name = op
        data.y_class = data.y_voltage
        if op in train_ops:
            train.append(data)
        elif op in validation_ops:
            validation.append(data)
        elif op in test_ops:
            test.append(data)
        else:
            unexpected.add(op)
    if unexpected:
        raise RuntimeError(f"Unexpected operating points in graph dataset: {sorted(unexpected)}")
    if not train:
        raise RuntimeError("Training graph split is empty")
    if validation_ops and not validation:
        raise RuntimeError("Validation graph split is empty")
    if test_ops and not test:
        raise RuntimeError("Test graph split is empty")
    return train, validation, test


def standardize_log_targets(train_graphs: list, all_splits: Sequence[list]) -> tuple[float, float]:
    values: list[np.ndarray] = []
    for data in train_graphs:
        mask = data.bus_node_mask.bool() & (data.y_class >= 0) & (data.y_class < ACTIVITY_CLASSES)
        target = data.y_log_kpi[mask]
        finite = torch.isfinite(target)
        if bool(finite.any()):
            values.append(target[finite].cpu().numpy())
    if not values:
        raise RuntimeError("No finite training log-KPI targets")
    merged = np.concatenate(values)
    mean = float(merged.mean())
    std = float(merged.std())
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 1.0e-12:
        raise RuntimeError(f"Invalid training log-KPI statistics mean={mean} std={std}")

    for split in all_splits:
        for data in split:
            standardized = torch.full_like(data.y_log_kpi, float("nan"))
            finite = torch.isfinite(data.y_log_kpi)
            standardized[finite] = (data.y_log_kpi[finite] - mean) / std
            data.y_log_kpi_std = standardized
    return mean, std


def attach_context(graphs: Iterable, context_map: dict[str, np.ndarray]) -> None:
    for data in graphs:
        op = normalize_op(data.op_name)
        if op not in context_map:
            raise KeyError(f"No OP context for {op}")
        data.op_context = torch.tensor(context_map[op], dtype=torch.float32).unsqueeze(0)


def main() -> None:
    args = parse_args()
    output_dir = prepare_output(args.output_dir, args.overwrite)
    logger = configure_logger(output_dir)
    config_path = args.config.expanduser().resolve()
    cfg = load_config(config_path)
    paths = resolve_paths(args.data_path, args.graph_dir)

    train_ops = normalize_ops(args.train_ops)
    validation_ops = normalize_ops(args.validation_ops)
    test_ops = normalize_ops(args.test_ops)
    groups = [set(train_ops), set(validation_ops), set(test_ops)]
    for i in range(3):
        for j in range(i + 1, 3):
            overlap = groups[i].intersection(groups[j])
            if overlap:
                raise ValueError(f"Operating-point split leakage: {sorted(overlap)}")
    allowed_ops = set(train_ops) | set(validation_ops) | set(test_ops)

    run_plan = {
        "scenario_name": args.scenario_name,
        "method": "direct pair-aware GINE with explicit component and contingency identities",
        "op_context_mode": args.op_context_mode,
        "train_ops": train_ops,
        "validation_ops": validation_ops,
        "test_ops": test_ops,
        "no_historical_kpi_prior": True,
        "num_classes": NUM_CLASSES,
        "class5_handling": "learned direct prediction; no deterministic override",
        "paths": {key: str(value) for key, value in paths.items()},
    }
    (output_dir / "run_plan.json").write_text(json.dumps(run_plan, indent=2), encoding="utf-8")
    shutil.copy2(config_path, output_dir / "source_config.yaml")

    logger.info("Scenario: %s", args.scenario_name)
    logger.info("Train OPs: %s", train_ops)
    logger.info("Validation OPs: %s", validation_ops)
    logger.info("Test OPs: %s", test_ops)
    logger.info("OP context mode: %s", args.op_context_mode)
    logger.info("Method: graph + target identity + contingency identity/location + learned six-class/log-KPI heads")
    logger.info("Historical KPI/class prior: disabled")
    logger.info("Class 5: learned by the model; deterministic override disabled")
    if args.dry_run:
        logger.info("Dry run completed")
        return

    set_seed(args.seed)
    voltage = filter_frame(pd.read_csv(paths["voltage"], low_memory=False), allowed_ops)
    spower = filter_frame(pd.read_csv(paths["spower"], low_memory=False), allowed_ops)
    kpi_voltage = filter_frame(pd.read_csv(paths["kpi_voltage"], low_memory=False), allowed_ops)
    disc_voltage = filter_frame(pd.read_csv(paths["disc_voltage"], low_memory=False), allowed_ops)
    country_filter = (cfg.get("network", {}) or {}).get("country_filter", []) or []

    graphs, skipped = build_graph_dataset_multi(
        dataset_voltage=voltage,
        dataset_spower=spower,
        graph_dir=paths["graphs"],
        num_classes=6,
        country_filter=country_filter,
        logger=logger,
    )
    logger.info("Built %d shared graphs; skipped=%d", len(graphs), len(skipped))
    if skipped:
        logger.info("Skipped preview: %s", skipped[:5])

    # Expose Voltage labels before tensor attachment so class-5 counts are available.
    for data in graphs:
        data.y_class = data.y_voltage
    append_electrical_distance_feature(
        graphs,
        graph_dir=paths["graphs"],
        electric_distance_dir=paths["electric"],
        logger=logger,
    )
    attachment = attach_pair_aware_tensors(
        graphs, kpi_voltage=kpi_voltage, disc_voltage=disc_voltage, epsilon=args.epsilon
    )
    logger.info("Pair-aware target attachment: %s", {k: v for k, v in attachment.items() if not isinstance(v, dict)})
    if int(attachment["class5_label_mask_mismatches"]) != 0:
        raise RuntimeError(
            "DISC_voltage structural mask does not match class-5 labels: "
            f"{attachment['class5_label_mask_mismatches']} mismatches"
        )

    train_raw, validation_raw, test_raw = split_graphs(
        graphs, set(train_ops), set(validation_ops), set(test_ops)
    )
    logger.info(
        "Graph splits: train=%d validation=%d test=%d",
        len(train_raw), len(validation_raw), len(test_raw),
    )

    node_cont_cols = [idx for idx in [1, 2, 3, 4, 6] if idx < int(train_raw[0].x.shape[1])]
    x_scaler = StandardScaler()
    x_scaler.fit(np.vstack([data.x[:, node_cont_cols].cpu().numpy() for data in train_raw]))
    edge_cont_cols = _edge_cont_cols_from_metadata(train_raw[0])
    edge_rows = [
        data.edge_attr[:, edge_cont_cols].cpu().numpy()
        for data in train_raw
        if int(data.edge_attr.shape[0]) > 0
    ]
    if not edge_rows:
        raise RuntimeError("No edges in training split")
    edge_scaler = StandardScaler()
    edge_scaler.fit(np.vstack(edge_rows))

    scaler_dir = output_dir / "scalers"
    scaler_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(x_scaler, scaler_dir / "x_scaler.pkl")
    joblib.dump(edge_scaler, scaler_dir / "edge_attr_scaler.pkl")

    train_scaled = _scale_split(
        train_raw,
        x_scaler=x_scaler,
        edge_attr_scaler=edge_scaler,
        node_cont_cols=node_cont_cols,
        edge_cont_cols=edge_cont_cols,
    )
    validation_scaled = _scale_split(
        validation_raw,
        x_scaler=x_scaler,
        edge_attr_scaler=edge_scaler,
        node_cont_cols=node_cont_cols,
        edge_cont_cols=edge_cont_cols,
    ) if validation_raw else []
    test_scaled = _scale_split(
        test_raw,
        x_scaler=x_scaler,
        edge_attr_scaler=edge_scaler,
        node_cont_cols=node_cont_cols,
        edge_cont_cols=edge_cont_cols,
    ) if test_raw else []

    log_mean, log_std = standardize_log_targets(
        train_scaled, [train_scaled, validation_scaled, test_scaled]
    )
    logger.info("Training log-KPI mean=%.6f std=%.6f", log_mean, log_std)

    use_op_context = args.op_context_mode == "aggregate"
    op_context_dim = 0
    if use_op_context:
        transformer = OPContextTransformer.fit(paths["graphs"], train_ops)
        transformer.save(output_dir / "op_context")
        context_map = transformer.transform_map(paths["graphs"], sorted(allowed_ops))
        attach_context(train_scaled + validation_scaled + test_scaled, context_map)
        op_context_dim = transformer.output_dim
        logger.info("Stable aggregate OP context dimension: %d", op_context_dim)

    train_loader = DataLoader(train_scaled, batch_size=args.batch_size, shuffle=True)
    validation_loader = (
        DataLoader(validation_scaled, batch_size=args.batch_size, shuffle=False)
        if validation_scaled else None
    )
    test_loader = (
        DataLoader(test_scaled, batch_size=args.batch_size, shuffle=False)
        if test_scaled else None
    )

    hparams = PairAwareHParams(
        hidden_dim=args.hidden_dim,
        node_id_dim=args.node_id_dim,
        contingency_id_dim=args.contingency_id_dim,
        type_dim=args.type_dim,
        pair_dim=args.pair_dim,
        op_context_embedding_dim=args.op_context_embedding_dim,
        num_gnn_layers=args.gnn_layers,
        decoder_hidden_dim=args.decoder_hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_weights = PairAwareLossWeights(
        classification=args.classification_weight,
        regression=args.regression_weight,
        inactive_gate=args.inactive_gate_weight,
        ordinal_cdf=args.ordinal_weight,
    )
    result = run_pair_aware_training(
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        output_dir=output_dir,
        num_node_tokens=int(attachment["node_vocab_size"]),
        num_contingency_tokens=int(attachment["contingency_vocab_size"]),
        op_context_dim=op_context_dim,
        use_op_context=use_op_context,
        hparams=hparams,
        loss_weights=loss_weights,
        log_mean=log_mean,
        log_std=log_std,
        cuts=np.asarray(args.cuts, dtype=np.float64),
        epsilon=args.epsilon,
        gate_threshold=args.gate_threshold,
        epochs=args.epochs,
        patience=args.patience,
        fixed_epochs=args.fixed_epochs,
        selection_output=args.selection_output,
        class_weight_mode=args.class_weight_mode,
        gate_pos_weight_mode=args.gate_pos_weight_mode,
        logger=logger,
    )

    summary = {
        **run_plan,
        "best_epoch": result["best_epoch"],
        "trained_epochs": result["trained_epochs"],
        "selected_output": result["selected_output"],
        "best_validation_score": result["best_validation_score"],
        "hparams": hparams.__dict__,
        "loss_weights": loss_weights.__dict__,
        "log_kpi_mean": log_mean,
        "log_kpi_std": log_std,
        "cuts": list(map(float, args.cuts)),
        "attachment_stats": {k: v for k, v in attachment.items() if not isinstance(v, dict)},
    }
    if "test" in result:
        summary["test"] = result["test"]
        summary["selected_test_six_class"] = result["selected_test_six_class"]
        summary["selected_test_activity"] = result["selected_test_activity"]
        # Compatibility alias: learned six-class metrics, not a hard override.
        summary["selected_test_combined"] = result["selected_test_six_class"]
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if "test" in result:
        six = result["selected_test_six_class"]
        print(f"Completed: {args.scenario_name}")
        print(
            f"Learned six-class GNN selected={result['selected_output']} — "
            f"{six['correct']}/{six['total']} acc={six['accuracy']:.4f} "
            f"bal={six['balanced_accuracy']:.4f} f1={six['macro_f1']:.4f}"
        )
        for cls in range(NUM_CLASSES):
            print(
                f"CLASS {cls}: {six['class_correct'][cls]}/{six['class_support'][cls]} "
                f"accuracy={six['class_accuracy'][cls]:.4f} "
                f"precision={six['class_precision'][cls]:.4f} "
                f"f1={six['class_f1'][cls]:.4f}"
            )
        offsets = []
        for offset in range(-(NUM_CLASSES - 1), NUM_CLASSES):
            key = str(offset)
            label = f"{offset:+d}" if offset != 0 else "0"
            offsets.append(
                f"{label}:{six['error_offset_count'][key]} "
                f"({six['error_offset_rate'][key]:.2%})"
            )
        print("ERROR OFFSETS pred-true | " + " | ".join(offsets))
        print(
            f"UNDER={six['under_count']}/{six['total']} ({six['under_rate']:.4f}) "
            f"OVER={six['over_count']}/{six['total']} ({six['over_rate']:.4f})"
        )
    else:
        print(
            f"Selection completed: best_epoch={result['best_epoch']} "
            f"selected_output={result['selected_output']}"
        )
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
