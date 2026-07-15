# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: repository integration for pair-aware six-class GINE training
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from modules.gat_spower_pair_aware_six_class import (
    PairAwareHParams as SpowerHParams,
    PairAwareLossWeights as SpowerLossWeights,
    run_pair_aware_training as run_spower_pair_aware_training,
)
from modules.gat_voltage_pair_aware_six_class import (
    PairAwareHParams as VoltageHParams,
    PairAwareLossWeights as VoltageLossWeights,
    run_pair_aware_training as run_voltage_pair_aware_training,
)
from modules.op_context_aggregate import normalize_op

PAIR_AWARE_ARCHITECTURE = "pair_aware_gine"
NUM_CLASSES = 6
ACTIVITY_CLASSES = 5


def _first_existing(paths: Sequence[Path], label: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing {label}. Tried: {[str(path) for path in paths]}")


def _task_paths(data_dir: Path) -> dict[str, Path]:
    return {
        "kpi_voltage": _first_existing(
            [data_dir / "KPI" / "KPI_voltage.csv", data_dir / "KPI_voltage.csv"],
            "KPI_voltage.csv",
        ),
        "kpi_spower": _first_existing(
            [data_dir / "KPI" / "KPI_spower.csv", data_dir / "KPI_spower.csv"],
            "KPI_spower.csv",
        ),
        "disc_voltage": _first_existing(
            [
                data_dir / "Disconnections" / "DISC_voltage.csv",
                data_dir / "DISC" / "DISC_voltage.csv",
                data_dir / "Dataset" / "DISC_voltage.csv",
                data_dir / "DISC_voltage.csv",
            ],
            "DISC_voltage.csv",
        ),
        "disc_spower": _first_existing(
            [
                data_dir / "Disconnections" / "DISC_spower.csv",
                data_dir / "DISC" / "DISC_spower.csv",
                data_dir / "Dataset" / "DISC_spower.csv",
                data_dir / "DISC_spower.csv",
            ],
            "DISC_spower.csv",
        ),
    }


def _prepare_lookup(path: Path) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.Series]]:
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [str(column).strip() for column in frame.columns]
    for column in ("OP", "Contingency"):
        if column not in frame.columns:
            raise KeyError(f"{path.name} is missing required column {column!r}")
        frame[column] = frame[column].astype("string").str.strip()
    frame["OP"] = frame["OP"].map(normalize_op)
    lookup = {
        (str(row["OP"]).strip(), str(row["Contingency"]).strip()): row
        for _, row in frame.iterrows()
    }
    return frame, lookup


def _node_id(meta_key: str, metadata: dict) -> str:
    return str(metadata.get("id", meta_key)).strip()


def _build_shared_vocab(graph_dataset: Sequence) -> tuple[dict[str, int], dict[str, int]]:
    node_ids: set[str] = set()
    contingencies: set[str] = set()
    for data in graph_dataset:
        contingencies.add(str(getattr(data, "event_id", "")).strip())
        metadata = getattr(data, "metadata", {}) or {}
        for key, node_meta in (metadata.get("node_metadata", {}) or {}).items():
            node_ids.add(_node_id(str(key), node_meta))
    node_vocab = {identifier: index + 1 for index, identifier in enumerate(sorted(node_ids))}
    contingency_vocab = {
        identifier: index + 1 for index, identifier in enumerate(sorted(contingencies))
    }
    return node_vocab, contingency_vocab


def _event_tensors(data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_nodes = int(data.x.shape[0])
    num_edges = int(data.edge_attr.shape[0])
    event_node_mask = torch.zeros(num_nodes, dtype=torch.bool)
    event_edge_mask = torch.zeros(num_edges, dtype=torch.bool)

    location_type = str(getattr(data, "event_location_type", ""))
    location_index = int(getattr(data, "event_location_index", -1))
    metadata = getattr(data, "metadata", {}) or {}

    if location_type == "node":
        if not (0 <= location_index < num_nodes):
            raise IndexError(f"Bad node event index {location_index}")
        event_node_mask[location_index] = True
        event_graph_type = 0
    elif location_type == "edge":
        edge_schema = list(metadata.get("edge_feature_schema", []) or [])
        if "fault_on" not in edge_schema:
            raise KeyError("fault_on missing from edge_feature_schema")
        fault_column = edge_schema.index("fault_on")
        event_edge_mask = data.edge_attr[:, fault_column] > 0.5
        if not bool(event_edge_mask.any()):
            if not (0 <= location_index < num_edges):
                raise IndexError(f"Bad edge event index {location_index}")
            event_edge_mask[location_index] = True
        endpoints = torch.unique(data.edge_index[:, event_edge_mask].reshape(-1))
        event_node_mask[endpoints] = True
        event_graph_type = 1
    else:
        raise ValueError(f"Unsupported event location type: {location_type!r}")

    return (
        event_node_mask,
        event_edge_mask,
        torch.tensor([event_graph_type], dtype=torch.long),
    )


def _numeric_value(row: pd.Series, columns: pd.Index, component_id: str) -> float | None:
    if component_id not in columns:
        return None
    value = pd.to_numeric(pd.Series([row[component_id]]), errors="coerce").iloc[0]
    if pd.isna(value) or not np.isfinite(float(value)):
        return None
    return float(value)


def attach_pair_aware_targets(
    graph_dataset: list,
    *,
    data_dir: Path,
    epsilon: float,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Attach shared identity/event tensors and task-specific KPI targets.

    The vocabularies are shared by Voltage and Spower so both deployment
    checkpoints use identical node and contingency tokenization.
    """
    if not graph_dataset:
        raise RuntimeError("Cannot attach pair-aware targets to an empty graph dataset")

    paths = _task_paths(Path(data_dir))
    kpi_voltage, kpi_voltage_lookup = _prepare_lookup(paths["kpi_voltage"])
    kpi_spower, kpi_spower_lookup = _prepare_lookup(paths["kpi_spower"])
    disc_voltage, disc_voltage_lookup = _prepare_lookup(paths["disc_voltage"])
    disc_spower, disc_spower_lookup = _prepare_lookup(paths["disc_spower"])

    node_vocab, contingency_vocab = _build_shared_vocab(graph_dataset)
    stats: dict[str, int] = {
        "graphs": len(graph_dataset),
        "node_vocab_size": len(node_vocab) + 1,
        "contingency_vocab_size": len(contingency_vocab) + 1,
        "node_event_graphs": 0,
        "edge_event_graphs": 0,
        "voltage_finite_kpi_targets": 0,
        "spower_finite_kpi_targets": 0,
        "voltage_class5_targets": 0,
        "spower_class5_targets": 0,
        "voltage_class5_label_mask_mismatches": 0,
        "spower_class5_label_mask_mismatches": 0,
    }

    for data in graph_dataset:
        data.op_name = normalize_op(getattr(data, "op_name", ""))
        event_id = str(getattr(data, "event_id", "")).strip()
        key = (data.op_name, event_id)

        rows = {
            "kpi_voltage": kpi_voltage_lookup.get(key),
            "kpi_spower": kpi_spower_lookup.get(key),
            "disc_voltage": disc_voltage_lookup.get(key),
            "disc_spower": disc_spower_lookup.get(key),
        }
        missing = [name for name, row in rows.items() if row is None]
        if missing:
            raise KeyError(f"Missing pair-aware rows for {key}: {missing}")

        num_nodes = int(data.x.shape[0])
        node_token = torch.zeros(num_nodes, dtype=torch.long)
        voltage_log = torch.full((num_nodes,), float("nan"), dtype=torch.float32)
        spower_log = torch.full((num_nodes,), float("nan"), dtype=torch.float32)
        voltage_structural = torch.zeros(num_nodes, dtype=torch.bool)
        spower_structural = torch.zeros(num_nodes, dtype=torch.bool)

        metadata = getattr(data, "metadata", {}) or {}
        for meta_key, node_meta in (metadata.get("node_metadata", {}) or {}).items():
            node_index = int(node_meta["index"])
            component_id = _node_id(str(meta_key), node_meta)
            node_token[node_index] = int(node_vocab.get(component_id, 0))
            node_type = str(node_meta.get("type", "")).lower()

            if node_type == "bus":
                value = _numeric_value(rows["kpi_voltage"], kpi_voltage.columns, component_id)
                if value is not None:
                    voltage_log[node_index] = float(np.log10(max(value, 0.0) + float(epsilon)))
                    stats["voltage_finite_kpi_targets"] += 1
                disc_value = _numeric_value(
                    rows["disc_voltage"], disc_voltage.columns, component_id
                )
                voltage_structural[node_index] = bool(
                    disc_value is not None and disc_value > 0.5
                )

            if node_type == "generator":
                value = _numeric_value(rows["kpi_spower"], kpi_spower.columns, component_id)
                if value is not None:
                    spower_log[node_index] = float(np.log10(max(value, 0.0) + float(epsilon)))
                    stats["spower_finite_kpi_targets"] += 1
                disc_value = _numeric_value(
                    rows["disc_spower"], disc_spower.columns, component_id
                )
                spower_structural[node_index] = bool(
                    disc_value is not None and disc_value > 0.5
                )

        event_node_mask, event_edge_mask, event_graph_type = _event_tensors(data)
        if int(event_graph_type.item()) == 0:
            stats["node_event_graphs"] += 1
        else:
            stats["edge_event_graphs"] += 1

        data.node_token = node_token
        data.contingency_token = torch.tensor(
            [int(contingency_vocab.get(event_id, 0))], dtype=torch.long
        )
        data.event_node_mask = event_node_mask
        data.event_edge_mask = event_edge_mask
        data.event_graph_type = event_graph_type
        data.y_voltage_log_kpi = voltage_log
        data.y_spower_log_kpi = spower_log
        data.voltage_structural_class5_mask = voltage_structural
        data.spower_structural_class5_mask = spower_structural

        voltage_labels = data.y_voltage[data.bus_node_mask]
        voltage_structural_targets = voltage_structural[data.bus_node_mask]
        spower_labels = data.y_spower[data.gen_node_mask]
        spower_structural_targets = spower_structural[data.gen_node_mask]

        stats["voltage_class5_targets"] += int(voltage_structural_targets.sum().item())
        stats["spower_class5_targets"] += int(spower_structural_targets.sum().item())
        stats["voltage_class5_label_mask_mismatches"] += int(
            ((voltage_labels == 5) != voltage_structural_targets).sum().item()
        )
        stats["spower_class5_label_mask_mismatches"] += int(
            ((spower_labels == 5) != spower_structural_targets).sum().item()
        )

    for task in ("voltage", "spower"):
        mismatch_key = f"{task}_class5_label_mask_mismatches"
        if int(stats[mismatch_key]) != 0:
            raise RuntimeError(
                f"{task} structural class-5 mask disagrees with labels: "
                f"{stats[mismatch_key]} mismatches"
            )

    logger.info("Pair-aware target attachment: %s", stats)
    return {
        **stats,
        "node_vocab": node_vocab,
        "contingency_vocab": contingency_vocab,
        "paths": {name: str(path) for name, path in paths.items()},
    }


def _task_attr(task: str) -> tuple[str, str, str, str]:
    if task == "voltage":
        return (
            "y_voltage",
            "y_voltage_log_kpi",
            "voltage_structural_class5_mask",
            "bus_node_mask",
        )
    if task == "spower":
        return (
            "y_spower",
            "y_spower_log_kpi",
            "spower_structural_class5_mask",
            "gen_node_mask",
        )
    raise ValueError(f"Unsupported task: {task}")


def _prepare_task(
    task: str,
    train_graphs: Sequence,
    all_splits: Sequence[Sequence],
) -> tuple[float, float]:
    label_attr, log_attr, structural_attr, mask_attr = _task_attr(task)
    values: list[np.ndarray] = []

    for data in train_graphs:
        data.y_class = getattr(data, label_attr)
        data.y_log_kpi = getattr(data, log_attr)
        data.structural_class5_mask = getattr(data, structural_attr)
        mask = getattr(data, mask_attr).bool() & (data.y_class >= 0) & (
            data.y_class < ACTIVITY_CLASSES
        )
        target = data.y_log_kpi[mask]
        finite = torch.isfinite(target)
        if bool(finite.any()):
            values.append(target[finite].cpu().numpy())

    if not values:
        raise RuntimeError(f"No finite training log-KPI targets for {task}")
    merged = np.concatenate(values)
    mean = float(merged.mean())
    std = float(merged.std())
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 1.0e-12:
        raise RuntimeError(f"Invalid {task} log-KPI statistics mean={mean} std={std}")

    for split in all_splits:
        for data in split:
            data.y_class = getattr(data, label_attr)
            data.y_log_kpi = getattr(data, log_attr)
            data.structural_class5_mask = getattr(data, structural_attr)
            standardized = torch.full_like(data.y_log_kpi, float("nan"))
            finite = torch.isfinite(data.y_log_kpi)
            standardized[finite] = (data.y_log_kpi[finite] - mean) / std
            data.y_log_kpi_std = standardized

    return mean, std


def _float_list(value: Any, *, expected: int, label: str) -> list[float]:
    if value is None:
        raise KeyError(f"Missing required config key: {label}")
    result = [float(item) for item in value]
    if len(result) != expected:
        raise ValueError(f"{label} must contain {expected} values, got {len(result)}")
    return result


def _pair_configs(config: dict, task: str):
    model_cfg = config.get("model", {}) or {}
    training_cfg = config.get("training", {}) or {}
    pair_model_cfg = model_cfg.get("pair_aware", {}) or {}
    pair_training_cfg = training_cfg.get("pair_aware", {}) or {}

    hparams_values = {
        "hidden_dim": int(pair_model_cfg.get("hidden_dim", 128)),
        "node_id_dim": int(pair_model_cfg.get("node_id_dim", 24)),
        "contingency_id_dim": int(pair_model_cfg.get("contingency_id_dim", 32)),
        "type_dim": int(pair_model_cfg.get("type_dim", 8)),
        "pair_dim": int(pair_model_cfg.get("pair_dim", 32)),
        "op_context_embedding_dim": int(pair_model_cfg.get("op_context_embedding_dim", 32)),
        "num_gnn_layers": int(pair_model_cfg.get("num_gnn_layers", 3)),
        "decoder_hidden_dim": int(pair_model_cfg.get("decoder_hidden_dim", 256)),
        "dropout": float(pair_model_cfg.get("dropout", 0.15)),
        "lr": float(pair_training_cfg.get("lr", 2.0e-4)),
        "weight_decay": float(pair_training_cfg.get("weight_decay", 1.0e-5)),
    }
    loss_values = {
        "classification": float(pair_training_cfg.get("classification_weight", 1.0)),
        "regression": float(pair_training_cfg.get("regression_weight", 0.30)),
        "inactive_gate": float(pair_training_cfg.get("inactive_gate_weight", 0.20)),
        "ordinal_cdf": float(pair_training_cfg.get("ordinal_weight", 0.10)),
    }
    if task == "voltage":
        return VoltageHParams(**hparams_values), VoltageLossWeights(**loss_values)
    return SpowerHParams(**hparams_values), SpowerLossWeights(**loss_values)


def _save_deployment_checkpoint(
    *,
    task: str,
    model_dir: Path,
    output_root: Path,
    result: dict,
    hparams,
    loss_weights,
    attachment: dict[str, Any],
    config: dict,
    log_mean: float,
    log_std: float,
    cuts: Sequence[float],
    epsilon: float,
    gate_threshold: float,
) -> Path:
    task_training_dir = (
        output_root
        / "training"
        / f"{task}_pair_aware_six_class_gnn"
    )
    state_path = task_training_dir / "model_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Pair-aware training did not produce {state_path}")
    state_dict = torch.load(state_path, map_location="cpu", weights_only=False)

    checkpoint = {
        "checkpoint_version": 2,
        "architecture": PAIR_AWARE_ARCHITECTURE,
        "task": task,
        "model_state_dict": state_dict,
        "hparams": asdict(hparams),
        "loss_weights": asdict(loss_weights),
        "num_classes": NUM_CLASSES,
        "num_node_tokens": int(attachment["node_vocab_size"]),
        "num_contingency_tokens": int(attachment["contingency_vocab_size"]),
        "node_vocab": attachment["node_vocab"],
        "contingency_vocab": attachment["contingency_vocab"],
        "selected_output": str(result["selected_output"]),
        "best_epoch": int(result["best_epoch"]),
        "best_validation_score": result.get("best_validation_score"),
        "log_kpi_mean": float(log_mean),
        "log_kpi_std": float(log_std),
        "cuts": [float(value) for value in cuts],
        "epsilon": float(epsilon),
        "gate_threshold": float(gate_threshold),
        "use_op_context": False,
        "op_context_dim": 0,
        "class5_handling": "learned direct prediction; no deterministic override",
        "node_continuous_columns": [1, 2, 3, 4, 6],
        "edge_continuous_features": ["r", "x", "b1", "g1", "b2", "g2"],
        "config_version": (config.get("dynagnn", {}) or {}).get("version"),
    }
    checkpoint_path = model_dir / f"pair_aware_{task}_best_model.pt"
    torch.save(checkpoint, checkpoint_path)

    metadata = {key: value for key, value in checkpoint.items() if key != "model_state_dict"}
    (model_dir / f"pair_aware_{task}_best_hparams.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return checkpoint_path


def run_pair_aware_repository_training(
    *,
    train_scaled: list,
    val_scaled: list,
    test_scaled: list,
    training_dir: Path,
    model_dir: Path,
    config: dict,
    attachment: dict[str, Any],
    logger: logging.Logger,
) -> dict[str, dict]:
    """Train Voltage and Spower pair-aware models through the standard pipeline."""
    del training_dir  # model modules write below output_root/training themselves
    model_cfg = config.get("model", {}) or {}
    training_cfg = config.get("training", {}) or {}
    pair_training_cfg = training_cfg.get("pair_aware", {}) or {}

    if int(model_cfg.get("num_classes", 0)) != NUM_CLASSES:
        raise ValueError(
            f"{PAIR_AWARE_ARCHITECTURE} requires model.num_classes={NUM_CLASSES}"
        )
    op_context_mode = str(pair_training_cfg.get("op_context_mode", "none")).lower()
    if op_context_mode != "none":
        raise ValueError(
            "Repository inference currently supports training.pair_aware.op_context_mode=none only"
        )

    output_root = Path(model_dir).parent
    batch_size = int(training_cfg.get("batch_size", 8))
    epochs = int(training_cfg.get("epochs", 60))
    patience = int(training_cfg.get("patience", 10))
    epsilon = float(pair_training_cfg.get("epsilon", 1.0e-10))
    gate_threshold = float(pair_training_cfg.get("gate_threshold", 0.5))
    class_weight_mode = str(pair_training_cfg.get("class_weight_mode", "sqrt_inverse"))
    gate_pos_weight_mode = str(pair_training_cfg.get("gate_pos_weight_mode", "balanced"))
    selected_output = pair_training_cfg.get("selection_output")
    if selected_output in ("", "auto", "none", "null"):
        selected_output = None

    results: dict[str, dict] = {}
    for task in ("voltage", "spower"):
        logger.info("Starting pair-aware %s training flow", task)
        log_mean, log_std = _prepare_task(
            task,
            train_scaled,
            [train_scaled, val_scaled, test_scaled],
        )
        logger.info("%s log-KPI mean=%.6f std=%.6f", task, log_mean, log_std)

        train_loader = DataLoader(train_scaled, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(val_scaled, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_scaled, batch_size=batch_size, shuffle=False)

        hparams, loss_weights = _pair_configs(config, task)
        kpi_cfg = config.get("kpi", {}) or {}
        class_bins_cfg = kpi_cfg.get("class_bins", {}) or {}
        task_kpi_cfg = class_bins_cfg.get(task, {}) or kpi_cfg.get(task, {}) or {}
        cuts = _float_list(
            task_kpi_cfg.get("cuts"),
            expected=4,
            label=f"kpi.class_bins.{task}.cuts",
        )

        common_kwargs = dict(
            train_loader=train_loader,
            validation_loader=validation_loader,
            test_loader=test_loader,
            output_dir=output_root,
            num_node_tokens=int(attachment["node_vocab_size"]),
            num_contingency_tokens=int(attachment["contingency_vocab_size"]),
            op_context_dim=0,
            use_op_context=False,
            hparams=hparams,
            loss_weights=loss_weights,
            log_mean=log_mean,
            log_std=log_std,
            cuts=np.asarray(cuts, dtype=np.float64),
            epsilon=epsilon,
            gate_threshold=gate_threshold,
            epochs=epochs,
            patience=patience,
            fixed_epochs=None,
            selection_output=selected_output,
            class_weight_mode=class_weight_mode,
            gate_pos_weight_mode=gate_pos_weight_mode,
            logger=logger,
        )
        if task == "voltage":
            result = run_voltage_pair_aware_training(**common_kwargs)
        else:
            result = run_spower_pair_aware_training(**common_kwargs)

        checkpoint_path = _save_deployment_checkpoint(
            task=task,
            model_dir=Path(model_dir),
            output_root=output_root,
            result=result,
            hparams=hparams,
            loss_weights=loss_weights,
            attachment=attachment,
            config=config,
            log_mean=log_mean,
            log_std=log_std,
            cuts=cuts,
            epsilon=epsilon,
            gate_threshold=gate_threshold,
        )
        logger.info(
            "Saved pair-aware %s deployment checkpoint: %s (selected_output=%s)",
            task,
            checkpoint_path,
            result["selected_output"],
        )
        results[task] = {
            **result,
            "checkpoint": str(checkpoint_path),
            "log_kpi_mean": log_mean,
            "log_kpi_std": log_std,
        }

    summary_path = Path(model_dir) / "pair_aware_training_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
