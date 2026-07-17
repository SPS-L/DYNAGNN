# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: repository integration for pair-aware GINE training
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import optuna
from torch_geometric.loader import DataLoader

from modules.pair_aware_gine import (
    PairAwareHParams,
    PairAwareLossWeights,
    compute_class_weights,
    compute_gate_pos_weight,
    evaluate_saved_pair_aware_model,
    run_pair_aware_training,
)


def normalize_op(value: object) -> str:
    import re

    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text.startswith("operating_point_"):
        suffix = text.rsplit("_", 1)[-1]
        if suffix.isdigit():
            return f"operating_point_{int(suffix)}"
    match = re.fullmatch(r"op_?(\d+)", text)
    if match:
        return f"operating_point_{int(match.group(1))}"
    if text.isdigit():
        return f"operating_point_{int(text)}"
    return text


MODEL_TYPE = "pair_aware_gine"


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

    node_vocab, contingency_vocab = _build_shared_vocab(graph_dataset)
    stats: dict[str, int] = {
        "graphs": len(graph_dataset),
        "node_vocab_size": len(node_vocab) + 1,
        "contingency_vocab_size": len(contingency_vocab) + 1,
        "node_event_graphs": 0,
        "edge_event_graphs": 0,
        "voltage_finite_kpi_targets": 0,
        "spower_finite_kpi_targets": 0,
    }

    for data in graph_dataset:
        data.op_name = normalize_op(getattr(data, "op_name", ""))
        event_id = str(getattr(data, "event_id", "")).strip()
        key = (data.op_name, event_id)

        rows = {
            "kpi_voltage": kpi_voltage_lookup.get(key),
            "kpi_spower": kpi_spower_lookup.get(key),
        }
        missing = [name for name, row in rows.items() if row is None]
        if missing:
            raise KeyError(f"Missing pair-aware rows for {key}: {missing}")

        num_nodes = int(data.x.shape[0])
        node_token = torch.zeros(num_nodes, dtype=torch.long)
        voltage_log = torch.full((num_nodes,), float("nan"), dtype=torch.float32)
        spower_log = torch.full((num_nodes,), float("nan"), dtype=torch.float32)

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

            if node_type == "generator":
                value = _numeric_value(rows["kpi_spower"], kpi_spower.columns, component_id)
                if value is not None:
                    spower_log[node_index] = float(np.log10(max(value, 0.0) + float(epsilon)))
                    stats["spower_finite_kpi_targets"] += 1

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

    logger.info("Pair-aware target attachment: %s", stats)
    return {
        **stats,
        "node_vocab": node_vocab,
        "contingency_vocab": contingency_vocab,
        "paths": {name: str(path) for name, path in paths.items()},
    }


def _task_attr(task: str) -> tuple[str, str, str]:
    if task == "voltage":
        return (
            "y_voltage",
            "y_voltage_log_kpi",
            "bus_node_mask",
        )
    if task == "spower":
        return (
            "y_spower",
            "y_spower_log_kpi",
            "gen_node_mask",
        )
    raise ValueError(f"Unsupported task: {task}")


def _prepare_task(
    task: str,
    train_graphs: Sequence,
    all_splits: Sequence[Sequence],
    *,
    num_classes: int,
) -> tuple[float, float]:
    label_attr, log_attr, mask_attr = _task_attr(task)
    flag_class = num_classes - 1
    values: list[np.ndarray] = []

    for data in train_graphs:
        data.y_class = getattr(data, label_attr)
        data.y_log_kpi = getattr(data, log_attr)
        # Regression targets are collected only for activity classes (< flag_class).
        mask = getattr(data, mask_attr).bool() & (data.y_class >= 0) & (
            data.y_class < flag_class
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



def _loss_weights(config: dict) -> PairAwareLossWeights:
    pair_cfg = ((config.get("training", {}) or {}).get("pair_aware", {}) or {})
    return PairAwareLossWeights(
        classification=float(pair_cfg.get("classification_weight", 1.0)),
        regression=float(pair_cfg.get("regression_weight", 0.30)),
        inactive_gate=float(pair_cfg.get("inactive_gate_weight", 0.20)),
        ordinal_cdf=float(pair_cfg.get("ordinal_weight", 0.10)),
    )


def _suggest(trial: optuna.Trial, name: str, spec: dict):
    if not isinstance(spec, dict):
        raise TypeError(f"optuna.hparams.{name} must be a mapping")
    kind = str(spec.get("type", "")).strip().lower()
    if kind == "categorical":
        choices = list(spec.get("choices") or [])
        if not choices:
            raise ValueError(f"optuna.hparams.{name}.choices cannot be empty")
        return trial.suggest_categorical(name, choices)
    if kind == "int":
        if spec.get("low") is None or spec.get("high") is None:
            raise ValueError(f"optuna.hparams.{name} requires low and high")
        return trial.suggest_int(
            name,
            int(spec["low"]),
            int(spec["high"]),
            step=int(spec.get("step", 1)),
            log=bool(spec.get("log", False)),
        )
    if kind == "float":
        if spec.get("low") is None or spec.get("high") is None:
            raise ValueError(f"optuna.hparams.{name} requires low and high")
        kwargs = {"log": bool(spec.get("log", False))}
        if spec.get("step") is not None:
            kwargs["step"] = float(spec["step"])
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), **kwargs)
    raise ValueError(
        f"Unsupported optuna.hparams.{name}.type={kind!r}; "
        "expected categorical, int, or float"
    )


def _sample_hparams(trial: optuna.Trial, space: dict) -> PairAwareHParams:
    required = (
        "hidden_dim",
        "node_id_dim",
        "contingency_id_dim",
        "type_dim",
        "pair_dim",
        "num_gnn_layers",
        "decoder_hidden_dim",
        "dropout",
        "lr",
        "weight_decay",
    )
    missing = [name for name in required if name not in space]
    if missing:
        raise KeyError(f"Missing Optuna hyperparameter definitions: {missing}")
    return PairAwareHParams(
        hidden_dim=int(_suggest(trial, "hidden_dim", space["hidden_dim"])),
        node_id_dim=int(_suggest(trial, "node_id_dim", space["node_id_dim"])),
        contingency_id_dim=int(
            _suggest(trial, "contingency_id_dim", space["contingency_id_dim"])
        ),
        type_dim=int(_suggest(trial, "type_dim", space["type_dim"])),
        pair_dim=int(_suggest(trial, "pair_dim", space["pair_dim"])),
        num_gnn_layers=int(
            _suggest(trial, "num_gnn_layers", space["num_gnn_layers"])
        ),
        decoder_hidden_dim=int(
            _suggest(trial, "decoder_hidden_dim", space["decoder_hidden_dim"])
        ),
        dropout=float(_suggest(trial, "dropout", space["dropout"])),
        lr=float(_suggest(trial, "lr", space["lr"])),
        weight_decay=float(_suggest(trial, "weight_decay", space["weight_decay"])),
    )


def _hparams_from_best_params(params: dict) -> PairAwareHParams:
    return PairAwareHParams(
        hidden_dim=int(params["hidden_dim"]),
        node_id_dim=int(params["node_id_dim"]),
        contingency_id_dim=int(params["contingency_id_dim"]),
        type_dim=int(params["type_dim"]),
        pair_dim=int(params["pair_dim"]),
        num_gnn_layers=int(params["num_gnn_layers"]),
        decoder_hidden_dim=int(params["decoder_hidden_dim"]),
        dropout=float(params["dropout"]),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )


def _task_spec(task: str) -> tuple[str, str]:
    if task == "voltage":
        return "bus_node_mask", "voltage"
    if task == "spower":
        return "gen_node_mask", "spower"
    raise ValueError(f"Unsupported task: {task}")


def _save_deployment_checkpoint(
    *,
    task: str,
    model_dir: Path,
    state_dict: dict,
    hparams: PairAwareHParams,
    loss_weights: PairAwareLossWeights,
    attachment: dict[str, Any],
    config: dict,
    log_mean: float,
    log_std: float,
    cuts: Sequence[float],
    epsilon: float,
    gate_threshold: float,
    selected_output: str,
    best_epoch: int,
    best_validation_score: float,
    num_classes: int,
) -> Path:
    checkpoint = {
        "checkpoint_version": 3,
        "model_type": MODEL_TYPE,
        "task": task,
        "model_state_dict": state_dict,
        "hparams": asdict(hparams),
        "loss_weights": asdict(loss_weights),
        "num_classes": num_classes,
        "num_node_tokens": int(attachment["node_vocab_size"]),
        "num_contingency_tokens": int(attachment["contingency_vocab_size"]),
        "node_vocab": attachment["node_vocab"],
        "contingency_vocab": attachment["contingency_vocab"],
        "selected_output": str(selected_output),
        "best_epoch": int(best_epoch),
        "best_validation_score": float(best_validation_score),
        "log_kpi_mean": float(log_mean),
        "log_kpi_std": float(log_std),
        "cuts": [float(value) for value in cuts],
        "epsilon": float(epsilon),
        "gate_threshold": float(gate_threshold),
        "flag_class_handling": "learned direct prediction; no deterministic override",
        "node_continuous_columns": [1, 2, 3, 4, 6],
        "edge_continuous_features": ["r", "x", "b1", "g1", "b2", "g2"],
        "config_version": (config.get("dynagnn", {}) or {}).get("version"),
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / f"{task}_best_model.pt"
    torch.save(checkpoint, checkpoint_path)
    metadata = {key: value for key, value in checkpoint.items() if key != "model_state_dict"}
    (model_dir / f"{task}_best_hparams.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return checkpoint_path


def run_task_training(
    *,
    task: str,
    train_scaled: list,
    val_scaled: list,
    test_scaled: list,
    training_dir: Path,
    model_dir: Path,
    config: dict,
    attachment: dict[str, Any],
    logger: logging.Logger,
) -> dict:
    training_cfg = config.get("training", {}) or {}
    pair_cfg = training_cfg.get("pair_aware", {}) or {}
    optuna_cfg = config.get("optuna", {}) or {}
    hparam_space = optuna_cfg.get("hparams", {}) or {}
    n_trials = int(optuna_cfg.get("n_trials", 15))
    if n_trials < 1:
        raise ValueError("optuna.n_trials must be at least 1")

    model_cfg = config.get("model", {}) or {}
    if "num_classes" not in model_cfg:
        raise KeyError("Missing required config key: model.num_classes")
    num_classes = int(model_cfg["num_classes"])
    if num_classes < 2:
        raise ValueError(f"model.num_classes must be >= 2, got {num_classes}")

    target_mask_attr, task_name = _task_spec(task)
    log_mean, log_std = _prepare_task(
        task, train_scaled, [train_scaled, val_scaled, test_scaled], num_classes=num_classes
    )
    logger.info("%s log-KPI mean=%.6f std=%.6f", task, log_mean, log_std)

    batch_size = int(training_cfg.get("batch_size", 8))
    train_loader = DataLoader(train_scaled, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_scaled, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_scaled, batch_size=batch_size, shuffle=False)

    class_bins = ((config.get("kpi", {}) or {}).get("class_bins", {}) or {})
    expected_cuts = num_classes - 2
    cuts = _float_list(
        (class_bins.get(task_name, {}) or {}).get("cuts"),
        expected=expected_cuts,
        label=f"kpi.class_bins.{task_name}.cuts",
    )
    cuts_array = np.asarray(cuts, dtype=np.float64)

    epochs = int(training_cfg.get("epochs", 60))
    patience = int(training_cfg.get("patience", 10))
    seed = int(training_cfg.get("seed", 42))
    epsilon = float(pair_cfg.get("epsilon", 1.0e-10))
    gate_threshold = float(pair_cfg.get("gate_threshold", 0.5))
    class_weight_mode = str(pair_cfg.get("class_weight_mode", "sqrt_inverse"))
    gate_pos_weight_mode = str(pair_cfg.get("gate_pos_weight_mode", "balanced"))
    selection_output = str(pair_cfg.get("selection_output", "auto")).strip().lower()
    if selection_output in {"", "auto", "none", "null"}:
        selection_output_arg = None
    elif selection_output in {"class", "gated", "log_kpi"}:
        selection_output_arg = selection_output
    else:
        raise ValueError(
            "training.pair_aware.selection_output must be auto, class, gated, or log_kpi"
        )
    loss_weights = _loss_weights(config)

    task_dir = Path(training_dir) / task
    task_dir.mkdir(parents=True, exist_ok=True)
    trials_root = task_dir / "optuna_trials"
    trials_root.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(task_dir / f'optuna_{task}.sqlite3').as_posix()}"
    # Folder name comes from config (optuna.study_name); Optuna DB study is per-task inside it.
    study_folder = Path(training_dir).name
    study_name = f"{study_folder}__pair_aware_{task}"

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    class_weights = compute_class_weights(
        train_loader, class_weight_mode, device, target_mask_attr, num_classes
    )
    inactive_pos_weight = compute_gate_pos_weight(
        train_loader, gate_pos_weight_mode, device, target_mask_attr, num_classes
    )

    logger.info("========== %s Optuna study ==========", task.capitalize())
    logger.info(
        "study_name=%s | n_trials=%d | epochs=%d | patience=%d | device=%s | num_classes=%d",
        study_folder,
        n_trials,
        epochs,
        patience,
        device,
        num_classes,
    )
    logger.info("Loss weights: %s", asdict(loss_weights))
    logger.info(
        "Class weights: %s",
        None
        if class_weights is None
        else [round(float(v), 4) for v in class_weights.detach().cpu().tolist()],
    )
    logger.info(
        "Inactive gate pos_weight: %s",
        None
        if inactive_pos_weight is None
        else [round(float(v), 4) for v in inactive_pos_weight.detach().cpu().tolist()],
    )
    logger.info("Artifacts: %s", task_dir)

    # Keep Optuna's own INFO chatter out of dynagnn.log / stdout tee.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        hparams = _sample_hparams(trial, hparam_space)
        trial_root = trials_root / f"trial_{trial.number}"
        try:
            result = run_pair_aware_training(
                task=task,
                target_mask_attr=target_mask_attr,
                train_loader=train_loader,
                validation_loader=val_loader,
                test_loader=None,
                output_dir=trial_root,
                num_node_tokens=int(attachment["node_vocab_size"]),
                num_contingency_tokens=int(attachment["contingency_vocab_size"]),
                hparams=hparams,
                loss_weights=loss_weights,
                log_mean=log_mean,
                log_std=log_std,
                cuts=cuts_array,
                epsilon=epsilon,
                gate_threshold=gate_threshold,
                epochs=epochs,
                patience=patience,
                fixed_epochs=None,
                selection_output=selection_output_arg,
                class_weight_mode=class_weight_mode,
                gate_pos_weight_mode=gate_pos_weight_mode,
                num_classes=num_classes,
                logger=logger,
                trial=trial,
            )
        except optuna.TrialPruned:
            raise
        score = result.get("best_validation_score")
        if score is None:
            raise RuntimeError(f"Optuna trial {trial.number} produced no validation score")
        trial.set_user_attr("model_state_path", result["model_state_path"])
        trial.set_user_attr("selected_output", result["selected_output"])
        trial.set_user_attr("best_epoch", int(result["best_epoch"]))
        return float(score)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=min(5, n_trials)),
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    if study.best_trial.value is None:
        raise RuntimeError(f"No completed Optuna trial for {task}")

    study.trials_dataframe().to_csv(task_dir / "optuna_trials.csv", index=False)
    best_trial = study.best_trial
    best_hparams = _hparams_from_best_params(best_trial.params)
    state_path = Path(str(best_trial.user_attrs.get("model_state_path", "")))
    if not state_path.exists():
        raise FileNotFoundError(
            f"Best {task} Optuna trial state was not found: {state_path}"
        )
    state_dict = torch.load(state_path, map_location="cpu", weights_only=False)
    selected_output = str(best_trial.user_attrs.get("selected_output", "class"))
    best_epoch = int(best_trial.user_attrs.get("best_epoch", 0))

    logger.info(
        "========== %s: best trial %d ==========",
        task.capitalize(),
        best_trial.number,
    )
    logger.info(
        "best_val=%.4f | best_epoch=%d | decode=%s | hparams=%s",
        float(best_trial.value),
        best_epoch,
        selected_output,
        best_trial.params,
    )
    logger.info("Evaluating best %s checkpoint on the held-out test set...", task)

    best_history_csv = state_path.parent / "history.csv"
    test_result = evaluate_saved_pair_aware_model(
        task=task,
        target_mask_attr=target_mask_attr,
        state_dict=state_dict,
        train_loader=train_loader,
        test_loader=test_loader,
        output_dir=training_dir,
        num_node_tokens=int(attachment["node_vocab_size"]),
        num_contingency_tokens=int(attachment["contingency_vocab_size"]),
        hparams=best_hparams,
        loss_weights=loss_weights,
        log_mean=log_mean,
        log_std=log_std,
        cuts=cuts_array,
        epsilon=epsilon,
        gate_threshold=gate_threshold,
        selected_output=selected_output,
        class_weight_mode=class_weight_mode,
        gate_pos_weight_mode=gate_pos_weight_mode,
        num_classes=num_classes,
        logger=logger,
        history_csv=best_history_csv,
    )

    checkpoint_path = _save_deployment_checkpoint(
        task=task,
        model_dir=model_dir,
        state_dict=state_dict,
        hparams=best_hparams,
        loss_weights=loss_weights,
        attachment=attachment,
        config=config,
        log_mean=log_mean,
        log_std=log_std,
        cuts=cuts,
        epsilon=epsilon,
        gate_threshold=gate_threshold,
        selected_output=selected_output,
        best_epoch=best_epoch,
        best_validation_score=float(best_trial.value),
        num_classes=num_classes,
    )
    logger.info(
        "Saved %s deployment checkpoint: %s (trial=%d, selected_output=%s)",
        task,
        checkpoint_path,
        best_trial.number,
        selected_output,
    )
    return {
        "best_trial": int(best_trial.number),
        "best_validation_score": float(best_trial.value),
        "best_epoch": best_epoch,
        "selected_output": selected_output,
        "hparams": asdict(best_hparams),
        "checkpoint": str(checkpoint_path),
        "test": test_result,
        "log_kpi_mean": log_mean,
        "log_kpi_std": log_std,
    }
