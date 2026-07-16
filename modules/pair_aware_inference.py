# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: pair-aware six-class GINE inference helpers
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Batch

from modules.pair_aware_gine import PairAwareGINE, PairAwareHParams

MODEL_TYPE = "pair_aware_gine"


def load_pair_aware_checkpoint(path: Path, *, expected_task: str) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing pair-aware checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dictionary in {path}")
    if checkpoint.get("model_type") != MODEL_TYPE:
        raise ValueError(
            f"Checkpoint {path} has model_type={checkpoint.get('model_type')!r}, "
            f"expected {MODEL_TYPE!r}"
        )
    if checkpoint.get("task") != expected_task:
        raise ValueError(
            f"Checkpoint {path} has task={checkpoint.get('task')!r}, "
            f"expected {expected_task!r}"
        )
    required = {
        "model_state_dict",
        "hparams",
        "num_node_tokens",
        "num_contingency_tokens",
        "node_vocab",
        "contingency_vocab",
        "selected_output",
        "cuts",
        "log_kpi_mean",
        "log_kpi_std",
        "epsilon",
        "gate_threshold",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise KeyError(f"Checkpoint {path} is missing fields: {missing}")
    return checkpoint


def load_pair_aware_model(checkpoint: dict[str, Any], device: torch.device):
    task = str(checkpoint["task"])
    hparams_dict = dict(checkpoint["hparams"])
    target_mask_attr = {"voltage": "bus_node_mask", "spower": "gen_node_mask"}.get(task)
    if target_mask_attr is None:
        raise ValueError(f"Unsupported checkpoint task: {task!r}")
    model = PairAwareGINE(
        num_node_tokens=int(checkpoint["num_node_tokens"]),
        num_contingency_tokens=int(checkpoint["num_contingency_tokens"]),
        target_mask_attr=target_mask_attr,
        hparams=PairAwareHParams(**hparams_dict),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _node_id(meta_key: str, metadata: dict) -> str:
    return str(metadata.get("id", meta_key)).strip()


def _attach_pair_tensors(sample, checkpoint: dict[str, Any]):
    data = sample.clone()
    metadata = getattr(data, "metadata", {}) or {}
    num_nodes = int(data.x.shape[0])
    num_edges = int(data.edge_attr.shape[0])

    node_vocab = {str(key): int(value) for key, value in checkpoint["node_vocab"].items()}
    contingency_vocab = {
        str(key): int(value) for key, value in checkpoint["contingency_vocab"].items()
    }
    node_token = torch.zeros(num_nodes, dtype=torch.long)
    for meta_key, node_meta in (metadata.get("node_metadata", {}) or {}).items():
        node_index = int(node_meta["index"])
        node_token[node_index] = int(
            node_vocab.get(_node_id(str(meta_key), node_meta), 0)
        )

    event_id = str(getattr(data, "event_id", "")).strip()
    contingency_token = torch.tensor(
        [int(contingency_vocab.get(event_id, 0))], dtype=torch.long
    )

    event_node_mask = torch.zeros(num_nodes, dtype=torch.bool)
    event_edge_mask = torch.zeros(num_edges, dtype=torch.bool)
    location_type = str(getattr(data, "event_location_type", ""))
    location_index = int(getattr(data, "event_location_index", -1))

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

    data.node_token = node_token
    data.contingency_token = contingency_token
    data.event_node_mask = event_node_mask
    data.event_edge_mask = event_edge_mask
    data.event_graph_type = torch.tensor([event_graph_type], dtype=torch.long)
    return data


def _decode(output: dict[str, torch.Tensor], checkpoint: dict[str, Any]) -> torch.Tensor:
    logits = output["class_logits"]
    selected_output = str(checkpoint.get("selected_output", "class"))

    if selected_output == "class":
        return logits.argmax(dim=1)

    if selected_output == "gated":
        inactive_probability = torch.sigmoid(output["inactive_logit"])
        active_prediction = logits[:, 1:].argmax(dim=1) + 1
        return torch.where(
            inactive_probability >= float(checkpoint.get("gate_threshold", 0.5)),
            torch.zeros_like(active_prediction),
            active_prediction,
        )

    if selected_output == "log_kpi":
        prediction_std = output["log_kpi_std"].detach().cpu().numpy()
        log_values = (
            prediction_std * float(checkpoint["log_kpi_std"])
            + float(checkpoint["log_kpi_mean"])
        )
        values = np.maximum(
            np.power(10.0, np.clip(log_values, -30.0, 30.0))
            - float(checkpoint["epsilon"]),
            0.0,
        )
        prediction = np.searchsorted(
            np.asarray(checkpoint["cuts"], dtype=np.float64),
            values,
            side="left",
        ).astype(np.int64)
        class_prediction = logits.argmax(dim=1).detach().cpu().numpy()
        prediction[class_prediction == 5] = 5
        return torch.tensor(prediction, dtype=torch.long, device=logits.device)

    raise ValueError(f"Unsupported selected_output in checkpoint: {selected_output!r}")


def predict_pair_aware(
    *,
    model,
    sample_cpu,
    checkpoint: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """Return one final activity class per target for a single scenario graph."""
    prepared = _attach_pair_tensors(sample_cpu, checkpoint)
    batch = Batch.from_data_list([prepared]).to(device)
    with torch.no_grad():
        output = model(batch)
        prediction = _decode(output, checkpoint)
    return prediction.detach().cpu()
