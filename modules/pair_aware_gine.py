# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: pair-aware residual GINE model, losses, and training loop
"""Pair-aware residual GINE model for DYNAGNN.

The GNN is the primary predictor. It receives graph topology and physical
features, explicit target-component identity, contingency identity/location,
and optional operating-point context.

The model uses a hierarchical output:
- a severity head for KPI-derived activity classes 0 .. num_classes-2;
- a binary flag-class protection gate for disconnected or controlled components (class K-1);
- the existing class-0 inactive gate inside the non-flag branch.

This keeps the flag class (K-1) separate from the ordinal severity ladder while learning all
decisions directly from graph, target, event, and operating-point features.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from modules.training_plots import save_training_plots

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GINEConv


def flag_class_index(num_classes: int) -> int:
    """Return the index of the flag (disconnected/controlled) class."""
    return int(num_classes) - 1


@dataclass(frozen=True)
class PairAwareHParams:
    hidden_dim: int
    node_id_dim: int
    contingency_id_dim: int
    type_dim: int
    pair_dim: int
    num_gnn_layers: int
    decoder_hidden_dim: int
    dropout: float
    lr: float
    weight_decay: float


@dataclass(frozen=True)
class PairAwareLossWeights:
    classification: float = 1.0
    regression: float = 0.30
    inactive_gate: float = 0.20
    flag_gate: float = 0.50
    ordinal_cdf: float = 0.10


@dataclass(frozen=True)
class SelectionScoreWeights:
    """Weights for validation protection_selection_score (Optuna / early stopping)."""

    balanced_accuracy: float = 0.30
    macro_f1: float = 0.25
    accuracy: float = 0.15
    within_one: float = 0.10
    flag_recall: float = 0.20


def _safe_global_mean_pool(x: torch.Tensor, batch: torch.Tensor, size: int) -> torch.Tensor:
    rows = []
    for graph_idx in range(int(size)):
        mask = batch == graph_idx
        rows.append(x[mask].mean(dim=0) if bool(mask.any()) else x.new_zeros((x.shape[1],)))
    return torch.stack(rows, dim=0)


def _safe_global_max_pool(x: torch.Tensor, batch: torch.Tensor, size: int) -> torch.Tensor:
    rows = []
    for graph_idx in range(int(size)):
        mask = batch == graph_idx
        rows.append(x[mask].max(dim=0).values if bool(mask.any()) else x.new_zeros((x.shape[1],)))
    return torch.stack(rows, dim=0)


class ResidualGINEBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.conv = GINEConv(mlp, train_eps=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        update = self.conv(x, edge_index, edge_attr)
        return F.relu(self.norm(x + self.dropout(update)))


class PairAwareGINE(nn.Module):
    """Direct event- and target-conditioned GINE model.

    This is intentionally close to the strongest pair-aware residual encoder,
    but removes every historical KPI prior. The output heads predict from the
    graph representation itself.

    ``num_classes`` is the total number of output classes including the flag
    class. Classes ``0 .. num_classes-2`` are KPI-derived activity levels;
    class ``num_classes-1`` is the flag (disconnected/controlled) class.
    """

    def __init__(
        self,
        *,
        num_node_tokens: int,
        num_contingency_tokens: int,
        target_mask_attr: str,
        hparams: PairAwareHParams,
        num_classes: int,
        num_node_types: int = 3,
        num_edge_types: int = 3,
    ) -> None:
        super().__init__()
        h = int(hparams.hidden_dim)
        self.hparams = hparams
        self.target_mask_attr = str(target_mask_attr)
        self.num_classes = int(num_classes)

        self.node_type_embedding = nn.Embedding(num_node_types, hparams.type_dim)
        self.edge_type_embedding = nn.Embedding(num_edge_types, hparams.type_dim)
        self.node_token_embedding = nn.Embedding(num_node_tokens, hparams.node_id_dim)
        self.contingency_embedding = nn.Embedding(num_contingency_tokens, hparams.contingency_id_dim)
        self.event_type_embedding = nn.Embedding(2, hparams.type_dim)

        # Node continuous input: v, angle, p, q, fault_on, electrical distance.
        self.node_input = nn.Sequential(
            nn.Linear(hparams.type_dim + hparams.node_id_dim + 6, h),
            nn.ReLU(),
            nn.LayerNorm(h),
        )
        # Edge continuous input: fault_on, r, x, b1, g1, b2, g2.
        self.edge_input = nn.Sequential(
            nn.Linear(hparams.type_dim + 7, h),
            nn.ReLU(),
            nn.LayerNorm(h),
        )

        self.blocks = nn.ModuleList(
            [ResidualGINEBlock(h, hparams.dropout) for _ in range(hparams.num_gnn_layers)]
        )
        self.jumping_projection = nn.Sequential(
            nn.Linear(h * (hparams.num_gnn_layers + 1), h),
            nn.ReLU(),
            nn.LayerNorm(h),
        )

        self.event_encoder = nn.Sequential(
            nn.Linear(h + h + hparams.type_dim + hparams.contingency_id_dim, h),
            nn.ReLU(),
            nn.Dropout(hparams.dropout),
            nn.Linear(h, h),
            nn.ReLU(),
        )
        self.event_to_node = nn.Linear(h, h)

        self.target_pair_projection = nn.Linear(hparams.node_id_dim, hparams.pair_dim)
        self.contingency_pair_projection = nn.Linear(hparams.contingency_id_dim, hparams.pair_dim)

        # target h + event h + global(mean|max)=2h + |h-e| + h*e + dz
        # + explicit target ID + explicit contingency ID + pair interaction
        decoder_in = (
            h * 6
            + 1
            + hparams.node_id_dim
            + hparams.contingency_id_dim
            + hparams.pair_dim
        )
        self.shared_decoder = nn.Sequential(
            nn.Linear(decoder_in, hparams.decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(hparams.dropout),
            nn.Linear(hparams.decoder_hidden_dim, hparams.decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(hparams.dropout),
        )
        self.class_head = nn.Linear(
            hparams.decoder_hidden_dim, self.num_classes - 1
        )
        self.flag_head = nn.Linear(hparams.decoder_hidden_dim, 1)
        self.inactive_head = nn.Linear(hparams.decoder_hidden_dim, 1)
        self.regression_head = nn.Linear(hparams.decoder_hidden_dim, 1)

    def forward(self, data) -> dict[str, torch.Tensor]:
        x = data.x
        edge_attr = data.edge_attr
        edge_index = data.edge_index
        batch = data.batch.to(x.device)
        num_graphs = int(data.num_graphs)

        node_type = x[:, 0].long().clamp(min=0, max=self.node_type_embedding.num_embeddings - 1)
        node_token = data.node_token.long().clamp(min=0, max=self.node_token_embedding.num_embeddings - 1)
        node_id_embedding = self.node_token_embedding(node_token)
        node_cont = torch.cat([x[:, 1:5], x[:, 5:6], x[:, 6:7]], dim=1)
        h = self.node_input(
            torch.cat(
                [self.node_type_embedding(node_type), node_id_embedding, node_cont],
                dim=1,
            )
        )

        edge_type = edge_attr[:, 0].long().clamp(min=0, max=self.edge_type_embedding.num_embeddings - 1)
        edge_cont = edge_attr[:, 1:8]
        e = self.edge_input(torch.cat([self.edge_type_embedding(edge_type), edge_cont], dim=1))

        states = [h]
        for block in self.blocks:
            h = block(h, edge_index, e)
            states.append(h)
        h = self.jumping_projection(torch.cat(states, dim=1))

        graph_mean = _safe_global_mean_pool(h, batch, size=num_graphs)
        graph_max = _safe_global_max_pool(h, batch, size=num_graphs)
        graph_context = torch.cat([graph_mean, graph_max], dim=1)

        event_node_mask = data.event_node_mask.bool()
        if not bool(event_node_mask.any()):
            raise RuntimeError("A batch contains no event anchor nodes")
        event_node_pool = _safe_global_mean_pool(
            h[event_node_mask], batch[event_node_mask], size=num_graphs
        )

        event_edge_mask = data.event_edge_mask.bool()
        if bool(event_edge_mask.any()):
            edge_batch = batch[edge_index[0]]
            event_edge_pool = _safe_global_mean_pool(
                e[event_edge_mask], edge_batch[event_edge_mask], size=num_graphs
            )
        else:
            event_edge_pool = h.new_zeros((num_graphs, h.shape[1]))

        event_type = data.event_graph_type.view(-1).long().clamp(min=0, max=1)
        contingency_token = data.contingency_token.view(-1).long().clamp(
            min=0, max=self.contingency_embedding.num_embeddings - 1
        )
        contingency_embedding = self.contingency_embedding(contingency_token)
        event_context = self.event_encoder(
            torch.cat(
                [
                    event_node_pool,
                    event_edge_pool,
                    self.event_type_embedding(event_type),
                    contingency_embedding,
                ],
                dim=1,
            )
        )

        target_mask = getattr(data, self.target_mask_attr).bool()
        target_batch = batch[target_mask]
        target_h = h[target_mask]
        target_event = event_context[target_batch]
        target_global = graph_context[target_batch]
        event_as_node = self.event_to_node(target_event)
        dz = x[target_mask, 6:7]

        target_node_id = node_id_embedding[target_mask]
        target_contingency = contingency_embedding[target_batch]
        pair_interaction = torch.tanh(
            self.target_pair_projection(target_node_id)
            * self.contingency_pair_projection(target_contingency)
        )

        parts = [
            target_h,
            target_event,
            target_global,
            torch.abs(target_h - event_as_node),
            target_h * event_as_node,
            dz,
            target_node_id,
            target_contingency,
            pair_interaction,
        ]
        shared = self.shared_decoder(torch.cat(parts, dim=1))
        return {
            "class_logits": self.class_head(shared),
            "flag_logit": self.flag_head(shared).squeeze(1),
            "inactive_logit": self.inactive_head(shared).squeeze(1),
            "log_kpi_std": self.regression_head(shared).squeeze(1),
        }


def _safe_div(numerator: float | int, denominator: float | int) -> float:
    return float(numerator) / float(denominator) if float(denominator) > 0 else 0.0


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    """Compute square confusion-matrix metrics and exact ordinal offsets.

    ``num_classes`` is the total number of output classes (KPI activity classes
    plus the flag class). It may also be used for an activity-only subset, which
    correctly counts a predicted flag class as an error while excluding absent
    true classes from balanced/macro averages.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch y_true={y_true.shape}, y_pred={y_pred.shape}")
    if y_true.size and (
        y_true.min() < 0
        or y_pred.min() < 0
        or y_true.max() >= int(num_classes)
        or y_pred.max() >= int(num_classes)
    ):
        raise ValueError(
            f"Labels outside 0..{int(num_classes)-1}: "
            f"true=[{y_true.min()},{y_true.max()}] pred=[{y_pred.min()},{y_pred.max()}]"
        )

    if y_true.size == 0:
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    else:
        flat = y_true * int(num_classes) + y_pred
        confusion = np.bincount(flat, minlength=num_classes * num_classes).reshape(
            num_classes, num_classes
        )

    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    tp = np.diag(confusion).astype(np.float64)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    present = support > 0
    total = int(y_true.size)
    diff = y_pred - y_true
    exact = int((diff == 0).sum())
    under = int((diff < 0).sum())
    over = int((diff > 0).sum())
    abs_diff = np.abs(diff)
    weighted_f1 = _safe_div(float((f1 * support).sum()), float(support.sum()))

    offsets = list(range(-(int(num_classes) - 1), int(num_classes)))
    offset_count = {str(offset): int((diff == offset).sum()) for offset in offsets}
    offset_rate = {str(offset): _safe_div(offset_count[str(offset)], total) for offset in offsets}

    metrics = {
        "total": total,
        "correct": exact,
        "accuracy": _safe_div(exact, total),
        "balanced_accuracy": float(recall[present].mean()) if present.any() else 0.0,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "weighted_f1": weighted_f1,
        "within_one_accuracy": _safe_div(int((abs_diff <= 1).sum()), total),
        "mae": float(abs_diff.mean()) if total else 0.0,
        "rmse": float(np.sqrt(np.mean(diff.astype(np.float64) ** 2))) if total else 0.0,
        "mean_signed_error": float(diff.mean()) if total else 0.0,
        "under_count": under,
        "under_rate": _safe_div(under, total),
        "over_count": over,
        "over_rate": _safe_div(over, total),
        "under_gt1_count": int((diff < -1).sum()),
        "under_gt1_rate": _safe_div(int((diff < -1).sum()), total),
        "over_gt1_count": int((diff > 1).sum()),
        "over_gt1_rate": _safe_div(int((diff > 1).sum()), total),
        "under_gt2_count": int((diff < -2).sum()),
        "over_gt2_count": int((diff > 2).sum()),
        "class_correct": tp.astype(np.int64).tolist(),
        "class_accuracy": recall.tolist(),
        "class_precision": precision.tolist(),
        "class_recall": recall.tolist(),
        "class_f1": f1.tolist(),
        "class_support": support.astype(np.int64).tolist(),
        "class_predicted": predicted.astype(np.int64).tolist(),
        "error_offset_count": offset_count,
        "error_offset_rate": offset_rate,
        "confusion_matrix": confusion.tolist(),
    }
    metrics["selection_score"] = selection_score(metrics)
    return metrics

def selection_score(metrics: dict) -> float:
    """Legacy reporting score (not used for Optuna / early stopping)."""
    return float(
        0.40 * metrics["balanced_accuracy"]
        + 0.30 * metrics["macro_f1"]
        + 0.20 * metrics["accuracy"]
        + 0.10 * metrics["within_one_accuracy"]
    )


def protection_selection_score(
    metrics: dict,
    weights: SelectionScoreWeights,
) -> float:
    """Validation score used for Optuna trials and early stopping."""
    return float(
        float(weights.balanced_accuracy) * metrics["balanced_accuracy"]
        + float(weights.macro_f1) * metrics["macro_f1"]
        + float(weights.accuracy) * metrics["accuracy"]
        + float(weights.within_one) * metrics["within_one_accuracy"]
        + float(weights.flag_recall) * float(metrics["flag_recall"])
    )


def _ordinal_cdf_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    severity_classes: int,
) -> torch.Tensor:
    """Ordinal CDF loss for KPI severity classes only, excluding the flag class."""
    probabilities = torch.softmax(logits, dim=1)
    pred_cdf = probabilities.cumsum(dim=1)[:, :-1]
    true_cdf = (
        F.one_hot(y, num_classes=int(severity_classes))
        .float()
        .cumsum(dim=1)[:, :-1]
    )
    return torch.abs(pred_cdf - true_cdf).mean()


def _classes_from_log_prediction(
    prediction_std: np.ndarray,
    *,
    log_mean: float,
    log_std: float,
    cuts: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    log_values = prediction_std * float(log_std) + float(log_mean)
    values = np.maximum(
        np.power(10.0, np.clip(log_values, -30.0, 30.0)) - float(epsilon),
        0.0,
    )
    return np.searchsorted(cuts, values, side="left").astype(np.int64)


def _move_batch(data, device: torch.device):
    tensor_keys = [
        "x",
        "batch",
        "edge_index",
        "edge_attr",
        "y_class",
        "bus_node_mask",
        "gen_node_mask",
        "node_token",
        "contingency_token",
        "event_node_mask",
        "event_edge_mask",
        "event_graph_type",
        "y_log_kpi_std",
    ]
    keys = [key for key in tensor_keys if hasattr(data, key)]
    return data.to(device, *keys)


def _decode_flag(
    severity_logits: torch.Tensor,
    flag_logit: torch.Tensor,
    flag_threshold: float,
    flag_class: int,
) -> torch.Tensor:
    severity_pred = severity_logits.argmax(dim=1)
    flag_probability = torch.sigmoid(flag_logit)
    return torch.where(
        flag_probability >= float(flag_threshold),
        torch.full_like(severity_pred, int(flag_class)),
        severity_pred,
    )


def _decode_gated(
    severity_logits: torch.Tensor,
    inactive_logit: torch.Tensor,
    flag_logit: torch.Tensor,
    inactive_threshold: float,
    flag_threshold: float,
    flag_class: int,
) -> torch.Tensor:
    flag_probability = torch.sigmoid(flag_logit)
    inactive_probability = torch.sigmoid(inactive_logit)
    active_pred = severity_logits[:, 1:].argmax(dim=1) + 1
    severity_pred = torch.where(
        inactive_probability >= float(inactive_threshold),
        torch.zeros_like(active_pred),
        active_pred,
    )
    return torch.where(
        flag_probability >= float(flag_threshold),
        torch.full_like(severity_pred, int(flag_class)),
        severity_pred,
    )


def _string_list(value, num_graphs: int) -> list[str]:
    if isinstance(value, (list, tuple)):
        result = [str(v) for v in value]
    else:
        result = [str(value)]
    if len(result) == 1 and num_graphs > 1:
        result = result * num_graphs
    if len(result) != num_graphs:
        result = (result + [""] * num_graphs)[:num_graphs]
    return result


def _run_epoch(
    *,
    model: nn.Module,
    loader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    class_weights: Optional[torch.Tensor],
    inactive_pos_weight: Optional[torch.Tensor],
    flag_pos_weight: Optional[torch.Tensor],
    loss_weights: PairAwareLossWeights,
    log_mean: float,
    log_std: float,
    cuts: np.ndarray,
    epsilon: float,
    gate_threshold: float,
    flag_gate_threshold: float,
    target_mask_attr: str,
    num_classes: int,
    selection_score_weights: SelectionScoreWeights,
    collect_predictions: bool = False,
) -> dict:
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    flag_cls = flag_class_index(num_classes)
    severity_classes = flag_cls
    sums = {
        "total": 0.0,
        "classification": 0.0,
        "regression": 0.0,
        "gate": 0.0,
        "flag_gate": 0.0,
        "ordinal": 0.0,
    }
    n_supervised = 0

    full_true_chunks: list[np.ndarray] = []
    full_class_chunks: list[np.ndarray] = []
    full_gated_chunks: list[np.ndarray] = []
    full_log_chunks: list[np.ndarray] = []
    activity_true_chunks: list[np.ndarray] = []
    activity_class_chunks: list[np.ndarray] = []
    activity_gated_chunks: list[np.ndarray] = []
    activity_log_chunks: list[np.ndarray] = []
    prediction_rows: list[dict] = []

    with torch.set_grad_enabled(train_mode):
        for data in loader:
            data = _move_batch(data, device)
            output = model(data)
            target_mask = getattr(data, target_mask_attr).bool()
            y_all = data.y_class[target_mask].long()
            log_target_all = data.y_log_kpi_std[target_mask]

            valid_mask = (y_all >= 0) & (y_all < num_classes)
            if not bool(valid_mask.any()):
                continue

            severity_logits = output["class_logits"][valid_mask]
            flag_logit = output["flag_logit"][valid_mask]
            inactive_logit = output["inactive_logit"][valid_mask]
            reg_prediction = output["log_kpi_std"][valid_mask]
            y = y_all[valid_mask]
            log_target = log_target_all[valid_mask]

            flag_target = (y == flag_cls).float()
            flag_loss = F.binary_cross_entropy_with_logits(
                flag_logit,
                flag_target,
                pos_weight=flag_pos_weight,
            )

            severity_mask = y < flag_cls
            if bool(severity_mask.any()):
                severity_y = y[severity_mask]
                classification_loss = F.cross_entropy(
                    severity_logits[severity_mask],
                    severity_y,
                    weight=class_weights,
                )
                gate_target = (severity_y == 0).float()
                gate_loss = F.binary_cross_entropy_with_logits(
                    inactive_logit[severity_mask],
                    gate_target,
                    pos_weight=inactive_pos_weight,
                )
                ordinal_loss = _ordinal_cdf_loss(
                    severity_logits[severity_mask],
                    severity_y,
                    severity_classes,
                )
            else:
                classification_loss = severity_logits.new_zeros(())
                gate_loss = severity_logits.new_zeros(())
                ordinal_loss = severity_logits.new_zeros(())

            finite_reg = severity_mask & torch.isfinite(log_target)
            regression_loss = (
                F.smooth_l1_loss(
                    reg_prediction[finite_reg], log_target[finite_reg]
                )
                if bool(finite_reg.any())
                else severity_logits.new_zeros(())
            )

            total_loss = (
                loss_weights.classification * classification_loss
                + loss_weights.regression * regression_loss
                + loss_weights.inactive_gate * gate_loss
                + loss_weights.flag_gate * flag_loss
                + loss_weights.ordinal_cdf * ordinal_loss
            )

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            n = int(y.numel())
            n_supervised += n
            sums["total"] += float(total_loss.detach().cpu()) * n
            sums["classification"] += float(classification_loss.detach().cpu()) * n
            sums["regression"] += float(regression_loss.detach().cpu()) * n
            sums["gate"] += float(gate_loss.detach().cpu()) * n
            sums["flag_gate"] += float(flag_loss.detach().cpu()) * n
            sums["ordinal"] += float(ordinal_loss.detach().cpu()) * n

            detached_severity = severity_logits.detach()
            detached_flag = flag_logit.detach()
            detached_inactive = inactive_logit.detach()

            class_pred = _decode_flag(
                detached_severity,
                detached_flag,
                flag_gate_threshold,
                flag_cls,
            )
            gated_pred = _decode_gated(
                detached_severity,
                detached_inactive,
                detached_flag,
                gate_threshold,
                flag_gate_threshold,
                flag_cls,
            )

            log_base_np = _classes_from_log_prediction(
                reg_prediction.detach().cpu().numpy(),
                log_mean=log_mean,
                log_std=log_std,
                cuts=cuts,
                epsilon=epsilon,
            )
            flag_probability_np = torch.sigmoid(detached_flag).cpu().numpy()
            log_pred_np = log_base_np.copy()
            log_pred_np[
                flag_probability_np >= float(flag_gate_threshold)
            ] = flag_cls

            y_np = y.detach().cpu().numpy()
            class_pred_np = class_pred.cpu().numpy()
            gated_pred_np = gated_pred.cpu().numpy()
            full_true_chunks.append(y_np)
            full_class_chunks.append(class_pred_np)
            full_gated_chunks.append(gated_pred_np)
            full_log_chunks.append(log_pred_np)

            activity_np = y_np < flag_cls
            if bool(activity_np.any()):
                activity_true_chunks.append(y_np[activity_np])
                activity_class_chunks.append(class_pred_np[activity_np])
                activity_gated_chunks.append(gated_pred_np[activity_np])
                activity_log_chunks.append(log_pred_np[activity_np])

            if collect_predictions:
                num_graphs = int(data.num_graphs)
                ops = _string_list(getattr(data, "op_name", ""), num_graphs)
                events = _string_list(getattr(data, "event_id", ""), num_graphs)
                target_graph_all = data.batch[target_mask].detach().cpu().numpy()
                target_token_all = data.node_token[target_mask].detach().cpu().numpy()
                valid_indices = np.flatnonzero(valid_mask.detach().cpu().numpy())
                severity_probabilities = torch.softmax(
                    detached_severity, dim=1
                ).cpu().numpy()
                flag_probabilities = torch.sigmoid(
                    detached_flag
                ).cpu().numpy()
                inactive_probabilities = torch.sigmoid(
                    detached_inactive
                ).cpu().numpy()
                severity_class_np = detached_severity.argmax(dim=1).cpu().numpy()
                active_severity_np = (
                    detached_severity[:, 1:].argmax(dim=1).cpu().numpy() + 1
                )
                gated_base_np = np.where(
                    inactive_probabilities >= float(gate_threshold),
                    0,
                    active_severity_np,
                ).astype(np.int64)

                for local_idx, target_idx in enumerate(valid_indices):
                    graph_idx = int(target_graph_all[target_idx])
                    true_class = int(y_np[local_idx])
                    class_value = int(class_pred_np[local_idx])
                    gated_value = int(gated_pred_np[local_idx])
                    log_value = int(log_pred_np[local_idx])
                    row = {
                        "operating_point": ops[graph_idx],
                        "contingency": events[graph_idx],
                        "node_token": int(target_token_all[target_idx]),
                        "true_class": true_class,
                        "flag_probability": float(
                            flag_probabilities[local_idx]
                        ),
                        "inactive_probability": float(
                            inactive_probabilities[local_idx]
                        ),
                        "base_class_prediction": int(
                            severity_class_np[local_idx]
                        ),
                        "base_gated_prediction": int(
                            gated_base_np[local_idx]
                        ),
                        "base_log_kpi_prediction": int(
                            log_base_np[local_idx]
                        ),
                        "class_prediction": class_value,
                        "gated_prediction": gated_value,
                        "log_kpi_prediction": log_value,
                        "class_error_offset": class_value - true_class,
                        "gated_error_offset": gated_value - true_class,
                        "log_kpi_error_offset": log_value - true_class,
                    }
                    for cls in range(severity_classes):
                        row[f"class_probability_{cls}"] = float(
                            severity_probabilities[local_idx, cls]
                        )
                    row[f"class_probability_{flag_cls}"] = float(
                        flag_probabilities[local_idx]
                    )
                    prediction_rows.append(row)

    def _cat(chunks: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(chunks) if chunks else np.empty((0,), dtype=np.int64)

    full_true = _cat(full_true_chunks)
    activity_true = _cat(activity_true_chunks)
    result = {
        "loss": {key: _safe_div(value, n_supervised) for key, value in sums.items()},
        "full_class_class": classification_metrics(
            full_true, _cat(full_class_chunks), num_classes
        ),
        "full_class_gated": classification_metrics(
            full_true, _cat(full_gated_chunks), num_classes
        ),
        "full_class_log_kpi": classification_metrics(
            full_true, _cat(full_log_chunks), num_classes
        ),
        "activity_class": classification_metrics(
            activity_true, _cat(activity_class_chunks), num_classes
        ),
        "activity_gated": classification_metrics(
            activity_true, _cat(activity_gated_chunks), num_classes
        ),
        "activity_log_kpi": classification_metrics(
            activity_true, _cat(activity_log_chunks), num_classes
        ),
    }
    for key in ("full_class_class", "full_class_gated", "full_class_log_kpi"):
        metrics = result[key]
        flag_recall = float(metrics["class_recall"][flag_cls])
        flag_precision = float(metrics["class_precision"][flag_cls])
        flag_f1 = float(metrics["class_f1"][flag_cls])
        metrics["flag_recall"] = flag_recall
        metrics["flag_precision"] = flag_precision
        metrics["flag_f1"] = flag_f1
        metrics["protection_selection_score"] = protection_selection_score(
            metrics, selection_score_weights
        )

    result["combined_class"] = result["full_class_class"]
    result["combined_gated"] = result["full_class_gated"]
    result["combined_log_kpi"] = result["full_class_log_kpi"]
    if collect_predictions:
        result["predictions"] = prediction_rows
    return result


def compute_class_weights(
    loader,
    mode: str,
    device: torch.device,
    target_mask_attr: str,
    num_classes: int,
) -> Optional[torch.Tensor]:
    """Weights for severity classes 0..num_classes-2."""
    mode = str(mode).strip().lower()
    if mode == "none":
        return None
    severity_classes = flag_class_index(num_classes)
    counts = torch.zeros(severity_classes, dtype=torch.float64)
    for data in loader:
        y = data.y_class[getattr(data, target_mask_attr)]
        y = y[(y >= 0) & (y < severity_classes)]
        counts += torch.bincount(y, minlength=severity_classes).double()
    inverse = counts.sum() / counts.clamp_min(1.0)
    if mode == "sqrt_inverse":
        weights = torch.sqrt(inverse)
    elif mode == "inverse":
        weights = inverse
    else:
        raise ValueError(
            f"Unsupported class_weight_mode={mode!r}; expected none, "
            "sqrt_inverse, or inverse"
        )
    weights = weights / weights.mean()
    return weights.float().to(device)


def compute_gate_pos_weight(
    loader,
    mode: str,
    device: torch.device,
    target_mask_attr: str,
    num_classes: int,
) -> Optional[torch.Tensor]:
    """Positive weight for class 0 versus classes 1..4, excluding the flag class."""
    mode = str(mode).strip().lower()
    if mode == "none":
        return None
    if mode != "balanced":
        raise ValueError(
            f"Unsupported gate_pos_weight_mode={mode!r}; expected none or balanced"
        )
    flag_cls = flag_class_index(num_classes)
    inactive = 0
    active = 0
    for data in loader:
        y = data.y_class[getattr(data, target_mask_attr)]
        y = y[(y >= 0) & (y < flag_cls)]
        inactive += int((y == 0).sum().item())
        active += int((y != 0).sum().item())
    return torch.tensor(
        [_safe_div(active, max(inactive, 1))],
        dtype=torch.float32,
        device=device,
    )


def compute_flag_pos_weight(
    loader,
    mode: str,
    device: torch.device,
    target_mask_attr: str,
    num_classes: int,
    multiplier: float = 1.0,
) -> Optional[torch.Tensor]:
    """Positive weight for the flag class (K-1) versus severity classes 0..K-2."""
    mode = str(mode).strip().lower()
    if mode == "none":
        return None
    if mode != "balanced":
        raise ValueError(
            f"Unsupported flag_gate_pos_weight_mode={mode!r}; "
            "expected none or balanced"
        )
    flag_cls = flag_class_index(num_classes)
    positive = 0
    negative = 0
    for data in loader:
        y = data.y_class[getattr(data, target_mask_attr)]
        y = y[(y >= 0) & (y < num_classes)]
        positive += int((y == flag_cls).sum().item())
        negative += int((y != flag_cls).sum().item())
    value = _safe_div(negative, max(positive, 1)) * float(multiplier)
    return torch.tensor([value], dtype=torch.float32, device=device)


def _metrics_frame(metrics: dict) -> pd.DataFrame:
    classes = range(len(metrics["class_support"]))
    return pd.DataFrame(
        {
            "class": list(classes),
            "support": metrics["class_support"],
            "correct": metrics["class_correct"],
            "class_accuracy": metrics["class_accuracy"],
            "predicted": metrics["class_predicted"],
            "precision": metrics["class_precision"],
            "recall": metrics["class_recall"],
            "f1": metrics["class_f1"],
        }
    )


def _error_offset_frame(metrics: dict) -> pd.DataFrame:
    rows = []
    for key, count in sorted(metrics["error_offset_count"].items(), key=lambda item: int(item[0])):
        offset = int(key)
        rows.append(
            {
                "error_offset_pred_minus_true": offset,
                "label": f"{offset:+d}" if offset != 0 else "0 (exact)",
                "count": int(count),
                "rate": float(metrics["error_offset_rate"][key]),
            }
        )
    return pd.DataFrame(rows)


def _error_offset_by_true_class(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> pd.DataFrame:
    rows = []
    diff = y_pred - y_true
    offsets = range(-(num_classes - 1), num_classes)
    for cls in range(num_classes):
        mask = y_true == cls
        total = int(mask.sum())
        row = {"true_class": cls, "total": total}
        for offset in offsets:
            count = int(((diff == offset) & mask).sum())
            row[f"offset_{offset:+d}_count"] = count
            row[f"offset_{offset:+d}_rate"] = _safe_div(count, total)
        rows.append(row)
    return pd.DataFrame(rows)

def _under_over_by_true_class(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> pd.DataFrame:
    rows = []
    diff = y_pred - y_true
    for cls in range(num_classes):
        mask = y_true == cls
        total = int(mask.sum())
        rows.append(
            {
                "true_class": cls,
                "total": total,
                "exact": int(((diff == 0) & mask).sum()),
                "under": int(((diff < 0) & mask).sum()),
                "over": int(((diff > 0) & mask).sum()),
                "under_rate": _safe_div(int(((diff < 0) & mask).sum()), total),
                "over_rate": _safe_div(int(((diff > 0) & mask).sum()), total),
                "mean_signed_error": float(diff[mask].mean()) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _save_test_outputs(output_dir: Path, result: dict, selected_output: str, num_classes: int) -> None:
    test_dir = output_dir / "test_outputs"
    test_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(result.get("predictions", []))
    predictions.to_csv(test_dir / "test_predictions.csv", index=False)

    full_key = f"full_class_{selected_output}"
    activity_key = f"activity_{selected_output}"
    full = result[full_key]
    activity = result[activity_key]
    (test_dir / "test_metrics.json").write_text(
        json.dumps(
            {
                "selected_output": selected_output,
                "full_class": full,
                "activity_subset": activity,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _metrics_frame(full).to_csv(test_dir / "test_per_class_metrics.csv", index=False)
    _metrics_frame(activity).to_csv(
        test_dir / "test_activity_subset_per_class_metrics.csv", index=False
    )
    pd.DataFrame(full["confusion_matrix"]).to_csv(
        test_dir / "test_confusion_matrix.csv", index=False
    )
    _error_offset_frame(full).to_csv(
        test_dir / "test_error_offset_distribution.csv", index=False
    )

    if not predictions.empty:
        pred_col = {
            "class": "class_prediction",
            "gated": "gated_prediction",
            "log_kpi": "log_kpi_prediction",
        }[selected_output]
        predictions["selected_prediction"] = predictions[pred_col].astype(int)
        predictions["selected_error_offset"] = (
            predictions["selected_prediction"] - predictions["true_class"]
        )
        predictions.to_csv(test_dir / "test_predictions.csv", index=False)

        y_true = predictions["true_class"].to_numpy(dtype=np.int64)
        y_pred = predictions["selected_prediction"].to_numpy(dtype=np.int64)
        _under_over_by_true_class(y_true, y_pred, num_classes).to_csv(
            test_dir / "test_under_over_by_true_class.csv", index=False
        )
        _error_offset_by_true_class(y_true, y_pred, num_classes).to_csv(
            test_dir / "test_error_offset_by_true_class.csv", index=False
        )
        pd.DataFrame(
            [
                {
                    "exact": full["correct"],
                    "total": full["total"],
                    "under_count": full["under_count"],
                    "under_rate": full["under_rate"],
                    "over_count": full["over_count"],
                    "over_rate": full["over_rate"],
                    "under_gt1_count": full["under_gt1_count"],
                    "under_gt1_rate": full["under_gt1_rate"],
                    "over_gt1_count": full["over_gt1_count"],
                    "over_gt1_rate": full["over_gt1_rate"],
                    "within_one_accuracy": full["within_one_accuracy"],
                    "mae": full["mae"],
                    "rmse": full["rmse"],
                    "mean_signed_error": full["mean_signed_error"],
                }
            ]
        ).to_csv(test_dir / "test_under_over_summary.csv", index=False)


def _log_test_breakdown(
    logger: logging.Logger, metrics: dict, *, selected_output: str
) -> None:
    num_classes = len(metrics["class_support"])
    logger.info("---------- TEST RESULTS (decode=%s) ----------", selected_output)
    logger.info(
        "Summary: %d/%d correct | acc=%.4f | bal_acc=%.4f | macro_F1=%.4f | MAE=%.4f",
        metrics["correct"],
        metrics["total"],
        metrics["accuracy"],
        metrics["balanced_accuracy"],
        metrics["macro_f1"],
        metrics["mae"],
    )
    per_class = []
    for cls in range(num_classes):
        support = int(metrics["class_support"][cls])
        correct = int(metrics["class_correct"][cls])
        recall = float(metrics["class_accuracy"][cls])
        per_class.append(f"{cls}:{correct}/{support} ({recall:.3f})")
    logger.info("Per-class recall: %s", "  ".join(per_class))
    flag_cls = num_classes - 1
    if int(metrics["class_support"][flag_cls]) > 0:
        logger.info(
            "Flag-class protection: recall=%.4f precision=%.4f F1=%.4f",
            float(metrics["class_recall"][flag_cls]),
            float(metrics["class_precision"][flag_cls]),
            float(metrics["class_f1"][flag_cls]),
        )

    # Compact offset summary: exact / within-1 / under / over
    logger.info(
        "Offsets: exact=%.3f | within_1=%.3f | under=%.3f | over=%.3f",
        float(metrics["error_offset_rate"].get("0", 0.0)),
        float(metrics["within_one_accuracy"]),
        float(metrics["under_rate"]),
        float(metrics["over_rate"]),
    )
    # Full offset table kept for inspection (short lines)
    offset_bits = []
    for offset in range(-(num_classes - 1), num_classes):
        key = str(offset)
        rate = float(metrics["error_offset_rate"][key])
        if rate <= 0.0:
            continue
        label = f"{offset:+d}" if offset != 0 else "0"
        offset_bits.append(f"{label}={rate:.3f}")
    if offset_bits:
        logger.info("Non-zero offset rates: %s", "  ".join(offset_bits))

def _save_validation_outputs(
    artifact_dir: Path,
    result: dict,
    selected_output: str,
    num_classes: int,
    flag_gate_threshold: float,
) -> Path:
    validation_dir = artifact_dir / "validation_outputs"
    validation_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(result.get("predictions", []))
    path = validation_dir / "validation_predictions.csv"
    predictions.to_csv(path, index=False)
    metrics = result[f"full_class_{selected_output}"]
    (validation_dir / "validation_metrics_at_training_threshold.json").write_text(
        json.dumps(
            {
                "selected_output": selected_output,
                "flag_gate_threshold": float(flag_gate_threshold),
                "full_class": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def run_pair_aware_training(
    *,
    task: str,
    target_mask_attr: str,
    train_loader,
    validation_loader,
    test_loader,
    output_dir: Path,
    num_node_tokens: int,
    num_contingency_tokens: int,
    hparams: PairAwareHParams,
    loss_weights: PairAwareLossWeights,
    log_mean: float,
    log_std: float,
    cuts: np.ndarray,
    epsilon: float,
    gate_threshold: float,
    flag_gate_threshold: float,
    epochs: int,
    patience: int,
    fixed_epochs: Optional[int],
    selection_output: Optional[str],
    class_weight_mode: str,
    gate_pos_weight_mode: str,
    flag_gate_pos_weight_mode: str,
    flag_pos_weight_multiplier: float,
    selection_score_weights: SelectionScoreWeights,
    num_classes: int,
    logger: logging.Logger,
    trial=None,
) -> dict:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = PairAwareGINE(
        num_node_tokens=num_node_tokens,
        num_contingency_tokens=num_contingency_tokens,
        target_mask_attr=target_mask_attr,
        hparams=hparams,
        num_classes=num_classes,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4, min_lr=1.0e-6
    )
    class_weights = compute_class_weights(
        train_loader, class_weight_mode, device, target_mask_attr, num_classes
    )
    inactive_pos_weight = compute_gate_pos_weight(
        train_loader, gate_pos_weight_mode, device, target_mask_attr, num_classes
    )
    flag_pos_weight = compute_flag_pos_weight(
        train_loader,
        flag_gate_pos_weight_mode,
        device,
        target_mask_attr,
        num_classes,
        multiplier=flag_pos_weight_multiplier,
    )

    trial_label = None if trial is None else int(trial.number)
    if trial_label is None:
        logger.info("Device: %s | num_classes=%d", device, num_classes)
        logger.info("HParams: %s", asdict(hparams))
        logger.info("Loss weights: %s", asdict(loss_weights))
        logger.info(
            "Class weights: %s",
            None if class_weights is None else [round(float(v), 4) for v in class_weights.detach().cpu().tolist()],
        )
        logger.info(
            "Inactive gate pos_weight: %s",
            None if inactive_pos_weight is None else [round(float(v), 4) for v in inactive_pos_weight.detach().cpu().tolist()],
        )
        logger.info(
            "Flag-gate pos_weight: %s | threshold=%.3f",
            None if flag_pos_weight is None else [round(float(v), 4) for v in flag_pos_weight.detach().cpu().tolist()],
            float(flag_gate_threshold),
        )
        logger.info("Selection score weights: %s", asdict(selection_score_weights))
    else:
        logger.info(
            "--- %s trial %d | %s ---",
            task,
            trial_label,
            ", ".join(f"{k}={v}" for k, v in asdict(hparams).items()),
        )

    history: list[dict] = []
    best_score = -float("inf")
    best_epoch = 0
    best_output = selection_output or "class"
    best_state: Optional[dict] = None
    stale = 0
    total_epochs = int(fixed_epochs) if fixed_epochs is not None else int(epochs)
    stopped_early = False
    pruned = False

    for epoch in range(1, total_epochs + 1):
        train_result = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            class_weights=class_weights,
            inactive_pos_weight=inactive_pos_weight,
            flag_pos_weight=flag_pos_weight,
            loss_weights=loss_weights,
            log_mean=log_mean,
            log_std=log_std,
            cuts=cuts,
            epsilon=epsilon,
            gate_threshold=gate_threshold,
            flag_gate_threshold=flag_gate_threshold,
            target_mask_attr=target_mask_attr,
            selection_score_weights=selection_score_weights,
            num_classes=num_classes,
        )
        row = {
            "epoch": epoch,
            **{f"train_{k}_loss": v for k, v in train_result["loss"].items()},
        }

        if validation_loader is not None:
            val_result = _run_epoch(
                model=model,
                loader=validation_loader,
                device=device,
                optimizer=None,
                class_weights=class_weights,
                inactive_pos_weight=inactive_pos_weight,
                flag_pos_weight=flag_pos_weight,
                loss_weights=loss_weights,
                log_mean=log_mean,
                log_std=log_std,
                cuts=cuts,
                epsilon=epsilon,
                gate_threshold=gate_threshold,
                flag_gate_threshold=flag_gate_threshold,
                target_mask_attr=target_mask_attr,
                selection_score_weights=selection_score_weights,
                num_classes=num_classes,
            )
            candidates = {
                "class": val_result["full_class_class"]["protection_selection_score"],
                "gated": val_result["full_class_gated"]["protection_selection_score"],
                "log_kpi": val_result["full_class_log_kpi"]["protection_selection_score"],
            }
            selected = selection_output or max(candidates, key=candidates.get)
            if selected not in candidates:
                raise ValueError(f"Unsupported selection_output: {selected!r}")
            score = float(candidates[selected])
            for key, value in val_result["loss"].items():
                row[f"val_{key}_loss"] = float(value)
            for name, value in candidates.items():
                row[f"val_{name}_score"] = float(value)
            row["val_selected_output"] = selected
            row["val_selected_score"] = score
            scheduler.step(score)

            improved = False
            if fixed_epochs is None and score > best_score + 1.0e-6:
                best_score = score
                best_epoch = epoch
                best_output = selected
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale = 0
                improved = True
            elif fixed_epochs is None:
                stale += 1

            marker = "*" if improved else " "
            selected_metrics = val_result[f"full_class_{selected}"]
            logger.info(
                "Epoch %03d %s | train=%.4f | val_loss=%.4f | val class=%.4f gated=%.4f logKPI=%.4f | selected=%s %.4f | flag R/P/F1=%.3f/%.3f/%.3f",
                epoch,
                marker,
                train_result["loss"]["total"],
                float(val_result["loss"]["total"]),
                float(candidates["class"]),
                float(candidates["gated"]),
                float(candidates["log_kpi"]),
                selected,
                score,
                float(selected_metrics["flag_recall"]),
                float(selected_metrics["flag_precision"]),
                float(selected_metrics["flag_f1"]),
            )
            history.append(row)

            if trial is not None:
                trial.report(score, step=epoch)
                if trial.should_prune():
                    import optuna
                    pruned = True
                    logger.info(
                        "Trial %d pruned at epoch %d (val_score=%.4f)",
                        trial_label,
                        epoch,
                        score,
                    )
                    raise optuna.TrialPruned()

            if fixed_epochs is None and stale >= int(patience):
                stopped_early = True
                logger.info(
                    "Early stopping at epoch %d (best_epoch=%d, best_val=%.4f [%s])",
                    epoch,
                    best_epoch,
                    best_score,
                    best_output,
                )
                break
        else:
            best_epoch = epoch
            logger.info(
                "Epoch %03d/%03d | train_loss=%.4f",
                epoch,
                total_epochs,
                train_result["loss"]["total"],
            )
            history.append(row)

    if validation_loader is not None and fixed_epochs is None:
        if best_state is None:
            raise RuntimeError("No validation checkpoint was selected")
        model.load_state_dict(best_state)
    elif fixed_epochs is not None:
        best_epoch = int(fixed_epochs)
        best_output = selection_output or "class"

    if trial_label is not None and not pruned:
        logger.info(
            "Trial %d finished | trained_epochs=%d | best_epoch=%d | best_val=%.4f [%s]%s",
            trial_label,
            len(history),
            best_epoch,
            best_score if best_score != -float("inf") else float("nan"),
            best_output,
            " | early_stop" if stopped_early else "",
        )

    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(artifact_dir / "history.csv", index=False)
    torch.save(model.state_dict(), artifact_dir / "model_state.pt")
    (artifact_dir / "model_metadata.json").write_text(
        json.dumps(
            {
                "hparams": asdict(hparams),
                "loss_weights": asdict(loss_weights),
                "best_epoch": best_epoch,
                "selected_output": best_output,
                "log_mean": log_mean,
                "log_std": log_std,
                "task": task,
                "target_mask_attr": target_mask_attr,
                "num_classes": num_classes,
                "flag_handling": "separate learned binary gate",
                "cuts": cuts.tolist(),
                "epsilon": epsilon,
                "gate_threshold": gate_threshold,
                "flag_gate_threshold": flag_gate_threshold,
                "selection_score_weights": asdict(selection_score_weights),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = {
        "best_epoch": int(best_epoch),
        "trained_epochs": int(len(history)),
        "selected_output": best_output,
        "best_validation_score": None if best_score == -float("inf") else float(best_score),
        "model_state_path": str(artifact_dir / "model_state.pt"),
    }

    if validation_loader is not None:
        validation_result = _run_epoch(
            model=model,
            loader=validation_loader,
            device=device,
            optimizer=None,
            class_weights=class_weights,
            inactive_pos_weight=inactive_pos_weight,
            flag_pos_weight=flag_pos_weight,
            loss_weights=loss_weights,
            log_mean=log_mean,
            log_std=log_std,
            cuts=cuts,
            epsilon=epsilon,
            gate_threshold=gate_threshold,
            flag_gate_threshold=flag_gate_threshold,
            target_mask_attr=target_mask_attr,
            selection_score_weights=selection_score_weights,
            num_classes=num_classes,
            collect_predictions=True,
        )
        result["validation_predictions_path"] = str(
            _save_validation_outputs(
                artifact_dir,
                validation_result,
                best_output,
                num_classes,
                flag_gate_threshold,
            )
        )

    if test_loader is not None:
        test_result = _run_epoch(
            model=model,
            loader=test_loader,
            device=device,
            optimizer=None,
            class_weights=class_weights,
            inactive_pos_weight=inactive_pos_weight,
            flag_pos_weight=flag_pos_weight,
            loss_weights=loss_weights,
            log_mean=log_mean,
            log_std=log_std,
            cuts=cuts,
            epsilon=epsilon,
            gate_threshold=gate_threshold,
            flag_gate_threshold=flag_gate_threshold,
            target_mask_attr=target_mask_attr,
            selection_score_weights=selection_score_weights,
            num_classes=num_classes,
            collect_predictions=True,
        )
        result["test"] = test_result
        result["selected_test_full_class"] = test_result[f"full_class_{best_output}"]
        result["selected_test_activity"] = test_result[f"activity_{best_output}"]
        result["selected_test_combined"] = result["selected_test_full_class"]
        _save_test_outputs(artifact_dir, test_result, best_output, num_classes)
        _log_test_breakdown(
            logger, result["selected_test_full_class"], selected_output=best_output
        )

        pred_col = {
            "class": "class_prediction",
            "gated": "gated_prediction",
            "log_kpi": "log_kpi_prediction",
        }.get(best_output, "class_prediction")
        predictions_df = pd.DataFrame(test_result.get("predictions", []))
        if "selected_prediction" in predictions_df.columns:
            pred_col = "selected_prediction"
        confusion = result["selected_test_full_class"].get("confusion_matrix", [])
        save_training_plots(
            training_dir=artifact_dir.parent,
            task=task,
            history_csv=artifact_dir / "history.csv",
            confusion_matrix=confusion,
            predictions_df=predictions_df,
            num_classes=num_classes,
            selected_pred_col=pred_col,
        )

    return result


def evaluate_saved_pair_aware_model(
    *,
    task: str,
    target_mask_attr: str,
    state_dict: dict,
    train_loader,
    test_loader,
    output_dir: Path,
    num_node_tokens: int,
    num_contingency_tokens: int,
    hparams: PairAwareHParams,
    loss_weights: PairAwareLossWeights,
    log_mean: float,
    log_std: float,
    cuts: np.ndarray,
    epsilon: float,
    gate_threshold: float,
    flag_gate_threshold: float,
    selected_output: str,
    class_weight_mode: str,
    gate_pos_weight_mode: str,
    flag_gate_pos_weight_mode: str,
    flag_pos_weight_multiplier: float,
    selection_score_weights: SelectionScoreWeights,
    num_classes: int,
    logger: logging.Logger,
    history_csv: Optional[Path] = None,
) -> dict:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = PairAwareGINE(
        num_node_tokens=num_node_tokens,
        num_contingency_tokens=num_contingency_tokens,
        target_mask_attr=target_mask_attr,
        hparams=hparams,
        num_classes=num_classes,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    class_weights = compute_class_weights(
        train_loader, class_weight_mode, device, target_mask_attr, num_classes
    )
    inactive_pos_weight = compute_gate_pos_weight(
        train_loader, gate_pos_weight_mode, device, target_mask_attr, num_classes
    )
    flag_pos_weight = compute_flag_pos_weight(
        train_loader,
        flag_gate_pos_weight_mode,
        device,
        target_mask_attr,
        num_classes,
        multiplier=flag_pos_weight_multiplier,
    )
    test_result = _run_epoch(
        model=model,
        loader=test_loader,
        device=device,
        optimizer=None,
        class_weights=class_weights,
        inactive_pos_weight=inactive_pos_weight,
        flag_pos_weight=flag_pos_weight,
        loss_weights=loss_weights,
        log_mean=log_mean,
        log_std=log_std,
        cuts=cuts,
        epsilon=epsilon,
        gate_threshold=gate_threshold,
        flag_gate_threshold=flag_gate_threshold,
        target_mask_attr=target_mask_attr,
        selection_score_weights=selection_score_weights,
        num_classes=num_classes,
        collect_predictions=True,
    )
    task_dir = Path(output_dir) / task
    task_dir.mkdir(parents=True, exist_ok=True)
    _save_test_outputs(task_dir, test_result, selected_output, num_classes)
    selected_metrics = test_result[f"full_class_{selected_output}"]
    _log_test_breakdown(logger, selected_metrics, selected_output=selected_output)

    pred_col_map = {
        "class": "class_prediction",
        "gated": "gated_prediction",
        "log_kpi": "log_kpi_prediction",
    }
    pred_col = pred_col_map.get(selected_output, "class_prediction")
    predictions_df = pd.DataFrame(test_result.get("predictions", []))
    if "selected_prediction" in predictions_df.columns:
        pred_col = "selected_prediction"
    confusion = selected_metrics.get("confusion_matrix", [])
    history_path = (
        Path(history_csv)
        if history_csv is not None
        else Path(output_dir) / task / "history.csv"
    )
    save_training_plots(
        training_dir=Path(output_dir),
        task=task,
        history_csv=history_path,
        confusion_matrix=confusion,
        predictions_df=predictions_df,
        num_classes=num_classes,
        selected_pred_col=pred_col,
    )

    return {
        "loss": test_result["loss"],
        "full_class_class": test_result["full_class_class"],
        "full_class_gated": test_result["full_class_gated"],
        "full_class_log_kpi": test_result["full_class_log_kpi"],
        "selected_test_full_class": selected_metrics,
        "selected_test_activity": test_result[f"activity_{selected_output}"],
    }
