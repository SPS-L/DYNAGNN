#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pair-aware direct six-class Voltage GNN for DYNAGNN.

The GNN is the primary predictor. It receives graph topology and physical
features, explicit target-component identity, contingency identity/location,
and optional operating-point context.

The classification head learns all six classes directly:
- classes 0..4: KPI-derived activity levels;
- class 5: disconnected or controlled component.

Class 5 is never overwritten deterministically during evaluation. The
structural disconnection mask is retained only for target construction and
audit. No historical KPI/class prior is used.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GINEConv

ACTIVITY_CLASSES = 5
NUM_CLASSES = 6
DISCONNECTED_CLASS = 5


@dataclass(frozen=True)
class PairAwareHParams:
    hidden_dim: int = 128
    node_id_dim: int = 24
    contingency_id_dim: int = 32
    type_dim: int = 8
    pair_dim: int = 32
    op_context_embedding_dim: int = 32
    num_gnn_layers: int = 3
    decoder_hidden_dim: int = 256
    dropout: float = 0.15
    lr: float = 2.0e-4
    weight_decay: float = 1.0e-5


@dataclass(frozen=True)
class PairAwareLossWeights:
    classification: float = 1.0
    regression: float = 0.30
    inactive_gate: float = 0.20
    ordinal_cdf: float = 0.10


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


class PairAwareDirectVoltageModel(nn.Module):
    """Direct event- and target-conditioned GINE model.

    This is intentionally close to the strongest pair-aware residual encoder,
    but removes every historical KPI prior. The output heads predict from the
    graph representation itself.
    """

    def __init__(
        self,
        *,
        num_node_tokens: int,
        num_contingency_tokens: int,
        op_context_dim: int,
        use_op_context: bool,
        hparams: PairAwareHParams,
        num_node_types: int = 3,
        num_edge_types: int = 3,
    ) -> None:
        super().__init__()
        h = int(hparams.hidden_dim)
        self.hparams = hparams
        self.use_op_context = bool(use_op_context)

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

        if self.use_op_context:
            if int(op_context_dim) <= 0:
                raise ValueError("op_context_dim must be positive when use_op_context=True")
            self.op_context_encoder = nn.Sequential(
                nn.Linear(int(op_context_dim), hparams.op_context_embedding_dim),
                nn.ReLU(),
                nn.LayerNorm(hparams.op_context_embedding_dim),
                nn.Dropout(hparams.dropout),
            )
            op_dim = hparams.op_context_embedding_dim
        else:
            self.op_context_encoder = None
            op_dim = 0

        # target h + event h + global(mean|max)=2h + |h-e| + h*e + dz
        # + explicit target ID + explicit contingency ID + pair interaction + optional OP context
        decoder_in = (
            h * 6
            + 1
            + hparams.node_id_dim
            + hparams.contingency_id_dim
            + hparams.pair_dim
            + op_dim
        )
        self.shared_decoder = nn.Sequential(
            nn.Linear(decoder_in, hparams.decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(hparams.dropout),
            nn.Linear(hparams.decoder_hidden_dim, hparams.decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(hparams.dropout),
        )
        self.class_head = nn.Linear(hparams.decoder_hidden_dim, NUM_CLASSES)
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

        target_mask = data.bus_node_mask.bool()
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
        if self.use_op_context:
            op_context = data.op_context
            if op_context.ndim == 1:
                op_context = op_context.view(num_graphs, -1)
            op_embedding = self.op_context_encoder(op_context.float())
            parts.append(op_embedding[target_batch])

        shared = self.shared_decoder(torch.cat(parts, dim=1))
        return {
            "class_logits": self.class_head(shared),
            "inactive_logit": self.inactive_head(shared).squeeze(1),
            "log_kpi_std": self.regression_head(shared).squeeze(1),
        }


def _safe_div(numerator: float | int, denominator: float | int) -> float:
    return float(numerator) / float(denominator) if float(denominator) > 0 else 0.0


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    """Compute square confusion-matrix metrics and exact ordinal offsets.

    ``num_classes`` is six for the learned class-0..5 task. It may also be six
    for the activity-only subset (true classes 0..4), which correctly counts a
    predicted class 5 as an error while excluding absent true classes from the
    balanced/macro averages.
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
    return float(
        0.40 * metrics["balanced_accuracy"]
        + 0.30 * metrics["macro_f1"]
        + 0.20 * metrics["accuracy"]
        + 0.10 * metrics["within_one_accuracy"]
    )


def _ordinal_cdf_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    pred_cdf = probabilities.cumsum(dim=1)[:, :-1]
    true_cdf = F.one_hot(y, num_classes=NUM_CLASSES).float().cumsum(dim=1)[:, :-1]
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
    values = np.maximum(np.power(10.0, np.clip(log_values, -30.0, 30.0)) - float(epsilon), 0.0)
    return np.searchsorted(cuts, values, side="left").astype(np.int64)


def _move_batch(data, device: torch.device):
    tensor_keys = [
        "x",
        "batch",
        "edge_index",
        "edge_attr",
        "y_class",
        "bus_node_mask",
        "node_token",
        "contingency_token",
        "event_node_mask",
        "event_edge_mask",
        "event_graph_type",
        "y_log_kpi_std",
        "op_context",
        "structural_class5_mask",
    ]
    keys = [key for key in tensor_keys if hasattr(data, key)]
    return data.to(device, *keys)


def _decode_gated(logits: torch.Tensor, gate_logit: torch.Tensor, threshold: float) -> torch.Tensor:
    inactive_probability = torch.sigmoid(gate_logit)
    active_pred = logits[:, 1:].argmax(dim=1) + 1
    return torch.where(
        inactive_probability >= float(threshold),
        torch.zeros_like(active_pred),
        active_pred,
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
    loss_weights: PairAwareLossWeights,
    log_mean: float,
    log_std: float,
    cuts: np.ndarray,
    epsilon: float,
    gate_threshold: float,
    collect_predictions: bool = False,
) -> dict:
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    sums = {"total": 0.0, "classification": 0.0, "regression": 0.0, "gate": 0.0, "ordinal": 0.0}
    n_supervised = 0

    six_true_chunks: list[np.ndarray] = []
    six_class_chunks: list[np.ndarray] = []
    six_gated_chunks: list[np.ndarray] = []
    six_log_chunks: list[np.ndarray] = []
    activity_true_chunks: list[np.ndarray] = []
    activity_class_chunks: list[np.ndarray] = []
    activity_gated_chunks: list[np.ndarray] = []
    activity_log_chunks: list[np.ndarray] = []
    prediction_rows: list[dict] = []

    with torch.set_grad_enabled(train_mode):
        for data in loader:
            data = _move_batch(data, device)
            output = model(data)
            y_all = data.y_class[data.bus_node_mask].long()
            log_target_all = data.y_log_kpi_std[data.bus_node_mask]
            structural_all = data.structural_class5_mask[data.bus_node_mask].bool()

            valid_mask = (y_all >= 0) & (y_all < NUM_CLASSES)
            if not bool(valid_mask.any()):
                continue

            logits = output["class_logits"][valid_mask]
            gate_logit = output["inactive_logit"][valid_mask]
            reg_prediction = output["log_kpi_std"][valid_mask]
            y = y_all[valid_mask]
            log_target = log_target_all[valid_mask]

            # All six labels, including structural class 5, supervise the classifier.
            classification_loss = F.cross_entropy(logits, y, weight=class_weights)
            gate_target = (y == 0).float()
            gate_loss = F.binary_cross_entropy_with_logits(
                gate_logit, gate_target, pos_weight=inactive_pos_weight
            )
            ordinal_loss = _ordinal_cdf_loss(logits, y)

            # Class 5 has no KPI target by design, so regression is learned only where
            # a finite KPI exists (normally classes 0..4).
            finite_reg = torch.isfinite(log_target)
            regression_loss = (
                F.smooth_l1_loss(reg_prediction[finite_reg], log_target[finite_reg])
                if bool(finite_reg.any())
                else logits.new_zeros(())
            )
            total_loss = (
                loss_weights.classification * classification_loss
                + loss_weights.regression * regression_loss
                + loss_weights.inactive_gate * gate_loss
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
            sums["ordinal"] += float(ordinal_loss.detach().cpu()) * n

            detached_logits = logits.detach()
            class_pred = detached_logits.argmax(dim=1)
            gated_pred = _decode_gated(detached_logits, gate_logit.detach(), gate_threshold)
            log_pred_np = _classes_from_log_prediction(
                reg_prediction.detach().cpu().numpy(),
                log_mean=log_mean,
                log_std=log_std,
                cuts=cuts,
                epsilon=epsilon,
            )
            # The regression branch can only discretize KPI classes 0..4. For its
            # class-5 decision, use the learned six-class classifier rather than a
            # deterministic structural override.
            class_pred_np = class_pred.cpu().numpy()
            log_pred_np[class_pred_np == DISCONNECTED_CLASS] = DISCONNECTED_CLASS

            y_np = y.detach().cpu().numpy()
            gated_pred_np = gated_pred.cpu().numpy()
            six_true_chunks.append(y_np)
            six_class_chunks.append(class_pred_np)
            six_gated_chunks.append(gated_pred_np)
            six_log_chunks.append(log_pred_np)

            activity_np = y_np < DISCONNECTED_CLASS
            if bool(activity_np.any()):
                activity_true_chunks.append(y_np[activity_np])
                activity_class_chunks.append(class_pred_np[activity_np])
                activity_gated_chunks.append(gated_pred_np[activity_np])
                activity_log_chunks.append(log_pred_np[activity_np])

            if collect_predictions:
                num_graphs = int(data.num_graphs)
                ops = _string_list(getattr(data, "op_name", ""), num_graphs)
                events = _string_list(getattr(data, "event_id", ""), num_graphs)
                target_graph_all = data.batch[data.bus_node_mask].detach().cpu().numpy()
                target_token_all = data.node_token[data.bus_node_mask].detach().cpu().numpy()
                structural_np_all = structural_all.detach().cpu().numpy()
                valid_indices = np.flatnonzero(valid_mask.detach().cpu().numpy())
                probabilities = torch.softmax(detached_logits, dim=1).cpu().numpy()

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
                        "structural_class5_target": bool(structural_np_all[target_idx]),
                        "class_prediction": class_value,
                        "gated_prediction": gated_value,
                        "log_kpi_prediction": log_value,
                        "class_error_offset": class_value - true_class,
                        "gated_error_offset": gated_value - true_class,
                        "log_kpi_error_offset": log_value - true_class,
                    }
                    for cls in range(NUM_CLASSES):
                        row[f"class_probability_{cls}"] = float(probabilities[local_idx, cls])
                    prediction_rows.append(row)

    def _cat(chunks: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(chunks) if chunks else np.empty((0,), dtype=np.int64)

    six_true = _cat(six_true_chunks)
    activity_true = _cat(activity_true_chunks)
    result = {
        "loss": {key: _safe_div(value, n_supervised) for key, value in sums.items()},
        "six_class_class": classification_metrics(six_true, _cat(six_class_chunks), NUM_CLASSES),
        "six_class_gated": classification_metrics(six_true, _cat(six_gated_chunks), NUM_CLASSES),
        "six_class_log_kpi": classification_metrics(six_true, _cat(six_log_chunks), NUM_CLASSES),
        # Secondary 0..4 view. Prediction 5 remains an error rather than being hidden.
        "activity_class": classification_metrics(activity_true, _cat(activity_class_chunks), NUM_CLASSES),
        "activity_gated": classification_metrics(activity_true, _cat(activity_gated_chunks), NUM_CLASSES),
        "activity_log_kpi": classification_metrics(activity_true, _cat(activity_log_chunks), NUM_CLASSES),
    }
    # Backward-compatible aliases; these are learned six-class metrics, not a
    # deterministic class-5 override.
    result["combined_class"] = result["six_class_class"]
    result["combined_gated"] = result["six_class_gated"]
    result["combined_log_kpi"] = result["six_class_log_kpi"]
    if collect_predictions:
        result["predictions"] = prediction_rows
    return result

def _compute_class_weights(loader, mode: str, device: torch.device) -> Optional[torch.Tensor]:
    if mode == "none":
        return None
    counts = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    for data in loader:
        y = data.y_class[data.bus_node_mask]
        y = y[(y >= 0) & (y < NUM_CLASSES)]
        counts += torch.bincount(y, minlength=NUM_CLASSES).double()
    weights = torch.sqrt(counts.sum() / counts.clamp_min(1.0))
    weights = weights / weights.mean()
    return weights.float().to(device)


def _compute_gate_pos_weight(loader, mode: str, device: torch.device) -> Optional[torch.Tensor]:
    if mode == "none":
        return None
    inactive = 0
    active = 0
    for data in loader:
        y = data.y_class[data.bus_node_mask]
        y = y[(y >= 0) & (y < NUM_CLASSES)]
        inactive += int((y == 0).sum().item())
        active += int((y != 0).sum().item())
    return torch.tensor([_safe_div(active, max(inactive, 1))], dtype=torch.float32, device=device)


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


def _save_test_outputs(output_dir: Path, result: dict, selected_output: str) -> None:
    test_dir = output_dir / "test_outputs"
    test_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(result.get("predictions", []))
    predictions.to_csv(test_dir / "test_predictions.csv", index=False)

    six_key = f"six_class_{selected_output}"
    activity_key = f"activity_{selected_output}"
    six = result[six_key]
    activity = result[activity_key]
    (test_dir / "test_metrics.json").write_text(
        json.dumps(
            {
                "selected_output": selected_output,
                "six_class_learned_0_to_5": six,
                "activity_subset_true_0_to_4": activity,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _metrics_frame(six).to_csv(test_dir / "test_per_class_metrics_0_to_5.csv", index=False)
    _metrics_frame(activity).to_csv(
        test_dir / "test_activity_subset_per_class_metrics.csv", index=False
    )
    pd.DataFrame(six["confusion_matrix"]).to_csv(
        test_dir / "test_confusion_matrix_0_to_5.csv", index=False
    )
    _error_offset_frame(six).to_csv(
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
        _under_over_by_true_class(y_true, y_pred, NUM_CLASSES).to_csv(
            test_dir / "test_under_over_by_true_class_0_to_5.csv", index=False
        )
        _error_offset_by_true_class(y_true, y_pred, NUM_CLASSES).to_csv(
            test_dir / "test_error_offset_by_true_class.csv", index=False
        )
        pd.DataFrame(
            [
                {
                    "exact": six["correct"],
                    "total": six["total"],
                    "under_count": six["under_count"],
                    "under_rate": six["under_rate"],
                    "over_count": six["over_count"],
                    "over_rate": six["over_rate"],
                    "under_gt1_count": six["under_gt1_count"],
                    "under_gt1_rate": six["under_gt1_rate"],
                    "over_gt1_count": six["over_gt1_count"],
                    "over_gt1_rate": six["over_gt1_rate"],
                    "within_one_accuracy": six["within_one_accuracy"],
                    "mae": six["mae"],
                    "rmse": six["rmse"],
                    "mean_signed_error": six["mean_signed_error"],
                }
            ]
        ).to_csv(test_dir / "test_under_over_summary_0_to_5.csv", index=False)


def _log_test_breakdown(
    logger: logging.Logger, metrics: dict, *, selected_output: str
) -> None:
    logger.info(
        "TEST learned six-class selected=%s — %d/%d acc=%.4f bal=%.4f macroF1=%.4f MAE=%.4f",
        selected_output,
        metrics["correct"],
        metrics["total"],
        metrics["accuracy"],
        metrics["balanced_accuracy"],
        metrics["macro_f1"],
        metrics["mae"],
    )
    logger.info("TEST per-class accuracy (correct/support; recall):")
    for cls in range(NUM_CLASSES):
        support = int(metrics["class_support"][cls])
        correct = int(metrics["class_correct"][cls])
        accuracy = float(metrics["class_accuracy"][cls])
        precision = float(metrics["class_precision"][cls])
        f1 = float(metrics["class_f1"][cls])
        logger.info(
            "  Class %d: %d/%d = %.4f | precision=%.4f F1=%.4f",
            cls,
            correct,
            support,
            accuracy,
            precision,
            f1,
        )
    logger.info("TEST exact ordinal error offsets (prediction - true):")
    for offset in range(-(NUM_CLASSES - 1), NUM_CLASSES):
        key = str(offset)
        label = f"{offset:+d}" if offset != 0 else "0 (exact)"
        logger.info(
            "  %s: %d/%d = %.4f",
            label,
            int(metrics["error_offset_count"][key]),
            int(metrics["total"]),
            float(metrics["error_offset_rate"][key]),
        )

def run_pair_aware_training(
    *,
    train_loader,
    validation_loader,
    test_loader,
    output_dir: Path,
    num_node_tokens: int,
    num_contingency_tokens: int,
    op_context_dim: int,
    use_op_context: bool,
    hparams: PairAwareHParams,
    loss_weights: PairAwareLossWeights,
    log_mean: float,
    log_std: float,
    cuts: np.ndarray,
    epsilon: float,
    gate_threshold: float,
    epochs: int,
    patience: int,
    fixed_epochs: Optional[int],
    selection_output: Optional[str],
    class_weight_mode: str,
    gate_pos_weight_mode: str,
    logger: logging.Logger,
) -> dict:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = PairAwareDirectVoltageModel(
        num_node_tokens=num_node_tokens,
        num_contingency_tokens=num_contingency_tokens,
        op_context_dim=op_context_dim,
        use_op_context=use_op_context,
        hparams=hparams,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4, min_lr=1.0e-6)
    class_weights = _compute_class_weights(train_loader, class_weight_mode, device)
    inactive_pos_weight = _compute_gate_pos_weight(train_loader, gate_pos_weight_mode, device)

    logger.info("Pair-aware direct GNN device: %s", device)
    logger.info("HParams: %s", asdict(hparams))
    logger.info("Loss weights: %s", asdict(loss_weights))
    logger.info("Use OP context: %s (dim=%d)", use_op_context, op_context_dim)
    logger.info("Class weights: %s", None if class_weights is None else class_weights.detach().cpu().tolist())
    logger.info("Inactive pos weight: %s", None if inactive_pos_weight is None else inactive_pos_weight.detach().cpu().tolist())

    history: list[dict] = []
    best_score = -float("inf")
    best_epoch = 0
    best_output = selection_output or "class"
    best_state: Optional[dict] = None
    stale = 0
    total_epochs = int(fixed_epochs) if fixed_epochs is not None else int(epochs)

    for epoch in range(1, total_epochs + 1):
        train_result = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            class_weights=class_weights,
            inactive_pos_weight=inactive_pos_weight,
            loss_weights=loss_weights,
            log_mean=log_mean,
            log_std=log_std,
            cuts=cuts,
            epsilon=epsilon,
            gate_threshold=gate_threshold,
        )
        row = {"epoch": epoch, **{f"train_{k}_loss": v for k, v in train_result["loss"].items()}}

        if validation_loader is not None:
            val_result = _run_epoch(
                model=model,
                loader=validation_loader,
                device=device,
                optimizer=None,
                class_weights=class_weights,
                inactive_pos_weight=inactive_pos_weight,
                loss_weights=loss_weights,
                log_mean=log_mean,
                log_std=log_std,
                cuts=cuts,
                epsilon=epsilon,
                gate_threshold=gate_threshold,
            )
            candidates = {
                "class": val_result["six_class_class"]["selection_score"],
                "gated": val_result["six_class_gated"]["selection_score"],
                "log_kpi": val_result["six_class_log_kpi"]["selection_score"],
            }
            selected = max(candidates, key=candidates.get)
            score = float(candidates[selected])
            for name, value in candidates.items():
                row[f"val_{name}_score"] = float(value)
            row["val_selected_output"] = selected
            row["val_selected_score"] = score
            scheduler.step(score)

            if fixed_epochs is None and score > best_score + 1.0e-6:
                best_score = score
                best_epoch = epoch
                best_output = selected
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale = 0
            elif fixed_epochs is None:
                stale += 1

            logger.info(
                "Epoch %03d | train=%.4f | val class=%.4f gated=%.4f logKPI=%.4f | selected=%s %.4f",
                epoch,
                train_result["loss"]["total"],
                candidates["class"],
                candidates["gated"],
                candidates["log_kpi"],
                selected,
                score,
            )
            if fixed_epochs is None and stale >= int(patience):
                logger.info("Early stopping at epoch %d", epoch)
                break
        else:
            best_epoch = epoch
            logger.info(
                "Epoch %03d/%03d | train=%.4f | train class=%.4f gated=%.4f logKPI=%.4f",
                epoch,
                total_epochs,
                train_result["loss"]["total"],
                train_result["six_class_class"]["accuracy"],
                train_result["six_class_gated"]["accuracy"],
                train_result["six_class_log_kpi"]["accuracy"],
            )
        history.append(row)

    if validation_loader is not None and fixed_epochs is None:
        if best_state is None:
            raise RuntimeError("No validation checkpoint was selected")
        model.load_state_dict(best_state)
    elif fixed_epochs is not None:
        best_epoch = int(fixed_epochs)
        best_output = selection_output or "class"

    training_dir = output_dir / "training" / "voltage_pair_aware_six_class_gnn"
    training_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(training_dir / "history.csv", index=False)
    torch.save(model.state_dict(), training_dir / "model_state.pt")
    (training_dir / "model_metadata.json").write_text(
        json.dumps(
            {
                "hparams": asdict(hparams),
                "loss_weights": asdict(loss_weights),
                "best_epoch": best_epoch,
                "selected_output": best_output,
                "use_op_context": use_op_context,
                "op_context_dim": op_context_dim,
                "log_mean": log_mean,
                "log_std": log_std,
                "num_classes": NUM_CLASSES,
                "class5_handling": "learned direct prediction; no deterministic override",
                "cuts": cuts.tolist(),
                "epsilon": epsilon,
                "gate_threshold": gate_threshold,
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
    }
    if test_loader is not None:
        test_result = _run_epoch(
            model=model,
            loader=test_loader,
            device=device,
            optimizer=None,
            class_weights=class_weights,
            inactive_pos_weight=inactive_pos_weight,
            loss_weights=loss_weights,
            log_mean=log_mean,
            log_std=log_std,
            cuts=cuts,
            epsilon=epsilon,
            gate_threshold=gate_threshold,
            collect_predictions=True,
        )
        result["test"] = test_result
        result["selected_test_six_class"] = test_result[f"six_class_{best_output}"]
        result["selected_test_activity"] = test_result[f"activity_{best_output}"]
        # Backward-compatible name; this is the learned six-class result.
        result["selected_test_combined"] = result["selected_test_six_class"]
        _save_test_outputs(output_dir, test_result, best_output)
        _log_test_breakdown(logger, result["selected_test_six_class"], selected_output=best_output)
    return result
