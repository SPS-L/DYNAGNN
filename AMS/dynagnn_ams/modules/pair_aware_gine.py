# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Pair-aware residual GINE model (inference-only copy aligned with DYNAGNN v1.2).
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv


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
    """Direct event- and target-conditioned GINE model (DYNAGNN v1.2)."""

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

        self.node_input = nn.Sequential(
            nn.Linear(hparams.type_dim + hparams.node_id_dim + 6, h),
            nn.ReLU(),
            nn.LayerNorm(h),
        )
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

        decoder_in = h * 6 + 1 + hparams.node_id_dim + hparams.contingency_id_dim + hparams.pair_dim
        self.shared_decoder = nn.Sequential(
            nn.Linear(decoder_in, hparams.decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(hparams.dropout),
            nn.Linear(hparams.decoder_hidden_dim, hparams.decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(hparams.dropout),
        )
        self.class_head = nn.Linear(hparams.decoder_hidden_dim, self.num_classes)
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
            "inactive_logit": self.inactive_head(shared).squeeze(1),
            "log_kpi_std": self.regression_head(shared).squeeze(1),
        }
