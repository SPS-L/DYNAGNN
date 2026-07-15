#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stable pre-contingency operating-point context for DYNAGNN.

This module deliberately uses only compact aggregate statistics of the raw
pre-contingency node state. It does not use KPI labels, nearest-operating-point
selection, Manhattan distance, PCA, post-contingency data, or contingency
features.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


DEFAULT_NODE_STATE_COLS = (1, 2, 3, 4)
STAT_NAMES = ("mean", "std", "min", "max", "median")


def normalize_op(value: object) -> str:
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


def normalize_ops(values: Iterable[object]) -> list[str]:
    result = [normalize_op(v) for v in values]
    if len(result) != len(set(result)):
        raise ValueError(f"Duplicate operating points after normalization: {result}")
    return result


def op_sort_key(value: object) -> tuple[int, str]:
    op = normalize_op(value)
    suffix = op.rsplit("_", 1)[-1]
    return (int(suffix), op) if suffix.isdigit() else (10**9, op)


def resolve_graph_path(graph_dir: Path, op_name: object) -> Path:
    op = normalize_op(op_name)
    candidates = [
        graph_dir / f"{op}.pt",
        graph_dir / op,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No graph for {op}. Tried: {[str(p) for p in candidates]}")


def _finite_column(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(1, dtype=np.float64)
    # Prevent accidental overflow from malformed graph values while preserving
    # all normal power-system feature magnitudes.
    return np.clip(finite, -1.0e12, 1.0e12)


def _summary(values: np.ndarray) -> list[float]:
    values = _finite_column(values)
    return [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.min(values)),
        float(np.max(values)),
        float(np.median(values)),
    ]


def _selected_node_columns(data, metadata: dict) -> tuple[list[int], list[str]]:
    num_features = int(data.x.shape[1])
    schema = list((metadata or {}).get("node_feature_schema", []) or [])

    selected = [idx for idx in DEFAULT_NODE_STATE_COLS if idx < num_features]
    if not selected:
        selected = [idx for idx in range(num_features) if idx != 0]
    if not selected:
        selected = [0]

    names: list[str] = []
    for idx in selected:
        if idx < len(schema):
            names.append(str(schema[idx]))
        else:
            names.append(f"node_feature_{idx}")
    return selected, names


def raw_descriptor(graph_path: Path) -> tuple[np.ndarray, list[str]]:
    loaded = torch.load(graph_path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict) or "data" not in loaded:
        raise ValueError(f"Unexpected graph format: {graph_path}")
    data = loaded["data"]
    metadata = loaded.get("metadata", {}) or {}
    if not hasattr(data, "x"):
        raise ValueError(f"Graph has no node feature tensor x: {graph_path}")

    columns, column_names = _selected_node_columns(data, metadata)
    x = data.x.detach().cpu().numpy().astype(np.float64, copy=False)

    descriptor: list[float] = []
    names: list[str] = []
    for idx, feature_name in zip(columns, column_names):
        descriptor.extend(_summary(x[:, idx]))
        names.extend([f"{feature_name}__{stat}" for stat in STAT_NAMES])

    array = np.asarray(descriptor, dtype=np.float64)
    array = np.nan_to_num(array, nan=0.0, posinf=1.0e12, neginf=-1.0e12)
    if not np.isfinite(array).all():
        raise FloatingPointError(f"Non-finite descriptor generated for {graph_path}")
    return array, names


@dataclass
class OPContextTransformer:
    """Train-only standardization for compact OP aggregate descriptors."""

    scaler: StandardScaler
    feature_names: list[str]
    train_ops: list[str]

    @classmethod
    def fit(cls, graph_dir: Path, train_ops: Sequence[str]) -> "OPContextTransformer":
        normalized = sorted(normalize_ops(train_ops), key=op_sort_key)
        rows: list[np.ndarray] = []
        feature_names: list[str] | None = None
        for op in normalized:
            row, names = raw_descriptor(resolve_graph_path(graph_dir, op))
            if feature_names is None:
                feature_names = names
            elif names != feature_names:
                raise RuntimeError(
                    f"Descriptor schema mismatch for {op}. Expected {feature_names}, got {names}"
                )
            rows.append(row)

        matrix = np.vstack(rows)
        if not np.isfinite(matrix).all():
            raise FloatingPointError("Training OP descriptor matrix contains non-finite values")
        scaler = StandardScaler()
        scaler.fit(matrix)
        return cls(scaler=scaler, feature_names=feature_names or [], train_ops=normalized)

    @property
    def output_dim(self) -> int:
        return int(len(self.feature_names))

    def transform_one(self, graph_dir: Path, op_name: str) -> np.ndarray:
        row, names = raw_descriptor(resolve_graph_path(graph_dir, op_name))
        if names != self.feature_names:
            raise RuntimeError(f"Descriptor schema mismatch for {op_name}")
        transformed = self.scaler.transform(row.reshape(1, -1))[0]
        transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(transformed).all():
            raise FloatingPointError(f"Scaled descriptor contains non-finite values for {op_name}")
        return transformed.astype(np.float32, copy=False)

    def transform_map(self, graph_dir: Path, ops: Sequence[str]) -> dict[str, np.ndarray]:
        return {
            normalize_op(op): self.transform_one(graph_dir, normalize_op(op))
            for op in normalize_ops(ops)
        }

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, output_dir / "op_context_scaler.pkl")
        metadata = {
            "feature_names": self.feature_names,
            "train_ops": self.train_ops,
            "output_dim": self.output_dim,
            "description": (
                "Compact aggregate descriptor of raw pre-contingency node-state "
                "columns 1..4; five statistics per feature; standardized on train OPs only."
            ),
        }
        (output_dir / "op_context_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
