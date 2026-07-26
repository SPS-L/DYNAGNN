# Pair-aware GINE inference and substation aggregation (DYNAGNN v1.2 compatible).

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import torch
from torch_geometric.data import Batch, Data

from .event_graph_construction import EDGE_CONT_FEATURE_NAMES, NODE_CONT_COLS
from .pair_aware_gine import PairAwareGINE, PairAwareHParams

MODEL_TYPE = "pair_aware_gine"


def resolve_torch_device(requested: str | torch.device | None = None) -> torch.device:
    """Resolve ``auto`` / ``cpu`` / ``cuda`` / ``cuda:N`` / ``mps`` to a ``torch.device``."""
    if isinstance(requested, torch.device):
        return requested

    raw = "auto" if requested is None else str(requested).strip().lower()
    if not raw or raw == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if raw == "cpu":
        return torch.device("cpu")

    if raw == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise RuntimeError(
                "Device 'mps' requested but torch.backends.mps is not available. "
                "Install the macOS PyTorch wheel (not the CPU-only Linux build)."
            )
        return torch.device("mps")

    if raw == "cuda" or raw.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Device {raw!r} requested but CUDA is not available in this PyTorch build."
            )
        return torch.device(raw)

    raise ValueError(
        f"Unknown device {requested!r}. Use auto, cpu, mps, cuda, or cuda:N."
    )


@dataclass
class InferencePredictionResult:
    voltage_predictions: Dict[str, int]
    spower_predictions: Dict[str, int]
    substation_predictions: Dict[str, int]


def _node_id(meta_key: str, metadata: dict) -> str:
    return str(metadata.get("id", meta_key)).strip()


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
            f"expected {MODEL_TYPE!r}. Copy DYNAGNN v1.2 deployment checkpoints "
            f"({expected_task}_best_model.pt), not legacy GAT weights."
        )
    if checkpoint.get("task") != expected_task:
        raise ValueError(
            f"Checkpoint {path} has task={checkpoint.get('task')!r}, "
            f"expected {expected_task!r}"
        )
    required = {
        "model_state_dict",
        "hparams",
        "num_classes",
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


def load_pair_aware_model(checkpoint: dict[str, Any], device: torch.device) -> PairAwareGINE:
    task = str(checkpoint["task"])
    target_mask_attr = {"voltage": "bus_node_mask", "spower": "gen_node_mask"}.get(task)
    if target_mask_attr is None:
        raise ValueError(f"Unsupported checkpoint task: {task!r}")
    model = PairAwareGINE(
        num_node_tokens=int(checkpoint["num_node_tokens"]),
        num_contingency_tokens=int(checkpoint["num_contingency_tokens"]),
        target_mask_attr=target_mask_attr,
        hparams=PairAwareHParams(**dict(checkpoint["hparams"])),
        num_classes=int(checkpoint["num_classes"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _attach_pair_tensors(sample: Data, checkpoint: dict[str, Any]) -> Data:
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
        node_token[node_index] = int(node_vocab.get(_node_id(str(meta_key), node_meta), 0))

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
        flag = int(checkpoint["num_classes"]) - 1
        class_prediction = logits.argmax(dim=1).detach().cpu().numpy()
        prediction[class_prediction == flag] = flag
        return torch.tensor(prediction, dtype=torch.long, device=logits.device)

    raise ValueError(f"Unsupported selected_output in checkpoint: {selected_output!r}")


def predict_pair_aware(
    *,
    model: PairAwareGINE,
    sample_cpu: Data,
    checkpoint: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """Return one predicted class per target node for a single scenario graph."""
    prepared = _attach_pair_tensors(sample_cpu, checkpoint)
    batch = Batch.from_data_list([prepared]).to(device)
    with torch.no_grad():
        output = model(batch)
        prediction = _decode(output, checkpoint)
    return prediction.detach().cpu()


def _find_checkpoint(models_dir: Path, task: str) -> Path:
    candidates: list[Path] = []
    for pattern in (
        f"{task}_best_model.pt",
        f"*{task}*best*model*.pt",
        f"*{task}*.pt",
    ):
        for path in sorted(models_dir.glob(pattern)):
            if path.is_file() and path not in candidates:
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"No {task} checkpoint found in {models_dir}. "
            f"Expected DYNAGNN export: {task}_best_model.pt"
        )
    preferred = [path for path in candidates if "model" in path.name.lower()]
    return preferred[0] if preferred else candidates[0]


def _find_scaler(models_dir: Path, filename: str) -> Path:
    direct = models_dir / filename
    if direct.is_file():
        return direct
    matches = sorted(models_dir.glob(f"*{filename}"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Missing scaler {filename} in {models_dir}. "
        "Copy x_scaler.pkl and edge_attr_scaler.pkl from DYNAGNN data/model/<study_name>/."
    )


def _edge_cont_cols_from_metadata(metadata: dict) -> list[int]:
    edge_schema = list(metadata.get("edge_feature_schema", []))
    if not edge_schema:
        raise RuntimeError("Missing edge_feature_schema on graph metadata")
    missing = [name for name in EDGE_CONT_FEATURE_NAMES if name not in edge_schema]
    if missing:
        raise RuntimeError(f"Graph edge_feature_schema is missing features: {missing}")
    return [edge_schema.index(name) for name in EDGE_CONT_FEATURE_NAMES]


def _scale_graph(data: Data, metadata: dict, x_scaler, edge_scaler) -> Data:
    scaled = data.clone()
    edge_cont_cols = _edge_cont_cols_from_metadata(metadata)

    x_part = scaled.x[:, NODE_CONT_COLS].cpu().numpy()
    scaled.x[:, NODE_CONT_COLS] = torch.tensor(
        x_scaler.transform(x_part),
        dtype=scaled.x.dtype,
        device=scaled.x.device,
    )

    if scaled.edge_attr.shape[0] > 0:
        edge_part = scaled.edge_attr[:, edge_cont_cols].cpu().numpy()
        scaled.edge_attr[:, edge_cont_cols] = torch.tensor(
            edge_scaler.transform(edge_part),
            dtype=scaled.edge_attr.dtype,
            device=scaled.edge_attr.device,
        )
    return scaled


def _predictions_to_components(
    predictions: torch.Tensor,
    mask: torch.Tensor,
    metadata: dict,
    *,
    node_type: str,
    require_dynamic_model: bool = False,
) -> Dict[str, int]:
    node_indices = mask.nonzero(as_tuple=False).view(-1).tolist()
    pred_values = predictions.detach().cpu().tolist()
    if len(node_indices) != len(pred_values):
        raise RuntimeError(
            f"Prediction count ({len(pred_values)}) does not match mask size ({len(node_indices)})"
        )
    idx_to_class = {int(idx): int(pred) for idx, pred in zip(node_indices, pred_values)}

    out: Dict[str, int] = {}
    for node_meta in metadata.get("node_metadata", {}).values():
        if str(node_meta.get("type", "")).lower() != node_type:
            continue
        if require_dynamic_model and not bool(node_meta.get("hasDynamicModel", False)):
            continue
        idx = int(node_meta["index"])
        if idx not in idx_to_class:
            continue
        component_id = str(node_meta.get("id", "")).strip()
        if component_id:
            out[component_id] = idx_to_class[idx]
    return out


def aggregate_substation_predictions(
    metadata: dict,
    voltage_predictions: Dict[str, int],
    spower_predictions: Dict[str, int],
) -> Dict[str, int]:
    substation_predictions: Dict[str, int] = {}

    for node_meta in metadata.get("node_metadata", {}).values():
        substation_id = str(node_meta.get("substationId", "")).strip()
        if not substation_id:
            continue

        node_type = str(node_meta.get("type", "")).lower()
        component_id = str(node_meta.get("id", "")).strip()
        value: Optional[int] = None

        if node_type == "bus":
            value = voltage_predictions.get(component_id)
        elif node_type == "generator" and bool(node_meta.get("hasDynamicModel", False)):
            value = spower_predictions.get(component_id)

        if value is None:
            continue
        substation_predictions[substation_id] = max(
            substation_predictions.get(substation_id, 0),
            int(value),
        )

    return substation_predictions


def aggregate_max_substation_predictions(
    predictions_per_graph: list[Dict[str, int]],
) -> Dict[str, int]:
    final: Dict[str, int] = {}
    for graph_predictions in predictions_per_graph:
        for substation_id, value in graph_predictions.items():
            final[substation_id] = max(final.get(substation_id, 0), int(value))
    return final


class PairAwareModels:
    """Load DYNAGNN v1.2 pair-aware GINE checkpoints and run AMS inference.

    Copy these files from ``<DYNAGNN data.path>/model/<study_name>/`` into
    ``AMS/dynagnn_ams/models/<network>/``:

    - ``voltage_best_model.pt``
    - ``spower_best_model.pt``
    - ``x_scaler.pkl``
    - ``edge_attr_scaler.pkl``

    Optional metadata JSON (``*_best_hparams.json``) is not required; all fields
    needed for inference live inside the ``.pt`` checkpoints.
    """

    def __init__(
        self,
        models_dir: str | Path,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        self.models_dir = Path(models_dir)
        if not self.models_dir.is_dir():
            raise FileNotFoundError(f"Models directory not found: {self.models_dir}")

        self.device = resolve_torch_device(device)

        self.voltage_checkpoint = load_pair_aware_checkpoint(
            _find_checkpoint(self.models_dir, "voltage"),
            expected_task="voltage",
        )
        self.spower_checkpoint = load_pair_aware_checkpoint(
            _find_checkpoint(self.models_dir, "spower"),
            expected_task="spower",
        )
        self.x_scaler = joblib.load(_find_scaler(self.models_dir, "x_scaler.pkl"))
        self.edge_scaler = joblib.load(_find_scaler(self.models_dir, "edge_attr_scaler.pkl"))

        self._voltage_model: Optional[PairAwareGINE] = None
        self._spower_model: Optional[PairAwareGINE] = None
        self._initialized = False

    def initialize(self, sample_graph: Data) -> None:
        if self._initialized:
            return
        self._voltage_model = load_pair_aware_model(self.voltage_checkpoint, self.device)
        self._spower_model = load_pair_aware_model(self.spower_checkpoint, self.device)
        self._initialized = True

    def _predict_voltage(self, graph: Data, metadata: dict) -> Dict[str, int]:
        assert self._voltage_model is not None
        scaled = _scale_graph(graph, metadata, self.x_scaler, self.edge_scaler)
        predictions = predict_pair_aware(
            model=self._voltage_model,
            sample_cpu=scaled,
            checkpoint=self.voltage_checkpoint,
            device=self.device,
        )
        return _predictions_to_components(
            predictions,
            scaled.bus_node_mask,
            metadata,
            node_type="bus",
        )

    def _predict_spower(self, graph: Data, metadata: dict) -> Dict[str, int]:
        assert self._spower_model is not None
        scaled = _scale_graph(graph, metadata, self.x_scaler, self.edge_scaler)
        predictions = predict_pair_aware(
            model=self._spower_model,
            sample_cpu=scaled,
            checkpoint=self.spower_checkpoint,
            device=self.device,
        )
        return _predictions_to_components(
            predictions,
            scaled.gen_node_mask,
            metadata,
            node_type="generator",
            require_dynamic_model=True,
        )

    def predict(self, graph: Data) -> InferencePredictionResult:
        if not self._initialized:
            raise RuntimeError("PairAwareModels.initialize() must be called before predict()")

        metadata = getattr(graph, "metadata", {}) or {}
        # MPS is not reliable with concurrent multi-threaded GPU submits.
        if self.device.type == "mps":
            voltage_predictions = self._predict_voltage(graph, metadata)
            spower_predictions = self._predict_spower(graph, metadata)
        else:
            with ThreadPoolExecutor(max_workers=2) as executor:
                voltage_future = executor.submit(self._predict_voltage, graph, metadata)
                spower_future = executor.submit(self._predict_spower, graph, metadata)
                voltage_predictions = voltage_future.result()
                spower_predictions = spower_future.result()

        substation_predictions = aggregate_substation_predictions(
            metadata,
            voltage_predictions,
            spower_predictions,
        )
        return InferencePredictionResult(
            voltage_predictions=voltage_predictions,
            spower_predictions=spower_predictions,
            substation_predictions=substation_predictions,
        )
