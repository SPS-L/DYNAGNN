# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: GAT apparent-power severity training

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
import torch
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GATv2Conv

from modules.gat_training_exports import (
    composite_selection_score,
    evaluate_detailed,
    export_optuna_trials_csv,
    log_detailed_metrics,
    plot_distance_histogram,
    plot_loss_curves,
    plot_pred_true_examples,
    resolve_selection_weights,
)


@dataclass(frozen=True)
class SpowerHParams:
    hidden_dim: int
    num_layers: int
    hidden_channels: int
    num_heads: int
    dropout: float
    num_gnn_layers: int
    lr: float
    weight_decay: float
    under_penalty_lambda: float
    coral_prediction_threshold: float


class GAT_S(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        edge_dim: int,
        hidden_channels: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int,
        num_gnn_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        h = hidden_channels * num_heads
        output_dim = num_classes - 1

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(
            GATv2Conv(
                in_channels=in_channels,
                out_channels=hidden_channels,
                heads=num_heads,
                edge_dim=edge_dim,
                dropout=dropout,
                concat=True,
            )
        )
        self.norms.append(nn.LayerNorm(h))

        for _ in range(int(num_gnn_layers) - 1):
            self.convs.append(
                GATv2Conv(
                    in_channels=h,
                    out_channels=hidden_channels,
                    heads=num_heads,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    concat=True,
                )
            )
            self.norms.append(nn.LayerNorm(h))

        self.dropout = nn.Dropout(dropout)

        lin_layers: list[nn.Module] = [nn.Linear(h, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(int(num_layers) - 1):
            lin_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        lin_layers.append(nn.Linear(hidden_dim, output_dim))
        self.head = nn.Sequential(*lin_layers)

    def forward(self, x, edge_index, edge_attr, gen_node_mask):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        gen_x = x[gen_node_mask]
        return self.head(gen_x)


def coral_transform(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    km1 = int(num_classes) - 1
    thresholds = torch.arange(km1, device=y.device, dtype=y.dtype).unsqueeze(0)
    return (y.unsqueeze(1) > thresholds).float()


def compute_coral_pos_weight(train_loader, *, num_classes: int, device: torch.device) -> torch.Tensor:
    km1 = int(num_classes) - 1
    pos = torch.zeros(km1, dtype=torch.float64)
    total = 0
    for data in train_loader:
        y = data.y_class[data.gen_node_mask].long().cpu()
        t = coral_transform(y, num_classes).cpu().to(torch.float64)
        pos += t.sum(dim=0)
        total += int(t.shape[0])
    neg = max(total, 1) - pos
    pos_weight = (neg / torch.clamp(pos, min=1.0)).float()
    return pos_weight.to(device)


def coral_loss(
    *,
    logits: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    pos_weight: Optional[torch.Tensor],
    under_penalty_lambda: float,
    high_class_threshold: Optional[int],
) -> torch.Tensor:
    targets = coral_transform(y, num_classes)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="mean")

    if under_penalty_lambda and under_penalty_lambda > 0 and high_class_threshold is not None:
        expected_class = torch.sigmoid(logits).sum(dim=1)
        under_amount = F.relu(y.float() - expected_class)
        high_mask = y >= int(high_class_threshold)
        if high_mask.any():
            under_penalty = (under_amount[high_mask] ** 2).mean()
        else:
            under_penalty = torch.tensor(0.0, device=logits.device)
        return bce + float(under_penalty_lambda) * under_penalty
    return bce


def coral_predict(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return (probs > float(threshold)).sum(dim=1).long()


def run_epoch(
    *,
    model: nn.Module,
    loader,
    optimizer: Optional[torch.optim.Optimizer],
    decode_threshold: float,
    device: torch.device,
    num_classes: int,
    pos_weight: torch.Tensor,
    under_penalty_lambda: float,
    high_class_threshold: Optional[int],
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    loss_list: list[float] = []
    correct = 0
    total = 0
    abs_err_sum = 0.0

    with torch.set_grad_enabled(train_mode):
        for data in loader:
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.edge_attr, data.gen_node_mask)
            y = data.y_class[data.gen_node_mask].long().to(device)

            loss = coral_loss(
                logits=logits,
                y=y,
                num_classes=num_classes,
                pos_weight=pos_weight,
                under_penalty_lambda=under_penalty_lambda,
                high_class_threshold=high_class_threshold,
            )

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            loss_list.append(float(loss.detach().cpu()))
            pred = coral_predict(logits.detach(), threshold=decode_threshold)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            abs_err_sum += float((pred - y).abs().sum().item())

    return {
        "loss": float(np.mean(loss_list)) if loss_list else float("nan"),
        "acc": float(correct) / float(total) if total else 0.0,
        "mae": float(abs_err_sum) / float(total) if total else 0.0,
    }


def _checkpoint_dir(training_dir: Path) -> Path:
    d = training_dir / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_ckpt(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load_ckpt(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def _sample_hparams(trial: optuna.Trial, space: dict) -> SpowerHParams:
    def _one(name: str, spec: dict):
        t = str(spec.get("type", "")).lower()
        if t == "categorical":
            return trial.suggest_categorical(name, list(spec["choices"]))
        if t == "int":
            return trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
        if t == "float":
            return trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        raise ValueError(f"Unsupported optuna.hparams type for {name}: {t}")

    return SpowerHParams(
        hidden_dim=int(_one("hidden_dim", space["hidden_dim"])),
        num_layers=int(_one("num_layers", space["num_layers"])),
        hidden_channels=int(_one("hidden_channels", space["hidden_channels"])),
        num_heads=int(_one("num_heads", space["num_heads"])),
        dropout=float(_one("dropout", space["dropout"])),
        num_gnn_layers=int(_one("num_gnn_layers", space["num_gnn_layers"])),
        lr=float(_one("lr", space["lr"])),
        weight_decay=float(_one("weight_decay", space["weight_decay"])),
        under_penalty_lambda=float(_one("under_penalty_lambda", space["under_penalty_lambda"])),
        coral_prediction_threshold=float(_one("coral_prediction_threshold", space["coral_prediction_threshold"])),
    )


def run_gat_spower_training(
    *,
    train_loader,
    val_loader,
    test_loader,
    training_dir: Path,
    model_dir: Path,
    config: dict,
    high_class_threshold: Optional[int],
    logger: logging.Logger,
) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.info("GAT spower training device: %s", device)
    model_cfg = (config.get("model", {}) or {})
    if "num_classes" not in model_cfg:
        raise KeyError("Missing required config key: model.num_classes (in config.yaml)")
    num_classes = int(model_cfg["num_classes"])
    training_cfg = config.get("training", {}) or {}
    epochs = int(training_cfg.get("epochs", 100))
    patience = int(training_cfg.get("patience", 10))
    seed = int(training_cfg.get("seed", 42))
    selection_f1_weight, selection_loss_weight = resolve_selection_weights(config)
    logger.info(
        "Spower selection score: high_recall + %.2f*high_f1 - %.2f*loss",
        selection_f1_weight,
        selection_loss_weight,
    )

    task_dir = training_dir / "spower"
    task_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = _checkpoint_dir(task_dir)
    last_ckpt = ckpt_dir / "gat_spower_last.pt"
    best_ckpt = ckpt_dir / "gat_spower_best.pt"

    optuna_cfg = (config.get("optuna", {}) or {})
    n_trials = int(optuna_cfg.get("n_trials", 15))
    hparam_space = (optuna_cfg.get("hparams", {}) or {})
    study_name = "gat_spower"
    storage = f"sqlite:///{(task_dir / 'optuna_gat_spower.sqlite3').as_posix()}"

    sample_graph = next(iter(train_loader))
    in_channels = int(sample_graph.x.shape[1])
    edge_dim = int(sample_graph.edge_attr.shape[1])
    logger.info("Spower model dims: in_channels=%d edge_dim=%d", in_channels, edge_dim)

    pos_weight = compute_coral_pos_weight(train_loader, num_classes=num_classes, device=device)
    logger.info("Spower pos_weight: %s", pos_weight.detach().cpu().numpy().tolist())

    def _forward(model_obj, data):
        return model_obj(data.x, data.edge_index, data.edge_attr, data.gen_node_mask)

    def _labels(data):
        return data.y_class[data.gen_node_mask]

    def objective(trial: optuna.Trial) -> float:
        hp = _sample_hparams(trial, hparam_space)
        model = GAT_S(
            in_channels=in_channels,
            edge_dim=edge_dim,
            hidden_channels=hp.hidden_channels,
            hidden_dim=hp.hidden_dim,
            num_classes=num_classes,
            num_layers=hp.num_layers,
            num_gnn_layers=hp.num_gnn_layers,
            num_heads=hp.num_heads,
            dropout=hp.dropout,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6)

        trial_ckpt_best = ckpt_dir / f"gat_spower_optuna_trial_{trial.number}_best.pt"
        trial_ckpt_last = ckpt_dir / f"gat_spower_optuna_trial_{trial.number}_last.pt"

        best_score = -float("inf")
        best_val_loss = float("inf")
        best_epoch = -1
        epochs_no_improve = 0
        train_history: list[float] = []
        val_history: list[float] = []

        for epoch in range(epochs):
            train_m = run_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                decode_threshold=hp.coral_prediction_threshold,
                device=device,
                num_classes=num_classes,
                pos_weight=pos_weight,
                under_penalty_lambda=hp.under_penalty_lambda,
                high_class_threshold=high_class_threshold,
            )
            val_m = evaluate_detailed(
                model=model,
                loader=val_loader,
                optimizer=None,
                device=device,
                num_classes=num_classes,
                decode_threshold=hp.coral_prediction_threshold,
                forward_fn=_forward,
                get_labels_fn=_labels,
                coral_loss_fn=coral_loss,
                predict_fn=coral_predict,
                pos_weight=pos_weight,
                under_penalty_lambda=hp.under_penalty_lambda,
                high_class_threshold=high_class_threshold,
            )
            val_loss = float(val_m["loss"])
            score = composite_selection_score(
                val_m,
                f1_weight=selection_f1_weight,
                loss_weight=selection_loss_weight,
            )
            train_history.append(float(train_m["loss"]))
            val_history.append(val_loss)
            scheduler.step(score)

            payload = {
                "trial": trial.number,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_score": best_score,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "hparams": hp.__dict__,
                "study": study_name,
            }
            _save_ckpt(trial_ckpt_last, payload)

            if score > best_score:
                best_score = score
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_no_improve = 0
                payload["best_score"] = best_score
                payload["best_val_loss"] = best_val_loss
                payload["best_epoch"] = best_epoch
                _save_ckpt(trial_ckpt_best, payload)
                trial.set_user_attr("best_checkpoint", str(trial_ckpt_best))
                trial.set_user_attr("best_epoch", int(best_epoch))
                trial.set_user_attr("best_score", float(best_score))
                trial.set_user_attr("best_val_loss", float(best_val_loss))
            else:
                epochs_no_improve += 1

            trial.report(best_score, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if epochs_no_improve >= patience:
                break

        trial.set_user_attr("train_history", train_history)
        trial.set_user_attr("val_history", val_history)
        return float(best_score)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    logger.info(
        "Spower Optuna study: %s (storage=%s) n_trials=%d objective=max composite score",
        study_name,
        storage,
        n_trials,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(
        "Spower best trial #%d score=%.6f val_loss=%.6f epoch=%s",
        study.best_trial.number,
        float(study.best_value),
        float(study.best_trial.user_attrs.get("best_val_loss", float("nan"))),
        study.best_trial.user_attrs.get("best_epoch"),
    )

    optuna_csv = task_dir / "optuna_trials.csv"
    export_optuna_trials_csv(study, optuna_csv)
    logger.info("Spower Optuna trials saved: %s", optuna_csv)

    best_params = dict(study.best_trial.params)
    best_params["high_class_threshold"] = high_class_threshold
    best_hp = SpowerHParams(
        hidden_dim=int(best_params["hidden_dim"]),
        num_layers=int(best_params["num_layers"]),
        hidden_channels=int(best_params["hidden_channels"]),
        num_heads=int(best_params["num_heads"]),
        dropout=float(best_params["dropout"]),
        num_gnn_layers=int(best_params["num_gnn_layers"]),
        lr=float(best_params["lr"]),
        weight_decay=float(best_params["weight_decay"]),
        under_penalty_lambda=float(best_params["under_penalty_lambda"]),
        coral_prediction_threshold=float(best_params["coral_prediction_threshold"]),
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "gat_spower_best_hparams.json").write_text(
        json.dumps(best_params, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("Spower best hparams saved: %s", model_dir / "gat_spower_best_hparams.json")

    # Retrieve best model directly from the best Optuna trial checkpoint (no extra full retrain)
    best_ckpt_path = study.best_trial.user_attrs.get("best_checkpoint")
    if not best_ckpt_path:
        raise RuntimeError("Best Optuna trial does not have a saved best_checkpoint user_attr.")
    best = _load_ckpt(Path(best_ckpt_path), device)

    model = GAT_S(
        in_channels=in_channels,
        edge_dim=edge_dim,
        hidden_channels=int(best["hparams"]["hidden_channels"]),
        hidden_dim=int(best["hparams"]["hidden_dim"]),
        num_classes=num_classes,
        num_layers=int(best["hparams"]["num_layers"]),
        num_gnn_layers=int(best["hparams"]["num_gnn_layers"]),
        num_heads=int(best["hparams"]["num_heads"]),
        dropout=float(best["hparams"]["dropout"]),
    ).to(device)
    model.load_state_dict(best["model"])

    # Also keep a stable pointer checkpoint for "best overall"
    _save_ckpt(best_ckpt, best)
    _save_ckpt(last_ckpt, best)
    best_model_path = model_dir / "gat_spower_best_model.pt"
    torch.save(model.state_dict(), best_model_path)
    logger.info("Spower best model saved: %s", best_model_path)

    decode_threshold = float(best["hparams"]["coral_prediction_threshold"])
    under_penalty = float(best["hparams"]["under_penalty_lambda"])

    train_history = list(study.best_trial.user_attrs.get("train_history", []))
    val_history = list(study.best_trial.user_attrs.get("val_history", []))
    loss_curve_path = task_dir / "loss_curves.png"
    plot_loss_curves(train_history, val_history, loss_curve_path, title="CORAL loss")
    logger.info("Saved plot: %s", loss_curve_path)

    logger.info("Spower test metrics:")
    test_m = evaluate_detailed(
        model=model,
        loader=test_loader,
        optimizer=None,
        device=device,
        num_classes=num_classes,
        decode_threshold=decode_threshold,
        forward_fn=_forward,
        get_labels_fn=_labels,
        coral_loss_fn=coral_loss,
        predict_fn=coral_predict,
        pos_weight=pos_weight,
        under_penalty_lambda=under_penalty,
        high_class_threshold=high_class_threshold,
    )
    log_detailed_metrics(logger, test_m, label="test", num_classes=num_classes)

    hist_path = task_dir / "test_distance_hist.png"
    plot_distance_histogram(test_m, num_classes, hist_path)
    logger.info("Saved plot: %s", hist_path)

    examples_loader = DataLoader(test_loader.dataset, batch_size=1, shuffle=False)
    plot_pred_true_examples(
        model=model,
        loader=examples_loader,
        device=device,
        decode_threshold=decode_threshold,
        forward_fn=_forward,
        get_labels_fn=_labels,
        predict_fn=coral_predict,
        num_classes=num_classes,
        output_dir=task_dir / "pred_true_examples",
        logger=logger,
        max_examples=5,
        target_class=num_classes - 1,
    )


