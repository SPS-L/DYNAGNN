# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: GAT voltage severity training

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GATv2Conv


@dataclass(frozen=True)
class VoltageHParams:
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


class GAT_V(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        edge_dim: int,
        hidden_channels: int,
        hidden_dim: int,
        num_layers: int,
        num_gnn_layers: int,
        num_classes: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        h = hidden_channels * num_heads

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
            lin_layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        lin_layers += [nn.Linear(hidden_dim, num_classes - 1)]
        self.head = nn.Sequential(*lin_layers)

    def forward(self, x, edge_index, edge_attr, bus_node_mask):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        bus_x = x[bus_node_mask]
        return self.head(bus_x)


def coral_transform(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    km1 = int(num_classes) - 1
    thresholds = torch.arange(km1, device=y.device, dtype=y.dtype).unsqueeze(0)
    return (y.unsqueeze(1) > thresholds).float()


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
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
        reduction="mean",
    )

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


def _compute_pos_weight_voltage(train_loader, *, device: torch.device, num_classes: int) -> torch.Tensor:
    all_y = []
    for data in train_loader:
        y = data.y_class[data.bus_node_mask].long()
        all_y.append(y)
    y_cat = torch.cat(all_y, dim=0)
    all_targets = coral_transform(y_cat, num_classes)
    pos_counts = all_targets.sum(dim=0)
    neg_counts = (1.0 - all_targets).sum(dim=0)
    pos_weight_cpu = (neg_counts / pos_counts.clamp(min=1.0)).float()
    return pos_weight_cpu.to(device)


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
            logits = model(data.x, data.edge_index, data.edge_attr, data.bus_node_mask)
            y = data.y_class[data.bus_node_mask].long().to(device)

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

    out = {
        "loss": float(np.mean(loss_list)) if loss_list else float("nan"),
        "acc": float(correct) / float(total) if total else 0.0,
        "mae": float(abs_err_sum) / float(total) if total else 0.0,
    }
    return out


def _checkpoint_dir(training_dir: Path) -> Path:
    d = training_dir / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_ckpt(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load_ckpt(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def _sample_hparams(trial: optuna.Trial, space: dict) -> VoltageHParams:
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

    return VoltageHParams(
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


def run_gat_voltage_training(
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
    logger.info("GAT voltage training device: %s", device)
    model_cfg = (config.get("model", {}) or {})
    if "num_classes" not in model_cfg:
        raise KeyError("Missing required config key: model.num_classes (in config.yaml)")
    num_classes = int(model_cfg["num_classes"])
    epochs = int((config.get("training", {}) or {}).get("epochs", 100))
    patience = int((config.get("training", {}) or {}).get("patience", 10))
    seed = int((config.get("training", {}) or {}).get("seed", 42))

    ckpt_dir = _checkpoint_dir(training_dir)
    last_ckpt = ckpt_dir / "gat_voltage_last.pt"
    best_ckpt = ckpt_dir / "gat_voltage_best.pt"

    # Optuna study (resume-safe)
    optuna_cfg = (config.get("optuna", {}) or {})
    n_trials = int(optuna_cfg.get("n_trials", 15))
    hparam_space = (optuna_cfg.get("hparams", {}) or {})
    study_name = "gat_voltage"
    storage = f"sqlite:///{(training_dir / 'optuna_gat_voltage.sqlite3').as_posix()}"

    sample_graph = next(iter(train_loader))
    in_channels = int(sample_graph.x.shape[1])
    edge_dim = int(sample_graph.edge_attr.shape[1])
    logger.info("Voltage model dims: in_channels=%d edge_dim=%d", in_channels, edge_dim)

    pos_weight = _compute_pos_weight_voltage(train_loader, device=device, num_classes=num_classes)
    logger.info("Voltage pos_weight: %s", pos_weight.detach().cpu().numpy().tolist())

    def objective(trial: optuna.Trial) -> float:
        hp = _sample_hparams(trial, hparam_space)
        model = GAT_V(
            in_channels=in_channels,
            edge_dim=edge_dim,
            hidden_channels=hp.hidden_channels,
            hidden_dim=hp.hidden_dim,
            num_layers=hp.num_layers,
            num_gnn_layers=hp.num_gnn_layers,
            num_classes=num_classes,
            num_heads=hp.num_heads,
            dropout=hp.dropout,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)

        # Full training inside Optuna: scheduler + early stopping (patience)
        trial_ckpt_best = ckpt_dir / f"gat_voltage_optuna_trial_{trial.number}_best.pt"
        trial_ckpt_last = ckpt_dir / f"gat_voltage_optuna_trial_{trial.number}_last.pt"

        best_val = float("inf")
        best_epoch = -1
        epochs_no_improve = 0

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
            val_m = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                decode_threshold=hp.coral_prediction_threshold,
                device=device,
                num_classes=num_classes,
                pos_weight=pos_weight,
                under_penalty_lambda=hp.under_penalty_lambda,
                high_class_threshold=high_class_threshold,
            )
            val_loss = float(val_m["loss"])
            scheduler.step(val_loss)

            payload = {
                "trial": trial.number,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val": best_val,
                "best_epoch": best_epoch,
                "hparams": hp.__dict__,
                "study": study_name,
            }
            _save_ckpt(trial_ckpt_last, payload)

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                epochs_no_improve = 0
                payload["best_val"] = best_val
                payload["best_epoch"] = best_epoch
                _save_ckpt(trial_ckpt_best, payload)
                trial.set_user_attr("best_checkpoint", str(trial_ckpt_best))
                trial.set_user_attr("best_epoch", int(best_epoch))
            else:
                epochs_no_improve += 1

            # prune support (optional)
            trial.report(best_val, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if epochs_no_improve >= patience:
                break

        return float(best_val)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    logger.info("Voltage Optuna study: %s (storage=%s) n_trials=%d", study_name, storage, n_trials)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(study.best_trial.params)
    best_params["high_class_threshold"] = high_class_threshold
    # If space omitted something (unlikely), fall back to defaults via config ordering
    best_hp = VoltageHParams(
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
    (model_dir / "gat_voltage_best_hparams.json").write_text(
        json.dumps(best_params, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("Voltage best hparams saved: %s", model_dir / "gat_voltage_best_hparams.json")

    # Retrieve best model directly from the best Optuna trial checkpoint (no extra full retrain)
    best_ckpt_path = study.best_trial.user_attrs.get("best_checkpoint")
    if not best_ckpt_path:
        raise RuntimeError("Best Optuna trial does not have a saved best_checkpoint user_attr.")
    best = _load_ckpt(Path(best_ckpt_path), device)

    model = GAT_V(
        in_channels=in_channels,
        edge_dim=edge_dim,
        hidden_channels=int(best["hparams"]["hidden_channels"]),
        hidden_dim=int(best["hparams"]["hidden_dim"]),
        num_layers=int(best["hparams"]["num_layers"]),
        num_gnn_layers=int(best["hparams"]["num_gnn_layers"]),
        num_classes=num_classes,
        num_heads=int(best["hparams"]["num_heads"]),
        dropout=float(best["hparams"]["dropout"]),
    ).to(device)
    model.load_state_dict(best["model"])

    # Also keep a stable pointer checkpoint for "best overall"
    _save_ckpt(best_ckpt, best)
    _save_ckpt(last_ckpt, best)
    best_model_path = model_dir / "gat_voltage_best_model.pt"
    torch.save(model.state_dict(), best_model_path)
    logger.info("Voltage best model saved: %s", best_model_path)

    # Testing + plots
    test_m = run_epoch(
        model=model,
        loader=test_loader,
        optimizer=None,
        decode_threshold=float(best["hparams"]["coral_prediction_threshold"]),
        device=device,
        num_classes=num_classes,
        pos_weight=pos_weight,
        under_penalty_lambda=float(best["hparams"]["under_penalty_lambda"]),
        high_class_threshold=high_class_threshold,
    )
    logger.info("Voltage test metrics: %s", test_m)

    # Plot 1: pred-true histogram (distance)
    diffs = []
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.edge_attr, data.bus_node_mask)
            y = data.y_class[data.bus_node_mask].long().to(device)
            pred = coral_predict(logits, threshold=float(best["hparams"]["coral_prediction_threshold"]))
            diffs.append((pred - y).detach().cpu().numpy())
    diffs = np.concatenate(diffs, axis=0) if diffs else np.array([], dtype=np.int64)

    if diffs.size:
        vals, counts = np.unique(diffs, return_counts=True)
        pct = 100.0 * counts / counts.sum()
        plt.figure(figsize=(10, 4.5))
        plt.bar(vals.astype(int), pct, width=0.9)
        plt.xlabel("Pred - True (distance)")
        plt.ylabel("Percent of test predictions (%)")
        plt.title("Voltage: test distance histogram (pred - true)")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        out = training_dir / "gat_voltage_test_distance_hist.png"
        plt.savefig(out, dpi=200)
        plt.close()
        logger.info("Saved plot: %s", out)


