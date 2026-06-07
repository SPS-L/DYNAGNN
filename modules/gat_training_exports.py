# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: Shared GAT training metrics, logging, and figure exports

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


CONSTRAINT_MAX_UNDER_CLASSES = 3


def resolve_selection_weights(config: dict) -> tuple[float, float]:
    training_cfg = config.get("training", {}) or {}
    missing = [
        key
        for key in ("selection_f1_weight", "selection_loss_weight")
        if key not in training_cfg or training_cfg[key] is None
    ]
    if missing:
        raise KeyError(
            "Missing required config key(s): "
            + ", ".join(f"training.{k}" for k in missing)
            + " (in config.yaml)"
        )
    return (
        float(training_cfg["selection_f1_weight"]),
        float(training_cfg["selection_loss_weight"]),
    )


def composite_selection_score(
    metrics: dict,
    *,
    f1_weight: float,
    loss_weight: float,
) -> float:
    """Checkpoint / Optuna objective: high_recall + f1_weight*f1 - loss_weight*loss."""
    return (
        float(metrics["high_recall"])
        + float(f1_weight) * float(metrics["high_f1"])
        - float(loss_weight) * float(metrics["loss"])
    )


def safe_div(numer: float | int, denom: float | int) -> float:
    return float(numer) / float(denom) if float(denom) > 0 else 0.0


def evaluate_detailed(
    *,
    model: nn.Module,
    loader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    num_classes: int,
    decode_threshold: float,
    forward_fn: Callable[[nn.Module, object], torch.Tensor],
    get_labels_fn: Callable[[object], torch.Tensor],
    coral_loss_fn: Callable[..., torch.Tensor],
    predict_fn: Callable[[torch.Tensor, float], torch.Tensor],
    pos_weight: torch.Tensor,
    under_penalty_lambda: float,
    high_class_threshold: Optional[int],
    constraint_max_under: int = CONSTRAINT_MAX_UNDER_CLASSES,
) -> dict:
    """Run one epoch and return full ordinal metrics (notebook-compatible)."""
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    loss_list: list[float] = []
    correct = 0
    total = 0
    false_under = 0
    false_over = 0
    hard_under_gt3 = 0
    abs_err_sum = 0.0

    class_total = torch.zeros(num_classes, dtype=torch.long)
    class_correct = torch.zeros(num_classes, dtype=torch.long)
    under_by_k = torch.zeros(num_classes, dtype=torch.long)
    over_by_k = torch.zeros(num_classes, dtype=torch.long)

    high_tp = 0
    high_fp = 0
    high_fn = 0

    high_thresh = int(high_class_threshold if high_class_threshold is not None else num_classes - 1)

    with torch.set_grad_enabled(train_mode):
        for data in loader:
            data = data.to(device)
            logits = forward_fn(model, data)
            y = get_labels_fn(data).long().to(device)

            loss = coral_loss_fn(
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
            pred = predict_fn(logits.detach(), decode_threshold)

            correct += int((pred == y).sum().item())
            total += int(y.numel())

            wrong = pred != y
            false_under += int((wrong & (pred < y)).sum().item())
            false_over += int((wrong & (pred > y)).sum().item())

            hard_under_gt3 += int(((y - pred) > int(constraint_max_under)).sum().item())
            abs_err_sum += float((pred - y).abs().sum().item())

            diff = (pred - y).to(torch.int64)
            under = (-diff).clamp(min=0)
            over = diff.clamp(min=0)

            if under.any():
                under_bins = torch.bincount(under[under > 0].detach().cpu(), minlength=num_classes)
                under_by_k += under_bins
            if over.any():
                over_bins = torch.bincount(over[over > 0].detach().cpu(), minlength=num_classes)
                over_by_k += over_bins

            class_total += torch.bincount(y.detach().cpu(), minlength=num_classes)
            class_correct += torch.bincount(y[pred == y].detach().cpu(), minlength=num_classes)

            true_high = y >= high_thresh
            pred_high = pred >= high_thresh
            high_tp += int((true_high & pred_high).sum().item())
            high_fp += int((~true_high & pred_high).sum().item())
            high_fn += int((true_high & ~pred_high).sum().item())

    high_recall = safe_div(high_tp, high_tp + high_fn)
    high_precision = safe_div(high_tp, high_tp + high_fp)
    high_f1 = safe_div(2.0 * high_precision * high_recall, high_precision + high_recall)

    return {
        "loss": float(np.mean(loss_list)) if loss_list else float("nan"),
        "acc": safe_div(correct, total),
        "correct": correct,
        "total": total,
        "false_under": false_under,
        "false_over": false_over,
        "under_by_k": under_by_k.tolist(),
        "over_by_k": over_by_k.tolist(),
        "class_correct": class_correct.tolist(),
        "class_total": class_total.tolist(),
        "hard_under_gt3": hard_under_gt3,
        "hard_rate": safe_div(hard_under_gt3, total),
        "mae": safe_div(abs_err_sum, total),
        "high_recall": high_recall,
        "high_precision": high_precision,
        "high_f1": high_f1,
        "high_tp": high_tp,
        "high_fp": high_fp,
        "high_fn": high_fn,
        "high_class_threshold": high_thresh,
    }


def log_detailed_metrics(
    logger: logging.Logger,
    metrics: dict,
    *,
    label: str,
    num_classes: int,
) -> None:
    """Log test metrics in notebook-compatible multi-line format."""
    logger.info("  loss: %.6f", metrics["loss"])
    logger.info(
        "  accuracy: %.6f (%.2f%%)",
        metrics["acc"],
        metrics["acc"] * 100.0,
    )

    if not metrics["total"]:
        return

    logger.info(
        "  outcome (%s)  — correct: %.2f%% | under: %.2f%% | over: %.2f%%  (%d / %d / %d of %d)",
        label,
        100.0 * metrics["correct"] / metrics["total"],
        100.0 * metrics["false_under"] / metrics["total"],
        100.0 * metrics["false_over"] / metrics["total"],
        metrics["correct"],
        metrics["false_under"],
        metrics["false_over"],
        metrics["total"],
    )
    logger.info(
        "  hard under > %d (%s): %d (%.2f%%) | MAE: %.4f",
        CONSTRAINT_MAX_UNDER_CLASSES,
        label,
        metrics["hard_under_gt3"],
        metrics["hard_rate"] * 100.0,
        metrics["mae"],
    )
    logger.info(
        "  under distance (%s):  %s",
        label,
        ", ".join(f"-{k}:{int(metrics['under_by_k'][k])}" for k in range(1, num_classes)),
    )
    logger.info(
        "  over  distance (%s):  %s",
        label,
        ", ".join(f"+{k}:{int(metrics['over_by_k'][k])}" for k in range(1, num_classes)),
    )

    parts = []
    for cls in range(num_classes):
        tot = int(metrics["class_total"][cls])
        cor = int(metrics["class_correct"][cls])
        if tot:
            parts.append(f"{cls}:{100.0 * cor / tot:.2f}% ({cor}/{tot})")
        else:
            parts.append(f"{cls}:— (0/0)")
    logger.info("  correct by true class (%s): %s", label, " | ".join(parts))

    high_thresh = metrics["high_class_threshold"]
    logger.info(
        "  high-class (%s, >= %d) — recall: %.4f | precision: %.4f | f1: %.4f | tp/fp/fn: %d/%d/%d",
        label,
        high_thresh,
        metrics["high_recall"],
        metrics["high_precision"],
        metrics["high_f1"],
        metrics["high_tp"],
        metrics["high_fp"],
        metrics["high_fn"],
    )


def export_optuna_trials_csv(study, path: Path) -> None:
    rows = []
    for trial in study.trials:
        row = {
            "trial": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "best_epoch": trial.user_attrs.get("best_epoch"),
            "best_score": trial.user_attrs.get("best_score"),
            "best_val_loss": trial.user_attrs.get("best_val_loss"),
        }
        row.update(trial.params)
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def plot_loss_curves(
    train_history: list[float],
    val_history: list[float],
    path: Path,
    *,
    title: str = "CORAL loss",
) -> None:
    if not train_history and not val_history:
        return
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 4))
    if train_history:
        ax.plot(train_history, label="train")
    if val_history:
        ax.plot(val_history, label="val")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def plot_distance_histogram(
    metrics: dict,
    num_classes: int,
    path: Path,
    *,
    title: str = "Test distance histogram (pred - true)",
) -> None:
    """Styled histogram from evaluate_detailed metrics (notebook-compatible)."""
    total = int(metrics.get("total", 0))
    if total <= 0:
        return

    x = np.arange(-num_classes + 1, num_classes)
    counts = np.zeros_like(x, dtype=np.int64)
    for k in range(1, num_classes):
        counts[x == -k] = int(metrics["under_by_k"][k])
        counts[x == +k] = int(metrics["over_by_k"][k])
    counts[x == 0] = int(metrics["correct"])

    pct = (counts / total) * 100.0
    colors = ["#d62728" if v < 0 else "#1f77b4" if v > 0 else "#7f7f7f" for v in x]

    plt.figure(figsize=(23, 6))
    bars = plt.bar(x, pct, color=colors, width=0.9)
    ax = plt.gca()
    ax.set_xlim(float(x[0]) - 0.5, float(x[-1]) + 0.5)
    ax.margins(x=0)
    plt.xlabel("Pred - True (distance)", fontsize=16)
    plt.ylabel("Percent of test predictions (%)", fontsize=16)
    plt.title(title, fontsize=16)
    plt.xticks(x, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="y", alpha=0.25)

    for rect, p in zip(bars, pct):
        pf = float(p)
        if pf != 0.0 and abs(pf) < 1e-3:
            label = f"{pf:.1e}%"
        else:
            label = f"{pf:.3f}%"
        plt.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() + 0.05,
            label,
            ha="center",
            va="bottom",
            fontsize=14,
        )

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def _save_pred_true_plot(
    *,
    sample_idx: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tag: str,
    num_classes: int,
    path: Path,
) -> None:
    node_idx = np.arange(len(y_true))
    diff = (y_pred - y_true).astype(int)
    colors = ["green" if v == 0 else "blue" if v > 0 else "red" for v in diff]

    plt.figure(figsize=(12, 4.5))
    plt.axhline(0, color="black", linewidth=1, alpha=0.9)
    plt.scatter(node_idx, diff, s=12, color=colors, alpha=0.9)
    plt.xlabel("Node index (FR bus order)")
    plt.ylabel("Pred - True")
    plt.title(f"{tag} (sample idx {sample_idx}) - pred-true per node (under < 0)")
    plt.yticks(np.arange(-num_classes + 1, num_classes))
    plt.grid(alpha=0.25)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def plot_pred_true_examples(
    *,
    model: nn.Module,
    loader,
    device: torch.device,
    decode_threshold: float,
    forward_fn: Callable[[nn.Module, object], torch.Tensor],
    get_labels_fn: Callable[[object], torch.Tensor],
    predict_fn: Callable[[torch.Tensor, float], torch.Tensor],
    num_classes: int,
    output_dir: Path,
    logger: logging.Logger,
    max_examples: int = 5,
    target_class: int = 9,
) -> None:
    """Save up to `max_examples` class-9 UNDER and OVER pred-true scatter plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    under_found = 0
    over_found = 0

    with torch.no_grad():
        for sample_idx, sample in enumerate(loader):
            if under_found >= max_examples and over_found >= max_examples:
                break

            sample = sample.to(device)
            logits = forward_fn(model, sample)
            y_true = get_labels_fn(sample).detach().cpu().numpy().astype(int)
            y_pred = predict_fn(logits, decode_threshold).detach().cpu().numpy().astype(int)

            if target_class not in y_true:
                continue

            diff = y_pred - y_true
            has_under = (diff < 0).any()
            has_over = (diff > 0).any()

            if has_under and under_found < max_examples:
                under_found += 1
                out = output_dir / f"pred_true_class{target_class}_under_{under_found}.png"
                _save_pred_true_plot(
                    sample_idx=sample_idx,
                    y_true=y_true,
                    y_pred=y_pred,
                    tag=f"Class-{target_class} UNDER example {under_found}/{max_examples}",
                    num_classes=num_classes,
                    path=out,
                )
                logger.info("Saved plot: %s", out)

            if has_over and over_found < max_examples:
                over_found += 1
                out = output_dir / f"pred_true_class{target_class}_over_{over_found}.png"
                _save_pred_true_plot(
                    sample_idx=sample_idx,
                    y_true=y_true,
                    y_pred=y_pred,
                    tag=f"Class-{target_class} OVER example {over_found}/{max_examples}",
                    num_classes=num_classes,
                    path=out,
                )
                logger.info("Saved plot: %s", out)

    if under_found < max_examples:
        logger.info(
            "Found only %d class-%d test sample(s) with at least one UNDER node.",
            under_found,
            target_class,
        )
    if over_found < max_examples:
        logger.info(
            "Found only %d class-%d test sample(s) with at least one OVER node.",
            over_found,
            target_class,
        )
    logger.info(
        "Plotted %d under-example(s) and %d over-example(s) containing true class %d.",
        under_found,
        over_found,
        target_class,
    )


def collect_pred_true_diffs(
    *,
    model: nn.Module,
    loader,
    device: torch.device,
    decode_threshold: float,
    forward_fn: Callable[[nn.Module, object], torch.Tensor],
    get_labels_fn: Callable[[object], torch.Tensor],
    predict_fn: Callable[[torch.Tensor, float], torch.Tensor],
) -> np.ndarray:
    diffs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = forward_fn(model, data)
            y = get_labels_fn(data).long().to(device)
            pred = predict_fn(logits, threshold=decode_threshold)
            diffs.append((pred - y).detach().cpu().numpy())
    return np.concatenate(diffs, axis=0) if diffs else np.array([], dtype=np.int64)
