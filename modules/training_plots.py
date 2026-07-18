# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: post-training diagnostic plots
"""Post-training diagnostic plots for pair-aware GINE models.

All figures are written under ``<training_dir>/<task>/plots/``
(where ``training_dir`` is ``data/training/<study_name>/``).
``loss_curve.png`` / ``score_curve.png`` are built from the winning Optuna
trial's ``history.csv`` (under ``optuna_trials/trial_N/``), but the PNGs
themselves are saved in the shared task ``plots/`` folder with the other
diagnostics.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MAX_NODE_EXAMPLES = 5
_DPI = 150


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_import_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def _ensure_plots_dir(training_dir: Path, task: str) -> Path:
    plots_dir = training_dir / task / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


# ---------------------------------------------------------------------------
# Loss curve
# ---------------------------------------------------------------------------

def plot_loss_curve(history_path: Path, plots_dir: Path, task: str) -> None:
    """Read ``history.csv`` and write ``loss_curve.png`` (train + val if present)."""
    plt = _safe_import_plt()
    if plt is None:
        logger.warning("matplotlib not available; skipping loss_curve.png")
        return

    df = pd.read_csv(history_path)
    train_col = "train_total_loss"
    val_col = "val_total_loss" if "val_total_loss" in df.columns else None

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["epoch"], df[train_col], label="train", color="#1f77b4")
    if val_col and df[val_col].notna().any():
        ax.plot(df["epoch"], df[val_col], label="val", color="#ff7f0e")
    ax.set_title(f"{task.capitalize()} — total loss curve", fontsize=14)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out = plots_dir / "loss_curve.png"
    fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved loss_curve.png → %s", out)


def plot_score_curve(history_path: Path, plots_dir: Path, task: str) -> None:
    """Plot validation selection scores from ``history.csv`` → ``score_curve.png``."""
    plt = _safe_import_plt()
    if plt is None:
        logger.warning("matplotlib not available; skipping score_curve.png")
        return

    df = pd.read_csv(history_path)
    series = [
        ("val_class_score", "class", "#1f77b4"),
        ("val_gated_score", "gated", "#2ca02c"),
        ("val_log_kpi_score", "logKPI", "#d62728"),
        ("val_selected_score", "selected", "#ff7f0e"),
    ]
    available = [(col, label, color) for col, label, color in series if col in df.columns]
    if not available:
        logger.warning("No val_*_score columns in %s; skipping score_curve.png", history_path)
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    for col, label, color in available:
        style = "--" if col == "val_selected_score" else "-"
        width = 2.0 if col == "val_selected_score" else 1.5
        ax.plot(df["epoch"], df[col], label=label, color=color, linestyle=style, linewidth=width)
    ax.set_title(f"{task.capitalize()} — validation selection scores", fontsize=14)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Selection score", fontsize=12)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out = plots_dir / "score_curve.png"
    fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved score_curve.png → %s", out)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    confusion: np.ndarray,
    num_classes: int,
    plots_dir: Path,
    task: str,
) -> None:
    """Write a row-normalised confusion matrix heat-map."""
    plt = _safe_import_plt()
    if plt is None:
        logger.warning("matplotlib not available; skipping confusion_matrix.png")
        return

    cm = confusion.astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(max(5, num_classes), max(4, num_classes - 1)))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = np.arange(num_classes)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks, fontsize=max(7, 12 - num_classes // 3))
    ax.set_yticklabels(ticks, fontsize=max(7, 12 - num_classes // 3))
    ax.set_xlabel("Predicted class", fontsize=12)
    ax.set_ylabel("True class", fontsize=12)
    ax.set_title(f"{task.capitalize()} — confusion matrix (row-normalised)", fontsize=13)

    thresh = 0.5
    fs = max(6, 10 - num_classes // 3)
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm_norm[i, j]
            if val > 0.0:
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=fs,
                    color="white" if val > thresh else "black",
                )

    plt.tight_layout()
    out = plots_dir / "confusion_matrix.png"
    fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion_matrix.png → %s", out)


# ---------------------------------------------------------------------------
# Distance histogram  (pred − true)
# ---------------------------------------------------------------------------

def plot_distance_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    plots_dir: Path,
    task: str,
) -> None:
    """Histogram of signed prediction offsets (pred − true)."""
    plt = _safe_import_plt()
    if plt is None:
        logger.warning("matplotlib not available; skipping distance_histogram.png")
        return

    diff = (y_pred.astype(np.int64) - y_true.astype(np.int64))
    x = np.arange(-(num_classes - 1), num_classes)
    counts = np.array([int((diff == v).sum()) for v in x], dtype=np.int64)
    total = max(int(diff.size), 1)
    pct = counts / total * 100.0

    colors = [
        "#d62728" if v < 0 else "#1f77b4" if v > 0 else "#7f7f7f"
        for v in x
    ]

    fig_w = max(10, len(x) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    bars = ax.bar(x, pct, color=colors, width=0.85, edgecolor="white", linewidth=0.4)
    ax.set_xlim(float(x[0]) - 0.5, float(x[-1]) + 0.5)
    ax.margins(x=0)
    ax.set_xlabel("Pred − True (class offset)", fontsize=13)
    ax.set_ylabel("% of test predictions", fontsize=13)
    ax.set_title(f"{task.capitalize()} — test distance histogram", fontsize=14)
    ax.set_xticks(x)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(axis="y", alpha=0.25)

    for rect, p in zip(bars, pct):
        if p == 0.0:
            continue
        label = f"{p:.1e}%" if 0 < p < 0.01 else f"{p:.2f}%"
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() + 0.05,
            label,
            ha="center", va="bottom", fontsize=max(7, 10 - num_classes // 4),
            rotation=90 if num_classes > 6 else 0,
        )

    # legend patch
    import matplotlib.patches as mpatches
    under_p = mpatches.Patch(color="#d62728", label="under-prediction")
    exact_p = mpatches.Patch(color="#7f7f7f", label="exact")
    over_p = mpatches.Patch(color="#1f77b4", label="over-prediction")
    ax.legend(handles=[under_p, exact_p, over_p], fontsize=10)

    plt.tight_layout()
    out = plots_dir / "distance_histogram.png"
    fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved distance_histogram.png → %s", out)


# ---------------------------------------------------------------------------
# Per-node pred-vs-true scatter examples
# ---------------------------------------------------------------------------

def _plot_one_graph(
    ax_class,
    ax_diff,
    y_true_row: np.ndarray,
    y_pred_row: np.ndarray,
    num_classes: int,
    title: str,
) -> None:
    """Fill a pair of axes (class scatter, diff scatter) for one graph."""
    node_idx = np.arange(len(y_true_row))
    diff = (y_pred_row - y_true_row).astype(int)

    # --- class per node ---
    ax_class.scatter(node_idx, y_true_row, s=14, label="True", color="#ff7f0e", alpha=0.9, zorder=3)
    ax_class.scatter(node_idx, y_pred_row, s=14, label="Pred", color="#1f77b4", alpha=0.9, zorder=2)
    ax_class.set_xlabel("Node index", fontsize=10)
    ax_class.set_ylabel("Class", fontsize=10)
    ax_class.set_title(f"{title} — class per node", fontsize=10)
    ax_class.set_yticks(np.arange(num_classes))
    ax_class.grid(alpha=0.25)
    ax_class.legend(fontsize=9, markerscale=1.2)

    # --- pred − true per node ---
    diff_colors = [
        "green" if v == 0 else "#1f77b4" if v > 0 else "#d62728"
        for v in diff
    ]
    ax_diff.axhline(0, color="black", linewidth=1, alpha=0.7)
    ax_diff.scatter(node_idx, diff, s=14, color=diff_colors, alpha=0.9)
    ax_diff.set_xlabel("Node index", fontsize=10)
    ax_diff.set_ylabel("Pred − True", fontsize=10)
    ax_diff.set_title(f"{title} — pred−true per node (under < 0)", fontsize=10)
    ax_diff.set_yticks(np.arange(-(num_classes - 1), num_classes))
    ax_diff.grid(alpha=0.25)


def plot_node_examples(
    predictions_df: pd.DataFrame,
    num_classes: int,
    plots_dir: Path,
    task: str,
    *,
    pred_col: str = "class_prediction",
    max_examples: int = MAX_NODE_EXAMPLES,
) -> None:
    """Plot up to *max_examples* graphs for both under- and over-prediction.

    Graphs are chosen from those that contain at least one node with the
    highest true class observed in the test set, mirroring the selection
    logic in ``GAT_Voltage_Final.ipynb``.
    """
    plt = _safe_import_plt()
    if plt is None:
        logger.warning("matplotlib not available; skipping node_examples plots")
        return

    if predictions_df.empty or pred_col not in predictions_df.columns:
        logger.warning("No predictions data for node_examples plots; skipping.")
        return

    required = {"operating_point", "contingency", "true_class", pred_col}
    if not required.issubset(predictions_df.columns):
        logger.warning("predictions_df missing columns %s; skipping node examples.", required - set(predictions_df.columns))
        return

    highest_class = int(predictions_df["true_class"].max())

    # Collect (op, contingency) pairs that have at least one node at highest_class
    interesting = (
        predictions_df[predictions_df["true_class"] == highest_class][["operating_point", "contingency"]]
        .drop_duplicates()
    )

    if interesting.empty:
        logger.info("No graphs with true class %d; skipping node_examples.", highest_class)
        return

    under_found = 0
    over_found = 0

    for _, row in interesting.iterrows():
        if under_found >= max_examples and over_found >= max_examples:
            break

        op, cont = row["operating_point"], row["contingency"]
        mask = (predictions_df["operating_point"] == op) & (predictions_df["contingency"] == cont)
        sub = predictions_df[mask]
        y_true_row = sub["true_class"].to_numpy(dtype=np.int64)
        y_pred_row = sub[pred_col].to_numpy(dtype=np.int64)
        diff = y_pred_row - y_true_row

        has_under = bool((diff < 0).any())
        has_over = bool((diff > 0).any())

        for direction, found, tag in [
            ("under", under_found, "UNDER"),
            ("over", over_found, "OVER"),
        ]:
            is_dir = has_under if direction == "under" else has_over
            cur_found = under_found if direction == "under" else over_found
            if is_dir and cur_found < max_examples:
                cur_found += 1
                label = f"cls{highest_class}_{tag}_ex{cur_found}_of_{max_examples}"
                fig, (ax_c, ax_d) = plt.subplots(1, 2, figsize=(16, 4.5))
                _plot_one_graph(
                    ax_c, ax_d,
                    y_true_row, y_pred_row,
                    num_classes=num_classes,
                    title=f"{task.capitalize()} {tag} ex {cur_found} | {op} / {cont}",
                )
                plt.tight_layout()
                out = plots_dir / f"node_example_{label}.png"
                fig.savefig(out, dpi=_DPI, bbox_inches="tight")
                plt.close(fig)
                logger.info("Saved node example → %s", out)
                if direction == "under":
                    under_found = cur_found
                else:
                    over_found = cur_found

    logger.info(
        "Node examples saved: %d under, %d over (class-%d graphs).",
        under_found, over_found, highest_class,
    )


# ---------------------------------------------------------------------------
# Master entry point called from pair_aware_gine.py / pair_aware_training.py
# ---------------------------------------------------------------------------

def save_training_plots(
    *,
    training_dir: Path,
    task: str,
    history_csv: Path,
    confusion_matrix: Sequence[Sequence[int]],
    predictions_df: pd.DataFrame,
    num_classes: int,
    selected_pred_col: str = "class_prediction",
) -> None:
    """Generate and save diagnostic figures under ``<training_dir>/<task>/plots/``.

    ``history_csv`` should point at the winning Optuna trial's ``history.csv``
    (the loss curve is still written to the shared task ``plots/`` folder).
    """
    plots_dir = _ensure_plots_dir(training_dir, task)

    # 1. Loss curve (history from best trial; plot in shared task plots/)
    if history_csv.exists():
        try:
            plot_loss_curve(history_csv, plots_dir, task)
        except Exception:
            logger.exception("Failed to save loss_curve.png")
        try:
            plot_score_curve(history_csv, plots_dir, task)
        except Exception:
            logger.exception("Failed to save score_curve.png")
    else:
        logger.warning("history.csv not found at %s; skipping loss/score curves.", history_csv)

    # 2. Confusion matrix
    try:
        cm = np.asarray(confusion_matrix, dtype=np.int64)
        plot_confusion_matrix(cm, num_classes, plots_dir, task)
    except Exception:
        logger.exception("Failed to save confusion_matrix.png")

    # 3. Distance histogram
    if not predictions_df.empty and "true_class" in predictions_df.columns and selected_pred_col in predictions_df.columns:
        try:
            y_true = predictions_df["true_class"].to_numpy(dtype=np.int64)
            y_pred = predictions_df[selected_pred_col].to_numpy(dtype=np.int64)
            plot_distance_histogram(y_true, y_pred, num_classes, plots_dir, task)
        except Exception:
            logger.exception("Failed to save distance_histogram.png")
    else:
        logger.warning("Insufficient predictions data for distance_histogram.png; skipping.")

    # 4. Per-node examples
    try:
        plot_node_examples(
            predictions_df,
            num_classes=num_classes,
            plots_dir=plots_dir,
            task=task,
            pred_col=selected_pred_col,
        )
    except Exception:
        logger.exception("Failed to save node_example plots")
