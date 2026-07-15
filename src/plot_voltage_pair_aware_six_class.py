#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NUM_CLASSES = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot pair-aware six-class Voltage GNN results")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def save(fig, path: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def confusion(metrics: dict, title: str, path: Path, dpi: int, normalized: bool) -> None:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
    display = matrix.copy()
    if normalized:
        denominator = display.sum(axis=1, keepdims=True)
        display = np.divide(display, denominator, out=np.zeros_like(display), where=denominator > 0)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(display, aspect="auto")
    fig.colorbar(image, ax=ax)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    ax.set_xticks(range(display.shape[1]))
    ax.set_yticks(range(display.shape[0]))
    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            text = f"{display[i, j]:.2f}" if normalized else str(int(display[i, j]))
            ax.text(j, i, text, ha="center", va="center", fontsize=8)
    save(fig, path, dpi)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    if "test" not in summary:
        raise RuntimeError(f"No test results in {run_dir / 'run_summary.json'}")

    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    test = summary["test"]
    selected = summary["selected_output"]
    selected_six = summary["selected_test_six_class"]

    labels = ["Accuracy", "Balanced accuracy", "Macro-F1", "Weighted-F1", "Within one"]
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "within_one_accuracy"]
    heads = ["class", "gated", "log_kpi"]
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, head in enumerate(heads):
        metrics = test[f"six_class_{head}"]
        ax.bar(x + (idx - 1) * width, [metrics[key] for key in keys], width, label=head)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(f"Learned six-class output-head comparison — selected={selected}")
    ax.legend()
    save(fig, figures / "six_class_output_head_comparison.png", args.dpi)

    classes = np.arange(NUM_CLASSES)
    width = 0.22
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(classes - width, selected_six["class_precision"], width, label="Precision")
    ax.bar(classes, selected_six["class_accuracy"], width, label="Class accuracy/recall")
    ax.bar(classes + width, selected_six["class_f1"], width, label="F1")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(classes)
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title("Selected output per-class metrics, including learned class 5")
    ax.legend()
    save(fig, figures / "selected_per_class_metrics_0_to_5.png", args.dpi)

    offsets = np.arange(-(NUM_CLASSES - 1), NUM_CLASSES)
    counts = [selected_six["error_offset_count"][str(int(value))] for value in offsets]
    rates = [selected_six["error_offset_rate"][str(int(value))] for value in offsets]
    labels_offset = [f"{value:+d}" if value != 0 else "0" for value in offsets]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels_offset, rates)
    ax.set_xlabel("Prediction − true class")
    ax.set_ylabel("Rate")
    ax.set_title("Exact ordinal error-offset distribution")
    for idx, (rate, count) in enumerate(zip(rates, counts)):
        if count:
            ax.text(idx, rate, str(count), ha="center", va="bottom", fontsize=8)
    save(fig, figures / "selected_error_offset_distribution.png", args.dpi)

    confusion(
        selected_six,
        f"Learned six-class confusion matrix (selected={selected})",
        figures / "selected_confusion_matrix_0_to_5.png",
        args.dpi,
        normalized=False,
    )
    confusion(
        selected_six,
        f"Normalized learned six-class confusion matrix (selected={selected})",
        figures / "selected_confusion_matrix_0_to_5_normalized.png",
        args.dpi,
        normalized=True,
    )

    history_path = run_dir / "training" / "voltage_pair_aware_six_class_gnn" / "history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        if not history.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(history["epoch"], history["train_total_loss"], label="Total")
            for name in ("classification", "regression", "gate", "ordinal"):
                column = f"train_{name}_loss"
                if column in history.columns:
                    ax.plot(history["epoch"], history[column], label=name)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Training losses")
            ax.legend()
            save(fig, figures / "training_losses.png", args.dpi)

            score_columns = [f"val_{name}_score" for name in heads if f"val_{name}_score" in history.columns]
            if score_columns:
                fig, ax = plt.subplots(figsize=(9, 5))
                for column in score_columns:
                    ax.plot(
                        history["epoch"],
                        history[column],
                        label=column.replace("val_", "").replace("_score", ""),
                    )
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Validation selection score")
                ax.set_title("Validation six-class output-head scores")
                ax.legend()
                save(fig, figures / "validation_scores.png", args.dpi)

    print(f"PNG figures saved to: {figures}")


if __name__ == "__main__":
    main()
