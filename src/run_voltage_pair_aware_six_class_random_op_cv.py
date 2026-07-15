#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Randomized whole-OP cross-validation for the learned six-class Voltage GNN.

Each fold holds out complete operating points. The model predicts classes 0..5;
class 5 is learned, not inserted with a deterministic post-processing rule.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

SRC_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SRC_DIR / "training_voltage_pair_aware_six_class.py"
PLOT_SCRIPT = SRC_DIR / "plot_voltage_pair_aware_six_class.py"
NUM_CLASSES = 6
DISCONNECTED_CLASS = 5

TOTAL_GENERATOR_MW = {
    "operating_point_1": 11506.0,
    "operating_point_2": 10398.3,
    "operating_point_3": 9359.5,
    "operating_point_4": 8309.6,
    "operating_point_5": 7783.4,
    "operating_point_6": 7241.1,
    "operating_point_7": 6169.1,
    "operating_point_8": 5658.8,
    "operating_point_9": 5207.7,
    "operating_point_10": 4609.8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random whole-OP CV for pair-aware learned six-class Voltage GNN"
    )
    parser.add_argument("--config", type=Path, default=SRC_DIR.parent / "config.yaml")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ops",
        nargs="+",
        default=[f"operating_point_{idx}" for idx in range(1, 11)],
    )
    parser.add_argument("--variants", nargs="+", choices=["none", "aggregate"], default=["none"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--validation-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=180)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--node-id-dim", type=int, default=24)
    parser.add_argument("--contingency-id-dim", type=int, default=32)
    parser.add_argument("--pair-dim", type=int, default=32)
    parser.add_argument("--op-context-embedding-dim", type=int, default=32)
    parser.add_argument("--gnn-layers", type=int, default=3)
    parser.add_argument("--decoder-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)

    parser.add_argument("--classification-weight", type=float, default=1.0)
    parser.add_argument("--regression-weight", type=float, default=0.30)
    parser.add_argument("--inactive-gate-weight", type=float, default=0.20)
    parser.add_argument("--ordinal-weight", type=float, default=0.10)
    parser.add_argument("--class-weight-mode", choices=["none", "sqrt_inverse"], default="sqrt_inverse")
    parser.add_argument("--gate-pos-weight-mode", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--gate-threshold", type=float, default=0.50)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument("--resume", action="store_true")
    return parser.parse_args()


def normalize_op(value: str) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text.startswith("op") and text[2:].isdigit():
        return f"operating_point_{int(text[2:])}"
    if text.startswith("operating_point_") and text.rsplit("_", 1)[-1].isdigit():
        return f"operating_point_{int(text.rsplit('_', 1)[-1])}"
    raise ValueError(f"Unsupported operating-point name: {value}")


def op_number(op: str) -> int:
    return int(normalize_op(op).rsplit("_", 1)[-1])


def variant_label(variant: str) -> str:
    return "Pair-aware six-class GNN" if variant == "none" else "Pair-aware six-class GNN + OP context"


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def prepare_root(path: Path, *, overwrite: bool, resume: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        if overwrite:
            shutil.rmtree(path)
        elif not resume:
            raise FileExistsError(f"Output directory is not empty: {path}. Use --overwrite or --resume.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_split_plan(
    ops: list[str], *, folds: int, repeats: int, validation_count: int, seed: int
) -> list[dict[str, Any]]:
    if folds < 2 or folds > len(ops):
        raise ValueError("--folds must be between 2 and the number of OPs")
    if validation_count < 1:
        raise ValueError("--validation-count must be at least 1")
    plan: list[dict[str, Any]] = []
    for repeat_idx in range(1, repeats + 1):
        shuffled = list(ops)
        random.Random(seed + 10000 * (repeat_idx - 1)).shuffle(shuffled)
        test_folds = [list(chunk) for chunk in np.array_split(np.asarray(shuffled, dtype=object), folds)]
        for fold_idx, test_array in enumerate(test_folds, start=1):
            test_ops = [str(op) for op in test_array]
            remaining = [op for op in shuffled if op not in set(test_ops)]
            if len(remaining) <= validation_count:
                raise ValueError("Not enough OPs left for training and validation")
            fold_seed = seed + 10000 * (repeat_idx - 1) + fold_idx
            validation_ops = random.Random(fold_seed + 1000).sample(remaining, validation_count)
            train_ops = [op for op in remaining if op not in set(validation_ops)]
            train_power = [TOTAL_GENERATOR_MW.get(op, math.nan) for op in train_ops]
            finite_train = [value for value in train_power if np.isfinite(value)]
            test_power = [TOTAL_GENERATOR_MW.get(op, math.nan) for op in test_ops]
            within_range = None
            if finite_train and all(np.isfinite(test_power)):
                low, high = min(finite_train), max(finite_train)
                within_range = all(low <= value <= high for value in test_power)
            plan.append(
                {
                    "repeat": repeat_idx,
                    "fold": fold_idx,
                    "seed": fold_seed,
                    "train_ops": train_ops,
                    "validation_ops": validation_ops,
                    "test_ops": test_ops,
                    "train_generator_mw_min": min(finite_train) if finite_train else None,
                    "train_generator_mw_max": max(finite_train) if finite_train else None,
                    "test_generator_mw": {op: TOTAL_GENERATOR_MW.get(op) for op in test_ops},
                    "all_test_ops_within_train_mw_range": within_range,
                }
            )
    return plan


def common_args(args: argparse.Namespace, fold_seed: int) -> list[str]:
    return [
        "--config", str(args.config.expanduser().resolve()),
        "--data-path", str(args.data_path.expanduser().resolve()),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--seed", str(fold_seed),
        "--hidden-dim", str(args.hidden_dim),
        "--node-id-dim", str(args.node_id_dim),
        "--contingency-id-dim", str(args.contingency_id_dim),
        "--pair-dim", str(args.pair_dim),
        "--op-context-embedding-dim", str(args.op_context_embedding_dim),
        "--gnn-layers", str(args.gnn_layers),
        "--decoder-hidden-dim", str(args.decoder_hidden_dim),
        "--dropout", str(args.dropout),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--classification-weight", str(args.classification_weight),
        "--regression-weight", str(args.regression_weight),
        "--inactive-gate-weight", str(args.inactive_gate_weight),
        "--ordinal-weight", str(args.ordinal_weight),
        "--class-weight-mode", args.class_weight_mode,
        "--gate-pos-weight-mode", args.gate_pos_weight_mode,
        "--gate-threshold", str(args.gate_threshold),
        "--overwrite",
    ]


def launch_fold(args: argparse.Namespace, *, run_dir: Path, variant: str, split: dict[str, Any]) -> None:
    scenario = f"Random OP CV repeat {split['repeat']} fold {split['fold']} — {variant_label(variant)}"
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--output-dir", str(run_dir),
        "--scenario-name", scenario,
        "--op-context-mode", variant,
        "--train-ops", *split["train_ops"],
        "--validation-ops", *split["validation_ops"],
        "--test-ops", *split["test_ops"],
        *common_args(args, int(split["seed"])),
    ]
    run(command)
    run([sys.executable, str(PLOT_SCRIPT), "--run-dir", str(run_dir), "--dpi", str(args.dpi)])


def load_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing completed run summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def prediction_column(selected_output: str) -> str:
    mapping = {"class": "class_prediction", "gated": "gated_prediction", "log_kpi": "log_kpi_prediction"}
    if selected_output not in mapping:
        raise KeyError(f"Unknown selected output: {selected_output}")
    return mapping[selected_output]


def metrics(y_true: np.ndarray, y_pred: np.ndarray, *, labels: list[int]) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    present = support > 0
    diff = y_pred - y_true
    offsets = range(-(NUM_CLASSES - 1), NUM_CLASSES)
    return {
        "total": int(y_true.size),
        "correct": int((diff == 0).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true.size else 0.0,
        "balanced_accuracy": float(recall[present].mean()) if np.any(present) else 0.0,
        "macro_f1": float(f1[present].mean()) if np.any(present) else 0.0,
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)) if y_true.size else 0.0,
        "mae": float(np.mean(np.abs(diff))) if y_true.size else 0.0,
        "mean_signed_error": float(np.mean(diff)) if y_true.size else 0.0,
        "under_rate": float(np.mean(diff < 0)) if y_true.size else 0.0,
        "over_rate": float(np.mean(diff > 0)) if y_true.size else 0.0,
        "within_one_accuracy": float(np.mean(np.abs(diff) <= 1)) if y_true.size else 0.0,
        "class_correct": [int(((y_true == cls) & (y_pred == cls)).sum()) for cls in labels],
        "class_precision": precision.tolist(),
        "class_accuracy": recall.tolist(),
        "class_recall": recall.tolist(),
        "class_f1": f1.tolist(),
        "class_support": support.astype(int).tolist(),
        "error_offset_count": {str(offset): int((diff == offset).sum()) for offset in offsets},
        "error_offset_rate": {str(offset): float(np.mean(diff == offset)) if y_true.size else 0.0 for offset in offsets},
    }


def fold_record(variant: str, split: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    six = summary["selected_test_six_class"]
    activity = summary["selected_test_activity"]
    return {
        "variant": variant,
        "model": variant_label(variant),
        "repeat": split["repeat"],
        "fold": split["fold"],
        "seed": split["seed"],
        "train_ops": ",".join(split["train_ops"]),
        "validation_ops": ",".join(split["validation_ops"]),
        "test_ops": ",".join(split["test_ops"]),
        "all_test_ops_within_train_mw_range": split["all_test_ops_within_train_mw_range"],
        "selected_output": summary["selected_output"],
        "best_epoch": summary["best_epoch"],
        "six_class_total": six["total"],
        "six_class_accuracy": six["accuracy"],
        "six_class_balanced_accuracy": six["balanced_accuracy"],
        "six_class_macro_f1": six["macro_f1"],
        "six_class_weighted_f1": six["weighted_f1"],
        "six_class_under_rate": six["under_rate"],
        "six_class_over_rate": six["over_rate"],
        "six_class_mae": six["mae"],
        "class5_support": six["class_support"][DISCONNECTED_CLASS],
        "class5_accuracy": six["class_accuracy"][DISCONNECTED_CLASS],
        "activity_0_to_4_total": activity["total"],
        "activity_0_to_4_accuracy": activity["accuracy"],
        "activity_0_to_4_balanced_accuracy": activity["balanced_accuracy"],
        "activity_0_to_4_macro_f1": activity["macro_f1"],
    }


def collect_predictions(
    run_dir: Path, *, variant: str, split: dict[str, Any], summary: dict[str, Any]
) -> pd.DataFrame:
    path = run_dir / "test_outputs" / "test_predictions.csv"
    frame = pd.read_csv(path)
    selected = str(summary["selected_output"])
    pred_col = prediction_column(selected)
    required = {"operating_point", "true_class", pred_col}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"{path} is missing columns: {sorted(missing)}")
    columns = [
        "operating_point", "contingency", "node_token", "true_class",
        "structural_class5_target", pred_col,
    ]
    out = frame[columns].copy()
    out = out.rename(columns={pred_col: "selected_prediction"})
    out["selected_error_offset"] = out["selected_prediction"] - out["true_class"]
    out["variant"] = variant
    out["repeat"] = int(split["repeat"])
    out["fold"] = int(split["fold"])
    out["selected_output"] = selected
    return out


def per_op_records(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for (variant, repeat, op), group in predictions.groupby(
        ["variant", "repeat", "operating_point"], sort=False
    ):
        y_true = group["true_class"].to_numpy(dtype=np.int64)
        y_pred = group["selected_prediction"].to_numpy(dtype=np.int64)
        six = metrics(y_true, y_pred, labels=list(range(NUM_CLASSES)))
        activity_mask = y_true < DISCONNECTED_CLASS
        activity = metrics(y_true[activity_mask], y_pred[activity_mask], labels=list(range(NUM_CLASSES)))
        record = {
            "variant": variant,
            "model": variant_label(str(variant)),
            "repeat": int(repeat),
            "operating_point": op,
            "op_number": op_number(str(op)),
            "total_generator_mw": TOTAL_GENERATOR_MW.get(str(op)),
            "selected_output": str(group["selected_output"].iloc[0]),
            "six_class_total": six["total"],
            "six_class_accuracy": six["accuracy"],
            "six_class_balanced_accuracy": six["balanced_accuracy"],
            "six_class_macro_f1": six["macro_f1"],
            "six_class_under_rate": six["under_rate"],
            "six_class_over_rate": six["over_rate"],
            "six_class_mae": six["mae"],
            "class0_support": six["class_support"][0],
            "class0_accuracy": six["class_accuracy"][0],
            "class5_support": six["class_support"][5],
            "class5_accuracy": six["class_accuracy"][5],
            "activity_0_to_4_accuracy": activity["accuracy"],
        }
        records.append(record)
    return records


def per_class_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant, group in predictions.groupby("variant", sort=False):
        y_true = group["true_class"].to_numpy(dtype=np.int64)
        y_pred = group["selected_prediction"].to_numpy(dtype=np.int64)
        result = metrics(y_true, y_pred, labels=list(range(NUM_CLASSES)))
        for cls in range(NUM_CLASSES):
            rows.append(
                {
                    "variant": variant,
                    "model": variant_label(str(variant)),
                    "class": cls,
                    "support": result["class_support"][cls],
                    "correct": result["class_correct"][cls],
                    "class_accuracy": result["class_accuracy"][cls],
                    "precision": result["class_precision"][cls],
                    "recall": result["class_recall"][cls],
                    "f1": result["class_f1"][cls],
                }
            )
    return pd.DataFrame(rows)


def offset_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant, group in predictions.groupby("variant", sort=False):
        diff = group["selected_prediction"].to_numpy(dtype=np.int64) - group["true_class"].to_numpy(dtype=np.int64)
        total = int(diff.size)
        for offset in range(-(NUM_CLASSES - 1), NUM_CLASSES):
            count = int((diff == offset).sum())
            rows.append(
                {
                    "variant": variant,
                    "model": variant_label(str(variant)),
                    "error_offset_pred_minus_true": offset,
                    "label": f"{offset:+d}" if offset != 0 else "0 (exact)",
                    "count": count,
                    "rate": count / total if total else 0.0,
                }
            )
    return pd.DataFrame(rows)


def make_plots(output_root: Path, folds: pd.DataFrame, per_op: pd.DataFrame, offsets: pd.DataFrame, dpi: int) -> None:
    figures = output_root / "comparison_figures"
    figures.mkdir(parents=True, exist_ok=True)
    for variant, subset in folds.groupby("variant", sort=False):
        labels = [f"R{r}F{f}" for r, f in zip(subset["repeat"], subset["fold"])]
        x = np.arange(len(subset))
        width = 0.25
        fig, ax = plt.subplots(figsize=(max(9, len(subset) * 1.4), 5))
        ax.bar(x - width, subset["six_class_accuracy"], width, label="Accuracy")
        ax.bar(x, subset["six_class_balanced_accuracy"], width, label="Balanced accuracy")
        ax.bar(x + width, subset["six_class_macro_f1"], width, label="Macro-F1")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Score")
        ax.set_title(f"Random OP-level learned six-class CV — {variant_label(str(variant))}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / f"{variant}_fold_metrics.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        op_subset = per_op[per_op["variant"] == variant].sort_values("total_generator_mw", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(op_subset["total_generator_mw"], op_subset["six_class_accuracy"], marker="o", label="Six-class accuracy")
        ax.plot(op_subset["total_generator_mw"], op_subset["class5_accuracy"], marker="o", label="Class-5 accuracy")
        for _, row in op_subset.iterrows():
            ax.annotate(f"OP{int(row['op_number'])}", (row["total_generator_mw"], row["six_class_accuracy"]))
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Total generator power (MW)")
        ax.set_ylabel("Score")
        ax.set_title(f"Per-OP held-out performance — {variant_label(str(variant))}")
        ax.invert_xaxis()
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / f"{variant}_per_op_by_loading.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        off = offsets[offsets["variant"] == variant]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(off["label"], off["rate"])
        ax.set_xlabel("Prediction − true class")
        ax.set_ylabel("Rate")
        ax.set_title(f"Pooled exact ordinal errors — {variant_label(str(variant))}")
        fig.tight_layout()
        fig.savefig(figures / f"{variant}_pooled_error_offsets.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    ops = list(dict.fromkeys(normalize_op(op) for op in args.ops))
    variants = list(dict.fromkeys(args.variants))
    output_root = prepare_root(args.output_dir, overwrite=args.overwrite, resume=args.resume)

    plan = build_split_plan(
        ops, folds=args.folds, repeats=args.repeats,
        validation_count=args.validation_count, seed=args.seed,
    )
    (output_root / "random_op_split_plan.json").write_text(
        json.dumps(
            {
                "description": "Whole-OP randomized CV with learned classes 0..5; no deterministic class-5 override.",
                "ops": ops,
                "folds": args.folds,
                "repeats": args.repeats,
                "validation_count": args.validation_count,
                "base_seed": args.seed,
                "total_generator_mw_reporting_only": TOTAL_GENERATOR_MW,
                "splits": plan,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fold_records: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for variant in variants:
        for split in plan:
            run_dir = output_root / variant / f"repeat_{int(split['repeat']):02d}" / f"fold_{int(split['fold']):02d}"
            completed = run_dir / "run_summary.json"
            if args.resume and completed.exists():
                print(f"Skipping completed fold: {run_dir}")
            else:
                launch_fold(args, run_dir=run_dir, variant=variant, split=split)
            summary = load_summary(run_dir)
            fold_records.append(fold_record(variant, split, summary))
            prediction_frames.append(collect_predictions(run_dir, variant=variant, split=split, summary=summary))

    folds_frame = pd.DataFrame(fold_records)
    folds_frame.to_csv(output_root / "random_op_cv_fold_summary.csv", index=False)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(output_root / "random_op_cv_all_six_class_predictions.csv", index=False)
    per_op = pd.DataFrame(per_op_records(predictions)).sort_values(["variant", "repeat", "op_number"])
    per_op.to_csv(output_root / "random_op_cv_per_op_metrics.csv", index=False)
    per_class = per_class_frame(predictions)
    per_class.to_csv(output_root / "random_op_cv_per_class_metrics_0_to_5.csv", index=False)
    offsets = offset_frame(predictions)
    offsets.to_csv(output_root / "random_op_cv_error_offset_distribution.csv", index=False)

    pooled_records: list[dict[str, Any]] = []
    stats_records: list[dict[str, Any]] = []
    for variant in variants:
        subset = predictions[predictions["variant"] == variant]
        y_true = subset["true_class"].to_numpy(dtype=np.int64)
        y_pred = subset["selected_prediction"].to_numpy(dtype=np.int64)
        six = metrics(y_true, y_pred, labels=list(range(NUM_CLASSES)))
        activity_mask = y_true < DISCONNECTED_CLASS
        activity = metrics(y_true[activity_mask], y_pred[activity_mask], labels=list(range(NUM_CLASSES)))
        pooled_records.append(
            {
                "variant": variant,
                "model": variant_label(variant),
                "learned_six_class_0_to_5": six,
                "activity_subset_true_0_to_4": activity,
            }
        )
        variant_folds = folds_frame[folds_frame["variant"] == variant]
        for metric_name in (
            "six_class_accuracy", "six_class_balanced_accuracy", "six_class_macro_f1",
            "six_class_weighted_f1", "six_class_under_rate", "six_class_over_rate",
            "six_class_mae", "class5_accuracy", "activity_0_to_4_accuracy",
        ):
            values = variant_folds[metric_name].to_numpy(dtype=float)
            stats_records.append(
                {
                    "variant": variant,
                    "model": variant_label(variant),
                    "metric": metric_name,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "n_folds": int(len(values)),
                }
            )

    (output_root / "random_op_cv_pooled_summary.json").write_text(
        json.dumps(pooled_records, indent=2), encoding="utf-8"
    )
    pd.DataFrame(stats_records).to_csv(output_root / "random_op_cv_statistics.csv", index=False)
    make_plots(output_root, folds_frame, per_op, offsets, args.dpi)

    report = [
        "# Random operating-point cross-validation — learned six classes",
        "",
        "All six classes (0–5) are predicted by the GNN. Class 5 is not inserted deterministically.",
        "All splits are performed at complete operating-point level.",
        "",
    ]
    for record in pooled_records:
        six = record["learned_six_class_0_to_5"]
        report.extend(
            [
                f"## {record['model']}",
                "",
                f"- Pooled six-class accuracy: {six['accuracy']:.2%}",
                f"- Balanced accuracy: {six['balanced_accuracy']:.2%}",
                f"- Macro-F1: {six['macro_f1']:.2%}",
                f"- Class-5 accuracy: {six['class_accuracy'][5]:.2%} ({six['class_correct'][5]}/{six['class_support'][5]})",
                f"- Underprediction: {six['under_rate']:.2%}",
                f"- Overprediction: {six['over_rate']:.2%}",
                "",
            ]
        )
    (output_root / "supervisor_results.md").write_text("\n".join(report), encoding="utf-8")

    print("\nRandom OP-level learned six-class cross-validation completed")
    for record in pooled_records:
        six = record["learned_six_class_0_to_5"]
        print(
            f"{record['model']}: acc={six['accuracy']:.4f} bal={six['balanced_accuracy']:.4f} "
            f"macroF1={six['macro_f1']:.4f}"
        )
        for cls in range(NUM_CLASSES):
            print(
                f"  CLASS {cls}: {six['class_correct'][cls]}/{six['class_support'][cls]} "
                f"accuracy={six['class_accuracy'][cls]:.4f}"
            )
        parts = []
        for offset in range(-(NUM_CLASSES - 1), NUM_CLASSES):
            key = str(offset)
            label = f"{offset:+d}" if offset != 0 else "0"
            parts.append(f"{label}:{six['error_offset_count'][key]} ({six['error_offset_rate'][key]:.2%})")
        print("  OFFSETS pred-true | " + " | ".join(parts))

    print(f"Split plan: {output_root / 'random_op_split_plan.json'}")
    print(f"Fold summary: {output_root / 'random_op_cv_fold_summary.csv'}")
    print(f"Per-OP metrics: {output_root / 'random_op_cv_per_op_metrics.csv'}")
    print(f"Per-class metrics: {output_root / 'random_op_cv_per_class_metrics_0_to_5.csv'}")
    print(f"Error offsets: {output_root / 'random_op_cv_error_offset_distribution.csv'}")
    print(f"Pooled summary: {output_root / 'random_op_cv_pooled_summary.json'}")


if __name__ == "__main__":
    main()
