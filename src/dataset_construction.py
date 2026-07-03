# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: Training dataset construction from KPI tables

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.paths import (
    ACTIONS_DIR,
    CONFIG_PATH,
    DATASET_DIR,
    DISCONNECTIONS_DIR,
    KPI_DIR,
)
from modules.pipeline_logging import get_logger, log_step_banner

ID_COLS = ["OP", "Contingency"]


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def raw_cuts_from_config(config: Mapping[str, object], key: str) -> np.ndarray:
    """Read strictly increasing raw KPI cut thresholds from ``kpi.class_bins.<key>.cuts``."""
    section = ((config.get("kpi") or {}).get("class_bins") or {}).get(key, {})
    if isinstance(section, dict):
        raw_cuts = section.get("cuts", [])
    elif isinstance(section, list):
        raw_cuts = section
    else:
        raise ValueError(
            f"Expected kpi.class_bins.{key} to be an object with 'cuts' or a list, "
            f"got {type(section).__name__}"
        )

    if not isinstance(raw_cuts, list) or not raw_cuts:
        raise ValueError(f"kpi.class_bins.{key}.cuts must be a non-empty list of raw KPI thresholds")

    arr = np.asarray(raw_cuts, dtype=float)
    if np.any(arr <= 0) or np.any(np.diff(arr) <= 0):
        raise ValueError(
            f"kpi.class_bins.{key}.cuts must be strictly increasing positive raw KPI thresholds: {arr}"
        )
    return arr


def kpi_values_to_class_array(kpi_arr: np.ndarray, raw_cuts: np.ndarray) -> np.ndarray:
    out = np.full(kpi_arr.shape, np.nan, dtype=float)
    finite = np.isfinite(kpi_arr)
    if finite.any():
        out[finite] = np.searchsorted(raw_cuts, kpi_arr[finite], side="left")
    return out


def apply_flag_mask_to_matching_rows(
    target_df: pd.DataFrame,
    flag_df: Optional[pd.DataFrame],
    replacement_value: object,
) -> pd.DataFrame:
    if target_df is None or target_df.empty or flag_df is None or flag_df.empty:
        return target_df

    target_cols = [str(c).strip() for c in target_df.columns]
    flag_cols = [str(c).strip() for c in flag_df.columns]
    out = target_df.copy()
    out.columns = target_cols
    flags = flag_df.copy()
    flags.columns = flag_cols

    if any(c not in out.columns for c in ID_COLS) or any(c not in flags.columns for c in ID_COLS):
        return out

    for id_col in ID_COLS:
        out[id_col] = out[id_col].astype("string").str.strip()
        flags[id_col] = flags[id_col].astype("string").str.strip()

    value_cols = [c for c in out.columns if c not in ID_COLS and c in flags.columns]
    if not value_cols:
        return out

    target_keyed = out[ID_COLS].copy()
    target_keyed["_occ"] = target_keyed.groupby(ID_COLS, dropna=False).cumcount()
    target_keyed["_row_idx"] = np.arange(len(out))

    flags_keyed = flags[ID_COLS].copy()
    flags_keyed["_occ"] = flags_keyed.groupby(ID_COLS, dropna=False).cumcount()
    flags_keyed = pd.concat([flags_keyed, flags[value_cols]], axis=1)

    merged = target_keyed.merge(flags_keyed, on=[*ID_COLS, "_occ"], how="left")

    for col in value_cols:
        is_flagged = pd.to_numeric(merged[col], errors="coerce").eq(1)
        if not is_flagged.any():
            continue
        row_indices = merged.loc[is_flagged, "_row_idx"].to_numpy(dtype=int, copy=False)
        if row_indices.size == 0:
            continue
        col_pos = out.columns.get_loc(col)
        out.iloc[row_indices, col_pos] = replacement_value

    return out


def _kpi_cut_column_name(index: int) -> str:
    return f"kpi_cut_{index}"


def build_class_bins_row(
    *,
    kpi_type: str,
    raw_cuts: np.ndarray,
    action_disconnect_class: int,
) -> pd.DataFrame:
    row: dict[str, object] = {"kpi_type": kpi_type}
    for idx, cut in enumerate(raw_cuts):
        row[_kpi_cut_column_name(idx)] = float(cut)
    n_kpi_classes = int(len(raw_cuts) + 1)
    row["n_kpi_classes"] = n_kpi_classes
    row["action_disconnect_class"] = int(action_disconnect_class)
    row["n_classes"] = int(action_disconnect_class) + 1
    return pd.DataFrame([row])


def save_class_bins_report(rows: Sequence[pd.DataFrame], output_path: Path) -> Path:
    report = pd.concat(list(rows), ignore_index=True, sort=False)
    cut_cols = sorted(c for c in report.columns if str(c).startswith("kpi_cut_"))
    ordered = ["kpi_type"] + cut_cols + [
        c for c in ("n_kpi_classes", "action_disconnect_class", "n_classes") if c in report.columns
    ]
    report = report.reindex(columns=ordered)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    return output_path


def build_class_dataset_for_type(
    *,
    kpi_type: str,
    kpi_df: pd.DataFrame,
    action_df: pd.DataFrame,
    disconnection_df: Optional[pd.DataFrame],
    raw_cuts: np.ndarray,
    output_dataset_path: Path,
    apply_disconnection_mask: bool,
) -> tuple[Path, pd.DataFrame]:
    if kpi_df is None or kpi_df.empty:
        raise ValueError(f"KPI table is empty for {kpi_type}")

    kpi_df = kpi_df.copy()
    kpi_df.columns = [str(c).strip() for c in kpi_df.columns]
    for col in ID_COLS:
        if col not in kpi_df.columns:
            raise ValueError(f"KPI table missing column {col!r} for {kpi_type}")

    value_cols = [c for c in kpi_df.columns if c not in ID_COLS]
    if not value_cols:
        raise ValueError(f"No value columns in KPI table for {kpi_type}")

    raw_cuts = np.asarray(raw_cuts, dtype=float)
    n_kpi_classes = int(raw_cuts.size) + 1
    action_disconnect_class = n_kpi_classes

    values = kpi_df[value_cols].to_numpy(dtype=float)
    class_df = kpi_df[ID_COLS].copy()
    for j, col in enumerate(value_cols):
        classes = kpi_values_to_class_array(values[:, j], raw_cuts)
        class_df[col] = pd.Series(classes, dtype="Int64")

    class_df = apply_flag_mask_to_matching_rows(
        class_df,
        action_df,
        replacement_value=action_disconnect_class,
    )

    if apply_disconnection_mask and disconnection_df is not None:
        class_df = apply_flag_mask_to_matching_rows(
            class_df,
            disconnection_df,
            replacement_value=action_disconnect_class,
        )

    output_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    class_df.to_csv(output_dataset_path, index=False)

    bins_row = build_class_bins_row(
        kpi_type=kpi_type,
        raw_cuts=raw_cuts,
        action_disconnect_class=action_disconnect_class,
    )
    return output_dataset_path, bins_row


def kpi_class_counts(csv_path: Path, n_classes: int, *, chunk_rows: int = 800) -> pd.DataFrame:
    header = pd.read_csv(csv_path, nrows=0)
    value_cols = [c for c in header.columns if c not in ID_COLS]
    counts = np.zeros(n_classes, dtype=np.int64)

    for chunk in pd.read_csv(csv_path, chunksize=chunk_rows, usecols=value_cols, low_memory=False):
        arr = chunk.to_numpy(dtype=float, copy=False)
        vals = np.rint(arr[np.isfinite(arr)]).astype(np.int64)
        vals = vals[(vals >= 0) & (vals < n_classes)]
        if vals.size:
            counts += np.bincount(vals, minlength=n_classes)

    return pd.DataFrame({"class": np.arange(n_classes), "count": counts})


def plot_voltage_spower_distribution(
    voltage_tbl: pd.DataFrame,
    spower_tbl: pd.DataFrame,
    *,
    title: str,
    n_classes: int,
    output_path: Path,
) -> Path:
    import matplotlib.pyplot as plt

    classes = np.arange(n_classes)
    bar_width = 0.46
    bar_offset = 0.25

    fig, ax = plt.subplots(figsize=(8, 4))
    voltage_bars = ax.bar(
        classes - bar_offset,
        voltage_tbl["count"],
        width=bar_width,
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
        label="Voltage",
    )
    spower_bars = ax.bar(
        classes + bar_offset,
        spower_tbl["count"],
        width=bar_width,
        color="darkorange",
        edgecolor="black",
        linewidth=0.5,
        label="Spower",
    )

    ax.set_xlabel("Class", fontsize=20)
    ax.set_ylabel("Count", fontsize=20)
    ax.set_title(title, fontsize=18)
    ax.set_xticks(classes)
    ax.tick_params(axis="both", labelsize=18)
    ax.legend(fontsize=18)
    ax.grid(axis="y", alpha=0.35)

    def _format_count(value: float) -> str:
        return f"{value:.1e}" if value > 1e6 else f"{value:.0f}"

    voltage_labels = [_format_count(v) for v in voltage_tbl["count"]]
    spower_labels = [_format_count(v) for v in spower_tbl["count"]]

    last_cls = n_classes - 1
    if voltage_tbl.loc[voltage_tbl["class"] == last_cls, "count"].iloc[0] > 1e6:
        voltage_labels[last_cls] = ""

    ax.bar_label(voltage_bars, labels=voltage_labels, padding=4, fontsize=14, rotation=90)
    ax.bar_label(spower_bars, labels=spower_labels, padding=4, fontsize=14, rotation=90)

    if voltage_labels[last_cls] == "":
        class_last_bar = voltage_bars[last_cls]
        ax.annotate(
            _format_count(voltage_tbl.loc[voltage_tbl["class"] == last_cls, "count"].iloc[0]),
            xy=(class_last_bar.get_x() + class_last_bar.get_width() / 2, class_last_bar.get_height()),
            xytext=(0, 22),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=14,
            rotation=90,
        )

    _, y_top = ax.get_ylim()
    ax.set_ylim(top=y_top * 1.4)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _read_required_table(path: Path, *, stage: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name} at {path}. Run the {stage} stage first (main.py --to-step {stage})."
        )
    return pd.read_csv(path)


def build_datasets() -> dict[str, Optional[Path]]:
    config = load_config()

    kpi_voltage_path = KPI_DIR / "KPI_voltage.csv"
    kpi_spower_path = KPI_DIR / "KPI_spower.csv"
    actions_voltage_path = ACTIONS_DIR / "ACTIONS_voltage.csv"
    actions_spower_path = ACTIONS_DIR / "ACTIONS_spower.csv"
    disc_voltage_path = DISCONNECTIONS_DIR / "DISC_voltage.csv"
    disc_spower_path = DISCONNECTIONS_DIR / "DISC_spower.csv"

    masked_voltage = _read_required_table(kpi_voltage_path, stage="curve_process")
    masked_spower = _read_required_table(kpi_spower_path, stage="curve_process")
    combined_actions_voltage = _read_required_table(actions_voltage_path, stage="curve_process")
    combined_actions_spower = _read_required_table(actions_spower_path, stage="curve_process")
    combined_disc_voltage = _read_required_table(disc_voltage_path, stage="curve_process")
    combined_disc_spower = _read_required_table(disc_spower_path, stage="curve_process")

    voltage_cuts = raw_cuts_from_config(config, "voltage")
    spower_cuts = raw_cuts_from_config(config, "spower")

    dataset_voltage_path, voltage_bins_row = build_class_dataset_for_type(
        kpi_type="voltage",
        kpi_df=masked_voltage,
        action_df=combined_actions_voltage,
        disconnection_df=combined_disc_voltage,
        raw_cuts=voltage_cuts,
        output_dataset_path=DATASET_DIR / "Dataset_Voltage.csv",
        apply_disconnection_mask=True,
    )

    dataset_spower_path, spower_bins_row = build_class_dataset_for_type(
        kpi_type="spower",
        kpi_df=masked_spower,
        action_df=combined_actions_spower,
        disconnection_df=combined_disc_spower,
        raw_cuts=spower_cuts,
        output_dataset_path=DATASET_DIR / "Dataset_Spower.csv",
        apply_disconnection_mask=True,
    )

    class_bins_path = save_class_bins_report(
        [voltage_bins_row, spower_bins_row],
        DATASET_DIR / "KPI_class_bins.csv",
    )

    n_classes_voltage = int(voltage_bins_row["n_classes"].iloc[0])
    n_classes_spower = int(spower_bins_row["n_classes"].iloc[0])
    n_classes_plot = max(n_classes_voltage, n_classes_spower)
    voltage_counts = kpi_class_counts(dataset_voltage_path, n_classes_plot)
    spower_counts = kpi_class_counts(dataset_spower_path, n_classes_plot)
    distribution_plot_path = plot_voltage_spower_distribution(
        voltage_counts,
        spower_counts,
        title="Dataset_Voltage / Dataset_Spower",
        n_classes=n_classes_plot,
        output_path=DATASET_DIR / "dataset_class_distribution.png",
    )

    return {
        "actions_voltage": actions_voltage_path,
        "actions_spower": actions_spower_path,
        "disconnections_voltage": disc_voltage_path,
        "disconnections_spower": disc_spower_path,
        "kpi_voltage": kpi_voltage_path,
        "kpi_spower": kpi_spower_path,
        "class_bins": class_bins_path,
        "dataset_voltage": dataset_voltage_path,
        "dataset_spower": dataset_spower_path,
        "distribution_plot": distribution_plot_path,
    }


def main() -> None:
    log_step_banner("dataset_construction")
    logger = get_logger()

    outputs = build_datasets()

    logger.info("Dataset construction finished.")
    for name, path in outputs.items():
        logger.info("%s: %s", name, path)
