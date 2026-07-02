# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: Training dataset construction from KPI tables

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules import dataset_split
from modules.paths import (
    ACTIONS_DIR,
    CONFIG_PATH,
    DATASET_DIR,
    DISCONNECTIONS_DIR,
    KPI_DIR,
    OP_GRAPHS_DIR,
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


def op_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def normalize_operating_point_name(value: object) -> Optional[str]:
    if value is None:
        return None
    txt = str(value).strip()
    if not txt or txt.lower() == "nan":
        return None
    if txt.startswith("operating_point_"):
        return txt

    match = re.search(r"(\d+)", txt)
    if not match:
        return None
    return f"operating_point_{int(match.group(1))}"


def extract_operating_point_name_from_filename(path: Path, prefix: str) -> Optional[str]:
    suffix = path.stem.split(prefix, 1)[-1]
    return normalize_operating_point_name(suffix)


def _torch_load_compat(torch_module, path: Path):
    try:
        return torch_module.load(path, map_location="cpu")
    except TypeError:
        return torch_module.load(path)
    except Exception as exc:
        try:
            return torch_module.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            raise exc


def load_op_graph_component_ids(op_name: str) -> Optional[set[str]]:
    graph_path = OP_GRAPHS_DIR / f"{op_name}.pt"
    if not graph_path.exists():
        return None

    try:
        import torch  # type: ignore
    except Exception:
        return None

    try:
        payload = _torch_load_compat(torch, graph_path)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None

    node_meta = metadata.get("node_metadata")
    if not isinstance(node_meta, dict):
        return None

    present: set[str] = set()
    for node_id, meta in node_meta.items():
        if node_id is not None:
            present.add(str(node_id).strip())
        if not isinstance(meta, dict):
            continue
        vl_id = meta.get("voltageLevelId")
        if vl_id:
            present.add(str(vl_id).strip())
        busbars = meta.get("busbarSectionIds")
        if isinstance(busbars, list):
            present.update(str(busbar).strip() for busbar in busbars if busbar)
        buses = meta.get("busIds")
        if isinstance(buses, list):
            present.update(str(bus).strip() for bus in buses if bus)

    edge_meta = metadata.get("edge_metadata")
    if isinstance(edge_meta, list):
        for item in edge_meta:
            if not isinstance(item, dict):
                continue
            edge_id = item.get("id")
            if edge_id:
                present.add(str(edge_id).strip())
            for key in ("bus1", "bus2"):
                bus_id = item.get(key)
                if bus_id:
                    present.add(str(bus_id).strip())

    return present


def filter_rows_contingency_in_op_graph(
    df: pd.DataFrame,
    op_name: str,
    graph_cache: dict[str, Optional[set[str]]],
) -> tuple[pd.DataFrame, int]:
    if df.empty or "Contingency" not in df.columns or not OP_GRAPHS_DIR.is_dir():
        return df, 0

    if op_name not in graph_cache:
        graph_cache[op_name] = load_op_graph_component_ids(op_name)

    present = graph_cache[op_name]
    if present is None:
        return df, 0

    before = len(df)

    def keep(contingency: object) -> bool:
        if contingency is None:
            return True
        try:
            if bool(pd.isna(contingency)):
                return True
        except Exception:
            pass
        contingency_id = str(contingency).strip()
        if not contingency_id or contingency_id.lower() == "nan":
            return True
        return contingency_id in present

    out = df.loc[df["Contingency"].map(keep)].copy()
    return out, before - len(out)


def collect_frames(prefix: str, source_dir: Path) -> list[pd.DataFrame]:
    paths = sorted(source_dir.glob(f"{prefix}operating_point_*.csv"), key=op_sort_key)
    frames = []
    graph_cache: dict[str, Optional[set[str]]] = {}
    for path in paths:
        df = pd.read_csv(path)
        if "OP" not in df.columns:
            op_name = path.stem.split(prefix, 1)[-1]
            df.insert(0, "OP", op_name)
        op_name = extract_operating_point_name_from_filename(path, prefix)
        if op_name is None and "OP" in df.columns:
            non_null_ops = df["OP"].dropna()
            if not non_null_ops.empty:
                op_name = normalize_operating_point_name(non_null_ops.iloc[0])
        if op_name is not None:
            df, dropped = filter_rows_contingency_in_op_graph(df, op_name, graph_cache)
            if dropped:
                print(f"{path.name}: dropped {dropped} row(s) with Contingency not in {op_name}.pt")
        frames.append(df)
    return frames


def combine_frames(prefix: str, source_dir: Path) -> Optional[pd.DataFrame]:
    frames = collect_frames(prefix, source_dir)
    if not frames:
        return None
    all_value_cols = sorted({
        col
        for df in frames
        for col in df.columns
        if col not in {"OP", "Contingency"}
    })
    aligned = [df.reindex(columns=["OP", "Contingency", *all_value_cols]) for df in frames]
    return pd.concat(aligned, ignore_index=True)


def write_table(df: Optional[pd.DataFrame], output_path: Path) -> Optional[Path]:
    if df is None:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def apply_flag_mask_to_kpi(kpi_df: Optional[pd.DataFrame], flags_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if kpi_df is None or flags_df is None:
        return kpi_df
    if "OP" not in kpi_df.columns or "Contingency" not in kpi_df.columns:
        return kpi_df
    if "OP" not in flags_df.columns or "Contingency" not in flags_df.columns:
        return kpi_df

    out = kpi_df.copy()
    flags = flags_df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    flags.columns = [str(col).strip() for col in flags.columns]

    for id_col in ("OP", "Contingency"):
        out[id_col] = out[id_col].astype("string").str.strip()
        flags[id_col] = flags[id_col].astype("string").str.strip()

    value_cols = [col for col in out.columns if col not in {"OP", "Contingency"} and col in flags.columns]
    if not value_cols:
        return out

    out_keys = out[["OP", "Contingency"]].copy()
    out_keys["_occ"] = out_keys.groupby(["OP", "Contingency"], dropna=False).cumcount()
    out_keys["_row_idx"] = np.arange(len(out))

    flag_keys = flags[["OP", "Contingency"]].copy()
    flag_keys["_occ"] = flag_keys.groupby(["OP", "Contingency"], dropna=False).cumcount()
    flag_keys = pd.concat([flag_keys, flags[value_cols]], axis=1)

    merged = out_keys.merge(flag_keys, on=["OP", "Contingency", "_occ"], how="left")
    for col in value_cols:
        is_flagged = pd.to_numeric(merged[col], errors="coerce").eq(1)
        if not is_flagged.any():
            continue
        row_indices = merged.loc[is_flagged, "_row_idx"].to_numpy(dtype=int, copy=False)
        out.loc[row_indices, col] = np.nan
    return out


def build_datasets() -> dict[str, Optional[Path]]:
    config = load_config()
    split_csv = DATASET_DIR / "train_val_test_split.csv"

    combined_voltage = combine_frames("KPI_voltage_", KPI_DIR)
    combined_spower = combine_frames("KPI_spower_", KPI_DIR)
    combined_actions_voltage = combine_frames("actions_voltage_", ACTIONS_DIR)
    combined_actions_spower = combine_frames("actions_spower_", ACTIONS_DIR)
    combined_disc_voltage = combine_frames("disconnections_voltage_", DISCONNECTIONS_DIR)
    combined_disc_spower = combine_frames("disconnections_spower_", DISCONNECTIONS_DIR)

    actions_voltage_path = write_table(combined_actions_voltage, ACTIONS_DIR / "ACTIONS_voltage.csv")
    actions_spower_path = write_table(combined_actions_spower, ACTIONS_DIR / "ACTIONS_spower.csv")
    disc_voltage_path = write_table(combined_disc_voltage, DISCONNECTIONS_DIR / "DISC_voltage.csv")
    disc_spower_path = write_table(combined_disc_spower, DISCONNECTIONS_DIR / "DISC_spower.csv")

    masked_voltage = apply_flag_mask_to_kpi(combined_voltage, combined_actions_voltage)
    masked_voltage = apply_flag_mask_to_kpi(masked_voltage, combined_disc_voltage)
    masked_spower = apply_flag_mask_to_kpi(combined_spower, combined_actions_spower)
    masked_spower = apply_flag_mask_to_kpi(masked_spower, combined_disc_spower)

    kpi_voltage_path = write_table(masked_voltage, KPI_DIR / "KPI_voltage.csv")
    kpi_spower_path = write_table(masked_spower, KPI_DIR / "KPI_spower.csv")

    if kpi_voltage_path is None:
        raise FileNotFoundError("No per-OP KPI_voltage tables found under data/KPI/")

    split_summary = dataset_split.build_dataset_split(
        kpi_voltage_path,
        output_csv=split_csv,
        config=config,
    )

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
        "split_csv": split_csv,
        "class_bins": class_bins_path,
        "dataset_voltage": dataset_voltage_path,
        "dataset_spower": dataset_spower_path,
        "distribution_plot": distribution_plot_path,
        "_split_summary": split_summary,
    }


def main() -> None:
    log_step_banner("dataset_construction")
    logger = get_logger()

    outputs = build_datasets()
    split_summary = outputs.pop("_split_summary")

    logger.info("Dataset construction finished.")
    logger.info(
        "Split built. total=%d train=%d val=%d test=%d mode=%s seed=%d",
        split_summary.total_examples,
        split_summary.train_examples,
        split_summary.validation_examples,
        split_summary.test_examples,
        split_summary.split_mode,
        split_summary.seed,
    )
    for name, path in outputs.items():
        logger.info("%s: %s", name, path)
