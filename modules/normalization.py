# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: KPI log-transform, z-score normalization, and class labeling

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ID_COLS = ["OP", "Contingency"]

_OP_NUMBER_RE = re.compile(r"(\d+)")


def normalize_operating_point_name(value: object) -> str:
    if value is None:
        return ""
    txt = str(value).strip()
    if not txt or txt.lower() == "nan":
        return ""
    if txt.startswith("operating_point_"):
        return txt
    match = _OP_NUMBER_RE.search(txt)
    if match:
        return f"operating_point_{int(match.group(1))}"
    return txt


def load_split_lookup(split_csv: Path) -> Dict[Tuple[str, str], str]:
    if not split_csv.is_file():
        raise FileNotFoundError(f"Split CSV not found: {split_csv}")

    split_df = pd.read_csv(split_csv)
    required = {"split", "operating_point", "contingency"}
    missing = required.difference(split_df.columns)
    if missing:
        raise ValueError(f"Split CSV missing columns: {sorted(missing)}")

    lookup: Dict[Tuple[str, str], str] = {}
    for row in split_df.itertuples(index=False):
        op = normalize_operating_point_name(getattr(row, "operating_point"))
        cont = str(getattr(row, "contingency")).strip()
        split_name = str(getattr(row, "split")).strip().lower()
        if op and cont:
            lookup[(op, cont)] = split_name

    valid = {"train", "validation", "test"}
    invalid = sorted(set(lookup.values()) - valid)
    if invalid:
        raise ValueError(f"Unexpected split names in {split_csv}: {invalid}")

    return lookup


def quantile_cuts_from_config(config: Mapping[str, object], key: str) -> np.ndarray:
    """Read quantile fractions in (0, 1) from ``kpi.class_bins.<key>.cuts``."""
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
        raise ValueError(
            f"kpi.class_bins.{key}.cuts must be a non-empty list of quantile fractions in (0, 1)"
        )

    arr = np.asarray(raw_cuts, dtype=float)
    if np.any(arr <= 0) or np.any(arr >= 1):
        raise ValueError(
            f"kpi.class_bins.{key}.cuts quantile fractions must be strictly between 0 and 1: {arr}"
        )
    arr.sort()
    return arr


def log_transform_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=series.index, dtype=float)
    valid = numeric.notna() & (numeric > -1.0)
    if valid.any():
        out.loc[valid] = np.log1p(numeric.loc[valid].to_numpy(dtype=float))
    return out


def log_transform_dataframe(df: pd.DataFrame, value_cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in value_cols:
        out[col] = log_transform_series(out[col])
    return out


def row_split_mask(
    df: pd.DataFrame,
    split_lookup: Dict[Tuple[str, str], str],
    split_name: str,
) -> np.ndarray:
    ops = df["OP"].map(normalize_operating_point_name)
    conts = df["Contingency"].astype("string").str.strip()
    keys = list(zip(ops.tolist(), conts.tolist()))
    return np.array([split_lookup.get(k) == split_name for k in keys], dtype=bool)


def fit_global_standard_scaler_on_train(
    values: np.ndarray,
    train_mask: np.ndarray,
) -> StandardScaler:
    """Fit one global mu and sigma on all finite log-KPI values from train rows."""
    train_values = values[train_mask]
    flat = train_values[np.isfinite(train_values)]
    if flat.size == 0:
        raise ValueError("No finite train KPI values available for global scaler fitting")

    scaler = StandardScaler()
    scaler.fit(flat.reshape(-1, 1))
    return scaler


def global_scaler_mu_sigma(scaler: StandardScaler) -> Tuple[float, float]:
    return float(scaler.mean_[0]), float(scaler.scale_[0])


def transform_values(scaler: StandardScaler, values: np.ndarray) -> np.ndarray:
    mu, sigma = global_scaler_mu_sigma(scaler)
    z = (values - mu) / sigma
    z[~np.isfinite(values)] = np.nan
    return z


def compute_z_cut_thresholds(
    z_values: np.ndarray,
    train_mask: np.ndarray,
    quantile_fractions: np.ndarray,
) -> np.ndarray:
    train_z = z_values[train_mask]
    flat = train_z[np.isfinite(train_z)]
    if flat.size == 0:
        raise ValueError("No finite training z-scores to compute class cut thresholds")
    return np.quantile(flat, quantile_fractions)


def z_values_to_class_array(z_arr: np.ndarray, z_cuts: np.ndarray) -> np.ndarray:
    out = np.full(z_arr.shape, np.nan, dtype=float)
    finite = np.isfinite(z_arr)
    if finite.any():
        out[finite] = np.searchsorted(z_cuts, z_arr[finite], side="right")
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


def _z_cut_column_name(quantile_fraction: float) -> str:
    pct = quantile_fraction * 100.0
    if abs(pct - round(pct)) < 1e-9:
        return f"z_cut_p{int(round(pct))}"
    label = f"{pct:.6g}".replace(".", "p")
    return f"z_cut_p{label}"


def build_normalization_row(
    *,
    kpi_type: str,
    scaler: StandardScaler,
    quantile_fractions: np.ndarray,
    z_cuts: np.ndarray,
    action_disconnect_class: int,
) -> pd.DataFrame:
    mu, sigma = global_scaler_mu_sigma(scaler)
    row: dict[str, object] = {
        "kpi_type": kpi_type,
        "mu": mu,
        "sigma": sigma,
    }

    for frac, z_cut in zip(quantile_fractions, z_cuts):
        row[_z_cut_column_name(float(frac))] = float(z_cut)

    n_kpi_classes = int(len(z_cuts) + 1)
    n_classes = int(action_disconnect_class) + 1

    row["n_kpi_classes"] = n_kpi_classes
    row["action_disconnect_class"] = int(action_disconnect_class)
    row["n_classes"] = n_classes

    return pd.DataFrame([row])


def build_class_dataset_for_type(
    *,
    kpi_type: str,
    kpi_df: pd.DataFrame,
    action_df: pd.DataFrame,
    disconnection_df: Optional[pd.DataFrame],
    split_lookup: Dict[Tuple[str, str], str],
    quantile_fractions: np.ndarray,
    output_dataset_path: Path,
    scaler_path: Path,
    apply_disconnection_mask: bool,
) -> Tuple[Path, pd.DataFrame]:
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

    logged_df = log_transform_dataframe(kpi_df, value_cols)

    train_mask = row_split_mask(logged_df, split_lookup, "train")
    if int(train_mask.sum()) == 0:
        raise ValueError(f"No train rows matched {kpi_type} KPI table against split CSV")

    values = logged_df[value_cols].to_numpy(dtype=float)
    scaler = fit_global_standard_scaler_on_train(values, train_mask)
    z_values = transform_values(scaler, values)

    quantile_fractions = np.asarray(quantile_fractions, dtype=float)
    z_cuts = compute_z_cut_thresholds(z_values, train_mask, quantile_fractions)
    n_kpi_classes = int(z_cuts.size) + 1
    action_disconnect_class = n_kpi_classes

    class_df = kpi_df[ID_COLS].copy()
    for j, col in enumerate(value_cols):
        classes = z_values_to_class_array(z_values[:, j], z_cuts)
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

    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)

    norm_row = build_normalization_row(
        kpi_type=kpi_type,
        scaler=scaler,
        quantile_fractions=quantile_fractions,
        z_cuts=z_cuts,
        action_disconnect_class=action_disconnect_class,
    )

    return output_dataset_path, norm_row


def save_normalization_report(rows: Sequence[pd.DataFrame], output_path: Path) -> Path:
    norm_report = pd.concat(list(rows), ignore_index=True, sort=False)
    cut_cols = sorted(c for c in norm_report.columns if str(c).startswith("z_cut_p"))
    ordered = ["kpi_type", "mu", "sigma"] + cut_cols + [
        c
        for c in ("n_kpi_classes", "action_disconnect_class", "n_classes")
        if c in norm_report.columns
    ]
    norm_report = norm_report.reindex(columns=ordered)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    norm_report.to_csv(output_path, index=False)
    return output_path


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
