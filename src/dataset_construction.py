# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: Training dataset construction from KPI tables

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

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
    OP_GRAPHS_DIR,
)
from modules.pipeline_logging import get_logger, log_step_banner

ID_COLS = ["OP", "Contingency"]


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


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


def read_normalization_bounds(path: Optional[Path]) -> dict[str, tuple[float, float]]:
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path)
    out: dict[str, tuple[float, float]] = {}
    for _, row in df.iterrows():
        file_name = str(row.get("File", "")).strip()
        lo = pd.to_numeric(row.get("GlobalMinUsed"), errors="coerce")
        hi = pd.to_numeric(row.get("GlobalMaxUsed"), errors="coerce")
        if file_name and pd.notna(lo) and pd.notna(hi):
            out[file_name] = (float(lo), float(hi))
    return out


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


def normalize_table(
    df: Optional[pd.DataFrame],
    file_name: str,
    *,
    reused_bounds: dict[str, tuple[float, float]],
    target_min: float = 0.0,
    target_max: float = 1.0,
    fixed_min: Optional[float] = None,
    fixed_max: Optional[float] = None,
) -> tuple[Optional[pd.DataFrame], Optional[float], Optional[float], str]:
    if df is None:
        return None, None, None, "missing"
    if target_max < target_min:
        raise ValueError("target_max must be >= target_min")
    use_fixed = fixed_min is not None or fixed_max is not None
    if use_fixed and (fixed_min is None or fixed_max is None):
        raise ValueError("fixed_min and fixed_max must both be set, or both omitted")
    value_cols = [col for col in df.columns if col not in {"OP", "Contingency"}]
    if not value_cols:
        return df.copy(), None, None, "empty"

    if use_fixed:
        global_min, global_max = float(fixed_min), float(fixed_max)
        source = "fixed"
    else:
        bounds = reused_bounds.get(file_name)
        if bounds is not None:
            global_min, global_max = bounds
            source = "reused"
        else:
            numeric = pd.concat([pd.to_numeric(df[col], errors="coerce") for col in value_cols])
            finite = numeric.dropna()
            if finite.empty:
                global_min = global_max = None
                source = "empty"
            else:
                global_min = float(finite.min())
                global_max = float(finite.max())
                source = "computed"

    out = df.copy()
    if global_min is not None and global_max is not None:
        if global_max == global_min:
            for col in value_cols:
                values = pd.to_numeric(out[col], errors="coerce")
                out[col] = values.mask(values.notna(), float(target_min))
        else:
            out_span = float(target_max) - float(target_min)
            for col in value_cols:
                values = pd.to_numeric(out[col], errors="coerce")
                out[col] = float(target_min) + ((values - global_min) / (global_max - global_min)) * out_span

    return out, global_min, global_max, source


def save_normalization_table(rows: list[dict], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def cuts_from_config(config: dict, key: str) -> np.ndarray:
    cuts = ((config.get("kpi") or {}).get("class_bins") or {}).get(key, {}).get("cuts", [])
    arr = np.asarray(cuts, dtype=float)
    arr.sort()
    return arr


def apply_flag_mask(dataset: pd.DataFrame, flags: Optional[pd.DataFrame], class_value: int) -> pd.DataFrame:
    if flags is None or flags.empty or dataset.empty:
        return dataset
    if any(col not in dataset.columns for col in ID_COLS) or any(col not in flags.columns for col in ID_COLS):
        return dataset
    out = dataset.copy()
    flags = flags.copy()
    out.columns = [str(col).strip() for col in out.columns]
    flags.columns = [str(col).strip() for col in flags.columns]

    for id_col in ID_COLS:
        out[id_col] = out[id_col].astype("string").str.strip()
        flags[id_col] = flags[id_col].astype("string").str.strip()

    common_cols = [col for col in out.columns if col not in ID_COLS and col in flags.columns]
    if not common_cols:
        return dataset

    target_keys = out[ID_COLS].copy()
    target_keys["_occ"] = target_keys.groupby(ID_COLS, dropna=False).cumcount()
    target_keys["_row_idx"] = np.arange(len(out))

    flag_keys = flags[ID_COLS].copy()
    flag_keys["_occ"] = flag_keys.groupby(ID_COLS, dropna=False).cumcount()
    flag_keys = pd.concat([flag_keys, flags[common_cols]], axis=1)

    merged = target_keys.merge(flag_keys, on=[*ID_COLS, "_occ"], how="left")
    for col in common_cols:
        flagged = pd.to_numeric(merged[col], errors="coerce").eq(1)
        if not flagged.any():
            continue
        row_indices = merged.loc[flagged, "_row_idx"].to_numpy(dtype=int, copy=False)
        out.loc[row_indices, col] = class_value
    return out


def kpi_values_to_classes(
    series: pd.Series,
    *,
    cuts: np.ndarray,
    norm_target_min: float = 0.0,
    norm_target_max: float = 1.0,
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    span = float(norm_target_max) - float(norm_target_min)
    if span == 0.0:
        unit = pd.Series(0.5, index=numeric.index, dtype=float).mask(numeric.isna())
    else:
        unit = (numeric - float(norm_target_min)) / span
    unit = unit.clip(lower=0.0, upper=1.0)
    classes = np.searchsorted(np.asarray(cuts, dtype=float), unit.to_numpy(dtype=float), side="right")
    return pd.Series(classes, index=series.index).mask(numeric.isna())


def build_dataset(
    normalized_kpi: Optional[pd.DataFrame],
    output_path: Path,
    *,
    cuts: np.ndarray,
    action_df: Optional[pd.DataFrame] = None,
    disconnection_df: Optional[pd.DataFrame] = None,
    norm_target_min: float = 0.0,
    norm_target_max: float = 1.0,
) -> Optional[Path]:
    if normalized_kpi is None:
        return None
    df = normalized_kpi
    value_cols = [col for col in df.columns if col not in {"OP", "Contingency"}]
    out = df.copy()
    for col in value_cols:
        out[col] = kpi_values_to_classes(
            out[col],
            cuts=cuts,
            norm_target_min=norm_target_min,
            norm_target_max=norm_target_max,
        )
    flag_class = int(len(cuts) + 1)
    out = apply_flag_mask(out, action_df, flag_class)
    out = apply_flag_mask(out, disconnection_df, flag_class)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return output_path


def build_datasets(
    *,
    reuse_normalized_table: Optional[Path] = None,
    norm_min: Optional[float] = None,
    norm_max: Optional[float] = None,
    norm_target_min: float = 0.0,
    norm_target_max: float = 1.0,
) -> dict[str, Optional[Path]]:
    config = load_config()
    reused_bounds = read_normalization_bounds(reuse_normalized_table)

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

    norm_rows = []
    norm_voltage, v_min, v_max, v_source = normalize_table(
        masked_voltage,
        "KPI_voltage.csv",
        reused_bounds=reused_bounds,
        target_min=norm_target_min,
        target_max=norm_target_max,
        fixed_min=norm_min,
        fixed_max=norm_max,
    )
    norm_rows.append({"File": "KPI_voltage.csv", "GlobalMinUsed": v_min, "GlobalMaxUsed": v_max, "BoundsSource": v_source})
    norm_spower, s_min, s_max, s_source = normalize_table(
        masked_spower,
        "KPI_spower.csv",
        reused_bounds=reused_bounds,
        target_min=norm_target_min,
        target_max=norm_target_max,
        fixed_min=norm_min,
        fixed_max=norm_max,
    )
    norm_rows.append({"File": "KPI_spower.csv", "GlobalMinUsed": s_min, "GlobalMaxUsed": s_max, "BoundsSource": s_source})
    kpi_voltage_path = write_table(norm_voltage, KPI_DIR / "KPI_voltage.csv")
    kpi_spower_path = write_table(norm_spower, KPI_DIR / "KPI_spower.csv")
    normalization_path = save_normalization_table(norm_rows, KPI_DIR / "KPI_normalization_minmax.csv")

    dataset_voltage = build_dataset(
        norm_voltage,
        DATASET_DIR / "Dataset_Voltage.csv",
        cuts=cuts_from_config(config, "voltage"),
        action_df=combined_actions_voltage,
        disconnection_df=combined_disc_voltage,
        norm_target_min=norm_target_min,
        norm_target_max=norm_target_max,
    )
    dataset_spower = build_dataset(
        norm_spower,
        DATASET_DIR / "Dataset_Spower.csv",
        cuts=cuts_from_config(config, "spower"),
        action_df=combined_actions_spower,
        disconnection_df=combined_disc_spower,
        norm_target_min=norm_target_min,
        norm_target_max=norm_target_max,
    )

    return {
        "actions_voltage": actions_voltage_path,
        "actions_spower": actions_spower_path,
        "disconnections_voltage": disc_voltage_path,
        "disconnections_spower": disc_spower_path,
        "kpi_voltage": kpi_voltage_path,
        "kpi_spower": kpi_spower_path,
        "normalization": normalization_path,
        "dataset_voltage": dataset_voltage,
        "dataset_spower": dataset_spower,
    }


def main() -> None:
    log_step_banner("dataset_construction")
    logger = get_logger()

    outputs = build_datasets()

    logger.info("Dataset construction finished.")
    for name, path in outputs.items():
        logger.info("%s: %s", name, path)
