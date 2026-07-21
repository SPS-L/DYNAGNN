# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: KPI, actions, and disconnections post-processing

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.actions_detection import run_actions_detection
from modules.disconnections_detection import run_disconnections_detection
from modules.kpi import run_kpi
from modules.paths import (
    ACTIONS_DIR,
    DISCONNECTIONS_DIR,
    KPI_DIR,
    OP_GRAPHS_DIR,
    SIMULATIONS_DIR,
)
from modules.pipeline_logging import get_logger, log_step_banner
from modules.simulation_results import load_successful_runs, resolve_results_csv

ID_COLS = ["OP", "Contingency"]


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


def build_combined_tables() -> dict[str, object]:
    """Merge per-OP KPI/flag tables and write combined CSVs."""
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

    return {
        "actions_voltage": actions_voltage_path,
        "actions_spower": actions_spower_path,
        "disconnections_voltage": disc_voltage_path,
        "disconnections_spower": disc_spower_path,
        "kpi_voltage": kpi_voltage_path,
        "kpi_spower": kpi_spower_path,
    }


def main() -> None:
    log_step_banner("curves_post_process")
    logger = get_logger()

    results_csv = resolve_results_csv(SIMULATIONS_DIR)
    successful_runs = load_successful_runs(results_csv)
    if not successful_runs:
        raise RuntimeError(
            f"No successful simulations found in {results_csv}. "
            "Run simulate.py first and ensure at least one scenario succeeds."
        )
    logger.info(
        "Post-processing %d successful simulation(s) from %s",
        len(successful_runs),
        results_csv.name,
    )

    logger.info("Step 1/5: KPI extraction")
    run_kpi(successful_runs=successful_runs)

    logger.info("Step 2/5: Actions detection")
    run_actions_detection(successful_runs=successful_runs)

    logger.info("Step 3/5: Disconnections detection")
    run_disconnections_detection(successful_runs=successful_runs)

    logger.info("Step 4/4: Combined KPI and flag tables")
    outputs = build_combined_tables()

    for name, path in outputs.items():
        logger.info("%s: %s", name, path)

    logger.info("curves_post_process completed.")
