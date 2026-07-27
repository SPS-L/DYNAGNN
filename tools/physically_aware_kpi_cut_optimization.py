#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: optional physically aware KPI cut optimization (outside the main pipeline)

"""Physically aware KPI cut optimization (voltage + spower).

Optional offline tool (not part of ``main.py``). Recommends raw KPI class cuts
from training scenarios using multi-objective usability + physical morphology
scores. See ``tools/README.md``.

Example::

    python3 tools/physically_aware_kpi_cut_optimization.py
    python3 tools/physically_aware_kpi_cut_optimization.py \\
        --data-path /path/to/my_network/data
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# Defaults — single source of truth for CLI and runtime globals
# =============================================================================

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PROJECT_ROOT = _REPO_ROOT
_DEFAULT_DATA_PATH = _REPO_ROOT / "examples" / "Nordic" / "data"
_DEFAULT_CLEAR_OUTPUT_DIRECTORY = True
_DEFAULT_EVENT_TIME_SEC = 10.0
_DEFAULT_POST_EVENT_WINDOW_SEC = 50.0
_DEFAULT_SETTLING_TAIL_FRACTION = 0.2
_DEFAULT_PHI_STABILIZER = 1e-12
_DEFAULT_SETTLING_WEIGHT = 0.5
_DEFAULT_NOISE_FLOOR = 1e-7
_DEFAULT_A_VALUES = [
    5e-5,
    7.5e-5,
    1e-4,
    1.5e-4,
    2e-4,
    3e-4,
    4e-4,
    5e-4,
    7.5e-4,
    1e-3,
]
_DEFAULT_P1_VALUES = [0.005, 0.01, 0.02, 0.05, 0.075, 0.10]
_DEFAULT_P2_VALUES = [0.03, 0.05, 0.075, 0.10, 0.15, 0.25, 0.40]
_DEFAULT_P3_VALUES = [0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75]
_DEFAULT_VOLTAGE_CUTS = (3e-6, 2e-5, 1e-4, 1e-3)
_DEFAULT_SPOWER_CUTS = (5e-6, 1e-5, 1e-4, 6e-4)
_DEFAULT_MAX_MORPHOLOGY_SAMPLES = 400
_DEFAULT_MORPHOLOGY_SEED = 42
_DEFAULT_MAX_CLASS1_SHARE = 0.92
_DEFAULT_MIN_CLASS2_SHARE = 0.02
_DEFAULT_MIN_CLASS3_SHARE = 0.005
_DEFAULT_MIN_CLASS4_SHARE = 0.001
_DEFAULT_USABILITY_WEIGHTS = (0.45, 0.35, 0.20)
_DEFAULT_PHYSICAL_WEIGHTS = (0.65, 0.35)
_DEFAULT_COMBINED_WEIGHTS = (0.40, 0.40, 0.20)
_DEFAULT_MIN_PHYSICAL_FOR_FEASIBLE = 0.15
_DEFAULT_SHARED_WEIGHTS = (0.35, 0.35, 0.15, 0.15)
_DEFAULT_SHARED_CUTS = True

# Runtime configuration (filled by apply_settings / CLI).
PROJECT_ROOT = _DEFAULT_PROJECT_ROOT
DATA_PATH = _DEFAULT_DATA_PATH
KPI_VOLTAGE_PATH = DATA_PATH / "KPI/KPI_voltage.csv"
KPI_SPOWER_PATH = DATA_PATH / "KPI/KPI_spower.csv"
CONTINGENCIES_CSV = DATA_PATH / "inputs/contingencies.csv"
SPLIT_CSV = DATA_PATH / "Dataset/train_val_test_split.csv"
SIM_RESULTS_ROOT = DATA_PATH / "Simulations_Scenarios"
SNOM_DIR = DATA_PATH / "generator_Snom"
OUTPUT_DIR = DATA_PATH / "kpi_cut_optimization"
CLEAR_OUTPUT_DIRECTORY = _DEFAULT_CLEAR_OUTPUT_DIRECTORY

EVENT_TIME_SEC = _DEFAULT_EVENT_TIME_SEC
POST_EVENT_WINDOW_SEC = _DEFAULT_POST_EVENT_WINDOW_SEC
SETTLING_TAIL_FRACTION = _DEFAULT_SETTLING_TAIL_FRACTION
PHI_STABILIZER = _DEFAULT_PHI_STABILIZER
SETTLING_WEIGHT = _DEFAULT_SETTLING_WEIGHT

SOLVER_NOISE_FLOOR = _DEFAULT_NOISE_FLOOR
A_VALUES = np.array(_DEFAULT_A_VALUES, dtype=float)
P1_VALUES = np.array(_DEFAULT_P1_VALUES, dtype=float)
P2_VALUES = np.array(_DEFAULT_P2_VALUES, dtype=float)
P3_VALUES = np.array(_DEFAULT_P3_VALUES, dtype=float)
CURRENT_VOLTAGE_CUTS = _DEFAULT_VOLTAGE_CUTS
CURRENT_SPOWER_CUTS = _DEFAULT_SPOWER_CUTS

MAX_MORPHOLOGY_SAMPLES_PER_KPI = _DEFAULT_MAX_MORPHOLOGY_SAMPLES
MORPHOLOGY_RANDOM_SEED = _DEFAULT_MORPHOLOGY_SEED
MIN_KPI_FOR_MORPHOLOGY = _DEFAULT_NOISE_FLOOR

MAX_DYNAMIC_CLASS1_SHARE = _DEFAULT_MAX_CLASS1_SHARE
MIN_DYNAMIC_CLASS2_SHARE = _DEFAULT_MIN_CLASS2_SHARE
MIN_DYNAMIC_CLASS3_SHARE = _DEFAULT_MIN_CLASS3_SHARE
MIN_DYNAMIC_CLASS4_SHARE = _DEFAULT_MIN_CLASS4_SHARE

USABILITY_W_ENTROPY, USABILITY_W_FLOOR, USABILITY_W_CLASS1 = _DEFAULT_USABILITY_WEIGHTS
PHYSICAL_W_SPEARMAN, PHYSICAL_W_MONO = _DEFAULT_PHYSICAL_WEIGHTS
COMBINED_W_USABILITY, COMBINED_W_PHYSICAL, COMBINED_W_SEVERITY = _DEFAULT_COMBINED_WEIGHTS
MIN_PHYSICAL_FOR_FEASIBLE = _DEFAULT_MIN_PHYSICAL_FOR_FEASIBLE
(
    SHARED_W_USABILITY,
    SHARED_W_PHYSICAL,
    SHARED_W_COMBINED,
    SHARED_W_CONSISTENCY,
) = _DEFAULT_SHARED_WEIGHTS

SHARED_CUTS = _DEFAULT_SHARED_CUTS

# Populated after CLI + project-root import.
load_generator_snom_for_operating_point = None  # type: ignore
GEN_P_SUFFIXES = None  # type: ignore
GEN_Q_SUFFIXES = None  # type: ignore
VOLTAGE_SUFFIXES = None  # type: ignore
build_dyd_id_to_staticid_map = None  # type: ignore
build_voltage_curve_to_voltage_level_map = None  # type: ignore
find_curves_file = None  # type: ignore
find_dyd_file = None  # type: ignore
find_iidm_file = None  # type: ignore
make_label = None  # type: ignore
matching_curves = None  # type: ignore
parse_curves_xml = None  # type: ignore
resolve_snom = None  # type: ignore


@dataclass
class Settings:
    """Tunable run configuration (CLI overrides use the same fields)."""

    project_root: Path
    data_path: Path
    output_dir: Path
    clear_output_directory: bool
    kpi_voltage: Path
    kpi_spower: Path
    contingencies: Path
    split: Path
    sim_results: Path
    snom_dir: Path
    event_time_sec: float
    post_event_window_sec: float
    settling_tail_fraction: float
    phi_stabilizer: float
    settling_weight: float
    noise_floor: float
    a_values: list[float]
    p1_values: list[float]
    p2_values: list[float]
    p3_values: list[float]
    current_voltage_cuts: tuple[float, float, float, float]
    current_spower_cuts: tuple[float, float, float, float]
    max_morphology_samples: int
    morphology_seed: int
    max_class1_share: float
    min_class2_share: float
    min_class3_share: float
    min_class4_share: float
    usability_weights: tuple[float, float, float]
    physical_weights: tuple[float, float]
    combined_weights: tuple[float, float, float]
    min_physical_for_feasible: float
    shared_weights: tuple[float, float, float, float]
    shared_cuts: bool


def _case_paths(data_path: Path) -> dict[str, Path]:
    """Standard DYNAGNN layout under a network ``data/`` directory."""
    data = data_path
    return {
        "kpi_voltage": data / "KPI/KPI_voltage.csv",
        "kpi_spower": data / "KPI/KPI_spower.csv",
        "contingencies": data / "inputs/contingencies.csv",
        "split": data / "Dataset/train_val_test_split.csv",
        "sim_results": data / "Simulations_Scenarios",
        "snom_dir": data / "generator_Snom",
    }


def _four_floats(values: list[float], name: str) -> tuple[float, float, float, float]:
    if len(values) != 4:
        raise argparse.ArgumentTypeError(f"{name} requires exactly 4 floats")
    return float(values[0]), float(values[1]), float(values[2]), float(values[3])


def parse_args(argv: Optional[list[str]] = None) -> Settings:
    p = argparse.ArgumentParser(
        description="Physically aware multi-objective KPI cut optimization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    paths = p.add_argument_group("paths")
    paths.add_argument(
        "--data-path",
        type=Path,
        default=_DEFAULT_DATA_PATH,
        help=(
            "Network data directory (same layout as config data.path). "
            "Default: bundled Nordic example."
        ),
    )
    paths.add_argument(
        "--project-root",
        type=Path,
        default=_DEFAULT_PROJECT_ROOT,
        help="DYNAGNN repository root (for importing modules/).",
    )
    paths.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Results directory (default: <data-path>/kpi_cut_optimization).",
    )
    paths.add_argument(
        "--clear-output",
        dest="clear_output_directory",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULT_CLEAR_OUTPUT_DIRECTORY,
        help="Delete output directory before writing results.",
    )
    paths.add_argument("--kpi-voltage", type=Path, default=None)
    paths.add_argument("--kpi-spower", type=Path, default=None)
    paths.add_argument("--contingencies", type=Path, default=None)
    paths.add_argument("--split", type=Path, default=None)
    paths.add_argument("--sim-results", type=Path, default=None)
    paths.add_argument("--snom-dir", type=Path, default=None)

    morph = p.add_argument_group("morphology")
    morph.add_argument(
        "--event-time-sec", type=float, default=_DEFAULT_EVENT_TIME_SEC
    )
    morph.add_argument(
        "--post-event-window-sec", type=float, default=_DEFAULT_POST_EVENT_WINDOW_SEC
    )
    morph.add_argument(
        "--settling-tail-fraction",
        type=float,
        default=_DEFAULT_SETTLING_TAIL_FRACTION,
    )
    morph.add_argument("--phi-stabilizer", type=float, default=_DEFAULT_PHI_STABILIZER)
    morph.add_argument("--settling-weight", type=float, default=_DEFAULT_SETTLING_WEIGHT)
    morph.add_argument(
        "--max-morphology-samples",
        type=int,
        default=_DEFAULT_MAX_MORPHOLOGY_SAMPLES,
    )
    morph.add_argument(
        "--morphology-seed", type=int, default=_DEFAULT_MORPHOLOGY_SEED
    )

    grid = p.add_argument_group("candidate grid")
    grid.add_argument("--noise-floor", type=float, default=_DEFAULT_NOISE_FLOOR)
    grid.add_argument("--a-values", type=float, nargs="+", default=_DEFAULT_A_VALUES)
    grid.add_argument("--p1-values", type=float, nargs="+", default=_DEFAULT_P1_VALUES)
    grid.add_argument("--p2-values", type=float, nargs="+", default=_DEFAULT_P2_VALUES)
    grid.add_argument("--p3-values", type=float, nargs="+", default=_DEFAULT_P3_VALUES)
    grid.add_argument(
        "--current-voltage-cuts",
        type=float,
        nargs=4,
        default=list(_DEFAULT_VOLTAGE_CUTS),
        metavar=("T0", "T1", "T2", "T3"),
    )
    grid.add_argument(
        "--current-spower-cuts",
        type=float,
        nargs=4,
        default=list(_DEFAULT_SPOWER_CUTS),
        metavar=("T0", "T1", "T2", "T3"),
    )

    floors = p.add_argument_group("class-share floors")
    floors.add_argument(
        "--max-class1-share", type=float, default=_DEFAULT_MAX_CLASS1_SHARE
    )
    floors.add_argument(
        "--min-class2-share", type=float, default=_DEFAULT_MIN_CLASS2_SHARE
    )
    floors.add_argument(
        "--min-class3-share", type=float, default=_DEFAULT_MIN_CLASS3_SHARE
    )
    floors.add_argument(
        "--min-class4-share", type=float, default=_DEFAULT_MIN_CLASS4_SHARE
    )

    scores = p.add_argument_group("score weights")
    scores.add_argument(
        "--usability-weights",
        type=float,
        nargs=3,
        default=list(_DEFAULT_USABILITY_WEIGHTS),
        metavar=("ENTROPY", "FLOOR", "CLASS1"),
        help="Weights for entropy, floor satisfaction, class-1 control.",
    )
    scores.add_argument(
        "--physical-weights",
        type=float,
        nargs=2,
        default=list(_DEFAULT_PHYSICAL_WEIGHTS),
        metavar=("SPEARMAN", "MONO"),
        help="Weights for Spearman(φ, class) and mean-φ monotonicity.",
    )
    scores.add_argument(
        "--combined-weights",
        type=float,
        nargs=3,
        default=list(_DEFAULT_COMBINED_WEIGHTS),
        metavar=("U", "P", "S"),
        help="Weights for usability, physical, scenario severity proxy.",
    )
    scores.add_argument(
        "--min-physical-for-feasible",
        type=float,
        default=_DEFAULT_MIN_PHYSICAL_FOR_FEASIBLE,
        help="Minimum physical score required for diagnostic feasibility.",
    )
    scores.add_argument(
        "--shared-weights",
        type=float,
        nargs=4,
        default=list(_DEFAULT_SHARED_WEIGHTS),
        metavar=("U_SH", "P_SH", "J_AVG", "C"),
        help="Shared-mode weights: worst U, worst P, avg combined, consistency.",
    )
    scores.add_argument(
        "--shared-cuts",
        dest="shared_cuts",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULT_SHARED_CUTS,
        help="Recommend one absolute cut set for voltage and spower.",
    )

    args = p.parse_args(argv)
    data_path = args.data_path.resolve()
    derived = _case_paths(data_path)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (data_path / "kpi_cut_optimization").resolve()
    )

    def _path(cli_val: Optional[Path], key: str) -> Path:
        return (cli_val if cli_val is not None else derived[key]).resolve()

    return Settings(
        project_root=args.project_root.resolve(),
        data_path=data_path,
        output_dir=output_dir,
        clear_output_directory=bool(args.clear_output_directory),
        kpi_voltage=_path(args.kpi_voltage, "kpi_voltage"),
        kpi_spower=_path(args.kpi_spower, "kpi_spower"),
        contingencies=_path(args.contingencies, "contingencies"),
        split=_path(args.split, "split"),
        sim_results=_path(args.sim_results, "sim_results"),
        snom_dir=_path(args.snom_dir, "snom_dir"),
        event_time_sec=float(args.event_time_sec),
        post_event_window_sec=float(args.post_event_window_sec),
        settling_tail_fraction=float(args.settling_tail_fraction),
        phi_stabilizer=float(args.phi_stabilizer),
        settling_weight=float(args.settling_weight),
        noise_floor=float(args.noise_floor),
        a_values=[float(x) for x in args.a_values],
        p1_values=[float(x) for x in args.p1_values],
        p2_values=[float(x) for x in args.p2_values],
        p3_values=[float(x) for x in args.p3_values],
        current_voltage_cuts=_four_floats(list(args.current_voltage_cuts), "current-voltage-cuts"),
        current_spower_cuts=_four_floats(list(args.current_spower_cuts), "current-spower-cuts"),
        max_morphology_samples=int(args.max_morphology_samples),
        morphology_seed=int(args.morphology_seed),
        max_class1_share=float(args.max_class1_share),
        min_class2_share=float(args.min_class2_share),
        min_class3_share=float(args.min_class3_share),
        min_class4_share=float(args.min_class4_share),
        usability_weights=(
            float(args.usability_weights[0]),
            float(args.usability_weights[1]),
            float(args.usability_weights[2]),
        ),
        physical_weights=(
            float(args.physical_weights[0]),
            float(args.physical_weights[1]),
        ),
        combined_weights=(
            float(args.combined_weights[0]),
            float(args.combined_weights[1]),
            float(args.combined_weights[2]),
        ),
        min_physical_for_feasible=float(args.min_physical_for_feasible),
        shared_weights=(
            float(args.shared_weights[0]),
            float(args.shared_weights[1]),
            float(args.shared_weights[2]),
            float(args.shared_weights[3]),
        ),
        shared_cuts=bool(args.shared_cuts),
    )


def apply_settings(s: Settings) -> None:
    """Push CLI/settings into module globals used by the scoring pipeline."""
    global PROJECT_ROOT, DATA_PATH
    global KPI_VOLTAGE_PATH, KPI_SPOWER_PATH, CONTINGENCIES_CSV, SPLIT_CSV
    global SIM_RESULTS_ROOT, SNOM_DIR, OUTPUT_DIR, CLEAR_OUTPUT_DIRECTORY
    global EVENT_TIME_SEC, POST_EVENT_WINDOW_SEC, SETTLING_TAIL_FRACTION
    global PHI_STABILIZER, SETTLING_WEIGHT
    global SOLVER_NOISE_FLOOR, A_VALUES, P1_VALUES, P2_VALUES, P3_VALUES
    global CURRENT_VOLTAGE_CUTS, CURRENT_SPOWER_CUTS
    global MAX_MORPHOLOGY_SAMPLES_PER_KPI, MORPHOLOGY_RANDOM_SEED, MIN_KPI_FOR_MORPHOLOGY
    global MAX_DYNAMIC_CLASS1_SHARE, MIN_DYNAMIC_CLASS2_SHARE
    global MIN_DYNAMIC_CLASS3_SHARE, MIN_DYNAMIC_CLASS4_SHARE
    global USABILITY_W_ENTROPY, USABILITY_W_FLOOR, USABILITY_W_CLASS1
    global PHYSICAL_W_SPEARMAN, PHYSICAL_W_MONO
    global COMBINED_W_USABILITY, COMBINED_W_PHYSICAL, COMBINED_W_SEVERITY
    global MIN_PHYSICAL_FOR_FEASIBLE
    global SHARED_W_USABILITY, SHARED_W_PHYSICAL, SHARED_W_COMBINED, SHARED_W_CONSISTENCY
    global SHARED_CUTS

    PROJECT_ROOT = s.project_root
    DATA_PATH = s.data_path
    KPI_VOLTAGE_PATH = s.kpi_voltage
    KPI_SPOWER_PATH = s.kpi_spower
    CONTINGENCIES_CSV = s.contingencies
    SPLIT_CSV = s.split
    SIM_RESULTS_ROOT = s.sim_results
    SNOM_DIR = s.snom_dir
    OUTPUT_DIR = s.output_dir
    CLEAR_OUTPUT_DIRECTORY = s.clear_output_directory

    EVENT_TIME_SEC = s.event_time_sec
    POST_EVENT_WINDOW_SEC = s.post_event_window_sec
    SETTLING_TAIL_FRACTION = s.settling_tail_fraction
    PHI_STABILIZER = s.phi_stabilizer
    SETTLING_WEIGHT = s.settling_weight

    SOLVER_NOISE_FLOOR = s.noise_floor
    A_VALUES = np.array(s.a_values, dtype=float)
    P1_VALUES = np.array(s.p1_values, dtype=float)
    P2_VALUES = np.array(s.p2_values, dtype=float)
    P3_VALUES = np.array(s.p3_values, dtype=float)
    CURRENT_VOLTAGE_CUTS = s.current_voltage_cuts
    CURRENT_SPOWER_CUTS = s.current_spower_cuts

    MAX_MORPHOLOGY_SAMPLES_PER_KPI = s.max_morphology_samples
    MORPHOLOGY_RANDOM_SEED = s.morphology_seed
    MIN_KPI_FOR_MORPHOLOGY = s.noise_floor

    MAX_DYNAMIC_CLASS1_SHARE = s.max_class1_share
    MIN_DYNAMIC_CLASS2_SHARE = s.min_class2_share
    MIN_DYNAMIC_CLASS3_SHARE = s.min_class3_share
    MIN_DYNAMIC_CLASS4_SHARE = s.min_class4_share

    USABILITY_W_ENTROPY, USABILITY_W_FLOOR, USABILITY_W_CLASS1 = s.usability_weights
    PHYSICAL_W_SPEARMAN, PHYSICAL_W_MONO = s.physical_weights
    COMBINED_W_USABILITY, COMBINED_W_PHYSICAL, COMBINED_W_SEVERITY = s.combined_weights
    MIN_PHYSICAL_FOR_FEASIBLE = s.min_physical_for_feasible
    (
        SHARED_W_USABILITY,
        SHARED_W_PHYSICAL,
        SHARED_W_COMBINED,
        SHARED_W_CONSISTENCY,
    ) = s.shared_weights
    SHARED_CUTS = s.shared_cuts


def import_dynagnn_modules(project_root: Path) -> None:
    """Import project helpers after --project-root is known."""
    global load_generator_snom_for_operating_point
    global GEN_P_SUFFIXES, GEN_Q_SUFFIXES, VOLTAGE_SUFFIXES
    global build_dyd_id_to_staticid_map, build_voltage_curve_to_voltage_level_map
    global find_curves_file, find_dyd_file, find_iidm_file
    global make_label, matching_curves, parse_curves_xml, resolve_snom

    root = str(project_root)
    if root not in sys.path:
        sys.path.insert(0, root)

    from modules.generator_snom import (  # noqa: WPS433
        load_generator_snom_for_operating_point as _load_snom,
    )
    from modules.kpi import (  # noqa: WPS433
        GEN_P_SUFFIXES as _GEN_P,
        GEN_Q_SUFFIXES as _GEN_Q,
        VOLTAGE_SUFFIXES as _VOLTAGE,
        build_dyd_id_to_staticid_map as _build_dyd,
        build_voltage_curve_to_voltage_level_map as _build_vl,
        find_curves_file as _find_curves,
        find_dyd_file as _find_dyd,
        find_iidm_file as _find_iidm,
        make_label as _make_label,
        matching_curves as _matching,
        parse_curves_xml as _parse,
        resolve_snom as _resolve,
    )

    load_generator_snom_for_operating_point = _load_snom
    GEN_P_SUFFIXES = _GEN_P
    GEN_Q_SUFFIXES = _GEN_Q
    VOLTAGE_SUFFIXES = _VOLTAGE
    build_dyd_id_to_staticid_map = _build_dyd
    build_voltage_curve_to_voltage_level_map = _build_vl
    find_curves_file = _find_curves
    find_dyd_file = _find_dyd
    find_iidm_file = _find_iidm
    make_label = _make_label
    matching_curves = _matching
    parse_curves_xml = _parse
    resolve_snom = _resolve


def settings_to_jsonable(s: Settings) -> dict[str, Any]:
    raw = asdict(s)
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, tuple):
            out[key] = list(value)
        else:
            out[key] = value
    return out


# =============================================================================
# Small utilities
# =============================================================================


def prepare_output_directory(path: Path) -> Path:
    if CLEAR_OUTPUT_DIRECTORY and path.exists():
        import shutil

        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "plots").mkdir(exist_ok=True)
    return path


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float("nan")
    xr = pd.Series(x[mask]).rank().to_numpy()
    yr = pd.Series(y[mask]).rank().to_numpy()
    if np.std(xr) < 1e-12 or np.std(yr) < 1e-12:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def normalized_entropy(probs: np.ndarray) -> float:
    p = np.asarray(probs, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    h = -np.sum(p * np.log(p))
    return float(h / np.log(len(probs)))


def clipped_ratio(value: float, target: float) -> float:
    if target <= 0:
        return 1.0
    return float(np.clip(value / target, 0.0, 1.0))


def assign_class(value: float, cuts: tuple[float, float, float, float]) -> int:
    """Class 0 <= t0 < class1 <= t1 < ... < class4. cuts = (t0,t1,t2,t3)."""
    if not np.isfinite(value):
        return -1
    t0, t1, t2, t3 = cuts
    if value <= t0:
        return 0
    if value <= t1:
        return 1
    if value <= t2:
        return 2
    if value <= t3:
        return 3
    return 4


# =============================================================================
# Paths / split helpers
# =============================================================================


def load_train_keys(split_path: Path) -> set[tuple[str, str]]:
    sdf = pd.read_csv(split_path)
    sdf.columns = [str(c).strip() for c in sdf.columns]
    train = sdf.loc[sdf["split"].astype(str).str.strip().str.lower() == "train"]
    ops = train["operating_point"].astype(str).str.strip()
    conts = train["contingency"].astype(str).str.strip()
    return set(zip(ops, conts))


def normalize_op(op_val: object) -> str:
    s = str(op_val).strip()
    if s.startswith("operating_point_"):
        return s
    m = __import__("re").search(r"(\d+)\s*$", s)
    if not m:
        raise ValueError(f"Cannot parse OP from {op_val!r}")
    return f"operating_point_{int(m.group(1))}"


def filter_train(df: pd.DataFrame, train_keys: set[tuple[str, str]]) -> pd.DataFrame:
    ops = df["OP"].map(normalize_op)
    conts = df["Contingency"].astype(str).str.strip()
    mask = [key in train_keys for key in zip(ops, conts)]
    out = df.loc[mask].copy()
    out.reset_index(drop=True, inplace=True)
    return out


def load_contingencies(path: Path) -> dict[str, str]:
    cdf = pd.read_csv(path)
    cdf.columns = [str(c).strip() for c in cdf.columns]
    return {
        str(r["Fault name"]).strip(): str(r["Contingency ID"]).strip()
        for _, r in cdf.iterrows()
    }


# =============================================================================
# Candidate cuts
# =============================================================================


def build_candidate_grid() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cid = 0
    for a in A_VALUES:
        for p1 in P1_VALUES:
            for p2 in P2_VALUES:
                for p3 in P3_VALUES:
                    if not (0.0 < p1 < p2 < p3 < 1.0):
                        continue
                    t0 = SOLVER_NOISE_FLOOR
                    t1 = float(a * p1)
                    t2 = float(a * p2)
                    t3 = float(a * p3)
                    if not (t0 < t1 < t2 < t3):
                        continue
                    rows.append(
                        {
                            "candidate_id": cid,
                            "A": float(a),
                            "p1": float(p1),
                            "p2": float(p2),
                            "p3": float(p3),
                            "t0": t0,
                            "t1": t1,
                            "t2": t2,
                            "t3": t3,
                            "source": "percentage_grid",
                        }
                    )
                    cid += 1

    # Current Nordic bins as named baselines (independent KPI tags later).
    rows.append(
        {
            "candidate_id": cid,
            "A": float("nan"),
            "p1": float("nan"),
            "p2": float("nan"),
            "p3": float("nan"),
            "t0": CURRENT_VOLTAGE_CUTS[0],
            "t1": CURRENT_VOLTAGE_CUTS[1],
            "t2": CURRENT_VOLTAGE_CUTS[2],
            "t3": CURRENT_VOLTAGE_CUTS[3],
            "source": "current_voltage_bins",
        }
    )
    cid += 1
    rows.append(
        {
            "candidate_id": cid,
            "A": float("nan"),
            "p1": float("nan"),
            "p2": float("nan"),
            "p3": float("nan"),
            "t0": CURRENT_SPOWER_CUTS[0],
            "t1": CURRENT_SPOWER_CUTS[1],
            "t2": CURRENT_SPOWER_CUTS[2],
            "t3": CURRENT_SPOWER_CUTS[3],
            "source": "current_spower_bins",
        }
    )
    return pd.DataFrame(rows)


# =============================================================================
# Morphology / physical severity φ
# =============================================================================


@dataclass
class MorphologySample:
    kpi: str
    op: str
    contingency: str
    component: str
    kpi_value: float
    phi: float
    peak_excursion: float
    post_var: float
    settling_dev: float


_VL_CACHE: dict[str, dict[str, list[str]]] = {}
_SNOM_CACHE: dict[str, dict[str, float]] = {}


def _vl_to_components(op: str, contingency_dir: Path) -> dict[str, list[str]]:
    if op not in _VL_CACHE:
        iidm = find_iidm_file(contingency_dir, contingency_dir.parent)
        curve_to_vl = build_voltage_curve_to_voltage_level_map(iidm)
        vl_map: dict[str, list[str]] = {}
        for comp, vl in curve_to_vl.items():
            vl_map.setdefault(vl, []).append(comp)
        for comps in vl_map.values():
            comps.sort()
        _VL_CACHE[op] = vl_map
    return _VL_CACHE[op]


def _snom(op: str) -> dict[str, float]:
    if op not in _SNOM_CACHE:
        path = SNOM_DIR / f"{op}.csv"
        _SNOM_CACHE[op] = load_generator_snom_for_operating_point(path)
    return _SNOM_CACHE[op]


def _series_morphology(time: np.ndarray, values: np.ndarray) -> tuple[float, float, float, float]:
    """Return (phi, peak_excursion, post_var, settling_dev)."""
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(time) & np.isfinite(values)
    time, values = time[finite], values[finite]
    if time.size < 5:
        return float("nan"), float("nan"), float("nan"), float("nan")

    pre = values[time < EVENT_TIME_SEC - 1e-9]
    post_mask = (time >= EVENT_TIME_SEC - 1e-9) & (
        time <= EVENT_TIME_SEC + POST_EVENT_WINDOW_SEC + 1e-9
    )
    post = values[post_mask]
    if pre.size == 0 or post.size < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")

    pre_mean = float(np.mean(pre))
    peak_excursion = float(np.max(np.abs(post - pre_mean)))
    post_var = float(np.var(post))
    # Settling: mean absolute deviation in last fraction of post window vs pre.
    n_tail = max(3, int(SETTLING_TAIL_FRACTION * post.size))
    settling_dev = float(np.mean(np.abs(post[-n_tail:] - pre_mean)))

    # Physical severity φ: combine excursion + dynamics + residual offset.
    # Log-scale so orders of magnitude behave like KPI classes.
    phi = math.log10(
        PHI_STABILIZER
        + peak_excursion
        + math.sqrt(max(post_var, 0.0))
        + SETTLING_WEIGHT * settling_dev
    )
    return phi, peak_excursion, post_var, settling_dev


def _read_voltage_series(
    contingency_dir: Path, vl_id: str, op: str
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    curves_file = find_curves_file(contingency_dir)
    if curves_file is None:
        return None
    dyd = find_dyd_file(contingency_dir) or find_dyd_file(contingency_dir.parent)
    id_map = build_dyd_id_to_staticid_map(dyd)
    wanted = set(_vl_to_components(op, contingency_dir).get(vl_id, []))
    wanted.add(vl_id)
    data = parse_curves_xml(curves_file)
    series: list[tuple[np.ndarray, np.ndarray]] = []
    for cname, cdata in matching_curves(data, VOLTAGE_SUFFIXES).items():
        comp = make_label(cname, VOLTAGE_SUFFIXES, id_map)
        if comp not in wanted:
            continue
        series.append((np.asarray(cdata["time"], float), np.asarray(cdata["value"], float)))
    if not series:
        return None
    # Aggregate VL by max |deviation from pre| across busbars: use mean of series.
    # For φ we take the component with largest peak excursion.
    best = None
    best_peak = -1.0
    for t, v in series:
        phi, peak, _, _ = _series_morphology(t, v)
        if np.isfinite(peak) and peak > best_peak:
            best_peak = peak
            best = (t, v)
    return best


def _read_spower_series(
    contingency_dir: Path, gen_id: str, op: str
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    curves_file = find_curves_file(contingency_dir)
    if curves_file is None:
        return None
    dyd = find_dyd_file(contingency_dir) or find_dyd_file(contingency_dir.parent)
    id_map = build_dyd_id_to_staticid_map(dyd)
    data = parse_curves_xml(curves_file)
    p_curves: dict[str, dict[str, list[float]]] = {}
    q_curves: dict[str, dict[str, list[float]]] = {}
    for cname, cdata in matching_curves(data, GEN_P_SUFFIXES).items():
        p_curves[make_label(cname, GEN_P_SUFFIXES, id_map)] = cdata
    for cname, cdata in matching_curves(data, GEN_Q_SUFFIXES).items():
        q_curves[make_label(cname, GEN_Q_SUFFIXES, id_map)] = cdata

    candidates = [gen_id]
    if "__" not in gen_id:
        candidates.extend(
            lab
            for lab in set(p_curves) | set(q_curves)
            if lab.split("__", 1)[0] == gen_id
        )
    label = next((c for c in candidates if c in p_curves or c in q_curves), None)
    if label is None:
        return None
    snom = resolve_snom(label, _snom(op))
    if not snom:
        return None
    p_data = p_curves.get(label)
    q_data = q_curves.get(label)
    time_values = (p_data or q_data)["time"]
    n = len(time_values)
    if p_data is not None:
        n = min(n, len(p_data["value"]))
    if q_data is not None:
        n = min(n, len(q_data["value"]))
    p_vals = p_data["value"][:n] if p_data is not None else [0.0] * n
    q_vals = q_data["value"][:n] if q_data is not None else [0.0] * n
    s_norm = [math.hypot(float(p), float(q)) / snom for p, q in zip(p_vals, q_vals)]
    return np.asarray(time_values[:n], float), np.asarray(s_norm, float)


def sample_morphology_cells(
    kpi_df: pd.DataFrame,
    kpi_name: str,
    fault_to_cid: dict[str, str],
    n_samples: int,
    rng: np.random.Generator,
) -> list[MorphologySample]:
    meta = {"OP", "Contingency"}
    components = [c for c in kpi_df.columns if c not in meta]
    # Build pool of finite above-noise cells.
    pool: list[tuple[int, str, float]] = []
    for i in range(len(kpi_df)):
        row = kpi_df.iloc[i]
        for comp in components:
            v = row[comp]
            if pd.isna(v):
                continue
            fv = float(v)
            if fv < MIN_KPI_FOR_MORPHOLOGY:
                continue
            pool.append((i, comp, fv))
    if not pool:
        return []

    n = min(n_samples, len(pool))
    # Stratify by log10(KPI) decades for coverage across physical regimes.
    decades = np.array([math.floor(math.log10(max(v, 1e-30))) for _, _, v in pool])
    chosen_idx: list[int] = []
    unique_dec = sorted(set(decades.tolist()))
    per = max(1, n // max(len(unique_dec), 1))
    for d in unique_dec:
        idxs = [j for j, dd in enumerate(decades) if dd == d]
        take = min(per, len(idxs))
        chosen_idx.extend(rng.choice(idxs, size=take, replace=False).tolist())
    if len(chosen_idx) < n:
        remaining = [j for j in range(len(pool)) if j not in set(chosen_idx)]
        need = min(n - len(chosen_idx), len(remaining))
        if need:
            chosen_idx.extend(rng.choice(remaining, size=need, replace=False).tolist())
    chosen_idx = chosen_idx[:n]

    samples: list[MorphologySample] = []
    t0 = time.time()
    for k, j in enumerate(chosen_idx, start=1):
        i, comp, fv = pool[j]
        row = kpi_df.iloc[i]
        op = normalize_op(row["OP"])
        fault = str(row["Contingency"]).strip()
        cid = fault_to_cid.get(fault)
        if cid is None:
            continue
        cont_dir = SIM_RESULTS_ROOT / op / f"contingency_{cid}"
        try:
            if kpi_name == "voltage":
                series = _read_voltage_series(cont_dir, comp, op)
            else:
                series = _read_spower_series(cont_dir, comp, op)
        except Exception:
            continue
        if series is None:
            continue
        phi, peak, pvar, sett = _series_morphology(series[0], series[1])
        if not np.isfinite(phi):
            continue
        samples.append(
            MorphologySample(
                kpi=kpi_name,
                op=op,
                contingency=fault,
                component=comp,
                kpi_value=fv,
                phi=phi,
                peak_excursion=peak,
                post_var=pvar,
                settling_dev=sett,
            )
        )
        if k % 50 == 0:
            print(
                f"  [{kpi_name}] morphology {k}/{len(chosen_idx)} "
                f"kept={len(samples)} ({time.time() - t0:.1f}s)"
            )
    print(f"  [{kpi_name}] morphology samples kept: {len(samples)}")
    return samples


# =============================================================================
# Scoring + Pareto
# =============================================================================


def class_share_vector(values: np.ndarray, cuts: tuple[float, float, float, float]) -> np.ndarray:
    t0, t1, t2, t3 = cuts
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.zeros(4, dtype=float)
    classes = np.empty(v.shape, dtype=np.int8)
    classes[v <= t0] = 0
    classes[(v > t0) & (v <= t1)] = 1
    classes[(v > t1) & (v <= t2)] = 2
    classes[(v > t2) & (v <= t3)] = 3
    classes[v > t3] = 4
    counts = np.bincount(classes, minlength=5).astype(float)
    dyn = counts[1:]
    total = dyn.sum()
    if total <= 0:
        return np.zeros(4, dtype=float)
    return dyn / total


def usability_score(dynamic_shares: np.ndarray) -> tuple[float, bool]:
    entropy = normalized_entropy(dynamic_shares)
    class_floor = min(
        clipped_ratio(dynamic_shares[1], MIN_DYNAMIC_CLASS2_SHARE),
        clipped_ratio(dynamic_shares[2], MIN_DYNAMIC_CLASS3_SHARE),
        clipped_ratio(dynamic_shares[3], MIN_DYNAMIC_CLASS4_SHARE),
    )
    class1_score = (
        1.0
        if dynamic_shares[0] <= MAX_DYNAMIC_CLASS1_SHARE
        else float(
            np.clip(
                (1.0 - dynamic_shares[0]) / (1.0 - MAX_DYNAMIC_CLASS1_SHARE),
                0.0,
                1.0,
            )
        )
    )
    score = (
        USABILITY_W_ENTROPY * entropy
        + USABILITY_W_FLOOR * class_floor
        + USABILITY_W_CLASS1 * class1_score
    )
    feasible = bool(
        dynamic_shares[0] <= MAX_DYNAMIC_CLASS1_SHARE
        and dynamic_shares[1] >= MIN_DYNAMIC_CLASS2_SHARE
        and dynamic_shares[2] >= MIN_DYNAMIC_CLASS3_SHARE
        and dynamic_shares[3] >= MIN_DYNAMIC_CLASS4_SHARE
    )
    return float(score), feasible


def physical_score(
    samples: list[MorphologySample],
    cuts: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """Return (physical_score, spearman_phi_class, monotonicity)."""
    if len(samples) < 10:
        return float("nan"), float("nan"), float("nan")
    kpis = np.array([s.kpi_value for s in samples], dtype=float)
    phis = np.array([s.phi for s in samples], dtype=float)
    classes = np.array([assign_class(v, cuts) for v in kpis], dtype=float)
    valid = classes >= 0
    spearman = safe_spearman(phis[valid], classes[valid])

    # Monotonicity: mean φ should non-decrease with class index (skip empty).
    means = []
    for c in range(5):
        m = phis[classes == c]
        means.append(float(np.mean(m)) if m.size else float("nan"))
    mono_pairs = 0
    mono_ok = 0
    for a, b in zip(means[:-1], means[1:]):
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        mono_pairs += 1
        if b + 1e-9 >= a:
            mono_ok += 1
    mono = mono_ok / mono_pairs if mono_pairs else float("nan")

    sp = 0.0 if not np.isfinite(spearman) else max(spearman, 0.0)
    mn = 0.0 if not np.isfinite(mono) else mono
    score = PHYSICAL_W_SPEARMAN * sp + PHYSICAL_W_MONO * mn
    return (
        float(score),
        float(spearman) if np.isfinite(spearman) else float("nan"),
        float(mono) if np.isfinite(mono) else float("nan"),
    )


def _scenario_value_matrix(kpi_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (p95_per_scenario, values[n_scenarios, n_components] with NaN)."""
    meta = {"OP", "Contingency"}
    comps = [c for c in kpi_df.columns if c not in meta]
    mat = kpi_df[comps].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    with np.errstate(all="ignore"):
        p95 = np.nanquantile(mat, 0.95, axis=1)
    return p95, mat


def severity_proxy_score_from_matrix(
    p95: np.ndarray,
    value_matrix: np.ndarray,
    cuts: tuple[float, float, float, float],
) -> float:
    """Scenario P95 KPI vs mean class score — vectorized distribution proxy."""
    t0, t1, t2, t3 = cuts
    classes = np.full(value_matrix.shape, np.nan, dtype=float)
    finite = np.isfinite(value_matrix)
    v = value_matrix
    classes = np.where(finite & (v <= t0), 0.0, classes)
    classes = np.where(finite & (v > t0) & (v <= t1), 1.0, classes)
    classes = np.where(finite & (v > t1) & (v <= t2), 2.0, classes)
    classes = np.where(finite & (v > t2) & (v <= t3), 3.0, classes)
    classes = np.where(finite & (v > t3), 4.0, classes)
    with np.errstate(all="ignore"):
        mean_cls = np.nanmean(classes, axis=1)
    return safe_spearman(p95, mean_cls)


def evaluate_candidates(
    candidates: pd.DataFrame,
    kpi_df: pd.DataFrame,
    samples: list[MorphologySample],
    kpi_name: str,
) -> pd.DataFrame:
    meta = {"OP", "Contingency"}
    comps = [c for c in kpi_df.columns if c not in meta]
    # Flatten finite KPI values once.
    flat_vals: list[float] = []
    for c in comps:
        col = pd.to_numeric(kpi_df[c], errors="coerce").to_numpy(dtype=float)
        flat_vals.extend(col[np.isfinite(col)].tolist())
    flat = np.asarray(flat_vals, dtype=float)
    p95, value_matrix = _scenario_value_matrix(kpi_df)

    rows = []
    n_cand = len(candidates)
    t_start = time.time()
    for n, cand in enumerate(candidates.itertuples(index=False), start=1):
        # Skip mismatched baseline tags when evaluating one KPI.
        if cand.source == "current_voltage_bins" and kpi_name != "voltage":
            continue
        if cand.source == "current_spower_bins" and kpi_name != "spower":
            continue
        cuts = (float(cand.t0), float(cand.t1), float(cand.t2), float(cand.t3))
        shares = class_share_vector(flat, cuts)
        use_score, feasible = usability_score(shares)
        phys, spearman, mono = physical_score(samples, cuts)
        sev = severity_proxy_score_from_matrix(p95, value_matrix, cuts)
        sev_pos = 0.0 if not np.isfinite(sev) else max(sev, 0.0)
        phys_pos = 0.0 if not np.isfinite(phys) else phys
        combined = (
            COMBINED_W_USABILITY * use_score
            + COMBINED_W_PHYSICAL * phys_pos
            + COMBINED_W_SEVERITY * sev_pos
        )
        rows.append(
            {
                "candidate_id": int(cand.candidate_id),
                "KPI": kpi_name,
                "A": cand.A,
                "p1": cand.p1,
                "p2": cand.p2,
                "p3": cand.p3,
                "t0": cuts[0],
                "t1": cuts[1],
                "t2": cuts[2],
                "t3": cuts[3],
                "source": cand.source,
                "dynamic_class1_share": shares[0],
                "dynamic_class2_share": shares[1],
                "dynamic_class3_share": shares[2],
                "dynamic_class4_share": shares[3],
                "usability_score": use_score,
                "physical_score": phys,
                "spearman_phi_class": spearman,
                "phi_class_monotonicity": mono,
                "severity_proxy_spearman": sev,
                "combined_score": combined,
                "diagnostic_feasible": bool(
                    feasible
                    and np.isfinite(phys)
                    and phys >= MIN_PHYSICAL_FOR_FEASIBLE
                ),
            }
        )
        if n % 500 == 0 or n == n_cand:
            print(
                f"  [{kpi_name}] scored {n}/{n_cand} candidates "
                f"({time.time() - t_start:.1f}s)"
            )
    return pd.DataFrame(rows)


def pareto_front(
    frame: pd.DataFrame,
    objectives: list[str],
    *,
    sort_by: Optional[str] = None,
) -> pd.DataFrame:
    """Nondominated points maximizing all listed objectives (NSGA-II spirit)."""
    vals = frame[objectives].to_numpy(dtype=float)
    finite = np.all(np.isfinite(vals), axis=1)
    idxs = np.where(finite)[0]
    keep = []
    for i in idxs:
        dominated = False
        for j in idxs:
            if i == j:
                continue
            ge = np.all(vals[j] >= vals[i])
            gt = np.any(vals[j] > vals[i])
            if ge and gt:
                dominated = True
                break
        if not dominated:
            keep.append(i)
    out = frame.iloc[keep].copy()
    out["on_pareto_front"] = True
    if sort_by is None:
        for col in ("combined_score", "shared_score", *objectives):
            if col in out.columns:
                sort_by = col
                break
    if sort_by is not None and sort_by in out.columns:
        out = out.sort_values(sort_by, ascending=False)
    return out.reset_index(drop=True)


def merge_shared(
    voltage_metrics: pd.DataFrame,
    spower_metrics: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    # Shared cuts: only percentage_grid candidates (same absolute thresholds).
    grid_ids = set(
        candidates.loc[candidates["source"] == "percentage_grid", "candidate_id"]
    )
    v = voltage_metrics.loc[voltage_metrics["candidate_id"].isin(grid_ids)].add_prefix("voltage_")
    s = spower_metrics.loc[spower_metrics["candidate_id"].isin(grid_ids)].add_prefix("spower_")
    merged = candidates.loc[candidates["candidate_id"].isin(grid_ids)].merge(
        v, left_on="candidate_id", right_on="voltage_candidate_id"
    ).merge(s, left_on="candidate_id", right_on="spower_candidate_id")

    v_shares = merged[
        [
            "voltage_dynamic_class1_share",
            "voltage_dynamic_class2_share",
            "voltage_dynamic_class3_share",
            "voltage_dynamic_class4_share",
        ]
    ].to_numpy(float)
    s_shares = merged[
        [
            "spower_dynamic_class1_share",
            "spower_dynamic_class2_share",
            "spower_dynamic_class3_share",
            "spower_dynamic_class4_share",
        ]
    ].to_numpy(float)
    l1 = 0.5 * np.abs(v_shares - s_shares).sum(axis=1)
    merged["cross_kpi_consistency"] = 1.0 - np.clip(l1, 0.0, 1.0)
    merged["worst_usability"] = np.minimum(
        merged["voltage_usability_score"], merged["spower_usability_score"]
    )
    merged["worst_physical"] = np.minimum(
        merged["voltage_physical_score"].fillna(0.0),
        merged["spower_physical_score"].fillna(0.0),
    )
    merged["shared_score"] = (
        SHARED_W_USABILITY * merged["worst_usability"]
        + SHARED_W_PHYSICAL * merged["worst_physical"]
        + SHARED_W_COMBINED
        * 0.5
        * (
            merged["voltage_combined_score"].fillna(0.0)
            + merged["spower_combined_score"].fillna(0.0)
        )
        + SHARED_W_CONSISTENCY * merged["cross_kpi_consistency"]
    )
    merged["shared_feasible"] = (
        merged["voltage_diagnostic_feasible"] & merged["spower_diagnostic_feasible"]
    )
    return merged.sort_values(
        ["shared_feasible", "shared_score"], ascending=[False, False]
    ).reset_index(drop=True)


# =============================================================================
# Plots / report
# =============================================================================


def plot_tradeoff(frame: pd.DataFrame, pareto: pd.DataFrame, path: Path, title: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(
        frame["usability_score"],
        frame["physical_score"],
        s=12,
        alpha=0.35,
        label="candidates",
    )
    if len(pareto):
        plt.scatter(
            pareto["usability_score"],
            pareto["physical_score"],
            s=40,
            c="C3",
            label="Pareto front",
        )
    plt.xlabel("Usability score (class balance / entropy)")
    plt.ylabel("Physical score (φ–class Spearman + monotonicity)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# =============================================================================
# Main
# =============================================================================


def main(argv: Optional[list[str]] = None) -> None:
    settings = parse_args(argv)
    apply_settings(settings)
    import_dynagnn_modules(PROJECT_ROOT)

    out = prepare_output_directory(OUTPUT_DIR)
    (out / "run_settings.json").write_text(
        json.dumps(settings_to_jsonable(settings), indent=2),
        encoding="utf-8",
    )

    require = [
        PROJECT_ROOT,
        DATA_PATH,
        KPI_VOLTAGE_PATH,
        KPI_SPOWER_PATH,
        CONTINGENCIES_CSV,
        SPLIT_CSV,
        SIM_RESULTS_ROOT,
        SNOM_DIR,
    ]
    missing = [str(p) for p in require if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing paths:\n  " + "\n  ".join(missing))

    print("Loading training split / KPIs...")
    train_keys = load_train_keys(SPLIT_CSV)
    fault_to_cid = load_contingencies(CONTINGENCIES_CSV)
    df_v = filter_train(pd.read_csv(KPI_VOLTAGE_PATH), train_keys)
    df_s = filter_train(pd.read_csv(KPI_SPOWER_PATH), train_keys)
    print(f"Train scenarios: {len(train_keys)}")
    print(f"Voltage KPI rows: {len(df_v)} | Spower KPI rows: {len(df_s)}")

    candidates = build_candidate_grid()
    print(f"Candidate cut sets: {len(candidates)}")
    candidates.to_csv(out / "candidates.csv", index=False)

    rng = np.random.default_rng(MORPHOLOGY_RANDOM_SEED)
    print("\nExtracting voltage morphology samples from curves.xml...")
    samples_v = sample_morphology_cells(
        df_v, "voltage", fault_to_cid, MAX_MORPHOLOGY_SAMPLES_PER_KPI, rng
    )
    print("Extracting spower morphology samples from curves.xml...")
    samples_s = sample_morphology_cells(
        df_s, "spower", fault_to_cid, MAX_MORPHOLOGY_SAMPLES_PER_KPI, rng
    )
    pd.DataFrame([s.__dict__ for s in samples_v]).to_csv(
        out / "morphology_samples_voltage.csv", index=False
    )
    pd.DataFrame([s.__dict__ for s in samples_s]).to_csv(
        out / "morphology_samples_spower.csv", index=False
    )

    print("\nScoring voltage candidates...")
    metrics_v = evaluate_candidates(candidates, df_v, samples_v, "voltage")
    print("Scoring spower candidates...")
    metrics_s = evaluate_candidates(candidates, df_s, samples_s, "spower")
    metrics_v.to_csv(out / "metrics_voltage.csv", index=False)
    metrics_s.to_csv(out / "metrics_spower.csv", index=False)

    pareto_v = pareto_front(metrics_v, ["usability_score", "physical_score"])
    pareto_s = pareto_front(metrics_s, ["usability_score", "physical_score"])
    pareto_v.to_csv(out / "pareto_voltage.csv", index=False)
    pareto_s.to_csv(out / "pareto_spower.csv", index=False)

    plot_tradeoff(
        metrics_v,
        pareto_v,
        out / "plots" / "voltage_usability_vs_physical.png",
        "Voltage: usability vs physical separation",
    )
    plot_tradeoff(
        metrics_s,
        pareto_s,
        out / "plots" / "spower_usability_vs_physical.png",
        "Spower: usability vs physical separation",
    )

    rec_v = metrics_v.sort_values(
        ["diagnostic_feasible", "combined_score"], ascending=[False, False]
    ).iloc[0]
    rec_s = metrics_s.sort_values(
        ["diagnostic_feasible", "combined_score"], ascending=[False, False]
    ).iloc[0]

    recommendation: dict[str, Any] = {
        "method": {
            "multi_objective": "NSGA-II / MODiTS-style Pareto cut selection",
            "physical_features": "post-event excursion / variance / settling → φ",
            "references": [
                "Deb et al., IEEE TEC 2002 (NSGA-II)",
                "Márquez-Grajales et al., MODiTS/eMODiTS (multi-objective discretization)",
                "Pinzón and Colomé, LAAR 2019 (STVS dynamic indices for contingency severity)",
                "Frontiers Energy Research 2024 (disturbance-signal energy thresholding)",
            ],
        },
        "settings": settings_to_jsonable(settings),
        "voltage_recommended": {
            "candidate_id": int(rec_v.candidate_id),
            "cuts": [float(rec_v.t0), float(rec_v.t1), float(rec_v.t2), float(rec_v.t3)],
            "combined_score": float(rec_v.combined_score),
            "usability_score": float(rec_v.usability_score),
            "physical_score": float(rec_v.physical_score)
            if pd.notna(rec_v.physical_score)
            else None,
            "source": rec_v.source,
        },
        "spower_recommended": {
            "candidate_id": int(rec_s.candidate_id),
            "cuts": [float(rec_s.t0), float(rec_s.t1), float(rec_s.t2), float(rec_s.t3)],
            "combined_score": float(rec_s.combined_score),
            "usability_score": float(rec_s.usability_score),
            "physical_score": float(rec_s.physical_score)
            if pd.notna(rec_s.physical_score)
            else None,
            "source": rec_s.source,
        },
    }

    if SHARED_CUTS:
        print("\nBuilding shared-cut ranking...")
        shared = merge_shared(metrics_v, metrics_s, candidates)
        shared.to_csv(out / "shared_ranking.csv", index=False)
        shared_for_pareto = shared.rename(
            columns={
                "worst_usability": "usability_score",
                "worst_physical": "physical_score",
            }
        )
        shared_pareto = pareto_front(
            shared_for_pareto,
            ["usability_score", "physical_score"],
            sort_by="shared_score",
        )
        shared_pareto.to_csv(out / "shared_pareto.csv", index=False)
        rec_shared = shared.iloc[0]
        recommendation["shared_recommended"] = {
            "candidate_id": int(rec_shared.candidate_id),
            "cuts": [
                float(rec_shared.t0),
                float(rec_shared.t1),
                float(rec_shared.t2),
                float(rec_shared.t3),
            ],
            "A": float(rec_shared.A) if pd.notna(rec_shared.A) else None,
            "p1": float(rec_shared.p1) if pd.notna(rec_shared.p1) else None,
            "p2": float(rec_shared.p2) if pd.notna(rec_shared.p2) else None,
            "p3": float(rec_shared.p3) if pd.notna(rec_shared.p3) else None,
            "shared_score": float(rec_shared.shared_score),
            "worst_usability": float(rec_shared.worst_usability),
            "worst_physical": float(rec_shared.worst_physical),
            "cross_kpi_consistency": float(rec_shared.cross_kpi_consistency),
        }
        print(
            "Shared recommended cuts: "
            f"{recommendation['shared_recommended']['cuts']}"
        )

    (out / "recommendation.json").write_text(
        json.dumps(recommendation, indent=2), encoding="utf-8"
    )

    print("\nVoltage recommended cuts:", recommendation["voltage_recommended"]["cuts"])
    print("Spower recommended cuts:", recommendation["spower_recommended"]["cuts"])
    print(f"\nResults written to: {out}")
    print("Copy recommended cuts into config.yaml under kpi.class_bins.*.cuts.")


if __name__ == "__main__":
    main()
