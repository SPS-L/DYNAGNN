# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: KPI preprocessing pipeline histograms

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from modules.normalization import (
    ID_COLS,
    log10_transform_values,
    prepare_kpi_matrix,
    transform_values,
)

FIGSIZE = (12, 7)
N_BINS = 100


def _flatten_finite(values: np.ndarray) -> np.ndarray:
    flat = values.ravel()
    return flat[np.isfinite(flat)]


def _positive_value_xlim(values: np.ndarray) -> tuple[float, float]:
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        raise ValueError("No positive KPI values available for histogram x-limits.")
    lo = float(positive.min())
    hi = float(positive.max())
    log_lo = np.log10(lo)
    log_hi = np.log10(hi)
    margin = max(0.05 * (log_hi - log_lo), 0.5)
    return 10 ** (log_lo - margin), 10 ** (log_hi + margin)


def _log_histogram_edges(xlim: tuple[float, float], n_bins: int = N_BINS) -> np.ndarray:
    return np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), n_bins)


def _compute_log_count_ylim(counts: np.ndarray) -> tuple[float, float]:
    positive_counts = counts[counts > 0]
    ymin = max(1e0, positive_counts.min() if positive_counts.size else 1e0)
    ymax = max(counts.max(), 1e1)
    return ymin, ymax


def plot_raw_kpi_histogram(
    values: np.ndarray,
    *,
    kpi_type: str,
    output_path: Path,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Histogram of raw KPI values with log-scaled x and y axes."""
    import matplotlib.pyplot as plt

    flat = _flatten_finite(values)
    if xlim is None:
        xlim = _positive_value_xlim(flat)

    bins = _log_histogram_edges(xlim)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    counts, _, _ = ax.hist(
        flat, bins=bins, color="steelblue", edgecolor="white", linewidth=0.2
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    if ylim is None:
        ylim = _compute_log_count_ylim(counts)
    ax.set_ylim(*ylim)
    ax.set_xlabel("KPI value")
    ax.set_ylabel("Count")
    ax.set_title(f"Raw KPI distribution ({kpi_type})")
    ax.grid(True, which="both", axis="both", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ylim


def plot_log10_kpi_histogram(
    log10_values: np.ndarray,
    *,
    kpi_type: str,
    output_path: Path,
    raw_xlim: tuple[float, float],
    ylim: tuple[float, float] | None = None,
) -> None:
    """Histogram of log10(KPI) values with log-scaled counts."""
    import matplotlib.pyplot as plt

    flat = _flatten_finite(log10_values)
    xlim_log10 = (np.log10(raw_xlim[0]), np.log10(raw_xlim[1]))
    bins = np.linspace(xlim_log10[0], xlim_log10[1], N_BINS)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    counts, _, _ = ax.hist(
        flat, bins=bins, color="steelblue", edgecolor="white", linewidth=0.2
    )
    ax.set_yscale("log")
    ax.set_xlim(*xlim_log10)
    if ylim is None:
        ylim = _compute_log_count_ylim(counts)
    ax.set_ylim(*ylim)
    ax.set_xlabel("log10(KPI value)")
    ax.set_ylabel("Count")
    ax.set_title(f"log10(KPI) distribution ({kpi_type})")
    ax.grid(True, which="both", axis="both", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_zscore_kpi_histogram(
    z_values: np.ndarray,
    *,
    kpi_type: str,
    output_path: Path,
    z_cuts: np.ndarray,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Histogram of z-scored log10(KPI) values with class-cut lines."""
    import matplotlib.pyplot as plt

    flat = _flatten_finite(z_values)
    if flat.size == 0:
        raise ValueError(f"No finite z-score values available to plot for {kpi_type}.")

    vmin = float(flat.min())
    vmax = float(flat.max())
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5
    margin = 0.05 * (vmax - vmin)
    xlim = (vmin - margin, vmax + margin)
    bins = np.linspace(xlim[0], xlim[1], N_BINS)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    counts, _, _ = ax.hist(
        flat, bins=bins, color="steelblue", edgecolor="white", linewidth=0.2
    )
    for z_cut in z_cuts:
        ax.axvline(float(z_cut), color="red", linewidth=2.0, linestyle="-")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    if ylim is None:
        ylim = _compute_log_count_ylim(counts)
    ax.set_ylim(*ylim)
    ax.set_xlabel("z-score of log10(KPI value)")
    ax.set_ylabel("Count")
    ax.set_title(f"z-score of log10(KPI) distribution ({kpi_type}) — range class cuts")
    ax.grid(True, which="both", axis="both", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_kpi_pipeline_histograms(
    *,
    kpi_df: pd.DataFrame,
    kpi_type: str,
    split_lookup: Dict[Tuple[str, str], str],
    scaler: StandardScaler,
    z_cuts: np.ndarray,
    output_dir: Path,
) -> list[Path]:
    """
    Plot raw KPI (log x-axis), log10(KPI), and z-score histograms with class cuts.

    Uses the same zero-replace and log10 steps as ``build_class_dataset_for_type``.
    """
    kpi_df = kpi_df.copy()
    kpi_df.columns = [str(c).strip() for c in kpi_df.columns]
    value_cols = [c for c in kpi_df.columns if c not in ID_COLS]
    if not value_cols:
        raise ValueError(f"No KPI value columns found for {kpi_type}")

    prepared_values, _ = prepare_kpi_matrix(kpi_df, value_cols)
    log10_values = log10_transform_values(prepared_values)
    z_values = transform_values(scaler, log10_values)

    flat_prepared = _flatten_finite(prepared_values)
    raw_xlim = _positive_value_xlim(flat_prepared)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"KPI_{kpi_type}_histogram.png"
    log10_path = output_dir / f"KPI_{kpi_type}_log10_histogram.png"
    zscore_path = output_dir / f"KPI_{kpi_type}_zscore_histogram_class_cuts.png"

    raw_ylim = plot_raw_kpi_histogram(
        prepared_values,
        kpi_type=kpi_type,
        output_path=raw_path,
        xlim=raw_xlim,
    )
    plot_log10_kpi_histogram(
        log10_values,
        kpi_type=kpi_type,
        output_path=log10_path,
        raw_xlim=raw_xlim,
        ylim=raw_ylim,
    )
    plot_zscore_kpi_histogram(
        z_values,
        kpi_type=kpi_type,
        output_path=zscore_path,
        z_cuts=z_cuts,
        ylim=raw_ylim,
    )
    return [raw_path, log10_path, zscore_path]


def plot_all_kpi_pipeline_histograms(
    *,
    voltage_df: pd.DataFrame,
    spower_df: pd.DataFrame,
    split_lookup: Dict[Tuple[str, str], str],
    voltage_scaler: StandardScaler,
    spower_scaler: StandardScaler,
    voltage_z_cuts: np.ndarray,
    spower_z_cuts: np.ndarray,
    output_dir: Path,
) -> list[Path]:
    """Write pipeline histograms for voltage and spower under ``output_dir``."""
    outputs: list[Path] = []
    outputs.extend(
        plot_kpi_pipeline_histograms(
            kpi_df=voltage_df,
            kpi_type="voltage",
            split_lookup=split_lookup,
            scaler=voltage_scaler,
            z_cuts=voltage_z_cuts,
            output_dir=output_dir,
        )
    )
    outputs.extend(
        plot_kpi_pipeline_histograms(
            kpi_df=spower_df,
            kpi_type="spower",
            split_lookup=split_lookup,
            scaler=spower_scaler,
            z_cuts=spower_z_cuts,
            output_dir=output_dir,
        )
    )
    return outputs
