# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: Electrical distance computation

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    import pypowsybl as pp
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit("Missing dependency: 'pypowsybl'.") from e

from scipy.linalg import solve as dense_solve
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu

# Reduced-bus count at or below which dense LAPACK (Z = Y^{-1} via solve(I)) beats sparse LU.
_DENSE_Z_CROSSOVER = 384

from .graph_construction import _parse_bool, _parse_float, _resolve_iidm_path

ElectricalDistanceRow = Tuple[str, str, float]
DistanceTable = Union[pd.DataFrame, Sequence[ElectricalDistanceRow]]
DistanceMode = str
_MODE_VOLTAGE_LEVEL = "voltage_level"
_MODE_SUBSTATION = "substation"
_VOLTAGE_LEVEL_COLUMNS = ("VLi", "VLj", "dij")
_SUBSTATION_COLUMNS = ("Substationi", "Substationj", "dij")


@dataclass
class ElectricDistanceResult:
    voltage_level_ids: List[str]
    distances: pd.DataFrame
    source_iidm_path: str
    y_bus: Optional[np.ndarray] = None
    z_bus: Optional[np.ndarray] = None


def branch_admittance(r: float, x: float) -> complex:
    """Series admittance y = 1 / (R + jX)."""
    z = complex(_parse_float(r), _parse_float(x))
    if abs(z) == 0.0:
        return 0j
    return 1.0 / z


def branch_admittances(r: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Vectorized series admittance y = 1 / (R + jX); zero-Z branches -> 0."""
    r = np.asarray(r, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    z = r + 1j * x
    y = np.zeros_like(z, dtype=np.complex128)
    nonzero = np.abs(z) > 0.0
    y[nonzero] = 1.0 / z[nonzero]
    return y


def build_y_bus_sparse(
    n: int,
    branch_i: np.ndarray,
    branch_j: np.ndarray,
    branch_y: np.ndarray,
    branch_y_shunt_i: Optional[np.ndarray] = None,
    branch_y_shunt_j: Optional[np.ndarray] = None,
) -> csc_matrix:
    """Assemble sparse complex Y-bus from branch endpoint indices and admittances."""
    if branch_i.size == 0:
        return csc_matrix((n, n), dtype=np.complex128)

    branch_i = np.asarray(branch_i, dtype=np.intp)
    branch_j = np.asarray(branch_j, dtype=np.intp)
    branch_y = np.asarray(branch_y, dtype=np.complex128)
    if branch_y_shunt_i is None:
        branch_y_shunt_i = np.zeros(branch_y.shape, dtype=np.complex128)
    else:
        branch_y_shunt_i = np.asarray(branch_y_shunt_i, dtype=np.complex128)
    if branch_y_shunt_j is None:
        branch_y_shunt_j = np.zeros(branch_y.shape, dtype=np.complex128)
    else:
        branch_y_shunt_j = np.asarray(branch_y_shunt_j, dtype=np.complex128)

    keep = (branch_y != 0) | (branch_y_shunt_i != 0) | (branch_y_shunt_j != 0)
    if not np.any(keep):
        return csc_matrix((n, n), dtype=np.complex128)

    branch_i = branch_i[keep]
    branch_j = branch_j[keep]
    branch_y = branch_y[keep]
    branch_y_shunt_i = branch_y_shunt_i[keep]
    branch_y_shunt_j = branch_y_shunt_j[keep]

    rows = np.concatenate([branch_i, branch_j, branch_i, branch_j])
    cols = np.concatenate([branch_i, branch_j, branch_j, branch_i])
    data = np.concatenate(
        [
            branch_y + branch_y_shunt_i,
            branch_y + branch_y_shunt_j,
            -branch_y,
            -branch_y,
        ]
    )
    y_bus = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsc()
    y_bus.eliminate_zeros()
    return y_bus


def build_y_bus(
    voltage_level_ids: Sequence[str],
    branch_i: np.ndarray,
    branch_j: np.ndarray,
    branch_y: np.ndarray,
    branch_y_shunt_i: Optional[np.ndarray] = None,
    branch_y_shunt_j: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Dense Y-bus (used only when callers request stored matrices)."""
    n = len(voltage_level_ids)
    y_sparse = build_y_bus_sparse(n, branch_i, branch_j, branch_y, branch_y_shunt_i, branch_y_shunt_j)
    return y_sparse.toarray()


def _connected_components_from_sparse_y(y_bus: csc_matrix) -> List[np.ndarray]:
    n = y_bus.shape[0]
    if n == 0:
        return []

    rows, cols = y_bus.nonzero()
    if rows.size == 0:
        return [np.array([i], dtype=np.intp) for i in range(n)]

    graph = csr_matrix(
        (np.ones(rows.size, dtype=np.uint8), (rows, cols)),
        shape=(n, n),
    )
    graph = graph + graph.T
    _, labels = connected_components(graph, directed=False, connection="weak")
    labels = np.asarray(labels, dtype=np.intp)
    return [np.flatnonzero(labels == label).astype(np.intp) for label in range(labels.max() + 1)]


def _invert_reduced_y(y_red: csc_matrix) -> np.ndarray:
    """Z_red = Y_red^{-1} for the grounded (reduced) admittance block."""
    n = y_red.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.complex128)
    if n == 1:
        y_ii = y_red[0, 0]
        if y_ii == 0:
            return np.zeros((1, 1), dtype=np.complex128)
        return np.array([[1.0 / y_ii]], dtype=np.complex128)

    if n <= _DENSE_Z_CROSSOVER:
        # Small islands: dense LAPACK zsysv on Y_red is faster than sparse LU setup.
        y_dense = y_red.toarray()
        rhs = np.eye(n, dtype=np.complex128)
        return dense_solve(y_dense, rhs, assume_a="sym")

    lu = splu(y_red, permc_spec="COLAMD")
    # Fortran-order RHS helps threaded BLAS in SuperLU's triangular solves.
    rhs = np.eye(n, dtype=np.complex128, order="F")
    return lu.solve(rhs)


def _invert_component_sparse(y_bus: csc_matrix, component: np.ndarray) -> Tuple[np.ndarray, int]:
    """Return (Z_red, ref_index) for one connected component."""
    component = np.asarray(component, dtype=np.intp)
    if component.size == 1:
        idx = int(component[0])
        y_ii = y_bus[idx, idx]
        if y_ii == 0:
            return np.zeros((1, 1), dtype=np.complex128), idx
        return np.array([[1.0 / y_ii]], dtype=np.complex128), idx

    ref = int(component[0])
    non_ref = component[1:]
    y_red = y_bus[non_ref][:, non_ref].tocsc()
    return _invert_reduced_y(y_red), ref


def _distances_from_z_red(
    component: np.ndarray,
    z_red: np.ndarray,
    ref: int,
    n_buses: int,
) -> np.ndarray:
    """|Z_ij| for all ordered pairs in *component* from grounded- bus reduction."""
    component = np.asarray(component, dtype=np.intp)
    m = component.size
    if m == 0:
        return np.empty(0, dtype=np.float64)

    if m == 1:
        return np.array([np.abs(z_red[0, 0])], dtype=np.float64)

    non_ref = component[1:]
    inv = np.full(n_buses, -1, dtype=np.intp)
    inv[non_ref] = np.arange(non_ref.size)

    ii, jj = _component_pair_indices(component)
    d_red = np.abs(z_red)
    diag = np.diag(d_red)

    out = np.empty(ii.size, dtype=np.float64)
    both_nr = (ii != ref) & (jj != ref)
    out[both_nr] = d_red[inv[ii[both_nr]], inv[jj[both_nr]]]

    ref_i = ii == ref
    ref_j = jj == ref
    out[ref_i & (jj != ref)] = diag[inv[jj[ref_i & (jj != ref)]]]
    out[ref_j & (ii != ref)] = diag[inv[ii[ref_j & (ii != ref)]]]
    # Grounded reference bus: Z_ref,ref is not part of the reduced matrix (legacy dense path = 0).
    out[ref_i & ref_j] = 0.0
    return out


def _component_pair_indices(component: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    component = np.asarray(component, dtype=np.intp)
    m = component.size
    if m == 0:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty
    return np.repeat(component, m), np.tile(component, m)


_CSV_CHUNK_ROWS = 500_000


def _component_distance_matrix(
    component: np.ndarray,
    z_red: np.ndarray,
    ref: int,
) -> np.ndarray:
    """|Z_ij| for all ordered pairs in *component* (m x m, row-major flatten order)."""
    component = np.asarray(component, dtype=np.intp)
    m = component.size
    if m == 1:
        return np.array([[float(np.abs(z_red[0, 0]))]], dtype=np.float64)

    d_red = np.abs(z_red)
    diag = np.diag(d_red)
    out = np.empty((m, m), dtype=np.float64)
    out[0, 0] = 0.0
    out[0, 1:] = diag
    out[1:, 0] = diag
    out[1:, 1:] = d_red
    return out


def _write_distance_rows_chunked(
    writer: csv.writer,
    vli: np.ndarray,
    vlj: np.ndarray,
    dij: np.ndarray,
    *,
    chunk_rows: int = _CSV_CHUNK_ROWS,
) -> int:
    """Write (VLi, VLj, dij) rows in chunks; return number of finite rows written."""
    n = dij.size
    if n == 0:
        return 0

    rows_written = 0
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        d_chunk = dij[start:end]
        finite = np.isfinite(d_chunk)
        if not np.any(finite):
            continue
        vli_chunk = vli[start:end][finite]
        vlj_chunk = vlj[start:end][finite]
        d_chunk = d_chunk[finite]
        writer.writerows(
            zip(
                vli_chunk.tolist(),
                vlj_chunk.tolist(),
                (f"{float(d):.12g}" for d in d_chunk),
                strict=True,
            )
        )
        rows_written += int(finite.sum())
    return rows_written


def _stream_component_distances(
    writer: csv.writer,
    vl: np.ndarray,
    component: np.ndarray,
    z_red: np.ndarray,
    ref: int,
) -> int:
    """Write all ordered distance pairs for one component (vectorized, chunked CSV)."""
    component = np.asarray(component, dtype=np.intp)
    m = component.size
    if m == 0:
        return 0

    vl_comp = vl[component]
    dist = _component_distance_matrix(component, z_red, ref)
    vli = np.repeat(vl_comp, m)
    vlj = np.tile(vl_comp, m)
    return _write_distance_rows_chunked(writer, vli, vlj, dist.ravel())


def stream_electrical_distance_csv(
    y_bus: csc_matrix,
    components: List[np.ndarray],
    voltage_level_ids: Sequence[str],
    output_csv: Path,
    *,
    columns: Tuple[str, str, str] = _VOLTAGE_LEVEL_COLUMNS,
) -> int:
    """Write VLi/VLj/dij CSV from sparse Y without building a full distance DataFrame."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if len(voltage_level_ids) == 0:
        pd.DataFrame(columns=list(columns)).to_csv(output_csv, index=False)
        return 0

    vl = np.asarray(voltage_level_ids, dtype=object)
    rows_written = 0
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        csv_writer = csv.writer(handle, lineterminator="\n")
        csv_writer.writerow(list(columns))
        for component in components:
            if component.size == 0:
                continue
            z_red, ref = _invert_component_sparse(y_bus, component)
            rows_written += _stream_component_distances(csv_writer, vl, component, z_red, ref)
    return rows_written


def electrical_distances_dataframe(
    y_bus: csc_matrix,
    components: List[np.ndarray],
    voltage_level_ids: Sequence[str],
    *,
    columns: Tuple[str, str, str] = _VOLTAGE_LEVEL_COLUMNS,
) -> pd.DataFrame:
    if len(voltage_level_ids) == 0:
        return pd.DataFrame(columns=list(columns))

    vl = np.asarray(voltage_level_ids)
    vli_parts: List[np.ndarray] = []
    vlj_parts: List[np.ndarray] = []
    dij_parts: List[np.ndarray] = []

    for component in components:
        if component.size == 0:
            continue
        z_red, ref = _invert_component_sparse(y_bus, component)
        ii, jj = _component_pair_indices(component)
        distances = _distances_from_z_red(component, z_red, ref, y_bus.shape[0])
        if not np.any(np.isfinite(distances)):
            continue
        vli_parts.append(vl[ii])
        vlj_parts.append(vl[jj])
        dij_parts.append(distances)

    if not dij_parts:
        return pd.DataFrame(columns=list(columns))

    return pd.DataFrame(
        {
            columns[0]: np.concatenate(vli_parts),
            columns[1]: np.concatenate(vlj_parts),
            columns[2]: np.concatenate(dij_parts),
        }
    )


def electrical_distances_from_z(
    z_bus: np.ndarray,
    y_bus: np.ndarray,
    voltage_level_ids: Sequence[str],
) -> List[ElectricalDistanceRow]:
    """Legacy API when a dense Z-bus is already available."""
    y_sparse = coo_matrix(y_bus).tocsc()
    df = electrical_distances_dataframe(
        y_sparse,
        _connected_components_from_sparse_y(y_sparse),
        voltage_level_ids,
    )
    if df.empty:
        return []
    return list(zip(df["VLi"].tolist(), df["VLj"].tolist(), df["dij"].tolist()))


def _connected_mask(series: pd.Series) -> np.ndarray:
    """Vectorized connected flag (bool column or 'true'/'false' strings)."""
    if series.dtype == bool:
        return series.to_numpy()
    return series.astype(str).str.strip().str.lower().eq("true").to_numpy()


def _active_topology_from_network(network) -> Tuple[frozenset, frozenset]:
    buses = network.get_buses()
    active = (buses["connected_component"].to_numpy() == 0) & (
        buses["v_mag"].fillna(0.0).to_numpy() != 0.0
    )
    kept = buses.loc[active]
    kept_bus_ids = frozenset(kept.index.astype(str))
    active_vls = frozenset(kept["voltage_level_id"].dropna().astype(str))
    return kept_bus_ids, active_vls


def _filter_branch_table(
    table: pd.DataFrame,
    kept_bus_ids: frozenset,
    active_voltage_levels: frozenset,
    *,
    bus1_col: str = "bus1_id",
    bus2_col: str = "bus2_id",
    vl1_col: str = "voltage_level1_id",
    vl2_col: str = "voltage_level2_id",
    r_col: str = "r",
    x_col: str = "x",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    empty = np.empty(0, dtype=np.float64)
    if table.empty:
        return empty, empty, empty, empty

    sub = table[
        table[bus1_col].astype(str).isin(kept_bus_ids)
        & table[bus2_col].astype(str).isin(kept_bus_ids)
    ]
    sub = sub[
        sub[vl1_col].astype(str).isin(active_voltage_levels)
        & sub[vl2_col].astype(str).isin(active_voltage_levels)
    ]
    if sub.empty:
        return empty, empty, empty, empty

    r = pd.to_numeric(sub[r_col], errors="coerce").to_numpy()
    x = pd.to_numeric(sub[x_col], errors="coerce").to_numpy()
    return (
        sub[vl1_col].astype(str).to_numpy(),
        sub[vl2_col].astype(str).to_numpy(),
        np.nan_to_num(r, nan=0.0),
        np.nan_to_num(x, nan=0.0),
    )


def _filter_line_table(
    table: pd.DataFrame,
    kept_bus_ids: frozenset,
    active_voltage_levels: frozenset,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    empty = np.empty(0, dtype=np.float64)
    if table.empty:
        return empty, empty, empty, empty, empty, empty

    vl1, vl2, r, x = _filter_branch_table(table, kept_bus_ids, active_voltage_levels)
    if vl1.size == 0:
        return empty, empty, empty, empty, empty, empty

    sub = table[
        table["bus1_id"].astype(str).isin(kept_bus_ids)
        & table["bus2_id"].astype(str).isin(kept_bus_ids)
    ]
    sub = sub[
        sub["voltage_level1_id"].astype(str).isin(active_voltage_levels)
        & sub["voltage_level2_id"].astype(str).isin(active_voltage_levels)
    ]

    g1 = _numeric_or_zero(sub, "g1")
    b1 = _numeric_or_zero(sub, "b1")
    g2 = _numeric_or_zero(sub, "g2")
    b2 = _numeric_or_zero(sub, "b2")
    y_shunt_1 = np.nan_to_num(g1, nan=0.0) + 1j * np.nan_to_num(b1, nan=0.0)
    y_shunt_2 = np.nan_to_num(g2, nan=0.0) + 1j * np.nan_to_num(b2, nan=0.0)
    return vl1, vl2, r, x, y_shunt_1, y_shunt_2


def _numeric_or_zero(table: pd.DataFrame, column: str) -> np.ndarray:
    if column not in table.columns:
        return np.zeros(len(table), dtype=np.float64)
    return pd.to_numeric(table[column], errors="coerce").to_numpy()


def _empty_branch_arrays() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    empty = np.empty(0, dtype=np.float64)
    return empty, empty, empty, empty, empty, empty


def _collect_branch_arrays(
    network,
    kept_bus_ids: frozenset,
    active_voltage_levels: frozenset,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vl1_parts: List[np.ndarray] = []
    vl2_parts: List[np.ndarray] = []
    r_parts: List[np.ndarray] = []
    x_parts: List[np.ndarray] = []
    y_shunt_1_parts: List[np.ndarray] = []
    y_shunt_2_parts: List[np.ndarray] = []

    lines = network.get_lines()
    if not lines.empty:
        line_mask = _connected_mask(lines["connected1"]) & _connected_mask(lines["connected2"])
        vl1, vl2, r, x, y_shunt_1, y_shunt_2 = _filter_line_table(
            lines.loc[line_mask],
            kept_bus_ids,
            active_voltage_levels,
        )
        if vl1.size:
            vl1_parts.append(vl1)
            vl2_parts.append(vl2)
            r_parts.append(r)
            x_parts.append(x)
            y_shunt_1_parts.append(y_shunt_1)
            y_shunt_2_parts.append(y_shunt_2)

    tr2 = network.get_2_windings_transformers()
    if not tr2.empty:
        tr2_mask = _connected_mask(tr2["connected1"]) & _connected_mask(tr2["connected2"])
        vl1, vl2, r, x = _filter_branch_table(tr2.loc[tr2_mask], kept_bus_ids, active_voltage_levels)
        if vl1.size:
            vl1_parts.append(vl1)
            vl2_parts.append(vl2)
            r_parts.append(r)
            x_parts.append(x)
            y_shunt_1_parts.append(np.zeros(vl1.shape, dtype=np.complex128))
            y_shunt_2_parts.append(np.zeros(vl2.shape, dtype=np.complex128))

    tr3 = network.get_3_windings_transformers()
    for _, row in tr3.iterrows():
        sides = []
        for side in (1, 2, 3):
            if not _parse_bool(row.get(f"connected{side}")):
                continue
            if str(row.get(f"bus{side}_id", "")) not in kept_bus_ids:
                continue
            vl_id = str(row.get(f"voltage_level{side}_id", ""))
            if vl_id not in active_voltage_levels:
                continue
            sides.append(
                (
                    vl_id,
                    _parse_float(row.get(f"r{side}")),
                    _parse_float(row.get(f"x{side}")),
                )
            )
        for a in range(len(sides)):
            for b in range(a + 1, len(sides)):
                vl_a, r_a, x_a = sides[a]
                vl_b, r_b, x_b = sides[b]
                vl1_parts.append(np.array([vl_a], dtype=object))
                vl2_parts.append(np.array([vl_b], dtype=object))
                r_parts.append(np.array([r_a + r_b], dtype=np.float64))
                x_parts.append(np.array([x_a + x_b], dtype=np.float64))
                y_shunt_1_parts.append(np.zeros(1, dtype=np.complex128))
                y_shunt_2_parts.append(np.zeros(1, dtype=np.complex128))

    if not vl1_parts:
        return _empty_branch_arrays()

    return (
        np.concatenate(vl1_parts),
        np.concatenate(vl2_parts),
        np.concatenate(r_parts),
        np.concatenate(x_parts),
        np.concatenate(y_shunt_1_parts),
        np.concatenate(y_shunt_2_parts),
    )


def _voltage_level_substation_id(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    if not text or text == "nan":
        return None
    return text


def _active_voltage_level_to_substation(network, active_voltage_levels: frozenset) -> Dict[str, str]:
    voltage_levels = network.get_voltage_levels()
    if voltage_levels.empty:
        return {}

    out: Dict[str, str] = {}
    for voltage_level_id in active_voltage_levels:
        if voltage_level_id not in voltage_levels.index:
            continue
        substation_id = _voltage_level_substation_id(voltage_levels.loc[voltage_level_id].get("substation_id"))
        if substation_id is not None:
            out[str(voltage_level_id)] = substation_id
    return out


def _collect_substation_branch_arrays(
    network,
    kept_bus_ids: frozenset,
    active_voltage_levels: frozenset,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, frozenset]:
    """Collect AC-line branches mapped to substation ids; transformers and HVDC are ignored."""
    vl_to_substation = _active_voltage_level_to_substation(network, active_voltage_levels)
    active_substations = frozenset(vl_to_substation.values())
    lines = network.get_lines()
    if lines.empty or not active_substations:
        return (*_empty_branch_arrays(), active_substations)

    line_mask = _connected_mask(lines["connected1"]) & _connected_mask(lines["connected2"])
    vl1, vl2, r, x, y_shunt_1, y_shunt_2 = _filter_line_table(
        lines.loc[line_mask],
        kept_bus_ids,
        active_voltage_levels,
    )
    if vl1.size == 0:
        return (*_empty_branch_arrays(), active_substations)

    sub1 = np.asarray([vl_to_substation.get(str(vl_id), "") for vl_id in vl1], dtype=object)
    sub2 = np.asarray([vl_to_substation.get(str(vl_id), "") for vl_id in vl2], dtype=object)
    keep = (sub1 != "") & (sub2 != "")
    if not np.any(keep):
        return (*_empty_branch_arrays(), active_substations)

    return (
        sub1[keep],
        sub2[keep],
        r[keep],
        x[keep],
        y_shunt_1[keep],
        y_shunt_2[keep],
        active_substations,
    )


def _validate_distance_mode(mode: DistanceMode) -> DistanceMode:
    if mode not in {_MODE_VOLTAGE_LEVEL, _MODE_SUBSTATION}:
        raise ValueError(f"Unsupported electrical-distance mode {mode!r}; use 'voltage_level' or 'substation'.")
    return mode


def _distance_columns_for_mode(mode: DistanceMode) -> Tuple[str, str, str]:
    return _SUBSTATION_COLUMNS if mode == _MODE_SUBSTATION else _VOLTAGE_LEVEL_COLUMNS


def _branch_arrays_for_mode(
    network,
    mode: DistanceMode,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Sequence[str]]:
    kept_bus_ids, active_voltage_levels = _active_topology_from_network(network)
    if mode == _MODE_SUBSTATION:
        sub1, sub2, r, x, y_shunt_1, y_shunt_2, active_substations = _collect_substation_branch_arrays(
            network,
            kept_bus_ids,
            active_voltage_levels,
        )
        return sub1, sub2, r, x, y_shunt_1, y_shunt_2, sorted(active_substations)

    vl1, vl2, r, x, y_shunt_1, y_shunt_2 = _collect_branch_arrays(
        network,
        kept_bus_ids,
        active_voltage_levels,
    )
    return vl1, vl2, r, x, y_shunt_1, y_shunt_2, sorted(active_voltage_levels)


def _branch_index_arrays(
    vl1: np.ndarray,
    vl2: np.ndarray,
    voltage_level_ids: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Map branch VL ids to indices in the full active-VL bus list.

    The Y-bus is ``len(voltage_level_ids) x len(voltage_level_ids)``. Entries
    without a series branch stay zero (``Y_ij = 0``); buses are not dropped.
    """
    index = {vl_id: i for i, vl_id in enumerate(voltage_level_ids)}
    branch_i = np.fromiter((index[str(v)] for v in vl1), dtype=np.intp, count=vl1.size)
    branch_j = np.fromiter((index[str(v)] for v in vl2), dtype=np.intp, count=vl2.size)
    return branch_i, branch_j


def _compute_from_branch_arrays(
    vl1: np.ndarray,
    vl2: np.ndarray,
    r: np.ndarray,
    x: np.ndarray,
    y_shunt_1: Optional[np.ndarray],
    y_shunt_2: Optional[np.ndarray],
    node_ids: Sequence[str],
    *,
    store_matrices: bool = False,
    materialize_distances: bool = True,
    distance_columns: Tuple[str, str, str] = _VOLTAGE_LEVEL_COLUMNS,
) -> Tuple[List[str], csc_matrix, List[np.ndarray], Optional[pd.DataFrame], Optional[np.ndarray], Optional[np.ndarray]]:
    node_ids = sorted(node_ids)
    if not node_ids:
        empty_df = pd.DataFrame(columns=list(distance_columns))
        return [], csc_matrix((0, 0), dtype=np.complex128), [], empty_df, None, None

    if vl1.size == 0:
        branch_i = np.empty(0, dtype=np.intp)
        branch_j = np.empty(0, dtype=np.intp)
        branch_y = np.empty(0, dtype=np.complex128)
    else:
        branch_i, branch_j = _branch_index_arrays(vl1, vl2, node_ids)
        branch_y = branch_admittances(r, x)
    y_bus = build_y_bus_sparse(len(node_ids), branch_i, branch_j, branch_y, y_shunt_1, y_shunt_2)
    components = _connected_components_from_sparse_y(y_bus)

    distances: Optional[pd.DataFrame]
    if materialize_distances:
        distances = electrical_distances_dataframe(y_bus, components, node_ids, columns=distance_columns)
    else:
        distances = pd.DataFrame(columns=list(distance_columns))

    y_dense: Optional[np.ndarray] = None
    z_dense: Optional[np.ndarray] = None
    if store_matrices:
        y_dense = build_y_bus(node_ids, branch_i, branch_j, branch_y, y_shunt_1, y_shunt_2)
        z_dense = np.zeros_like(y_dense)
        for component in components:
            z_red, ref = _invert_component_sparse(y_bus, component)
            if component.size == 1:
                z_dense[component[0], component[0]] = z_red[0, 0]
                continue
            non_ref = component[1:]
            z_dense[np.ix_(non_ref, non_ref)] = z_red
            z_dense[ref, non_ref] = z_dense[non_ref, ref] = np.diag(z_red)

    return node_ids, y_bus, components, distances, y_dense, z_dense


def compute_electrical_distances_from_network(
    network,
    *,
    mode: DistanceMode = _MODE_VOLTAGE_LEVEL,
    store_matrices: bool = False,
    materialize_distances: bool = True,
    source_iidm_path: str = "",
) -> ElectricDistanceResult:
    """Compute distances from an already-loaded pypowsybl network."""
    mode = _validate_distance_mode(mode)
    vl1, vl2, r, x, y_shunt_1, y_shunt_2, node_ids = _branch_arrays_for_mode(network, mode)
    voltage_level_ids, _y_bus, _components, distances, y_dense, z_dense = _compute_from_branch_arrays(
        vl1,
        vl2,
        r,
        x,
        y_shunt_1,
        y_shunt_2,
        node_ids,
        store_matrices=store_matrices,
        materialize_distances=materialize_distances,
        distance_columns=_distance_columns_for_mode(mode),
    )

    return ElectricDistanceResult(
        voltage_level_ids=voltage_level_ids,
        distances=distances if distances is not None else pd.DataFrame(columns=list(_distance_columns_for_mode(mode))),
        source_iidm_path=source_iidm_path,
        y_bus=y_dense,
        z_bus=z_dense,
    )


def compute_electrical_distances_from_iidm(
    path: Path,
    *,
    mode: DistanceMode = _MODE_VOLTAGE_LEVEL,
    store_matrices: bool = False,
) -> ElectricDistanceResult:
    """Run all three steps: Y-bus, Z = Y^{-1}, electrical distances."""
    iidm_path = _resolve_iidm_path(Path(path))
    network = pp.network.load(str(iidm_path))
    return compute_electrical_distances_from_network(
        network,
        mode=mode,
        store_matrices=store_matrices,
        source_iidm_path=str(iidm_path),
    )


def distances_to_dataframe(distances: DistanceTable) -> pd.DataFrame:
    if isinstance(distances, pd.DataFrame):
        return distances
    return pd.DataFrame(distances, columns=["VLi", "VLj", "dij"])


def write_electrical_distance_csv(distances: DistanceTable, output_csv: Path) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = distances_to_dataframe(distances)
    if df.empty:
        df.to_csv(output_csv, index=False)
        return

    chunksize = 500_000
    if len(df) <= chunksize:
        df.to_csv(output_csv, index=False)
        return

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        columns = list(df.columns[:3]) if len(df.columns) >= 3 else ["VLi", "VLj", "dij"]
        writer.writerow(columns)
        for start in range(0, len(df), chunksize):
            chunk = df.iloc[start : start + chunksize]
            writer.writerows(zip(chunk[columns[0]], chunk[columns[1]], chunk[columns[2]]))


def write_electrical_distance_csv_from_network(
    network,
    output_csv: Path,
    *,
    mode: DistanceMode = _MODE_VOLTAGE_LEVEL,
    source_iidm_path: str = "",
) -> int:
    """Compute distances from a loaded network and stream CSV (fast path for batch jobs).

    Unlike :func:`compute_electrical_distances_from_iidm`, this never builds a full
    ``DataFrame`` of all (VLi, VLj) pairs in memory.
    """
    del source_iidm_path
    mode = _validate_distance_mode(mode)
    vl1, vl2, r, x, y_shunt_1, y_shunt_2, node_ids = _branch_arrays_for_mode(network, mode)
    node_ids = sorted(node_ids)
    columns = _distance_columns_for_mode(mode)
    if not node_ids:
        write_electrical_distance_csv(pd.DataFrame(columns=list(columns)), output_csv)
        return 0

    if vl1.size == 0:
        branch_i = np.empty(0, dtype=np.intp)
        branch_j = np.empty(0, dtype=np.intp)
        branch_y = np.empty(0, dtype=np.complex128)
    else:
        branch_i, branch_j = _branch_index_arrays(vl1, vl2, node_ids)
        branch_y = branch_admittances(r, x)
    y_bus = build_y_bus_sparse(len(node_ids), branch_i, branch_j, branch_y, y_shunt_1, y_shunt_2)
    components = _connected_components_from_sparse_y(y_bus)
    return stream_electrical_distance_csv(y_bus, components, node_ids, output_csv, columns=columns)


def write_electrical_distance_csv_from_iidm(
    iidm_path: Path,
    output_csv: Path,
    *,
    mode: DistanceMode = _MODE_VOLTAGE_LEVEL,
    store_matrices: bool = False,
) -> int:
    """Load IIDM, compute distances with sparse LU, stream CSV; return row count.

    Preferred entry point for ``create_op_electric_distance.py``: same numerical
    results as :func:`compute_electrical_distances_from_iidm`, but writes rows
    incrementally instead of materializing millions of rows in a ``DataFrame`` first.

    ``store_matrices`` is accepted for API compatibility but ignored (matrices are
    not written to CSV).
    """
    del store_matrices
    iidm_path = _resolve_iidm_path(Path(iidm_path))
    network = pp.network.load(str(iidm_path))
    return write_electrical_distance_csv_from_network(
        network,
        output_csv,
        mode=mode,
        source_iidm_path=str(iidm_path),
    )


def write_substation_electrical_distance_csv_from_iidm(
    iidm_path: Path,
    output_csv: Path,
    *,
    store_matrices: bool = False,
) -> int:
    """Convenience wrapper for substation-level electrical distances."""
    return write_electrical_distance_csv_from_iidm(
        iidm_path,
        output_csv,
        mode=_MODE_SUBSTATION,
        store_matrices=store_matrices,
    )
