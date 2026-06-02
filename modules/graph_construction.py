# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: PyG graph construction from IIDM

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch_geometric.data import Data

try:
    from .dyd_mapping import build_generator_static_to_dynamic_map
except ImportError:  # pragma: no cover - allows running this module as a script
    from dyd_mapping import build_generator_static_to_dynamic_map

try:
    import pypowsybl as pp
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit("Missing dependency: 'pypowsybl'.") from e


TYPE_BUS = 0
TYPE_GENERATOR = 1
TYPE_LOAD = 2

EDGE_LINE = 0
EDGE_TRANSFORMER = 1
EDGE_CONNECTION = 2

EDGE_TYPE_NAMES = {
    EDGE_LINE: "line",
    EDGE_TRANSFORMER: "transformer",
    EDGE_CONNECTION: "connection",
}


def _parse_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        return default if math.isnan(out) else out
    except (TypeError, ValueError):
        return default


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _node_feature(node_type: int, *, v=0.0, angle=0.0, p=0.0, q=0.0, fault_on=False) -> list:
    return [
        float(node_type),
        float(v),
        float(angle),
        float(p),
        float(q),
        1.0 if fault_on else 0.0,
    ]


def _edge_feature(
    edge_type: int,
    *,
    r=0.0,
    x=0.0,
    b1=0.0,
    g1=0.0,
    b2=0.0,
    g2=0.0,
    fault_on=False,
) -> list:
    return [
        float(edge_type),
        1.0 if fault_on else 0.0,
        float(r),
        float(x),
        float(b1),
        float(g1),
        float(b2),
        float(g2),
    ]


def _resolve_iidm_path(path: Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path

    candidates = [
        path / "outputIIDM.xml",
        *sorted(path.glob("*.iidm")),
        *sorted(path.glob("*.xiidm")),
        *sorted(path.glob("*.xml")),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find an IIDM/XML file in {path}")


def _resolve_dyd_path(path: Path, dyd_path: Optional[Path]) -> Optional[Path]:
    if dyd_path is not None:
        dyd_path = Path(dyd_path)
        return dyd_path if dyd_path.exists() else None

    path = Path(path)
    search_dir = path if path.is_dir() else path.parent
    dyds = sorted(search_dir.glob("*.dyd"))
    return dyds[0] if dyds else None


def _busbar_sections_by_voltage_level(busbar_sections) -> Dict[str, list]:
    """Return busbar-section ids grouped by voltage level.

    This is only a metadata mapping. Voltage-level node inclusion is decided
    from pypowsybl buses and branch/equipment connectivity.
    """
    if busbar_sections.empty:
        return {}

    connected = busbar_sections[busbar_sections["connected"].map(_parse_bool)]
    out: Dict[str, list] = {}
    for voltage_level_id, group in connected.groupby("voltage_level_id", sort=True):
        out[str(voltage_level_id)] = sorted(group.index.astype(str).tolist())
    return out


def _substation_country(substations, substation_id: str):
    if substation_id and substation_id in substations.index:
        value = substations.loc[substation_id].get("country")
        return None if value is None or str(value) == "nan" else str(value)
    return None


def _voltage_level_substation(voltage_levels, voltage_level_id: str):
    if voltage_level_id and voltage_level_id in voltage_levels.index:
        value = voltage_levels.loc[voltage_level_id].get("substation_id")
        return None if value is None or str(value) == "nan" else str(value)
    return None


def _iter_hvdc_voltage_level_links(
    network,
    kept_bus_ids: set,
    active_voltage_levels: set,
):
    """Yield (hvdc_id, voltage_level1_id, voltage_level2_id, r) for active HVDC links.

    An HVDC line connects two voltage levels via VSC converter stations. Both
    ``connected1``/``connected2`` on the line and ``connected`` on each converter
    must hold; converter buses must be in the active topology.
    """
    hvdc_lines = network.get_hvdc_lines()
    vsc_stations = network.get_vsc_converter_stations()
    if hvdc_lines.empty or vsc_stations.empty:
        return

    vsc_by_id = {str(idx): row for idx, row in vsc_stations.iterrows()}

    def _endpoint_voltage_level(converter_station_id: str) -> Optional[str]:
        row = vsc_by_id.get(converter_station_id)
        if row is None:
            return None
        if not _parse_bool(row.get("connected")):
            return None
        bus_id = str(row.get("bus_id", ""))
        if bus_id not in kept_bus_ids:
            return None
        voltage_level_id = str(row.get("voltage_level_id", ""))
        if voltage_level_id not in active_voltage_levels:
            return None
        return voltage_level_id

    connected_hvdc = hvdc_lines[
        hvdc_lines["connected1"].map(_parse_bool)
        & hvdc_lines["connected2"].map(_parse_bool)
    ]
    for hvdc_id, row in connected_hvdc.iterrows():
        voltage_level1 = _endpoint_voltage_level(str(row["converter_station1_id"]))
        voltage_level2 = _endpoint_voltage_level(str(row["converter_station2_id"]))
        if voltage_level1 is None or voltage_level2 is None or voltage_level1 == voltage_level2:
            continue
        yield str(hvdc_id), voltage_level1, voltage_level2, _parse_float(row.get("r"))


def _add_undirected_edge(
    edge_index: list,
    edge_attr: list,
    edge_metadata: list,
    src_idx: int,
    dst_idx: int,
    edge_type: int,
    edge_id: str,
    *,
    bus1: str,
    bus2: str,
    r=0.0,
    x=0.0,
    b1=0.0,
    g1=0.0,
    b2=0.0,
    g2=0.0,
) -> None:
    r = _parse_float(r)
    x = _parse_float(x)
    b1 = _parse_float(b1)
    g1 = _parse_float(g1)
    b2 = _parse_float(b2)
    g2 = _parse_float(g2)

    def _one(source, target, source_bus, target_bus, source_b, source_g, target_b, target_g):
        edge_index.append([source, target])
        edge_attr.append(
            _edge_feature(
                edge_type,
                r=r,
                x=x,
                b1=source_b,
                g1=source_g,
                b2=target_b,
                g2=target_g,
            )
        )
        edge_metadata.append(
            {
                "source": source,
                "target": target,
                "type": EDGE_TYPE_NAMES[edge_type],
                "id": edge_id,
                "bus1": source_bus,
                "bus2": target_bus,
                "r": r,
                "x": x,
                "b1": source_b,
                "g1": source_g,
                "b2": target_b,
                "g2": target_g,
            }
        )

    _one(src_idx, dst_idx, bus1, bus2, b1, g1, b2, g2)
    if src_idx != dst_idx:
        _one(dst_idx, src_idx, bus2, bus1, b2, g2, b1, g1)


def _compact_graph(node_features, node_metadata, edge_index, edge_attr, edge_metadata):
    nodes_with_edges = set()
    for src, dst in edge_index:
        nodes_with_edges.add(src)
        nodes_with_edges.add(dst)

    if not nodes_with_edges:
        return [], {}, [], [], []

    keep_nodes = sorted(nodes_with_edges)
    index_map = {old: new for new, old in enumerate(keep_nodes)}
    compact_features = [node_features[i] for i in keep_nodes]

    compact_metadata = {}
    for node_id, meta in node_metadata.items():
        old_idx = meta["index"]
        if old_idx not in index_map:
            continue
        new_meta = dict(meta)
        new_meta["index"] = index_map[old_idx]
        compact_metadata[node_id] = new_meta

    compact_edge_index = []
    compact_edge_attr = []
    compact_edge_metadata = []
    for (src, dst), attr, meta in zip(edge_index, edge_attr, edge_metadata):
        if src not in index_map or dst not in index_map:
            continue
        compact_edge_index.append([index_map[src], index_map[dst]])
        compact_edge_attr.append(attr)
        new_meta = dict(meta)
        new_meta["source"] = index_map[src]
        new_meta["target"] = index_map[dst]
        compact_edge_metadata.append(new_meta)

    return compact_features, compact_metadata, compact_edge_index, compact_edge_attr, compact_edge_metadata


def _has_symmetric_edges(edge_index_tensor: torch.Tensor) -> bool:
    if edge_index_tensor.numel() == 0:
        return True
    edges = {
        (int(edge_index_tensor[0, i]), int(edge_index_tensor[1, i]))
        for i in range(edge_index_tensor.shape[1])
    }
    return all(src == dst or (dst, src) in edges for src, dst in edges)


def build_graph(
    path: Path,
    *,
    dyd_path: Optional[Path] = None,
    compact: bool = True,
) -> Tuple[Data, dict]:
    """Build a PyTorch Geometric graph from an IIDM file/directory using pypowsybl.

    Construction rules:
    - pypowsybl buses are used to identify active voltage levels
      (``connected_component == 0`` and ``v_mag != 0``);
    - one graph ``bus`` node is created per active voltage level;
    - ``network.get_busbar_sections()`` maps connected busbar-section ids to
      the voltage-level node metadata;
    - generators/loads come from pypowsybl tables and are kept only when
      ``connected == True`` and their voltage level is active;
    - lines and transformers are kept only when both sides are connected and
      both side voltage levels are active;
    - HVDC lines link voltage levels via ``get_hvdc_lines()`` and
      ``get_vsc_converter_stations()`` as line edges with ``x=0``;
    - nodes without any edge are removed by the final compaction step.
    """
    input_path = Path(path)
    iidm_path = _resolve_iidm_path(input_path)
    dyd_path = _resolve_dyd_path(input_path, dyd_path)
    dynamic_generator_by_static_id = build_generator_static_to_dynamic_map(dyd_path)

    network = pp.network.load(str(iidm_path))

    buses = network.get_buses()
    voltage_levels = network.get_voltage_levels()
    busbar_sections = network.get_busbar_sections()
    substations = network.get_substations()
    generators = network.get_generators()
    loads = network.get_loads()
    lines = network.get_lines()
    transformers_2w = network.get_2_windings_transformers()
    transformers_3w = network.get_3_windings_transformers()

    busbar_ids_by_voltage_level = _busbar_sections_by_voltage_level(busbar_sections)

    kept_powsybl_buses = buses[
        (buses["connected_component"] == 0)
        & (buses["v_mag"].fillna(0.0) != 0.0)
    ]
    kept_bus_ids = set(kept_powsybl_buses.index.astype(str))

    active_voltage_levels = set(kept_powsybl_buses["voltage_level_id"].dropna().astype(str))
    voltage_level_values = {}
    for voltage_level_id, group in kept_powsybl_buses.groupby("voltage_level_id", sort=True):
        voltage_level_values[str(voltage_level_id)] = {
            "v": float(group["v_mag"].mean()),
            "angle": float(group["v_angle"].mean()),
        }

    node_features = []
    node_metadata = {}
    voltage_level_node_index = {}

    for voltage_level_id in sorted(active_voltage_levels):
        substation_id = _voltage_level_substation(voltage_levels, voltage_level_id)
        values = voltage_level_values.get(voltage_level_id, {"v": 0.0, "angle": 0.0})
        idx = len(node_features)
        node_features.append(
            _node_feature(
                TYPE_BUS,
                v=_parse_float(values["v"]),
                angle=_parse_float(values["angle"]),
            )
        )
        node_metadata[voltage_level_id] = {
            "index": idx,
            "type": "bus",
            "id": voltage_level_id,
            "voltageLevelId": voltage_level_id,
            "substationId": substation_id,
            "country": _substation_country(substations, substation_id),
            "busbarSectionIds": list(busbar_ids_by_voltage_level.get(voltage_level_id, [])),
        }
        voltage_level_node_index[voltage_level_id] = idx

    edge_index = []
    edge_attr = []
    edge_metadata = []

    connected_lines = lines[
        lines["connected1"].map(_parse_bool)
        & lines["connected2"].map(_parse_bool)
    ]
    for line_id, row in connected_lines.iterrows():
        bus1 = str(row["bus1_id"])
        bus2 = str(row["bus2_id"])
        if bus1 not in kept_bus_ids or bus2 not in kept_bus_ids:
            continue
        voltage_level1 = str(row["voltage_level1_id"])
        voltage_level2 = str(row["voltage_level2_id"])
        if voltage_level1 not in active_voltage_levels or voltage_level2 not in active_voltage_levels:
            continue
        _add_undirected_edge(
            edge_index,
            edge_attr,
            edge_metadata,
            voltage_level_node_index[voltage_level1],
            voltage_level_node_index[voltage_level2],
            EDGE_LINE,
            str(line_id),
            bus1=voltage_level1,
            bus2=voltage_level2,
            r=row.get("r"),
            x=row.get("x"),
            b1=row.get("b1"),
            g1=row.get("g1"),
            b2=row.get("b2"),
            g2=row.get("g2"),
        )

    connected_transformers_2w = transformers_2w[
        transformers_2w["connected1"].map(_parse_bool)
        & transformers_2w["connected2"].map(_parse_bool)
    ]
    for transformer_id, row in connected_transformers_2w.iterrows():
        bus1 = str(row["bus1_id"])
        bus2 = str(row["bus2_id"])
        if bus1 not in kept_bus_ids or bus2 not in kept_bus_ids:
            continue
        voltage_level1 = str(row["voltage_level1_id"])
        voltage_level2 = str(row["voltage_level2_id"])
        if voltage_level1 not in active_voltage_levels or voltage_level2 not in active_voltage_levels:
            continue
        _add_undirected_edge(
            edge_index,
            edge_attr,
            edge_metadata,
            voltage_level_node_index[voltage_level1],
            voltage_level_node_index[voltage_level2],
            EDGE_TRANSFORMER,
            str(transformer_id),
            bus1=voltage_level1,
            bus2=voltage_level2,
            r=row.get("r"),
            x=row.get("x"),
        )

    for transformer_id, row in transformers_3w.iterrows():
        sides = []
        for side in (1, 2, 3):
            if not _parse_bool(row.get(f"connected{side}")):
                continue
            bus_id = str(row.get(f"bus{side}_id", ""))
            if bus_id not in kept_bus_ids:
                continue
            voltage_level_id = str(row.get(f"voltage_level{side}_id", ""))
            if voltage_level_id not in active_voltage_levels:
                continue
            sides.append(
                {
                    "side": side,
                    "bus_id": bus_id,
                    "voltage_level_id": voltage_level_id,
                    "r": row.get(f"r{side}"),
                    "x": row.get(f"x{side}"),
                }
            )
        for i in range(len(sides)):
            for j in range(i + 1, len(sides)):
                side_i = sides[i]
                side_j = sides[j]
                _add_undirected_edge(
                    edge_index,
                    edge_attr,
                    edge_metadata,
                    voltage_level_node_index[side_i["voltage_level_id"]],
                    voltage_level_node_index[side_j["voltage_level_id"]],
                    EDGE_TRANSFORMER,
                    f"{transformer_id}_side{side_i['side']}_side{side_j['side']}",
                    bus1=side_i["voltage_level_id"],
                    bus2=side_j["voltage_level_id"],
                    r=_parse_float(side_i["r"]) + _parse_float(side_j["r"]),
                    x=_parse_float(side_i["x"]) + _parse_float(side_j["x"]),
                )

    for hvdc_id, voltage_level1, voltage_level2, hvdc_r in _iter_hvdc_voltage_level_links(
        network, kept_bus_ids, active_voltage_levels
    ):
        _add_undirected_edge(
            edge_index,
            edge_attr,
            edge_metadata,
            voltage_level_node_index[voltage_level1],
            voltage_level_node_index[voltage_level2],
            EDGE_LINE,
            hvdc_id,
            bus1=voltage_level1,
            bus2=voltage_level2,
            r=hvdc_r,
            x=0.0,
        )

    def _add_equipment_nodes(table, node_type: int, type_name: str, prefix: str) -> None:
        connected_table = table[table["connected"].map(_parse_bool)]
        if type_name == "load" and "type" in connected_table.columns:
            connected_table = connected_table[connected_table["type"].astype(str).str.upper() != "FICTITIOUS"]

        for equipment_id, row in connected_table.sort_index().iterrows():
            equipment_id = str(equipment_id)
            bus_id = str(row["bus_id"])
            if bus_id not in kept_bus_ids:
                continue
            voltage_level_id = str(row["voltage_level_id"])
            if voltage_level_id not in active_voltage_levels:
                continue
            substation_id = _voltage_level_substation(voltage_levels, voltage_level_id)
            values = voltage_level_values.get(voltage_level_id, {"v": 0.0, "angle": 0.0})
            idx = len(node_features)
            node_features.append(
                _node_feature(
                    node_type,
                    v=_parse_float(values["v"]),
                    angle=_parse_float(values["angle"]),
                    p=_parse_float(row.get("p")),
                    q=_parse_float(row.get("q")),
                )
            )
            metadata = {
                "index": idx,
                "type": type_name,
                "id": equipment_id,
                "voltageLevelId": voltage_level_id,
                "substationId": substation_id,
                "country": _substation_country(substations, substation_id),
            }
            if type_name == "generator":
                dynamic_model = dynamic_generator_by_static_id.get(equipment_id)
                metadata["hasDynamicModel"] = dynamic_model is not None
                metadata["dynamicModelId"] = dynamic_model[0] if dynamic_model else None
                metadata["dynamicModelLib"] = dynamic_model[1] if dynamic_model else None
            node_metadata[equipment_id] = metadata
            _add_undirected_edge(
                edge_index,
                edge_attr,
                edge_metadata,
                idx,
                voltage_level_node_index[voltage_level_id],
                EDGE_CONNECTION,
                f"{prefix}_{equipment_id}_to_{voltage_level_id}",
                bus1=equipment_id,
                bus2=voltage_level_id,
            )

    _add_equipment_nodes(generators, TYPE_GENERATOR, "generator", "gen")
    _add_equipment_nodes(loads, TYPE_LOAD, "load", "load")

    if compact:
        node_features, node_metadata, edge_index, edge_attr, edge_metadata = _compact_graph(
            node_features,
            node_metadata,
            edge_index,
            edge_attr,
            edge_metadata,
        )

    x = torch.tensor(node_features, dtype=torch.float) if node_features else torch.empty((0, 6), dtype=torch.float)
    if edge_index:
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
        edge_attr_tensor = torch.empty((0, 8), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor)
    metadata = {
        "node_metadata": node_metadata,
        "edge_metadata": edge_metadata,
        "node_feature_schema": ["node_type", "v", "angle", "p", "q", "fault_on"],
        "edge_feature_schema": ["edge_type", "fault_on", "r", "x", "b1", "g1", "b2", "g2"],
        "is_undirected": _has_symmetric_edges(edge_index_tensor),
        "source_iidm_path": str(iidm_path),
        "source_dyd_path": str(dyd_path) if dyd_path is not None else None,
        "dynamic_model_metadata_available": dyd_path is not None,
        "construction_backend": "pypowsybl",
    }
    data.metadata = metadata
    data.node_metadata = node_metadata
    data.edge_metadata = edge_metadata
    return data, metadata


def build_pyg_graph(path: Path, *, dyd_path: Optional[Path] = None, compact: bool = True) -> Data:
    data, _ = build_graph(path, dyd_path=dyd_path, compact=compact)
    return data


def build_graph_from_iidm(path: Path, *, dyd_path: Optional[Path] = None, compact: bool = True) -> Tuple[Data, dict]:
    return build_graph(path, dyd_path=dyd_path, compact=compact)


def build_pyg_graph_from_iidm(path: Path, *, dyd_path: Optional[Path] = None, compact: bool = True) -> Data:
    return build_pyg_graph(path, dyd_path=dyd_path, compact=compact)


def build_graph_from_directory(directory: Path, *, dyd_path: Optional[Path] = None, compact: bool = True) -> Tuple[Data, dict]:
    return build_graph(directory, dyd_path=dyd_path, compact=compact)


def build_pyg_graph_from_directory(directory: Path, *, dyd_path: Optional[Path] = None, compact: bool = True) -> Data:
    return build_pyg_graph(directory, dyd_path=dyd_path, compact=compact)


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build a PyG graph from an IIDM file using pypowsybl tables.")
    parser.add_argument("input", type=Path, help="IIDM/XML file or directory containing outputIIDM.xml / *.iidm")
    parser.add_argument("--dyd", type=Path, default=None, help="Optional DYD file for synchronous-generator metadata")
    parser.add_argument("--output", type=Path, default=None, help="Optional .pt output path")
    args = parser.parse_args(list(argv) if argv is not None else None)

    data, metadata = build_graph(args.input, dyd_path=args.dyd)
    print(f"Built graph: {data.x.shape[0]} nodes, {data.edge_index.shape[1]} directed edges")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"data": data, "metadata": metadata}, args.output)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
