# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: Operating-point graph, electrical distance, and SNom assets

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.curve_generation import op_sort_key
from modules.electric_distance import write_electrical_distance_csv_from_iidm
from modules.generator_snom import build_generator_snom_for_operating_point
from modules.graph_construction import build_graph
from modules.paths import INPUTS_DIR, OP_ELECTRIC_DISTANCE_DIR, OP_GRAPHS_DIR, SNOM_DIR
from modules.pipeline_logging import get_logger, log_step_banner


def discover_operating_points(inputs_dir: Path) -> list[Path]:
    return sorted(
        [path for path in inputs_dir.iterdir() if path.is_dir() and path.name.startswith("operating_point_")],
        key=op_sort_key,
    )


def resolve_op_dyd_path(op_dir: Path) -> Optional[Path]:
    local_dyds = sorted(op_dir.glob("*.dyd"))
    return local_dyds[0] if local_dyds else None


def print_examples_once(data, metadata) -> None:
    node_types = {"bus": None, "generator": None, "load": None}
    for _, meta in metadata["node_metadata"].items():
        node_type = meta["type"]
        if node_type in node_types and node_types[node_type] is None:
            idx = meta["index"]
            print(f"\nExample node type: {node_type}")
            print("  Data:", data.x[idx].tolist())
            print("  Metadata:", meta)
            node_types[node_type] = True

    edge_types = {"line": None, "transformer": None, "connection": None, "hvdc": None}
    for meta, attr in zip(metadata["edge_metadata"], data.edge_attr):
        edge_type = meta["type"]
        if edge_type in edge_types and edge_types[edge_type] is None:
            print(f"\nExample edge type: {edge_type}")
            print("  Data:", attr.tolist())
            print("  Metadata:", meta)
            edge_types[edge_type] = True


def process_distance_task(iidm_path: str, output_csv: str) -> tuple[str, int, str | None]:
    iidm = Path(iidm_path)
    op_name = iidm.name if iidm.is_dir() else iidm.parent.name
    try:
        row_count = write_electrical_distance_csv_from_iidm(
            iidm,
            Path(output_csv),
            store_matrices=False,
        )
        return op_name, row_count, None
    except Exception as exc:
        return op_name, 0, str(exc)


def main() -> None:
    log_step_banner("build_op_assets")
    logger = get_logger()

    inputs_dir = INPUTS_DIR
    graph_output = OP_GRAPHS_DIR
    electric_output = OP_ELECTRIC_DISTANCE_DIR
    snom_output = SNOM_DIR
    show_examples = True

    if not inputs_dir.exists():
        raise SystemExit(f"Missing inputs directory: {inputs_dir}")

    operating_points = discover_operating_points(inputs_dir)
    if not operating_points:
        raise SystemExit(f"No operating point folders found in {inputs_dir}")

    graph_output.mkdir(parents=True, exist_ok=True)
    electric_output.mkdir(parents=True, exist_ok=True)
    snom_output.mkdir(parents=True, exist_ok=True)

    logger.info("Inputs: %s", inputs_dir)
    logger.info("Graph output: %s", graph_output)
    logger.info("Electrical-distance output: %s", electric_output)
    logger.info("Generator SNom output: %s", snom_output)
    logger.info("Operating points to build: %d", len(operating_points))

    distance_tasks: list[tuple[str, str]] = []
    printed_examples = False
    for index, op_dir in enumerate(operating_points, start=1):
        output_path = graph_output / f"{op_dir.name}.pt"
        distance_csv = electric_output / f"{op_dir.name}.csv"
        snom_csv = snom_output / f"{op_dir.name}.csv"
        dyd_path = resolve_op_dyd_path(op_dir)

        logger.info("%d/%d %s: generator SNom -> %s", index, len(operating_points), op_dir.name, snom_csv.name)
        try:
            _, gen_count = build_generator_snom_for_operating_point(
                op_dir,
                output_dir=snom_output,
            )
            logger.info("  %d generators", gen_count)
        except Exception as exc:
            logger.error("  generator SNom failed: %s", exc)

        distance_tasks.append((str(op_dir), str(distance_csv)))

        logger.info("%d/%d %s: graph -> %s", index, len(operating_points), op_dir.name, output_path.name)
        try:
            data, metadata = build_graph(op_dir, dyd_path=dyd_path)
            torch.save({"data": data, "metadata": metadata}, output_path)
            num_nodes = int(data.x.shape[0])
            num_edges = int(data.edge_index.shape[1])
            logger.info("  %d nodes, %d directed edges", num_nodes, num_edges)
            if show_examples and not printed_examples:
                print_examples_once(data, metadata)
                printed_examples = True
        except Exception as exc:
            logger.error("  graph build failed: %s", exc)

    for iidm_path, output_csv in distance_tasks:
        op_name = Path(iidm_path).name
        logger.info("%s: electrical distance -> %s", op_name, Path(output_csv).name)
        _, row_count, error = process_distance_task(iidm_path, output_csv)
        if error:
            logger.error("  failed: %s", error)
        else:
            logger.info("  %d pairs", row_count)

    logger.info("Finished building operating-point assets.")
