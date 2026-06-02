# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: Operating-point graph, electrical distance, and SNom assets

from __future__ import annotations

import argparse
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

    parser = argparse.ArgumentParser(
        description="Build PyG graphs, electrical-distance CSVs, and generator SNom tables per operating point."
    )
    parser.add_argument("--inputs", type=Path, default=INPUTS_DIR, help="Directory containing operating_point_N folders.")
    parser.add_argument("--graph-output", type=Path, default=OP_GRAPHS_DIR, help="Directory for .pt graph bundles.")
    parser.add_argument(
        "--electric-output",
        type=Path,
        default=OP_ELECTRIC_DISTANCE_DIR,
        help="Directory for electrical-distance CSVs.",
    )
    parser.add_argument(
        "--snom-output",
        type=Path,
        default=SNOM_DIR,
        help="Directory for per-OP generator SNom CSVs (operating_point_N.csv).",
    )
    parser.add_argument("--skip-electrical-distance", action="store_true", help="Do not write electrical-distance CSVs.")
    parser.add_argument("--skip-generator-snom", action="store_true", help="Do not write generator SNom CSVs.")
    parser.add_argument("--skip-existing-graphs", action="store_true", help="Skip graph bundles that already exist.")
    parser.add_argument(
        "--skip-existing-electric-distance",
        action="store_true",
        help="Skip electrical-distance CSVs that already exist.",
    )
    parser.add_argument(
        "--skip-existing-generator-snom",
        action="store_true",
        help="Skip generator SNom CSVs that already exist.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-graph node and edge type counts.")
    parser.add_argument(
        "--show-examples",
        action="store_true",
        help="Print one example node/edge metadata record per type.",
    )
    args = parser.parse_args()

    if not args.inputs.exists():
        raise SystemExit(f"Missing inputs directory: {args.inputs}")

    operating_points = discover_operating_points(args.inputs)
    if not operating_points:
        raise SystemExit(f"No operating point folders found in {args.inputs}")

    args.graph_output.mkdir(parents=True, exist_ok=True)
    if not args.skip_electrical_distance:
        args.electric_output.mkdir(parents=True, exist_ok=True)
    if not args.skip_generator_snom:
        args.snom_output.mkdir(parents=True, exist_ok=True)

    logger.info("Inputs: %s", args.inputs)
    logger.info("Graph output: %s", args.graph_output)
    if not args.skip_electrical_distance:
        logger.info("Electrical-distance output: %s", args.electric_output)
    if not args.skip_generator_snom:
        logger.info("Generator SNom output: %s", args.snom_output)
    logger.info("Operating points to build: %d", len(operating_points))

    distance_tasks: list[tuple[str, str]] = []
    printed_examples = False
    for index, op_dir in enumerate(operating_points, start=1):
        output_path = args.graph_output / f"{op_dir.name}.pt"
        distance_csv = args.electric_output / f"{op_dir.name}.csv"
        snom_csv = args.snom_output / f"{op_dir.name}.csv"
        dyd_path = resolve_op_dyd_path(op_dir)

        if not args.skip_generator_snom:
            if args.skip_existing_generator_snom and snom_csv.is_file():
                logger.info("%d/%d %s: generator SNom skipped (%s exists)", index, len(operating_points), op_dir.name, snom_csv.name)
            else:
                logger.info("%d/%d %s: generator SNom -> %s", index, len(operating_points), op_dir.name, snom_csv.name)
                try:
                    _, gen_count = build_generator_snom_for_operating_point(
                        op_dir,
                        output_dir=args.snom_output,
                    )
                    logger.info("  %d generators", gen_count)
                except Exception as exc:
                    logger.error("  generator SNom failed: %s", exc)

        if not args.skip_electrical_distance:
            if args.skip_existing_electric_distance and distance_csv.is_file():
                logger.info(
                    "%d/%d %s: electrical distance skipped (%s exists)",
                    index,
                    len(operating_points),
                    op_dir.name,
                    distance_csv.name,
                )
            else:
                distance_tasks.append((str(op_dir), str(distance_csv)))

        if args.skip_existing_graphs and output_path.is_file():
            logger.info("%d/%d %s: graph skipped (%s exists)", index, len(operating_points), op_dir.name, output_path.name)
            continue

        logger.info("%d/%d %s: graph -> %s", index, len(operating_points), op_dir.name, output_path.name)
        try:
            data, metadata = build_graph(op_dir, dyd_path=dyd_path)
            torch.save({"data": data, "metadata": metadata}, output_path)
            num_nodes = int(data.x.shape[0])
            num_edges = int(data.edge_index.shape[1])
            logger.info("  %d nodes, %d directed edges", num_nodes, num_edges)
            if args.show_examples and not printed_examples:
                print_examples_once(data, metadata)
                printed_examples = True
            if args.verbose:
                node_counts = {}
                for meta in metadata["node_metadata"].values():
                    node_counts[meta["type"]] = node_counts.get(meta["type"], 0) + 1
                edge_counts = {}
                for meta in metadata["edge_metadata"]:
                    edge_counts[meta["type"]] = edge_counts.get(meta["type"], 0) + 1
                logger.info("  Node counts: %s", dict(sorted(node_counts.items())))
                logger.info("  Edge counts: %s", dict(sorted(edge_counts.items())))
        except Exception as exc:
            logger.error("  graph build failed: %s", exc)

    if not args.skip_electrical_distance and distance_tasks:
        for iidm_path, output_csv in distance_tasks:
            op_name = Path(iidm_path).name
            logger.info("%s: electrical distance -> %s", op_name, Path(output_csv).name)
            _, row_count, error = process_distance_task(iidm_path, output_csv)
            if error:
                logger.error("  failed: %s", error)
            else:
                logger.info("  %d pairs", row_count)

    logger.info("Finished building operating-point assets.")
