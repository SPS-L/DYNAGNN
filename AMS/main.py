#!/usr/bin/env python3
"""CLI entry point for AMS model reduction (TwinEU DSL + node-breaker simplification)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from modules.base_graph_construction import build_base_graph
from modules.DSL_reader import read_dsl
from modules.electric_distance import build_electric_distance_table
from modules.event_graph_construction import build_event_graphs
from modules.pair_aware_models import PairAwareModels, aggregate_max_substation_predictions
from modules.node_breaker_simplification import apply_node_breaker_simplification

AMS_DIR = Path(__file__).resolve().parent
MODELS_DIR = AMS_DIR / "models"
DEFAULT_EPSILON = 1.0


def resolve_models_dir(network: str, *, models_root: Path | None = None) -> Path:
    """Return ``AMS/models/<network>/``."""
    name = str(network).strip()
    if not name:
        raise ValueError("network name must be non-empty")
    root = MODELS_DIR if models_root is None else Path(models_root)
    return root / name


def run(
    dsl_path: str | Path,
    iidm_path: str | Path,
    dyd_path: str | Path,
    *,
    network: str,
    models_dir: str | Path | None = None,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[list[str], list[str], dict[str, int]]:
    """Run AMS model reduction: predict activity, then simplify node-breaker switches."""
    resolved_models_dir = (
        Path(models_dir) if models_dir is not None else resolve_models_dir(network)
    )
    if not resolved_models_dir.is_dir():
        raise FileNotFoundError(
            f"Models directory not found: {resolved_models_dir}. "
            f"Create AMS/models/{network.strip()}/ and copy DYNAGNN deployment checkpoints there."
        )

    action_locations, events_list = read_dsl(dsl_path, iidm_path)
    graph_data, graph_metadata = build_base_graph(iidm_path, dyd_path)
    electric_distance_table = build_electric_distance_table(iidm_path)

    event_graphs = build_event_graphs(
        graph_data,
        graph_metadata,
        events_list,
        electric_distance_table,
    )

    pair_aware_models = PairAwareModels(resolved_models_dir)
    if event_graphs:
        pair_aware_models.initialize(event_graphs[0])

    substation_predictions_per_graph: list[dict[str, int]] = []
    for graph in event_graphs:
        result = pair_aware_models.predict(graph)
        substation_predictions_per_graph.append(result.substation_predictions)

    final_substation_predictions = aggregate_max_substation_predictions(
        substation_predictions_per_graph
    )

    apply_node_breaker_simplification(
        iidm_path,
        epsilon,
        action_locations,
        final_substation_predictions,
    )
    return action_locations, events_list, final_substation_predictions


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "AMS model reduction: parse a TwinEU DSL scenario, run pair-aware GINE "
            "predictions, and simplify node-breaker switch retention in the IIDM."
        ),
    )
    parser.add_argument("dsl_path", type=Path, help="Path to the .dsl scenario file")
    parser.add_argument("iidm_path", type=Path, help="Path to the IIDM network file")
    parser.add_argument("dyd_path", type=Path, help="Path to the DYD dynamic models file")
    parser.add_argument(
        "--network",
        "-n",
        required=True,
        metavar="NAME",
        help="Network name; loads checkpoints from AMS/models/<NAME>/ (e.g. Nordic)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help="Retain switches in substations with prediction >= epsilon (default: %(default)s)",
    )
    parser.add_argument(
        "--json",
        nargs="?",
        const="",
        metavar="PATH",
        help="Save DSL location lists to JSON under AMS/ (default: <dsl_stem>.json)",
    )
    args = parser.parse_args()

    if not args.dsl_path.is_file():
        print(f"DSL file not found: {args.dsl_path}", file=sys.stderr)
        return 1
    if not args.iidm_path.is_file():
        print(f"IIDM file not found: {args.iidm_path}", file=sys.stderr)
        return 1
    if not args.dyd_path.is_file():
        print(f"DYD file not found: {args.dyd_path}", file=sys.stderr)
        return 1

    try:
        action_locations, events_list, final_substation_predictions = run(
            args.dsl_path,
            args.iidm_path,
            args.dyd_path,
            network=args.network,
            epsilon=args.epsilon,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.json is not None:
        if args.json:
            out_path = Path(args.json)
            if not out_path.is_absolute():
                out_path = AMS_DIR / out_path
        else:
            out_path = AMS_DIR / f"{args.dsl_path.stem}.json"
        payload = {
            "action_locations": action_locations,
            "events_list": events_list,
        }
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
