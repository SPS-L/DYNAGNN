# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
"""CLI entry point for AMS model reduction (scenario DSL + node-breaker simplification)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_EPSILON = 1.0
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = PACKAGE_ROOT / "models"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _is_checkpoint_bundle(path: Path) -> bool:
    """True if ``path`` looks like a DYNAGNN deployment folder (``.pt`` + scalers)."""
    if not path.is_dir():
        return False
    checkpoint_names = (
        "voltage_best_model.pt",
        "spower_best_model.pt",
    )
    return any((path / name).is_file() for name in checkpoint_names)


def resolve_models_dir(network: str, *, models_root: Path | None = None) -> Path:
    """Return the checkpoint folder for ``network``.

    Resolution order when ``models_root`` is set:

    1. ``models_root`` itself, if it already contains deployment checkpoints.
    2. ``models_root / <network> /``, if that subfolder exists.

    When ``models_root`` is omitted, use packaged ``dynagnn_ams/models/<network>/``.
    """
    name = str(network).strip()
    if not name:
        raise ValueError("network name must be non-empty")

    if models_root is None:
        return DEFAULT_MODELS_DIR / name

    root = Path(models_root).expanduser().resolve()
    if _is_checkpoint_bundle(root):
        return root

    nested = root / name
    if nested.is_dir():
        return nested

    raise FileNotFoundError(
        f"Models directory not found for network {name!r}. "
        f"Looked for checkpoints in {root} and {nested}. "
        f"Expected layout: <models-root>/{name}/{{voltage_best_model.pt, spower_best_model.pt, "
        f"x_scaler.pkl, edge_attr_scaler.pkl}} or pass the checkpoint folder directly via "
        f"--models-dir."
    )


def run(
    dsl_path: str | Path,
    iidm_path: str | Path,
    dyd_path: str | Path,
    *,
    network: str,
    models_dir: str | Path | None = None,
    epsilon: float = DEFAULT_EPSILON,
    device: str | None = None,
) -> tuple[list[str], list[str], dict[str, int]]:
    """Run AMS model reduction: predict activity, then simplify node-breaker switches."""
    from tqdm import tqdm

    from dynagnn_ams.modules.base_graph_construction import build_base_graph
    from dynagnn_ams.modules.DSL_reader import read_dsl
    from dynagnn_ams.modules.electric_distance import build_electric_distance_table
    from dynagnn_ams.modules.event_graph_construction import build_event_graphs
    from dynagnn_ams.modules.node_breaker_simplification import (
        apply_node_breaker_simplification,
    )
    from dynagnn_ams.modules.pair_aware_models import (
        PairAwareModels,
        aggregate_max_substation_predictions,
    )

    resolved_models_dir = resolve_models_dir(
        network,
        models_root=Path(models_dir) if models_dir is not None else None,
    )
    if not resolved_models_dir.is_dir():
        raise FileNotFoundError(
            f"Models directory not found: {resolved_models_dir}. "
            f"Install dynagnn-ams with bundled models, or pass --models-dir."
        )
    _log(f"[dynagnn-ams] models={resolved_models_dir}")

    # Fixed pipeline steps (ETA is approximate: early steps often dominate).
    with tqdm(
        total=6, desc="AMS", unit="step", dynamic_ncols=True, file=sys.stderr
    ) as pbar:
        pbar.set_postfix_str("read DSL")
        action_locations, events_list = read_dsl(dsl_path, iidm_path)
        pbar.update(1)

        pbar.set_postfix_str("build graph")
        graph_data, graph_metadata = build_base_graph(iidm_path, dyd_path)
        pbar.update(1)

        pbar.set_postfix_str("electric distance")
        electric_distance_table = build_electric_distance_table(iidm_path)
        pbar.update(1)

        pbar.set_postfix_str(f"event graphs ({len(events_list)})")
        event_graphs = build_event_graphs(
            graph_data,
            graph_metadata,
            events_list,
            electric_distance_table,
        )
        pbar.update(1)

        pbar.set_postfix_str("load models")
        pair_aware_models = PairAwareModels(resolved_models_dir, device=device)
        _log(f"[dynagnn-ams] device={pair_aware_models.device}")
        if event_graphs:
            pair_aware_models.initialize(event_graphs[0])
        pbar.update(1)

        pbar.set_postfix_str(f"predict ({len(event_graphs)} events)")
        substation_predictions_per_graph: list[dict[str, int]] = []
        for graph in tqdm(
            event_graphs,
            desc="Events",
            unit="event",
            leave=False,
            dynamic_ncols=True,
            file=sys.stderr,
        ):
            result = pair_aware_models.predict(graph)
            substation_predictions_per_graph.append(result.substation_predictions)

        final_substation_predictions = aggregate_max_substation_predictions(
            substation_predictions_per_graph
        )

        pbar.set_postfix_str("patch IIDM")
        apply_node_breaker_simplification(
            iidm_path,
            epsilon,
            action_locations,
            final_substation_predictions,
        )
        pbar.update(1)

    return action_locations, events_list, final_substation_predictions


def main(argv: list[str] | None = None) -> int:
    _log("[dynagnn-ams] started")
    parser = argparse.ArgumentParser(
        prog="dynagnn-ams",
        description=(
            "AMS model reduction: parse a scenario DSL, run pair-aware GINE "
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
        help="Network name; loads checkpoints from packaged models/<NAME>/ (e.g. Nordic)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help=(
            "Override packaged checkpoints: path to models root (<network>/ subfolder) "
            "or direct checkpoint folder (voltage_best_model.pt, spower_best_model.pt, …)"
        ),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help="Retain switches where predicted class >= epsilon (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        metavar="NAME",
        help="Inference device: auto, cpu, mps, cuda, or cuda:N (default: auto)",
    )
    parser.add_argument(
        "--json",
        nargs="?",
        const="",
        metavar="PATH",
        help="Save DSL location lists to JSON (default: <cwd>/<dsl_stem>.json)",
    )
    args = parser.parse_args(argv)

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
        _log("[dynagnn-ams] loading libraries (torch / pypowsybl; can take a while)…")
        action_locations, events_list, _predictions = run(
            args.dsl_path,
            args.iidm_path,
            args.dyd_path,
            network=args.network,
            models_dir=args.models_dir,
            epsilon=args.epsilon,
            device=args.device,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    except (RuntimeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.json is not None:
        if args.json:
            out_path = Path(args.json)
            if not out_path.is_absolute():
                out_path = Path.cwd() / out_path
        else:
            out_path = Path.cwd() / f"{args.dsl_path.stem}.json"
        payload = {
            "action_locations": action_locations,
            "events_list": events_list,
        }
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    _log("[dynagnn-ams] finished")
    return 0
