#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


YAML_TEMPLATE = """# Configuration for DYNAGNN scripts.
#
# Quick smoke test defaults for the bundled Nordic example (examples/Nordic/data).
# Before running main.py, set dynawo.path and data.path to absolute paths on your machine.

dynagnn:
  version: 1

dynawo:
  path: "{dynawo_env_sh}"

data:
  path: "{nordic_data_path}"

simulation:
  event_time: 10.0
  initialization_duration: 10.0  # steady-state init per OP before contingencies; use 0 to skip

network:
  country_filter: []

kpi:
  window_sec: 5.0
  step_sec: 1.0
  class_bins:
    voltage:
      cuts: [0.33, 0.66]  # 3 severity bins on [0, 1] + 1 flag class => model.num_classes: 4
    spower:
      cuts: [0.33, 0.66]

model:
  num_classes: 4

training:
  epochs: 30          # keep low for a quick smoke test
  patience: 8
  batch_size: 16
  split_mode: operating_point
  seed: 42
  training: 0.8
  validation: 0.1
  testing: 0.1
  high_class_threshold: null

optuna:
  n_trials: 5
  hparams:
    hidden_dim:
      type: categorical
      choices: [64, 128, 256]
    num_layers:
      type: int
      low: 2
      high: 4
    hidden_channels:
      type: categorical
      choices: [16, 32, 64]
    num_heads:
      type: categorical
      choices: [1, 2, 4, 8]
    dropout:
      type: float
      low: 0.1
      high: 0.5
    num_gnn_layers:
      type: int
      low: 2
      high: 4
    lr:
      type: float
      low: 1.0e-4
      high: 5.0e-3
      log: true
    weight_decay:
      type: float
      low: 1.0e-6
      high: 1.0e-3
      log: true
    under_penalty_lambda:
      type: float
      low: 0.0
      high: 2.0
    coral_prediction_threshold:
      type: float
      low: 0.3
      high: 0.7

inference:
  initialization_duration: 10.0  # steady-state run before graph build; use 0 to skip
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Nordic_test_setup.py",
        description=(
            "Write a ready-to-run Nordic smoke-test configuration into "
            "the project-root config.yaml."
        ),
    )
    parser.add_argument(
        "--dynawo-env",
        required=True,
        help="Absolute path to myEnvDynawo.sh (including the .sh filename).",
    )
    parser.add_argument(
        "--dynagnn-root",
        default=None,
        help=(
            "Absolute path to the DYNAGNN repository root. "
            "Defaults to the directory containing this script."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to config.yaml to write. Defaults to <dynagnn-root>/config.yaml."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing config.yaml without prompting.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    dynawo_env_sh = Path(args.dynawo_env).expanduser().resolve()
    if not dynawo_env_sh.is_file():
        raise SystemExit(f"--dynawo-env does not exist or is not a file: {dynawo_env_sh}")
    if dynawo_env_sh.suffix != ".sh":
        raise SystemExit(f"--dynawo-env must point to a .sh file: {dynawo_env_sh}")

    dynagnn_root = (
        Path(args.dynagnn_root).expanduser().resolve()
        if args.dynagnn_root
        else Path(__file__).resolve().parent
    )
    if not dynagnn_root.is_dir():
        raise SystemExit(f"--dynagnn-root is not a directory: {dynagnn_root}")

    nordic_data_path = (dynagnn_root / "examples" / "Nordic" / "data").resolve()
    if not nordic_data_path.is_dir():
        raise SystemExit(
            "Could not find Nordic example data folder at: "
            f"{nordic_data_path} (expected <dynagnn-root>/examples/Nordic/data)"
        )

    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else (dynagnn_root / "config.yaml").resolve()
    )

    if config_path.exists() and not args.force:
        raise SystemExit(
            f"Refusing to overwrite existing {config_path}. Re-run with --force."
        )

    text = YAML_TEMPLATE.format(
        dynawo_env_sh=str(dynawo_env_sh),
        nordic_data_path=str(nordic_data_path),
    )
    config_path.write_text(text, encoding="utf-8")

    print(f"Wrote {config_path}")
    print(f"  dynawo.path = {dynawo_env_sh}")
    print(f"  data.path   = {nordic_data_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

