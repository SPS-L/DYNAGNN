# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: Nordic example config writer

from __future__ import annotations

import argparse
from pathlib import Path


YAML_TEMPLATE = """# Configuration for DYNAGNN scripts.
#
# Nordic example defaults (examples/Nordic/data).
# Before running main.py, set dynawo.path and data.path to absolute paths on your machine.

dynagnn:
  version: 1.2

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
      cuts: [1e-06, 1.4999999999999999e-05, 5.9999999999999995e-05, 0.000225] # fair log-ish spacing; rebuild Dataset_*.csv to train on these
    spower:
      cuts: [1e-06, 2.9999999999999997e-05, 0.00011999999999999999, 0.00045]

model:
  num_classes: 6

training:
  epochs: 100
  patience: 12
  batch_size: 16
  split_mode: operating_point
  seed: 42
  training: 0.7142857143
  validation: 0.1428571429
  testing: 0.1428571429 

  # Fixed loss construction and output-decoding settings.
  pair_aware:
    classification_weight: 1.0
    regression_weight: 0.30
    inactive_gate_weight: 0.2
    ordinal_weight: 0.1
    class_weight_mode: inverse
    gate_pos_weight_mode: balanced
    gate_threshold: 0.50
    epsilon: 1.0e-10
    selection_output: class

optuna:
  n_trials: 15
  study_name: nordic
  hparams:
    hidden_dim:
      type: categorical
      choices: [128, 256]
    node_id_dim:
      type: categorical
      choices: [16, 24, 32]
    contingency_id_dim:
      type: categorical
      choices: [16, 32, 64]
    type_dim:
      type: categorical
      choices: [8, 16]
    pair_dim:
      type: categorical
      choices: [8, 16, 32]
    num_gnn_layers:
      type: int
      low: 3
      high: 5
    decoder_hidden_dim:
      type: categorical
      choices: [128, 256, 512]
    dropout:
      type: float
      low: 0.02
      high: 0.25
    lr:
      type: float
      low: 0.00001
      high: 0.0015
      log: true
    weight_decay:
      type: float
      low: 0.0000001
      high: 0.001
      log: true

inference:
  initialization_duration: 10.0  # steady-state run before graph build; use 0 to skip
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Nordic_test_setup.py",
        description=(
            "Write a ready-to-run Nordic configuration into "
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
