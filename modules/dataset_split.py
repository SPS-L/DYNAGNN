# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: Train, validation, and test split generation

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

from .paths import CONFIG_PATH, DATASET_DIR


@dataclass(frozen=True)
class Example:
    operating_point: str
    contingency: str


@dataclass(frozen=True)
class SplitSettings:
    split_mode: str
    seed: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float


@dataclass
class SplitSummary:
    split_mode: str
    seed: int
    input_csv: Path
    output_csv: Path
    total_examples: int
    train_examples: int
    validation_examples: int
    test_examples: int


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    import yaml

    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_split_settings(
    config: Optional[Mapping[str, object]] = None,
    *,
    config_path: Path = CONFIG_PATH,
) -> SplitSettings:
    """Read split settings from ``config.yaml`` under ``training``."""
    cfg = dict(config) if config is not None else load_config(config_path)
    training = cfg.get("training", {}) or {}

    split_mode = str(training.get("split_mode", "scenario")).strip().lower()
    if split_mode not in {"scenario", "operating_point"}:
        raise ValueError(
            f"Unsupported training.split_mode '{split_mode}'. "
            "Expected 'scenario' or 'operating_point'."
        )

    train_fraction = float(training.get("training", 0.8))
    validation_fraction = float(training.get("validation", 0.1))
    test_fraction = float(training.get("testing", 0.1))

    total = train_fraction + validation_fraction + test_fraction
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            "training.training + training.validation + training.testing "
            f"must sum to 1.0 (got {total:.6f})."
        )

    return SplitSettings(
        split_mode=split_mode,
        seed=int(training.get("seed", 42)),
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )


def resolve_dataset_csv_path(dataset_csv: str | Path, *, dataset_dir: Path = DATASET_DIR) -> Path:
    """Resolve a dataset CSV name or path under ``data/Dataset``."""
    path = Path(dataset_csv)
    if path.is_absolute():
        return path
    if path.parent != Path("."):
        return path
    return dataset_dir / path.name


def default_split_output_path(input_csv: Path, *, dataset_dir: Path = DATASET_DIR) -> Path:
    return dataset_dir / "train_val_test_split.csv"


def read_examples(input_csv: Path) -> list[Example]:
    examples: list[Example] = []
    seen: set[tuple[str, str]] = set()

    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = {"OP", "Contingency"}
        missing = required.difference(fieldnames)
        if missing:
            raise ValueError(f"{input_csv} missing required columns: {sorted(missing)}")

        for row in reader:
            operating_point = str(row.get("OP", "")).strip()
            contingency = str(row.get("Contingency", "")).strip()
            if not operating_point or not contingency:
                continue

            key = (operating_point, contingency)
            if key in seen:
                continue
            seen.add(key)
            examples.append(Example(operating_point=operating_point, contingency=contingency))

    return examples


def _allocate_split_sizes(
    n: int,
    *,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[int, int, int]:
    if n == 0:
        return 0, 0, 0

    n_train = int(round(n * train_fraction))
    n_validation = int(round(n * validation_fraction)) if validation_fraction > 0 else 0
    n_test = n - n_train - n_validation

    if n_test < 0:
        n_validation = max(0, n_validation + n_test)
        n_test = 0

    assigned = n_train + n_validation + n_test
    if assigned < n:
        n_test += n - assigned
    elif assigned > n:
        overflow = assigned - n
        if n_validation >= overflow:
            n_validation -= overflow
        else:
            overflow -= n_validation
            n_validation = 0
            n_train = max(0, n_train - overflow)

    return n_train, n_validation, n_test


def _partition_items(
    items: Sequence[str],
    *,
    rng: random.Random,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[list[str], list[str], list[str]]:
    shuffled = list(items)
    rng.shuffle(shuffled)

    n_train, n_validation, n_test = _allocate_split_sizes(
        len(shuffled),
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )

    train = shuffled[:n_train]
    validation = shuffled[n_train : n_train + n_validation]
    test = shuffled[n_train + n_validation : n_train + n_validation + n_test]
    return train, validation, test


def _split_examples(
    examples: Sequence[Example],
    *,
    rng: random.Random,
    settings: SplitSettings,
) -> dict[str, list[Example]]:
    shuffled = list(examples)
    rng.shuffle(shuffled)

    n_train, n_validation, n_test = _allocate_split_sizes(
        len(shuffled),
        train_fraction=settings.train_fraction,
        validation_fraction=settings.validation_fraction,
        test_fraction=settings.test_fraction,
    )

    split = {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train : n_train + n_validation],
        "test": shuffled[n_train + n_validation : n_train + n_validation + n_test],
    }
    for rows in split.values():
        rows.sort(key=lambda ex: (ex.operating_point, ex.contingency))
    return split


def _split_by_operating_point(
    examples: Sequence[Example],
    *,
    rng: random.Random,
    settings: SplitSettings,
) -> dict[str, list[Example]]:
    operating_points = sorted({ex.operating_point for ex in examples})
    train_ops, validation_ops, test_ops = _partition_items(
        operating_points,
        rng=rng,
        train_fraction=settings.train_fraction,
        validation_fraction=settings.validation_fraction,
        test_fraction=settings.test_fraction,
    )

    op_to_split = {
        **{op: "train" for op in train_ops},
        **{op: "validation" for op in validation_ops},
        **{op: "test" for op in test_ops},
    }

    split: dict[str, list[Example]] = {"train": [], "validation": [], "test": []}
    for example in examples:
        split[op_to_split[example.operating_point]].append(example)

    for rows in split.values():
        rows.sort(key=lambda ex: (ex.operating_point, ex.contingency))
    return split


def build_split(
    examples: Sequence[Example],
    *,
    settings: SplitSettings,
) -> dict[str, list[Example]]:
    rng = random.Random(settings.seed)
    if settings.split_mode == "operating_point":
        return _split_by_operating_point(examples, rng=rng, settings=settings)
    return _split_examples(examples, rng=rng, settings=settings)


def write_split_csv(split: Mapping[str, Sequence[Example]], output_csv: Path) -> Path:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "operating_point", "contingency"],
        )
        writer.writeheader()
        for split_name in ("train", "validation", "test"):
            for example in split[split_name]:
                writer.writerow(
                    {
                        "split": split_name,
                        "operating_point": example.operating_point,
                        "contingency": example.contingency,
                    }
                )
    return output_csv


def build_dataset_split(
    dataset_csv: str | Path,
    *,
    output_csv: str | Path | None = None,
    dataset_dir: Path = DATASET_DIR,
    config: Optional[Mapping[str, object]] = None,
    config_path: Path = CONFIG_PATH,
    settings: Optional[SplitSettings] = None,
) -> SplitSummary:
    """Read a dataset CSV from ``data/Dataset``, split it, and write the split CSV.

    Parameters
    ----------
    dataset_csv:
        File name (for example ``Dataset_Voltage.csv``) or path under ``data/Dataset``.
    output_csv:
        Optional output path. Defaults to ``data/Dataset/train_val_test_split.csv``.
    """
    input_path = resolve_dataset_csv_path(dataset_csv, dataset_dir=dataset_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing dataset CSV: {input_path}")

    split_settings = settings or load_split_settings(config, config_path=config_path)
    examples = read_examples(input_path)
    if not examples:
        raise ValueError(f"No examples found in {input_path}")

    split = build_split(examples, settings=split_settings)

    if output_csv is None:
        out_path = default_split_output_path(input_path, dataset_dir=dataset_dir)
    else:
        out_path = resolve_dataset_csv_path(output_csv, dataset_dir=dataset_dir)

    write_split_csv(split, out_path)

    return SplitSummary(
        split_mode=split_settings.split_mode,
        seed=split_settings.seed,
        input_csv=input_path,
        output_csv=out_path,
        total_examples=len(examples),
        train_examples=len(split["train"]),
        validation_examples=len(split["validation"]),
        test_examples=len(split["test"]),
    )
