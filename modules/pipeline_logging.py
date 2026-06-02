# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: Unified pipeline logging

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, TextIO

from .paths import DATA_DIR

LOG_NAME = "dynagnn"
DEFAULT_LOG_PATH = DATA_DIR / "dynagnn.log"

_configured = False
_log_path: Path = DEFAULT_LOG_PATH


LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
LOG_FORMAT = "%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s"


def build_pipeline_formatter() -> logging.Formatter:
    """Return the canonical DYNAGNN pipeline log formatter."""
    return logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)


class _StdoutToLogger(TextIO):
    """Redirect writes to a logger (one log line per printed line)."""

    def __init__(self, logger: logging.Logger, level: int, *, prefix: str = "") -> None:
        self._logger = logger
        self._level = int(level)
        self._prefix = str(prefix)
        self._buffer = ""

    def write(self, data: str) -> int:
        if not data:
            return 0

        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            msg = line.rstrip("\r")
            if msg:
                self._logger.log(self._level, "%s%s", self._prefix, msg)
        return len(data)

    def flush(self) -> None:
        msg = self._buffer.strip("\r\n")
        self._buffer = ""
        if msg:
            self._logger.log(self._level, "%s%s", self._prefix, msg)


def get_pipeline_log_path() -> Path:
    return _log_path


def configure_pipeline_logging(
    log_path: Optional[Path] = None,
    *,
    tee_stdout: bool = True,
    force: bool = False,
) -> logging.Logger:
    """Configure a single pipeline logger (file + console) and optional stdout tee."""
    global _configured, _log_path

    _log_path = (log_path or DEFAULT_LOG_PATH).resolve()
    _log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOG_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if _configured and not force:
        return logger

    logger.handlers.clear()
    fmt = build_pipeline_formatter()

    file_handler = logging.FileHandler(_log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler(sys.__stdout__)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    if tee_stdout and not getattr(sys.stdout, "_dynagnn_tee", False):
        redirect = _StdoutToLogger(logger, logging.INFO)
        redirect._dynagnn_tee = True  # type: ignore[attr-defined]
        sys.stdout = redirect  # type: ignore[assignment]

    _configured = True
    return logger


def get_logger() -> logging.Logger:
    if not _configured:
        configure_pipeline_logging()
    return logging.getLogger(LOG_NAME)


def log_step_banner(step_name: str) -> None:
    """Write a visible section header for one pipeline step."""
    logger = get_logger()
    separator = "=" * 60
    logger.info(separator)
    logger.info("STEP: %s", step_name)
    logger.info(separator)
