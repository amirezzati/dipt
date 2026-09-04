"""Console + file logging."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(levelname)s %(asctime)s | %(message)s"
_DATE_FORMAT = "%m/%d %H:%M:%S"


def get_logger(
    name: str = "dipt",
    log_file: str | Path | None = None,
    level: str = "INFO",
) -> logging.Logger:
    """Return a logger writing to stdout and, optionally, to ``log_file``.

    Repeated calls with the same ``name`` reuse the same logger; a new
    ``log_file`` adds a handler rather than replacing the logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(_FORMAT, _DATE_FORMAT)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        existing = {
            Path(h.baseFilename)
            for h in logger.handlers
            if isinstance(h, logging.FileHandler)
        }
        if log_file.resolve() not in existing:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
