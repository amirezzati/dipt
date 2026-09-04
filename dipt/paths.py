"""Project-relative path resolution.

Every default path in this repository is expressed relative to the repository
root, so the code runs unchanged on any machine as long as the layout below is
respected::

    <repo>/data       raw + prepared datasets   (scripts/download_datasets.py)
    <repo>/outputs    checkpoints, logs, results
"""

from __future__ import annotations

import os
from pathlib import Path

#: Repository root (the directory that contains ``dipt/`` and ``scripts/``).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Default dataset root. Override with the ``DIPT_DATA_ROOT`` env var or ``--data-root``.
DATA_ROOT = Path(os.environ.get("DIPT_DATA_ROOT", PROJECT_ROOT / "data"))

#: Default output root. Override with the ``DIPT_OUTPUT_ROOT`` env var or ``--output-root``.
OUTPUT_ROOT = Path(os.environ.get("DIPT_OUTPUT_ROOT", PROJECT_ROOT / "outputs"))


def resolve(path: str | os.PathLike, base: Path = PROJECT_ROOT) -> Path:
    """Resolve ``path``; relative paths are taken w.r.t. ``base`` (repo root)."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else (base / p)


def ensure_dir(path: str | os.PathLike) -> Path:
    """Create ``path`` (and parents) if missing and return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
