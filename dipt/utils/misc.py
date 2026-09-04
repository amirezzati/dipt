"""Small helpers shared across scripts."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(gpu_index: int = 0) -> torch.device:
    """Return ``cuda:<gpu_index>`` when available, otherwise CPU."""
    if torch.cuda.is_available() and gpu_index < torch.cuda.device_count():
        device = torch.device(f"cuda:{gpu_index}")
        print(f"Using device: {device} ({torch.cuda.get_device_name(gpu_index)})")
        return device
    print("CUDA not available - using CPU.")
    return torch.device("cpu")


def count_parameters(module: torch.nn.Module, trainable_only: bool = False) -> int:
    params = module.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def print_config(args: Any, title: str = "Configuration") -> None:
    """Pretty-print an ``argparse.Namespace``."""
    items = vars(args) if hasattr(args, "__dict__") else dict(args)
    width = max((len(k) for k in items), default=0)
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for key in sorted(items):
        print(f"{key:<{width}} : {items[key]}")
    print("=" * 60 + "\n")


def save_json(payload: dict, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def domain_tag(domains: list[str] | tuple[str, ...]) -> str:
    """``["0", "2", "3"] -> "0_2_3"`` - used in checkpoint file names."""
    return "_".join(str(d) for d in domains)


def run_name(train_domains: list[str], val_domain: str) -> str:
    """Canonical run identifier, e.g. ``t_0_2_3_v_1``."""
    return f"t_{domain_tag(train_domains)}_v_{val_domain}"


def to_row(values, colwidth: int = 12) -> str:
    """Format a list of values as a fixed-width log row."""

    def fmt(x):
        if isinstance(x, float) or np.issubdtype(type(x), np.floating):
            x = f"{x:.6f}"
        return str(x).ljust(colwidth)[:colwidth]

    return "  ".join(fmt(v) for v in values)
