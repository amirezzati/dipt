"""Classification metrics and result reporting."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


@dataclass
class Metrics:
    """Weighted-average metrics for one evaluation pass."""

    accuracy: float
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"acc {self.accuracy * 100:6.2f} | prec {self.precision * 100:6.2f} | "
            f"rec {self.recall * 100:6.2f} | f1 {self.f1 * 100:6.2f}"
        )


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_metrics(predictions, targets) -> Metrics:
    """Accuracy plus weighted precision/recall/F1 (multi-class safe)."""
    preds = _to_numpy(predictions)
    labels = _to_numpy(targets)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return Metrics(float(accuracy), float(precision), float(recall), float(f1))


def summarise(per_domain: dict[str, Metrics]) -> dict[str, float]:
    """Mean and worst-case accuracy/F1 across domains (Table 2 in the paper)."""
    accuracies = [m.accuracy for m in per_domain.values()]
    f1_scores = [m.f1 for m in per_domain.values()]
    if not accuracies:
        return {}
    return {
        "mean_accuracy": float(np.mean(accuracies)),
        "mean_f1": float(np.mean(f1_scores)),
        "worst_accuracy": float(np.min(accuracies)),
        "worst_f1": float(np.min(f1_scores)),
    }


def save_report(
    path: str | Path,
    per_domain: dict[str, Metrics],
    meta: dict | None = None,
) -> Path:
    """Write a JSON report (and a human-readable ``.txt`` next to it)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": meta or {},
        "per_domain": {d: m.as_dict() for d, m in per_domain.items()},
        "summary": summarise(per_domain),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

    txt_path = path.with_suffix(".txt")
    with open(txt_path, "w", encoding="utf-8") as handle:
        for key, value in (meta or {}).items():
            handle.write(f"{key}: {value}\n")
        handle.write("\nPer-domain results\n")
        handle.write("-" * 60 + "\n")
        for domain, metrics in per_domain.items():
            handle.write(f"domain {domain}: {metrics}\n")
        handle.write("\nSummary\n")
        handle.write("-" * 60 + "\n")
        for key, value in payload["summary"].items():
            handle.write(f"{key}: {value * 100:.2f}\n")

    print(f"Results written to {path} and {txt_path}")
    return path
