"""Evaluation loops shared by every script."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dipt.evaluation.metrics import Metrics, compute_metrics
from dipt.models.students import StudentModel


@torch.no_grad()
def evaluate_student(
    student: StudentModel,
    loader: DataLoader,
    device: torch.device | str,
    desc: str = "eval",
) -> Metrics:
    """Top-1 metrics for a RISE-style student (encoder + linear head)."""
    student.eval()
    predictions, targets = [], []

    for batch in tqdm(loader, desc=desc, unit="batch", leave=False):
        images, labels = batch[0].to(device), batch[1].to(device)
        logits = student(images)
        predictions.append(logits.argmax(dim=1).cpu())
        targets.append(labels.cpu())

    student.train()
    return compute_metrics(torch.cat(predictions), torch.cat(targets))


@torch.no_grad()
def evaluate_zero_shot(
    model,
    processor,
    text_features: torch.Tensor,
    loader: DataLoader,
    device: torch.device | str,
    logit_scale: float = 100.0,
    desc: str = "zero-shot",
) -> Metrics:
    """Zero-shot metrics for a VLM against a fixed set of class text features."""
    from dipt.models.vlm import encode_image, preprocess_images

    model.eval()
    text_features = text_features.to(device)
    predictions, targets = [], []

    for batch in tqdm(loader, desc=desc, unit="batch", leave=False):
        images, labels = batch[0].to(device), batch[1].to(device)
        pixel_values = preprocess_images(model, images)
        image_features = F.normalize(encode_image(model, pixel_values), dim=-1)
        logits = (logit_scale * image_features @ text_features.T).float()
        predictions.append(logits.argmax(dim=1).cpu())
        targets.append(labels.cpu())

    return compute_metrics(torch.cat(predictions), torch.cat(targets))


@torch.no_grad()
def evaluate_algorithm(
    algorithm,
    loader: DataLoader,
    device: torch.device | str,
    desc: str = "eval",
) -> tuple[Metrics, float]:
    """Metrics plus mean cross-entropy for a VL2V-ADiP algorithm."""
    algorithm.eval()
    predictions, targets = [], []
    loss_sum, n_batches = 0.0, 0

    for batch in tqdm(loader, desc=desc, unit="batch", leave=False):
        images, labels = batch[0].to(device), batch[1].to(device)
        logits = algorithm.predict(images)
        loss_sum += F.cross_entropy(logits, labels).item()
        n_batches += 1
        predictions.append(logits.argmax(dim=1).cpu())
        targets.append(labels.cpu())

    algorithm.train()
    metrics = compute_metrics(torch.cat(predictions), torch.cat(targets))
    return metrics, loss_sum / max(n_batches, 1)
