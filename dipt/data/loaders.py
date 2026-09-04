"""DataLoader construction shared by every training and evaluation script."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torchvision.transforms as T
from torch.utils.data import DataLoader

from dipt.data.datasets import MultipleDomainDataset
from dipt.data.registry import DatasetSpec


def default_transform(image_size: int | None = None) -> T.Compose:
    """Dataset-side transform.

    Patches are kept at their native resolution and only converted to tensors;
    resizing/normalisation is the responsibility of the teacher's CLIP processor
    and of each student's own preprocessing (see ``dipt.models.students``).
    """
    steps: list = []
    if image_size is not None:
        steps.append(T.Resize(image_size))
    steps.append(T.ToTensor())
    return T.Compose(steps)


def build_dataset(
    spec: DatasetSpec,
    domains: Sequence[str],
    data_root: str | Path | None = None,
    image_size: int | None = None,
) -> MultipleDomainDataset:
    return MultipleDomainDataset(
        root_dir=spec.root(data_root),
        domains=domains,
        transform=default_transform(image_size),
    )


def build_loader(
    spec: DatasetSpec,
    domains: Sequence[str],
    batch_size: int,
    shuffle: bool,
    data_root: str | Path | None = None,
    num_workers: int = 4,
    image_size: int | None = None,
) -> DataLoader:
    dataset = build_dataset(spec, domains, data_root=data_root, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
