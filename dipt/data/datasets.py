"""Datasets over a ``<root>/<domain>/<class>/<image>`` directory tree.

Both prepared datasets follow this layout, so a single class covers them::

    data/camelyon17/0/normal/patch_....png      # domain = WILDS hospital/center
    data/kather19/0/TUM/TUM-....tif            # domain = stain-augmented pseudo-center
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class MultipleDomainDataset(Dataset):
    """Images from one or more domains of a ``root/domain/class/*.png`` tree.

    Class indices are assigned by sorting the union of class names across the
    requested domains, so they are consistent no matter which domains are used.

    Each item is ``(image, class_label, domain_label)`` where ``domain_label``
    indexes into ``domains`` as given (not the folder name).
    """

    def __init__(
        self,
        root_dir: str | os.PathLike,
        domains: Sequence[str],
        transform: Callable | None = None,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.domains = [str(d) for d in domains]
        self.transform = transform

        if len(self.domains) != len(set(self.domains)):
            raise ValueError(f"Duplicate domains requested: {self.domains}")

        self.domain_paths = [self.root_dir / d for d in self.domains]
        for domain_path in self.domain_paths:
            if not domain_path.is_dir():
                raise FileNotFoundError(
                    f"Domain directory {domain_path} does not exist. "
                    "Did you run scripts/download_datasets.py?"
                )

        all_classes: set[str] = set()
        for domain_path in self.domain_paths:
            all_classes.update(p.name for p in domain_path.iterdir() if p.is_dir())
        self.classes = sorted(all_classes)
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.domain_to_idx = {d: i for i, d in enumerate(self.domains)}

        self.image_paths: list[Path] = []
        self.class_labels: list[int] = []
        self.domain_labels: list[int] = []

        for domain, domain_path in zip(self.domains, self.domain_paths):
            domain_label = self.domain_to_idx[domain]
            for class_dir in sorted(p for p in domain_path.iterdir() if p.is_dir()):
                class_label = self.class_to_idx[class_dir.name]
                for img_path in sorted(class_dir.iterdir()):
                    if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                        self.image_paths.append(img_path)
                        self.class_labels.append(class_label)
                        self.domain_labels.append(domain_label)

        if not self.image_paths:
            raise RuntimeError(f"No images found under {self.root_dir} for domains {self.domains}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.class_labels[idx], self.domain_labels[idx]

    def get_class_mapping(self) -> dict[str, int]:
        return dict(self.class_to_idx)

    def get_domain_mapping(self) -> dict[str, int]:
        return dict(self.domain_to_idx)

    def __repr__(self) -> str:
        return (
            f"MultipleDomainDataset(root={self.root_dir}, domains={self.domains}, "
            f"classes={self.classes}, n={len(self)})"
        )


class CyclicDataLoader:
    """Infinite iterator over a ``DataLoader`` (restarts when exhausted).

    VL2V-ADiP trains for a fixed number of steps rather than epochs.
    """

    def __init__(self, data_loader) -> None:
        self.data_loader = data_loader
        self.iterator = iter(data_loader)

    def __iter__(self) -> "CyclicDataLoader":
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.data_loader)
            return next(self.iterator)
