"""On-device image preprocessing shared by teachers and students.

Patches leave the DataLoader as ``[B, 3, H, W]`` float tensors in ``[0, 1]`` at
their native resolution. Resizing and normalisation happen here, on the GPU,
rather than by round-tripping every batch through PIL and ``CLIPProcessor`` -
same statistics, substantially faster.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

#: CLIP / PLIP / QuiltNet statistics (identical to ``CLIPProcessor``).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

#: ImageNet statistics, used by the timm students.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

INPUT_SIZE = 224


def resize_normalize(
    images: torch.Tensor,
    size: int = INPUT_SIZE,
    mean: tuple[float, ...] = CLIP_MEAN,
    std: tuple[float, ...] = CLIP_STD,
) -> torch.Tensor:
    """Bicubic-resize to ``size`` (if needed) and normalise with ``mean``/``std``."""
    if images.shape[-2:] != (size, size):
        images = F.interpolate(images, size=(size, size), mode="bicubic", align_corners=False)
    mean_t = torch.tensor(mean, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean_t) / std_t


def clip_preprocess(images: torch.Tensor, size: int = INPUT_SIZE) -> torch.Tensor:
    return resize_normalize(images, size, CLIP_MEAN, CLIP_STD)


def imagenet_preprocess(images: torch.Tensor, size: int = INPUT_SIZE) -> torch.Tensor:
    return resize_normalize(images, size, IMAGENET_MEAN, IMAGENET_STD)
