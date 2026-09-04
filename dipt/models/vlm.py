"""Pathology vision-language teachers.

Every entry point in this repository takes ``--vlm {plip,quiltnet,keep}``; this
module is the single place where those names map to HuggingFace checkpoints, and
the single place that knows how each checkpoint's architecture differs.

PLIP and QuiltNet are plain ``CLIPModel`` checkpoints - image encoder, text
encoder and tokenizer all match ``transformers``' CLIP classes. KEEP is not: its
image tower is a timm ViT-L/16 with its own projection head and its text tower is
BERT, wired together by custom ``trust_remote_code`` modelling code
(``Astaxanthin/KEEP``). ``encode_image``/``encode_text``/``preprocess_images``
below are the seam that lets every other module - ``students.py``,
``templates.py``, ``evaluate.py`` - call one API regardless of which backend
loaded.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor

from dipt.models.preprocessing import INPUT_SIZE, clip_preprocess, imagenet_preprocess


@dataclass(frozen=True)
class VLMSpec:
    key: str
    hf_id: str
    embed_dim: int
    description: str
    backend: str = "clip"  # "clip" (transformers.CLIPModel) or "keep" (custom AutoModel)


VLMS: dict[str, VLMSpec] = {
    "plip": VLMSpec(
        key="plip",
        hf_id="vinid/plip",
        embed_dim=512,
        description="PLIP - CLIP ViT-B/32 fine-tuned on pathology image-text pairs (OpenPath).",
    ),
    "quiltnet": VLMSpec(
        key="quiltnet",
        hf_id="wisdomik/QuiltNet-B-32",
        embed_dim=512,
        description="QuiltNet-B-32 - CLIP ViT-B/32 trained on the Quilt-1M histopathology corpus.",
    ),
    "keep": VLMSpec(
        key="keep",
        hf_id="Astaxanthin/KEEP",
        embed_dim=768,
        description=(
            "KEEP - ViT-L/16 + BERT, knowledge-graph-enhanced pathology VLM "
            "(MAGIC-AI4Med, Cancer Cell 2026)."
        ),
        backend="keep",
    ),
}

VLM_CHOICES = sorted(VLMS)


def get_vlm_spec(name: str) -> VLMSpec:
    """Accept either the short key (``plip``) or the full HF id (``vinid/plip``)."""
    if name in VLMS:
        return VLMS[name]
    for spec in VLMS.values():
        if spec.hf_id == name:
            return spec
    raise KeyError(f"Unknown VLM '{name}'. Available: {VLM_CHOICES}")


def load_vlm(
    name: str,
    device: torch.device | str = "cpu",
    freeze: bool = True,
):
    """Load a teacher VLM and its processor/tokenizer.

    Args:
        name: ``plip``/``quiltnet``/``keep`` or a full HuggingFace id.
        device: device to place the model on.
        freeze: put the model in eval mode and disable gradients (teacher use).

    Returns:
        ``(model, processor, spec)``. ``processor`` is a ``CLIPProcessor`` for the
        ``clip`` backend or a ``BertTokenizer``-family ``AutoTokenizer`` for
        ``keep`` - both are called the same way by ``encode_text`` below.
    """
    spec = get_vlm_spec(name)
    if spec.backend == "keep":
        model = AutoModel.from_pretrained(spec.hf_id, trust_remote_code=True).to(device)
        processor = AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=True)
    else:
        model = CLIPModel.from_pretrained(spec.hf_id).to(device)
        processor = CLIPProcessor.from_pretrained(spec.hf_id)
    if freeze:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    return model, processor, spec


def is_keep_model(model) -> bool:
    return getattr(getattr(model, "config", None), "model_type", None) == "keep"


def preprocess_images(model, images: torch.Tensor, size: int = INPUT_SIZE) -> torch.Tensor:
    """Resize + normalise a batch for whichever backend ``model`` is.

    KEEP's ViT was trained with ImageNet statistics (it starts from a timm
    checkpoint); PLIP/QuiltNet use CLIP statistics. Both are 224x224.
    """
    if is_keep_model(model):
        return imagenet_preprocess(images, size)
    return clip_preprocess(images, size)


def encode_image(model, pixel_values: torch.Tensor) -> torch.Tensor:
    """Raw (unnormalised) image embeddings - matches ``CLIPModel.get_image_features``."""
    if is_keep_model(model):
        return model.visual_head(model.visual(pixel_values))
    return model.get_image_features(pixel_values=pixel_values)


def encode_text(model, processor, texts: list[str], device: torch.device | str) -> torch.Tensor:
    """Raw (unnormalised) text embeddings - matches ``CLIPModel.get_text_features``."""
    if is_keep_model(model):
        tokens = processor(
            texts, max_length=256, padding="max_length", truncation=True, return_tensors="pt"
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        return model.text(**tokens).pooler_output
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return model.get_text_features(**inputs)
