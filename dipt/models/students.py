"""Student backbones with one common interface.

Both distillation methods only need three things from a student:

* ``preprocess(images)`` - turn a batch of ``[B, 3, H, W]`` float tensors in
  ``[0, 1]`` into whatever the backbone expects,
* ``encode(inputs)``     - an L2-normalised feature vector, and
* ``classify(features)`` - class logits.

That uniformity is what lets ``--student vlm`` (self-distillation: the student is
a trainable copy of the teacher's image encoder) sit next to ``--student
resnet50_bit`` behind the same training loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, CLIPModel

from dipt.models.preprocessing import INPUT_SIZE, imagenet_preprocess
from dipt.models.vlm import encode_image, get_vlm_spec, is_keep_model, preprocess_images


class StudentModel(nn.Module):
    """Common student interface. Subclasses set ``feature_dim``."""

    feature_dim: int

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classify(self.encode(self.preprocess(images)))


class TimmStudent(StudentModel):
    """A ``timm`` backbone with a linear head on top of normalised features.

    Default is ``resnetv2_50x1_bit.goog_in21k_ft_in1k``, the ResNet-50 student
    used by RISE.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        image_size: int = INPUT_SIZE,
        embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        import timm  # imported lazily so the package is optional for VLM-only runs

        # num_classes=0 -> the backbone returns pooled features, no head.
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        backbone_dim = self.backbone.num_features

        # RISE's original reference hand-patches its ResNetV2 with exactly this
        # projector (see WisconsinAIVision/RISE, timm/models/resnetv2.py) so the
        # pooled backbone feature can be compared directly against the VLM's
        # (embed_dim-sized) text embeddings. Left as identity when the caller
        # doesn't ask for it (e.g. VL2V-ADiP, which builds its own projection on
        # top of the raw backbone feature).
        if embed_dim is not None and embed_dim != backbone_dim:
            self.projector: nn.Module = nn.Linear(backbone_dim, embed_dim)
            self.feature_dim = embed_dim
        else:
            self.projector = nn.Identity()
            self.feature_dim = backbone_dim

        self.head = nn.Linear(self.feature_dim, num_classes)
        self.image_size = image_size

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        return imagenet_preprocess(images, self.image_size)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projector(self.backbone(inputs)), dim=-1)

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)

    @torch.no_grad()
    def init_head_from_text(self, text_features: torch.Tensor) -> None:
        """Initialise the classifier with the class text embeddings (RISE trick)."""
        if text_features.shape != self.head.weight.shape:
            raise ValueError(
                f"Cannot initialise head {tuple(self.head.weight.shape)} from text features "
                f"{tuple(text_features.shape)}."
            )
        self.head.weight.copy_(text_features.to(self.head.weight.dtype))
        self.head.bias.zero_()


class VLMStudent(StudentModel):
    """Self-distillation student: a *trainable copy* of the teacher's image encoder.

    The teacher stays frozen; this copy is fine-tuned so its visual features align
    with the domain-invariant text embeddings produced by DIPT.
    """

    def __init__(self, vlm: str, num_classes: int, image_size: int = INPUT_SIZE) -> None:
        super().__init__()
        spec = get_vlm_spec(vlm)
        if spec.backend == "keep":
            self.model = AutoModel.from_pretrained(spec.hf_id, trust_remote_code=True)
        else:
            self.model = CLIPModel.from_pretrained(spec.hf_id)
        self.feature_dim = spec.embed_dim
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.image_size = image_size

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        return preprocess_images(self.model, images, self.image_size)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.normalize(encode_image(self.model, inputs), dim=-1)

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


@dataclass(frozen=True)
class StudentSpec:
    key: str
    kind: str  # "timm" or "vlm"
    timm_name: str | None
    description: str


STUDENTS: dict[str, StudentSpec] = {
    "resnet50_bit": StudentSpec(
        "resnet50_bit",
        "timm",
        "resnetv2_50x1_bit.goog_in21k_ft_in1k",
        "ResNet-50 (BiT, IN21k->IN1k) - the RISE student in the paper.",
    ),
    "resnet50": StudentSpec(
        "resnet50", "timm", "resnet50", "Torchvision-style ResNet-50 via timm."
    ),
    "vit_base": StudentSpec(
        "vit_base", "timm", "vit_base_patch16_224", "ViT-B/16 student."
    ),
    "vit_small": StudentSpec(
        "vit_small", "timm", "vit_small_patch16_224", "ViT-S/16 student."
    ),
    "vlm": StudentSpec(
        "vlm", "vlm", None, "Self-distillation: trainable copy of the teacher image encoder."
    ),
}

STUDENT_CHOICES = sorted(STUDENTS)


def build_student(
    student: str,
    num_classes: int,
    vlm: str = "plip",
    pretrained: bool = True,
    image_size: int = INPUT_SIZE,
    embed_dim: int | None = None,
) -> StudentModel:
    """Instantiate a student by short name (``--student``).

    ``embed_dim``, when given, projects a timm student's pooled feature to that
    size (see ``TimmStudent``). Leave it unset to get the student's native
    feature space untouched, e.g. for VL2V-ADiP's own projection layer.
    """
    try:
        spec = STUDENTS[student]
    except KeyError:
        raise KeyError(f"Unknown student '{student}'. Available: {STUDENT_CHOICES}") from None

    if spec.kind == "vlm":
        return VLMStudent(vlm, num_classes, image_size=image_size)
    return TimmStudent(
        spec.timm_name, num_classes, pretrained=pretrained, image_size=image_size, embed_dim=embed_dim
    )


class FrozenTeacher(nn.Module):
    """Frozen VLM teacher exposing normalised image features and text logits."""

    def __init__(
        self,
        model,
        processor,
        text_features: torch.Tensor | None,
        image_size: int = INPUT_SIZE,
    ) -> None:
        super().__init__()
        self.model = model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.processor = processor
        self.register_buffer("text_features", text_features)
        self.image_size = image_size
        # CLIP checkpoints train logit_scale to ~100 (the standard clamp), so keep
        # that literal default for them; KEEP's learned scale is very different
        # (init exp(ln(1/0.04)) = 25) and using 100 here would over-sharpen its
        # soft labels for RISE's KD loss.
        self.default_scale = (
            float(self.model.logit_scale.exp().detach().cpu())
            if is_keep_model(self.model)
            else 100.0
        )

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        return preprocess_images(self.model, images, self.image_size)

    @torch.no_grad()
    def image_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.normalize(encode_image(self.model, inputs), dim=-1)

    @torch.no_grad()
    def logits(self, inputs: torch.Tensor, scale: float | None = None) -> torch.Tensor:
        scale = self.default_scale if scale is None else scale
        return (scale * self.image_features(inputs) @ self.text_features.T).float()
