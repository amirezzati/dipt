"""Stage 2 - VL2V-ADiP distillation (Addepalli et al., CVPR 2024).

Two training stages followed by inference:

* **stage 1** - freeze the student encoder, train only the projection layer so
  that projected features align with the teacher's image *and* text embeddings::

      L = -lambda * cos(proj, teacher image feat) - (1 - lambda) * cos(proj, class text feat)

* **stage 2** - keep the projection, unfreeze and fine-tune the student encoder
  with the same objective.
* **stage 3** - inference with the frozen zero-shot classification head.

The classification head is built from class text embeddings. With
``--prompt-source dipt`` those are the aggregated domain-invariant DIPT prompts;
``handcrafted``/``template`` reproduce the original VL2V-ADiP baseline.

``--student vlm`` turns this into **self-distillation**: the student encoder is a
trainable copy of the teacher's image encoder.
"""

from __future__ import annotations

from itertools import chain
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from dipt.models.students import StudentModel

STAGE_TRAIN_PROJECTION = 1
STAGE_TRAIN_ENCODER = 2
STAGE_INFERENCE = 3


class ClassificationHead(nn.Linear):
    """Frozen linear head initialised from (scaled) class text embeddings."""

    def __init__(self, weights: torch.Tensor, normalize: bool = True) -> None:
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        with torch.no_grad():
            self.weight.copy_(weights.clone())
            self.bias.zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)


def build_zero_shot_head(
    text_features: torch.Tensor, logit_scale: torch.Tensor | float
) -> ClassificationHead:
    """Scale L2-normalised class text features by ``exp(logit_scale)`` into a head."""
    scale = logit_scale.exp() if isinstance(logit_scale, torch.Tensor) else float(logit_scale)
    weights = (text_features / text_features.norm(dim=-1, keepdim=True)) * scale
    return ClassificationHead(weights.float(), normalize=True)


class VL2VADiP(nn.Module):
    """Student encoder + projection layer + frozen zero-shot head."""

    def __init__(
        self,
        stage: int,
        student: StudentModel,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor | float,
        embed_dim: int = 512,
        lam: float = 0.5,
        lr: float = 5e-5,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.stage = stage
        self.lam = lam
        self.student = student
        self.embed_dim = embed_dim

        self.classifier = build_zero_shot_head(text_features, logit_scale)
        for param in self.classifier.parameters():
            param.requires_grad = False

        self.proj_lyr = nn.Linear(student.feature_dim, embed_dim)

        if stage == STAGE_TRAIN_PROJECTION:
            train_params = chain(self.proj_lyr.parameters())
        elif stage == STAGE_TRAIN_ENCODER:
            # The projection is kept as learned in stage 1; only the encoder moves.
            train_params = chain(self.student.parameters())
        else:
            train_params = None

        self.optimizer = (
            torch.optim.Adam(train_params, lr=lr, weight_decay=weight_decay)
            if train_params is not None
            else None
        )
        if stage == STAGE_TRAIN_PROJECTION:
            for param in self.student.parameters():
                param.requires_grad = False

    # ------------------------------------------------------------------ loss

    @staticmethod
    def dfc_loss(
        image_feat: torch.Tensor, text_feat: torch.Tensor, proj: torch.Tensor, lam: float
    ) -> torch.Tensor:
        """Dual feature-consistency loss: align the projection to image and text."""
        to_image = -torch.mean(F.cosine_similarity(proj, image_feat))
        to_text = -torch.mean(F.cosine_similarity(proj, text_feat))
        return lam * to_image + (1.0 - lam) * to_text

    # --------------------------------------------------------------- forward

    def features(self, images: torch.Tensor) -> torch.Tensor:
        return self.proj_lyr(self.student.encode(self.student.preprocess(images)))

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.predict(images)

    def update(self, images: torch.Tensor, labels: torch.Tensor, teacher) -> dict[str, float]:
        teacher_inputs = teacher.preprocess(images)
        teacher_image_feat = teacher.image_features(teacher_inputs)
        teacher_text_feat = teacher.text_features[labels]

        proj = self.features(images)
        proj = proj / proj.norm(dim=-1, keepdim=True)

        loss = self.dfc_loss(teacher_image_feat, teacher_text_feat, proj, self.lam)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}


def checkpoint_name(run_name: str, stage: int, best: bool = True) -> str:
    prefix = "best" if best else "last"
    return f"{run_name}_stage{stage}_{prefix}.pth"


def save_checkpoint(
    algorithm: VL2VADiP,
    path: str | Path,
    meta: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    device = next(algorithm.parameters()).device
    torch.save(
        {"model_dict": algorithm.cpu().state_dict(), "meta": meta or {}},
        path,
    )
    algorithm.to(device)


def load_checkpoint(
    algorithm: VL2VADiP,
    path: str | Path,
    device: torch.device | str,
    strict: bool = True,
) -> VL2VADiP:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"VL2V-ADiP checkpoint not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    algorithm.load_state_dict(payload["model_dict"], strict=strict)
    return algorithm.to(device)
