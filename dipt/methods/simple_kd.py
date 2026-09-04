"""Stage 2 baseline - simple image-encoder-only knowledge distillation.

The paper's "KD" comparison row: no prompts, no text encoder at all. The
student is pulled directly toward the teacher's own image embedding (MSE,
both L2-normalised) alongside the usual classification loss::

    L = distill_weight * MSE(student_feature, teacher_image_feature)
      + classification_weight * CE(student_logits, y)

Everything else - the training loop, validation, checkpointing, optimizer -
is inherited unchanged from ``RiseTrainer``; only the loss differs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dipt.methods.rise import RiseTrainer
from dipt.models.students import FrozenTeacher, StudentModel


class SimpleKDTrainer(RiseTrainer):
    """MSE-to-teacher-image-embedding + CE. No text encoder involved."""

    def __init__(
        self,
        student: StudentModel,
        teacher: FrozenTeacher,
        train_loader: DataLoader,
        val_loaders: dict[str, DataLoader],
        device: torch.device | str,
        output_dir,
        run_name: str,
        epochs: int = 1,
        lr: float = 1e-3,
        optimizer_name: str = "sgd",
        distill_weight: float = 0.5,
        classification_weight: float = 0.5,
        eval_interval: int = 200,
        logger=None,
    ) -> None:
        super().__init__(
            student=student,
            teacher=teacher,
            train_loader=train_loader,
            val_loaders=val_loaders,
            text_features=None,
            device=device,
            output_dir=output_dir,
            run_name=run_name,
            epochs=epochs,
            lr=lr,
            optimizer_name=optimizer_name,
            distill_weight=distill_weight,
            classification_weight=classification_weight,
            distance_weight=0.0,
            temperature=1.0,
            eval_interval=eval_interval,
            logger=logger,
        )

    def _compute_losses(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
        teacher_features = self.teacher.image_features(self.teacher.preprocess(images))

        features = self.student.encode(self.student.preprocess(images))
        logits = self.student.classify(features)

        supervised = self.criterion(logits, labels)
        distill = F.mse_loss(features, teacher_features)

        loss = self.distill_weight * distill + self.classification_weight * supervised
        stats = {"loss": loss.item(), "ce": supervised.item(), "mse": distill.item()}
        return loss, stats, logits
