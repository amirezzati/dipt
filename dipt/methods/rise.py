"""Stage 2 - RISE distillation from the VLM text encoder.

The student is trained with three terms:

    L = w_kd  * KL(student / T || teacher / T) * T^2      soft-label distillation
      + w_cls * CE(student logits, y)                      supervised
      + w_dist* (1 - cos(student feature, E_y))            absolute-distance loss

``E`` is the class text embedding matrix. With ``--prompt-source template`` it is
the aggregated hand-written templates (the original RISE); with
``--prompt-source dipt`` it is the aggregated domain-invariant DIPT prompts,
which is the contribution of this work.

Setting ``--student vlm`` makes the student a trainable copy of the teacher's own
image encoder, i.e. **self-distillation**.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dipt.evaluation.evaluate import evaluate_student
from dipt.evaluation.metrics import Metrics, compute_metrics
from dipt.models.students import FrozenTeacher, StudentModel


def build_optimizer(
    student: StudentModel,
    epochs: int,
    lr: float,
    optimizer_name: str = "sgd",
    nesterov: bool = False,
) -> tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler]:
    """SGD (as in RISE) or Adam, with a step schedule at 80% of training."""
    params = student.parameters()
    if optimizer_name == "sgd":
        optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=nesterov)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(params, lr=lr, weight_decay=5e-4)
    else:
        raise ValueError(f"Unknown optimizer '{optimizer_name}' (use sgd or adam).")

    step_size = max(1, int(epochs * 0.8))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    return optimizer, scheduler


class RiseTrainer:
    """RISE training loop with periodic validation and best-checkpoint saving."""

    def __init__(
        self,
        student: StudentModel,
        teacher: FrozenTeacher,
        train_loader: DataLoader,
        val_loaders: dict[str, DataLoader],
        text_features: torch.Tensor | None,
        device: torch.device | str,
        output_dir: str | Path,
        run_name: str,
        epochs: int = 1,
        lr: float = 8e-4,
        optimizer_name: str = "sgd",
        distill_weight: float = 0.3,
        classification_weight: float = 0.4,
        distance_weight: float = 0.3,
        temperature: float = 2.0,
        eval_interval: int = 200,
        logger=None,
    ) -> None:
        self.student = student.to(device)
        self.teacher = teacher.to(device)
        self.train_loader = train_loader
        self.val_loaders = val_loaders
        self.text_features = text_features.to(device) if text_features is not None else None
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name

        self.epochs = epochs
        self.distill_weight = distill_weight
        self.classification_weight = classification_weight
        self.distance_weight = distance_weight
        self.temperature = temperature
        self.eval_interval = eval_interval
        self.logger = logger

        self.optimizer, self.scheduler = build_optimizer(
            self.student, epochs, lr, optimizer_name
        )
        self.criterion = nn.CrossEntropyLoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.best_accuracy: dict[str, float] = {}

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.info(message)
        else:
            print(message)

    # ------------------------------------------------------------------ loss

    def _compute_losses(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
        teacher_logits = self.teacher.logits(self.teacher.preprocess(images))

        features = self.student.encode(self.student.preprocess(images))
        logits = self.student.classify(features)

        supervised = self.criterion(logits, labels)

        kd = F.kl_div(
            F.log_softmax(logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction="batchmean",
        ) * (self.temperature ** 2)

        # Pull each image embedding toward the text embedding of its own class.
        target_text = self.text_features[labels]
        ones = torch.ones(images.size(0), device=self.device)
        distance = self.cosine_loss(features, target_text, ones)

        loss = (
            self.distill_weight * kd
            + self.classification_weight * supervised
            + self.distance_weight * distance
        )
        stats = {
            "loss": loss.item(),
            "ce": supervised.item(),
            "kd": kd.item(),
            "dist": distance.item(),
        }
        return loss, stats, logits

    # -------------------------------------------------------------- training

    def _validate_and_save(self, tag: str) -> dict[str, Metrics]:
        results: dict[str, Metrics] = {}
        for domain, loader in self.val_loaders.items():
            metrics = evaluate_student(self.student, loader, self.device, desc=f"val d{domain}")
            results[domain] = metrics
            self._log(f"[{tag}] validation domain {domain}: {metrics}")

            if metrics.accuracy > self.best_accuracy.get(domain, 0.0):
                self.best_accuracy[domain] = metrics.accuracy
                path = self.output_dir / f"{self.run_name}_best.pth"
                torch.save(
                    {
                        "student_state_dict": self.student.state_dict(),
                        "val_domain": domain,
                        "val_accuracy": metrics.accuracy,
                        "run_name": self.run_name,
                    },
                    path,
                )
                self._log(f"[{tag}] new best for domain {domain} -> {path}")
        return results

    def run(self) -> dict:
        self.student.train()
        for epoch in range(1, self.epochs + 1):
            self._log(f"===== epoch {epoch}/{self.epochs} =====")
            progress = tqdm(self.train_loader, desc=f"epoch {epoch}", unit="batch")

            for it, batch in enumerate(progress):
                images, labels = batch[0].to(self.device), batch[1].to(self.device)

                loss, stats, logits = self._compute_losses(images, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_metrics = compute_metrics(logits.argmax(dim=1).cpu(), labels.cpu())
                progress.set_postfix(
                    loss=f"{stats['loss']:.3f}",
                    ce=f"{stats['ce']:.3f}",
                    acc=f"{batch_metrics.accuracy:.3f}",
                )

                if self.eval_interval and (it + 1) % self.eval_interval == 0:
                    self._validate_and_save(f"epoch {epoch} it {it + 1}")

            self.scheduler.step()

        final = self._validate_and_save("final")
        last_path = self.output_dir / f"{self.run_name}_last.pth"
        torch.save(
            {"student_state_dict": self.student.state_dict(), "run_name": self.run_name},
            last_path,
        )
        self._log(f"last checkpoint -> {last_path}")
        return {
            "best_val_accuracy": self.best_accuracy,
            "final_metrics": {d: m.as_dict() for d, m in final.items()},
        }
