"""Stage 1 - Domain-Invariant Prompt Tuning (DIPT).

One prompt is learned per domain. Only the K context tokens are trainable; the
VLM stays frozen. The objective is

    L = CE(logits, y) + score_weight * (1 - cos(text_features, aggregated templates))

i.e. a supervised term plus a KgCoOp-style drift penalty that keeps the learned
prompt near the class-generic anchor, which is what makes the per-domain prompts
averageable in stage 2.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dipt.evaluation.metrics import compute_metrics
from dipt.models.prompt_learner import LearnableAggPromptLearner, PromptedCLIP
from dipt.prompts.learned import BEST_CHECKPOINT, LAST_CHECKPOINT
from dipt.utils.misc import save_json


class PromptTuningTrainer:
    """Trains a single domain's prompt and checkpoints the best one by val accuracy."""

    def __init__(
        self,
        model: PromptedCLIP,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str | Path,
        device: torch.device | str,
        num_epochs: int = 1,
        lr: float = 5e-5,
        score_weight: float = 0.5,
        eval_interval: int = 100,
        logger=None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.num_epochs = num_epochs
        self.score_weight = score_weight
        self.eval_interval = eval_interval
        self.logger = logger

        self._freeze_all_but_prompt()
        self.optimizer = optim.Adam(
            [p for p in self.model.prompt_learner.parameters() if p.requires_grad], lr=lr
        )

        self.best_val_acc = 0.0
        self.best_state: dict | None = None
        self.history: list[dict] = []

    # ------------------------------------------------------------------ setup

    def _freeze_all_but_prompt(self) -> None:
        """Train the context tokens (and the agg. vector in the learnable-agg ablation)."""
        learnable_agg = isinstance(self.model.prompt_learner, LearnableAggPromptLearner)
        for name, param in self.model.named_parameters():
            is_context = "prompt_learner.context_tokens" in name
            is_agg = learnable_agg and "prompt_learner.agg_vector" in name
            param.requires_grad = is_context or is_agg

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.info(message)
        else:
            print(message)

    def save_setup(self, extra: dict | None = None) -> None:
        setup = {
            "class_names": self.model.prompt_learner.class_names,
            "num_context_tokens": self.model.prompt_learner.num_context_tokens,
            "embedding_dim": self.model.prompt_learner.embedding_dim,
            "score_weight": self.score_weight,
            "eval_interval": self.eval_interval,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "num_epochs": self.num_epochs,
            "batch_size": self.train_loader.batch_size,
            "device": str(self.device),
        }
        setup.update(extra or {})
        save_json(setup, self.output_dir / "training_setup.json")

    # --------------------------------------------------------------- training

    def _step(self, images: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        logits, drift = self.model(images)
        loss_ce = F.cross_entropy(logits, labels)
        loss = loss_ce + self.score_weight * drift

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        metrics = compute_metrics(logits.argmax(dim=1).cpu(), labels.cpu())
        return {
            "loss": loss.item(),
            "loss_ce": loss_ce.item(),
            "drift": drift.item(),
            "accuracy": metrics.accuracy,
            "f1": metrics.f1,
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        predictions, targets, total_loss = [], [], 0.0

        for batch in tqdm(loader, desc="validation", unit="batch", leave=False):
            images, labels = batch[0].to(self.device), batch[1].to(self.device)
            logits, drift = self.model(images)
            total_loss += (F.cross_entropy(logits, labels) + self.score_weight * drift).item()
            predictions.append(logits.argmax(dim=1).cpu())
            targets.append(labels.cpu())

        self.model.train()
        metrics = compute_metrics(torch.cat(predictions), torch.cat(targets))
        self._log(f"validation loss {total_loss / max(len(loader), 1):.4f} | {metrics}")
        return metrics.accuracy

    def _maybe_checkpoint(self, val_acc: float, tag: str) -> None:
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_state = {
                k: v.detach().cpu().clone()
                for k, v in self.model.prompt_learner.state_dict().items()
            }
            self._log(f"new best prompt at {tag} (val accuracy {val_acc:.4f})")

    def run(self) -> dict:
        self.model.train()
        for epoch in range(1, self.num_epochs + 1):
            self._log(f"===== epoch {epoch}/{self.num_epochs} =====")
            progress = tqdm(self.train_loader, desc=f"epoch {epoch}", unit="batch")

            for it, batch in enumerate(progress):
                images, labels = batch[0].to(self.device), batch[1].to(self.device)
                stats = self._step(images, labels)
                progress.set_postfix(loss=f"{stats['loss']:.3f}", acc=f"{stats['accuracy']:.3f}")
                self.history.append({"epoch": epoch, "iter": it, **stats})

                if self.eval_interval and (it + 1) % self.eval_interval == 0:
                    self._maybe_checkpoint(self.evaluate(self.val_loader), f"epoch {epoch} it {it + 1}")

            self._maybe_checkpoint(self.evaluate(self.val_loader), f"end of epoch {epoch}")

        last_path = self.output_dir / LAST_CHECKPOINT
        torch.save(self.model.prompt_learner.state_dict(), last_path)
        if self.best_state is not None:
            torch.save(self.best_state, self.output_dir / BEST_CHECKPOINT)

        save_json({"best_val_accuracy": self.best_val_acc, "history": self.history},
                  self.output_dir / "history.json")
        self._log(f"prompts saved to {self.output_dir} (best val accuracy {self.best_val_acc:.4f})")
        return {"best_val_accuracy": self.best_val_acc, "output_dir": str(self.output_dir)}
