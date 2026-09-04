"""Loading and aggregating DIPT prompts learned per domain.

The original code hard-coded one absolute checkpoint path per domain. Here the
layout is deterministic, so every downstream stage can find the prompts it needs
from ``(dataset, vlm, k, domain)`` alone::

    outputs/prompts/<dataset>/<vlm>/k<K>/domain<D>/best_prompt_learner.pth
                                              .../last_prompt_learner.pth
                                              .../training_setup.json

Aggregation (Eq. 3 in the paper): encode each domain's prompt with the VLM text
encoder, average across the training domains, and re-normalise. The result is the
domain-invariant, class-generic text embedding used by RISE and VL2V-ADiP.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from transformers import CLIPModel, CLIPProcessor

from dipt.models.prompt_learner import PromptLearner, TextEncoder
from dipt.paths import OUTPUT_ROOT, resolve

BEST_CHECKPOINT = "best_prompt_learner.pth"
LAST_CHECKPOINT = "last_prompt_learner.pth"


def default_prompt_root() -> Path:
    return OUTPUT_ROOT / "prompts"


def prompt_run_dir(
    dataset: str,
    vlm: str,
    num_context_tokens: int,
    domain: str,
    prompt_root: str | Path | None = None,
) -> Path:
    """Directory holding the prompt learned on a single domain."""
    root = resolve(prompt_root) if prompt_root is not None else default_prompt_root()
    return Path(root) / dataset / vlm / f"k{num_context_tokens}" / f"domain{domain}"


def prompt_checkpoint_path(
    dataset: str,
    vlm: str,
    num_context_tokens: int,
    domain: str,
    prompt_root: str | Path | None = None,
    best: bool = True,
) -> Path:
    run_dir = prompt_run_dir(dataset, vlm, num_context_tokens, domain, prompt_root)
    name = BEST_CHECKPOINT if best else LAST_CHECKPOINT
    path = run_dir / name
    if best and not path.exists() and (run_dir / LAST_CHECKPOINT).exists():
        # Short runs may never trigger a "best" save; fall back to the last one.
        return run_dir / LAST_CHECKPOINT
    return path


def load_prompt_learner(
    checkpoint_path: str | Path,
    classnames: list[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    num_context_tokens: int,
    device: torch.device | str = "cpu",
) -> PromptLearner:
    """Rebuild a ``PromptLearner`` and load a trained checkpoint into it."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Prompt checkpoint not found: {checkpoint_path}\n"
            "Run scripts/train_domain_prompts.py for this (dataset, vlm, k, domain) first."
        )
    learner = PromptLearner(classnames, model, processor, num_context_tokens)
    learner.load_state_dict(torch.load(checkpoint_path, map_location=device))
    learner.to(device)
    learner.eval()
    return learner


@torch.no_grad()
def aggregate_learned_prompts(
    classnames: list[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    num_context_tokens: int,
    domains: Sequence[str],
    dataset: str,
    vlm: str,
    prompt_root: str | Path | None = None,
    device: torch.device | str = "cpu",
    best: bool = True,
) -> torch.Tensor:
    """Average the per-domain DIPT prompts into domain-invariant text features.

    Returns:
        ``[num_classes, embed_dim]`` L2-normalised tensor.
    """
    if not domains:
        raise ValueError("At least one domain is required to aggregate prompts.")

    text_encoder = TextEncoder(model).to(device)
    per_domain = []
    for domain in domains:
        path = prompt_checkpoint_path(
            dataset, vlm, num_context_tokens, str(domain), prompt_root, best=best
        )
        learner = load_prompt_learner(
            path, classnames, model, processor, num_context_tokens, device
        )
        per_domain.append(text_encoder(learner(), learner.tokenized_learnable_prompts))

    stacked = torch.stack(per_domain)  # [num_domains, num_classes, embed_dim]
    mean = stacked.mean(dim=0)
    mean = mean / mean.norm(dim=-1, keepdim=True)
    return mean.detach().clone()


@torch.no_grad()
def build_text_targets(
    prompt_source: str,
    classnames: list[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    dataset: str,
    vlm: str,
    domains: Sequence[str],
    num_context_tokens: int = 4,
    prompt_root: str | Path | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Single entry point for the class text embeddings used by the KD stage.

    Args:
        prompt_source:
            ``dipt``      - aggregated DIPT prompts (this paper's contribution),
            ``template``  - aggregated hand-written templates (RISE baseline),
            ``handcrafted`` - one prompt per class (VL2V baseline).
    """
    from dipt.prompts.templates import (
        aggregate_template_features,
        get_handcrafted_prompts,
        handcrafted_features,
    )

    if prompt_source == "dipt":
        return aggregate_learned_prompts(
            classnames,
            model,
            processor,
            num_context_tokens,
            domains,
            dataset,
            vlm,
            prompt_root=prompt_root,
            device=device,
        )
    if prompt_source == "template":
        return aggregate_template_features(model, processor, classnames, dataset, device)
    if prompt_source == "handcrafted":
        prompts = get_handcrafted_prompts(dataset, classnames)
        return handcrafted_features(model, processor, prompts, device)

    raise ValueError(
        f"Unknown prompt source '{prompt_source}'. Use dipt, template or handcrafted."
    )


PROMPT_SOURCES = ("dipt", "template", "handcrafted")
