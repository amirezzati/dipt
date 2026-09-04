"""Class-generic prompts built from hand-written templates.

This is the clean port of the original ``utils/avg_template_prompts.py``. For each
class we encode every template with the VLM text encoder and average the
embeddings; the result ``E`` (shape ``[num_classes, embed_dim]``) is:

* the frozen class-generic token inside every DIPT prompt (stage 1),
* the ``Agg. prompt Zero-Shot`` baseline, and
* the text target of the original (non-DIPT) RISE distance loss.

Templates live in ``configs/class_templates.json`` so they can be edited without
touching code.

Aggregation order
-----------------
The original averaged the **raw** text embeddings and let each caller normalise
the result once (``agg /= agg.norm(...)``). That is the default here
(``normalize_each=False``). Passing ``normalize_each=True`` averages unit vectors
instead, which is the more common CLIP convention but does *not* reproduce the
published numbers.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch

from dipt.models.vlm import encode_text
from dipt.paths import PROJECT_ROOT

DEFAULT_TEMPLATE_FILE = PROJECT_ROOT / "configs" / "class_templates.json"

#: Fallback used when a class has no entry in the template file.
GENERIC_TEMPLATES = [
    "a histopathology image of {class_name}",
    "a patch of {class_name}",
    "an H&E stained image of {class_name}",
]

#: Fallback single prompt for the one-prompt zero-shot baseline.
GENERIC_HANDCRAFTED = "a photo of a {class_name}"


@lru_cache(maxsize=4)
def load_template_file(path: str | Path = DEFAULT_TEMPLATE_FILE) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def get_class_templates(
    dataset: str,
    classnames: list[str],
    template_file: str | Path = DEFAULT_TEMPLATE_FILE,
) -> dict[str, list[str]]:
    """Return ``{classname: [prompt, ...]}`` for one dataset."""
    table = load_template_file(str(template_file)).get(dataset, {})
    templates: dict[str, list[str]] = {}
    for name in classnames:
        entry = table.get(name)
        if not entry:
            entry = [t.format(class_name=name) for t in GENERIC_TEMPLATES]
        templates[name] = list(entry)
    return templates


def get_handcrafted_prompts(
    dataset: str,
    classnames: list[str],
    template_file: str | Path = DEFAULT_TEMPLATE_FILE,
) -> list[str]:
    """One prompt per class, in ``classnames`` order (single-prompt baseline)."""
    table = load_template_file(str(template_file)).get("_handcrafted", {}).get(dataset, {})
    return [
        table.get(name) or GENERIC_HANDCRAFTED.format(class_name=name) for name in classnames
    ]


@torch.no_grad()
def aggregate_template_features(
    model,
    processor,
    classnames: list[str],
    dataset: str,
    device: torch.device | str = "cpu",
    template_file: str | Path = DEFAULT_TEMPLATE_FILE,
    normalize_each: bool = False,
    normalize_output: bool = True,
) -> torch.Tensor:
    """Aggregated class-generic embeddings, shape ``[num_classes, embed_dim]``.

    Args:
        normalize_each: L2-normalise every template embedding before averaging.
            ``False`` (default) reproduces the original implementation.
        normalize_output: L2-normalise the per-class mean. The original callers
            all did this immediately after aggregating.
    """
    templates = get_class_templates(dataset, classnames, template_file)

    features = []
    for name in classnames:
        embeddings = encode_text(model, processor, templates[name], device)
        if normalize_each:
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        features.append(embeddings.mean(dim=0))

    stacked = torch.stack(features, dim=0).to(device)
    if normalize_output:
        stacked = stacked / stacked.norm(dim=-1, keepdim=True)
    return stacked


@torch.no_grad()
def handcrafted_features(
    model,
    processor,
    prompts: list[str],
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Encode one hand-written prompt per class (single-prompt zero-shot baseline)."""
    features = encode_text(model, processor, prompts, device)
    return features / features.norm(dim=-1, keepdim=True)
