"""Shared argparse building blocks, so every script speaks the same language."""

from __future__ import annotations

import argparse

from dipt.data.registry import DATASETS
from dipt.models.students import STUDENT_CHOICES
from dipt.models.vlm import VLM_CHOICES
from dipt.paths import DATA_ROOT, OUTPUT_ROOT
from dipt.prompts.learned import PROMPT_SOURCES, default_prompt_root


def add_common_args(
    parser: argparse.ArgumentParser, multi_vlm: bool = False
) -> argparse.ArgumentParser:
    """Dataset / VLM / device / path options used by every entry point.

    Args:
        multi_vlm: accept several teachers at once (``--vlm plip quiltnet`` or
            ``--vlm all``). Only the zero-shot script needs this.
    """
    group = parser.add_argument_group("common")
    if multi_vlm:
        group.add_argument(
            "--vlm",
            nargs="+",
            default=["all"],
            choices=[*VLM_CHOICES, "all"],
            help=f"Vision-language teacher(s): {' '.join(VLM_CHOICES)} or all (default: all).",
        )
    else:
        group.add_argument(
            "--vlm",
            default="plip",
            choices=VLM_CHOICES,
            help="Vision-language teacher (default: plip).",
        )
    group.add_argument(
        "--dataset",
        default="camelyon17",
        choices=sorted(DATASETS),
        help="Prepared dataset to use (default: camelyon17).",
    )
    group.add_argument(
        "--data-root",
        default=str(DATA_ROOT),
        help="Root holding the prepared datasets (default: <repo>/data).",
    )
    group.add_argument(
        "--output-root",
        default=str(OUTPUT_ROOT),
        help="Root for checkpoints, logs and results (default: <repo>/outputs).",
    )
    group.add_argument("--gpu-index", type=int, default=0, help="CUDA device index.")
    group.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    group.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    group.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser


PROMPT_SOURCE_HELP = (
    "dipt: aggregated domain-invariant learned prompts (ours); "
    "template: aggregated hand-written templates (RISE baseline); "
    "handcrafted: one prompt per class (VL2V baseline)."
)


def add_prompt_args(
    parser: argparse.ArgumentParser,
    default_source: str = "dipt",
    multi_source: bool = False,
) -> argparse.ArgumentParser:
    """Options selecting which class text embeddings the KD stage should use.

    Args:
        multi_source: accept several prompt sources at once (``--prompt-source
            template handcrafted`` or ``all``). Only the zero-shot script needs this.
    """
    group = parser.add_argument_group("prompts")
    if multi_source:
        group.add_argument(
            "--prompt-source",
            nargs="+",
            default=[default_source],
            choices=[*PROMPT_SOURCES, "all"],
            help=PROMPT_SOURCE_HELP + " Pass several, or 'all'.",
        )
    else:
        group.add_argument(
            "--prompt-source",
            default=default_source,
            choices=PROMPT_SOURCES,
            help=PROMPT_SOURCE_HELP,
        )
    group.add_argument(
        "--num-context-tokens",
        "-k",
        type=int,
        default=4,
        help="K, the number of learnable context tokens (only used with --prompt-source dipt).",
    )
    group.add_argument(
        "--prompt-root",
        default=str(default_prompt_root()),
        help="Where stage-1 prompt checkpoints live (default: <repo>/outputs/prompts).",
    )
    return parser


def add_student_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Student backbone selection (``vlm`` == self-distillation)."""
    group = parser.add_argument_group("student")
    group.add_argument(
        "--student",
        default="resnet50_bit",
        choices=STUDENT_CHOICES,
        help=(
            "Student backbone. Use 'vlm' for self-distillation, i.e. a trainable "
            "copy of the teacher's own image encoder (default: resnet50_bit)."
        ),
    )
    group.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Randomly initialise the student instead of loading pretrained weights.",
    )
    return parser


def expand_choices(values: list[str], available: list[str]) -> list[str]:
    """Resolve an ``all`` entry and de-duplicate while preserving order."""
    if "all" in values:
        return list(available)
    seen, ordered = set(), []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def parse_domain_groups(values: list[str]) -> list[list[str]]:
    """Turn ``["0,2,3", "0,2,4"]`` into ``[["0","2","3"], ["0","2","4"]]``."""
    return [[d.strip() for d in group.split(",") if d.strip()] for group in values]
