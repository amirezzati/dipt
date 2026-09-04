#!/usr/bin/env python
"""Zero-shot inference for every pathology VLM, across every domain.

This is the clean port of ``dipt/ZeroShot/Plip_zero_shot_domain.py``, generalised
to sweep teachers and prompt sources in a single run and to print a comparison
table (the "Zero-Shot" rows of Tables 1-3).

Prompt sources
--------------
``template``     aggregated hand-written templates ("Agg. prompt Zero-Shot")
``handcrafted``  one prompt per class
``dipt``         aggregated DIPT prompts - use this to sanity-check a stage-1 run
                 before spending GPU time on distillation

Both ``--vlm`` and ``--prompt-source`` accept several values, or ``all``.

Examples
--------
    # every teacher x both prompt-free baselines (the default)
    python scripts/evaluate_zero_shot.py

    # one teacher, one source
    python scripts/evaluate_zero_shot.py --vlm plip --prompt-source template

    # check learned prompts for both teachers
    python scripts/evaluate_zero_shot.py --vlm all --prompt-source dipt -k 4 \
        --prompt-domains 0 2 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from dipt.cli import add_common_args, add_prompt_args, expand_choices  # noqa: E402
from dipt.data.loaders import build_loader  # noqa: E402
from dipt.data.registry import get_dataset_spec  # noqa: E402
from dipt.evaluation.evaluate import evaluate_zero_shot  # noqa: E402
from dipt.evaluation.metrics import Metrics, save_report, summarise  # noqa: E402
from dipt.models.vlm import VLM_CHOICES, load_vlm  # noqa: E402
from dipt.paths import resolve  # noqa: E402
from dipt.prompts.learned import PROMPT_SOURCES, build_text_targets  # noqa: E402
from dipt.utils.logging import get_logger  # noqa: E402
from dipt.utils.misc import print_config, select_device  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Zero-shot inference for PLIP / QuiltNet across all domains.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_args(parser, multi_vlm=True)
    add_prompt_args(parser, default_source="template", multi_source=True)
    parser.set_defaults(prompt_source=["template", "handcrafted"])

    group = parser.add_argument_group("zero-shot")
    group.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Domains to evaluate on (default: every domain of the dataset).",
    )
    group.add_argument(
        "--prompt-domains",
        nargs="+",
        default=None,
        help=(
            "Domains whose DIPT prompts are averaged (only for --prompt-source dipt; "
            "default: every non-validation domain)."
        ),
    )
    group.add_argument(
        "--results-dir",
        default=None,
        help="Where to write reports (default: <output-root>/zero_shot).",
    )
    return parser


def run_one(
    args, spec, vlm: str, prompt_source: str, eval_domains: list[str],
    prompt_domains: list[str], device, logger,
) -> dict[str, Metrics]:
    logger.info(f"--- {vlm} | {prompt_source} ---")
    model, processor, vlm_spec = load_vlm(vlm, device=device, freeze=True)

    text_features = build_text_targets(
        prompt_source=prompt_source,
        classnames=spec.classnames,
        model=model,
        processor=processor,
        dataset=spec.name,
        vlm=vlm,
        domains=prompt_domains,
        num_context_tokens=args.num_context_tokens,
        prompt_root=args.prompt_root,
        device=device,
    )
    logger.info(f"text features {tuple(text_features.shape)} from {vlm_spec.hf_id}")

    per_domain: dict[str, Metrics] = {}
    for domain in eval_domains:
        loader = build_loader(
            spec, [domain], args.batch_size, shuffle=False,
            data_root=args.data_root, num_workers=args.num_workers,
        )
        metrics = evaluate_zero_shot(
            model, processor, text_features, loader, device, desc=f"{vlm}/{domain}"
        )
        per_domain[domain] = metrics
        logger.info(f"domain {domain}: {metrics}")

    results_dir = (
        resolve(args.results_dir) if args.results_dir else resolve(args.output_root) / "zero_shot"
    )
    save_report(
        results_dir / f"{spec.name}_{vlm}_{prompt_source}.json",
        per_domain,
        meta={
            "dataset": spec.name,
            "vlm": vlm,
            "vlm_hf_id": vlm_spec.hf_id,
            "prompt_source": prompt_source,
            "num_context_tokens": args.num_context_tokens if prompt_source == "dipt" else None,
            "prompt_domains": prompt_domains if prompt_source == "dipt" else None,
            "eval_domains": eval_domains,
        },
    )

    del model, processor
    torch.cuda.empty_cache()
    return per_domain


def print_table(table: dict[tuple[str, str], dict[str, Metrics]], domains: list[str]) -> None:
    """One row per (teacher, prompt source); accuracy / F1 per domain plus summary."""
    header = (
        f"{'teacher':<10} {'prompts':<12} "
        + " ".join(f"{'d' + d:>13}" for d in domains)
        + f" {'mean':>13} {'worst':>13}"
    )
    print("\n" + "=" * len(header))
    print("Zero-shot results  (accuracy / F1, %)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for (vlm, source), per_domain in table.items():
        cells = []
        for domain in domains:
            metrics = per_domain.get(domain)
            cells.append(
                f"{metrics.accuracy * 100:5.2f}/{metrics.f1 * 100:5.2f}" if metrics else " " * 11
            )
        summary = summarise(per_domain)
        row = (
            f"{vlm:<10} {source:<12} "
            + " ".join(f"{c:>13}" for c in cells)
            + f" {summary['mean_accuracy'] * 100:5.2f}/{summary['mean_f1'] * 100:5.2f}".rjust(14)
            + f" {summary['worst_accuracy'] * 100:5.2f}/{summary['worst_f1'] * 100:5.2f}".rjust(14)
        )
        print(row)
    print("=" * len(header) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    args.data_root = str(resolve(args.data_root))
    args.prompt_root = str(resolve(args.prompt_root))
    args.vlm = expand_choices(args.vlm, VLM_CHOICES)
    args.prompt_source = expand_choices(args.prompt_source, list(PROMPT_SOURCES))
    print_config(args, "Zero-shot inference")

    spec = get_dataset_spec(args.dataset)
    device = select_device(args.gpu_index)

    eval_domains = args.domains or spec.domains
    prompt_domains = args.prompt_domains or [d for d in spec.domains if d != spec.val_domain]

    results_dir = (
        resolve(args.results_dir) if args.results_dir else resolve(args.output_root) / "zero_shot"
    )
    logger = get_logger("dipt.zeroshot", results_dir / "zero_shot.log")

    table: dict[tuple[str, str], dict[str, Metrics]] = {}
    for vlm in args.vlm:
        for prompt_source in args.prompt_source:
            try:
                table[(vlm, prompt_source)] = run_one(
                    args, spec, vlm, prompt_source, eval_domains, prompt_domains, device, logger
                )
            except FileNotFoundError as exc:
                # A missing stage-1 checkpoint should not abort the whole sweep.
                logger.warning(f"skipping {vlm}/{prompt_source}: {exc}")

    if table:
        print_table(table, eval_domains)
        print(f"Per-configuration reports written to {results_dir}")
    else:
        raise SystemExit("No configuration produced results.")


if __name__ == "__main__":
    main()
