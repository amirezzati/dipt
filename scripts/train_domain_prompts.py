#!/usr/bin/env python
"""Stage 1 - learn domain-specific prompts (DIPT).

Trains K context tokens on ONE domain, anchored to the class-generic aggregated
template embedding. Repeat for every training domain; stage 2 averages the
resulting prompts into domain-invariant text embeddings.

Checkpoints land at a deterministic location so later stages find them
automatically::

    outputs/prompts/<dataset>/<vlm>/k<K>/domain<D>/best_prompt_learner.pth

Examples
--------
    # one domain
    python scripts/train_domain_prompts.py --vlm plip --domain 0 --num-context-tokens 4

    # every training domain of Camelyon17 (validation domain excluded)
    python scripts/train_domain_prompts.py --vlm quiltnet --domain all -k 3

    # explicit validation domain and longer schedule
    python scripts/train_domain_prompts.py --vlm plip --domain 2 --val-domain 1 \
        --num-epochs 3 --lr 5e-5 --score-weight 0.5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dipt.cli import add_common_args  # noqa: E402
from dipt.data.loaders import build_loader  # noqa: E402
from dipt.data.registry import get_dataset_spec  # noqa: E402
from dipt.methods.dipt import PromptTuningTrainer  # noqa: E402
from dipt.models.prompt_learner import PromptedCLIP  # noqa: E402
from dipt.models.vlm import load_vlm  # noqa: E402
from dipt.paths import resolve  # noqa: E402
from dipt.prompts.learned import prompt_run_dir  # noqa: E402
from dipt.prompts.templates import aggregate_template_features  # noqa: E402
from dipt.utils.logging import get_logger  # noqa: E402
from dipt.utils.misc import print_config, select_device, set_seed  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 1: train domain-specific prompts (DIPT).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_args(parser)

    group = parser.add_argument_group("prompt tuning")
    group.add_argument(
        "--domain",
        required=True,
        help="Training domain, or 'all' to loop over every non-validation domain.",
    )
    group.add_argument(
        "--val-domain",
        default=None,
        help="Validation domain (default: the dataset's reserved validation domain).",
    )
    group.add_argument(
        "--num-context-tokens", "-k", type=int, default=4, help="K learnable context tokens."
    )
    group.add_argument("--num-epochs", type=int, default=1, help="Training epochs.")
    group.add_argument("--lr", type=float, default=5e-5, help="Adam learning rate.")
    group.add_argument(
        "--score-weight",
        type=float,
        default=0.5,
        help="Weight of the prompt-drift penalty (KgCoOp term).",
    )
    group.add_argument(
        "--eval-interval",
        type=int,
        default=100,
        help="Validate every N batches (0 disables mid-epoch validation).",
    )
    group.add_argument(
        "--learnable-agg",
        action="store_true",
        help="Ablation: also train the class-generic aggregated token.",
    )
    return parser


def train_one_domain(args, spec, domain: str, val_domain: str) -> dict:
    device = select_device(args.gpu_index)
    set_seed(args.seed)

    output_dir = prompt_run_dir(
        spec.name, args.vlm, args.num_context_tokens, domain, args.prompt_root
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("dipt.prompts", output_dir / "train.log")
    logger.info(f"training prompt | dataset={spec.name} vlm={args.vlm} "
                f"k={args.num_context_tokens} domain={domain} val={val_domain}")

    model, processor, vlm_spec = load_vlm(args.vlm, device=device, freeze=True)

    train_loader = build_loader(
        spec, [domain], args.batch_size, shuffle=True,
        data_root=args.data_root, num_workers=args.num_workers,
    )
    val_loader = build_loader(
        spec, [val_domain], args.batch_size, shuffle=False,
        data_root=args.data_root, num_workers=args.num_workers,
    )
    logger.info(f"train patches: {len(train_loader.dataset):,} | val patches: {len(val_loader.dataset):,}")

    # Class-generic anchor: aggregated hand-written templates.
    agg_vector = aggregate_template_features(
        model, processor, spec.classnames, spec.name, device
    )
    logger.info(f"aggregated template embedding: {tuple(agg_vector.shape)}")

    prompted_clip = PromptedCLIP(
        class_names=spec.classnames,
        clip_model=model,
        processor=processor,
        num_context_tokens=args.num_context_tokens,
        agg_vector=agg_vector,
        learnable_agg=args.learnable_agg,
    )

    trainer = PromptTuningTrainer(
        model=prompted_clip,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        device=device,
        num_epochs=args.num_epochs,
        lr=args.lr,
        score_weight=args.score_weight,
        eval_interval=args.eval_interval,
        logger=logger,
    )
    trainer.save_setup(
        {
            "dataset": spec.name,
            "vlm": args.vlm,
            "vlm_hf_id": vlm_spec.hf_id,
            "train_domain": domain,
            "val_domain": val_domain,
            "learnable_agg": args.learnable_agg,
        }
    )

    started = time.time()
    result = trainer.run()
    elapsed = time.time() - started
    logger.info(f"finished in {int(elapsed // 3600)}h {int(elapsed % 3600 // 60)}m {int(elapsed % 60)}s")
    return result


def main() -> None:
    args = build_parser().parse_args()
    args.data_root = str(resolve(args.data_root))
    args.prompt_root = str(resolve(Path(args.output_root) / "prompts"))
    print_config(args, "Stage 1 - DIPT prompt tuning")

    spec = get_dataset_spec(args.dataset)
    val_domain = args.val_domain or spec.val_domain

    if args.domain == "all":
        domains = [d for d in spec.domains if d != val_domain]
    else:
        domains = [args.domain]

    for domain in domains:
        if domain == val_domain:
            print(f"Skipping domain {domain}: it is the validation domain.")
            continue
        print(f"\n{'=' * 70}\nDomain {domain} (validation: {val_domain})\n{'=' * 70}")
        train_one_domain(args, spec, domain, val_domain)


if __name__ == "__main__":
    main()
