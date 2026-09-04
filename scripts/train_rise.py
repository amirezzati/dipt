#!/usr/bin/env python
"""Stage 2 - RISE knowledge distillation from a pathology VLM.

Runs the leave-one-domain-out protocol: for each split the student is trained on
the source domains, validated on the reserved validation domain, and later tested
on the held-out target domain (``scripts/evaluate_students.py``).

Prompt source
-------------
``--prompt-source dipt``      aggregated domain-invariant DIPT prompts (ours)
``--prompt-source template``  aggregated hand-written templates (original RISE)

Student
-------
``--student resnet50_bit``    ResNet-50 student, as reported in the paper
``--student vlm``             self-distillation: a trainable copy of the
                              teacher's own image encoder

Checkpoints::

    outputs/rise/<dataset>/<vlm>/<student>/<prompt-source>_k<K>/t_<train>_v_<val>_best.pth

Examples
--------
    # RISE + DIPT with a ResNet-50 student, all four Camelyon17 splits
    python scripts/train_rise.py --vlm plip --prompt-source dipt -k 4

    # self-distillation with QuiltNet
    python scripts/train_rise.py --vlm quiltnet --student vlm --prompt-source dipt -k 3

    # original RISE baseline (template prompts) on one split only
    python scripts/train_rise.py --vlm plip --prompt-source template --train-doms 0,2,3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from dipt.cli import (  # noqa: E402
    add_common_args,
    add_prompt_args,
    add_student_args,
    parse_domain_groups,
)
from dipt.data.loaders import build_loader  # noqa: E402
from dipt.data.registry import get_dataset_spec  # noqa: E402
from dipt.methods.rise import RiseTrainer  # noqa: E402
from dipt.models.students import FrozenTeacher, TimmStudent, build_student  # noqa: E402
from dipt.models.vlm import load_vlm  # noqa: E402
from dipt.paths import resolve  # noqa: E402
from dipt.prompts.learned import build_text_targets  # noqa: E402
from dipt.utils.logging import get_logger  # noqa: E402
from dipt.utils.misc import (  # noqa: E402
    print_config,
    run_name,
    save_json,
    select_device,
    set_seed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 2: RISE distillation with aggregated learned prompts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_args(parser)
    add_prompt_args(parser)
    add_student_args(parser)

    group = parser.add_argument_group("rise")
    group.add_argument(
        "--train-doms",
        nargs="+",
        default=None,
        help="Comma-separated domain groups, e.g. 0,2,3 0,2,4 (default: the dataset's splits).",
    )
    group.add_argument(
        "--val-domain",
        default=None,
        help="Validation domain (default: the dataset's reserved validation domain).",
    )
    group.add_argument("--epochs", type=int, default=1, help="Training epochs per split.")
    group.add_argument("--lr", type=float, default=8e-4, help="Learning rate.")
    group.add_argument(
        "--optimizer", default="sgd", choices=["sgd", "adam"], help="Optimizer (RISE uses SGD)."
    )
    group.add_argument("--distill-weight", type=float, default=0.3, help="Weight of the KL term.")
    group.add_argument(
        "--classification-weight", type=float, default=0.4, help="Weight of the CE term."
    )
    group.add_argument(
        "--distance-weight", type=float, default=0.3, help="Weight of the absolute-distance term."
    )
    group.add_argument("--temperature", "-T", type=float, default=2.0, help="Distillation temperature.")
    group.add_argument("--eval-interval", type=int, default=200, help="Validate every N batches.")
    group.add_argument(
        "--init-head-from-prompts",
        action="store_true",
        help=(
            "Initialise the student's linear head with the class text embeddings. "
            "Off by default, matching the original implementation."
        ),
    )
    return parser


def experiment_dir(args, spec) -> Path:
    tag = args.prompt_source
    if args.prompt_source == "dipt":
        tag = f"dipt_k{args.num_context_tokens}"
    return (
        resolve(args.output_root)
        / "rise"
        / spec.name
        / args.vlm
        / args.student
        / tag
    )


def train_split(args, spec, train_domains: list[str], val_domain: str, out_dir: Path) -> dict:
    device = select_device(args.gpu_index)
    set_seed(args.seed)

    name = run_name(train_domains, val_domain)
    logger = get_logger("dipt.rise", out_dir / f"{name}.log")
    logger.info(
        f"RISE | dataset={spec.name} vlm={args.vlm} student={args.student} "
        f"prompts={args.prompt_source} train={train_domains} val={val_domain}"
    )

    teacher_model, processor, _ = load_vlm(args.vlm, device=device, freeze=True)

    # Class text embeddings: DIPT prompts averaged over the training domains,
    # or the template/handcrafted baselines.
    text_features = build_text_targets(
        prompt_source=args.prompt_source,
        classnames=spec.classnames,
        model=teacher_model,
        processor=processor,
        dataset=spec.name,
        vlm=args.vlm,
        domains=train_domains,
        num_context_tokens=args.num_context_tokens,
        prompt_root=args.prompt_root,
        device=device,
    )
    logger.info(f"text targets ({args.prompt_source}): {tuple(text_features.shape)}")

    teacher = FrozenTeacher(teacher_model, processor, text_features).to(device)

    student = build_student(
        args.student,
        num_classes=spec.num_classes,
        vlm=args.vlm,
        pretrained=not args.no_pretrained,
        embed_dim=text_features.shape[-1],
    )
    if args.init_head_from_prompts:
        if isinstance(student, TimmStudent):
            student.init_head_from_text(text_features)
            logger.info("student head initialised from the class text embeddings")
        else:
            logger.warning("--init-head-from-prompts is only supported for timm students; ignored")

    train_loader = build_loader(
        spec, train_domains, args.batch_size, shuffle=True,
        data_root=args.data_root, num_workers=args.num_workers,
    )
    val_loaders = {
        val_domain: build_loader(
            spec, [val_domain], args.batch_size, shuffle=False,
            data_root=args.data_root, num_workers=args.num_workers,
        )
    }
    logger.info(f"train patches: {len(train_loader.dataset):,}")

    trainer = RiseTrainer(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        val_loaders=val_loaders,
        text_features=text_features,
        device=device,
        output_dir=out_dir,
        run_name=name,
        epochs=args.epochs,
        lr=args.lr,
        optimizer_name=args.optimizer,
        distill_weight=args.distill_weight,
        classification_weight=args.classification_weight,
        distance_weight=args.distance_weight,
        temperature=args.temperature,
        eval_interval=args.eval_interval,
        logger=logger,
    )
    result = trainer.run()

    del student, teacher, teacher_model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = build_parser().parse_args()
    args.data_root = str(resolve(args.data_root))
    args.prompt_root = str(resolve(args.prompt_root))
    print_config(args, "Stage 2 - RISE distillation")

    spec = get_dataset_spec(args.dataset)
    val_domain = args.val_domain or spec.val_domain

    if args.train_doms:
        splits = [(group, None) for group in parse_domain_groups(args.train_doms)]
    else:
        splits = [(train, test) for train, test in spec.splits]

    out_dir = experiment_dir(args, spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), out_dir / "config.json")

    results = {}
    for train_domains, test_domain in splits:
        header = f"train {train_domains} | val {val_domain}"
        if test_domain:
            header += f" | test {test_domain}"
        print(f"\n{'=' * 70}\n{header}\n{'=' * 70}")
        results[run_name(train_domains, val_domain)] = train_split(
            args, spec, train_domains, val_domain, out_dir
        )

    save_json(results, out_dir / "training_summary.json")
    print(f"\nAll splits finished. Checkpoints and logs in {out_dir}")
    print("Next: python scripts/evaluate_students.py --run-dir "
          f"{out_dir} --vlm {args.vlm} --student {args.student}")


if __name__ == "__main__":
    main()
