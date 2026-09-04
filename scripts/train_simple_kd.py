#!/usr/bin/env python
"""Stage 2 baseline - simple image-encoder-only KD ("KD" row in the paper's table).

No prompts, no text encoder: the student is distilled directly from the
teacher's own image embedding (MSE) plus the usual classification loss. Runs
the same leave-one-domain-out protocol as train_rise.py, and its checkpoints
use the same layout, so scripts/evaluate_students.py works unchanged.

Checkpoints::

    outputs/simple_kd/<dataset>/<vlm>/<student>/t_<train>_v_<val>_best.pth

Examples
--------
    python scripts/train_simple_kd.py --vlm plip --student resnet50_bit
    python scripts/train_simple_kd.py --vlm quiltnet --student vit_base
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from dipt.cli import add_common_args, add_student_args, parse_domain_groups  # noqa: E402
from dipt.data.loaders import build_loader  # noqa: E402
from dipt.data.registry import get_dataset_spec  # noqa: E402
from dipt.methods.simple_kd import SimpleKDTrainer  # noqa: E402
from dipt.models.students import FrozenTeacher, build_student  # noqa: E402
from dipt.models.vlm import load_vlm  # noqa: E402
from dipt.paths import resolve  # noqa: E402
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
        description="Stage 2 baseline: simple image-encoder-only KD, no prompts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_args(parser)
    add_student_args(parser)

    group = parser.add_argument_group("simple-kd")
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
    group.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    group.add_argument(
        "--optimizer", default="sgd", choices=["sgd", "adam"], help="Optimizer."
    )
    group.add_argument(
        "--distill-weight", type=float, default=0.5,
        help="Weight of the MSE-to-teacher-image-feature term.",
    )
    group.add_argument(
        "--classification-weight", type=float, default=0.5, help="Weight of the CE term."
    )
    group.add_argument("--eval-interval", type=int, default=200, help="Validate every N batches.")
    return parser


def experiment_dir(args, spec) -> Path:
    return resolve(args.output_root) / "simple_kd" / spec.name / args.vlm / args.student


def train_split(args, spec, train_domains: list[str], val_domain: str, out_dir: Path) -> dict:
    device = select_device(args.gpu_index)
    set_seed(args.seed)

    name = run_name(train_domains, val_domain)
    logger = get_logger("dipt.simple_kd", out_dir / f"{name}.log")
    logger.info(
        f"simple-kd | dataset={spec.name} vlm={args.vlm} student={args.student} "
        f"train={train_domains} val={val_domain}"
    )

    teacher_model, processor, vlm_spec = load_vlm(args.vlm, device=device, freeze=True)
    teacher = FrozenTeacher(teacher_model, processor, text_features=None).to(device)

    student = build_student(
        args.student,
        num_classes=spec.num_classes,
        vlm=args.vlm,
        pretrained=not args.no_pretrained,
        embed_dim=vlm_spec.embed_dim,
    )

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

    trainer = SimpleKDTrainer(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        val_loaders=val_loaders,
        device=device,
        output_dir=out_dir,
        run_name=name,
        epochs=args.epochs,
        lr=args.lr,
        optimizer_name=args.optimizer,
        distill_weight=args.distill_weight,
        classification_weight=args.classification_weight,
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
    print_config(args, "Stage 2 - Simple KD (image-encoder-only baseline)")

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
