#!/usr/bin/env python
"""Stage 2 - VL2V-ADiP distillation with aggregated learned prompts.

Runs both VL2V-ADiP stages back to back for every leave-one-domain-out split:

    stage 1  train the projection layer only (student encoder frozen)
    stage 2  resume from stage 1 and fine-tune the whole student encoder

Test-set evaluation is a separate step (``scripts/evaluate_vl2v_adip.py``).

Prompt source
-------------
``--prompt-source dipt``        aggregated domain-invariant DIPT prompts (ours)
``--prompt-source handcrafted`` one prompt per class (original VL2V-ADiP)
``--prompt-source template``    aggregated hand-written templates

Student
-------
``--student resnet50`` / ``vit_base``  the students reported in the paper
``--student vlm``                      self-distillation from the teacher's own
                                       image encoder

Checkpoints::

    outputs/vl2v_adip/<dataset>/<vlm>/<student>/<prompt-source>_k<K>/
        t_<train>_v_<val>_stage1_best.pth
        t_<train>_v_<val>_stage2_best.pth

Examples
--------
    python scripts/train_vl2v_adip.py --vlm plip --student resnet50 --prompt-source dipt -k 4
    python scripts/train_vl2v_adip.py --vlm quiltnet --student vlm --prompt-source dipt -k 3
    python scripts/train_vl2v_adip.py --vlm plip --student vit_base --prompt-source handcrafted
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from dipt.cli import (  # noqa: E402
    add_common_args,
    add_prompt_args,
    add_student_args,
    parse_domain_groups,
)
from dipt.data.datasets import CyclicDataLoader  # noqa: E402
from dipt.data.loaders import build_loader  # noqa: E402
from dipt.data.registry import get_dataset_spec  # noqa: E402
from dipt.evaluation.evaluate import evaluate_algorithm  # noqa: E402
from dipt.methods.vl2v_adip import (  # noqa: E402
    STAGE_TRAIN_ENCODER,
    STAGE_TRAIN_PROJECTION,
    VL2VADiP,
    checkpoint_name,
    load_checkpoint,
    save_checkpoint,
)
from dipt.models.students import FrozenTeacher, build_student  # noqa: E402
from dipt.models.vlm import load_vlm  # noqa: E402
from dipt.paths import resolve  # noqa: E402
from dipt.prompts.learned import build_text_targets  # noqa: E402
from dipt.utils.logging import get_logger  # noqa: E402
from dipt.utils.misc import (  # noqa: E402
    count_parameters,
    print_config,
    run_name,
    save_json,
    select_device,
    set_seed,
    to_row,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 2: VL2V-ADiP distillation with aggregated learned prompts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_args(parser)
    add_prompt_args(parser)
    add_student_args(parser)
    parser.set_defaults(student="resnet50", batch_size=128)

    group = parser.add_argument_group("vl2v-adip")
    group.add_argument(
        "--train-doms",
        nargs="+",
        default=None,
        help="Comma-separated domain groups (default: the dataset's splits).",
    )
    group.add_argument("--val-domain", default=None, help="Validation domain.")
    group.add_argument("--steps", type=int, default=2500, help="Optimisation steps per stage.")
    group.add_argument("--lr", type=float, default=5e-5, help="Adam learning rate.")
    group.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay.")
    group.add_argument(
        "--lam",
        type=float,
        default=0.5,
        help="Trade-off between the image and text alignment terms of the DFC loss.",
    )
    group.add_argument("--embed-dim", type=int, default=512, help="Projection output dimension.")
    group.add_argument("--checkpoint-freq", type=int, default=200, help="Validate every N steps.")
    group.add_argument(
        "--save-after",
        type=int,
        default=200,
        help="Do not save a best checkpoint before this step.",
    )
    group.add_argument(
        "--stages",
        nargs="+",
        type=int,
        default=[STAGE_TRAIN_PROJECTION, STAGE_TRAIN_ENCODER],
        choices=[STAGE_TRAIN_PROJECTION, STAGE_TRAIN_ENCODER],
        help="Which stages to run (default: 1 2).",
    )
    return parser


def experiment_dir(args, spec) -> Path:
    tag = args.prompt_source
    if args.prompt_source == "dipt":
        tag = f"dipt_k{args.num_context_tokens}"
    return resolve(args.output_root) / "vl2v_adip" / spec.name / args.vlm / args.student / tag


def build_algorithm(args, spec, text_features, logit_scale, stage, device) -> VL2VADiP:
    student = build_student(
        args.student,
        num_classes=spec.num_classes,
        vlm=args.vlm,
        pretrained=not args.no_pretrained,
    )
    algorithm = VL2VADiP(
        stage=stage,
        student=student,
        text_features=text_features,
        logit_scale=logit_scale,
        embed_dim=args.embed_dim,
        lam=args.lam,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    return algorithm.to(device)


def train_stage(
    args, spec, stage: int, train_domains: list[str], val_domain: str, out_dir: Path,
    text_features, teacher, logit_scale, device, logger,
) -> dict:
    name = run_name(train_domains, val_domain)
    logger.info(f"===== stage {stage} | {name} =====")

    algorithm = build_algorithm(args, spec, text_features, logit_scale, stage, device)

    if stage == STAGE_TRAIN_ENCODER:
        previous = out_dir / checkpoint_name(name, STAGE_TRAIN_PROJECTION, best=True)
        if not previous.exists():
            previous = out_dir / checkpoint_name(name, STAGE_TRAIN_PROJECTION, best=False)
        logger.info(f"resuming from stage 1 checkpoint {previous}")
        load_checkpoint(algorithm, previous, device)

    logger.info(f"trainable parameters: {count_parameters(algorithm, trainable_only=True):,}")

    train_loader = build_loader(
        spec, train_domains, args.batch_size, shuffle=True,
        data_root=args.data_root, num_workers=args.num_workers,
    )
    val_loader = build_loader(
        spec, [val_domain], args.batch_size, shuffle=False,
        data_root=args.data_root, num_workers=args.num_workers,
    )

    train_iter = CyclicDataLoader(train_loader)
    records: list[dict] = []
    best_val_acc = 0.0
    header_written = False

    algorithm.train()
    for step in tqdm(range(1, args.steps + 1), desc=f"stage {stage}", unit="step"):
        batch = next(train_iter)
        images, labels = batch[0].to(device), batch[1].to(device)
        stats = algorithm.update(images, labels, teacher)

        if step % args.checkpoint_freq == 0:
            metrics, val_loss = evaluate_algorithm(algorithm, val_loader, device, desc="val")
            record = {
                "step": step,
                "epoch": step / max(len(train_loader), 1),
                "train_loss": stats["loss"],
                "val_loss": val_loss,
                "val_acc": metrics.accuracy,
                "val_precision": metrics.precision,
                "val_recall": metrics.recall,
                "val_f1": metrics.f1,
            }
            records.append(record)

            if not header_written:
                logger.info(to_row(list(record.keys())))
                header_written = True
            logger.info(to_row(list(record.values())))

            if metrics.accuracy > best_val_acc and step >= args.save_after:
                best_val_acc = metrics.accuracy
                path = out_dir / checkpoint_name(name, stage, best=True)
                save_checkpoint(
                    algorithm,
                    path,
                    meta={
                        "stage": stage,
                        "train_domains": train_domains,
                        "val_domain": val_domain,
                        "val_acc": metrics.accuracy,
                        "args": {k: str(v) for k, v in vars(args).items()},
                    },
                )
                logger.info(f"new best (val acc {metrics.accuracy:.4f}) -> {path}")

    last_path = out_dir / checkpoint_name(name, stage, best=False)
    save_checkpoint(
        algorithm,
        last_path,
        meta={"stage": stage, "train_domains": train_domains, "val_domain": val_domain},
    )
    logger.info(f"last checkpoint -> {last_path}")

    save_json(records, out_dir / f"{name}_stage{stage}_records.json")

    del algorithm
    torch.cuda.empty_cache()
    return {"best_val_acc": best_val_acc, "records": records}


def main() -> None:
    args = build_parser().parse_args()
    args.data_root = str(resolve(args.data_root))
    args.prompt_root = str(resolve(args.prompt_root))
    print_config(args, "Stage 2 - VL2V-ADiP distillation")

    spec = get_dataset_spec(args.dataset)
    val_domain = args.val_domain or spec.val_domain
    device = select_device(args.gpu_index)
    set_seed(args.seed)

    out_dir = experiment_dir(args, spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), out_dir / "config.json")
    logger = get_logger("dipt.vl2v", out_dir / "train.log")

    teacher_model, processor, _ = load_vlm(args.vlm, device=device, freeze=True)
    logit_scale = teacher_model.logit_scale.detach()

    if args.train_doms:
        splits = [(group, None) for group in parse_domain_groups(args.train_doms)]
    else:
        splits = list(spec.splits)

    results: dict[str, dict] = {}
    for train_domains, test_domain in splits:
        header = f"train {train_domains} | val {val_domain}"
        if test_domain:
            header += f" | test {test_domain}"
        print(f"\n{'=' * 70}\n{header}\n{'=' * 70}")

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

        name = run_name(train_domains, val_domain)
        results[name] = {}
        for stage in args.stages:
            results[name][f"stage{stage}"] = train_stage(
                args, spec, stage, train_domains, val_domain, out_dir,
                text_features, teacher, logit_scale, device, logger,
            )

    save_json(results, out_dir / "training_summary.json")
    print(f"\nAll splits finished. Checkpoints and logs in {out_dir}")
    print(f"Next: python scripts/evaluate_vl2v_adip.py --run-dir {out_dir}")


if __name__ == "__main__":
    main()
