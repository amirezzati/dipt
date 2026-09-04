#!/usr/bin/env python
"""Evaluate RISE students (including self-distilled ones) on held-out domains.

For every ``t_<train>_v_<val>_best.pth`` checkpoint in a run directory, the test
domain is inferred as the complement of the train and validation domains, and the
student is evaluated there. Mean and worst-case accuracy/F1 across splits are
reported, matching Table 2 of the paper.

Examples
--------
    python scripts/evaluate_students.py \
        --run-dir outputs/rise/camelyon17/plip/resnet50_bit/dipt_k4 \
        --vlm plip --student resnet50_bit

    # a single checkpoint on an explicit test domain
    python scripts/evaluate_students.py \
        --checkpoints outputs/rise/.../t_0_2_3_v_1_best.pth \
        --test-domains 4 --vlm quiltnet --student vlm
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from dipt.cli import add_common_args, add_student_args  # noqa: E402
from dipt.data.loaders import build_loader  # noqa: E402
from dipt.data.registry import get_dataset_spec  # noqa: E402
from dipt.evaluation.evaluate import evaluate_student  # noqa: E402
from dipt.evaluation.metrics import save_report, summarise  # noqa: E402
from dipt.models.students import build_student  # noqa: E402
from dipt.models.vlm import get_vlm_spec  # noqa: E402
from dipt.paths import resolve  # noqa: E402
from dipt.utils.misc import print_config, select_device  # noqa: E402

RUN_NAME_RE = re.compile(r"t_(?P<train>[\d_]+?)_v_(?P<val>\d+)")


def infer_domains(checkpoint_name: str, all_domains: list[str]) -> tuple[list[str], str, list[str]]:
    """Recover (train, val, test) domains from a ``t_0_2_3_v_1_best.pth`` filename."""
    match = RUN_NAME_RE.search(checkpoint_name)
    if not match:
        raise ValueError(
            f"Cannot parse domains from '{checkpoint_name}'. "
            "Pass --test-domains explicitly."
        )
    train = [d for d in match.group("train").split("_") if d]
    val = match.group("val")
    test = [d for d in all_domains if d not in train and d != val]
    return train, val, test


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate RISE / self-distilled students on held-out domains.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_args(parser)
    add_student_args(parser)

    group = parser.add_argument_group("evaluation")
    group.add_argument("--run-dir", default=None, help="Directory of *_best.pth checkpoints.")
    group.add_argument("--checkpoints", nargs="+", default=None, help="Explicit checkpoint paths.")
    group.add_argument(
        "--test-domains",
        nargs="+",
        default=None,
        help="Override the inferred test domains (applies to every checkpoint).",
    )
    group.add_argument(
        "--use-last",
        action="store_true",
        help="Evaluate *_last.pth instead of *_best.pth when scanning --run-dir.",
    )
    group.add_argument(
        "--results-file",
        default=None,
        help="Where to write the report (default: <run-dir>/evaluation.json).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.data_root = str(resolve(args.data_root))
    print_config(args, "Evaluation - RISE students")

    spec = get_dataset_spec(args.dataset)
    device = select_device(args.gpu_index)

    if args.checkpoints:
        checkpoints = [resolve(p) for p in args.checkpoints]
        run_dir = checkpoints[0].parent
    elif args.run_dir:
        run_dir = resolve(args.run_dir)
        pattern = "*_last.pth" if args.use_last else "*_best.pth"
        checkpoints = sorted(run_dir.glob(pattern))
    else:
        raise SystemExit("Pass either --run-dir or --checkpoints.")

    if not checkpoints:
        raise SystemExit(f"No checkpoints found in {run_dir}")

    per_split = {}
    for checkpoint in checkpoints:
        print(f"\n{'=' * 70}\n{checkpoint.name}\n{'=' * 70}")

        if args.test_domains:
            test_domains = list(args.test_domains)
        else:
            _, _, test_domains = infer_domains(checkpoint.name, spec.domains)
        print(f"test domains: {test_domains}")

        student = build_student(
            args.student,
            num_classes=spec.num_classes,
            vlm=args.vlm,
            pretrained=False,
            embed_dim=get_vlm_spec(args.vlm).embed_dim,
        ).to(device)

        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        state_dict = payload.get("student_state_dict", payload)
        student.load_state_dict(state_dict)

        for domain in test_domains:
            loader = build_loader(
                spec, [domain], args.batch_size, shuffle=False,
                data_root=args.data_root, num_workers=args.num_workers,
            )
            metrics = evaluate_student(student, loader, device, desc=f"test d{domain}")
            key = f"{checkpoint.stem}|domain{domain}"
            per_split[key] = metrics
            print(f"  domain {domain}: {metrics}")

        del student
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    for key, value in summarise(per_split).items():
        print(f"{key:16s}: {value * 100:.2f}")
    print("=" * 70)

    results_file = (
        resolve(args.results_file) if args.results_file else run_dir / "evaluation.json"
    )
    save_report(
        results_file,
        per_split,
        meta={
            "dataset": spec.name,
            "vlm": args.vlm,
            "student": args.student,
            "checkpoints": [str(c) for c in checkpoints],
        },
    )


if __name__ == "__main__":
    main()
