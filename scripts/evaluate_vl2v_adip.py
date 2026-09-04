#!/usr/bin/env python
"""Stage 3 - evaluate trained VL2V-ADiP models on held-out domains.

Loads the stage-2 checkpoints of a run directory, rebuilds the algorithm with the
same prompt source it was trained with, and reports per-split plus mean/worst-case
metrics on the target domains.

Examples
--------
    python scripts/evaluate_vl2v_adip.py \
        --run-dir outputs/vl2v_adip/camelyon17/plip/resnet50/dipt_k4

    python scripts/evaluate_vl2v_adip.py \
        --run-dir outputs/vl2v_adip/camelyon17/quiltnet/vlm/dipt_k3 \
        --vlm quiltnet --student vlm --prompt-source dipt -k 3 --use-last
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from dipt.cli import add_common_args, add_prompt_args, add_student_args  # noqa: E402
from dipt.data.loaders import build_loader  # noqa: E402
from dipt.data.registry import get_dataset_spec  # noqa: E402
from dipt.evaluation.evaluate import evaluate_algorithm  # noqa: E402
from dipt.evaluation.metrics import save_report, summarise  # noqa: E402
from dipt.methods.vl2v_adip import STAGE_INFERENCE, VL2VADiP, load_checkpoint  # noqa: E402
from dipt.models.students import build_student  # noqa: E402
from dipt.models.vlm import load_vlm  # noqa: E402
from dipt.paths import resolve  # noqa: E402
from dipt.prompts.learned import build_text_targets  # noqa: E402
from dipt.utils.misc import print_config, select_device  # noqa: E402

from scripts.evaluate_students import infer_domains  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 3: evaluate VL2V-ADiP models on held-out domains.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_args(parser)
    add_prompt_args(parser)
    add_student_args(parser)
    parser.set_defaults(student="resnet50")

    group = parser.add_argument_group("evaluation")
    group.add_argument("--run-dir", required=True, help="Directory holding the stage-2 checkpoints.")
    group.add_argument("--stage", type=int, default=2, help="Which stage's checkpoints to load.")
    group.add_argument("--use-last", action="store_true", help="Load *_last.pth instead of *_best.pth.")
    group.add_argument("--embed-dim", type=int, default=512, help="Projection output dimension.")
    group.add_argument("--lam", type=float, default=0.5, help="DFC trade-off (unused at inference).")
    group.add_argument("--test-domains", nargs="+", default=None, help="Override test domains.")
    group.add_argument("--results-file", default=None, help="Report path (default: <run-dir>/evaluation.json).")
    return parser


def load_run_config(run_dir: Path, args) -> None:
    """Adopt the settings recorded at training time unless overridden on the CLI."""
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    supplied = set(sys.argv)
    for key in ("vlm", "student", "prompt_source", "num_context_tokens", "embed_dim", "dataset"):
        flag = "--" + key.replace("_", "-")
        if flag not in supplied and key in config:
            setattr(args, key, config[key])
    print(f"Loaded training configuration from {config_path}")


def main() -> None:
    args = build_parser().parse_args()
    run_dir = resolve(args.run_dir)
    load_run_config(run_dir, args)
    args.data_root = str(resolve(args.data_root))
    args.prompt_root = str(resolve(args.prompt_root))
    print_config(args, "Stage 3 - VL2V-ADiP evaluation")

    spec = get_dataset_spec(args.dataset)
    device = select_device(args.gpu_index)

    suffix = "last" if args.use_last else "best"
    checkpoints = sorted(run_dir.glob(f"*_stage{args.stage}_{suffix}.pth"))
    if not checkpoints:
        raise SystemExit(f"No stage-{args.stage} '{suffix}' checkpoints found in {run_dir}")

    teacher_model, processor, _ = load_vlm(args.vlm, device=device, freeze=True)
    logit_scale = teacher_model.logit_scale.detach()

    per_split = {}
    for checkpoint in checkpoints:
        print(f"\n{'=' * 70}\n{checkpoint.name}\n{'=' * 70}")
        train_domains, _, test_domains = infer_domains(checkpoint.name, spec.domains)
        if args.test_domains:
            test_domains = list(args.test_domains)
        print(f"train domains: {train_domains} | test domains: {test_domains}")

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

        student = build_student(
            args.student, num_classes=spec.num_classes, vlm=args.vlm, pretrained=False
        )
        algorithm = VL2VADiP(
            stage=STAGE_INFERENCE,
            student=student,
            text_features=text_features,
            logit_scale=logit_scale,
            embed_dim=args.embed_dim,
            lam=args.lam,
        ).to(device)
        load_checkpoint(algorithm, checkpoint, device)

        for domain in test_domains:
            loader = build_loader(
                spec, [domain], args.batch_size, shuffle=False,
                data_root=args.data_root, num_workers=args.num_workers,
            )
            metrics, _ = evaluate_algorithm(algorithm, loader, device, desc=f"test d{domain}")
            per_split[f"{checkpoint.stem}|domain{domain}"] = metrics
            print(f"  domain {domain}: {metrics}")

        del algorithm, student
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
            "prompt_source": args.prompt_source,
            "stage": args.stage,
            "checkpoints": [str(c) for c in checkpoints],
        },
    )


if __name__ == "__main__":
    main()
