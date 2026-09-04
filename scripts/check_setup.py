#!/usr/bin/env python
"""Sanity-check the installation and the data layout before launching a run.

Verifies that every module imports, that the configured VLMs and students can be
named, and reports which dataset domains are actually present on disk. It never
downloads anything and needs no GPU.

    python scripts/check_setup.py
    python scripts/check_setup.py --dataset kather19 --data-root ./data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dipt.data.registry import DATASETS, get_dataset_spec  # noqa: E402
from dipt.models.students import STUDENTS  # noqa: E402
from dipt.models.vlm import VLMS  # noqa: E402
from dipt.paths import DATA_ROOT, OUTPUT_ROOT, PROJECT_ROOT, resolve  # noqa: E402
from dipt.prompts.templates import get_class_templates  # noqa: E402


def check_imports() -> list[str]:
    problems = []
    for module, hint in [
        ("torch", "pip install -r requirements.txt"),
        ("torchvision", "pip install -r requirements.txt"),
        ("transformers", "pip install -r requirements.txt"),
        ("sklearn", "pip install scikit-learn"),
        ("timm", "pip install timm (needed for ResNet/ViT students)"),
        ("pandas", "pip install pandas (needed by the download script)"),
    ]:
        try:
            __import__(module)
            print(f"  [ok]   {module}")
        except ImportError:
            print(f"  [MISS] {module} - {hint}")
            problems.append(module)

    try:
        import torch

        if torch.cuda.is_available():
            print(f"  [ok]   CUDA: {torch.cuda.device_count()} device(s), "
                  f"{torch.cuda.get_device_name(0)}")
        else:
            print("  [warn] CUDA not available - training will fall back to CPU")
    except ImportError:
        pass
    return problems


def check_data(dataset: str, data_root: Path) -> list[str]:
    spec = get_dataset_spec(dataset)
    root = spec.root(data_root)
    problems = []

    print(f"  dataset root: {root}")
    if not root.is_dir():
        print("  [MISS] not prepared - run scripts/download_datasets.py")
        return [f"{dataset} not prepared"]

    for domain in spec.domains:
        domain_dir = root / domain
        if not domain_dir.is_dir():
            print(f"  [MISS] domain {domain}: {domain_dir}")
            problems.append(f"{dataset} domain {domain}")
            continue
        per_class = {
            p.name: sum(1 for _ in p.iterdir())
            for p in sorted(domain_dir.iterdir())
            if p.is_dir()
        }
        total = sum(per_class.values())
        role = " (validation)" if domain == spec.val_domain else ""
        print(f"  [ok]   domain {domain}{role}: {total:,} images {per_class}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the DIPT setup.")
    parser.add_argument("--dataset", default="all", choices=[*sorted(DATASETS), "all"])
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    args = parser.parse_args()

    print(f"repository root : {PROJECT_ROOT}")
    print(f"data root       : {resolve(args.data_root)}")
    print(f"output root     : {OUTPUT_ROOT}\n")

    print("Dependencies")
    problems = check_imports()

    print("\nRegistered teachers (--vlm)")
    for key, spec in VLMS.items():
        print(f"  {key:10s} -> {spec.hf_id}")

    print("\nRegistered students (--student)")
    for key, spec in STUDENTS.items():
        print(f"  {key:14s} {spec.description}")

    datasets = sorted(DATASETS) if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        spec = get_dataset_spec(dataset)
        print(f"\nDataset '{dataset}' ({spec.num_classes} classes, {spec.num_domains} domains)")
        templates = get_class_templates(dataset, spec.classnames)
        missing = [c for c, t in templates.items() if not t]
        if missing:
            print(f"  [warn] no templates for {missing}")
        else:
            print(f"  [ok]   templates defined for all {len(templates)} classes")
        problems += check_data(dataset, resolve(args.data_root))

    print("\n" + "=" * 60)
    if problems:
        print(f"{len(problems)} issue(s) found:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)
    print("Setup looks good.")


if __name__ == "__main__":
    main()
