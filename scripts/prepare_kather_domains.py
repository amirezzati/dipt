#!/usr/bin/env python
"""Build Kather19 pseudo-domains by stain augmentation.

Kather19 has no per-center metadata, so - as in the thesis experiments - the
domains are synthesised: each source patch is re-stained with a different
randomly sampled (alpha, beta) pair in Macenko/Vahadane stain-concentration
space, and each pair defines one pseudo-center.

Output layout (what ``MultipleDomainDataset`` expects)::

    data/kather19/0/ADI/ADI-....tif
    data/kather19/0/TUM/...
    data/kather19/1/...

Requires ``staintools`` (``pip install staintools spams``); it is not part of the
core requirements because only this script needs it.

Examples
--------
    python scripts/prepare_kather_domains.py --num-domains 6
    python scripts/prepare_kather_domains.py --num-domains 6 --max-per-class 2000 --method macenko
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dipt.data.registry import KATHER19  # noqa: E402
from dipt.paths import DATA_ROOT, ensure_dir, resolve  # noqa: E402


def _require_staintools():
    try:
        import numpy
        # spams (an unmaintained staintools dependency) still calls the
        # `np.bool` alias, removed in numpy>=1.24; shim it before it's used
        # rather than pin numpy down for the whole environment.
        if not hasattr(numpy, "bool"):
            numpy.bool = bool
        import staintools
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "staintools is required for Kather19 domain synthesis.\n"
            "  pip install staintools spams\n"
            f"(import error: {exc})"
        ) from exc
    return staintools


class MultiDomainStainAugmentor:
    """Stain augmentor that keeps one fixed (alpha, beta) pair per pseudo-domain.

    ``staintools.StainAugmentor`` resamples the perturbation on every call, which
    would make each image its own domain. Sampling the pairs once up front makes
    a domain a consistent, reproducible stain style across the whole dataset.
    """

    def __init__(
        self,
        method: str,
        num_domains: int,
        sigma1: float = 0.4,
        sigma2: float = 0.4,
        augment_background: bool = False,
        seed: int = 0,
    ) -> None:
        staintools = _require_staintools()
        self._augmentor = staintools.StainAugmentor(
            method=method,
            sigma1=sigma1,
            sigma2=sigma2,
            augment_background=augment_background,
        )
        rng = np.random.RandomState(seed)
        self.alphas = [rng.uniform(1 - sigma1, 1 + sigma1) for _ in range(num_domains)]
        self.betas = [rng.uniform(-sigma2, sigma2) for _ in range(num_domains)]
        self.augment_background = augment_background

    def fit(self, image: np.ndarray) -> None:
        self._augmentor.fit(image)

    def transform(self, domain: int) -> np.ndarray:
        """Re-stain the fitted image with the (alpha, beta) of ``domain``."""
        aug = self._augmentor
        alpha, beta = self.alphas[domain], self.betas[domain]

        concentrations = copy.deepcopy(aug.source_concentrations)
        for stain in range(aug.n_stains):
            if self.augment_background:
                concentrations[:, stain] = concentrations[:, stain] * alpha + beta
            else:
                concentrations[aug.tissue_mask, stain] = (
                    concentrations[aug.tissue_mask, stain] * alpha + beta
                )

        image = 255 * np.exp(-1 * np.dot(concentrations, aug.stain_matrix))
        image = np.clip(image.reshape(aug.image_shape), 0, 255)
        return image.astype("uint8")


def augment_class(
    class_name: str,
    source_dir: Path,
    target_root: Path,
    augmentor: MultiDomainStainAugmentor,
    num_domains: int,
    max_per_class: int | None,
) -> int:
    staintools = _require_staintools()

    class_dir = source_dir / class_name
    if not class_dir.is_dir():
        print(f"  skipping {class_name}: {class_dir} not found")
        return 0

    for domain in range(num_domains):
        ensure_dir(target_root / str(domain) / class_name)

    files = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in {".tif", ".png", ".jpg"})
    if max_per_class is not None:
        files = files[:max_per_class]

    written = 0
    for image_path in tqdm(files, desc=f"  {class_name}", unit="img", leave=False):
        try:
            image = staintools.read_image(str(image_path))
            augmentor.fit(image)
        except Exception:
            # Blank/background patches can yield an empty tissue mask - skip them.
            continue

        for domain in range(num_domains):
            out_path = target_root / str(domain) / class_name / image_path.name
            if out_path.exists():
                continue
            Image.fromarray(augmentor.transform(domain), mode="RGB").save(out_path)
        written += 1

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesise Kather19 pseudo-domains by stain augmentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data-root", default=str(DATA_ROOT), help="Root holding data/.")
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Raw class folders (default: <data-root>/raw/kather19/NCT-CRC-HE-100K).",
    )
    parser.add_argument(
        "--target-dir",
        default=None,
        help="Output root (default: <data-root>/kather19).",
    )
    parser.add_argument("--num-domains", type=int, default=6, help="Number of pseudo-domains.")
    parser.add_argument(
        "--method",
        default="vahadane",
        choices=["vahadane", "macenko"],
        help="Stain extraction method (vahadane is slower but cleaner).",
    )
    parser.add_argument("--sigma1", type=float, default=0.4, help="Multiplicative jitter range.")
    parser.add_argument("--sigma2", type=float, default=0.4, help="Additive jitter range.")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=2000,
        help="Source patches per class (default: 2000; use 0 for all).",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=KATHER19.classnames,
        help="Class folders to process.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for the stain perturbations.")
    args = parser.parse_args()

    data_root = resolve(args.data_root)
    source_dir = (
        resolve(args.source_dir)
        if args.source_dir
        else data_root / "raw" / "kather19" / "NCT-CRC-HE-100K"
    )
    target_dir = ensure_dir(resolve(args.target_dir) if args.target_dir else data_root / "kather19")

    if not source_dir.is_dir():
        raise SystemExit(
            f"Source directory {source_dir} not found.\n"
            "Run: python scripts/download_datasets.py --dataset kather19"
        )

    max_per_class = None if args.max_per_class in (0, None) else args.max_per_class

    print(f"Source      : {source_dir}")
    print(f"Target      : {target_dir}")
    print(f"Domains     : {args.num_domains} ({args.method}, sigma1={args.sigma1}, sigma2={args.sigma2})")
    print(f"Max / class : {max_per_class or 'all'}\n")

    augmentor = MultiDomainStainAugmentor(
        method=args.method,
        num_domains=args.num_domains,
        sigma1=args.sigma1,
        sigma2=args.sigma2,
        seed=args.seed,
    )

    total = 0
    for class_name in args.classes:
        total += augment_class(
            class_name, source_dir, target_dir, augmentor, args.num_domains, max_per_class
        )

    print(f"\nDone: {total:,} source patches x {args.num_domains} domains -> {target_dir}")


if __name__ == "__main__":
    main()
