#!/usr/bin/env python
"""Download and prepare the datasets used in this repository.

Both datasets end up in the same ``<root>/<domain>/<class>/<image>`` layout that
:class:`dipt.data.datasets.MultipleDomainDataset` expects.

Camelyon17-WILDS
----------------
The official WILDS release ships one flat patch directory plus ``metadata.csv``
with a ``center`` column (the hospital the slide came from) and a ``tumor``
label. **We deliberately ignore the official train/val/test split** and instead
group patches by *center*, because domain generalisation here means holding out
an entire hospital::

    data/camelyon17/0/normal/patch_patient_004_node_4_x_1234_y_5678.png
    data/camelyon17/0/tumor/...
    data/camelyon17/1/...            <- reserved for validation (see registry)

Kather19
--------
Downloads the Zenodo archives (NCT-CRC-HE-100K and CRC-VAL-HE-7K) into
``data/kather19/raw``. Because Kather19 has no per-center metadata, the
pseudo-domains used in the paper are produced by stain augmentation - run
``scripts/prepare_kather_domains.py`` afterwards.

Examples
--------
    python scripts/download_datasets.py --dataset camelyon17
    python scripts/download_datasets.py --dataset camelyon17 --max-per-class-per-center 20000
    python scripts/download_datasets.py --dataset kather19
    python scripts/download_datasets.py --dataset all --link-mode copy
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dipt.paths import DATA_ROOT, ensure_dir, resolve  # noqa: E402

CAMELYON17_CLASS_NAMES = {0: "normal", 1: "tumor"}

KATHER19_URLS = {
    "NCT-CRC-HE-100K.zip": "https://zenodo.org/records/1214456/files/NCT-CRC-HE-100K.zip",
    "CRC-VAL-HE-7K.zip": "https://zenodo.org/records/1214456/files/CRC-VAL-HE-7K.zip",
}


# --------------------------------------------------------------------- helpers


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    """Materialise ``dst`` from ``src`` using hardlink / symlink / copy.

    Hardlinks are the default: instant and no extra disk usage, but they require
    both paths to sit on the same filesystem. We fall back to copying otherwise.
    """
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if mode == "hardlink":
            os.link(src, dst)
        elif mode == "symlink":
            os.symlink(src, dst)
        else:
            shutil.copy2(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def download_file(url: str, dst: Path) -> Path:
    """Stream ``url`` to ``dst`` with a simple progress readout."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"  already downloaded: {dst.name}")
        return dst

    print(f"  downloading {url}")
    tmp = dst.with_suffix(dst.suffix + ".part")

    def hook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            done = min(block_num * block_size, total_size)
            pct = 100.0 * done / total_size
            print(f"\r    {done / 1e6:8.1f} / {total_size / 1e6:.1f} MB ({pct:5.1f}%)", end="")

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    print()
    tmp.rename(dst)
    return dst


def extract_zip(archive: Path, target_dir: Path) -> None:
    marker = target_dir / f".{archive.stem}.extracted"
    if marker.exists():
        print(f"  already extracted: {archive.name}")
        return
    print(f"  extracting {archive.name} -> {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target_dir)
    marker.touch()


# ------------------------------------------------------------------ camelyon17


def download_camelyon17(raw_dir: Path) -> Path:
    """Fetch the WILDS release via the ``wilds`` package."""
    try:
        from wilds import get_dataset
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "The 'wilds' package is required to download Camelyon17.\n"
            "  pip install wilds\n"
            f"(import error: {exc})"
        ) from exc

    ensure_dir(raw_dir)
    print(f"Downloading Camelyon17-WILDS into {raw_dir} (~10 GB, this takes a while)...")
    get_dataset(dataset="camelyon17", download=True, root_dir=str(raw_dir))
    return raw_dir / "camelyon17_v1.0"


def prepare_camelyon17(
    source_dir: Path,
    target_dir: Path,
    link_mode: str = "hardlink",
    max_per_class_per_center: int | None = None,
) -> None:
    """Group WILDS patches by hospital (``center``) and tumour label."""
    import pandas as pd

    metadata_path = source_dir / "metadata.csv"
    if not metadata_path.exists():
        raise SystemExit(
            f"metadata.csv not found under {source_dir}.\n"
            "Run with --download first, or point --raw-dir at an existing WILDS copy."
        )

    metadata = pd.read_csv(metadata_path, index_col=0)
    required = {"patient", "node", "x_coord", "y_coord", "tumor", "center"}
    missing = required - set(metadata.columns)
    if missing:
        raise SystemExit(
            f"{metadata_path} is missing the column(s) {sorted(missing)}.\n"
            f"Found: {list(metadata.columns)}\n"
            "This script expects the Camelyon17-WILDS v1.0 metadata format."
        )

    print(f"Loaded {len(metadata):,} patch records from {metadata_path}")
    print("Splitting by the 'center' column (hospital), NOT the official train/val/test split.")

    counts: dict[tuple[int, int], int] = {}
    written = 0

    for row in metadata.itertuples(index=False):
        center = int(row.center)
        label = int(row.tumor)

        key = (center, label)
        if max_per_class_per_center is not None and counts.get(key, 0) >= max_per_class_per_center:
            continue

        patch_dir = f"patient_{int(row.patient):03d}_node_{int(row.node)}"
        filename = (
            f"patch_patient_{int(row.patient):03d}_node_{int(row.node)}"
            f"_x_{int(row.x_coord)}_y_{int(row.y_coord)}.png"
        )
        src = source_dir / "patches" / patch_dir / filename
        if not src.exists():
            continue

        dst = target_dir / str(center) / CAMELYON17_CLASS_NAMES[label] / filename
        link_or_copy(src, dst, link_mode)

        counts[key] = counts.get(key, 0) + 1
        written += 1
        if written % 50_000 == 0:
            print(f"  {written:,} patches placed...")

    print(f"\nPrepared Camelyon17 at {target_dir}")
    for (center, label), count in sorted(counts.items()):
        print(f"  center {center} / {CAMELYON17_CLASS_NAMES[label]:6s}: {count:,} patches")
    print(
        "\nDomain indices map directly to WILDS center ids 0-4. "
        "The dataset registry reserves domain 1 for validation."
    )


# -------------------------------------------------------------------- kather19


def download_kather19(raw_dir: Path) -> Path:
    ensure_dir(raw_dir)
    print(f"Downloading Kather19 archives into {raw_dir} (~11 GB)...")
    for name, url in KATHER19_URLS.items():
        archive = download_file(url, raw_dir / name)
        extract_zip(archive, raw_dir)
    return raw_dir


def report_kather19(raw_dir: Path) -> None:
    train_dir = raw_dir / "NCT-CRC-HE-100K"
    print(f"\nKather19 raw data ready under {raw_dir}")
    if train_dir.is_dir():
        classes = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
        print(f"  NCT-CRC-HE-100K classes: {classes}")
    print(
        "\nKather19 ships no per-center metadata, so the paper's pseudo-domains are\n"
        "produced by stain augmentation. Next step:\n\n"
        "  python scripts/prepare_kather_domains.py --num-domains 6\n"
    )


# ------------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare datasets for the DIPT pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        default="camelyon17",
        choices=["camelyon17", "kather19", "all"],
        help="Which dataset to fetch (default: camelyon17).",
    )
    parser.add_argument(
        "--data-root",
        default=str(DATA_ROOT),
        help="Destination root for prepared data (default: <repo>/data).",
    )
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="Where raw downloads live (default: <data-root>/raw/<dataset>).",
    )
    parser.add_argument(
        "--link-mode",
        default="hardlink",
        choices=["hardlink", "symlink", "copy"],
        help="How prepared patches reference the raw files (default: hardlink, no extra disk).",
    )
    parser.add_argument(
        "--max-per-class-per-center",
        type=int,
        default=None,
        help="Subsample Camelyon17 to at most N patches per class per center (default: use all).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only run the preparation step on an already-downloaded copy.",
    )
    args = parser.parse_args()

    data_root = ensure_dir(resolve(args.data_root))
    targets = ["camelyon17", "kather19"] if args.dataset == "all" else [args.dataset]

    for dataset in targets:
        print("\n" + "=" * 70)
        print(f"Dataset: {dataset}")
        print("=" * 70)

        raw_dir = (
            resolve(args.raw_dir) if args.raw_dir else data_root / "raw" / dataset
        )

        if dataset == "camelyon17":
            source_dir = raw_dir / "camelyon17_v1.0"
            if not args.skip_download:
                source_dir = download_camelyon17(raw_dir)
            prepare_camelyon17(
                source_dir=source_dir,
                target_dir=ensure_dir(data_root / "camelyon17"),
                link_mode=args.link_mode,
                max_per_class_per_center=args.max_per_class_per_center,
            )
        else:
            if not args.skip_download:
                download_kather19(raw_dir)
            report_kather19(raw_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
