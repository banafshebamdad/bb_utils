#
# File: evaluation/segmentation_spotcheck.py
# Author: Banafshe Bamdad + GitHub Copilot (Claude Sonnet 4.6)
# Created: 2026-04-23 CET
#

#!/usr/bin/env python3
"""
Visual spot-check for bb-run-segmentation output.

Samples a small number of frames from a flat mask directory, overlays each
binary mask on its source image as a semi-transparent red highlight, and
writes the blended PNGs to an output directory.

Frames are drawn from four density buckets so both empty scenes and heavily
masked frames are always represented:
  zero   — mask density == 0   (no pedestrians detected)
  sparse — density in (0, 0.05]
  medium — density in (0.05, 0.30]
  dense  — density > 0.30      (heavily masked; worth checking for FP)

Each output filename is prefixed with its bucket and density value, e.g.:
  dense_d0.996_BIN_Hrsaal1B01_to_restroom_L_120528236.png

Usage
-----
    python evaluation/segmentation_spotcheck.py \
        --masks-dir  dataset/incrowdvi/semantic_masks/train \
        --images-dir dataset/incrowdvi/frames/train \
        --output-dir logs/mask_spotcheck

    # Custom sample size, seed, and overlay opacity
    python evaluation/segmentation_spotcheck.py \
        --masks-dir  dataset/incrowdvi/semantic_masks/train \
        --images-dir dataset/incrowdvi/frames/train \
        --output-dir logs/mask_spotcheck \
        --n-frames 40 --seed 123 --alpha 0.5

    # Process every mask file (no sampling)
    python evaluation/segmentation_spotcheck.py \
        --masks-dir  dataset/incrowdvi/semantic_masks/train \
        --images-dir dataset/incrowdvi/frames/train \
        --output-dir logs/mask_spotcheck_all \
        --all
"""

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def _load_rgb(image_path: Path) -> np.ndarray:
    """Load an image file as uint8 (H, W, 3) RGB, auto-converting grayscale."""
    try:
        import cv2
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"cv2 returned None for {image_path}")
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        return img[:, :, ::-1].astype(np.uint8)
    except ImportError:
        pass
    from PIL import Image
    return np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)


def _save_rgb(image_rgb: np.ndarray, path: Path) -> None:
    """Save a uint8 (H, W, 3) RGB array to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
        cv2.imwrite(str(path), image_rgb[:, :, ::-1])
        return
    except ImportError:
        pass
    from PIL import Image
    Image.fromarray(image_rgb).save(path)


def _load_preprocessing_cfg(masks_dir: Path) -> dict:
    """Return the ``preprocessing:`` section from the config YAML in *masks_dir*.

    ``bb-run-segmentation`` copies the active config into the output directory
    for reproducibility.  If a YAML file is found, this function parses it and
    returns its ``preprocessing:`` sub-dict (or ``{}`` when the key is absent).
    Returns an empty dict when no YAML file exists in *masks_dir*.
    """
    yaml_files = sorted(masks_dir.glob("*.yaml"))
    if not yaml_files:
        return {}
    import yaml
    cfg_path = yaml_files[0]
    if len(yaml_files) > 1:
        logger.debug(
            "Multiple YAML files in %s; using %s", masks_dir, cfg_path.name
        )
    try:
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh) or {}
        return cfg.get("preprocessing") or {}
    except Exception as exc:
        logger.warning("Could not parse config %s: %s", cfg_path, exc)
        return {}


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

def _overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """Blend a red highlight onto *image_rgb* wherever *mask* == 1.

    Args:
        image_rgb: uint8 (H, W, 3) source frame.
        mask:      uint8 (H, W) binary mask, values {0, 1}.
        alpha:     Opacity of the red overlay in [0, 1].

    Returns:
        uint8 (H, W, 3) blended image.
    """
    result = image_rgb.astype(np.float32)
    m = mask.astype(bool)
    red = np.array([220.0, 0.0, 0.0], dtype=np.float32)
    result[m] = result[m] * (1.0 - alpha) + red * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

_BUCKETS = [
    ("zero",   0.0,  0.0),
    ("sparse", 0.0,  0.05),
    ("medium", 0.05, 0.30),
    ("dense",  0.30, 1.01),
]


def _bucket_label(density: float) -> str:
    if density == 0.0:
        return "zero"
    if density <= 0.05:
        return "sparse"
    if density <= 0.30:
        return "medium"
    return "dense"


def _sample_frames(mask_files: list, n_frames: int, seed: int) -> list:
    """Return up to *n_frames* (mask_path, density) pairs spread across buckets.

    To avoid scanning the full dataset, a candidate pool of
    ``min(total, n_frames * 50)`` files is drawn first; densities are computed
    only for that pool.  Each of the four density buckets then gets
    n_frames // 4 samples; any shortfall is filled with random draws.
    """
    rng = np.random.default_rng(seed)
    per_bucket = max(1, n_frames // 4)

    # Pre-subsample to keep density scanning fast on large directories
    pool_size = min(len(mask_files), n_frames * 50)
    if pool_size < len(mask_files):
        idx = rng.choice(len(mask_files), size=pool_size, replace=False)
        pool = [mask_files[i] for i in idx]
    else:
        pool = list(mask_files)

    buckets: dict[str, list] = {b[0]: [] for b in _BUCKETS}
    logger.info("Computing mask densities for %d files (pool of %d) …", pool_size, len(mask_files))

    for f in pool:
        try:
            d = float(np.load(f)["mask"].mean())
        except Exception as exc:
            logger.warning("Could not load %s: %s", f.name, exc)
            continue
        buckets[_bucket_label(d)].append((f, d))

    logger.info(
        "Buckets — zero: %d  sparse: %d  medium: %d  dense: %d",
        len(buckets["zero"]), len(buckets["sparse"]),
        len(buckets["medium"]), len(buckets["dense"]),
    )

    selected = []
    for label, entries in buckets.items():
        if not entries:
            continue
        n = min(per_bucket, len(entries))
        idx = rng.choice(len(entries), size=n, replace=False)
        selected.extend(entries[i] for i in idx)

    # Fill remaining slots with random draws from unselected frames
    if len(selected) < n_frames:
        already = {f for f, _ in selected}
        pool = [(f, d) for b in buckets.values() for f, d in b if f not in already]
        extra = min(n_frames - len(selected), len(pool))
        if extra > 0:
            idx = rng.choice(len(pool), size=extra, replace=False)
            selected.extend(pool[i] for i in idx)

    return selected


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run_spotcheck(
    masks_dir: Path,
    images_dir: Path,
    output_dir: Path,
    n_frames: int,
    seed: int,
    alpha: float,
    all_frames: bool = False,
) -> None:
    mask_files = sorted(masks_dir.glob("*.npz"))
    if not mask_files:
        logger.error("No NPZ files found in %s", masks_dir)
        return

    # Read the preprocessing config that was copied into the output directory
    # by bb-run-segmentation so we can reproduce the rotation that was applied.
    pre_cfg = _load_preprocessing_cfg(masks_dir)
    rotate_mask_back = pre_cfg.get("rotate_mask_back", True)
    if pre_cfg:
        logger.info(
            "Preprocessing config found in masks_dir — rotate_mask_back: %s",
            rotate_mask_back,
        )
        if not rotate_mask_back:
            logger.info(
                "Masks are in the rotated frame; source images will be "
                "rotated forward before overlaying."
            )

    if all_frames:
        logger.info("--all: processing all %d mask files", len(mask_files))
        selected = [(f, float(np.load(f)["mask"].mean())) for f in mask_files]
    else:
        selected = _sample_frames(mask_files, n_frames, seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_saved = n_skipped = 0
    for mask_path, density in selected:
        stem = mask_path.stem
        image_path = images_dir / f"{stem}.png"

        if not image_path.exists():
            logger.warning("Source image not found, skipping: %s.png", stem)
            n_skipped += 1
            continue

        try:
            image_rgb = _load_rgb(image_path)
            mask = np.load(mask_path)["mask"]
        except Exception as exc:
            logger.error("Failed to load %s: %s", stem, exc)
            n_skipped += 1
            continue

        if mask.shape != image_rgb.shape[:2]:
            if not rotate_mask_back and pre_cfg:
                # Masks were saved in the rotated frame (rotate_mask_back: false).
                # Rotate the source image forward to match the mask orientation.
                try:
                    from bb_utils.data_preparation.segmentation_runner import (
                        _resolve_rotation,
                        _resolve_direction,
                    )
                    from bb_utils.segmentation.utils import rotate_image
                    rotation = _resolve_rotation(stem, pre_cfg)
                    if rotation:
                        direction = _resolve_direction(stem, pre_cfg)
                        effective = rotation if direction == "cw" else -rotation
                        image_rgb = rotate_image(image_rgb, effective)
                except Exception as exc:
                    logger.warning(
                        "Could not apply forward rotation for %s: %s", stem, exc
                    )

            if mask.shape != image_rgb.shape[:2]:
                logger.warning(
                    "Shape mismatch for %s: mask %s vs image %s — skipping",
                    stem, mask.shape, image_rgb.shape[:2],
                )
                n_skipped += 1
                continue

        overlay = _overlay_mask(image_rgb, mask, alpha=alpha)
        label = _bucket_label(density)
        out_name = f"{label}_d{density:.3f}_{stem}.png"
        _save_rgb(overlay, output_dir / out_name)
        n_saved += 1

    logger.info(
        "Done — %d overlays saved to %s  (%d skipped)",
        n_saved, output_dir, n_skipped,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Visual spot-check: overlay segmentation masks on source frames "
            "and save blended PNGs for manual inspection."
        )
    )
    parser.add_argument(
        "--masks-dir", type=Path, required=True,
        help="Directory of mask NPZ files produced by bb-run-segmentation.",
    )
    parser.add_argument(
        "--images-dir", type=Path, required=True,
        help="Directory of source PNG frames.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to write overlay PNG images.",
    )
    parser.add_argument(
        "--n-frames", type=int, default=20,
        help="Total frames to sample, spread across density buckets (default: 20).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.45,
        help="Opacity of the red mask overlay, 0–1 (default: 0.45).",
    )
    parser.add_argument(
        "--all", dest="all_frames", action="store_true",
        help="Process every mask in --masks-dir instead of sampling.",
    )
    args = parser.parse_args()

    for path, flag in [(args.masks_dir, "--masks-dir"), (args.images_dir, "--images-dir")]:
        if not path.exists():
            logger.error("%s does not exist: %s", flag, path)
            raise SystemExit(1)

    run_spotcheck(
        masks_dir=args.masks_dir,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        n_frames=args.n_frames,
        seed=args.seed,
        alpha=args.alpha,
        all_frames=args.all_frames,
    )


if __name__ == "__main__":
    main()
