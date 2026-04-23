#
# File: data_preparation/segmentation_runner.py
# Author: Banafshe Bamdad + Claude Code
# Created: 2026-04-22 CET
#

"""
Generic per-frame segmentation runner for image datasets.

This module orchestrates per-frame segmentation for any directory of PNG images.
It handles:

  - Loading PNG frames (grayscale or RGB) and converting to 3-channel RGB
  - Delegating inference to a ``bb_utils.segmentation.SegmentationBackend``
  - Applying mask dilation via ``bb_utils.segmentation.utils.dilate_mask``
  - Writing per-frame mask NPZ files to::

      {output_dir}/{stem}.npz  (key: ``mask``)

Mask encoding convention:
  ``1`` = detected instance of a target class (dynamic), ``0`` = background (static).

The runner discovers frames by globbing ``{images_dir}/*.png`` directly.
``images_dir`` should point to a flat directory of PNG files; no subdirectory
structure is assumed.  For dataset splits, call the tool once per split.

The runner is idempotent: frames whose mask NPZ already exists are skipped
unless ``--force`` is passed.

CLI
---
    bb-run-segmentation --config model.yaml --images-dir /path/to/images --output-dir /path/to/masks
    bb-run-segmentation --config model.yaml --images-dir /path/to/images --output-dir /path/to/masks --force
    bb-run-segmentation --config model.yaml --images-dir /path/to/images --output-dir /path/to/masks --dry-run

Config schema (model settings only)
------------------------------------
    model:
      backend: yolo
      model_name: yolov8n-seg
      device: cpu
      confidence_threshold: 0.25
      iou_threshold: 0.45
      target_classes: [0]   # 0 = person in COCO
      mask_dilation_px: 5

    model:
      backend: yolo
      model_name: yolov8n-seg
      device: cpu
      confidence_threshold: 0.25
      iou_threshold: 0.45
      target_classes: [0]   # 0 = person in COCO
      mask_dilation_px: 5
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-frame segmentation
# ---------------------------------------------------------------------------

def segment_frame(
    image_path: Path,
    out_path: Path,
    backend,
    target_classes: List[int],
    dilation_radius: int,
    force: bool = False,
) -> bool:
    """Segment one frame and write the mask NPZ.

    Args:
        image_path:      Path to the source PNG (grayscale or RGB, any depth).
        out_path:        Destination NPZ path.
        backend:         ``SegmentationBackend`` instance.
        target_classes:  Class indices to include in the mask.
        dilation_radius: Dilation radius in pixels (0 = no dilation).
        force:           Overwrite existing NPZ when True.

    Returns:
        True if the mask was written; False if skipped (already exists).

    Raises:
        FileNotFoundError: If *image_path* does not exist.
    """
    if out_path.exists() and not force:
        return False

    if not image_path.exists():
        raise FileNotFoundError(f"Frame image not found: {image_path}")

    # Load image — convert grayscale to 3-channel RGB
    image_rgb = _load_as_rgb(image_path)

    # Run segmentation
    mask = backend.segment(image_rgb, target_classes)

    # Apply dilation
    if dilation_radius > 0:
        from bb_utils.segmentation.utils import dilate_mask
        mask = dilate_mask(mask, dilation_radius)

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, mask=mask)
    return True


def _load_as_rgb(image_path: Path) -> np.ndarray:
    """Load an image file and return a uint8 (H, W, 3) RGB array."""
    try:
        import cv2
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"cv2.imread returned None for {image_path}")
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        # OpenCV loads BGR; convert to RGB
        return img[:, :, ::-1].astype(np.uint8)
    except ImportError:
        pass

    try:
        from PIL import Image as _PIL
        img = _PIL.open(image_path).convert("RGB")
        return np.array(img, dtype=np.uint8)
    except ImportError:
        pass

    raise RuntimeError(
        "Neither 'cv2' nor 'PIL' is available.  "
        "Install one of them: pip install opencv-python  or  pip install Pillow"
    )


# ---------------------------------------------------------------------------
# Split-level runner
# ---------------------------------------------------------------------------

def run_on_dir(
    images_dir: Path,
    output_dir: Path,
    backend,
    target_classes: List[int],
    dilation_radius: int,
    force: bool = False,
) -> Dict:
    """Segment all PNG frames in *images_dir* and write mask NPZ files to *output_dir*.

    Args:
        images_dir:      Flat directory containing ``*.png`` frames.
        output_dir:      Destination directory for mask NPZ files (created if absent).
        backend:         Configured ``SegmentationBackend`` instance.
        target_classes:  Class indices to include in the mask.
        dilation_radius: Dilation radius in pixels (0 = no dilation).
        force:           Overwrite existing masks when True.

    Returns:
        Summary dict with ``"total"``, ``"written"``, ``"skipped"``, ``"failed"``.
    """
    image_files = sorted(images_dir.glob("*.png"))
    if not image_files:
        logger.warning("No PNG frames found in %s", images_dir)
        return {"total": 0, "written": 0, "skipped": 0, "failed": 0}

    logger.info("%d frames to segment in %s", len(image_files), images_dir)
    n_written = n_skip = n_fail = 0

    try:
        from tqdm import tqdm
        file_iter = tqdm(image_files, unit="frame")
    except ImportError:
        file_iter = image_files

    for image_path in file_iter:
        stem = image_path.stem
        out_path = output_dir / f"{stem}.npz"

        try:
            written = segment_frame(
                image_path=image_path,
                out_path=out_path,
                backend=backend,
                target_classes=target_classes,
                dilation_radius=dilation_radius,
                force=force,
            )
            if written:
                n_written += 1
            else:
                n_skip += 1
        except Exception as exc:
            logger.error("Failed on %s: %s", stem, exc, exc_info=True)
            n_fail += 1

    return {"total": len(image_files), "written": n_written, "skipped": n_skip, "failed": n_fail}

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    """CLI entry point: bb-run-segmentation.

    Usage::

        bb-run-segmentation --config model.yaml --images-dir /path/to/images --output-dir /path/to/masks
        bb-run-segmentation --config model.yaml --images-dir /path/to/images --output-dir /path/to/masks --force
        bb-run-segmentation --config model.yaml --images-dir /path/to/images --output-dir /path/to/masks --dry-run
    """
    parser = argparse.ArgumentParser(
        description="Run per-frame segmentation to produce binary mask NPZ files."
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to segmentation YAML config (model settings).",
    )
    parser.add_argument(
        "--images-dir", type=Path, required=True,
        help="Flat directory of *.png frames to segment.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Destination directory for mask NPZ files.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing mask NPZ files.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate config and paths without running inference.",
    )

    args = parser.parse_args()
    _setup_logging(args.verbose)

    if not args.config.exists():
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_cfg = config.get("model", {})

    if args.dry_run:
        logger.info("DRY RUN — configuration summary")
        logger.info("  backend            : %s", model_cfg.get("backend"))
        logger.info("  model_name         : %s", model_cfg.get("model_name"))
        logger.info("  device             : %s", model_cfg.get("device"))
        logger.info("  target_classes     : %s", model_cfg.get("target_classes"))
        logger.info("  confidence_threshold: %s", model_cfg.get("confidence_threshold"))
        logger.info("  mask_dilation_px   : %s", model_cfg.get("mask_dilation_px"))
        logger.info("  images_dir         : %s", args.images_dir)
        logger.info("  output_dir         : %s", args.output_dir)
        return

    # Instantiate backend
    try:
        from bb_utils.segmentation import create_backend
        backend = create_backend(config)
    except Exception as exc:
        logger.error("Failed to create segmentation backend: %s", exc)
        sys.exit(1)

    target_classes  = model_cfg.get("target_classes", [0])
    dilation_radius = model_cfg.get("mask_dilation_px", 0) or 0

    t0 = time.time()
    summary = run_on_dir(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        backend=backend,
        target_classes=target_classes,
        dilation_radius=dilation_radius,
        force=args.force,
    )
    elapsed = time.time() - t0
    logger.info(
        "Done — %d written, %d skipped, %d failed  (%.1fs)",
        summary["written"], summary["skipped"], summary["failed"], elapsed,
    )

    if summary["failed"] > 0:
        sys.exit(1)
