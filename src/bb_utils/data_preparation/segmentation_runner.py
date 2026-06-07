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
  - Optionally rotating the image before passing it to the backend
  - Delegating inference to a ``bb_utils.segmentation.SegmentationBackend``
  - Optionally rotating the generated mask back to the original orientation
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

Pre-rotation config (optional, top-level)
-----------------------------------------
    # Global rotation — same angle applied to all images:
    preprocessing:
      pre_rotation_deg: 90   # clockwise; must be a multiple of 90; null = no rotation
      rotate_mask_back: true # rotate mask back to original orientation (default true)

    # Per-camera rotation — camera inferred from filename stem (_L_ / _R_):
    preprocessing:
      rotate_mask_back: true
      cameras:
        L: 90
        R: 270

    # Mixed — cameras dict takes precedence; pre_rotation_deg is the fallback:
    preprocessing:
      pre_rotation_deg: 90
      rotate_mask_back: true
      cameras:
        L: 90
        R: 270
"""

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pre-rotation helpers
# ---------------------------------------------------------------------------

def _camera_from_stem(stem: str) -> Optional[str]:
    """Return the camera indicator (``'L'`` or ``'R'``) embedded in *stem*.

    Expects the InCrowd-VI filename convention::

        {sequence}_{L|R}_{timestamp_us}

    The camera indicator is the second-to-last ``_``-separated token.
    Returns ``None`` when the stem does not match the expected pattern.
    """
    parts = stem.split("_")
    if len(parts) >= 2 and parts[-2] in ("L", "R"):
        return parts[-2]
    return None


def _resolve_rotation(stem: str, pre_cfg: Optional[dict]) -> Optional[int]:
    """Return the pre-rotation angle (degrees, CW) for the frame identified by *stem*.

    Resolution order:

    1. If *pre_cfg* contains a ``cameras`` dict, detect the camera from *stem*
       and look up its rotation.  If the camera is not found in the dict, fall
       through to step 2.
    2. Return ``pre_cfg.get("pre_rotation_deg")``, which may be ``None``.

    Args:
        stem:    Filename stem (without extension) of the source image.
        pre_cfg: Contents of the ``preprocessing:`` YAML section, or ``None``.

    Returns:
        Integer rotation in degrees (clockwise), or ``None`` when no rotation
        should be applied.
    """
    if not pre_cfg:
        return None
    cameras_map = pre_cfg.get("cameras")
    if cameras_map:
        camera = _camera_from_stem(stem)
        if camera is None:
            logger.debug(
                "Could not detect camera from stem '%s'; "
                "falling back to pre_rotation_deg.",
                stem,
            )
        else:
            rotation = cameras_map.get(camera)
            if rotation is not None:
                return int(rotation)
            logger.debug(
                "Camera '%s' not found in preprocessing.cameras; "
                "falling back to pre_rotation_deg.",
                camera,
            )
    return pre_cfg.get("pre_rotation_deg")


def _resolve_direction(stem: str, pre_cfg: Optional[dict]) -> str:
    """Return the rotation direction (``'cw'`` or ``'ccw'``) for *stem*.

    Resolution order:

    1. If *pre_cfg* contains a ``camera_directions`` dict, detect the camera
       from *stem* and look up its direction.  If the camera is not found,
       fall through to step 2.
    2. Return ``pre_cfg.get("rotation_direction", "cw")``, lowercased.

    Args:
        stem:    Filename stem (without extension) of the source image.
        pre_cfg: Contents of the ``preprocessing:`` YAML section, or ``None``.

    Returns:
        ``'cw'`` or ``'ccw'``.

    Raises:
        ValueError: If the resolved direction is not ``'cw'`` or ``'ccw'``.
    """
    direction = "cw"  # hardcoded default
    if pre_cfg:
        cam_dir_map = pre_cfg.get("camera_directions")
        if cam_dir_map:
            camera = _camera_from_stem(stem)
            if camera is not None and camera in cam_dir_map:
                direction = str(cam_dir_map[camera]).lower()
            else:
                direction = pre_cfg.get("rotation_direction", "cw").lower()
        else:
            direction = pre_cfg.get("rotation_direction", "cw").lower()
    if direction not in ("cw", "ccw"):
        raise ValueError(
            f"rotation_direction must be 'cw' or 'ccw', got '{direction}'."
        )
    return direction

def segment_frame(
    image_path: Path,
    out_path: Path,
    backend,
    target_classes: List[int],
    dilation_radius: int,
    force: bool = False,
    preprocessing_cfg: Optional[dict] = None,
    rotate_mask_back: bool = True,
) -> bool:
    """Segment one frame and write the mask NPZ.

    Args:
        image_path:       Path to the source PNG (grayscale or RGB, any depth).
        out_path:         Destination NPZ path.
        backend:          ``SegmentationBackend`` instance.
        target_classes:   Class indices to include in the mask.
        dilation_radius:  Dilation radius in pixels (0 = no dilation).
        force:            Overwrite existing NPZ when True.
        preprocessing_cfg: Contents of the ``preprocessing:`` YAML section.
                          Used to resolve the per-frame rotation angle.  When
                          ``None`` or empty, no rotation is applied.
        rotate_mask_back: When ``True`` (default) and a pre-rotation was
                          applied, the generated mask is rotated back by the
                          inverse angle so it remains pixel-aligned with the
                          original source image.

    Returns:
        True if the mask was written; False if skipped (already exists).

    Raises:
        FileNotFoundError: If *image_path* does not exist.
    """
    if out_path.exists() and not force:
        return False

    if not image_path.exists():
        raise FileNotFoundError(f"Frame image not found: {image_path}")

    stem = image_path.stem

    # Load image — convert grayscale to 3-channel RGB
    image_rgb = _load_as_rgb(image_path)

    # Optionally rotate before segmentation
    rotation = _resolve_rotation(stem, preprocessing_cfg)
    effective_rotation = None
    if rotation:
        direction = _resolve_direction(stem, preprocessing_cfg)
        effective_rotation = rotation if direction == "cw" else -rotation
        from bb_utils.segmentation.utils import rotate_image
        logger.debug(
            "Pre-rotating '%s' by %d° %s.", stem, abs(rotation), direction.upper()
        )
        image_rgb = rotate_image(image_rgb, effective_rotation)

    # Run segmentation
    mask = backend.segment(image_rgb, target_classes)

    # Rotate mask back to original image orientation
    if effective_rotation and rotate_mask_back:
        from bb_utils.segmentation.utils import rotate_mask
        logger.debug("Rotating mask for '%s' back by %d°.", stem, -effective_rotation)
        mask = rotate_mask(mask, -effective_rotation)

    # Apply dilation (in original image coordinates)
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
    sequence: Optional[str] = None,
    preprocessing_cfg: Optional[dict] = None,
    rotate_mask_back: bool = True,
) -> Dict:
    """Segment PNG frames in *images_dir* and write mask NPZ files to *output_dir*.

    Args:
        images_dir:       Flat directory containing ``*.png`` frames.
        output_dir:       Destination directory for mask NPZ files (created if absent).
        backend:          Configured ``SegmentationBackend`` instance.
        target_classes:   Class indices to include in the mask.
        dilation_radius:  Dilation radius in pixels (0 = no dilation).
        force:            Overwrite existing masks when True.
        sequence:         Optional sequence name prefix.  When provided, only
                          frames whose stem starts with this string are processed
                          (glob pattern ``{sequence}*.png``).
        preprocessing_cfg: Contents of the ``preprocessing:`` YAML section.
                          Passed through to :func:`segment_frame` for per-frame
                          rotation resolution.
        rotate_mask_back: When ``True`` (default), the mask is rotated back to
                          the original image orientation after segmentation.

    Returns:
        Summary dict with ``"total"``, ``"written"``, ``"skipped"``, ``"failed"``.
    """
    glob_pattern = f"{sequence}*.png" if sequence else "*.png"
    image_files = sorted(images_dir.glob(glob_pattern))
    if not image_files:
        if sequence:
            logger.warning(
                "No PNG frames matching '%s' found in %s", glob_pattern, images_dir
            )
        else:
            logger.warning("No PNG frames found in %s", images_dir)
        return {"total": 0, "written": 0, "skipped": 0, "failed": 0}

    if sequence:
        logger.info(
            "%d frames matching sequence '%s' to segment in %s",
            len(image_files), sequence, images_dir,
        )
    else:
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
                preprocessing_cfg=preprocessing_cfg,
                rotate_mask_back=rotate_mask_back,
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
        bb-run-segmentation --config model.yaml --images-dir /path/to/images --output-dir /path/to/masks --sequence Kiko_loop_R
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
        "--sequence", type=str, default=None,
        help=(
            "Only process frames whose filename starts with this sequence name "
            "(e.g. 'Kiko_loop_R').  Equivalent to globbing '{sequence}*.png' "
            "instead of '*.png'.  Omit to process all frames."
        ),
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

    pre_cfg = config.get("preprocessing") or {}
    rotate_mask_back = pre_cfg.get("rotate_mask_back", True)

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
        logger.info("  sequence filter    : %s", args.sequence or "(all frames)")
        rotation_direction = pre_cfg.get("rotation_direction", "cw").lower()
        cam_dir_map = pre_cfg.get("camera_directions", {})
        cameras_map = pre_cfg.get("cameras")
        if cameras_map:
            logger.info("  rotation mode      : per-camera")
            for cam, deg in cameras_map.items():
                cam_dir = cam_dir_map.get(cam, rotation_direction).lower()
                logger.info("    camera %-3s       : %s° %s", cam, deg, cam_dir.upper())
        else:
            logger.info(
                "  pre_rotation_deg   : %s",
                pre_cfg.get("pre_rotation_deg") or "(none)",
            )
            if pre_cfg.get("pre_rotation_deg"):
                logger.info("  rotation_direction : %s", rotation_direction.upper())
        logger.info("  rotate_mask_back   : %s", rotate_mask_back)
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

    # Copy config into output directory for reproducibility
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, args.output_dir / args.config.name)
    logger.info("Config copied to %s", args.output_dir / args.config.name)

    t0 = time.time()
    summary = run_on_dir(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        backend=backend,
        target_classes=target_classes,
        dilation_radius=dilation_radius,
        force=args.force,
        sequence=args.sequence,
        preprocessing_cfg=pre_cfg or None,
        rotate_mask_back=rotate_mask_back,
    )
    elapsed = time.time() - t0
    logger.info(
        "Done — %d written, %d skipped, %d failed  (%.1fs)",
        summary["written"], summary["skipped"], summary["failed"], elapsed,
    )

    if summary["failed"] > 0:
        sys.exit(1)
