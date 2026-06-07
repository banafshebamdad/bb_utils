#
# File: segmentation/utils.py
# Author: Banafshe Bamdad + Claude Code
# Created: 2026-04-21 CET
#

"""
Mask post-processing utilities for segmentation outputs.

All functions operate on uint8 (H, W) masks with values in {0, 1} and
return masks satisfying the same contract.

Functions
---------
dilate_mask
    Morphological dilation by a circular structuring element of given radius.
    Used to compensate for boundary inaccuracy in fisheye segmentation.
union_masks
    Elementwise OR of a list of masks (same as logical union).
resize_mask
    Resize a mask to a target (H, W) using nearest-neighbour interpolation
    to preserve binary values.
rotate_image
    Rotate an RGB image clockwise by a multiple of 90 degrees using exact
    np.rot90 — no interpolation, no pixel-value changes.
rotate_mask
    Rotate a binary mask clockwise by a multiple of 90 degrees.  Negative
    degrees are accepted and are used to invert a prior rotation.
"""

from typing import List, Tuple

import numpy as np


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a binary mask by *radius* pixels using a circular kernel.

    Uses a pure-numpy rolling-max approach so that ``scipy`` is not a hard
    dependency.  For large radii (> ~20), a scipy or OpenCV-based
    implementation would be faster, but correctness takes precedence here.

    Args:
        mask:   uint8 (H, W) binary mask with values in {0, 1}.
        radius: Dilation radius in pixels (0 = no-op).

    Returns:
        uint8 (H, W) dilated mask.

    Raises:
        ValueError: If *radius* is negative.
    """
    if radius < 0:
        raise ValueError(f"Dilation radius must be non-negative, got {radius}.")
    if radius == 0:
        return mask.astype(np.uint8)

    try:
        from scipy.ndimage import binary_dilation
        from scipy.ndimage import generate_binary_structure

        # Build a disk-shaped structuring element of the given radius
        y, x = np.ogrid[-radius: radius + 1, -radius: radius + 1]
        disk = (x ** 2 + y ** 2 <= radius ** 2)
        dilated = binary_dilation(mask.astype(bool), structure=disk)
        return dilated.astype(np.uint8)

    except ImportError:
        # Fallback: iterative row/column max (slower but dependency-free)
        result = mask.astype(np.float32)
        for _ in range(radius):
            shifted = [
                np.roll(result, 1,  axis=0),
                np.roll(result, -1, axis=0),
                np.roll(result, 1,  axis=1),
                np.roll(result, -1, axis=1),
            ]
            result = np.maximum(result, np.max(shifted, axis=0))
        return (result > 0).astype(np.uint8)


def union_masks(masks: List[np.ndarray]) -> np.ndarray:
    """Compute the elementwise union (logical OR) of a list of binary masks.

    Args:
        masks: List of uint8 (H, W) masks with values in {0, 1}.
               All masks must have the same shape.

    Returns:
        uint8 (H, W) union mask.

    Raises:
        ValueError: If *masks* is empty or shapes are inconsistent.
    """
    if not masks:
        raise ValueError("union_masks requires at least one mask.")
    shapes = {m.shape for m in masks}
    if len(shapes) > 1:
        raise ValueError(f"All masks must have the same shape; got {shapes}.")
    result = np.zeros_like(masks[0], dtype=np.uint8)
    for m in masks:
        result = np.bitwise_or(result, m.astype(np.uint8))
    return np.clip(result, 0, 1).astype(np.uint8)


def resize_mask(
    mask: np.ndarray,
    target_shape: Tuple[int, int],
) -> np.ndarray:
    """Resize a binary mask to *target_shape* using nearest-neighbour interpolation.

    Nearest-neighbour is required to preserve binary values; bilinear
    interpolation would introduce fractional values.

    Args:
        mask:         uint8 (H, W) binary mask.
        target_shape: ``(H_new, W_new)`` target spatial dimensions.

    Returns:
        uint8 (H_new, W_new) binary mask.
    """
    H_new, W_new = target_shape
    if mask.shape == target_shape:
        return mask.astype(np.uint8)

    try:
        from PIL import Image as _PIL_Image
        pil = _PIL_Image.fromarray(mask * 255, mode="L")
        pil = pil.resize((W_new, H_new), resample=_PIL_Image.NEAREST)
        return (np.array(pil) > 0).astype(np.uint8)
    except ImportError:
        pass

    # Numpy fallback: index-based nearest-neighbour
    H_src, W_src = mask.shape
    row_idx = (np.arange(H_new) * H_src / H_new).astype(int)
    col_idx = (np.arange(W_new) * W_src / W_new).astype(int)
    return mask[np.ix_(row_idx, col_idx)].astype(np.uint8)


def rotate_image(image: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate an RGB image clockwise by *degrees* using exact ``np.rot90``.

    Only multiples of 90 are supported.  No interpolation is performed, so
    pixel values are never altered — the output is a view or copy of the
    input with axes permuted.

    Args:
        image:   uint8 (H, W, 3) RGB image.
        degrees: Clockwise rotation angle in degrees.  Must be a multiple of
                 90.  Values are normalised modulo 360, so 360 is a no-op.

    Returns:
        uint8 array with shape ``(H, W, 3)`` (0° / 180°) or
        ``(W, H, 3)`` (90° / 270°).

    Raises:
        ValueError: If *degrees* is not a multiple of 90.
    """
    if degrees % 90 != 0:
        raise ValueError(
            f"rotate_image: degrees must be a multiple of 90, got {degrees}."
        )
    # np.rot90(m, k=1) rotates 90° CCW; negative k gives CW.
    k = -(degrees // 90) % 4
    if k == 0:
        return image.astype(np.uint8)
    return np.rot90(image, k=k).astype(np.uint8)


def rotate_mask(mask: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate a binary mask clockwise by *degrees* using exact ``np.rot90``.

    Negative *degrees* are accepted; they are used to invert a prior forward
    rotation (e.g. ``rotate_mask(mask, -90)`` undoes a ``rotate_image(..., 90)``).

    Args:
        mask:    uint8 (H, W) binary mask with values in {0, 1}.
        degrees: Clockwise rotation angle in degrees.  Must be a multiple of
                 90.  Negative values rotate counter-clockwise by that amount.

    Returns:
        uint8 array with shape ``(H, W)`` (0° / 180°) or
        ``(W, H)`` (90° / 270°).

    Raises:
        ValueError: If *degrees* is not a multiple of 90.
    """
    if degrees % 90 != 0:
        raise ValueError(
            f"rotate_mask: degrees must be a multiple of 90, got {degrees}."
        )
    k = -(degrees // 90) % 4
    if k == 0:
        return mask.astype(np.uint8)
    return np.rot90(mask, k=k).astype(np.uint8)
