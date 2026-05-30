#
# File: segmentation/__init__.py
# Author: Banafshe Bamdad + Claude Code
# Created: 2026-04-21 CET
#

"""
General-purpose segmentation backends for bb_utils.

This package provides a common interface for running instance/semantic
segmentation models on RGB images, normalising each model's output into a
single binary mask contract:

    mask: np.ndarray  shape (H, W), dtype uint8, values in {0, 1}
    - 1 = pixel belongs to a detected instance of a target class
    - 0 = does not belong to any detected target class

Sub-modules
-----------
base
    Abstract ``SegmentationBackend`` class and the common output contract.
factory
    ``create_backend(config)`` — instantiates the backend selected by
    ``config["model"]["backend"]``.
yolo_backend
    ``YoloBackend`` — wraps Ultralytics YOLOv8/11-seg (instance segmentation).
mask2former_backend
    ``Mask2FormerBackend`` — wraps HuggingFace Mask2Former (instance/panoptic).
deeplab_backend
    ``DeepLabBackend`` — DeepLabv3+ with ASPP module (semantic segmentation).
sam3_backend
    ``Sam3Backend`` — SAM 3 (Meta AI) open-vocabulary text-prompt segmentation.
utils
    ``dilate_mask``, ``union_masks``, ``resize_mask`` — mask post-processing
    utilities used by callers (e.g. ``segmentation_runner.py`` in sp-score).
"""

from bb_utils.segmentation.base import SegmentationBackend
from bb_utils.segmentation.factory import create_backend
from bb_utils.segmentation.utils import dilate_mask, resize_mask, union_masks

__all__ = [
    "SegmentationBackend",
    "create_backend",
    "dilate_mask",
    "resize_mask",
    "union_masks",
]
