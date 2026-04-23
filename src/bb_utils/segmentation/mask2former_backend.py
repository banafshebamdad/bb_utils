#
# File: segmentation/mask2former_backend.py
# Author: Banafshe Bamdad + GitHub Copilot (Claude Sonnet 4.6)
# Created: 2026-04-23 CET
#

"""
Mask2Former segmentation backend using HuggingFace Transformers.

Supports any ``Mask2FormerForUniversalSegmentation`` checkpoint from the
HuggingFace Hub.  Typical checkpoints for pedestrian detection:

  Instance segmentation (recommended for pedestrian masking):
    facebook/mask2former-swin-tiny-coco-instance
    facebook/mask2former-swin-small-coco-instance
    facebook/mask2former-swin-base-coco-instance
    facebook/mask2former-swin-large-coco-instance

  Panoptic segmentation (combines thing + stuff classes):
    facebook/mask2former-swin-tiny-coco-panoptic
    facebook/mask2former-swin-base-coco-panoptic
    facebook/mask2former-swin-large-coco-panoptic

COCO label IDs
--------------
For both instance and panoptic COCO checkpoints the label IDs are 0-indexed
COCO thing classes: ``0 = person``, ``1 = bicycle``, ``2 = car``, …  This is
the same convention as YOLOv8-seg, so ``target_classes: [0]`` selects person
in both backends.

Output contract compliance
--------------------------
The ``segment`` method satisfies the ``SegmentationBackend`` contract:
- Returns uint8 (H, W) with values in {0, 1}.
- Shape matches the input image dimensions.
- No detections → all-zeros mask.

Config keys read by this backend (via ``Mask2FormerBackend.from_config``)
-------------------------------------------------------------------------
  ``model_name``            str   — HuggingFace repo ID or local path.
                                    Default: "facebook/mask2former-swin-tiny-coco-instance"
  ``device``                str   — "cuda", "cpu", or "cuda:0".
                                    Default: "cpu"
  ``confidence_threshold``  float — Minimum prediction score in (0, 1).
                                    Default: 0.5
  ``segmentation_type``     str   — "instance" or "panoptic".
                                    If omitted, auto-detected from model_name
                                    ("panoptic" if the name contains "panoptic",
                                    else "instance").

Dependencies
------------
    pip install transformers torch

The ``transformers`` and ``torch`` packages are not installed by bb_utils and
must be present in the environment before using this backend.
"""

import logging
from typing import List

import numpy as np

from bb_utils.segmentation.base import SegmentationBackend

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "facebook/mask2former-swin-tiny-coco-instance"


class Mask2FormerBackend(SegmentationBackend):
    """Segmentation backend wrapping HuggingFace Mask2Former.

    Args:
        model_name:           HuggingFace repo ID or local path.
        device:               Inference device (``"cuda"``, ``"cpu"``).
        confidence_threshold: Minimum prediction score in (0, 1).
        segmentation_type:    ``"instance"`` or ``"panoptic"``.
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        confidence_threshold: float,
        segmentation_type: str,
    ) -> None:
        try:
            import torch
            from transformers import (
                AutoImageProcessor,
                Mask2FormerForUniversalSegmentation,
            )
        except ImportError as exc:
            raise RuntimeError(
                "transformers and torch are required for Mask2FormerBackend. "
                "Install them with: pip install transformers torch"
            ) from exc

        self._conf = float(confidence_threshold)
        self._device = device
        self._segmentation_type = segmentation_type
        self._torch = torch

        try:
            self._processor = AutoImageProcessor.from_pretrained(model_name)
            self._model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
            self._model.eval()
            self._model.to(device)
            logger.info(
                "Loaded Mask2Former model: %s on %s (segmentation_type=%s)",
                model_name, device, segmentation_type,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Mask2Former model '{model_name}': {exc}"
            ) from exc

    def segment(
        self,
        image: np.ndarray,
        target_classes: List[int],
    ) -> np.ndarray:
        """Run Mask2Former on *image* and return a binary mask.

        Args:
            image:          uint8 RGB image of shape (H, W, 3).
            target_classes: List of COCO label IDs to include (0 = person).

        Returns:
            uint8 (H, W) mask; 1 = detected target-class pixel, 0 = background.
        """
        from PIL import Image as _PIL

        H, W = image.shape[:2]
        expected_shape = (H, W)
        mask = np.zeros(expected_shape, dtype=np.uint8)

        pil_image = _PIL.fromarray(image)
        inputs = self._processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with self._torch.no_grad():
            outputs = self._model(**inputs)

        if self._segmentation_type == "panoptic":
            results = self._processor.post_process_panoptic_segmentation(
                outputs,
                threshold=self._conf,
                target_sizes=[(H, W)],
                label_ids_to_fuse=[],
            )
        else:
            results = self._processor.post_process_instance_segmentation(
                outputs,
                threshold=self._conf,
                target_sizes=[(H, W)],
            )

        if not results:
            return self._validate_output(mask, expected_shape)

        result = results[0]
        segments_info = result.get("segments_info", [])
        if not segments_info:
            return self._validate_output(mask, expected_shape)

        seg_map = result["segmentation"].cpu().numpy()  # (H, W), values = segment IDs

        for seg_info in segments_info:
            if seg_info["label_id"] in target_classes:
                seg_id = seg_info["id"]
                mask = np.bitwise_or(mask, (seg_map == seg_id).astype(np.uint8))

        return self._validate_output(mask, expected_shape)

    @classmethod
    def from_config(cls, model_cfg: dict) -> "Mask2FormerBackend":
        """Instantiate from a ``model`` config dict section.

        Auto-detects ``segmentation_type`` from ``model_name`` if the key is
        absent: names containing ``"panoptic"`` default to ``"panoptic"``,
        everything else defaults to ``"instance"``.
        """
        model_name = model_cfg.get("model_name", _DEFAULT_MODEL)
        seg_type = model_cfg.get("segmentation_type")
        if seg_type is None:
            seg_type = "panoptic" if "panoptic" in model_name else "instance"
        return cls(
            model_name=model_name,
            device=model_cfg.get("device", "cpu"),
            confidence_threshold=model_cfg.get("confidence_threshold", 0.5),
            segmentation_type=seg_type,
        )
