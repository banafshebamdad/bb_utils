#
# File: segmentation/yolo_backend.py
# Author: Banafshe Bamdad + Claude Code
# Created: 2026-04-21 CET
#

"""
YOLOv8-seg segmentation backend using Ultralytics.

This backend supports any Ultralytics YOLOv8 (or later) segmentation model
(``*-seg`` variant).  It returns the union of all instance masks whose
predicted class is in *target_classes*.

Output contract compliance
--------------------------
The ``segment`` method satisfies the ``SegmentationBackend`` contract:
- Returns uint8 (H, W) with values in {0, 1}.
- Shape matches the input image dimensions.
- No detections → all-zeros mask.

Class indices follow the model's training dataset (COCO by default).
COCO class 0 = person.

Config keys read by this backend (via ``YoloBackend.__init__``)
------------------------------------------------------------
  ``model_name``            str   — model identifier, e.g. "yolov8n-seg"
  ``device``                str   — "cuda", "cpu", or "cuda:0" etc.
  ``confidence_threshold``  float — required; minimum detection confidence
  ``iou_threshold``         float — required; NMS IoU threshold
"""

import logging
from typing import List

import numpy as np

from bb_utils.segmentation.base import SegmentationBackend

logger = logging.getLogger(__name__)


class YoloBackend(SegmentationBackend):
    """Segmentation backend wrapping Ultralytics YOLOv8-seg.

    Args:
        model_name:           Model identifier or path, e.g. ``"yolov8n-seg"``.
        device:               Inference device string (``"cuda"``, ``"cpu"``).
        confidence_threshold: Minimum detection confidence in (0, 1).
        iou_threshold:        NMS IoU threshold in (0, 1).
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        confidence_threshold: float,
        iou_threshold: float,
    ) -> None:
        if confidence_threshold is None:
            raise ValueError("YoloBackend: 'confidence_threshold' is required (currently null).")
        if iou_threshold is None:
            raise ValueError("YoloBackend: 'iou_threshold' is required (currently null).")

        self._conf  = float(confidence_threshold)
        self._iou   = float(iou_threshold)
        self._device = device

        try:
            from ultralytics import YOLO
            self._model = YOLO(model_name)
            logger.info("Loaded YOLO model: %s on %s", model_name, device)
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed.  Install it with: pip install ultralytics"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to load YOLO model '{model_name}': {exc}") from exc

    def segment(
        self,
        image: np.ndarray,
        target_classes: List[int],
    ) -> np.ndarray:
        """Run YOLOv8-seg on *image* and return a binary mask.

        Args:
            image:          uint8 RGB image of shape (H, W, 3).
            target_classes: List of COCO class indices to include.

        Returns:
            uint8 (H, W) mask; 1 = detected target-class pixel, 0 = background.
        """
        H, W = image.shape[:2]
        expected_shape = (H, W)
        mask = np.zeros(expected_shape, dtype=np.uint8)

        results = self._model.predict(
            source=image,
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            verbose=False,
        )

        if not results or results[0].masks is None:
            return self._validate_output(mask, expected_shape)

        result = results[0]
        boxes = result.boxes

        for i, cls_id in enumerate(boxes.cls.cpu().numpy().astype(int)):
            if cls_id not in target_classes:
                continue
            # Instance mask: float (H', W') in [0, 1] at model resolution
            inst_mask = result.masks.data[i].cpu().numpy()
            # Resize to input resolution if needed
            if inst_mask.shape != (H, W):
                from bb_utils.segmentation.utils import resize_mask
                inst_mask_bin = (inst_mask > 0.5).astype(np.uint8)
                inst_mask_bin = resize_mask(inst_mask_bin, (H, W))
            else:
                inst_mask_bin = (inst_mask > 0.5).astype(np.uint8)
            mask = np.bitwise_or(mask, inst_mask_bin)

        return self._validate_output(mask, expected_shape)

    @classmethod
    def from_config(cls, model_cfg: dict) -> "YoloBackend":
        """Instantiate from a ``model`` config dict section."""
        return cls(
            model_name=model_cfg.get("model_name", "yolov8n-seg"),
            device=model_cfg.get("device", "cpu"),
            confidence_threshold=model_cfg.get("confidence_threshold"),
            iou_threshold=model_cfg.get("iou_threshold"),
        )
