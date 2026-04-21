#
# File: segmentation/base.py
# Author: Banafshe Bamdad + Claude Code
# Created: 2026-04-21 CET
#

"""
Abstract base class and common output contract for segmentation backends.

All backends in this package must satisfy the contract defined here so that
callers (e.g. ``sp-score``'s ``segmentation_runner.py``) can swap backends
by changing a config key without modifying any other code.

Output contract
---------------
``SegmentationBackend.segment(image, target_classes) → np.ndarray``

Input
  image:          np.ndarray  shape (H, W, 3), dtype uint8, colour order RGB.
  target_classes: List[int]   Class indices to include in the output mask.
                              Interpretation is model-specific and documented
                              per backend.

Output
  mask: np.ndarray  shape (H, W), dtype uint8
  - Value 1: pixel belongs to a detected instance of a target class.
  - Value 0: does not belong to any detected target class.
  - Shape (H, W) exactly equals the input image spatial dimensions.
  - Values strictly in {0, 1}; no probabilities, no multi-class integers.
  - Output is the union over all instances of all target classes.
  - No detections → all-zeros mask (valid result, not an error).

Coordinate system
  Pixel (row i, col j) in the output mask corresponds to pixel (row i, col j)
  in the input image.  No spatial transformation is applied.

Error handling
  - Raises ValueError for unsupported target_class values (model-specific).
  - Raises RuntimeError on model loading or inference failure.
  - Does NOT raise for empty detection results.

Normalisation responsibility
  Each backend converts its model-specific output into the common binary mask:
  - Instance segmentation (YOLO): union of per-instance masks for target classes.
  - Semantic segmentation (DeepLab): class probability map thresholded to binary.
  - Promptable segmentation (SAM): masks for regions matching target classes.

The backend does NOT apply dilation and does NOT convert grayscale to RGB.
Both are the caller's responsibility (``segmentation_runner.py``).
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class SegmentationBackend(ABC):
    """Abstract base class for all segmentation backends.

    Concrete subclasses must implement :meth:`segment`.  Construction
    parameters (model path, confidence threshold, device, etc.) are
    backend-specific and accepted via the constructor.
    """

    @abstractmethod
    def segment(
        self,
        image: np.ndarray,
        target_classes: List[int],
    ) -> np.ndarray:
        """Run segmentation and return a binary mask.

        Args:
            image:          uint8 RGB image of shape (H, W, 3).
            target_classes: List of integer class indices to include.

        Returns:
            uint8 ndarray of shape (H, W) with values in {0, 1}.
            1 = detected target class pixel, 0 = background.

        Raises:
            ValueError:   If any value in *target_classes* is unsupported.
            RuntimeError: On model inference failure.
        """

    def _validate_output(self, mask: np.ndarray, expected_shape: tuple) -> np.ndarray:
        """Validate and enforce the output contract.

        Should be called by every concrete ``segment`` implementation before
        returning.  Raises ``RuntimeError`` if the shape contract is violated;
        coerces dtype to uint8 and clips to {0, 1}.

        Args:
            mask:           Candidate output mask.
            expected_shape: (H, W) tuple from the input image.

        Returns:
            Validated uint8 mask of shape (H, W) with values in {0, 1}.
        """
        if mask.shape != expected_shape:
            raise RuntimeError(
                f"Segmentation backend produced mask shape {mask.shape} "
                f"but input image shape is {expected_shape}.  "
                f"Backend must return a mask with the same spatial dimensions."
            )
        mask = np.asarray(mask, dtype=np.uint8)
        mask = np.clip(mask, 0, 1)
        return mask
