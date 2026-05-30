#
# File: segmentation/sam3_backend.py
# Author: Banafshe Bamdad + GitHub Copilot (Claude Sonnet 4.6)
# Date: 2026-05-30
#

"""
SAM 3 open-vocabulary segmentation backend.

SAM 3 (Segment Anything with Concepts, Meta AI 2025) is a unified foundation
model for promptable image and video segmentation.  Unlike class-index-based
backends (YOLO, Mask2Former, DeepLab), SAM 3 uses **free-form text prompts**
for open-vocabulary instance detection and segmentation.

This backend bridges SAM 3's text-prompt interface to the
``SegmentationBackend`` contract.  Integer ``target_classes`` supplied by the
pipeline are translated to text prompts via a configurable ``class_to_text``
dictionary.

Architecture
------------
SAM 3 consists of a ViT visual backbone (~848 M parameters total) shared
between:

  - A DETR-based detector conditioned on text, geometry, and image exemplars.
  - A SAM 2-derived tracker (video; not used in single-image mode here).

The detector accepts a short text phrase (e.g. ``"person"``,
``"cyclist"``, ``"person carrying a bag"``) and produces per-instance binary
masks along with confidence scores and bounding boxes.

Inference flow
--------------
1. Precompute visual backbone features once per image via
   ``Sam3Processor.set_image``.
2. For each unique text prompt derived from *target_classes*:
   a. Shallow-copy the image-state dict so each text run is independent.
   b. Run ``Sam3Processor.set_text_prompt`` — executes the detector head.
   c. Filter detections by ``confidence_threshold``.
   d. Union all per-instance boolean masks for this prompt.
3. Accumulate per-prompt masks via element-wise maximum.
4. Return a ``uint8 (H, W)`` mask with values in ``{0, 1}``.

Open-vocabulary prompts
-----------------------
Because SAM 3 is open-vocabulary, ``class_to_text`` can contain arbitrary
text descriptions beyond the standard COCO class names.  For example:

    class_to_text:
      0: "person"            # standard
      0: "pedestrian"        # equivalent but more descriptive
      0: "person in a crowd" # context-specific phrasing

The backend provides a built-in default mapping that covers all 80 COCO
categories (keyed by COCO class ID 0–79).  Override any entry by setting
``class_to_text`` in the config.

Authentication
--------------
SAM 3 checkpoints require access to be granted on HuggingFace before the
automatic download will succeed.

  1. Request access at https://huggingface.co/facebook/sam3
  2. Authenticate:  ``huggingface-cli login``

The checkpoint (~3.4 GB for ``sam3``) is downloaded automatically on first use
to the HuggingFace cache (``~/.cache/huggingface/hub``).  Subsequent runs
reuse the cached file.  Set ``checkpoint_path`` to a local ``.pt`` file to
avoid repeated downloads.

Output contract compliance
--------------------------
``segment`` satisfies the ``SegmentationBackend`` contract:

- Returns ``uint8 (H, W)`` with values in ``{0, 1}``.
- Shape matches the input image dimensions exactly.
- No detections → all-zeros mask (valid result, not an error).

Config keys (all under ``model:``)
-----------------------------------
  ``version``               str   — SAM 3 checkpoint version.
                                    ``"sam3"``  → ``facebook/sam3`` (sam3.pt)
                                    ``"sam3.1"`` → ``facebook/sam3.1``
                                                   (sam3.1_multiplex.pt)
                                    Default: ``"sam3"``.
  ``checkpoint_path``       str   — Absolute path to a local ``.pt`` file.
                                    If ``null``, auto-downloads from HuggingFace
                                    (requires prior authentication).
                                    Default: ``null``.
  ``device``                str   — ``"cuda"``, ``"cpu"``, or ``"cuda:0"``.
                                    Default: ``"cuda"``.
  ``confidence_threshold``  float — Minimum detection score in (0, 1).
                                    Default: ``0.5``.
  ``class_to_text``         dict  — Maps integer class IDs to text prompts.
                                    Merged on top of the built-in COCO default
                                    map.  E.g. ``{0: "person"}`` or
                                    ``{0: "pedestrian"}``.
                                    Default: ``{}`` (use built-in COCO map).

Dependencies
------------
    git clone https://github.com/facebookresearch/sam3.git
    cd sam3
    pip install -e .

The ``sam3`` package and its dependencies (``torch``, ``torchvision``,
``huggingface_hub``, ``iopath``, etc.) are not installed by bb_utils and must
be present in the environment before using this backend.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from bb_utils.segmentation.base import SegmentationBackend

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default class-ID → text-prompt mapping (COCO 80-class label space).
# Users can override individual entries or all entries via ``class_to_text``
# in the config.
# ---------------------------------------------------------------------------
_DEFAULT_CLASS_TO_TEXT: Dict[int, str] = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    4:  "airplane",
    5:  "bus",
    6:  "train",
    7:  "truck",
    8:  "boat",
    9:  "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}


class Sam3Backend(SegmentationBackend):
    """Segmentation backend wrapping SAM 3 (Segment Anything with Concepts).

    SAM 3 uses free-form text prompts to locate and segment objects in images.
    Integer ``target_classes`` from the pipeline contract are translated to
    text prompts via the ``class_to_text`` dictionary.

    Args:
        version:              SAM 3 checkpoint variant: ``"sam3"`` or
                              ``"sam3.1"``.
        checkpoint_path:      Absolute path to a local ``.pt`` checkpoint.
                              ``None`` triggers an automatic HuggingFace
                              download (requires prior authentication).
        device:               Inference device (``"cuda"``, ``"cpu"``).
        confidence_threshold: Minimum detection confidence in (0, 1).
        class_to_text:        Dict mapping integer class IDs to text prompts.
                              Merged on top of the built-in COCO default map;
                              user values take precedence on collision.
    """

    def __init__(
        self,
        version: str,
        checkpoint_path: Optional[str],
        device: str,
        confidence_threshold: float,
        class_to_text: Dict[int, str],
    ) -> None:
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as exc:
            raise RuntimeError(
                "The 'sam3' package is required for Sam3Backend.  "
                "Install it with:\n"
                "  git clone https://github.com/facebookresearch/sam3.git\n"
                "  cd sam3 && pip install -e ."
            ) from exc

        self._device = device
        self._confidence_threshold = float(confidence_threshold)
        # SAM 3 checkpoint weights are saved in bfloat16.  Wrapping forward
        # passes in torch.autocast resolves the BFloat16/Float dtype mismatch
        # that occurs when strict=False loading leaves some layers in float32.
        self._autocast_device = "cuda" if device.startswith("cuda") else "cpu"

        # Build the effective class→text map: defaults overridden by user config.
        self._class_to_text: Dict[int, str] = {
            **_DEFAULT_CLASS_TO_TEXT,
            **{int(k): str(v) for k, v in class_to_text.items()},
        }

        logger.info(
            "Loading SAM 3 model (version=%s, device=%s, checkpoint=%s) …",
            version,
            device,
            checkpoint_path if checkpoint_path is not None else "HuggingFace auto-download",
        )
        try:
            model = build_sam3_image_model(
                device=device,
                eval_mode=True,
                checkpoint_path=checkpoint_path,
                load_from_HF=(checkpoint_path is None),
            )
            self._processor = Sam3Processor(
                model=model,
                device=device,
                confidence_threshold=self._confidence_threshold,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load SAM 3 model: {exc}\n"
                "If the download failed, ensure you have:\n"
                "  1. Requested access at https://huggingface.co/facebook/sam3\n"
                "  2. Authenticated with: huggingface-cli login"
            ) from exc

        logger.info("SAM 3 model loaded successfully.")

    @classmethod
    def from_config(cls, model_cfg: dict) -> "Sam3Backend":
        """Instantiate from the ``model`` section of the pipeline config.

        Args:
            model_cfg: Dict containing model configuration keys.

        Returns:
            Configured :class:`Sam3Backend` instance.
        """
        raw_map = model_cfg.get("class_to_text", {})
        # YAML may deserialise integer keys as int or str; normalise to int.
        class_to_text: Dict[int, str] = {int(k): str(v) for k, v in raw_map.items()}
        return cls(
            version=str(model_cfg.get("version", "sam3")),
            checkpoint_path=model_cfg.get("checkpoint_path", None),
            device=str(model_cfg.get("device", "cuda")),
            confidence_threshold=float(model_cfg.get("confidence_threshold", 0.5)),
            class_to_text=class_to_text,
        )

    def segment(
        self,
        image: np.ndarray,
        target_classes: List[int],
    ) -> np.ndarray:
        """Run SAM 3 on *image* and return a binary mask.

        For each integer class ID in *target_classes* the backend resolves a
        text prompt from ``class_to_text`` and runs SAM 3's detector.  All
        per-instance masks from all text prompts are unioned into a single
        binary output mask.

        The visual backbone is run **once** per call; only the detector head
        is re-executed for each distinct text prompt.

        Args:
            image:          uint8 RGB image of shape (H, W, 3).
            target_classes: List of integer class IDs to include.

        Returns:
            uint8 ndarray of shape (H, W) with values in {0, 1}.
            1 = detected target-class pixel, 0 = background.
        """
        from PIL import Image as PILImage

        H, W = image.shape[:2]
        output_mask = np.zeros((H, W), dtype=np.uint8)

        # ------------------------------------------------------------------
        # Resolve text prompts; deduplicate while preserving first occurrence.
        # ------------------------------------------------------------------
        seen_prompts: set = set()
        prompts: List[str] = []
        for cls_id in target_classes:
            prompt = self._class_to_text.get(cls_id)
            if prompt is None:
                logger.warning(
                    "Sam3Backend: no text prompt mapped to class ID %d.  "
                    "Add an entry to 'class_to_text' in the config.",
                    cls_id,
                )
                continue
            if prompt not in seen_prompts:
                prompts.append(prompt)
                seen_prompts.add(prompt)

        if not prompts:
            return output_mask

        # ------------------------------------------------------------------
        # Precompute visual backbone features once for all prompts.
        # set_image returns a state dict with ``backbone_out`` (image-only).
        # SAM 3 weights are bfloat16; autocast ensures a consistent dtype
        # across all layers regardless of strict=False loading gaps.
        # ------------------------------------------------------------------
        import torch
        pil_image = PILImage.fromarray(image)
        with torch.autocast(self._autocast_device, dtype=torch.bfloat16):
            base_state = self._processor.set_image(pil_image)

        # ------------------------------------------------------------------
        # Run the detector head independently for each text prompt.
        # set_text_prompt updates ``state["backbone_out"]`` in-place with
        # text features, so we shallow-copy the backbone_out dict before
        # each call to avoid contaminating subsequent prompts.
        # ------------------------------------------------------------------
        for prompt in prompts:
            # Shallow-copy the top-level state dict and the backbone_out
            # sub-dict so that text-feature updates stay local to this prompt.
            prompt_state: dict = dict(base_state)
            prompt_state["backbone_out"] = dict(base_state["backbone_out"])

            with torch.autocast(self._autocast_device, dtype=torch.bfloat16):
                result_state = self._processor.set_text_prompt(
                    prompt=prompt,
                    state=prompt_state,
                )

            masks_tensor = result_state.get("masks")
            if masks_tensor is None or masks_tensor.numel() == 0:
                logger.debug(
                    "SAM 3 returned no masks for prompt '%s' (no detections "
                    "above confidence_threshold=%.2f).",
                    prompt,
                    self._confidence_threshold,
                )
                continue

            # masks_tensor: bool tensor (N, 1, H_orig, W_orig)
            # Union all N instance masks across the spatial dimensions.
            union = masks_tensor.any(dim=0).squeeze(0)  # (H_orig, W_orig) bool
            output_mask = np.maximum(
                output_mask,
                union.cpu().numpy().astype(np.uint8),
            )

        return output_mask
