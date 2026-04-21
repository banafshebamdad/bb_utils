#
# File: segmentation/factory.py
# Author: Banafshe Bamdad + Claude Code
# Created: 2026-04-21 CET
#

"""
Factory for instantiating segmentation backends from a config dict.

Usage::

    from bb_utils.segmentation import create_backend

    backend = create_backend(config)
    mask = backend.segment(image_rgb, target_classes=[0])

The ``config["model"]["backend"]`` key selects the implementation.
Currently registered backends:
  ``"yolo"``  — ``YoloBackend`` (Ultralytics YOLOv8-seg)

New backends can be registered via ``register_backend(name, cls)``.
"""

import logging
from typing import Dict, Type

from bb_utils.segmentation.base import SegmentationBackend

logger = logging.getLogger(__name__)

# Registry maps backend name → backend class
_BACKEND_REGISTRY: Dict[str, Type[SegmentationBackend]] = {}


def register_backend(name: str, cls: Type[SegmentationBackend]) -> None:
    """Register a backend class under *name*."""
    _BACKEND_REGISTRY[name] = cls


def create_backend(config: Dict) -> SegmentationBackend:
    """Instantiate the segmentation backend specified in *config*.

    Reads ``config["model"]["backend"]`` and passes the full
    ``config["model"]`` dict to the backend constructor.

    Args:
        config: Full pipeline config dict containing a ``"model"`` section.

    Returns:
        Configured :class:`SegmentationBackend` instance.

    Raises:
        KeyError:   If ``config["model"]`` or ``config["model"]["backend"]``
                    is absent.
        ValueError: If the requested backend name is not registered.
    """
    model_cfg = config.get("model")
    if model_cfg is None:
        raise KeyError("Config section 'model' is missing.")

    backend_name = model_cfg.get("backend")
    if backend_name is None:
        raise KeyError("Config key 'model.backend' is missing.")

    cls = _BACKEND_REGISTRY.get(backend_name)
    if cls is None:
        known = sorted(_BACKEND_REGISTRY.keys())
        raise ValueError(
            f"Unknown segmentation backend '{backend_name}'.  "
            f"Registered backends: {known}."
        )

    logger.info("Creating segmentation backend: %s", backend_name)
    return _instantiate(cls, model_cfg)


def _instantiate(cls: Type[SegmentationBackend], model_cfg: Dict) -> SegmentationBackend:
    """Instantiate *cls* from *model_cfg*, mapping config keys to constructor args."""
    return cls(
        model_name=model_cfg.get("model_name", "yolov8n-seg"),
        device=model_cfg.get("device", "cpu"),
        confidence_threshold=model_cfg.get("confidence_threshold"),
        iou_threshold=model_cfg.get("iou_threshold"),
    )


# Register built-in backends lazily to avoid importing heavy dependencies
# at module load time.
def _register_builtin_backends() -> None:
    from bb_utils.segmentation.yolo_backend import YoloBackend
    register_backend("yolo", YoloBackend)


_register_builtin_backends()
