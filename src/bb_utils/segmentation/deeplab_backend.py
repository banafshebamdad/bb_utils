#
# File: segmentation/deeplab_backend.py
# Author: Banafshe Bamdad + GitHub Copilot (Claude Sonnet 4.6)
# Created: 2026-05-12 CET
#

"""
DeepLabv3+ semantic segmentation backend with an explicit ASPP module.

Architecture
------------
DeepLabv3+ (Chen et al., 2018) extends DeepLabv3 with an encoder–decoder
structure:

  Input image
    │
    ▼
  ResNet backbone (with dilated convolutions, output_stride=8)
    ├─► layer1  ──────────────────────────────► low-level features (256 ch)
    │                                                │
    └─► layer4  ──► ASPP ──► (1/8 feature map)    │
                                │                    │
                                ▼                    ▼
                          DeepLabV3+ Decoder ──── concat
                                │
                                ▼
                          logits (H/4, W/4)
                                │
                           bilinear upsample
                                │
                                ▼
                          logits (H, W)

The ASPP module (Atrous Spatial Pyramid Pooling) captures multi-scale context
by running five parallel branches on the high-level backbone features:
  1. 1×1 convolution  (rate=1)
  2. 3×3 atrous conv  (rate=rates[0])
  3. 3×3 atrous conv  (rate=rates[1])
  4. 3×3 atrous conv  (rate=rates[2])
  5. Global average pooling branch

All branches are concatenated and projected back to `aspp_channels`.

Pretrained weights
------------------
``pretrained_weights="coco_voc"`` loads **torchvision's fully-pretrained
DeepLabV3** model (backbone + ASPP + classifier head, all pretrained on COCO
and fine-tuned on Pascal VOC).  In this mode the torchvision DeepLabV3 model
is used directly — no custom decoder is involved, so inference works
out-of-the-box with no fine-tuning required.

To use the full DeepLabv3+ architecture (backbone + ASPP + decoder) with
custom weights, save the state-dict of a fine-tuned
:class:`_DeepLabV3PlusModel` to a ``.pth`` file and point
``pretrained_weights`` at that path.

    pretrained_weights: "/path/to/deeplab_plus.pth"

Class indices
-------------
When using the default COCO/VOC pretrained weights the label space follows
Pascal VOC (21 classes):

    0  background
    1  aeroplane   2  bicycle     3  bird        4  boat
    5  bottle      6  bus         7  car          8  cat
    9  chair      10  cow        11  diningtable 12  dog
   13  horse      14  motorbike  15  person      16  potted plant
   17  sheep      18  sofa       19  train       20  tv/monitor

**Person (pedestrian) = class 15.**  Set ``target_classes: [15]`` in the
pipeline config (unlike YOLO / Mask2Former which use ``target_classes: [0]``).

Inference flow
--------------
1. Normalise the input image to ImageNet statistics and batch it.
2. Forward pass → raw logits of shape (H, W, num_classes).
3a. ``segmentation_mode="argmax"`` (default): argmax over classes; pixel is
    foreground if the winning class is in ``target_classes``.
3b. ``segmentation_mode="threshold"``: softmax over classes; pixel is
    foreground if the max probability for any target class ≥ ``mask_threshold``.
4. Return uint8 (H, W) mask with values in {0, 1}.

Output contract compliance
--------------------------
``segment`` satisfies the ``SegmentationBackend`` contract:
- Returns uint8 (H, W) with values in {0, 1}.
- Shape matches the input image dimensions.
- No detections → all-zeros mask.

Config keys (all under ``model:``)
-----------------------------------
  ``backbone``          str   — ``"resnet50"`` or ``"resnet101"``.
                                Default: ``"resnet101"``.
  ``pretrained_weights``str   — ``"coco_voc"`` to load the complete torchvision
                                pretrained DeepLabV3 model (backbone + ASPP +
                                classifier head, fully trained — no random
                                decoder); an absolute path to a ``.pth``
                                state-dict to load a custom DeepLabv3+
                                checkpoint (backbone + ASPP + decoder); or
                                ``null`` for random initialisation.
                                Default: ``"coco_voc"``.
  ``device``            str   — ``"cuda"``, ``"cpu"``, or ``"cuda:0"``.
                                Default: ``"cpu"``.
  ``num_classes``       int   — Number of output classes.  Default: ``21``
                                (Pascal VOC / COCO-VOC label space).
  ``atrous_rates``      list  — Three ASPP dilation rates.
                                Default: ``[12, 24, 36]`` (matches torchvision
                                pretrained; use ``[6, 12, 18]`` for custom
                                checkpoints trained with output_stride=16).
  ``aspp_channels``     int   — ASPP output (and projection) channels.
                                Default: ``256``.
  ``segmentation_mode`` str   — ``"argmax"`` or ``"threshold"``.
                                Default: ``"argmax"``.
  ``mask_threshold``    float — Confidence threshold for ``"threshold"`` mode.
                                Default: ``0.5``.

Dependencies
------------
    pip install torch torchvision

``torch`` and ``torchvision`` are not installed by bb_utils and must be
present in the environment before using this backend.
"""

import logging
from typing import List, Tuple

import numpy as np

from bb_utils.segmentation.base import SegmentationBackend

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported torchvision pretrained model descriptors
# ---------------------------------------------------------------------------

# Maps a user-facing ``pretrained_weights`` shorthand to
# (backbone_name, torchvision_fn_name, torchvision_weights_enum_str).
# The backbone_name here is the torchvision resnet variant used internally.
_COCO_VOC_PRETRAINS = {
    "resnet50": (
        "torchvision.models.segmentation",
        "deeplabv3_resnet50",
        "DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1",
    ),
    "resnet101": (
        "torchvision.models.segmentation",
        "deeplabv3_resnet101",
        "DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1",
    ),
}

# Default atrous rates used by torchvision DeepLabV3 (output_stride=8)
_DEFAULT_ATROUS_RATES = (12, 24, 36)

# ImageNet normalisation constants (mean, std) used for all torchvision models
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# ASPP building blocks
# ---------------------------------------------------------------------------

class ASPPConv:
    """Atrous (dilated) 3×3 convolution + BatchNorm + ReLU block.

    This helper is used by :class:`ASPP` to build the parallel atrous branches.
    It is a plain ``nn.Sequential`` subclass so that its weights share the
    same key structure as torchvision's ``ASPPConv``, enabling direct weight
    loading from torchvision pretrained checkpoints.
    """

    # The actual class is created dynamically inside _build_aspp_modules() to
    # avoid a hard import of torch at module load time.  The docstring above
    # serves as public documentation.


class ASPPPooling:
    """Global average-pooling branch for :class:`ASPP`.

    Applies adaptive average pooling to size (1, 1), projects to
    ``out_channels`` via a 1×1 convolution, then bilinearly upsamples back to
    the input spatial size.

    The class is a plain ``nn.Sequential`` subclass with a custom ``forward``
    so that its weights share the same key structure as torchvision's
    ``ASPPPooling``, enabling direct weight loading from pretrained checkpoints.
    """


class ASPP:
    """Atrous Spatial Pyramid Pooling (ASPP) module.

    Aggregates multi-scale context from a high-level feature map by running
    five parallel branches:

    - Branch 0: 1×1 convolution (rate=1, captures local context)
    - Branch 1–3: 3×3 atrous convolutions at ``atrous_rates[0..2]``
    - Branch 4: global average pooling followed by 1×1 projection and
      bilinear upsampling back to the input spatial resolution

    All five branch outputs are concatenated along the channel axis and then
    projected to ``out_channels`` by a 1×1 convolution followed by BN, ReLU,
    and a 0.5 Dropout.

    The internal attribute layout (``convs: nn.ModuleList``, ``project:
    nn.Sequential``) is intentionally identical to torchvision's private
    ``ASPP`` implementation so that ``state_dict`` keys are compatible and
    pretrained torchvision weights can be loaded directly.

    Args:
        in_channels:  Number of channels in the input feature map (default
                      2048, matching ResNet50/101 ``layer4`` output).
        out_channels: Number of output channels for every branch and for the
                      final projection (default 256).
        atrous_rates: Three-element tuple of dilation rates for the atrous
                      branches.  Use ``(12, 24, 36)`` when the backbone was
                      trained with ``output_stride=8`` (torchvision default);
                      use ``(6, 12, 18)`` for ``output_stride=16``.
    """


# ---------------------------------------------------------------------------
# DeepLabv3+ decoder
# ---------------------------------------------------------------------------

class _DeepLabV3PlusDecoder:
    """Decoder for DeepLabv3+.

    Fuses high-level ASPP features with low-level backbone features (the
    "+" component absent in the original DeepLabv3):

    1. Project low-level features (256 ch) → 48 ch via 1×1 conv + BN + ReLU.
    2. Bilinearly upsample ASPP output to the spatial size of the projected
       low-level features (1/4 input resolution).
    3. Concatenate → (aspp_channels + 48) channels.
    4. Two 3×3 convs + BN + ReLU to refine the fused representation.
    5. 1×1 conv to produce per-class logits.

    The final spatial size is 1/4 of the input image; the backend upsamples
    to full resolution inside the model's ``forward`` method.
    """


# ---------------------------------------------------------------------------
# Full DeepLabv3+ model (internal, not part of the public API)
# ---------------------------------------------------------------------------

class _DeepLabV3PlusModel:
    """Full DeepLabv3+ model: ResNet backbone + ASPP + decoder.

    Constructed from a backbone name; weights can be loaded from torchvision
    pretrained DeepLabV3 checkpoints or from a custom ``.pth`` state-dict.
    """


# ---------------------------------------------------------------------------
# Actual nn.Module implementations (torch imported lazily)
# ---------------------------------------------------------------------------

def _build_nn_modules() -> None:
    """Patch the stub classes above with real ``nn.Module`` implementations.

    Called once on first use of :class:`DeepLabBackend` so that ``torch`` and
    ``torchvision`` are not imported at package load time.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # ------------------------------------------------------------------
    # ASPPConv
    # ------------------------------------------------------------------
    class _ASPPConv(nn.Sequential):
        def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
            super().__init__(
                nn.Conv2d(
                    in_channels, out_channels, 3,
                    padding=dilation, dilation=dilation, bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    # ------------------------------------------------------------------
    # ASPPPooling
    # ------------------------------------------------------------------
    class _ASPPPooling(nn.Sequential):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[override]
            size = x.shape[-2:]
            x = super().forward(x)
            return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    # ------------------------------------------------------------------
    # ASPP
    # ------------------------------------------------------------------
    class _ASPP(nn.Module):
        def __init__(
            self,
            in_channels: int = 2048,
            out_channels: int = 256,
            atrous_rates: Tuple[int, int, int] = _DEFAULT_ATROUS_RATES,
        ) -> None:
            super().__init__()
            # Build the five branches:
            # index 0   → 1×1 conv (local context)
            # indices 1-3 → atrous convs at each rate in atrous_rates
            # index 4   → global average pooling
            convs: List[nn.Module] = [
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            ]
            for rate in atrous_rates:
                convs.append(_ASPPConv(in_channels, out_channels, rate))
            convs.append(_ASPPPooling(in_channels, out_channels))

            # ``convs`` attribute name and layout must stay identical to
            # torchvision's private ASPP so that ``state_dict`` keys match
            # and pretrained weights can be loaded without remapping.
            self.convs = nn.ModuleList(convs)
            self.project = nn.Sequential(
                nn.Conv2d(len(convs) * out_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.project(
                torch.cat([conv(x) for conv in self.convs], dim=1)
            )

    # ------------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------------
    class _Decoder(nn.Module):
        def __init__(
            self,
            low_level_channels: int = 256,
            aspp_channels: int = 256,
            num_classes: int = 21,
        ) -> None:
            super().__init__()
            # Project low-level features to a much smaller channel count so
            # they don't dominate the ASPP features after concatenation.
            self.low_proj = nn.Sequential(
                nn.Conv2d(low_level_channels, 48, 1, bias=False),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True),
            )
            # Two 3×3 convs to refine the fused representation, then classify.
            self.refine = nn.Sequential(
                nn.Conv2d(aspp_channels + 48, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Conv2d(256, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Conv2d(256, num_classes, 1),
            )

        def forward(
            self,
            aspp_features: "torch.Tensor",
            low_level_features: "torch.Tensor",
        ) -> "torch.Tensor":
            # low_level_features: (B, C, H/4, W/4)
            # aspp_features:      (B, C, H/8, W/8) or (H/16, W/16)
            low = self.low_proj(low_level_features)
            aspp_up = F.interpolate(
                aspp_features,
                size=low.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            return self.refine(torch.cat([aspp_up, low], dim=1))

    # ------------------------------------------------------------------
    # Full model
    # ------------------------------------------------------------------
    class _Model(nn.Module):
        """DeepLabv3+ model: ResNet backbone + ASPP + decoder."""

        def __init__(
            self,
            backbone_name: str,
            num_classes: int,
            aspp_channels: int,
            atrous_rates: Tuple[int, int, int],
        ) -> None:
            super().__init__()
            import torchvision.models as tv_models

            _REPLACE_STRIDE = {
                # output_stride=8  (matches torchvision pretrained)
                (12, 24, 36): [False, True, True],
                # output_stride=16 (faster; use [6,12,18] rates)
                (6, 12, 18):  [False, False, True],
            }
            replace_stride = _REPLACE_STRIDE.get(
                tuple(atrous_rates), [False, True, True]
            )

            backbone_fn = getattr(tv_models, backbone_name, None)
            if backbone_fn is None:
                raise ValueError(
                    f"Unsupported backbone '{backbone_name}'. "
                    "Supported values: 'resnet50', 'resnet101'."
                )
            # Load bare backbone (no pretrained weights here; loaded separately
            # when the user requests 'coco_voc').
            backbone = backbone_fn(
                weights=None,
                replace_stride_with_dilation=replace_stride,
            )

            # Stem: conv1 + bn1 + relu + maxpool
            self.backbone_stem = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
            )
            # Residual stages
            self.layer1 = backbone.layer1  # 256 ch  – low-level features
            self.layer2 = backbone.layer2
            self.layer3 = backbone.layer3
            self.layer4 = backbone.layer4  # 2048 ch – ASPP input

            low_level_ch = backbone.layer1[-1].bn3.num_features  # 256 for both 50 and 101

            self.aspp    = _ASPP(2048, aspp_channels, atrous_rates)
            self.decoder = _Decoder(low_level_ch, aspp_channels, num_classes)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            input_size = x.shape[-2:]

            x           = self.backbone_stem(x)
            low_level   = self.layer1(x)
            x           = self.layer2(low_level)
            x           = self.layer3(x)
            x           = self.layer4(x)

            aspp_out = self.aspp(x)
            logits   = self.decoder(aspp_out, low_level)

            return F.interpolate(
                logits,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )

    # Patch the module-level stubs with the real implementations.
    globals()["ASPPConv"]          = _ASPPConv
    globals()["ASPPPooling"]       = _ASPPPooling
    globals()["ASPP"]              = _ASPP
    globals()["_DeepLabV3PlusDecoder"]  = _Decoder
    globals()["_DeepLabV3PlusModel"]    = _Model


_NN_MODULES_BUILT = False


def _ensure_nn_modules() -> None:
    global _NN_MODULES_BUILT
    if not _NN_MODULES_BUILT:
        _build_nn_modules()
        _NN_MODULES_BUILT = True


# ---------------------------------------------------------------------------
# Weight loading helpers
# ---------------------------------------------------------------------------

def _load_full_tv_model(backbone_name: str, num_classes: int):
    """Load the complete, fully-pretrained torchvision DeepLabV3 model.

    Unlike :func:`_load_torchvision_pretrained`, which only transfers backbone
    and ASPP weights into the custom :class:`_DeepLabV3PlusModel` (leaving the
    decoder randomly initialised), this function returns the **entire**
    torchvision ``DeepLabV3`` object — backbone + ASPP + classifier head —
    with all weights pretrained.  This is the correct model to use when
    ``pretrained_weights="coco_voc"`` because it produces meaningful
    predictions immediately, without any fine-tuning.

    The torchvision model's ``forward`` returns an ``OrderedDict`` with key
    ``"out"`` containing the logit tensor of shape ``(B, num_classes, H, W)``.
    :class:`DeepLabBackend` handles this automatically via its
    ``_is_tv_model`` flag.

    Args:
        backbone_name: ``"resnet50"`` or ``"resnet101"``.
        num_classes:   Must be 21 for the COCO/VOC pretrained weights.

    Returns:
        Fully-pretrained ``torchvision.models.segmentation.DeepLabV3`` model
        in eval mode (not yet moved to device).

    Raises:
        RuntimeError:  If torchvision is unavailable or the download fails.
        ValueError:    If ``backbone_name`` is not supported or ``num_classes``
                       is not 21 (the pretrained head has 21 output classes).
    """
    if num_classes != 21:
        raise ValueError(
            f"pretrained_weights='coco_voc' requires num_classes=21 "
            f"(Pascal VOC label space), got {num_classes}.  "
            "Set num_classes=21 or use a custom .pth checkpoint."
        )

    entry = _COCO_VOC_PRETRAINS.get(backbone_name)
    if entry is None:
        raise ValueError(
            f"No torchvision COCO/VOC pretrained weights for backbone "
            f"'{backbone_name}'.  Supported: {sorted(_COCO_VOC_PRETRAINS)}."
        )

    _pkg, fn_name, weights_str = entry
    try:
        import torchvision.models.segmentation as tv_seg
        from torchvision.models import get_weight
    except ImportError as exc:
        raise RuntimeError(
            "torchvision is required to load pretrained weights. "
            "Install with: pip install torchvision"
        ) from exc

    weights = get_weight(weights_str)
    model = getattr(tv_seg, fn_name)(weights=weights)
    model.eval()
    logger.info(
        "DeepLabBackend: loaded complete torchvision '%s' "
        "(backbone + ASPP + classifier head, fully pretrained).",
        fn_name,
    )
    return model


def _load_torchvision_pretrained(
    model: "_DeepLabV3PlusModel",  # type: ignore[name-defined]
    backbone_name: str,
) -> None:
    """Initialise *model* backbone + ASPP from a torchvision DeepLabV3 checkpoint.

    The torchvision DeepLabV3 uses exactly the same ResNet backbone structure
    and the same ASPP layout (``convs: ModuleList``, ``project: Sequential``)
    as our :class:`ASPP`, so the state-dict keys map 1-to-1 after a simple
    prefix substitution.  The decoder is left randomly initialised because
    torchvision's DeepLabV3 has no decoder (the "+" addition).

    Key remapping::

        backbone.conv1.*         → backbone_stem.0.*
        backbone.bn1.*           → backbone_stem.1.*
        backbone.layer{1-4}.*   → layer{1-4}.*
        classifier.0.*           → aspp.*
        (all other keys skipped)

    Args:
        model:         The :class:`_DeepLabV3PlusModel` instance to fill.
        backbone_name: ``"resnet50"`` or ``"resnet101"``; selects which
                       torchvision weight set to download.

    Raises:
        RuntimeError: If torchvision or torch is unavailable, or if the
                      weight download fails.
    """
    import torch

    entry = _COCO_VOC_PRETRAINS.get(backbone_name)
    if entry is None:
        raise ValueError(
            f"No torchvision COCO/VOC pretrained weights for backbone "
            f"'{backbone_name}'.  Supported: {sorted(_COCO_VOC_PRETRAINS)}."
        )

    _pkg, fn_name, weights_str = entry
    try:
        import torchvision.models.segmentation as tv_seg
        from torchvision.models import get_weight
    except ImportError as exc:
        raise RuntimeError(
            "torchvision is required to load pretrained weights. "
            "Install with: pip install torchvision"
        ) from exc

    weights = get_weight(weights_str)
    tv_model = getattr(tv_seg, fn_name)(weights=weights)
    tv_sd = tv_model.state_dict()

    remap: dict = {}
    for key, val in tv_sd.items():
        if key.startswith("backbone.conv1."):
            remap["backbone_stem.0." + key[len("backbone.conv1."):]] = val
        elif key.startswith("backbone.bn1."):
            remap["backbone_stem.1." + key[len("backbone.bn1."):]] = val
        elif key.startswith("backbone.layer"):
            remap[key[len("backbone."):]] = val   # "layer1.xxx" → "layer1.xxx"
        elif key.startswith("classifier.0."):
            remap["aspp." + key[len("classifier.0."):]] = val
        # Skip classifier.1.* (final per-class conv; different num_classes)
        # and auxiliary head keys.

    missing, unexpected = model.load_state_dict(remap, strict=False)
    decoder_keys = {k for k in missing if k.startswith("decoder.")}
    truly_missing = set(missing) - decoder_keys
    if truly_missing:
        logger.warning(
            "DeepLabBackend: %d backbone/ASPP keys not found in torchvision "
            "checkpoint: %s", len(truly_missing), sorted(truly_missing)[:10],
        )
    if unexpected:
        logger.debug(
            "DeepLabBackend: %d unexpected keys ignored: %s",
            len(unexpected), sorted(unexpected)[:10],
        )
    logger.info(
        "DeepLabBackend: loaded torchvision '%s' weights "
        "(backbone + ASPP; decoder randomly initialised).",
        fn_name,
    )


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------

class DeepLabBackend(SegmentationBackend):
    """Semantic segmentation backend using DeepLabv3+ with an ASPP module.

    Two operational modes depending on ``pretrained_weights``:

    * ``pretrained_weights="coco_voc"`` (default): wraps the **complete**
      torchvision DeepLabV3 model (backbone + ASPP + classifier head, all
      weights pretrained on COCO/VOC).  No decoder is used; the model
      produces useful predictions immediately.  This is the recommended
      mode for pedestrian segmentation with ``target_classes=[15]``.

    * ``pretrained_weights="/path/to/checkpoint.pth"`` or ``None``: builds
      the full custom DeepLabv3+ (backbone + ASPP + decoder) and loads the
      provided state-dict (or uses random initialisation).  Use this mode
      when you have a fine-tuned checkpoint that includes decoder weights.

    Args:
        backbone:           ``"resnet50"`` or ``"resnet101"`` (default).
        pretrained_weights: ``"coco_voc"`` to use the complete pretrained
                            torchvision DeepLabV3; a path string to a ``.pth``
                            state-dict to load a custom DeepLabv3+ checkpoint;
                            ``None`` for random initialisation.
        device:             Inference device (``"cuda"``, ``"cpu"``).
        num_classes:        Number of output classes (default 21 for COCO/VOC).
                            Must be 21 when ``pretrained_weights="coco_voc"``.
        atrous_rates:       Three ASPP dilation rates (default ``(12, 24, 36)``).
                            Only used for the custom model path.
        aspp_channels:      ASPP output channels (default 256).
                            Only used for the custom model path.
        segmentation_mode:  ``"argmax"`` (default) or ``"threshold"``.
        mask_threshold:     Confidence threshold for ``"threshold"`` mode
                            (default 0.5).
    """

    def __init__(
        self,
        backbone: str = "resnet101",
        pretrained_weights: str = "coco_voc",
        device: str = "cpu",
        num_classes: int = 21,
        atrous_rates: Tuple[int, int, int] = _DEFAULT_ATROUS_RATES,
        aspp_channels: int = 256,
        segmentation_mode: str = "argmax",
        mask_threshold: float = 0.5,
    ) -> None:
        import torch

        self._device    = device
        self._seg_mode  = segmentation_mode
        self._thresh    = float(mask_threshold)
        self._torch     = torch
        # Tracks whether self._model is a torchvision DeepLabV3 (returns a
        # dict with key "out") vs. the custom _DeepLabV3PlusModel (returns
        # a plain tensor).
        self._is_tv_model = False

        if pretrained_weights == "coco_voc":
            # Use the complete, fully-pretrained torchvision model.
            # The custom decoder is NOT built or used in this path.
            try:
                self._model = _load_full_tv_model(backbone, num_classes)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load torchvision DeepLabV3 "
                    f"(backbone={backbone!r}): {exc}"
                ) from exc
            self._is_tv_model = True
        else:
            # Custom DeepLabv3+ with decoder (backbone + ASPP + decoder).
            _ensure_nn_modules()
            try:
                self._model = _DeepLabV3PlusModel(  # type: ignore[operator]
                    backbone_name=backbone,
                    num_classes=num_classes,
                    aspp_channels=aspp_channels,
                    atrous_rates=tuple(atrous_rates),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to build DeepLabV3+ model "
                    f"(backbone={backbone!r}): {exc}"
                ) from exc

            if pretrained_weights is None or pretrained_weights == "":
                logger.info("DeepLabBackend: using randomly initialised weights.")
            elif isinstance(pretrained_weights, str):
                try:
                    sd = torch.load(
                        pretrained_weights,
                        map_location="cpu",
                        weights_only=True,
                    )
                    self._model.load_state_dict(sd, strict=True)
                    logger.info(
                        "DeepLabBackend: loaded custom checkpoint '%s'.",
                        pretrained_weights,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to load DeepLabV3+ checkpoint "
                        f"'{pretrained_weights}': {exc}"
                    ) from exc
            else:
                raise ValueError(
                    f"Unsupported pretrained_weights value: {pretrained_weights!r}. "
                    "Use 'coco_voc', a path to a .pth file, or None."
                )

        self._model.eval()
        self._model.to(device)
        logger.info(
            "DeepLabBackend: ready (backbone=%s, tv_model=%s, device=%s, mode=%s).",
            backbone, self._is_tv_model, device, segmentation_mode,
        )

    # ------------------------------------------------------------------
    # SegmentationBackend interface
    # ------------------------------------------------------------------

    def segment(
        self,
        image: np.ndarray,
        target_classes: List[int],
    ) -> np.ndarray:
        """Run DeepLabv3+ on *image* and return a binary mask.

        Args:
            image:          uint8 RGB image of shape (H, W, 3).
            target_classes: Class indices to include in the mask.  With
                            COCO/VOC pretrained weights, use ``[15]`` for
                            person / pedestrian.

        Returns:
            uint8 (H, W) mask; 1 = target-class pixel, 0 = background.
        """
        import torch

        H, W = image.shape[:2]
        expected_shape = (H, W)
        mask = np.zeros(expected_shape, dtype=np.uint8)

        # Normalise to [0, 1] and apply ImageNet mean/std
        img_f = image.astype(np.float32) / 255.0
        mean = np.array(_IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std  = np.array(_IMAGENET_STD,  dtype=np.float32).reshape(1, 1, 3)
        img_f = (img_f - mean) / std

        # (H, W, 3) → (1, 3, H, W)
        tensor = torch.from_numpy(img_f.transpose(2, 0, 1)).unsqueeze(0)
        tensor = tensor.to(self._device)

        with torch.no_grad():
            out = self._model(tensor)
            # torchvision DeepLabV3 returns an OrderedDict {"out": tensor, ...};
            # the custom _DeepLabV3PlusModel returns a plain tensor.
            logits = out["out"] if self._is_tv_model else out  # (1, num_classes, H, W)

        logits_np = logits.squeeze(0).cpu().numpy()  # (num_classes, H, W)

        if self._seg_mode == "argmax":
            pred_classes = np.argmax(logits_np, axis=0)  # (H, W)
            for cls_id in target_classes:
                mask = np.bitwise_or(mask, (pred_classes == cls_id).astype(np.uint8))
        else:
            # Softmax probabilities
            logits_t = torch.from_numpy(logits_np)
            probs_np = torch.softmax(logits_t, dim=0).numpy()  # (num_classes, H, W)
            for cls_id in target_classes:
                prob_map = probs_np[cls_id]
                mask = np.bitwise_or(
                    mask, (prob_map >= self._thresh).astype(np.uint8)
                )

        return self._validate_output(mask, expected_shape)

    # ------------------------------------------------------------------
    # Factory classmethod
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, model_cfg: dict) -> "DeepLabBackend":
        """Instantiate from a ``model`` config dict section.

        Example YAML block::

            model:
              backend: "deeplab"
              backbone: "resnet101"
              pretrained_weights: "coco_voc"
              device: "cpu"
              num_classes: 21
              atrous_rates: [12, 24, 36]
              aspp_channels: 256
              segmentation_mode: "argmax"
              mask_threshold: 0.5
        """
        raw_rates = model_cfg.get("atrous_rates", list(_DEFAULT_ATROUS_RATES))
        if len(raw_rates) != 3:
            raise ValueError(
                f"'atrous_rates' must have exactly 3 elements, got {raw_rates}."
            )
        return cls(
            backbone=model_cfg.get("backbone", "resnet101"),
            pretrained_weights=model_cfg.get("pretrained_weights", "coco_voc"),
            device=model_cfg.get("device", "cpu"),
            num_classes=model_cfg.get("num_classes", 21),
            atrous_rates=tuple(raw_rates),
            aspp_channels=model_cfg.get("aspp_channels", 256),
            segmentation_mode=model_cfg.get("segmentation_mode", "argmax"),
            mask_threshold=model_cfg.get("mask_threshold", 0.5),
        )
