# Segmentation Backend

`bb_utils.segmentation` provides a common interface for running instance and semantic segmentation models on RGB images, normalising each model's output into a single binary mask format that callers (e.g. `bb-run-segmentation`) can consume without knowing which model produced it.

---

## Table of Contents

- [Output interface](#output-interface)
- [Architecture](#architecture)
- [Available backends](#available-backends)
  - [YoloBackend](#yolobackend)
  - [Mask2FormerBackend](#mask2formerbackend)
  - [DeepLabBackend](#deeplabbackend)
  - [Sam3Backend](#sam3backend)
- [CLI usage](#cli-usage)
- [Config format](#config-format)
- [Output files](#output-files)
- [Mask utilities](#mask-utilities)
- [Adding a new backend](#adding-a-new-backend)
- [Dependencies](#dependencies)

---

## Output contract

Every backend satisfies the same interface defined in `base.SegmentationBackend`:

```
mask = backend.segment(image, target_classes)
```

| Property | Value |
|---|---|
| Input `image` | `np.ndarray`, shape `(H, W, 3)`, dtype `uint8`, colour order **RGB** |
| Input `target_classes` | `List[int]`: model-specific class indices to include |
| Output `mask` | `np.ndarray`, shape `(H, W)`, dtype `uint8`, values in `{0, 1}` |
| Output semantics | `1` = pixel belongs to a detected instance of a target class; `0` = background |
| No detections | Returns an all-zeros mask (valid result, not an error) |
| Spatial alignment | `mask[i, j]` corresponds to `image[i, j]`, no spatial transformation applied |

The backend does **not** apply dilation and does **not** convert grayscale input
to RGB; both are the caller's responsibility.

---

## Architecture

The package follows a **registry + factory** pattern so that new backends can
be added without modifying any caller code.

```
caller
  └── create_backend(config)          ← factory.py
        ├── reads config["model"]["backend"]
        └── looks up _BACKEND_REGISTRY[name]
              └── instantiates the registered SegmentationBackend subclass
                    └── .segment(image, target_classes) → uint8 (H, W) mask
```

New backends are registered at import time:

```python
from bb_utils.segmentation.factory import register_backend
from my_package.my_backend import MyBackend

register_backend("my_backend", MyBackend)
```

---

## Available backends

### YoloBackend

**Module**: `bb_utils.segmentation.yolo_backend`  
**Config key**: `"yolo"`  
**Dependency**: `ultralytics` (`pip install ultralytics`)

Wraps Ultralytics YOLOv8-seg (or any later `*-seg` variant). Returns the
union of all per-instance masks whose predicted class is in `target_classes`.

**Class indices** follow the model's training dataset. For the default
YOLOv8 weights (COCO): class `0` = `person`.

**Inference flow:**

1. `model.predict(image_rgb, conf=..., iou=..., device=...)`, runs NMS internally.
2. For each detected instance whose class is in `target_classes`, extract the
   float instance mask at model resolution.
3. Threshold at `mask_threshold` (default `0.5`) → binary; resize to input `(H, W)` with nearest-neighbour
   if the model output resolution differs.
4. Accumulate via bitwise OR into the output mask.

**Config keys** (all under `model:`):

| Key | Type | Required | Default | Description |
|---|---|---|---|---|
| `backend` | str | yes | — | Must be `"yolo"` |
| `model_name` | str | no | `"yolov8n-seg"` | Model identifier or path accepted by `YOLO()` |
| `device` | str | no | `"cpu"` | `"cuda"`, `"cuda:0"`, `"cpu"` |
| `confidence_threshold` | float | **yes** | — | Minimum detection confidence (0, 1) |
| `iou_threshold` | float | **yes** | — | NMS IoU threshold (0, 1) |
| `mask_threshold` | float | no | `0.5` | Pixel-level binarization threshold applied to the raw float instance mask (0, 1) |

---

### Mask2FormerBackend

**Module**: `bb_utils.segmentation.mask2former_backend`  
**Config key**: `"mask2former"`  
**Dependencies**: `transformers`, `torch` (`pip install transformers torch`)

Wraps [HuggingFace Mask2Former](https://huggingface.co/docs/transformers/model_doc/mask2former) for instance or panoptic segmentation. Any `Mask2FormerForUniversalSegmentation` checkpoint from the HuggingFace Hub is supported. Weights are downloaded automatically on first use.

**Class indices** follow COCO (same convention as YOLOv8-seg): class `0` = `person`.

**Inference flow:**

1. Preprocess image via `AutoImageProcessor`; run model forward pass.
2. Post-process via `post_process_instance_segmentation` or `post_process_panoptic_segmentation` depending on `segmentation_type`.
3. For each segment whose `label_id` is in `target_classes` and score ≥ `confidence_threshold`, merge it into the output mask via bitwise OR.

**Config keys** (all under `model:`):

| Key | Type | Required | Default | Description |
|---|---|---|---|---|
| `backend` | str | yes | — | Must be `"mask2former"` |
| `model_name` | str | no | `"facebook/mask2former-swin-tiny-coco-instance"` | HuggingFace repo ID or local path |
| `device` | str | no | `"cpu"` | `"cuda"`, `"cuda:0"`, `"cpu"` |
| `confidence_threshold` | float | no | `0.5` | Minimum prediction score (0, 1) |
| `segmentation_type` | str | no | auto | `"instance"` or `"panoptic"`, auto-detected from `model_name` |

`iou_threshold` is not used (Mask2Former does not apply NMS).

**Recommended checkpoints for pedestrian masking:**

```
facebook/mask2former-swin-tiny-coco-instance    # fastest
facebook/mask2former-swin-small-coco-instance
facebook/mask2former-swin-base-coco-instance
facebook/mask2former-swin-large-coco-instance   # most accurate

facebook/mask2former-swin-large-coco-panoptic   # panoptic; see note below
```

> **Note: panoptic checkpoint and safetensors workaround**
>
> `facebook/mask2former-swin-large-coco-panoptic` (and other panoptic checkpoints)
> ship only `pytorch_model.bin`, not `model.safetensors`.
> Since `transformers` ≥ 5.6 blocks loading `.bin` files with `torch` < 2.6
> (CVE-2025-32434), using the Hub ID directly will fail unless torch is upgraded.
>
> **Workaround**: download the safetensors weights from the model's open
> [PR#3](https://huggingface.co/facebook/mask2former-swin-large-coco-panoptic/discussions/3)
> and point `model_name` at the local directory:
>
> ```python
> from huggingface_hub import snapshot_download
> snapshot_download(
>     "facebook/mask2former-swin-large-coco-panoptic",
>     revision="refs/pr/3",
>     local_dir="~/.cache/huggingface/hub/mask2former-swin-large-coco-panoptic-safetensors",
>     ignore_patterns=["*.bin", "*.msgpack", "flax_model*"],
> )
> ```
>
> Then in the config:
>
> ```yaml
> model_name: "~/.cache/huggingface/hub/mask2former-swin-large-coco-panoptic-safetensors"
> segmentation_type: "panoptic"
> ```
>
> This is machine-specific. The permanent fix is to upgrade torch to ≥ 2.6.

---

### DeepLabBackend

**Module**: `bb_utils.segmentation.deeplab_backend`  
**Config key**: `"deeplab"`  
**Dependencies**: `torch`, `torchvision` (`pip install torch torchvision`)

Custom DeepLabv3+ implementation with an explicit **ASPP (Atrous Spatial
Pyramid Pooling)** module.  The backend operates in two modes depending on
`pretrained_weights`:

- **`"coco_voc"` (default)**: uses the **complete, fully-pretrained
  torchvision DeepLabV3** model (backbone + ASPP + classifier head).  The
  custom decoder is not built.  Predictions are useful immediately.
- **Custom `.pth` checkpoint or `null`**: builds the full **DeepLabv3+**
  model (backbone + ASPP + `_DeepLabV3PlusDecoder`) and loads the provided
  weights (or random initialisation).

**Architecture (`pretrained_weights: "coco_voc"`):**

```
Input image
  │
  ▼
ResNet backbone (dilated, output_stride=8)
  └─► layer4 ──► ASPP ──► (H/8, W/8)
                    │
                    ▼
              classifier head (1×1 conv, 21 classes)
                    │
                    ▼
              logits (H/8, W/8) ──► bilinear upsample ──► (H, W)
```

**Architecture (custom `.pth` checkpoint — DeepLabv3+ with decoder):**

```
Input image
  │
  ▼
ResNet backbone (dilated, output_stride=8)
  ├─► layer1 ──────────────────────────► low-level features (256 ch)
  │                                            │
  └─► layer4 ──► ASPP ──► (H/8, W/8)        │
                    │                           │
                    ▼                           ▼
              Decoder (concat + 2× 3×3 conv)──┘
                    │
                    ▼
              logits (H/4, W/4) ──► bilinear upsample ──► (H, W)
```

The **ASPP** module applies five parallel branches to the high-level feature
map:

| Branch | Operation | Purpose |
|---|---|---|
| 0 | 1×1 conv | Local context |
| 1 | 3×3 atrous conv, rate=12 | Medium-scale context |
| 2 | 3×3 atrous conv, rate=24 | Large-scale context |
| 3 | 3×3 atrous conv, rate=36 | Very large-scale context |
| 4 | Global avg pool + 1×1 conv | Image-level context |

All five outputs are concatenated and projected to 256 channels.

> The default rates `(12, 24, 36)` match the torchvision pretrained
> checkpoint (trained with `output_stride=8`).  Use `(6, 12, 18)` for
> checkpoints trained with `output_stride=16`.

**Class indices** follow the **Pascal VOC label space** (21 classes) used by
the torchvision COCO pretrained weights — *different from YOLO / Mask2Former*:

| Class ID | Label |
|---|---|
| **15** | **person / pedestrian** |
| 0 | background |
| 1–14, 16–20 | other VOC categories |

Use `target_classes: [15]` for pedestrian masking with the default pretrained
weights.

**Inference flow:**

1. Normalise input to ImageNet statistics (mean/std).
2. Forward pass through backbone → ASPP → classifier head (torchvision mode)
   or backbone → ASPP → decoder (custom checkpoint mode) → upsample to `(H, W)`.
3a. `segmentation_mode="argmax"` (default): argmax over classes; pixel is
    foreground if the winning class is in `target_classes`.
3b. `segmentation_mode="threshold"`: softmax; pixel is foreground if the
    max probability for any target class ≥ `mask_threshold`.

**Config keys** (all under `model:`):

| Key | Type | Required | Default | Description |
|---|---|---|---|---|
| `backend` | str | yes | — | Must be `"deeplab"` |
| `backbone` | str | no | `"resnet101"` | `"resnet50"` or `"resnet101"` |
| `pretrained_weights` | str | no | `"coco_voc"` | `"coco_voc"` (complete torchvision pretrained model — fully trained, no random decoder), path to `.pth` state-dict for a custom DeepLabv3+ checkpoint, or `null` |
| `device` | str | no | `"cpu"` | `"cuda"`, `"cuda:0"`, `"cpu"` |
| `num_classes` | int | no | `21` | Number of output classes |
| `atrous_rates` | list | no | `[12, 24, 36]` | Three ASPP dilation rates — ignored when `pretrained_weights: "coco_voc"` |
| `aspp_channels` | int | no | `256` | ASPP output channels — ignored when `pretrained_weights: "coco_voc"` |
| `segmentation_mode` | str | no | `"argmax"` | `"argmax"` or `"threshold"` |
| `mask_threshold` | float | no | `0.5` | Confidence threshold for `"threshold"` mode |

**Pretrained weight loading (`pretrained_weights: "coco_voc"`):**

When `pretrained_weights` is `"coco_voc"`, the backend loads the **complete**
torchvision `DeepLabV3_ResNet50/101_Weights.COCO_WITH_VOC_LABELS_V1` model
(backbone + ASPP + classifier head, all fully pretrained).  The torchvision
model is used directly — the custom decoder is not built and no fine-tuning
is required.  The model produces useful predictions immediately.

The custom decoder (`_DeepLabV3PlusDecoder`) is only used when
`pretrained_weights` points to a `.pth` state-dict file containing a fully
fine-tuned DeepLabv3+ checkpoint.

> **Note**: `atrous_rates` and `aspp_channels` are ignored when
> `pretrained_weights: "coco_voc"` — the torchvision model's architecture
> is fixed.  These parameters only take effect for custom `.pth` checkpoints.

**Example YAML config for pedestrian segmentation:**

```yaml
model:
  backend: "deeplab"
  backbone: "resnet101"
  pretrained_weights: "coco_voc"
  device: "cpu"
  num_classes: 21
  atrous_rates: [12, 24, 36]
  aspp_channels: 256
  segmentation_mode: "argmax"
```

And in the pipeline config:

```yaml
segmentation:
  target_classes: [15]   # person in Pascal VOC label space
```

---

### Sam3Backend

**Module**: `bb_utils.segmentation.sam3_backend`  
**Config key**: `"sam3"`  
**Dependencies**: `sam3`, `torch` (see installation below)

Wraps [SAM 3 (Segment Anything with Concepts)](https://github.com/facebookresearch/sam3) from Meta AI.
SAM 3 is an open-vocabulary, text-prompted foundation model that can detect and segment any object described by a short text phrase, without retraining or fine-tuning.  It achieves 75–80% of human performance on the SA-Co benchmark (270 K unique concepts).

Unlike the other backends, SAM 3 does **not** use integer class indices as primary prompts.  Instead, integer `target_classes` are translated to text prompts via a configurable `class_to_text` dictionary.  The built-in default covers all 80 COCO categories (e.g. class `0` → `"person"`).

**Class indices** follow the COCO convention (same as YOLO / Mask2Former): class `0` = `person`.  The `class_to_text` map can be overridden with arbitrary open-vocabulary phrases:

```yaml
class_to_text:
  0: "pedestrian"          # more descriptive than "person"
  0: "person in a crowd"   # context-specific
```

**Inference flow:**

1. Precompute visual backbone features once per image (`Sam3Processor.set_image`).
2. For each unique text prompt (derived from `target_classes` + `class_to_text`):
   a. Shallow-copy the image-state dict (keeps backbone features, isolates text features).
   b. Run `Sam3Processor.set_text_prompt` — executes the DETR-based detector head.
   c. Filter detections: keep instances with score ≥ `confidence_threshold`.
   d. Union all per-instance boolean masks for this prompt.
3. Accumulate masks from all prompts via element-wise maximum.
4. Return `uint8 (H, W)` mask with values in `{0, 1}`.

**Authentication: required before first use:**

SAM 3 checkpoints are gated on HuggingFace and require explicit access:

1. Request access at <https://huggingface.co/facebook/sam3> and wait for approval.
2. Once approved, generate a HuggingFace access token at <https://huggingface.co/settings/tokens> (role: **Read**).
3. Authenticate locally using the `hf` CLI (`huggingface-cli` is deprecated):
   ```bash
   hf auth login
   ```
   Paste your token when prompted.  The token is saved to `~/.cache/huggingface/token`.
4. Verify authentication:
   ```bash
   hf whoami
   ```

The checkpoint (~3.4 GB for `sam3`, ~3.7 GB for `sam3.1`) is downloaded
automatically to `~/.cache/huggingface/hub/` on first use.

**Config keys** (all under `model:`):

| Key | Type | Required | Default | Description |
|---|---|---|---|---|
| `backend` | str | yes | — | Must be `"sam3"` |
| `version` | str | no | `"sam3"` | `"sam3"` (base) or `"sam3.1"` (Object Multiplex, faster multi-object tracking) |
| `checkpoint_path` | str | no | `null` | Absolute path to a local `.pt` checkpoint.  `null` = auto-download from HuggingFace |
| `device` | str | no | `"cuda"` | `"cuda"`, `"cuda:0"`, `"cpu"` |
| `confidence_threshold` | float | no | `0.5` | Minimum detection score (0, 1) |
| `class_to_text` | dict | no | `{}` | Maps integer class IDs to text prompts.  Merged on top of the built-in 80-class COCO default; user values take precedence |

`iou_threshold`, `mask_threshold`, `segmentation_mode`, and `backbone` are not used by this backend.

**Example YAML config for pedestrian segmentation:**

```yaml
model:
  backend:              "sam3"
  version:              "sam3"
  checkpoint_path:      null          # null = auto-download (requires HF auth)
  device:               "cuda"
  confidence_threshold: 0.5
  class_to_text:
    0: "person"
```

And in the pipeline config:

```yaml
segmentation:
  target_classes: [0]   # COCO class 0 = person
```

**Installation:**

```bash
# 1. Clone and install sam3
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .

# 2. Request access at https://huggingface.co/facebook/sam3 and wait for approval.
# 3. Generate a HuggingFace token at https://huggingface.co/settings/tokens (role: Read).
# 4. Authenticate (huggingface-cli is deprecated; use hf instead):
hf auth login
# 5. Verify:
hf whoami
```

> **Note: SAM 3 requires Python ≥ 3.12 and PyTorch ≥ 2.7 (CUDA 12.6+).**
> See the [SAM 3 README](https://github.com/facebookresearch/sam3#installation)
> for full prerequisites.

---

## CLI usage

**Prerequisites**: install the backend dependency before running:

```bash
pip install ultralytics          # required for YoloBackend (default)
pip install transformers torch   # required for Mask2FormerBackend
pip install torch torchvision    # required for DeepLabBackend
# For Sam3Backend: clone and install from source (see Sam3Backend section above)
```

```bash
# Segment a directory of images
bb-run-segmentation \
  --config configs/preprocessing_segmentation.yaml \
  --images-dir /path/to/images \
  --output-dir /path/to/masks

# Process only one sequence from a mixed flat directory
bb-run-segmentation \
  --config configs/preprocessing_segmentation.yaml \
  --images-dir /path/to/images \
  --output-dir /path/to/masks \
  --sequence Kiko_loop_R

# Overwrite already-generated masks
bb-run-segmentation \
  --config configs/preprocessing_segmentation.yaml \
  --images-dir /path/to/images \
  --output-dir /path/to/masks \
  --force

# Validate config and paths without running inference
bb-run-segmentation \
  --config configs/preprocessing_segmentation.yaml \
  --images-dir /path/to/images \
  --output-dir /path/to/masks \
  --dry-run

# For a split-based dataset, call once per split
bb-run-segmentation --config configs/preprocessing_segmentation.yaml \
  --images-dir dataset/incrowdvi/images/train \
  --output-dir dataset/incrowdvi/semantic_masks/train
bb-run-segmentation --config configs/preprocessing_segmentation.yaml \
  --images-dir dataset/incrowdvi/images/val \
  --output-dir dataset/incrowdvi/semantic_masks/val
```

CLI flags:

| Flag | Required | Description |
|---|---|---|
| `--config` | yes | Path to segmentation YAML config (model settings) |
| `--images-dir` | yes | Flat directory of `*.png` frames to segment |
| `--output-dir` | yes | Destination directory for mask NPZ files |
| `--sequence` | no | Only process frames whose filename starts with this string (e.g. `Kiko_loop_R`); equivalent to globbing `{sequence}*.png` |
| `--force` | no | Overwrite existing mask NPZ files |
| `--dry-run` | no | Log config summary and exit without running inference |
| `--verbose` | no | Enable DEBUG-level logging |

---

## Output files

The runner writes one compressed NPZ per source frame:

```
{output_dir}/{stem}.npz
```

Each file contains a single array:

| Key | dtype | Shape | Description |
|---|---|---|---|
| `mask` | `uint8` | `(H, W)` | `1` = detected pedestrian pixel; `0` = static background |

The mask is in the same pixel space as the source image (no spatial transformation applied).

Loading a mask:

```python
import numpy as np

data = np.load("dataset/incrowdvi/semantic_masks/train/frame_stem.npz", allow_pickle=False)
mask = data["mask"]   # uint8 (H, W), values in {0, 1}
```

### Visual spot-check

Use the bundled example script to overlay a sample of masks on their source
frames and save the blended PNGs for manual inspection:

```bash
# Default: 20 frames spread across zero/sparse/medium/dense density buckets
python evaluation/segmentation_spotcheck.py \
  --masks-dir  dataset/incrowdvi/semantic_masks/train \
  --images-dir dataset/incrowdvi/frames/train \
  --output-dir logs/mask_spotcheck

# Custom sample size, random seed, and overlay opacity
python evaluation/segmentation_spotcheck.py \
  --masks-dir  dataset/incrowdvi/semantic_masks/train \
  --images-dir dataset/incrowdvi/frames/train \
  --output-dir logs/mask_spotcheck \
  --n-frames 40 \
  --seed 123 \
  --alpha 0.5
```

| Flag | Default | Description |
|---|---|---|
| `--n-frames` | `20` | Total frames to sample, spread evenly across the four buckets |
| `--seed` | `42` | Random seed for reproducible sampling |
| `--alpha` | `0.45` | Opacity of the red mask overlay (0 = invisible, 1 = solid) |
| `--all` | off | Process every mask in `--masks-dir` instead of sampling |

Frames are drawn from four density buckets (`zero`, `sparse`, `medium`,
`dense`) so both empty scenes and heavily masked frames are always represented.
Output filenames encode the bucket and density value, e.g.
`dense_d0.45_Kiko_loop_R_1058664352.png`.

---

## Mask utilities

`bb_utils.segmentation.utils` provides post-processing helpers. All functions
accept and return `uint8 (H, W)` masks with values in `{0, 1}`.

### `dilate_mask(mask, radius)`

Morphological dilation by a circular structuring element of `radius` pixels.

- Uses `scipy.ndimage.binary_dilation` when available; falls back to a
  pure-numpy iterative approach.
- `radius=0` is a no-op.

```python
from bb_utils.segmentation.utils import dilate_mask

dilated = dilate_mask(mask, radius=5)
```

### `union_masks(masks)`

Elementwise OR of a list of masks (same shape required).

```python
from bb_utils.segmentation.utils import union_masks

combined = union_masks([mask_a, mask_b, mask_c])
```

### `resize_mask(mask, target_shape)`

Resize a binary mask to `(H_new, W_new)` using nearest-neighbour interpolation
to preserve binary values.

- Uses `PIL.Image.NEAREST` when Pillow is available; falls back to a
  pure-numpy index-based approach.

```python
from bb_utils.segmentation.utils import resize_mask

resized = resize_mask(mask, (480, 640))
```

---

## Adding a new backend

1. Create a subclass of `SegmentationBackend` in a new file, e.g.
   `bb_utils/segmentation/sam_backend.py`:

   ```python
   from bb_utils.segmentation.base import SegmentationBackend

   class SamBackend(SegmentationBackend):
       def __init__(self, model_name, device, **kwargs):
           ...

       def segment(self, image, target_classes):
           # image: uint8 (H, W, 3) RGB
           # must return uint8 (H, W) with values in {0, 1}
           ...
   ```

2. Register it in `factory.py` (or at import time in your submodule):

   ```python
   from bb_utils.segmentation.factory import register_backend
   from bb_utils.segmentation.sam_backend import SamBackend

   register_backend("sam", SamBackend)
   ```

3. Use it by setting `model.backend: sam` in the pipeline YAML config.

No changes are required in `segmentation_runner.py` (`bb_utils.data_preparation`) or any other caller.

---

## Dependencies

| Dependency | Required by | Install |
|---|---|---|
| `numpy` | all modules | `pip install numpy` |
| `ultralytics` | `YoloBackend` | `pip install ultralytics` |
| `transformers` | `Mask2FormerBackend` | `pip install transformers` |
| `torch` | `Mask2FormerBackend`, `DeepLabBackend`, `Sam3Backend` | `pip install torch` |
| `torchvision` | `DeepLabBackend` | `pip install torchvision` |
| `sam3` | `Sam3Backend` | `git clone https://github.com/facebookresearch/sam3.git && cd sam3 && pip install -e .` |
| `scipy` | `dilate_mask` (optional) | `pip install scipy` |
| `Pillow` | `resize_mask`, `Mask2FormerBackend`, `Sam3Backend` (optional) | `pip install Pillow` |

`scipy` and `Pillow` are optional: both `dilate_mask` and `resize_mask` fall
back to pure-numpy implementations when they are not installed, at a small
performance cost for large masks or radii.
