# Segmentation Backend

`bb_utils.segmentation` provides a common interface for running instance and
semantic segmentation models on RGB images, normalising each model's output
into a single **binary mask contract** that callers (e.g. `sp-run-segmentation`
in sp-score) can consume without knowing which model produced it.

---

## Table of Contents

- [Output contract](#output-contract)
- [Package layout](#package-layout)
- [Architecture](#architecture)
- [Available backends](#available-backends)
  - [YoloBackend](#yolobackend)
- [Config format](#config-format)
- [Mask utilities](#mask-utilities)
- [Adding a new backend](#adding-a-new-backend)
- [Dependencies](#dependencies)

---

## Output contract

Every backend satisfies the same contract defined in `base.SegmentationBackend`:

```
mask = backend.segment(image, target_classes)
```

| Property | Value |
|---|---|
| Input `image` | `np.ndarray`, shape `(H, W, 3)`, dtype `uint8`, colour order **RGB** |
| Input `target_classes` | `List[int]` — model-specific class indices to include |
| Output `mask` | `np.ndarray`, shape `(H, W)`, dtype `uint8`, values strictly in `{0, 1}` |
| Output semantics | `1` = pixel belongs to a detected instance of a target class; `0` = background |
| No detections | Returns an all-zeros mask (valid result, not an error) |
| Spatial alignment | `mask[i, j]` corresponds to `image[i, j]` — no spatial transformation applied |

The backend does **not** apply dilation and does **not** convert grayscale input
to RGB; both are the caller's responsibility.

---

## Package layout

```
bb_utils/segmentation/
    __init__.py       — public API: SegmentationBackend, create_backend, mask utils
    base.py           — abstract SegmentationBackend (ABC) + contract documentation
    factory.py        — create_backend(config), register_backend(name, cls)
    yolo_backend.py   — YoloBackend: Ultralytics YOLOv8-seg
    utils.py          — dilate_mask, union_masks, resize_mask
```

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

1. `model.predict(image_rgb, conf=..., iou=..., device=...)` — runs NMS internally.
2. For each detected instance whose class is in `target_classes`, extract the
   float instance mask at model resolution.
3. Threshold at `0.5` → binary; resize to input `(H, W)` with nearest-neighbour
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

---

## Config format

`create_backend(config)` reads the `model` section of the pipeline config dict.
A minimal YAML excerpt:

```yaml
model:
  backend: yolo
  model_name: yolov8n-seg
  device: cuda
  confidence_threshold: 0.25
  iou_threshold: 0.45
```

The full segmentation pipeline config (used by `sp-run-segmentation`) is at
[sp-score/configs/preprocessing_segmentation.yaml](../../sp-score/configs/preprocessing_segmentation.yaml).

---

## Mask utilities

`bb_utils.segmentation.utils` provides post-processing helpers. All functions
accept and return `uint8 (H, W)` masks with values in `{0, 1}`.

### `dilate_mask(mask, radius)`

Morphological dilation by a circular structuring element of `radius` pixels.
Used to compensate for boundary inaccuracy in fisheye segmentation.

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

No changes are required in `segmentation_runner.py` or any other caller.

---

## Dependencies

| Dependency | Required by | Install |
|---|---|---|
| `numpy` | all modules | `pip install numpy` |
| `ultralytics` | `YoloBackend` | `pip install ultralytics` |
| `scipy` | `dilate_mask` (optional) | `pip install scipy` |
| `Pillow` | `resize_mask` (optional) | `pip install Pillow` |

`scipy` and `Pillow` are optional: both `dilate_mask` and `resize_mask` fall
back to pure-numpy implementations when they are not installed, at a small
performance cost for large masks or radii.
