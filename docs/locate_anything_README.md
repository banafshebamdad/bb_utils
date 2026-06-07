# LocateAnything: Object Detection and Bounding-Box Annotation

`bb_utils.locate_anything` provides a script that runs the
[NVIDIA LocateAnything](https://huggingface.co/nvidia/LocateAnything-3B) model
on one image or an entire folder of images and saves annotated copies with
bounding boxes drawn on them.

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Image pre-rotation](#image-pre-rotation)
- [Usage](#usage)
  - [Single image](#single-image)
  - [Folder of images](#folder-of-images)
- [Output](#output)
- [Script internals](#script-internals)

---

## Overview

`bb_locate_and_annotate.py` loads the LocateAnything model once and then:

1. Runs object detection on each input image.
2. Parses the raw model answer to extract bounding box coordinates.
3. Rescales the normalised coordinates (0–1000 grid) to actual pixel coordinates.
4. Draws the boxes, and optionally the label and coordinates, on a **copy** of
   the image (the original is never modified).
5. Saves the annotated image to the specified output path.

---

## Prerequisites

LocateAnything must be installed in the Python environment used to run this
script. It is not a declared dependency of `bb_utils` because it lives in a
separate repository (`~/eagle`).

### Setting up the LocateAnything environment

```bash
conda update -n base -c defaults conda
conda create -n locateanything python=3.10 pip -y
conda activate locateanything

git clone https://github.com/NVlabs/Eagle.git eagle
cd eagle/Embodied
pip install -e .
```

### Troubleshooting: PyTorch / CUDA mismatch

If you encounter CUDA-related errors at runtime, reinstall PyTorch with the
correct CUDA build (adjust the `cu126` suffix to match your driver):

```bash
conda activate locateanything
python -m pip uninstall torch torchvision torchaudio -y
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### Minimum additional requirements

The following package is required by `bb_locate_and_annotate.py` and is already
satisfied by the LocateAnything environment:

- `Pillow`

---

## Installation

`bb_locate_and_annotate.py` imports `locateanything_worker`, which is only
available in the **`locateanything`** conda environment. Both this script and
`bb_utils` must therefore be used from that environment.

After setting up the `locateanything` environment (see [Prerequisites](#prerequisites)),
install `bb_utils` into it so the console script entry point is registered:

```bash
conda activate locateanything
pip install -e ~/bb_utils
```

This makes the command `bb-locate-and-annotate` available on `PATH` within the
`locateanything` environment.

> **Note:** Do **not** run this script from any other
> environment, `locateanything_worker` will not be found there.

---

## Configuration

All tuneable parameters are stored in **`configs/locate_anything.yaml`** and
loaded at runtime via the required `--config` flag.  The file has three
top-level sections:

### `model:`

| Key | Default | Description |
|---|---|---|
| `model_id` | `"nvidia/LocateAnything-3B"` | HuggingFace model identifier |
| `locateanything_dir` | `"~/eagle/Embodied"` | Directory containing `locateanything_worker.py`; added to `sys.path` at startup |
| `query_labels` | `["person"]` | List of object classes to detect |
| `random_seed` | `42` | RNG seed reset before every image (see note below) |
| `iou_threshold` | `0.5` | IoU above which a box is suppressed as a near-duplicate (see note below); set to `1.0` to disable |

### `annotation:`

| Key | Default | Description |
|---|---|---|
| `box_color` | `"green"` | Bounding box outline and text-background colour |
| `text_color` | `"white"` | Colour of the overlay text |
| `line_width` | `1` | Bounding box outline thickness in pixels |
| `write_label` | `false` | Write the object label above each box |
| `write_coords` | `false` | Write `(x1,y1,x2,y2)` above each box |
| `font_size` | `14` | Font size for label and coordinate text |

### `preprocessing:` (optional)

See [Image pre-rotation](#image-pre-rotation) below for the full key table and
configuration examples.

> **Batch consistency and near-duplicate suppression note:**
> The LocateAnything model uses stochastic sampling (`do_sample=True`,
> `temperature=0.7`), which causes two related issues:
>
> 1. **Runaway generation**: without a fixed seed, each image's generation
>    shifts the GPU random state. Occasionally an image gets an unlucky state
>    where the model never produces the end-of-sequence token, generating
>    `<box>` entries until the 2048-token limit (hundreds of spurious boxes).
>    `random_seed` resets the RNG to the same state before every call so batch
>    mode and single-image mode behave identically.
>
> 2. **Near-duplicate boxes**: even with a fixed seed the model can enter a
>    semi-repetitive state where it emits many slightly-shifted boxes for the
>    same location before eventually stopping (e.g. 31 boxes instead of ~8 for
>    a crowd frame). `iou_threshold` enables greedy IoU-based suppression: the
>    first box in each overlapping cluster is kept and any subsequent box with
>    IoU ≥ threshold against an accepted box is discarded.
>
> The image below illustrates the near-duplicate issue
> annotated **without** the fixes, showing spurious overlapping boxes produced
> by the model's repetitive generation:
>
> ![Near-duplicate box example](Orell_strait_L_671630631_annotated.png)

---

## Image pre-rotation

Frames can optionally be rotated before being passed to the LocateAnything
model.  This is useful for cameras whose physical orientation differs from the
expected portrait/landscape input.

Rotation is controlled by an optional `--config` YAML file.  The
`preprocessing:` section of that file uses the **same schema** as
`bb-run-segmentation` — the two pipelines can share a single config file.

**Only multiples of 90° are supported.**  Rotation direction defaults to
clockwise (`cw`) and is configurable per-camera.

### Config modes

**Option A — same rotation for all images:**

```yaml
preprocessing:
  pre_rotation_deg: 90          # rotation magnitude; must be a multiple of 90
  rotation_direction: cw        # "cw" (clockwise, default) or "ccw"
  rotate_annotated_back: true   # transform boxes back and save in original orientation (default: true)
```

**Option B — per-camera rotation** (camera inferred from filename `{sequence}_{L|R}_{timestamp}.png`):

```yaml
preprocessing:
  rotation_direction: cw        # global default
  camera_directions:             # optional per-camera direction override
    L: cw
    R: ccw
  rotate_annotated_back: true
  cameras:
    L: 90
    R: 90
```

**Option C — per-camera with global fallback:**

```yaml
preprocessing:
  pre_rotation_deg: 90          # used when camera cannot be detected from filename
  rotation_direction: cw
  camera_directions:
    L: cw
    R: ccw
  rotate_annotated_back: true
  cameras:
    L: 90
    R: 90
```

### `preprocessing:` config keys

| Key | Type | Default | Description |
|---|---|---|---|
| `pre_rotation_deg` | int | `null` | Rotation magnitude in degrees (must be a multiple of 90) |
| `rotation_direction` | str | `"cw"` | Global direction: `"cw"` (clockwise) or `"ccw"` (counter-clockwise) |
| `camera_directions` | dict | `{}` | Per-camera direction overrides; keys `"L"` / `"R"`; values `"cw"` / `"ccw"` |
| `cameras` | dict | `{}` | Per-camera rotation magnitude overrides; keys `"L"` / `"R"`; values are degrees |
| `rotate_annotated_back` | bool | `true` | When `true`, bounding boxes are inverse-transformed to the original coordinate space and the annotated output is saved in the original image orientation.  When `false`, the annotated output is saved in the rotated orientation |

**Pixel-alignment guarantee**: when `rotate_annotated_back: true` (the default),
the saved annotated image has the same shape `(H, W)` as the source image and
each bounding box is correctly positioned in the original pixel space.
When `rotate_annotated_back: false`, the annotated image is in the rotated
orientation — useful when downstream code consumes the rotated frames directly.

---

## Usage

Always activate the `locateanything` environment before running the script:

```bash
conda activate locateanything
```

The script accepts absolute or relative paths for both input and output and can
be invoked from any working directory; `locateanything_worker` is resolved via
`model.locateanything_dir` in the config file.

### Single image

```bash
# Using the console script (recommended after pip install -e)
bb-locate-and-annotate /path/to/photo.png /path/to/photo_annotated.png \
    --config configs/locate_anything.yaml

# Or directly with python (from any directory)
python ~/bb_utils/src/bb_utils/locate_anything/bb_locate_and_annotate.py \
    /path/to/photo.png /path/to/photo_annotated.png \
    --config ~/bb_utils/configs/locate_anything.yaml

# Or with python after cd into the script directory
cd ~/bb_utils/src/bb_utils/locate_anything
python bb_locate_and_annotate.py photo.png photo_annotated.png \
    --config ~/bb_utils/configs/locate_anything.yaml
```

### Folder of images

```bash
# Using the console script (recommended after pip install -e)
bb-locate-and-annotate /path/to/images/ /path/to/images_annotated/ \
    --config configs/locate_anything.yaml

# Or directly with python (from any directory)
python ~/bb_utils/src/bb_utils/locate_anything/bb_locate_and_annotate.py \
    /path/to/images/ /path/to/images_annotated/ \
    --config ~/bb_utils/configs/locate_anything.yaml
```

All files with extensions `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, or `.webp`
inside the input directory are processed. The output directory is created
automatically if it does not exist.

---

## Output

| Mode | Output naming |
|---|---|
| Single image | Exactly the path given as the second argument |
| Folder | `<output_dir>/<stem>_annotated<ext>` for each input file |

The original images are never modified.

---

## Script internals

| Function | Description |
|---|---|
| `parse_boxes(answer, img_width, img_height)` | Extracts bounding boxes from the raw model answer string and rescales them from the 0–1000 normalised grid to pixel coordinates |
| `_transform_boxes_inverse(boxes, effective_rotation, orig_w, orig_h)` | Inverse-transforms bounding boxes from rotated-image space back to original-image space (exact integer arithmetic, all multiples of 90°) |
| `draw_boxes(img, boxes, ...)` | Draws boxes on a copy of the image; respects all visual configuration options |
| `process_image(worker, input_path, output_path, model_cfg, annotation_cfg, preprocessing_cfg, rotate_annotated_back)` | Runs detection and annotation for a single image; reads all parameters from the config dicts; applies pre-rotation and optional inverse transform when a preprocessing config is provided |
| `main()` | Entry point; loads `--config` YAML, logs effective settings, dispatches to single-image or batch mode |
