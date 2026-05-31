# LocateAnything — Object Detection and Bounding-Box Annotation

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
4. Draws the boxes — and optionally the label and coordinates — on a **copy** of
   the image (the original is never modified).
5. Saves the annotated image to the specified output path.

---

## Prerequisites

LocateAnything must be installed in the Python environment used to run this
script. It is not a declared dependency of `bb_utils` because it lives in a
separate repository (`~/eagle`). Follow the setup instructions in
`~/eagle/Embodied/` to install it.

Minimum Python packages required (already satisfied by the LocateAnything
environment):

- `Pillow`

---

## Installation

After cloning / pulling `bb_utils`, reinstall the package so the new console
script entry point is registered:

```bash
pip install -e /home/ubuntu/bb_utils
```

This makes the command `bb-locate-and-annotate` available on `PATH`.

---

## Configuration

All tuneable parameters are collected in the **Configuration block** near the
top of `bb_locate_and_annotate.py`:

| Constant | Default | Description |
|---|---|---|
| `MODEL_ID` | `"nvidia/LocateAnything-3B"` | HuggingFace model identifier |
| `QUERY_LABELS` | `["person"]` | List of object classes to detect |
| `BOX_COLOR` | `"green"` | Bounding box outline and text-background colour |
| `TEXT_COLOR` | `"white"` | Colour of the overlay text |
| `LINE_WIDTH` | `1` | Bounding box outline thickness in pixels |
| `WRITE_LABEL` | `False` | Write the object label above each box |
| `WRITE_COORDS` | `False` | Write `(x1,y1,x2,y2)` above each box |
| `FONT_SIZE` | `14` | Font size for label and coordinate text |

---

## Usage

### Single image

```bash
python bb_locate_and_annotate.py <input_image> <output_image>

# Example
python bb_locate_and_annotate.py photo.png photo_annotated.png
```

Or via the registered console script (after `pip install -e`):

```bash
bb-locate-and-annotate photo.png photo_annotated.png
```

### Folder of images

```bash
python bb_locate_and_annotate.py <input_dir> <output_dir>

# Example
python bb_locate_and_annotate.py images/ images_annotated/
```

All files with extensions `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, or `.webp`
inside `<input_dir>` are processed. The output directory is created
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
| `draw_boxes(img, boxes, ...)` | Draws boxes on a copy of the image; respects all visual configuration options |
| `process_image(worker, input_path, output_path)` | Runs detection and annotation for a single image |
| `main()` | Entry point; dispatches to single-image or batch mode based on whether the input path is a file or a directory |
