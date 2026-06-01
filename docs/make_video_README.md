# bb-make-video

<!-- Author: Banafshe Bamdad, Date: 2026-05-30 -->

`bb-make-video` generates a video file from a flat directory of images using
OpenCV's `VideoWriter`.  All encoding and frame-selection parameters are
controlled through a YAML config file; the image directory and output path are
supplied as CLI flags.

---

## Table of Contents

- [Quick start](#quick-start)
- [CLI reference](#cli-reference)
- [Config reference](#config-reference)
  - [fps](#fps)
  - [codec](#codec)
  - [image\_glob](#image_glob)
  - [sort\_by](#sort_by)
  - [resize](#resize)
  - [frame\_range](#frame_range)
  - [overlay\_filename](#overlay_filename)
- [Image ordering](#image-ordering)
- [Output resolution](#output-resolution)
- [Supported image formats](#supported-image-formats)
- [Dependencies](#dependencies)

---

## Quick start

```bash
# Install the package (editable)
pip install -e bb_utils

bb-make-video --config configs/make_video.yaml \
              --images-dir /path/to/images \
              --output /path/to/output.mp4

# Validate config and discovered frames without writing anything
bb-make-video --config configs/make_video.yaml \
              --images-dir /path/to/images \
              --output /path/to/output.mp4 \
              --dry-run

# Burn the image filename onto every frame
bb-make-video --config configs/make_video.yaml \
              --images-dir /path/to/images \
              --output /path/to/output.mp4 \
              --overlay-filename

```

---

## CLI reference

| Flag | Required | Description |
|---|---|---|
| `--config PATH` | yes | Path to the YAML config file |
| `--images-dir PATH` | yes | Directory containing the source image files |
| `--output PATH` | yes | Destination video file (extension determines container, e.g. `.mp4`, `.avi`) |
| `--force` | no | Overwrite the output file if it already exists |
| `--dry-run` | no | Validate config and discover frames without writing the video |
| `--verbose` | no | Enable debug-level logging |
| `--overlay-filename` | no | Burn the image filename onto each frame (overrides `overlay_filename` in config) |

---

## Config reference

All settings live under the top-level `video:` key.  A fully annotated
example is provided in [`configs/make_video.yaml`](configs/make_video.yaml).

```yaml
video:
  fps: 30
  codec: "mp4v"
  image_glob: "*.png"
  sort_by: "name"
  resize:
    width: null
    height: null
  frame_range:
    start: null
    end: null
  overlay_filename: false
```


### codec

[FourCC](https://www.fourcc.org/codecs/) codec string passed to
`cv2.VideoWriter_fourcc`.

| Type | Default |
|---|---|
| str | `"mp4v"` |

| Value | Container | Notes |
|---|---|---|
| `"mp4v"` | `.mp4` | MPEG-4 Part 2; widely compatible |
| `"avc1"` | `.mp4` | H.264; smaller files; requires H.264 in your OpenCV build |
| `"XVID"` | `.avi` | Xvid MPEG-4 |
| `"MJPG"` | `.avi` | Motion JPEG; large files, near-lossless quality |

```yaml
video:
  codec: "XVID"
```

> **Note:** codec availability depends on your OpenCV build.  If encoding
> fails, try `"XVID"` with an `.avi` output path or rebuild OpenCV with
> the desired codec support.

### image_glob

Glob pattern used to discover image files inside `--images-dir`.

| Type | Default |
|---|---|
| str | `"*.png"` |

```yaml
video:
  image_glob: "frame_*.jpg"
```

### sort_by

Determines the frame order.

| Type | Default |
|---|---|
| str | `"name"` |

| Value | Behaviour |
|---|---|
| `"name"` | Lexicographic sort on the full filename (matches numeric frame naming) |
| `"mtime"` | Sort by file modification time (ascending) |

```yaml
video:
  sort_by: "mtime"
```

> **Tip:** For lexicographic sort to match numeric order, zero-pad frame
> indices (e.g. `frame_00001.png`, not `frame_1.png`).

### resize

Scale all frames to a fixed resolution before encoding.

| Sub-key | Type | Default | Description |
|---|---|---|---|
| `width` | int or null | `null` | Target width in pixels |
| `height` | int or null | `null` | Target height in pixels |

- Both `null` → keep the native resolution of the first frame.
- One dimension given → the other is derived from the original aspect ratio.
- Both given → frames are resized to exactly `width × height` (aspect ratio
  is **not** preserved).

```yaml
video:
  resize:
    width: 1280
    height: null   # height derived automatically from aspect ratio
```

### frame_range

Select a contiguous subset of the sorted frame list (0-based, both bounds
inclusive).

| Sub-key | Type | Default | Description |
|---|---|---|---|
| `start` | int or null | `null` | First frame index to include (`null` = 0) |
| `end` | int or null | `null` | Last frame index to include (`null` = last frame) |

```yaml
video:
  frame_range:
    start: 100
    end: 499   # encodes frames 100–499 (400 frames total)
```

### overlay_filename

When `true`, the filename of each source image is rendered onto the
bottom-left corner of the corresponding video frame.  Text is drawn in
white with a black outline so it remains readable on any background.

| Type | Default |
|---|---|
| bool | `false` |

```yaml
video:
  overlay_filename: true
```

The same behaviour can be enabled at the command line with `--overlay-filename`,
which overrides the config value.

---

## Image ordering

Frames are sorted **before** the `frame_range` slice is applied.  The
pipeline is:

```
discover all files matching image_glob
        │
        ▼
sort (by name or mtime)
        │
        ▼
slice [frame_range.start : frame_range.end + 1]
        │
        ▼
encode frames in order
```

---

## Output resolution

The output resolution is determined from the **first frame** in the
(possibly sliced) frame list:

1. If `resize.width` and `resize.height` are both `null` → use the
   native image size.
2. If only `width` is given → derive `height = round(native_height * width / native_width)`.
3. If only `height` is given → derive `width = round(native_width * height / native_height)`.
4. If both are given → use `(width, height)` exactly.

Frames whose native size differs from the target resolution are resized
with `cv2.INTER_AREA` (good quality for downscaling).

---

## Supported image formats

Any format readable by `cv2.imread` is supported, including:

- PNG (`.png`)
- JPEG (`.jpg`, `.jpeg`)
- BMP (`.bmp`)
- TIFF (`.tiff`, `.tif`)

Adjust `image_glob` accordingly (e.g. `"*.jpg"`).

Grayscale images are handled automatically by OpenCV and written as
colour frames.

---

## Dependencies

| Package | Purpose | How to install |
|---|---|---|
| `opencv-python` | Image loading, resizing, video encoding | `pip install opencv-python` |
| `PyYAML` | Config parsing | included in `bb_utils` dependencies |
| `tqdm` | Progress bar (optional) | `pip install tqdm` |

`tqdm` is optional; if it is not installed the tool runs without a progress
bar.
