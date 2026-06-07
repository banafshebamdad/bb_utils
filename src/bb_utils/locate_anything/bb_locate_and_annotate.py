#
# Banafshe Bamdad
# Sun May 31, 2026 15:41 CET
#
"""bb_locate_and_annotate.py: Object detection and bounding-box annotation using LocateAnything.

This script uses the NVIDIA LocateAnything model to detect objects in images and
overlays the resulting bounding boxes on annotated copies of those images.

It can operate in two modes:

    Single-image mode:
        Processes one image file and writes the annotated result to the specified
        output path.

    Batch mode:
        Processes all images inside an input directory and writes one annotated
        output image per input image to the specified output directory, using the
        naming convention ``<stem>_annotated<ext>``. The output directory is
        created automatically if it does not exist.

Usage:
    # Single image
    python bb_locate_and_annotate.py <input_image> <output_image>

    # Folder of images
    python bb_locate_and_annotate.py <input_dir> <output_dir>

Configuration:
    All tuneable parameters are stored in a YAML config file.
    Pass it with --config on every invocation::

        bb-locate-and-annotate <input> <output> --config configs/locate_anything.yaml

    The config has three top-level sections:

    ``model:``        model ID, locateanything_worker directory, query labels,
                      random seed, IoU deduplication threshold.
    ``annotation:``   bounding-box colour, line width, font size, label / coord
                      overlay switches.
    ``preprocessing:`` optional frame pre-rotation before inference (same schema
                      as ``bb-run-segmentation``; see Image pre-rotation in the
                      README).
"""

import argparse
import re
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect objects in images using LocateAnything and annotate bounding boxes."
    )
    parser.add_argument(
        "input",
        help="Path to a single image file or a folder of images.",
    )
    parser.add_argument(
        "output",
        help=(
            "Output path. For a single image: path to the annotated output file. "
            "For a folder: path to the output directory (created if it does not exist)."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help=(
            "Path to the YAML config file (e.g. configs/locate_anything.yaml). "
            "Contains model, annotation, and optional preprocessing sections."
        ),
    )
    return parser.parse_args()



def parse_boxes(
    answer: str,
    img_width: int,
    img_height: int,
    coord_scale: int = 1000,
) -> list[tuple[int, int, int, int]]:
    """Parse bounding box coordinates from LocateAnything answer string.

    LocateAnything outputs coordinates normalized to [0, coord_scale].
    They are rescaled here to actual pixel coordinates.

    Expected format: <box><x1><y1><x2><y2></box>
    Returns a list of (x1, y1, x2, y2) tuples in pixel coordinates.
    """
    boxes = []
    for match in re.findall(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
        nx1, ny1, nx2, ny2 = (int(v) for v in match)
        x1 = round(nx1 * img_width / coord_scale)
        y1 = round(ny1 * img_height / coord_scale)
        x2 = round(nx2 * img_width / coord_scale)
        y2 = round(ny2 * img_height / coord_scale)
        # Normalise in case the model outputs inverted coordinates
        boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
    return boxes


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union for two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def deduplicate_boxes(
    boxes: list[tuple[int, int, int, int]],
    iou_threshold: float = 0.5,
) -> list[tuple[int, int, int, int]]:
    """Remove near-duplicate boxes using greedy IoU-based suppression.

    Boxes are processed in the order returned by the model. The first box in
    each overlapping cluster is kept; subsequent boxes whose IoU with any
    already-accepted box exceeds ``iou_threshold`` are discarded.
    """
    kept: list[tuple[int, int, int, int]] = []
    for box in boxes:
        if all(_iou(box, k) < iou_threshold for k in kept):
            kept.append(box)
    return kept


def _transform_boxes_inverse(
    boxes: list[tuple[int, int, int, int]],
    effective_rotation: int,
    orig_w: int,
    orig_h: int,
) -> list[tuple[int, int, int, int]]:
    """Transform bounding boxes from the rotated image back to the original image space.

    This is the inverse of the rotation applied to the source image before
    inference.  The transformation is exact and lossless (integer arithmetic,
    no rounding error) because only multiples of 90° are supported.

    Derivation of the coordinate mappings (let W=orig_w, H=orig_h):

    - 90° CW:   original (x, y) → rotated (H-1-y, x)
                inverse:  rotated (rx, ry) → original (ry, H-1-rx)
    - 90° CCW:  original (x, y) → rotated (y, W-1-x)
                inverse:  rotated (rx, ry) → original (W-1-ry, rx)
    - 180°:    original (x, y) → rotated (W-1-x, H-1-y)  [self-inverse]
                inverse:  rotated (rx, ry) → original (W-1-rx, H-1-ry)

    Args:
        boxes:              List of (x1, y1, x2, y2) boxes in the rotated image
                            coordinate system.
        effective_rotation: The CW angle (degrees) that was applied to the
                            original image before inference.  Negative values
                            denote CCW rotation (e.g. -90 = 90° CCW).  Must be
                            a multiple of 90.
        orig_w:             Width of the original (pre-rotation) image in pixels.
        orig_h:             Height of the original (pre-rotation) image in pixels.

    Returns:
        List of (x1, y1, x2, y2) boxes in the original image coordinate space.
    """
    norm = effective_rotation % 360
    if norm == 0:
        return list(boxes)
    result = []
    for rx1, ry1, rx2, ry2 in boxes:
        if norm == 90:
            # CW 90°: rotated (rx, ry) → original (x=ry, y=H-1-rx)
            ox1, oy1 = ry1, orig_h - 1 - rx2
            ox2, oy2 = ry2, orig_h - 1 - rx1
        elif norm == 270:
            # CCW 90° (= CW 270°): rotated (rx, ry) → original (x=W-1-ry, y=rx)
            ox1, oy1 = orig_w - 1 - ry2, rx1
            ox2, oy2 = orig_w - 1 - ry1, rx2
        elif norm == 180:
            # 180° (self-inverse): rotated (rx, ry) → original (W-1-rx, H-1-ry)
            ox1, oy1 = orig_w - 1 - rx2, orig_h - 1 - ry2
            ox2, oy2 = orig_w - 1 - rx1, orig_h - 1 - ry1
        else:
            result.append((rx1, ry1, rx2, ry2))
            continue
        result.append((
            min(ox1, ox2), min(oy1, oy2),
            max(ox1, ox2), max(oy1, oy2),
        ))
    return result


def draw_boxes(
    img: Image.Image,
    boxes: list[tuple[int, int, int, int]],
    label: str = "",
    write_label: bool = True,
    write_coords: bool = True,
    box_color: str = "red",
    text_color: str = "white",
    line_width: int = 2,
    font_size: int = 14,
) -> Image.Image:
    """Draw bounding boxes on a copy of the image.

    Args:
        img: Source image (will not be modified).
        boxes: List of (x1, y1, x2, y2) bounding box coordinates.
        label: Optional label text.
        write_label: If True, write the label above each box.
        write_coords: If True, write the (x1,y1,x2,y2) coordinates on the box.
        box_color: Color of the bounding box outline.
        text_color: Color of the overlay text.
        line_width: Thickness of the bounding box outline.
        font_size: Font size for label and coordinate text.

    Returns:
        A new image with the bounding boxes drawn.
    """
    out = img.copy()
    draw = ImageDraw.Draw(out)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=font_size)
    except OSError:
        font = ImageFont.load_default()

    for x1, y1, x2, y2 in boxes:
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=line_width)

        text_parts = []
        if write_label and label:
            text_parts.append(label)
        if write_coords:
            text_parts.append(f"({x1},{y1},{x2},{y2})")
        text = " ".join(text_parts)

        if text:
            # Draw a small filled rectangle behind text for readability
            bbox = draw.textbbox((x1, y1 - 16), text, font=font)
            draw.rectangle(bbox, fill=box_color)
            draw.text((x1, y1 - 16), text, fill=text_color, font=font)

    return out


def process_image(
    worker,
    input_path: Path,
    output_path: Path,
    model_cfg: dict,
    annotation_cfg: dict,
    preprocessing_cfg: Optional[dict] = None,
    rotate_annotated_back: bool = True,
) -> None:
    """Run detection on one image and save the annotated result.

    Args:
        worker:               Loaded ``LocateAnythingWorker`` instance.
        input_path:           Source image path.
        output_path:          Destination path for the annotated image.
        model_cfg:            ``model:`` section of the YAML config.
        annotation_cfg:       ``annotation:`` section of the YAML config.
        preprocessing_cfg:    ``preprocessing:`` section of the YAML config.
                              When provided, a per-frame rotation is resolved
                              from the filename stem and applied before inference.
        rotate_annotated_back: When ``True`` (default) and a pre-rotation was
                              applied, bounding boxes are transformed back to
                              the original image coordinate space and the
                              annotated output is saved in the original
                              orientation.  When ``False``, the annotated
                              output is saved in the rotated orientation.
    """
    random_seed   = model_cfg.get("random_seed",   42)
    iou_threshold = model_cfg.get("iou_threshold", 0.5)
    query_labels  = model_cfg.get("query_labels",  ["person"])

    box_color    = annotation_cfg.get("box_color",    "green")
    text_color   = annotation_cfg.get("text_color",   "white")
    line_width   = annotation_cfg.get("line_width",   1)
    write_label  = annotation_cfg.get("write_label",  False)
    write_coords = annotation_cfg.get("write_coords", False)
    font_size    = annotation_cfg.get("font_size",    14)

    # Reset RNG state before each call so stochastic sampling (do_sample=True)
    # in the model gives the same result regardless of batch position.
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    stem = input_path.stem
    img_orig = Image.open(input_path).convert("RGB")
    orig_w, orig_h = img_orig.width, img_orig.height

    # Resolve and apply pre-rotation
    effective_rotation: Optional[int] = None
    if preprocessing_cfg:
        from bb_utils.data_preparation.segmentation_runner import (
            _resolve_rotation,
            _resolve_direction,
        )
        rotation = _resolve_rotation(stem, preprocessing_cfg)
        if rotation:
            direction = _resolve_direction(stem, preprocessing_cfg)
            effective_rotation = rotation if direction == "cw" else -rotation

    if effective_rotation:
        from bb_utils.segmentation.utils import rotate_image
        rotated_arr = rotate_image(np.array(img_orig, dtype=np.uint8), effective_rotation)
        img_for_detection = Image.fromarray(rotated_arr)
        print(
            f"[{input_path.name}] Pre-rotated by {abs(effective_rotation)}° "
            f"{'CW' if effective_rotation > 0 else 'CCW'}."
        )
    else:
        img_for_detection = img_orig

    result = worker.detect(img_for_detection, query_labels)
    answer = result["answer"]
    print(f"[{input_path.name}] {answer}")

    boxes = parse_boxes(answer, img_for_detection.width, img_for_detection.height)
    boxes = deduplicate_boxes(boxes, iou_threshold=iou_threshold)
    print(f"[{input_path.name}] Detected {len(boxes)} boxes: {boxes}")

    if effective_rotation and rotate_annotated_back:
        # Transform boxes back to original coordinate space and annotate the
        # original (non-rotated) image so the output is pixel-aligned.
        boxes = _transform_boxes_inverse(boxes, effective_rotation, orig_w, orig_h)
        img_to_annotate = img_orig
    else:
        img_to_annotate = img_for_detection

    annotated = draw_boxes(
        img_to_annotate, boxes,
        label=query_labels[0] if query_labels else "",
        write_label=write_label,
        write_coords=write_coords,
        box_color=box_color,
        text_color=text_color,
        line_width=line_width,
        font_size=font_size,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path)
    print(f"[{input_path.name}] Annotated image saved to {output_path}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    # Load config
    with open(args.config) as fh:
        full_cfg = yaml.safe_load(fh) or {}

    model_cfg      = full_cfg.get("model",      {})
    annotation_cfg = full_cfg.get("annotation", {})
    pre            = full_cfg.get("preprocessing") or {}
    preprocessing_cfg      = pre if pre else None
    rotate_annotated_back  = pre.get("rotate_annotated_back", True)

    print(f"[config] Loaded from {args.config}")
    print(f"[config]   model_id          : {model_cfg.get('model_id', 'nvidia/LocateAnything-3B')}")
    print(f"[config]   query_labels       : {model_cfg.get('query_labels', ['person'])}")
    print(f"[config]   iou_threshold      : {model_cfg.get('iou_threshold', 0.5)}")
    print(f"[config]   random_seed        : {model_cfg.get('random_seed', 42)}")
    if preprocessing_cfg:
        cameras_map = pre.get("cameras")
        if cameras_map:
            cam_dirs   = pre.get("camera_directions", {})
            global_dir = pre.get("rotation_direction", "cw").upper()
            for cam, deg in cameras_map.items():
                d = cam_dirs.get(cam, global_dir).upper()
                print(f"[config]   camera {cam}: {deg}° {d}")
        else:
            deg = pre.get("pre_rotation_deg")
            if deg:
                d = pre.get("rotation_direction", "cw").upper()
                print(f"[config]   global rotation    : {deg}° {d}")
        print(f"[config]   rotate_annotated_back: {rotate_annotated_back}")

    # Register locateanything_worker on sys.path
    locate_dir = str(
        Path(model_cfg.get("locateanything_dir", "~/eagle/Embodied"))
        .expanduser().resolve()
    )
    if locate_dir not in sys.path:
        sys.path.insert(0, locate_dir)
    from locateanything_worker import LocateAnythingWorker  # noqa: E402

    model_id = model_cfg.get("model_id", "nvidia/LocateAnything-3B")
    worker = LocateAnythingWorker(model_id)

    if input_path.is_dir():
        image_files = sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_files:
            print(f"No images found in {input_path}")
            return
        output_path.mkdir(parents=True, exist_ok=True)
        for img_file in image_files:
            out_file = output_path / (img_file.stem + "_annotated" + img_file.suffix)
            process_image(
                worker, img_file, out_file,
                model_cfg=model_cfg,
                annotation_cfg=annotation_cfg,
                preprocessing_cfg=preprocessing_cfg,
                rotate_annotated_back=rotate_annotated_back,
            )
    else:
        process_image(
            worker, input_path, output_path,
            model_cfg=model_cfg,
            annotation_cfg=annotation_cfg,
            preprocessing_cfg=preprocessing_cfg,
            rotate_annotated_back=rotate_annotated_back,
        )


if __name__ == "__main__":
    main()

