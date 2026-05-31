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
    Edit the Configuration block near the top of the file to change the model,
    query labels, bounding-box color, line thickness, font size, and whether
    labels or coordinates are written on the boxes.
"""

import argparse
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from locateanything_worker import LocateAnythingWorker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID        = "nvidia/LocateAnything-3B"
QUERY_LABELS    = ["person"]

BOX_COLOR       = "green"    # outline and text-background color
TEXT_COLOR      = "white"    # text color
LINE_WIDTH      = 1          # bounding box outline thickness in pixels
WRITE_COORDS    = False      # write (x1,y1,x2,y2) on each box
WRITE_LABEL     = False      # write the object label on each box
FONT_SIZE       = 14         # font size for label and coordinate text
# ---------------------------------------------------------------------------

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
        boxes.append((x1, y1, x2, y2))
    return boxes


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


def process_image(worker: LocateAnythingWorker, input_path: Path, output_path: Path) -> None:
    """Run detection on one image and save the annotated result."""
    img = Image.open(input_path).convert("RGB")

    result = worker.detect(img, QUERY_LABELS)
    answer = result["answer"]
    print(f"[{input_path.name}] {answer}")

    boxes = parse_boxes(answer, img.width, img.height)
    print(f"[{input_path.name}] Detected {len(boxes)} boxes: {boxes}")

    annotated = draw_boxes(
        img, boxes,
        label=QUERY_LABELS[0],
        write_label=WRITE_LABEL,
        write_coords=WRITE_COORDS,
        box_color=BOX_COLOR,
        text_color=TEXT_COLOR,
        line_width=LINE_WIDTH,
        font_size=FONT_SIZE,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path)
    print(f"[{input_path.name}] Annotated image saved to {output_path}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    worker = LocateAnythingWorker(MODEL_ID)
    # worker = LocateAnythingWorker(MODEL_ID, device="cpu")

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
            process_image(worker, img_file, out_file)
    else:
        process_image(worker, input_path, output_path)


if __name__ == "__main__":
    main()

