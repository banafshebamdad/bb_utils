#
# Author: Banafshe Bamdad
# Date: 2026-05-30
#

"""
Generate a video from a flat directory of images.

The tool discovers image files via a configurable glob pattern, sorts them,
optionally resizes each frame, and encodes the result with OpenCV's
``VideoWriter``.

CLI
---
    bb-make-video --config make_video.yaml \\
                  --images-dir /path/to/images \\
                  --output /path/to/output.mp4

    # Overwrite an existing output file
    bb-make-video --config make_video.yaml \\
                  --images-dir /path/to/images \\
                  --output /path/to/output.mp4 --force

    # Validate config and discovered frames without writing
    bb-make-video --config make_video.yaml \\
                  --images-dir /path/to/images \\
                  --output /path/to/output.mp4 --dry-run

Config schema
-------------
    video:
      fps: 30
      codec: "mp4v"        # FourCC: "mp4v" (.mp4), "XVID" (.avi), "avc1" (H.264)
      image_glob: "*.png"  # glob pattern for image discovery
      sort_by: "name"      # "name" (lexicographic) or "mtime" (modification time)
      resize:
        width: null        # null = keep original resolution
        height: null       # null = keep original resolution
      frame_range:
        start: null        # null = first discovered frame (0-based inclusive)
        end: null          # null = last discovered frame (0-based inclusive)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _discover_frames(
    images_dir: Path,
    image_glob: str,
    sort_by: str,
    frame_start: Optional[int],
    frame_end: Optional[int],
) -> List[Path]:
    """Return a sorted, optionally sliced list of image paths.

    Args:
        images_dir:  Directory to search for images.
        image_glob:  Glob pattern relative to *images_dir* (e.g. ``*.png``).
        sort_by:     ``"name"`` for lexicographic sort; ``"mtime"`` for
                     modification-time sort.
        frame_start: 0-based inclusive start index; ``None`` = 0.
        frame_end:   0-based inclusive end index; ``None`` = last frame.

    Returns:
        Ordered list of ``Path`` objects.

    Raises:
        ValueError: If *sort_by* is not ``"name"`` or ``"mtime"``.
        FileNotFoundError: If no files match the glob.
    """
    if sort_by not in ("name", "mtime"):
        raise ValueError(f"sort_by must be 'name' or 'mtime', got '{sort_by}'")

    files = list(images_dir.glob(image_glob))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{image_glob}' found in {images_dir}"
        )

    if sort_by == "mtime":
        files.sort(key=lambda p: p.stat().st_mtime)
    else:
        files.sort()

    start = frame_start or 0
    end = (frame_end + 1) if frame_end is not None else len(files)
    sliced = files[start:end]

    if not sliced:
        raise ValueError(
            f"frame_range [{frame_start}, {frame_end}] produced an empty list "
            f"(total frames discovered: {len(files)})"
        )

    return sliced


def _target_size(
    first_frame_path: Path,
    resize_width: Optional[int],
    resize_height: Optional[int],
) -> Tuple[int, int]:
    """Resolve the output (width, height) for the video.

    If both *resize_width* and *resize_height* are ``None`` the native size of
    *first_frame_path* is returned.  If only one dimension is given the other
    is derived from the original aspect ratio.

    Args:
        first_frame_path:  Path to the first image frame.
        resize_width:      Desired output width in pixels, or ``None``.
        resize_height:     Desired output height in pixels, or ``None``.

    Returns:
        ``(width, height)`` tuple.
    """
    import cv2  # local import so the module is importable without cv2

    img = cv2.imread(str(first_frame_path))
    if img is None:
        raise RuntimeError(f"Could not read image: {first_frame_path}")
    native_h, native_w = img.shape[:2]

    if resize_width is None and resize_height is None:
        return native_w, native_h
    if resize_width is not None and resize_height is not None:
        return resize_width, resize_height
    if resize_width is not None:
        scale = resize_width / native_w
        return resize_width, max(1, round(native_h * scale))
    # resize_height is not None
    scale = resize_height / native_h
    return max(1, round(native_w * scale)), resize_height


def make_video(
    images_dir: Path,
    output_path: Path,
    fps: float = 30.0,
    codec: str = "mp4v",
    image_glob: str = "*.png",
    sort_by: str = "name",
    resize_width: Optional[int] = None,
    resize_height: Optional[int] = None,
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
    force: bool = False,
) -> int:
    """Write a video file from images in *images_dir*.

    Args:
        images_dir:     Flat directory of image files.
        output_path:    Destination video file path.  The file extension
                        determines the container (e.g. ``.mp4``, ``.avi``).
        fps:            Frames per second of the output video.
        codec:          FourCC codec string (``"mp4v"``, ``"XVID"``, ``"avc1"``).
        image_glob:     Glob pattern used to discover images (e.g. ``"*.png"``).
        sort_by:        Sort order for discovered files: ``"name"`` or ``"mtime"``.
        resize_width:   Target width in pixels, or ``None`` to keep native.
        resize_height:  Target height in pixels, or ``None`` to keep native.
        frame_start:    0-based inclusive start index; ``None`` = first frame.
        frame_end:      0-based inclusive end index; ``None`` = last frame.
        force:          Overwrite *output_path* if it already exists.

    Returns:
        Number of frames written.

    Raises:
        FileExistsError:    If *output_path* exists and *force* is ``False``.
        FileNotFoundError:  If no images match the glob.
        RuntimeError:       If the ``VideoWriter`` cannot be opened.
    """
    import cv2

    if output_path.exists() and not force:
        raise FileExistsError(
            f"Output file already exists: {output_path}  (pass --force to overwrite)"
        )

    frames = _discover_frames(
        images_dir=images_dir,
        image_glob=image_glob,
        sort_by=sort_by,
        frame_start=frame_start,
        frame_end=frame_end,
    )
    logger.info("Discovered %d frames in %s", len(frames), images_dir)

    width, height = _target_size(frames[0], resize_width, resize_height)
    logger.info("Output resolution: %d x %d  |  fps: %s  |  codec: %s", width, height, fps, codec)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(
            f"cv2.VideoWriter failed to open '{output_path}' "
            f"(codec={codec}, fps={fps}, size={width}x{height})"
        )

    try:
        file_iter = _progress_iter(frames)
        for frame_path in file_iter:
            img = cv2.imread(str(frame_path))
            if img is None:
                logger.warning("Skipping unreadable frame: %s", frame_path)
                continue
            if (img.shape[1], img.shape[0]) != (width, height):
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(img)
    finally:
        writer.release()

    n_frames = len(frames)
    logger.info("Video written to %s  (%d frames)", output_path, n_frames)
    return n_frames


def _progress_iter(items):
    """Wrap *items* with tqdm if available, otherwise return as-is."""
    try:
        from tqdm import tqdm
        return tqdm(items, unit="frame")
    except ImportError:
        return items


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: Path) -> dict:
    with open(config_path) as fh:
        return yaml.safe_load(fh) or {}


def _parse_video_cfg(config: dict) -> dict:
    """Extract and validate the ``video`` section of the config."""
    video = config.get("video", {})

    resize = video.get("resize", {}) or {}
    frame_range = video.get("frame_range", {}) or {}

    return {
        "fps": float(video.get("fps", 30)),
        "codec": str(video.get("codec", "mp4v")),
        "image_glob": str(video.get("image_glob", "*.png")),
        "sort_by": str(video.get("sort_by", "name")),
        "resize_width": resize.get("width"),
        "resize_height": resize.get("height"),
        "frame_start": frame_range.get("start"),
        "frame_end": frame_range.get("end"),
    }


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point: bb-make-video."""
    parser = argparse.ArgumentParser(
        description="Generate a video from a directory of images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to the YAML config file (video settings).",
    )
    parser.add_argument(
        "--images-dir", type=Path, required=True,
        help="Directory containing the source image files.",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Destination video file path (e.g. /out/clip.mp4).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug-level logging.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate config and discover frames without writing the video.",
    )

    args = parser.parse_args()
    _setup_logging(args.verbose)

    # Validate inputs
    if not args.config.exists():
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)

    if not args.images_dir.is_dir():
        logger.error("images-dir is not a directory: %s", args.images_dir)
        sys.exit(1)

    config = _load_config(args.config)
    cfg = _parse_video_cfg(config)

    if args.dry_run:
        logger.info("DRY RUN — configuration summary")
        logger.info("  images_dir   : %s", args.images_dir)
        logger.info("  output       : %s", args.output)
        logger.info("  fps          : %s", cfg["fps"])
        logger.info("  codec        : %s", cfg["codec"])
        logger.info("  image_glob   : %s", cfg["image_glob"])
        logger.info("  sort_by      : %s", cfg["sort_by"])
        logger.info("  resize       : %s x %s", cfg["resize_width"], cfg["resize_height"])
        logger.info("  frame_range  : [%s, %s]", cfg["frame_start"], cfg["frame_end"])
        try:
            frames = _discover_frames(
                images_dir=args.images_dir,
                image_glob=cfg["image_glob"],
                sort_by=cfg["sort_by"],
                frame_start=cfg["frame_start"],
                frame_end=cfg["frame_end"],
            )
            logger.info("  frames found : %d", len(frames))
            logger.info("  first frame  : %s", frames[0].name)
            logger.info("  last frame   : %s", frames[-1].name)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Frame discovery failed: %s", exc)
            sys.exit(1)
        sys.exit(0)

    try:
        n = make_video(
            images_dir=args.images_dir,
            output_path=args.output,
            fps=cfg["fps"],
            codec=cfg["codec"],
            image_glob=cfg["image_glob"],
            sort_by=cfg["sort_by"],
            resize_width=cfg["resize_width"],
            resize_height=cfg["resize_height"],
            frame_start=cfg["frame_start"],
            frame_end=cfg["frame_end"],
            force=args.force,
        )
        logger.info("Done. %d frames encoded.", n)
    except FileExistsError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("Video generation failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
