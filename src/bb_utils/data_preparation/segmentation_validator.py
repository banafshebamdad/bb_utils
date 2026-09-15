#
# Created on Tue Sep 15 2026 14:19:46
#
# Banafshe Bamdad
#
# segmentation_validator.py
#
"""Validate saved segmentation artifacts against their source PNG images."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path
from typing import Optional

import numpy as np


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _sequence_from_stem(stem: str) -> str:
    """Extract sequence from {sequence}_{L|R}_{timestamp}."""
    parts = stem.rsplit("_", 2)
    return parts[0] if len(parts) == 3 else stem


def _png_shape(path: Path) -> tuple[int, int]:
    """Return PNG shape as (height, width) without decoding the image."""
    with path.open("rb") as stream:
        header = stream.read(24)

    if (
        len(header) != 24
        or header[:8] != PNG_SIGNATURE
        or header[12:16] != b"IHDR"
    ):
        raise ValueError("not a valid PNG header")

    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid PNG dimensions: {(width, height)}")

    return height, width


def _validate_soft_confidence(path: Path, expected_shape: tuple[int, int]) -> None:
    with np.load(path, allow_pickle=False) as data:
        if data.files != ["pedestrian_confidence"]:
            raise ValueError(
                "expected exactly one key 'pedestrian_confidence'; "
                f"found {data.files}"
            )

        confidence = data["pedestrian_confidence"]

    if confidence.dtype != np.float32:
        raise ValueError(f"expected float32; found {confidence.dtype}")
    if confidence.ndim != 2:
        raise ValueError(f"expected a 2-D array; found shape {confidence.shape}")
    if confidence.shape != expected_shape:
        raise ValueError(
            f"shape {confidence.shape} does not match image {expected_shape}"
        )
    if not np.isfinite(confidence).all():
        raise ValueError("contains non-finite values")
    if np.any(confidence < 0.0) or np.any(confidence > 1.0):
        raise ValueError("contains values outside [0, 1]")


def _validate_binary_mask(path: Path, expected_shape: tuple[int, int]) -> None:
    with np.load(path, allow_pickle=False) as data:
        if data.files != ["mask"]:
            raise ValueError(f"expected exactly one key 'mask'; found {data.files}")

        mask = data["mask"]

    if mask.dtype != np.uint8:
        raise ValueError(f"expected uint8; found {mask.dtype}")
    if mask.ndim != 2:
        raise ValueError(f"expected a 2-D array; found shape {mask.shape}")
    if mask.shape != expected_shape:
        raise ValueError(f"shape {mask.shape} does not match image {expected_shape}")
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("contains values outside {0, 1}")


def validate_output_directory(
    images_dir: Path,
    output_dir: Path,
    output_mode: str,
    sequence: Optional[str] = None,
) -> dict[str, object]:
    """Validate completeness and contents of one segmentation output directory."""
    if not images_dir.is_dir():
        raise NotADirectoryError(f"images directory not found: {images_dir}")
    if not output_dir.is_dir():
        raise NotADirectoryError(f"output directory not found: {output_dir}")

    images = {
        path.stem: path
        for path in sorted(images_dir.glob("*.png"))
        if sequence is None or _sequence_from_stem(path.stem) == sequence
    }
    artifacts = {
        path.stem: path
        for path in sorted(output_dir.glob("*.npz"))
        if sequence is None or _sequence_from_stem(path.stem) == sequence
    }

    missing = sorted(set(images) - set(artifacts))
    extra = sorted(set(artifacts) - set(images))
    invalid: list[tuple[str, str]] = []

    validator = (
        _validate_soft_confidence
        if output_mode == "soft_confidence"
        else _validate_binary_mask
    )

    for stem in sorted(set(images) & set(artifacts)):
        try:
            expected_shape = _png_shape(images[stem])
            validator(artifacts[stem], expected_shape)
        except Exception as exc:
            invalid.append((stem, str(exc)))

    valid_count = len(images) - len(missing) - len(invalid)

    return {
        "expected": len(images),
        "artifacts": len(artifacts),
        "valid": valid_count,
        "missing": missing,
        "extra": extra,
        "invalid": invalid,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate segmentation NPZ artifacts against source PNG images."
    )
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--output-mode",
        choices=("soft_confidence", "binary_mask"),
        required=True,
    )
    parser.add_argument("--sequence")
    parser.add_argument("--max-errors", type=int, default=20)
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    try:
        result = validate_output_directory(
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            output_mode=args.output_mode,
            sequence=args.sequence,
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    print(f"expected images : {result['expected']}")
    print(f"artifacts found : {result['artifacts']}")
    print(f"valid artifacts : {result['valid']}")
    print(f"missing         : {len(result['missing'])}")
    print(f"extra           : {len(result['extra'])}")
    print(f"invalid         : {len(result['invalid'])}")

    shown = 0
    for category in ("missing", "extra"):
        for stem in result[category]:
            if shown >= args.max_errors:
                break
            print(f"{category.upper()}: {stem}")
            shown += 1

    for stem, error in result["invalid"]:
        if shown >= args.max_errors:
            break
        print(f"INVALID: {stem}: {error}")
        shown += 1

    has_errors = bool(result["missing"] or result["extra"] or result["invalid"])
    raise SystemExit(1 if has_errors else 0)


if __name__ == "__main__":
    main()