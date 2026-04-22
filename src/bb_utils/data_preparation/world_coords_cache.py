#
# File: data_preparation/world_coords_cache.py
# Author: Banafshe Bamdad + Claude Code
# Created: 2026-04-22 CET
#

"""
Generic world-coordinate cache builder for Aria MPS semidense point clouds.

Reads ``*_semidense_points.csv.gz`` files produced by Meta's Aria MPS
pipeline and writes compact NPZ caches that map each global UID to its
3-D world coordinates.

This module contains **no project-specific logic**.  It operates entirely on
explicit ``Path`` arguments so it can be reused across any Aria MPS dataset,
not only InCrowd-VI.  Project-specific CLI wrappers (e.g. ``sp-cache-world-
coords`` in sp-score) are responsible for extracting paths from their own
config schemas and delegating to :func:`build_world_coords_cache` or
:func:`run_split`.

Output NPZ contract
-------------------
Each output file ``{output_dir}/{split}/{sequence}_world_coords.npz``
contains two arrays:

  ``uid`` : int64, shape (K,)
      Global world-point identifiers matching those in
      ``semidense_observations.csv.gz``.

  ``xyz`` : float64, shape (K, 3)
      ``(px_world, py_world, pz_world)`` in the Aria MPS odometry frame
      (metres).

This contract is consumed by callers that need to look up 3-D positions
by UID (e.g. the static reliability temporal score computation in sp-score).

Public API
----------
build_world_coords_cache
    Build a single cache file from one semidense_points CSV.
discover_sequences
    List sequence names available in a split directory.
run_split
    Build caches for all sequences in one split directory.
"""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Required columns in the semidense_points CSV (Aria MPS invariants)
_REQUIRED_COLS = {"uid", "px_world", "py_world", "pz_world"}
_CHUNK_SIZE = 50_000


# ---------------------------------------------------------------------------
# Core cache builder
# ---------------------------------------------------------------------------

def build_world_coords_cache(
    points_csv_path: Path,
    out_path: Path,
    force: bool = False,
) -> int:
    """Read a semidense_points CSV and write a uid→xyz NPZ cache.

    This function has no knowledge of any project's config schema or directory
    conventions; all paths are passed explicitly.

    Args:
        points_csv_path: Path to ``{sequence}_semidense_points.csv.gz``.
        out_path:        Destination NPZ path; parent directory is created if
                         absent.
        force:           When False (default), skip writing if *out_path*
                         already exists and return the sentinel value ``-1``.
                         When True, overwrite unconditionally.

    Returns:
        Number of world points written, or ``-1`` if the file was skipped
        (already exists and *force* is False).

    Raises:
        FileNotFoundError: If *points_csv_path* does not exist.
        KeyError:          If required CSV columns are missing.
        RuntimeError:      On any other CSV read failure.
    """
    if not points_csv_path.exists():
        raise FileNotFoundError(
            f"Semidense points CSV not found: {points_csv_path}"
        )

    if out_path.exists() and not force:
        logger.info("Cache already exists, skipping: %s", out_path.name)
        return -1  # sentinel: skipped

    logger.info(
        "Building world-coord cache: %s → %s",
        points_csv_path.name, out_path.name,
    )

    uid_chunks: List[np.ndarray] = []
    xyz_chunks: List[np.ndarray] = []

    try:
        for chunk in pd.read_csv(
            points_csv_path,
            compression="gzip",
            chunksize=_CHUNK_SIZE,
            usecols=list(_REQUIRED_COLS),
            dtype={
                "uid":      np.int64,
                "px_world": np.float64,
                "py_world": np.float64,
                "pz_world": np.float64,
            },
        ):
            missing = _REQUIRED_COLS - set(chunk.columns)
            if missing:
                raise KeyError(
                    f"Semidense points CSV is missing required columns: {missing}"
                )
            uid_chunks.append(chunk["uid"].to_numpy(dtype=np.int64))
            xyz_chunks.append(
                np.stack([
                    chunk["px_world"].to_numpy(dtype=np.float64),
                    chunk["py_world"].to_numpy(dtype=np.float64),
                    chunk["pz_world"].to_numpy(dtype=np.float64),
                ], axis=1)
            )
    except KeyError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read {points_csv_path}: {exc}"
        ) from exc

    if not uid_chunks:
        logger.warning(
            "No data found in %s; empty cache will be written.",
            points_csv_path.name,
        )
        uids = np.empty(0, dtype=np.int64)
        xyz = np.empty((0, 3), dtype=np.float64)
    else:
        uids = np.concatenate(uid_chunks)
        xyz = np.vstack(xyz_chunks)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, uid=uids, xyz=xyz)
    logger.info("Wrote %d world points → %s", len(uids), out_path.name)
    return len(uids)


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def discover_sequences(raw_3d_dir: Path, split: str) -> List[str]:
    """Return sequence names for a split by scanning for semidense_points files.

    Looks for files matching ``*_semidense_points.csv.gz`` inside
    ``{raw_3d_dir}/{split}/`` and strips the suffix to derive sequence names.

    Args:
        raw_3d_dir: Root directory containing per-split sub-directories.
        split:      Split name, e.g. ``"train"``, ``"val"``, ``"test"``.

    Returns:
        Sorted list of sequence name strings.  Empty list if the directory
        does not exist or contains no matching files.
    """
    split_dir = raw_3d_dir / split
    if not split_dir.exists():
        logger.warning("Raw 3-D directory not found: %s", split_dir)
        return []
    sequences = []
    for p in sorted(split_dir.glob("*_semidense_points.csv.gz")):
        seq = p.name.replace("_semidense_points.csv.gz", "")
        sequences.append(seq)
    return sequences


# ---------------------------------------------------------------------------
# Split-level runner (explicit paths, no config schema knowledge)
# ---------------------------------------------------------------------------

def run_split(
    raw_3d_dir: Path,
    output_dir: Path,
    split: str,
    force: bool = False,
) -> Dict[str, int]:
    """Build world-coord caches for all sequences in one split directory.

    This function accepts explicit directory paths and has no knowledge of any
    project's config schema.

    Input pattern::

        {raw_3d_dir}/{split}/{sequence}_semidense_points.csv.gz

    Output pattern::

        {output_dir}/{split}/{sequence}_world_coords.npz

    Args:
        raw_3d_dir: Root directory that contains per-split sub-directories
                    with ``*_semidense_points.csv.gz`` files.
        output_dir: Root directory for output NPZ files; per-split and
                    per-sequence sub-directories are created automatically.
        split:      Split name, e.g. ``"train"``, ``"val"``, ``"test"``.
        force:      Rebuild even if the cache file already exists.

    Returns:
        Summary dict with integer keys ``"total"``, ``"built"``,
        ``"skipped"``, ``"failed"``.
    """
    sequences = discover_sequences(raw_3d_dir, split)
    if not sequences:
        logger.warning(
            "No semidense_points CSV files found in %s/%s", raw_3d_dir, split
        )
        return {"total": 0, "built": 0, "skipped": 0, "failed": 0}

    logger.info("Split %s: %d sequences found", split, len(sequences))
    n_built = n_skip = n_fail = 0

    for seq in sequences:
        csv_path = raw_3d_dir / split / f"{seq}_semidense_points.csv.gz"
        npz_path = output_dir / split / f"{seq}_world_coords.npz"
        try:
            n = build_world_coords_cache(csv_path, npz_path, force=force)
            if n == -1:
                n_skip += 1
            else:
                n_built += 1
        except Exception as exc:
            logger.error(
                "Failed for sequence %s: %s", seq, exc, exc_info=True
            )
            n_fail += 1

    return {
        "total": len(sequences),
        "built": n_built,
        "skipped": n_skip,
        "failed": n_fail,
    }
