# World-Coordinate Cache Builder

`bb_utils.data_preparation.world_coords_cache` provides a generic,
project-independent utility for converting Aria MPS semidense point cloud CSV
files into compact NumPy NPZ caches that map each global UID to its 3-D world
coordinates.

The module contains **no project-specific logic**.  All paths are passed
explicitly so it can be reused across any Aria MPS dataset.

It ships with a standalone CLI (`bb-cache-world-coords`) that can be run
directly without any project config file, and is also used internally by
sp-score via `bb_utils.data_preparation.world_coords_cache.run_split`.

---

## Table of Contents

- [Purpose](#purpose)
- [CLI usage](#cli-usage)
- [Input format](#input-format)
- [Output contract (NPZ)](#output-contract-npz)
- [Public API](#public-api)
  - [build_world_coords_cache](#build_world_coords_cache)
  - [discover_sequences](#discover_sequences)
  - [run_split](#run_split)
- [Directory layout](#directory-layout)
- [Python usage example](#python-usage-example)
- [sp-score integration](#sp-score-integration)
- [Dependencies](#dependencies)

---

## Purpose

The Aria MPS semidense SLAM pipeline outputs two gzipped CSV files per
recording sequence:

- `*_semidense_points.csv.gz` — 3-D world positions of tracked points
- `*_semidense_observations.csv.gz` — per-frame 2-D observations of those points

Loading and joining these files at runtime for every pipeline run is slow.
This module pre-builds a compact NPZ cache (`uid → xyz`) from the points CSV
so that downstream consumers can look up world coordinates by UID in O(1) with
a simple dictionary, without reading the large CSV again.

---

## CLI usage

```bash
# Cache one split
bb-cache-world-coords \
  --raw-3d-dir /home/ubuntu/raw_3d_observations \
  --output-dir dataset/incrowdvi/world_coords \
  --split train

# Cache multiple splits at once
bb-cache-world-coords \
  --raw-3d-dir /home/ubuntu/raw_3d_observations \
  --output-dir dataset/incrowdvi/world_coords \
  --split train --split val

# Rebuild even if cache already exists
bb-cache-world-coords \
  --raw-3d-dir /home/ubuntu/raw_3d_observations \
  --output-dir dataset/incrowdvi/world_coords \
  --split train --force
```

CLI flags:

| Flag | Required | Description |
|---|---|---|
| `--raw-3d-dir` | yes | Root dir with `{split}/*_semidense_points.csv.gz` files |
| `--output-dir` | yes | Root dir for output NPZ files (sub-dirs created automatically) |
| `--split` | no | `train`/`val`/`test`; repeat for multiple splits (default: all three) |
| `--force` | no | Overwrite existing NPZ files |
| `--verbose` | no | Enable DEBUG-level logging |

---

## Input format

Each input file must be a gzip-compressed CSV with at least the following
columns:

| Column | dtype | Description |
|---|---|---|
| `uid` | int64 | Global world-point identifier (matches `semidense_observations.csv.gz`) |
| `px_world` | float64 | X coordinate in the Aria MPS odometry frame (metres) |
| `py_world` | float64 | Y coordinate (metres) |
| `pz_world` | float64 | Z coordinate (metres) |

Additional columns are silently ignored.

---

## Output contract (NPZ)

Each output file `{output_dir}/{split}/{sequence}_world_coords.npz` contains:

| Key | dtype | Shape | Description |
|---|---|---|---|
| `uid` | int64 | (K,) | World-point identifiers |
| `xyz` | float64 | (K, 3) | `(px_world, py_world, pz_world)` in the Aria MPS odometry frame |

Row `i` of `xyz` corresponds to `uid[i]`.  The order within the NPZ matches
the order of rows in the source CSV.

Loading the cache into a uid→xyz dict:

```python
import numpy as np

data = np.load("sequence_world_coords.npz", allow_pickle=False)
world_coords = {int(uid): data["xyz"][i] for i, uid in enumerate(data["uid"])}
# world_coords[uid]  →  float64 array of shape (3,)
```

---

## Public API

### `build_world_coords_cache`

```python
from bb_utils.data_preparation.world_coords_cache import build_world_coords_cache

n = build_world_coords_cache(
    points_csv_path=Path("raw/train/corridor_01_semidense_points.csv.gz"),
    out_path=Path("cache/train/corridor_01_world_coords.npz"),
    force=False,
)
# Returns: number of points written, or -1 if skipped (already exists)
```

| Arg | Type | Default | Description |
|---|---|---|---|
| `points_csv_path` | `Path` | required | Source gzipped CSV |
| `out_path` | `Path` | required | Destination NPZ; parent created if absent |
| `force` | `bool` | `False` | Overwrite existing NPZ when `True` |

Raises `FileNotFoundError`, `KeyError` (missing columns), or `RuntimeError`
(CSV read failure).

---

### `discover_sequences`

```python
from bb_utils.data_preparation.world_coords_cache import discover_sequences

seqs = discover_sequences(raw_3d_dir=Path("/data/raw"), split="train")
# Returns: ["corridor_01", "library_02", ...]
```

Scans `{raw_3d_dir}/{split}/` for files matching
`*_semidense_points.csv.gz` and returns the sorted list of sequence name
strings (suffix stripped).

---

### `run_split`

```python
from bb_utils.data_preparation.world_coords_cache import run_split

summary = run_split(
    raw_3d_dir=Path("/data/raw"),
    output_dir=Path("/data/cache"),
    split="train",
    force=False,
)
# Returns: {"total": 12, "built": 10, "skipped": 2, "failed": 0}
```

Calls `build_world_coords_cache` for every sequence discovered in the split
directory.  Per-sequence failures are logged and counted but do not abort the
remaining sequences; the summary dict lets the caller decide whether to exit
non-zero.

| Arg | Type | Default | Description |
|---|---|---|---|
| `raw_3d_dir` | `Path` | required | Root dir with per-split sub-directories |
| `output_dir` | `Path` | required | Root dir for output NPZ files |
| `split` | `str` | required | `"train"`, `"val"`, or `"test"` |
| `force` | `bool` | `False` | Rebuild even if cache already exists |

---

## Directory layout

```
raw_3d_dir/
└── train/
    ├── corridor_01_semidense_points.csv.gz
    └── library_02_semidense_points.csv.gz

output_dir/                    ← created automatically
└── train/
    ├── corridor_01_world_coords.npz
    └── library_02_world_coords.npz
```

---

## Python usage example

```python
from pathlib import Path
from bb_utils.data_preparation.world_coords_cache import run_split

summary = run_split(
    raw_3d_dir=Path("/home/ubuntu/raw_3d_observations"),
    output_dir=Path("dataset/incrowdvi/world_coords"),
    split="train",
)
print(summary)
# {'total': 8, 'built': 8, 'skipped': 0, 'failed': 0}
```

---

## sp-score integration

`bb-cache-world-coords` is the preferred way to build the world-coordinate
cache.  It requires no sp-score config file and can be run as a standalone
data preparation step before setting up sp-score.

The sp-score pipeline reads the resulting NPZ files via
`sp_score.static_reliability.frame.load_world_coords_npz`, which loads them
into a `Dict[int, np.ndarray]` (uid → xyz) for use in the temporal score
computation.

```
bb-cache-world-coords                           ← standalone, no config needed
  └── bb_utils.data_preparation.world_coords_cache.run_split()
        └── build_world_coords_cache() per sequence → NPZ

sp-score static reliability pipeline
  └── frame.load_world_coords_npz()             ← reads the NPZ
        └── {world_coords_dir}/{split}/{sequence}_world_coords.npz
```

---

## Dependencies

| Dependency | Required | Install |
|---|---|---|
| `numpy` | yes | `pip install numpy` |
| `pandas` | yes | `pip install pandas` |
