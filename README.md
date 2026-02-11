# bb_utils

`bb_utils` is a lightweight Python utility package developed to support [SP-SCore](https://github.com/banafshebamdad/SP-SCore) research and development.  
It provides tools for dataset processing, label generation, and inspection utilities for SuperPoint training.

---

## Installation

### Local editable install

```bash
cd /path/to/bb_utils
pip install -e .
```

This will install the package and all dependencies.

---

## Tools

### NPZ Inspector

Inspect the contents of NumPy `.npz` files with detailed statistics:

```bash
# Using console script
inspect-npz path/to/file.npz

# Or directly
python -m bb_utils.inspect_npz path/to/file.npz
```

**Features:**
- Data type, shape, and dimensionality
- Memory usage in human-readable format
- Per-column statistics for 2D arrays
- Min/max/mean values
- Non-finite value detection (NaN, Inf)
- Support for boolean, string, and object arrays

---

### InCrowd-VI Label Generator for Superpoint Training

Generate SuperPoint training labels from InCrowd-VI dataset:

```bash
# Using console script
generate-superpoint-labels configs/incrowdvi_superpoint_labels.yaml

# Or directly
python -m bb_utils.incrowdvi_superpoint_labels config.yaml

# With verbose logging
generate-superpoint-labels configs/incrowdvi_superpoint_labels.yaml --verbose
```

**Input Data:**
- `semidense_observations.csv.gz` - 2D keypoint observations
- `semidense_points.csv.gz` - 3D points with uncertainty estimates

**Output Format:**
- One `.npz` file per frame
- Key: `'pts'`
- Shape: `(N, 3)` where columns are `[x, y, confidence]`
- Compatible with SuperPoint training pipeline

**Configuration:**

Create a YAML config file (see `configs/incrowdvi_superpoint_labels.yaml`):

**Features:**
- Memory-efficient chunked processing
- Configurable confidence computation
- Per-frame keypoint filtering
- Automatic report generation with statistics
- Progress tracking with tqdm

**Confidence Methods:**
- `inverse_normalized`: `confidence = 1 - (inv_dist_std / max_inv_dist_std)`
- `exponential`: `confidence = exp(-scale * inv_dist_std)`

---

### 3D Observation Organizer

Organize 3D observation data (semidense points and observations) into train/val/test directories based on a split CSV file:

```bash
# Basic usage
bb-organize-3d-obs \
  --data-root /path/to/ROOT \
  --split-csv split_crowd_density_with_frames.csv \
  --output-dir /path/to/output

# Preview operations without copying (dry run)
bb-organize-3d-obs \
  --data-root /path/to/ROOT \
  --split-csv split.csv \
  --output-dir /path/to/output \
  --dry-run

# Create symlinks instead of copying files (saves disk space)
bb-organize-3d-obs \
  --data-root /path/to/ROOT \
  --split-csv split.csv \
  --output-dir /path/to/output \
  --symlink

# Verbose output
bb-organize-3d-obs \
  --data-root /path/to/ROOT \
  --split-csv split.csv \
  --output-dir /path/to/output \
  --verbose
```

**Input Structure:**

This structure is based on data created from the [InCrowd-VI](https://github.com/banafshebamdad/InCrowd-VI) dataset:

```
ROOT/
├── Scene1/
│   └── mps_sequence_name_vrs/
│       ├── semidense_points.csv.gz
│       └── semidense_observations.csv.gz
└── Scene2/
    └── mps_another_sequence_vrs/
        ├── semidense_points.csv.gz
        └── semidense_observations.csv.gz
```

**Split CSV Format:**

A sample split CSV file is provided in [configs/split_crowd_density_with_frames.csv](configs/split_crowd_density_with_frames.csv).


**Output Structure:**
```
output_dir/
└── raw_3d_observations/
    ├── train/
    │   ├── IMS_TE21_LEA_lab_semidense_points.csv.gz
    │   └── IMS_TE21_LEA_lab_semidense_observations.csv.gz
    ├── val/
    │   ├── AND_Lib_floor5_1_semidense_points.csv.gz
    │   └── AND_Lib_floor5_1_semidense_observations.csv.gz
    └── test/
        └── ...
```

**Features:**
- Automatically creates train/val/test directory structure
- Copies files with sequence name prefix for easy identification
- Optional symlink mode to save disk space
- Dry run mode to preview operations
- Detailed logging and error reporting
- Progress tracking with tqdm
- Validates source files exist before processing

---

## Usage as Python Module

```python
# Option 1: Import from main package
from bb_utils import generate_labels, inspect_npz_file, organize_3d_observations

# Generate labels
generate_labels('config.yaml')

# Organize 3D observations
from pathlib import Path
organize_3d_observations(
    data_root_dir=Path('/path/to/ROOT'),
    split_csv_path=Path('split.csv'),
    output_dir=Path('/path/to/output'),
    dry_run=False,
    symlink=False,
    verbose=True
)

# Option 2: Import from submodules
from bb_utils.label_generation import generate_labels
from bb_utils.utils import inspect_npz_file
from bb_utils.data_preparation import organize_3d_observations
```
