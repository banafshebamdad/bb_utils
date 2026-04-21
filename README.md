# bb_utils

`bb_utils` is a lightweight Python utility package developed to support [SP-SCore](https://github.com/banafshebamdad/SP-SCore) research and development.  
It provides tools for dataset processing, label generation, and inspection utilities for SuperPoint training.

---

## Table of Contents

- [Installation](#installation)
  - [Local editable install](#local-editable-install)
- [Tools](#tools)
  - [NPZ Inspector](#npz-inspector)
  - [InCrowd-VI Label Generator for Superpoint Training](#incrowd-vi-label-generator-for-superpoint-training)
  - [3D Observation Organizer](#3d-observation-organizer)
  - [Confidence Decay Rate Analyzer](#confidence-decay-rate-analyzer)

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

| Tool | Description |
|------|-------------|
| [NPZ Inspector](#npz-inspector) | Inspect contents of NumPy `.npz` files with detailed statistics |
| [InCrowd-VI Label Generator for Superpoint Training](#incrowd-vi-label-generator-for-superpoint-training) | Generate SuperPoint training labels from InCrowd-VI dataset |
| [3D Observation Organizer](#3d-observation-organizer) | Organize 3D observation data into train/val/test directories |
| [Confidence Decay Rate Analyzer](#confidence-decay-rate-analyzer) | Analyze `inv_dist_std` distribution and compute optimal confidence decay parameters |

---

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

### Confidence Decay Rate Analyzer

Analyze the distribution of `inv_dist_std` values from InCrowd-VI semi-dense 3D point cloud and compute optimal confidence decay rate parameters for computing confidence:

```bash
# Basic usage
bb-analyze-decay-rate \
  --input /path/to/raw_3d_observations/train \
  --output-dir logs/decay-analysis

# Full analysis with plots and per-file statistics with custom MPS threshold parameters
bb-analyze-decay-rate \
  --input /path/to/raw_3d_observations/train \
  --output-dir logs/decay-analysis \
  --x-thr 0.005 \
  --lambda 100.0 \
  --c-min 0.1 \
  --per-file \
  --plot \
  --percentiles \
  --verbose

```

**Purpose:**

This tool helps determine the optimal `decay_rate` (α) parameter for the exponential confidence function used in heatmap generation. The confidence function is:

```
confidence(x) = exp(-α × x)
```

where `x` is the `inv_dist_std` value (inverse distance standard deviation) from Meta MPS semi-dense SLAM output.

**Output Files:**

- `confidence_analysis.json`: nalysis report including:
  - Confidence parameters (α, x_thr, x_max, λ, c_min)
  - Distribution statistics (min, max, mean, std)
  - Filtering report (outlier counts and percentages)
  - Confidence distribution statistics
  - Sanity checks for confidence function
  - Per-file statistics (if `--per-file` enabled)

- `confidence_summary.csv` - Tabular summary with:
  - Overall and per-file statistics
  - Confidence parameters
  - Filtering and confidence distribution metrics

- **Plots** (if `--plot` enabled):
  - `inv_dist_std_distribution.png` - Histogram of inv_dist_std values with threshold markers
  - `confidence_curve.png` - Confidence function visualization
  - `confidence_histogram.png` - Distribution of confidence values

**CLI Parameters:**

- `--input`: Path(s) to CSV.gz files or directories (can specify multiple)
- `--output-dir`: Directory where results will be saved
- `--column`: Column name to analyze (default: `inv_dist_std`)
- `--x-thr`: MPS nominal threshold (default: `0.005`)
- `--lambda`: Safety factor λ > 1, controls outlier cutoff: x_max = λ × x_thr
- `--c-min`: Minimum confidence at x_max (default: `0.1`)
- `--per-file`: Compute statistics for each input file separately
- `--plot`: Generate visualization plots
- `--percentiles`: Compute distribution percentiles (requires tdigest)
- `--chunk-size`: CSV chunk size for memory-efficient processing (default: `100000`)
- `--verbose`, `-v`: Enable detailed logging

**Confidence Function:**

The MPS-threshold-based confidence function uses three user-specified parameters:

1. **x_thr**: MPS nominal threshold (0.005)
2. **λ**: Safety factor for outlier rejection (e.g., 100.0 means x_max = 100 × 0.005 = 0.5)
3. **c_min**: Minimum acceptable confidence at x_max (e.g., 0.1 = 10%)

From these, the tool automatically computes:
- **x_max** = λ × x_thr (outlier cutoff threshold)
- **α (alpha/decay_rate)** = -ln(c_min) / (x_max - x_thr)

The computed `decay_rate` can then be used in the [heatmap generation config](https://github.com/banafshebamdad/sp-score/blob/main/configs/preprocessing_heatmap_generation.yaml).

**Dependencies:**

```bash
pip install bb-utils[all]
```

---

