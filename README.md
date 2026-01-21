# bb_utils

`bb_utils` is a lightweight Python utility package developed to support [SP-SCore](https://github.com/banafshebamdad/SP-SCore) research and development.  
It provides tools for dataset processing, label generation, and inspection utilities for SuperPoint training.

---

## Features

### 1. NPZ File Inspector
Comprehensive inspection tool for NumPy `.npz` files with detailed statistics and per-column analysis.

### 2. InCrowd-VI Label Generator
Converts InCrowd-VI semi-dense SLAM observations into SuperPoint-compatible training labels.

---

## Installation

### Local editable install (recommended during development)

```bash
cd /path/to/bb_utils
pip install -e .
```

This will install the package and all dependencies including:
- numpy
- pandas
- PyYAML
- tqdm

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

**Example output:**
```
================================================================================
NPZ File Inspection: file.npz
================================================================================
File size: 6.79 KB
Number of arrays: 1
Keys: pts
================================================================================

[1] Key: 'pts'
--------------------------------------------------------------------------------
  Data type:       float64
  Shape:           (468, 3)
  Dimensions:      2
  Total elements:  1,404
  Memory usage:    10.97 KB
  Category:        Numeric
  Minimum:         0.015067268162965775
  Maximum:         634.9155260324478
  Mean:            200.4165050833177

  Per-column statistics:
    Column 0:
      Min:         12.067086696624756
      Max:         634.9155260324478
      Mean:        324.1712256368154
    Column 1:
      Min:         3.739638328552246
      Max:         475.4317111968994
      Mean:        276.9944434749265
    Column 2:
      Min:         0.015067268162965775
      Max:         0.27862390875816345
      Mean:        0.08384613821115823
================================================================================
```

---

### InCrowd-VI Label Generator

Generate SuperPoint training labels from InCrowd-VI dataset:

```bash
# Using console script
generate-superpoint-labels config.yaml

# Or directly
python -m bb_utils.incrowdvi_superpoint_labels config.yaml

# With verbose logging
generate-superpoint-labels config.yaml --verbose
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

```yaml
data:
  mps_root: "/path/to/incrowdvi/mps_data"
  frames_root: "/path/to/incrowdvi/frames"
  output_labels_root: "/path/to/output/labels"
  
  camera_serials:
    left: "0072510f1b2107010500001910150001"
    right: "0072510f1b2107010800001127080001"

model:
  detection_threshold: 0.015  # minimum confidence
  top_k: 1000                 # max keypoints per frame
  
  confidence:
    method: "inverse_normalized"
    params: {}

output:
  generate_report: true
  report_file: "label_generation_report.txt"
```

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

## Usage as Python Module

```python
from bb_utils import generate_labels

# Generate labels
generate_labels('config.yaml')
```

---

## Project Structure

```
bb_utils/
├── configs/
│   └── incrowdvi_superpoint_labels.yaml  # Example config for SuperPoint
├── src/
│   └── bb_utils/
│       ├── __init__.py
│       ├── inspect_npz.py                    # NPZ inspector
│       └── incrowdvi_superpoint_labels.py    # SuperPoint label generator
├── pyproject.toml
└── README.md
```

---

## Development

The package is designed to be modular and extensible:

- Add new confidence computation methods in `incrowdvi_label_generator.py`
- Extend the NPZ inspector for custom array types
- Create new dataset processors following the same pattern

---

## Requirements

- Python >= 3.9
- numpy >= 1.23
- pandas >= 1.5.0
- PyYAML >= 6.0
- tqdm >= 4.65.0

---

## License

See repository for license information.

---

## Author

Banafshe Bamdad  
Development assisted by GitHub Copilot (Claude Sonnet 4.5)  
January 2026