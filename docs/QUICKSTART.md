# InCrowd-VI Label Generator - Quick Start Guide

## Installation

1. **Install the package:**
   ```bash
   cd /home/ubuntu/bb_utils
   pip install -e .
   ```

2. **Verify installation:**
   ```bash
   generate-superpoint-labels --help
   inspect-npz --help
   ```

## Configuration

1. **Copy the example config:**
   ```bash
   cp configs/incrowdvi_superpoint_labels.yaml my_config.yaml
   ```

2. **Edit the configuration:**
   ```yaml
   data:
     mps_root: "/path/to/your/incrowdvi/mps_data"
     frames_root: "/path/to/your/incrowdvi/frames"
     output_labels_root: "/path/to/output/labels"
   
   model:
     detection_threshold: 0.015
     top_k: 1000
   ```

## Usage

### Generate Labels

```bash
# Basic usage
generate-superpoint-labels my_config.yaml

# With verbose output
generate-superpoint-labels my_config.yaml --verbose
```

### Inspect Generated Labels

```bash
# Inspect a single label file
inspect-npz /path/to/output/labels/scene/frame.npz

# The output shows per-column statistics:
# Column 0: x-coordinates
# Column 1: y-coordinates  
# Column 2: confidence scores
```

## Expected Dataset Structure

### Input (MPS Data):
```
mps_root/
├── scene1/
│   ├── mps_sequence1_vrs/
│   │   ├── semidense_observations.csv.gz
│   │   └── semidense_points.csv.gz
│   └── mps_sequence2_vrs/
│       ├── semidense_observations.csv.gz
│       └── semidense_points.csv.gz
└── scene2/
    └── mps_sequence3_vrs/
        ├── semidense_observations.csv.gz
        └── semidense_points.csv.gz
```

### Input (Frames):
Frames are directly in frames_root (not nested by scene):
```
frames_root/
├── sequence1_L_1234567890.png
├── sequence1_R_1234567890.png
├── sequence2_L_1234567891.png
├── sequence2_R_1234567891.png
└── ...
```
Note: Timestamps in filenames are in microseconds (same as CSV).

### Output (Labels):
Labels are directly in output_labels_root (not nested by scene):
```
output_labels_root/
├── sequence1_L_1234567890.npz
├── sequence1_R_1234567890.npz
├── sequence2_L_1234567891.npz
├── sequence2_R_1234567891.npz
├── ...
└── label_generation_report.txt
```

## Output Report

After processing, a report file will be generated with statistics:

```
================================================================================
InCrowd-VI SuperPoint Label Generation Report
================================================================================

Configuration:
  Detection threshold: 0.015
  Top-k keypoints: 1000
  Confidence method: inverse_normalized

Sequence: scene1/sequence1
  Frames processed: 1234
  Keypoints per frame:
    Min:  459
    Max:  1000
    Mean: 687.45
    Std:  123.21

================================================================================
Global Statistics
================================================================================
Total frames: 1234
Keypoints per frame (global):
  Min:  459
  Max:  1000
  Mean: 687.45
  Std:  123.21
```

## Troubleshooting

### No frames found
- Check that `frames_root` and `mps_root` paths are correct
- Verify frame filename format matches: `[sequence]_[L/R]_[timestamp_us].png`
- Ensure camera serials in config match your dataset
- Remember: frames are directly in frames_root, not in scene subdirectories

### Low keypoint counts
- Lower `detection_threshold` (e.g., 0.01 or 0.005)
- Increase `top_k` if many keypoints have high confidence
- Check the report to see confidence distribution

### Memory issues
- The script processes in chunks, but very large datasets may need more RAM
- Process scenes one at a time by organizing your directory structure

## Python API Usage

```python
from bb_utils import generate_labels

# Generate SuperPoint labels
generate_labels('my_config.yaml')
```

## Testing with Example Script

```bash
cd /home/ubuntu/bb_utils
python examples/incrowdvi_superpoint_labels_example.py
```

This will demonstrate:
- Confidence computation methods
- Keypoint filtering
- Available options

## Next Steps

1. Generate labels for your training set
2. Generate labels for your validation set (update `export_folder` in config)
3. Use `inspect-npz` to verify a few random output files
4. Check the report for quality metrics
5. Use the labels for SuperPoint training

## Support

For issues or questions:
- Check module docstrings: `python -c "import bb_utils.incrowdvi_superpoint_labels; help(bb_utils.incrowdvi_superpoint_labels)"`
- Review the comprehensive documentation in the script itself
- Check the example config file for all available options
