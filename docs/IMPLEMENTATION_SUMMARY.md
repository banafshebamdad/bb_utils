# InCrowd-VI SuperPoint Label Generator - Implementation Summary

## Project Overview

This implementation provides a production-ready Python script for generating **SuperPoint** training labels from the InCrowd-VI dataset. The script processes semi-dense SLAM observations and converts them into the format required by the SuperPoint training pipeline.

**Note**: This is specifically for SuperPoint labels. A separate script will be developed for SP-SCore label generation.

## Files Created

```
/home/ubuntu/bb_utils/
├── src/bb_utils/
│   ├── __init__.py                          # Package initialization
│   ├── incrowdvi_superpoint_labels.py       # SuperPoint label generation (700+ lines)
│   └── inspect_npz.py                        # NPZ file inspector (updated)
│
├── configs/
│   └── incrowdvi_superpoint_labels.yaml     # Config for SuperPoint labels
│
├── examples/
│   └── incrowdvi_superpoint_labels_example.py # Usage examples
│
├── docs/
│   └── QUICKSTART.md                         # Quick start guide
│
├── pyproject.toml                            # Package metadata (updated)
└── README.md                                 # Main documentation (updated)
```

## Key Features Implemented

### 1. **Modular Architecture**
- **Confidence computation**: Pluggable system with multiple methods
  - `inverse_normalized`: Default method using normalized uncertainty
  - `exponential`: Exponential decay alternative
  - Easy to add new methods via registry pattern

- **Separate functions for**:
  - Frame discovery and matching
  - Timestamp conversion
  - Keypoint filtering
  - Report generation

### 2. **Memory Efficiency**
- **Chunked CSV processing**: Reads large files in 100K row chunks
- **Streaming approach**: Processes frames as data arrives
- **Efficient groupby**: Uses pandas groupby for per-frame aggregation

### 3. **Robustness**
- **Comprehensive error handling**: File validation, missing data checks
- **Logging**: Configurable logging levels (INFO/DEBUG)
- **Progress tracking**: tqdm progress bars for long operations
- **Validation**: Input path checking, camera serial verification

### 4. **Configurability**
- **YAML configuration**: All parameters externalized
- **Flexible paths**: Separate MPS data, frames, and output locations
- **Adjustable thresholds**: Confidence and top-k filtering
- **Method selection**: Choose confidence computation strategy

### 5. **Quality Assurance**
- **Automatic reporting**: Per-sequence and global statistics
- **Keypoint counts**: Min/max/mean/std for quality monitoring
- **Inspection tools**: Compatible with updated inspect-npz tool

## Label Format Compatibility

The generated labels match the exact format expected by SuperPoint training:

```python
# Output NPZ file structure
{
    'pts': np.array([
        [x1, y1, confidence1],
        [x2, y2, confidence2],
        ...
        [xN, yN, confidenceN]
    ], dtype=np.float64)
}
```

This is identical to the output of `export_detector_homoAdapt` from the original SuperPoint pipeline.

## Data Flow

```
┌─────────────────────────────────────────┐
│ InCrowd-VI Dataset                      │
│ ├── semidense_observations.csv.gz       │
│ │   (uid, timestamp, camera, u, v)      │
│ └── semidense_points.csv.gz             │
│     (uid, 3D pos, inv_dist_std, ...)    │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ Label Generator                          │
│ 1. Load points → compute confidence     │
│ 2. Create uid → confidence mapping      │
│ 3. Read observations in chunks          │
│ 4. Join with confidence                 │
│ 5. Group by frame                       │
│ 6. Filter & sort keypoints              │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ Output Labels (.npz files)               │
│ ├── scene1/                             │
│ │   ├── sequence1_L_timestamp.npz      │
│ │   └── sequence1_R_timestamp.npz      │
│ └── label_generation_report.txt        │
└─────────────────────────────────────────┘
```

## Configuration Parameters

### Critical Parameters
- `detection_threshold`: 0.015 (matches SuperPoint default)
- `top_k`: 600-1000 (typical for crowd scenes)
- `confidence.method`: 'inverse_normalized' (recommended)

### Camera Mapping
```yaml
camera_serials:
  left: "0072510f1b2107010500001910150001"
  right: "0072510f1b2107010800001127080001"
```

### Timestamp Handling
- CSV timestamps: microseconds
- Frame filenames: nanoseconds
- Conversion: multiply by 1000

## Installation & Usage

### Install Package
```bash
cd /home/ubuntu/bb_utils
pip install -e .
```

### Generate Labels
```bash
# Create config from template
cp configs/incrowdvi_superpoint_labels.yaml my_config.yaml

# Edit paths in my_config.yaml

# Run SuperPoint label generation
generate-superpoint-labels my_config.yaml --verbose
```

### Inspect Results
```bash
# Inspect a generated label file
inspect-npz output/labels/scene/frame.npz

# Check the report
cat output/labels/label_generation_report.txt
```

## Performance Characteristics

### Memory Usage
- **Minimal**: Points loaded once, observations streamed
- **Scalable**: Can handle datasets with millions of observations
- **Estimated RAM**: 2-4 GB for typical InCrowd-VI sequences

### Processing Speed
- **Chunked reading**: 100K observations per chunk
- **Progress tracking**: Real-time feedback via tqdm
- **Estimated time**: ~1-5 minutes per sequence (depends on size)

## Extensibility

### Adding New Confidence Methods

```python
def compute_confidence_my_method(inv_dist_std: np.ndarray, 
                                 param1: float,
                                 **kwargs) -> np.ndarray:
    """Your custom confidence computation."""
    confidence = ... # your formula
    return confidence

# Register the method
CONFIDENCE_METHODS['my_method'] = compute_confidence_my_method
```

Then update config:
```yaml
model:
  confidence:
    method: "my_method"
    params:
      param1: 0.5
```

### Processing Multiple Datasets

The script automatically discovers all scenes and sequences in the MPS root directory, making it easy to process entire datasets at once.

## Validation

### Expected Output
- One `.npz` file per frame
- Keypoint counts matching the report
- Confidence values in [0, 1] range
- Compatible with SuperPoint training

### Quality Checks
1. **Verify keypoint counts**: Should be reasonable (100-1000 per frame)
2. **Check confidence distribution**: Mean should be > threshold
3. **Inspect sample files**: Use `inspect-npz` on random outputs
4. **Review report statistics**: Look for anomalies

## Integration with SuperPoint Training

The generated labels can be used directly with the SuperPoint training pipeline:

1. **Place labels** in the expected directory structure
2. **Update dataset loader** to point to label directory
3. **Configure training** with matching threshold
4. **Train detector** using the pseudo-ground truth labels
5. **Train descriptor** using homography adaptation (automatic)

## Documentation

- **Module docstring**: 80+ lines of comprehensive documentation
- **Function docstrings**: All functions documented with parameters
- **README.md**: User-facing documentation
- **QUICKSTART.md**: Step-by-step guide
- **Example script**: Demonstrates all features

## Testing

Run the example script to verify installation:
```bash
python examples/incrowdvi_superpoint_labels_example.py
```

This tests:
- Module imports
- Confidence computation
- Keypoint filtering
- Method registry

## Known Limitations & Assumptions

1. **Timestamp precision**: Assumes microsecond/nanosecond units
2. **Camera serials**: Must match InCrowd-VI dataset
3. **Frame format**: Expects specific filename pattern
4. **CSV format**: Assumes standard InCrowd-VI column order

## Future Enhancements

Potential improvements:
- [ ] Multi-processing for parallel scene processing
- [ ] Adaptive thresholding based on keypoint density
- [ ] Visualization tools for quality assessment
- [ ] Support for other SLAM output formats
- [ ] Integration with SuperPoint dataloader

## Author & License

**Author**: Banafshe Bamdad  
**Date**: January 21, 2026  
**Assistant**: GitHub Copilot (Claude Sonnet 4.5)

See repository for license information.
