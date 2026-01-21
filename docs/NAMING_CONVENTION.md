# bb_utils Naming Convention

## Overview

This document defines the naming convention for label generation scripts in the bb_utils package to ensure clarity and avoid confusion between different training targets.

## Naming Pattern

```
incrowdvi_<model>_labels.py
```

Where `<model>` identifies the target training framework.

## Current Scripts

### SuperPoint Label Generator
- **Script**: `incrowdvi_superpoint_labels.py`
- **Config**: `configs/incrowdvi_superpoint_labels.yaml`
- **Example**: `examples/incrowdvi_superpoint_labels_example.py`
- **Console command**: `generate-superpoint-labels`
- **Purpose**: Generate labels for training original SuperPoint
- **Output format**: 
  - Key: `'pts'`
  - Shape: `(N, 3)`
  - Columns: `[x, y, confidence]`

### SP-SCore Label Generator (Future)
- **Script**: `incrowdvi_spscore_labels.py` *(to be developed)*
- **Config**: `configs/incrowdvi_spscore_labels.yaml`
- **Example**: `examples/incrowdvi_spscore_labels_example.py`
- **Console command**: `generate-spscore-labels`
- **Purpose**: Generate labels for training SP-SCore
- **Output format**: *(to be defined based on SP-SCore requirements)*

## Rationale

### Why separate scripts?

1. **Different output formats**: SuperPoint and SP-SCore may require different label structures
2. **Different requirements**: Processing logic may differ (e.g., matching requirements, confidence computation)
3. **Clarity**: Clear naming prevents accidental use of wrong label generator
4. **Maintainability**: Easier to update one without affecting the other
5. **Documentation**: Each script can have specific documentation for its target

## Adding New Label Generators

When adding a new label generator for a different model or dataset:

1. **Choose a clear name**: `<dataset>_<model>_labels.py`
2. **Create matching files**:
   - Script: `src/bb_utils/<dataset>_<model>_labels.py`
   - Config: `configs/<dataset>_<model>_labels.yaml`
   - Example: `examples/<dataset>_<model>_labels_example.py`
3. **Add console script** in `pyproject.toml`:
   ```toml
   [project.scripts]
   generate-<model>-labels = "bb_utils.<dataset>_<model>_labels:main"
   ```
4. **Update documentation**:
   - Add to README.md
   - Create QUICKSTART section
   - Update this naming convention doc

## File Organization

```
bb_utils/
├── src/bb_utils/
│   ├── incrowdvi_superpoint_labels.py    # SuperPoint labels
│   ├── incrowdvi_spscore_labels.py       # SP-SCore labels (future)
│   └── <dataset>_<model>_labels.py       # Other label generators
│
├── configs/
│   ├── incrowdvi_superpoint_labels.yaml  # SuperPoint config
│   ├── incrowdvi_spscore_labels.yaml     # SP-SCore config (future)
│   └── <dataset>_<model>_labels.yaml     # Other configs
│
└── examples/
    ├── incrowdvi_superpoint_labels_example.py
    ├── incrowdvi_spscore_labels_example.py     # (future)
    └── <dataset>_<model>_labels_example.py
```

## Console Commands

Follow the pattern: `generate-<model>-labels`

Examples:
- `generate-superpoint-labels config.yaml`
- `generate-spscore-labels config.yaml`

## Backward Compatibility

If you have scripts using the old names:
- Old: `generate-incrowdvi-labels` 
- New: `generate-superpoint-labels`

The old command will no longer work after renaming. Update your scripts accordingly.

## Questions?

Contact: Banafshe Bamdad  
Date: January 21, 2026
