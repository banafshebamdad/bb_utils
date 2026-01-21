# File Renaming Summary - SuperPoint vs SP-SCore Distinction

## Date: January 21, 2026

## Purpose
Renamed label generation files to clearly indicate they are for **SuperPoint** training, leaving room for future **SP-SCore** label generators.

## Files Renamed

### Main Script
- **Old**: `src/bb_utils/incrowdvi_label_generator.py`
- **New**: `src/bb_utils/incrowdvi_superpoint_labels.py`

### Configuration
- **Old**: `configs/incrowdvi_label_generation.yaml`
- **New**: `configs/incrowdvi_superpoint_labels.yaml`

### Example Script
- **Old**: `examples/incrowdvi_label_generator_example.py`
- **New**: `examples/incrowdvi_superpoint_labels_example.py`

## Console Command Changed

### Old Command
```bash
generate-incrowdvi-labels config.yaml
```

### New Command
```bash
generate-superpoint-labels config.yaml
```

## Files Updated

All references to the old filenames were updated in:

1. `src/bb_utils/__init__.py` - Import statements
2. `pyproject.toml` - Console script entry point
3. `README.md` - Documentation and examples
4. `docs/QUICKSTART.md` - Usage instructions
5. `docs/IMPLEMENTATION_SUMMARY.md` - File paths and commands
6. `examples/incrowdvi_superpoint_labels_example.py` - Import and config references
7. `configs/incrowdvi_superpoint_labels.yaml` - Header comments

## New Files Created

1. `docs/NAMING_CONVENTION.md` - Comprehensive naming convention guide

## Future Development

The naming convention now supports:

### SuperPoint (Current)
- `incrowdvi_superpoint_labels.py`
- `generate-superpoint-labels` command

### SP-SCore (Planned)
- `incrowdvi_spscore_labels.py` *(to be developed)*
- `generate-spscore-labels` command *(to be added)*

## Benefits

1. **Clear distinction**: Immediately obvious which model the labels are for
2. **No confusion**: Scripts explicitly named for their target training framework
3. **Extensible**: Easy to add new label generators for other models
4. **Maintainable**: Changes to one don't affect the other
5. **Self-documenting**: Filenames indicate purpose

## Migration Notes

If you have existing code using the old names:

```python
# Old import (no longer works)
from bb_utils.incrowdvi_label_generator import generate_labels

# New import
from bb_utils.incrowdvi_superpoint_labels import generate_labels
```

```bash
# Old command (no longer works)
generate-incrowdvi-labels config.yaml

# New command
generate-superpoint-labels config.yaml
```

## Current Project Structure

```
bb_utils/
├── src/bb_utils/
│   ├── __init__.py
│   ├── incrowdvi_superpoint_labels.py    ✓ SuperPoint labels
│   └── inspect_npz.py
│
├── configs/
│   └── incrowdvi_superpoint_labels.yaml  ✓ SuperPoint config
│
├── examples/
│   └── incrowdvi_superpoint_labels_example.py
│
├── docs/
│   ├── NAMING_CONVENTION.md              ✓ NEW
│   ├── QUICKSTART.md
│   └── IMPLEMENTATION_SUMMARY.md
│
├── README.md
├── CHANGELOG.md
└── pyproject.toml
```

## Verification

To verify the installation works:

```bash
# Install package
cd /home/ubuntu/bb_utils
pip install -e .

# Check new command is available
generate-superpoint-labels --help

# Run example
python examples/incrowdvi_superpoint_labels_example.py
```

---

**Author**: Banafshe Bamdad  
**Date**: January 21, 2026
