# Package Restructuring - Migration Guide

**Date:** January 21, 2026  
**Reason:** Organize package for scalability as new features are added

---

## What Changed

The `bb_utils` package has been reorganized from a flat structure into logical subdirectories:

### Old Structure (Flat)
```
src/bb_utils/
├── __init__.py
├── inspect_npz.py
└── incrowdvi_superpoint_labels.py
```

### New Structure (Organized)
```
src/bb_utils/
├── __init__.py
├── utils/
│   ├── __init__.py
│   └── inspect_npz.py
└── label_generation/
    ├── __init__.py
    └── incrowdvi_superpoint_labels.py
```

---

## Why This Change?

1. **Future SP-SCore labels** - Multiple label generators warrant their own subdirectory
2. **Clear separation** - Utilities vs. label generation vs. future data processing
3. **Scalability** - Easier to add new functionality without cluttering root
4. **Better organization** - Clear intent from directory structure

---

## Migration Required

### Console Scripts (No change needed)
Console scripts still work the same way:
```bash
inspect-npz file.npz
generate-superpoint-labels config.yaml
```

### Python Imports (Updated paths)

**Option 1: Use top-level imports (Recommended)**
```python
# Still works - imports are re-exported from main __init__.py
from bb_utils import generate_labels
from bb_utils import inspect_npz_file
```

**Option 2: Use explicit submodule imports**
```python
# New: More explicit about where functions come from
from bb_utils.label_generation import generate_labels
from bb_utils.utils import inspect_npz_file
```

**Old imports that NO LONGER WORK:**
```python
# ❌ BROKEN - module moved
from bb_utils.incrowdvi_superpoint_labels import generate_labels

# ❌ BROKEN - module moved  
from bb_utils.inspect_npz import inspect_npz_file
```

**Fixed versions:**
```python
# ✓ Works - use top-level import
from bb_utils import generate_labels

# ✓ Works - use new submodule path
from bb_utils.label_generation import generate_labels

# ✓ Works - use top-level import
from bb_utils import inspect_npz_file

# ✓ Works - use new submodule path
from bb_utils.utils import inspect_npz_file
```

---

## Updated Files

### Core Package Files
- `src/bb_utils/__init__.py` - Updated to import from submodules
- `src/bb_utils/utils/__init__.py` - New, exports inspect_npz utilities
- `src/bb_utils/label_generation/__init__.py` - New, exports label generation functions
- `pyproject.toml` - Updated console script entry points

### Examples and Documentation
- `examples/incrowdvi_superpoint_labels_example.py` - Updated imports
- `docs/QUICKSTART.md` - Added new import examples
- `docs/IMPLEMENTATION_SUMMARY.md` - Updated file structure diagram
- `README.md` - Added import options

---

## What to Do

### If you have local code using bb_utils:

1. **Reinstall the package:**
   ```bash
   cd /home/ubuntu/bb_utils
   pip install -e .
   ```

2. **Update your imports** (choose one):
   
   **Easiest:** Use top-level imports (no code changes if you were already doing this)
   ```python
   from bb_utils import generate_labels
   ```
   
   **More explicit:** Use submodule imports
   ```python
   from bb_utils.label_generation import generate_labels
   from bb_utils.utils import inspect_npz_file
   ```

3. **Test your code** to ensure imports work

---

## Future Structure

As more functionality is added, the structure will grow like this:

```
src/bb_utils/
├── __init__.py
├── utils/                      # General utilities
│   ├── __init__.py
│   ├── inspect_npz.py
│   └── ...                     # Future: file_utils.py, etc.
│
├── label_generation/           # Training label generators
│   ├── __init__.py
│   ├── incrowdvi_superpoint_labels.py
│   └── incrowdvi_spscore_labels.py    # Coming soon
│
└── dataset_processing/         # Future: Dataset preprocessing
    ├── __init__.py
    └── ...
```

This organization makes it clear where new features belong and keeps related functionality together.

---

## Benefits

✓ **Clearer intent** - Directory names show purpose  
✓ **Easier navigation** - Related code is grouped  
✓ **Scalable** - Easy to add new categories  
✓ **Maintainable** - Logical organization  
✓ **Backward compatible** - Top-level imports still work

---

## Questions?

The main package API (`from bb_utils import ...`) remains stable. Only direct submodule imports changed paths.
