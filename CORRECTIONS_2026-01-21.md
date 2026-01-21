# Directory Structure and Timestamp Corrections

**Date:** January 21, 2026  
**Author:** GitHub Copilot  
**Reason:** User provided corrections to implementation assumptions

---

## Issues Corrected

### 1. **Directory Structure - Frames**
**Incorrect assumption:**
- Frames were nested in scene subdirectories: `frames_root/scene_name/sequence_L_timestamp.png`

**Corrected to:**
- Frames are directly in frames_root: `frames_root/sequence_L_timestamp_us.png`

**Files modified:**
- `src/bb_utils/incrowdvi_superpoint_labels.py`:
  - Updated `discover_frames()` to search in `frames_root` directly, not `frames_root/scene_name`
  - Removed `scene_name` parameter from `discover_frames()`
  - Updated call to `discover_frames()` in `generate_labels()`
- `configs/incrowdvi_superpoint_labels.yaml`: Updated comments
- `docs/QUICKSTART.md`: Updated directory structure examples
- `docs/IMPLEMENTATION_SUMMARY.md`: Updated data flow diagram
- `examples/incrowdvi_superpoint_labels_example.py`: Updated example paths

---

### 2. **Directory Structure - Output Labels**
**Incorrect assumption:**
- Labels were nested in scene subdirectories: `output_labels_root/scene_name/frame_name.npz`

**Corrected to:**
- Labels are directly in output_labels_root: `output_labels_root/frame_name.npz`

**Files modified:**
- `src/bb_utils/incrowdvi_superpoint_labels.py`:
  - Updated `process_sequence()` to save to `output_root` directly
  - Removed creation of `output_scene_dir = output_root / scene_name`
  - Changed output path from `output_scene_dir / f"{frame_path.stem}.npz"` to `output_root / f"{frame_path.stem}.npz"`
- `configs/incrowdvi_superpoint_labels.yaml`: Updated comments
- `docs/QUICKSTART.md`: Updated directory structure examples
- `docs/IMPLEMENTATION_SUMMARY.md`: Updated data flow diagram
- `examples/incrowdvi_superpoint_labels_example.py`: Updated example paths

---

### 3. **Timestamp Units**
**Incorrect assumption:**
- CSV timestamps were in microseconds
- Frame filename timestamps were in nanoseconds
- Required conversion: multiply by 1000

**Corrected to:**
- Both CSV and frame filename timestamps are in **microseconds**
- No conversion needed

**Files modified:**
- `src/bb_utils/incrowdvi_superpoint_labels.py`:
  - Removed `timestamp_us_to_ns()` function entirely
  - Updated `build_frame_id()` to return `f"{cam_indicator}_{timestamp_us}"` directly
  - Updated docstring showing format as `[L/R]_[timestamp_us]` instead of `[L/R]_[timestamp_ns]`
  - Updated ASSUMPTIONS section in module docstring
  - Updated frame filename format references throughout
- `configs/incrowdvi_superpoint_labels.yaml`:
  - Removed `timestamp_unit_csv` and `timestamp_unit_frame` fields
  - Updated comments showing `timestamp_us` instead of `timestamp_ns`
- `docs/QUICKSTART.md`: Updated troubleshooting section
- Module docstring: Updated CONFIG FILE section to remove timestamp conversion comments

---

## Summary of Changes

### Code Changes
1. **Removed function:** `timestamp_us_to_ns()`
2. **Modified function:** `discover_frames()` - removed `scene_name` parameter, searches root directly
3. **Modified function:** `build_frame_id()` - no longer calls conversion, uses microseconds directly
4. **Modified function:** `process_sequence()` - saves to output_root directly, not nested
5. **Modified function:** `generate_labels()` - updated call to `discover_frames()`

### Documentation Changes
1. **Module docstring:** Updated ASSUMPTIONS, directory structure examples, filename formats
2. **Config file:** Removed timestamp unit fields, updated directory structure comments
3. **QUICKSTART.md:** Updated directory structure, troubleshooting tips
4. **IMPLEMENTATION_SUMMARY.md:** Updated data flow diagram, output structure
5. **Example script:** Updated example label file path

---

## Verification

All changes have been applied and verified:
- ✅ No syntax errors in Python files
- ✅ Directory structure logic corrected
- ✅ Timestamp conversion removed
- ✅ Documentation updated consistently
- ✅ All references to old structure updated

---

## Migration Notes

**If you have already generated labels with the old version:**
- Old labels were saved in nested structure: `output_root/scene_name/*.npz`
- New labels will be saved flat: `output_root/*.npz`
- You may need to reorganize existing labels or regenerate them

**Frame discovery:**
- Old version looked in: `frames_root/scene_name/*.png`
- New version looks in: `frames_root/*.png`
- Ensure your frames are in the correct location

**Timestamp matching:**
- Both versions use the same timestamp values (microseconds)
- Only the internal logic changed (removal of conversion)
- Existing labels should still be valid
