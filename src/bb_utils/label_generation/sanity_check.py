# 
# Banafshe Bamdad + GitHub Copilot (Claude Sonnet 4.5)
# January 23, 2026 13:10 CET
# 
# sanity_check.py
# 

#!/usr/bin/env python3
"""
Quick Sanity Checks for InCrowd-VI SuperPoint Labels

This script performs rapid validation of generated label files (.npz format) to ensure
basic correctness before using them for training. It checks structural integrity and
catches common issues that would cause training failures.

Checks Performed:
-----------------
1. File not empty - Ensures each .npz file contains at least one keypoint
2. Correct shape - Verifies pts array has shape (N, 3) with columns [x, y, confidence]
3. Coordinates in bounds - Checks x ∈ [0, 640] and y ∈ [0, 480] (after resize)
4. Confidence range - Ensures all confidence values are in [0, 1]
5. Spatial variation - Verifies keypoints aren't all at the same location
6. No invalid values - Checks for NaN or Inf values

Usage:
------
    python sanity_check.py <labels_dir>

Example:
    python sanity_check.py /home/ubuntu/pytorch-superpoint/datasets/incrowdvi/superpoint_labels/train

Output:
-------
    - Progress bar showing validation progress
    - Summary of total files checked and issues found
    - List of problematic files (if any) with specific error types
    - Exit code 0 if all checks pass, 1 if issues detected

Performance:
------------
    Processes ~1000-2000 files/second (depends on disk I/O)
    Expected runtime for 250K files: 2-5 minutes

"""

import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

def sanity_check_labels(labels_dir: Path):
    """Run quick sanity checks."""
    issues = []
    
    label_files = list(labels_dir.glob('*.npz'))
    print(f"Checking {len(label_files)} label files...")
    
    for npz_file in tqdm(label_files, desc="Validating labels"):
        data = np.load(npz_file)
        pts = data['pts']
        
        # Check 1: File not empty
        if len(pts) == 0:
            issues.append(f"EMPTY: {npz_file.name}")
            continue
        
        # Check 2: Shape is correct
        if pts.shape[1] != 3:
            issues.append(f"WRONG_SHAPE: {npz_file.name} has shape {pts.shape}")
        
        x, y, conf = pts[:, 0], pts[:, 1], pts[:, 2]
        
        # Check 3: Coordinates in reasonable range (assuming 640x480 resize)
        if x.min() < 0 or x.max() > 650 or y.min() < 0 or y.max() > 490:
            issues.append(f"OUT_OF_BOUNDS: {npz_file.name} - x:[{x.min():.1f}, {x.max():.1f}], y:[{y.min():.1f}, {y.max():.1f}]")
        
        # Check 4: Confidence in [0, 1]
        if conf.min() < 0 or conf.max() > 1:
            issues.append(f"BAD_CONFIDENCE: {npz_file.name} - conf:[{conf.min():.4f}, {conf.max():.4f}]")
        
        # Check 5: Not all keypoints at same location
        if x.std() < 1 or y.std() < 1:
            issues.append(f"NO_SPATIAL_VARIATION: {npz_file.name}")
        
        # Check 6: Check for NaN/Inf
        if np.isnan(pts).any() or np.isinf(pts).any():
            issues.append(f"NAN_OR_INF: {npz_file.name}")
    
    # Report
    print(f"\n{'='*60}")
    print(f"Sanity Check Results")
    print(f"{'='*60}")
    print(f"Total files checked: {len(label_files)}")
    print(f"Issues found: {len(issues)}")
    
    if issues:
        print(f"\n⚠️  Issues detected:\n")
        for issue in issues[:20]:  # Show first 20
            print(f"  - {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues)-20} more")
        return False
    else:
        print("\n✓ All checks passed!")
        return True

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python sanity_check.py <labels_dir>")
        sys.exit(1)
    
    labels_dir = Path(sys.argv[1])
    success = sanity_check_labels(labels_dir)
    sys.exit(0 if success else 1)
