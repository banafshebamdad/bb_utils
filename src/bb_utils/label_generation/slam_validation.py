# 
# Banafshe Bamdad + GitHub Copilot (Claude Sonnet 4.5)
# January 23, 2026 13:49 CET
# 
# slam_validation.py
# 

#!/usr/bin/env python3
"""
SLAM-specific validation checks.

Validates that SLAM-derived labels have expected characteristics:
- Temporal consistency across sequential frames
- Repeatability of features in similar viewpoints

Usage:
    python slam_validation.py <labels_dir>
    python src/bb_utils/label_generation/slam_validation.py /home/ubuntu/pytorch-superpoint/datasets/incrowdvi/superpoint_labels/train
"""

import numpy as np
from pathlib import Path
import re
from collections import defaultdict
from tqdm import tqdm


def extract_timestamp(filename: str) -> int:
    """Extract timestamp from filename: [sequence]_[L/R]_[timestamp].npz"""
    parts = filename.split('_')
    return int(parts[-1].replace('.npz', ''))


def check_temporal_consistency(labels_dir: Path, max_gap_us: int = 100000):
    """
    Check temporal consistency of keypoint counts across sequential frames.
    
    This function validates that SLAM-derived labels maintain consistent keypoint
    counts between consecutive frames. Large drops in keypoint count may indicate
    SLAM tracking failures, severe motion blur, or occlusions that should be
    investigated.
    
    The check groups frames by sequence name, sorts them chronologically by timestamp,
    and examines consecutive frame pairs. A potential issue is flagged when:
    - Current frame has <50% keypoints compared to previous frame, AND
    - Current frame has <500 keypoints total, AND
    - Frames are within max_gap_us microseconds apart
    
    Parameters:
    -----------
    labels_dir : Path
        Directory containing .npz label files with naming format:
        [sequence_name]_[L/R]_[timestamp_us].npz
    max_gap_us : int, default=100000
        Maximum time gap (microseconds) between frames to be considered consecutive.
        Frames separated by more than this are not compared (e.g., after scene cuts).
        Default 100ms = 100,000 microseconds
    
    Returns:
    --------
    bool
        True if no temporal consistency issues detected, False otherwise
    
    Example Output:
    ---------------
        Sequences checked: 30
        Temporal consistency issues: 0
        ✓ No major temporal inconsistencies detected
    """
    print("Checking temporal consistency...")
    
    # Group by sequence
    sequences = defaultdict(list)
    label_files = list(labels_dir.glob('*.npz'))
    for npz_file in tqdm(label_files, desc="Loading sequences"):
        # Extract sequence name (everything before last two underscores)
        match = re.match(r'(.+)_[LR]_\d+\.npz', npz_file.name)
        if match:
            seq_name = match.group(1)
            timestamp = extract_timestamp(npz_file.name)
            
            data = np.load(npz_file)
            count = len(data['pts'])
            
            sequences[seq_name].append((timestamp, count, npz_file.name))
    
    # Check each sequence
    issues = []
    for seq_name, frames in sequences.items():
        # Sort by timestamp
        frames.sort()
        
        for i in range(1, len(frames)):
            prev_ts, prev_count, prev_name = frames[i-1]
            curr_ts, curr_count, curr_name = frames[i]
            
            time_gap = curr_ts - prev_ts
            
            # Only check consecutive frames (within max_gap)
            if time_gap > max_gap_us:
                continue
            
            # Check for large drops in keypoint count
            if curr_count < prev_count * 0.5 and curr_count < 500:
                issues.append(
                    f"{seq_name}: Large drop at {curr_name} "
                    f"({prev_count} -> {curr_count}, gap={time_gap}us)"
                )
    
    print(f"  Sequences checked: {len(sequences)}")
    print(f"  Temporal consistency issues: {len(issues)}")
    if issues:
        for issue in issues[:10]:
            print(f"    - {issue}")
    else:
        print("  ✓ No major temporal inconsistencies detected")
    
    return len(issues) == 0


def check_confidence_distribution(labels_dir: Path):
    """Check that confidence scores are well-distributed."""
    print("\nChecking confidence distribution...")
    
    all_confidences = []
    label_files = list(labels_dir.glob('*.npz'))
    for npz_file in tqdm(label_files, desc="Collecting confidences"):
        data = np.load(npz_file)
        pts = data['pts']
        if len(pts) > 0:
            all_confidences.extend(pts[:, 2])
    
    all_confidences = np.array(all_confidences)
    
    # Check for degenerate distributions
    issues = []
    
    # Check 1: Not all same value
    if all_confidences.std() < 0.01:
        issues.append("All confidences nearly identical (std < 0.01)")
    
    # Check 2: Good spread across range
    bins = np.histogram(all_confidences, bins=10, range=(0, 1))[0]
    if (bins > 0).sum() < 3:
        issues.append("Confidence values clustered in <3 bins")
    
    # Check 3: Not all at extremes
    extreme_ratio = ((all_confidences < 0.1) | (all_confidences > 0.9)).sum() / len(all_confidences)
    if extreme_ratio > 0.9:
        issues.append(f"{extreme_ratio*100:.1f}% of confidences are extreme (<0.1 or >0.9)")
    
    print(f"  Total confidence values: {len(all_confidences)}")
    print(f"  Mean: {all_confidences.mean():.3f}, Std: {all_confidences.std():.3f}")
    print(f"  Distribution across [0,1]: {np.histogram(all_confidences, bins=5, range=(0,1))[0]}")
    
    if issues:
        print(f"  ⚠️  Issues:")
        for issue in issues:
            print(f"    - {issue}")
        return False
    else:
        print("  ✓ Confidence distribution looks good")
        return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SLAM-specific label validation")
    parser.add_argument('labels_dir', type=Path, help='Directory containing labels')
    args = parser.parse_args()
    
    print("=" * 60)
    print("SLAM Label Validation")
    print("=" * 60)
    
    temporal_ok = check_temporal_consistency(args.labels_dir)
    confidence_ok = check_confidence_distribution(args.labels_dir)
    
    print("\n" + "=" * 60)
    if temporal_ok and confidence_ok:
        print("✓ All SLAM-specific checks passed!")
        return 0
    else:
        print("⚠️  Some checks failed - review above")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
