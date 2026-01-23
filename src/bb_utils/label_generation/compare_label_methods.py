#!/usr/bin/env python3
"""
Compare InCrowd-VI labels with traditional SuperPoint MagicPoint exports.

This helps validate that SLAM-based labels have similar characteristics
to labels generated via homography adaptation.

Usage:
    python compare_label_methods.py \
        --slam_labels /path/to/slam/labels \
        --magicpoint_labels /path/to/magicpoint/labels \
        --output comparison_report.txt
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm


def load_labels(labels_dir: Path) -> Dict[str, np.ndarray]:
    """Load all labels from directory."""
    labels = {}
    for npz_file in labels_dir.glob('*.npz'):
        data = np.load(npz_file)
        labels[npz_file.stem] = data['pts']
    return labels


def compute_statistics(labels: Dict[str, np.ndarray]) -> dict:
    """Compute aggregate statistics across all labels."""
    all_counts = []
    all_confidences = []
    all_x_spreads = []
    all_y_spreads = []
    
    for pts in labels.values():
        if len(pts) == 0:
            continue
        
        all_counts.append(len(pts))
        all_confidences.extend(pts[:, 2])
        all_x_spreads.append(pts[:, 0].std())
        all_y_spreads.append(pts[:, 1].std())
    
    return {
        'num_files': len(labels),
        'count_mean': np.mean(all_counts) if all_counts else 0,
        'count_std': np.std(all_counts) if all_counts else 0,
        'count_min': np.min(all_counts) if all_counts else 0,
        'count_max': np.max(all_counts) if all_counts else 0,
        'conf_mean': np.mean(all_confidences) if all_confidences else 0,
        'conf_std': np.std(all_confidences) if all_confidences else 0,
        'conf_min': np.min(all_confidences) if all_confidences else 0,
        'conf_max': np.max(all_confidences) if all_confidences else 0,
        'x_spread_mean': np.mean(all_x_spreads) if all_x_spreads else 0,
        'y_spread_mean': np.mean(all_y_spreads) if all_y_spreads else 0,
    }


def compare_label_sets(slam_labels: Dict[str, np.ndarray],
                       mp_labels: Dict[str, np.ndarray],
                       common_only: bool = True) -> dict:
    """Compare two sets of labels."""
    # Find common files
    slam_keys = set(slam_labels.keys())
    mp_keys = set(mp_labels.keys())
    common_keys = slam_keys & mp_keys
    
    logging.info(f"SLAM labels: {len(slam_keys)} files")
    logging.info(f"MagicPoint labels: {len(mp_keys)} files")
    logging.info(f"Common files: {len(common_keys)}")
    
    if common_only and len(common_keys) == 0:
        logging.warning("No common files found!")
        return {}
    
    # Compare on common files
    count_diffs = []
    overlap_ratios = []
    
    for key in tqdm(common_keys, desc="Comparing labels"):
        slam_pts = slam_labels[key]
        mp_pts = mp_labels[key]
        
        # Count difference
        count_diffs.append(abs(len(slam_pts) - len(mp_pts)))
        
        # Spatial overlap (rough estimate)
        # Count how many SLAM points are close to MagicPoint points
        if len(slam_pts) > 0 and len(mp_pts) > 0:
            slam_coords = slam_pts[:, :2]
            mp_coords = mp_pts[:, :2]
            
            # For each SLAM point, find nearest MP point
            matches = 0
            for slam_pt in slam_coords:
                dists = np.sqrt(((mp_coords - slam_pt)**2).sum(axis=1))
                if dists.min() < 5:  # within 5 pixels
                    matches += 1
            
            overlap_ratios.append(matches / len(slam_pts))
    
    return {
        'common_files': len(common_keys),
        'count_diff_mean': np.mean(count_diffs) if count_diffs else 0,
        'count_diff_std': np.std(count_diffs) if count_diffs else 0,
        'overlap_mean': np.mean(overlap_ratios) if overlap_ratios else 0,
        'overlap_std': np.std(overlap_ratios) if overlap_ratios else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare SLAM and MagicPoint labels")
    parser.add_argument('--slam_labels', type=Path, required=True)
    parser.add_argument('--magicpoint_labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('comparison_report.txt'))
    parser.add_argument('-v', '--verbose', action='store_true')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='[%(levelname)s] %(message)s'
    )
    
    # Load labels
    logging.info("Loading SLAM labels...")
    slam_labels = load_labels(args.slam_labels)
    
    logging.info("Loading MagicPoint labels...")
    mp_labels = load_labels(args.magicpoint_labels)
    
    # Compute statistics
    slam_stats = compute_statistics(slam_labels)
    mp_stats = compute_statistics(mp_labels)
    
    # Compare
    comparison = compare_label_sets(slam_labels, mp_labels)
    
    # Write report
    with open(args.output, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Label Method Comparison: SLAM vs MagicPoint\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("SLAM Labels:\n")
        for k, v in slam_stats.items():
            f.write(f"  {k}: {v:.4f}\n")
        
        f.write("\nMagicPoint Labels:\n")
        for k, v in mp_stats.items():
            f.write(f"  {k}: {v:.4f}\n")
        
        f.write("\nComparison:\n")
        for k, v in comparison.items():
            f.write(f"  {k}: {v:.4f}\n")
    
    logging.info(f"Report saved to: {args.output}")


if __name__ == '__main__':
    main()
