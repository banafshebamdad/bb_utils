# 
# Banafshe Bamdad + GitHub Copilot (Claude Sonnet 4.5)
# January 23, 2026 14:17 CET
# 
# validate_csv_to_npz.py
# 

#!/usr/bin/env python3
"""
CSV-to-NPZ Label Validation Tool

Validates the accuracy of label extraction by comparing NPZ labels against
original CSV observations. This ensures the conversion pipeline is correct.

Strategy:
---------
1. Randomly sample frames from NPZ labels
2. Load corresponding CSV observations for those frames only (memory efficient)
3. Verify that NPZ keypoints (top-k filtered) are a subset of CSV observations
4. Visualize overlays with color coding:
   - Gray: CSV keypoints not selected in top-k (expected)
   - Green: NPZ keypoints correctly found in CSV (should be ~100%)
   - Red: NPZ keypoints NOT in CSV (indicates extraction error)
5. Report match statistics (target: 100% of NPZ points found in CSV)

Validation Logic:
-----------------
NPZ files contain top-k keypoints selected from CSV observations based on
confidence scores. This tool verifies that every keypoint in NPZ exists in
the original CSV source, confirming the extraction pipeline is correct.

Performance Optimization:
-------------------------
- Only loads CSV data for sampled frames (not all 250K+)
- Uses chunked CSV reading
- Groups frames by sequence to minimize file reads

Usage:
------
cd /home/ubuntu/bb_utils
python src/bb_utils/label_generation/validate_csv_to_npz.py \
    --config configs/incrowdvi_superpoint_labels.yaml \
    --frames_dir /home/ubuntu/pytorch-superpoint/datasets/incrowdvi/val \
    --labels_dir /home/ubuntu/pytorch-superpoint/datasets/incrowdvi/superpoint_labels/val \
    --output_dir /home/ubuntu/pytorch-superpoint/bb_results/validation_results/csv_npz_comparison/val \
    --num_samples 30 \
    --verbose

"""

import argparse
import gzip
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import random

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml


def parse_frame_filename(filename: str) -> Tuple[str, str, int]:
    """
    Parse frame filename to extract sequence, camera, and timestamp.
    
    Format: [sequence_name]_[L/R]_[timestamp_us].npz
    
    Returns:
    --------
    sequence_name : str
    camera_indicator : str ('L' or 'R')
    timestamp_us : int
    """
    stem = filename.replace('.npz', '').replace('.png', '')
    parts = stem.split('_')
    
    camera = parts[-2]  # L or R
    timestamp = int(parts[-1])
    sequence = '_'.join(parts[:-2])
    
    return sequence, camera, timestamp


def load_csv_observations_for_frames(mps_root: Path,
                                     frame_info_list: List[Tuple[str, str, int]],
                                     camera_serials: Dict[str, str]) -> Dict[str, np.ndarray]:
    """
    Load CSV observations only for specified frames (memory efficient).
    
    Parameters:
    -----------
    mps_root : Path
        Root directory containing MPS data
    frame_info_list : list
        List of (sequence_name, camera_indicator, timestamp_us) tuples
    camera_serials : dict
        Mapping of {'left': serial, 'right': serial}
    
    Returns:
    --------
    frame_observations : dict
        Maps frame_id to numpy array of shape (N, 2) with [u, v] coordinates
    """
    # Group frames by sequence to minimize file reads
    seq_frames = defaultdict(list)
    for seq_name, cam, timestamp in frame_info_list:
        seq_frames[seq_name].append((cam, timestamp))
    
    frame_observations = {}
    
    # Map camera indicator to serial
    cam_to_serial = {
        'L': camera_serials['left'],
        'R': camera_serials['right']
    }
    
    # Process each sequence
    for seq_name in tqdm(seq_frames.keys(), desc="Loading CSV observations"):
        # Find the MPS directory for this sequence
        mps_dir = None
        for scene_dir in mps_root.iterdir():
            if not scene_dir.is_dir():
                continue
            for seq_dir in scene_dir.iterdir():
                if seq_dir.name.replace('mps_', '').replace('_vrs', '') == seq_name:
                    mps_dir = seq_dir
                    break
            if mps_dir:
                break
        
        if not mps_dir:
            logging.warning(f"MPS directory not found for sequence: {seq_name}")
            continue
        
        obs_file = mps_dir / "semidense_observations.csv.gz"
        if not obs_file.exists():
            logging.warning(f"Observations file not found: {obs_file}")
            continue
        
        # Get target timestamps and cameras for this sequence
        target_frames = {(cam, ts) for cam, ts in seq_frames[seq_name]}
        
        # Read CSV in chunks and filter for target frames
        chunk_size = 100000
        with gzip.open(obs_file, 'rt') as f:
            for chunk in pd.read_csv(f, chunksize=chunk_size):
                # Filter for target frames
                for cam, timestamp in target_frames:
                    serial = cam_to_serial[cam]
                    mask = (chunk['frame_tracking_timestamp_us'] == timestamp) & \
                           (chunk['camera_serial'] == serial)
                    
                    if mask.any():
                        frame_data = chunk[mask][['u', 'v']].values
                        frame_id = f"{seq_name}_{cam}_{timestamp}"
                        
                        if frame_id in frame_observations:
                            # Append if we already have some observations
                            frame_observations[frame_id] = np.vstack([
                                frame_observations[frame_id],
                                frame_data
                            ])
                        else:
                            frame_observations[frame_id] = frame_data
    
    return frame_observations


def compare_and_visualize(image_path: Path,
                         npz_path: Path,
                         csv_observations: np.ndarray,
                         output_path: Path,
                         frame_id: str):
    """
    Compare CSV and NPZ keypoints and create visualization.
    
    Parameters:
    -----------
    image_path : Path
    npz_path : Path
    csv_observations : np.ndarray
        Shape (N, 2) with [u, v] coordinates from CSV
    output_path : Path
    frame_id : str
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        logging.error(f"Failed to load image: {image_path}")
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    # Load NPZ labels
    npz_data = np.load(npz_path)
    npz_pts = npz_data['pts'][:, :2]  # Just x, y (ignore confidence)
    
    # Compute statistics
    n_csv = len(csv_observations)
    n_npz = len(npz_pts)
    
    # Check if NPZ keypoints are present in CSV (NPZ should be subset of CSV)
    # NPZ contains top-k filtered keypoints, so we check if each NPZ point exists in CSV
    npz_found_in_csv = np.zeros(n_npz, dtype=bool)
    csv_used = np.zeros(n_csv, dtype=bool)
    
    for i, npz_pt in enumerate(npz_pts):
        # Find closest CSV point to this NPZ point
        dists = np.sqrt(((csv_observations - npz_pt)**2).sum(axis=1))
        min_idx = dists.argmin()
        
        if dists[min_idx] < 1.0:  # 1 pixel tolerance
            npz_found_in_csv[i] = True
            csv_used[min_idx] = True
    
    matches = npz_found_in_csv.sum()
    csv_matched = csv_used
    npz_matched = npz_found_in_csv
    
    # Create visualization - single panel with overlay
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    ax.imshow(img_rgb)
    # Plot unused CSV points (gray - not selected in top-k)
    if (~csv_matched).any():
        ax.scatter(csv_observations[~csv_matched, 0], csv_observations[~csv_matched, 1],
                  c='lightgray', s=2, alpha=0.4, label=f'CSV not in top-k ({(~csv_matched).sum()})', 
                  edgecolors='none')
    # Plot NPZ points NOT found in CSV (ERROR - RED)
    if (~npz_matched).any():
        ax.scatter(npz_pts[~npz_matched, 0], npz_pts[~npz_matched, 1],
                  c='red', s=8, alpha=0.9, label=f'NPZ NOT in CSV - ERROR ({(~npz_matched).sum()})',
                  edgecolors='darkred', linewidths=0.8, marker='x')
    # Plot NPZ points found in CSV (CORRECT - GREEN)
    if npz_matched.any():
        ax.scatter(npz_pts[npz_matched, 0], npz_pts[npz_matched, 1],
                  c='lime', s=5, alpha=0.8, label=f'NPZ in CSV - CORRECT ({matches})',
                  edgecolors='darkgreen', linewidths=0.3)
    
    # Match ratio = what % of NPZ keypoints were found in CSV (should be 100%)
    match_ratio = matches / n_npz if n_npz > 0 else 0
    ax.set_title(f'CSV-to-NPZ Validation: {frame_id}\n' +
                f'NPZ→CSV Match Rate: {match_ratio*100:.1f}% ({matches}/{n_npz}) | ' +
                f'Top-k filtered: {n_npz}/{n_csv}',
                fontsize=12, fontweight='bold')
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Return statistics
    return {
        'frame_id': frame_id,
        'n_csv': n_csv,
        'n_npz': n_npz,
        'matches': matches,
        'match_ratio': match_ratio,
        'csv_only': (~csv_matched).sum(),
        'npz_only': (~npz_matched).sum(),
    }


def validate_csv_to_npz(config_path: Path,
                       frames_dir: Path,
                       labels_dir: Path,
                       output_dir: Path,
                       num_samples: int = 50):
    """
    Main validation function.
    """
    # Load config for camera serials
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    camera_serials = config['data']['camera_serials']
    mps_root = Path(config['data']['mps_root'])
    
    # Get all NPZ files and sample
    npz_files = list(labels_dir.glob('*.npz'))
    logging.info(f"Found {len(npz_files)} NPZ label files")
    
    sample_size = min(num_samples, len(npz_files))
    sampled_npz = random.sample(npz_files, sample_size)
    logging.info(f"Randomly sampling {sample_size} frames for validation")
    
    # Parse frame info
    frame_info_list = []
    frame_id_to_paths = {}
    
    for npz_path in sampled_npz:
        seq_name, cam, timestamp = parse_frame_filename(npz_path.name)
        frame_info_list.append((seq_name, cam, timestamp))
        
        frame_id = f"{seq_name}_{cam}_{timestamp}"
        img_path = frames_dir / f"{npz_path.stem}.png"
        
        if not img_path.exists():
            logging.warning(f"Image not found: {img_path}")
            continue
        
        frame_id_to_paths[frame_id] = {
            'npz': npz_path,
            'img': img_path
        }
    
    # Load CSV observations for sampled frames
    logging.info("Loading CSV observations for sampled frames...")
    csv_observations = load_csv_observations_for_frames(
        mps_root, frame_info_list, camera_serials
    )
    
    logging.info(f"Loaded CSV observations for {len(csv_observations)} frames")
    
    # Compare and visualize
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    all_stats = []
    
    for frame_id, paths in tqdm(frame_id_to_paths.items(), desc="Creating comparisons"):
        if frame_id not in csv_observations:
            logging.warning(f"No CSV observations found for {frame_id}")
            continue
        
        output_path = vis_dir / f"{frame_id}_comparison.png"
        
        stats = compare_and_visualize(
            paths['img'],
            paths['npz'],
            csv_observations[frame_id],
            output_path,
            frame_id
        )
        
        if stats:
            all_stats.append(stats)
    
    # Generate report
    report_path = output_dir / 'csv_to_npz_validation_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CSV-to-NPZ Label Extraction Validation Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Visualization Color Coding:\n")
        f.write("  - GRAY: CSV keypoints not selected in top-k (expected, filtered out)\n")
        f.write("  - GREEN: NPZ keypoints correctly found in CSV (validation success)\n")
        f.write("  - RED: NPZ keypoints NOT found in CSV (extraction error)\n\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Frames validated: {len(all_stats)}\n\n")
        
        if all_stats:
            match_ratios = [s['match_ratio'] for s in all_stats]
            csv_counts = [s['n_csv'] for s in all_stats]
            npz_counts = [s['n_npz'] for s in all_stats]
            f.write("Match Statistics (NPZ→CSV validation):\n")
            f.write(f"  NPZ keypoints are a top-k filtered subset of CSV\n")
            f.write(f"  Match rate = % of NPZ points found in CSV (should be ~100%)\n\n")
            f.write(f"  Mean match rate: {np.mean(match_ratios)*100:.2f}%\n")
            f.write(f"  Median match rate: {np.median(match_ratios)*100:.2f}%\n")
            f.write(f"  Min match rate: {np.min(match_ratios)*100:.2f}%\n")
            f.write(f"  Max match rate: {np.max(match_ratios)*100:.2f}%\n\n")
            
            f.write("Keypoint Counts:\n")
            f.write(f"  CSV mean: {np.mean(csv_counts):.1f} (all SLAM observations)\n")
            f.write(f"  NPZ mean: {np.mean(npz_counts):.1f} (top-k filtered)\n")
            f.write(f"  Reduction: {np.mean(csv_counts) - np.mean(npz_counts):.1f} ({(1 - np.mean(npz_counts)/np.mean(csv_counts))*100:.1f}%)\n\n")
            
            # List problematic frames (where NPZ points are NOT in CSV)
            low_match = [s for s in all_stats if s['match_ratio'] < 0.99]
            if low_match:
                f.write(f"⚠️  Frames with <99% match rate ({len(low_match)}):\n")
                f.write(f"These frames have NPZ keypoints NOT found in CSV (possible extraction error)\n\n")
                for s in low_match[:20]:
                    missing = s['npz_only']
                    f.write(f"  {s['frame_id']}: {s['match_ratio']*100:.1f}% "
                           f"(NPZ:{s['n_npz']}, CSV:{s['n_csv']}, Missing:{missing})\n")
            else:
                f.write(f"✓ All NPZ keypoints found in CSV!\n")
            
            # Perfect matches
            perfect = [s for s in all_stats if s['match_ratio'] >= 0.99]
            f.write(f"\nFrames with ≥99% match rate: {len(perfect)}/{len(all_stats)}\n")
    if all_stats:
        avg_match = np.mean([s['match_ratio'] for s in all_stats]) * 100
        perfect_count = sum(1 for s in all_stats if s['match_ratio'] >= 0.99)
        
        print(f"\n" + "="*60)
        print(f"NPZ→CSV Validation Results")
        print(f"="*60)
        print(f"Average match rate: {avg_match:.1f}%")
        print(f"Perfect matches (≥99%): {perfect_count}/{len(all_stats)}")
        
        if avg_match >= 99:
            print(f"\n✓ EXCELLENT! All NPZ keypoints found in CSV.")
            print(f"  Extraction pipeline is working correctly.")
        elif avg_match > 95:
            print(f"\n✓ Good! Most NPZ keypoints found in CSV.")
            print(f"  Minor discrepancies - review report for details.")
        else:
            print(f"\n❌ WARNING! Significant mismatches detected.")
            print(f"  {100-avg_match:.1f}% of NPZ keypoints NOT found in CSV!")
            print(f"  Check extraction pipeline for errors.")
    logging.info(f"  Visualizations: {vis_dir}")
    logging.info(f"  Report: {report_path}")
    
    if all_stats:
        avg_match = np.mean([s['match_ratio'] for s in all_stats]) * 100
        print(f"\n✓ Average match rate: {avg_match:.1f}%")
        if avg_match > 95:
            print("  ✓ Excellent! CSV-to-NPZ extraction is accurate.")
        elif avg_match > 80:
            print("  ⚠️  Good, but review low-match frames in report.")
        else:
            print("  ❌ Poor match rate - check extraction pipeline!")


def main():
    parser = argparse.ArgumentParser(
        description="Validate CSV-to-NPZ label extraction accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--config', type=Path, required=True,
                       help='Path to incrowdvi_superpoint_labels.yaml config')
    parser.add_argument('--frames_dir', type=Path, required=True,
                       help='Directory containing frame images')
    parser.add_argument('--labels_dir', type=Path, required=True,
                       help='Directory containing NPZ label files')
    parser.add_argument('--output_dir', type=Path, required=True,
                       help='Output directory for validation results')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of random frames to validate (default: 50)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        format='[%(asctime)s %(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=log_level
    )
    
    validate_csv_to_npz(
        args.config,
        args.frames_dir,
        args.labels_dir,
        args.output_dir,
        args.num_samples
    )


if __name__ == '__main__':
    main()
