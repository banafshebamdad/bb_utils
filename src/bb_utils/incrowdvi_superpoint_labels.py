#!/usr/bin/env python3
"""
InCrowd-VI SuperPoint Label Generator

This script generates **SuperPoint-compatible** training labels from the InCrowd-VI dataset's
semi-dense SLAM observations. It processes multi-view visual-inertial SLAM data and outputs
per-frame keypoint labels in the format expected by the SuperPoint training pipeline.

**IMPORTANT**: This script is specifically designed for SuperPoint training labels.
              For SP-SCore label generation, use incrowdvi_spscore_labels.py (to be developed).

================================================================================
PURPOSE
================================================================================
Convert InCrowd-VI semi-dense 3D point observations into 2D keypoint labels suitable
for training SuperPoint detectors. Each output file contains keypoint pixel coordinates
and confidence scores derived from the SLAM system's uncertainty estimates.

================================================================================
INPUT DATA STRUCTURE
================================================================================
The InCrowd-VI dataset provides:

1. semidense_observations.csv.gz
   - Columns: uid, frame_tracking_timestamp_us, camera_serial, u, v
   - Contains 2D pixel observations (u, v) of 3D points
   - Timestamps in microseconds
   - Camera serial identifies left/right camera

2. semidense_points.csv.gz
   - Columns: uid, graph_uid, px_world, py_world, pz_world, inv_dist_std, dist_std
   - Contains 3D point positions and uncertainty estimates
   - uid links to observations
   - inv_dist_std is inverse distance standard deviation (uncertainty metric)

Dataset structure:
    [mps_root]/[scene_name]/mps_[sequence_name]_vrs/
        ├── semidense_observations.csv.gz
        └── semidense_points.csv.gz

Frame images are stored separately:
    [frames_root]/[scene_name]/
        └── [sequence_name]_[L/R]_[timestamp_ns].png

================================================================================
OUTPUT FORMAT
================================================================================
One .npz file per frame containing:
    Key: 'pts'
    Shape: (N, 3)
    Columns: [x, y, confidence]
    
Where:
    - x, y: pixel coordinates (float)
    - confidence: [0, 1] derived from inv_dist_std
    - N: number of keypoints (≤ top_k, filtered by confidence_threshold)

Output location:
    [output_labels_root]/[scene_name]/[frame_filename_without_ext].npz

================================================================================
CONFIG FILE (YAML)
================================================================================
Required fields:

data:
    mps_root: "/path/to/incrowdvi/mps_data"
    frames_root: "/path/to/incrowdvi/frames"
    output_labels_root: "/path/to/output/labels"
    
    camera_serials:
        left: "0072510f1b2107010500001910150001"
        right: "0072510f1b2107010800001127080001"
    
    # Timestamp conversion: frame filenames use nanoseconds, CSV uses microseconds
    timestamp_unit_csv: "microseconds"
    timestamp_unit_frame: "nanoseconds"

model:
    detection_threshold: 0.015  # minimum confidence to keep keypoint
    top_k: 1000                  # maximum keypoints per frame
    
    confidence:
        method: "inverse_normalized"  # confidence computation method
        params:
            # Add method-specific parameters here if needed

output:
    generate_report: true
    report_file: "label_generation_report.txt"

================================================================================
CONFIDENCE COMPUTATION
================================================================================
The confidence score is derived from the SLAM uncertainty estimate (inv_dist_std).
Higher inv_dist_std indicates higher uncertainty, thus lower confidence.

Current implementation uses normalized inverse approach, but the design allows
for easy replacement with alternative formulas.

================================================================================
ASSUMPTIONS
================================================================================
1. Timestamps in CSV files are in microseconds
2. Timestamps in frame filenames are in nanoseconds (multiply CSV by 1000)
3. Camera serial mapping:
   - Right camera: 0072510f1b2107010800001127080001 → 'R' in filename
   - Left camera:  0072510f1b2107010500001910150001 → 'L' in filename
4. Frame filename format: [sequence]_[L/R]_[timestamp_ns].png
5. Both CSV files use 'uid' as the linking key
6. Coordinates (u, v) in observations correspond to (x, y) pixel positions

================================================================================
USAGE
================================================================================
Command line:
    python incrowdvi_label_generator.py config.yaml

As a module:
    from bb_utils.incrowdvi_label_generator import generate_labels
    generate_labels('config.yaml')

================================================================================
PERFORMANCE
================================================================================
- Uses chunked CSV reading to handle large files
- Processes frames in streaming fashion
- Memory-efficient groupby operations
- Progress tracking via tqdm

================================================================================
Author: Banafshe Bamdad + GitHub Copilot (Claude Sonnet 4.5)
Date: January 21, 2026
================================================================================
"""

import argparse
import gzip
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


# ============================================================================
# CONFIDENCE COMPUTATION
# ============================================================================

def compute_confidence_inverse_normalized(inv_dist_std: np.ndarray, 
                                         max_inv_dist_std: float,
                                         **kwargs) -> np.ndarray:
    """
    Compute confidence as normalized inverse of uncertainty.
    
    Higher inv_dist_std (uncertainty) → lower confidence
    
    Formula: confidence = 1 - (inv_dist_std / max_inv_dist_std)
    
    Parameters:
    -----------
    inv_dist_std : np.ndarray
        Inverse distance standard deviation (uncertainty metric)
    max_inv_dist_std : float
        Maximum inv_dist_std value for normalization
    
    Returns:
    --------
    confidence : np.ndarray
        Confidence scores in [0, 1]
    """
    if max_inv_dist_std == 0:
        return np.ones_like(inv_dist_std)
    
    confidence = 1.0 - (inv_dist_std / max_inv_dist_std)
    confidence = np.clip(confidence, 0.0, 1.0)
    return confidence


def compute_confidence_exponential(inv_dist_std: np.ndarray,
                                   scale: float = 1.0,
                                   **kwargs) -> np.ndarray:
    """
    Compute confidence using exponential decay.
    
    Formula: confidence = exp(-scale * inv_dist_std)
    
    Parameters:
    -----------
    inv_dist_std : np.ndarray
        Inverse distance standard deviation
    scale : float
        Decay rate parameter
    
    Returns:
    --------
    confidence : np.ndarray
        Confidence scores in (0, 1]
    """
    confidence = np.exp(-scale * inv_dist_std)
    return confidence


# Confidence computation method registry
CONFIDENCE_METHODS = {
    'inverse_normalized': compute_confidence_inverse_normalized,
    'exponential': compute_confidence_exponential,
}


def compute_confidence(inv_dist_std: np.ndarray, 
                      method: str = 'inverse_normalized',
                      **params) -> np.ndarray:
    """
    Compute confidence scores from uncertainty estimates.
    
    This function serves as a dispatcher to different confidence computation
    strategies, making it easy to swap formulas.
    
    Parameters:
    -----------
    inv_dist_std : np.ndarray
        Inverse distance standard deviation from SLAM
    method : str
        Confidence computation method name
    **params : dict
        Method-specific parameters
    
    Returns:
    --------
    confidence : np.ndarray
        Confidence scores
    """
    if method not in CONFIDENCE_METHODS:
        raise ValueError(f"Unknown confidence method: {method}. "
                        f"Available: {list(CONFIDENCE_METHODS.keys())}")
    
    return CONFIDENCE_METHODS[method](inv_dist_std, **params)


# ============================================================================
# FRAME IDENTIFICATION
# ============================================================================

def get_camera_indicator(camera_serial: str, camera_serials: Dict[str, str]) -> str:
    """
    Map camera serial to L/R indicator.
    
    Parameters:
    -----------
    camera_serial : str
        Camera serial number from CSV
    camera_serials : dict
        Mapping of {'left': serial, 'right': serial}
    
    Returns:
    --------
    indicator : str
        'L' for left, 'R' for right
    """
    if camera_serial == camera_serials['left']:
        return 'L'
    elif camera_serial == camera_serials['right']:
        return 'R'
    else:
        raise ValueError(f"Unknown camera serial: {camera_serial}")


def timestamp_us_to_ns(timestamp_us: int) -> int:
    """Convert microsecond timestamp to nanoseconds."""
    return timestamp_us * 1000


def build_frame_id(timestamp_us: int, camera_serial: str, 
                   camera_serials: Dict[str, str]) -> str:
    """
    Build frame identifier for matching with frame filenames.
    
    Returns a string that can be matched against frame filename patterns.
    Format: [L/R]_[timestamp_ns]
    
    Parameters:
    -----------
    timestamp_us : int
        Timestamp in microseconds from CSV
    camera_serial : str
        Camera serial from CSV
    camera_serials : dict
        Camera serial mappings
    
    Returns:
    --------
    frame_id : str
        Frame identifier (e.g., "L_1234567890000")
    """
    cam_indicator = get_camera_indicator(camera_serial, camera_serials)
    timestamp_ns = timestamp_us_to_ns(timestamp_us)
    return f"{cam_indicator}_{timestamp_ns}"


# ============================================================================
# FRAME DISCOVERY
# ============================================================================

def discover_frames(frames_root: Path, scene_name: str) -> Dict[str, Path]:
    """
    Discover all frame files for a scene and build lookup dictionary.
    
    Parameters:
    -----------
    frames_root : Path
        Root directory containing frame images
    scene_name : str
        Scene/sequence name
    
    Returns:
    --------
    frame_map : dict
        Mapping from frame_id to file path
        frame_id format: "[L/R]_[timestamp_ns]"
    """
    scene_frames_dir = frames_root / scene_name
    
    if not scene_frames_dir.exists():
        logging.warning(f"Frames directory not found: {scene_frames_dir}")
        return {}
    
    frame_map = {}
    for frame_file in scene_frames_dir.glob("*.png"):
        # Parse filename: [sequence]_[L/R]_[timestamp_ns].png
        parts = frame_file.stem.split('_')
        if len(parts) < 3:
            logging.warning(f"Unexpected frame filename format: {frame_file.name}")
            continue
        
        # Extract camera indicator and timestamp
        cam_indicator = parts[-2]  # L or R
        timestamp_ns = parts[-1]
        
        frame_id = f"{cam_indicator}_{timestamp_ns}"
        frame_map[frame_id] = frame_file
    
    logging.info(f"Discovered {len(frame_map)} frames for scene '{scene_name}'")
    return frame_map


# ============================================================================
# LABEL GENERATION
# ============================================================================

def filter_and_sort_keypoints(pts: np.ndarray, 
                              confidence_threshold: float,
                              top_k: int) -> np.ndarray:
    """
    Filter keypoints by confidence and keep top-k.
    
    Parameters:
    -----------
    pts : np.ndarray
        Keypoints array of shape (N, 3) with columns [x, y, confidence]
    confidence_threshold : float
        Minimum confidence to keep
    top_k : int
        Maximum number of keypoints to keep
    
    Returns:
    --------
    filtered_pts : np.ndarray
        Filtered and sorted keypoints
    """
    # Filter by confidence threshold
    mask = pts[:, 2] >= confidence_threshold
    filtered = pts[mask]
    
    if len(filtered) == 0:
        return filtered
    
    # Sort by confidence (descending) and take top-k
    sorted_indices = np.argsort(-filtered[:, 2])  # negative for descending
    filtered = filtered[sorted_indices[:top_k]]
    
    return filtered


def process_sequence(mps_sequence_dir: Path,
                     sequence_name: str,
                     scene_name: str,
                     frame_map: Dict[str, Path],
                     output_root: Path,
                     config: dict) -> Dict[str, int]:
    """
    Process a single MPS sequence and generate labels for all frames.
    
    Parameters:
    -----------
    mps_sequence_dir : Path
        Directory containing semidense_observations.csv.gz and semidense_points.csv.gz
    sequence_name : str
        Sequence name (for filtering frame_map)
    scene_name : str
        Scene name for output organization
    frame_map : dict
        Mapping from frame_id to frame file path
    output_root : Path
        Root directory for output labels
    config : dict
        Configuration dictionary
    
    Returns:
    --------
    stats : dict
        Statistics about processed frames
    """
    obs_file = mps_sequence_dir / "semidense_observations.csv.gz"
    points_file = mps_sequence_dir / "semidense_points.csv.gz"
    
    if not obs_file.exists() or not points_file.exists():
        logging.warning(f"Missing CSV files in {mps_sequence_dir}")
        return {}
    
    # Load configuration
    camera_serials = config['data']['camera_serials']
    confidence_threshold = config['model']['detection_threshold']
    top_k = config['model']['top_k']
    confidence_method = config['model']['confidence']['method']
    confidence_params = config['model']['confidence'].get('params', {})
    
    # Read points to get uncertainty data
    logging.info(f"Loading points from {points_file.name}")
    with gzip.open(points_file, 'rt') as f:
        points_df = pd.read_csv(f)
    
    # Compute max inv_dist_std for normalization
    max_inv_dist_std = points_df['inv_dist_std'].max()
    confidence_params['max_inv_dist_std'] = max_inv_dist_std
    
    # Create uid -> confidence mapping
    points_df['confidence'] = compute_confidence(
        points_df['inv_dist_std'].values,
        method=confidence_method,
        **confidence_params
    )
    uid_to_confidence = dict(zip(points_df['uid'], points_df['confidence']))
    
    # Read observations in chunks for memory efficiency
    logging.info(f"Processing observations from {obs_file.name}")
    
    # First pass: collect all observations
    frame_keypoints = defaultdict(list)
    
    chunk_size = 100000
    with gzip.open(obs_file, 'rt') as f:
        for chunk in pd.read_csv(f, chunksize=chunk_size):
            # Add confidence from points
            chunk['confidence'] = chunk['uid'].map(uid_to_confidence)
            
            # Drop observations without confidence (points not in points_df)
            chunk = chunk.dropna(subset=['confidence'])
            
            # Build frame identifiers
            chunk['frame_id'] = chunk.apply(
                lambda row: build_frame_id(
                    row['frame_tracking_timestamp_us'],
                    row['camera_serial'],
                    camera_serials
                ),
                axis=1
            )
            
            # Group by frame and collect keypoints
            for frame_id, group in chunk.groupby('frame_id'):
                # Check if this frame_id matches our sequence
                if frame_id not in frame_map:
                    continue
                
                # Extract x, y, confidence
                keypoints = group[['u', 'v', 'confidence']].values
                frame_keypoints[frame_id].extend(keypoints)
    
    # Second pass: save labels for each frame
    output_scene_dir = output_root / scene_name
    output_scene_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {}
    for frame_id, keypoints_list in tqdm(frame_keypoints.items(), 
                                         desc=f"Saving labels ({sequence_name})"):
        # Convert to numpy array
        pts = np.array(keypoints_list, dtype=np.float64)
        
        # Filter and sort
        pts = filter_and_sort_keypoints(pts, confidence_threshold, top_k)
        
        if len(pts) == 0:
            logging.debug(f"No keypoints after filtering for frame {frame_id}")
            stats[frame_id] = 0
            continue
        
        # Get frame path to determine output filename
        frame_path = frame_map[frame_id]
        output_file = output_scene_dir / f"{frame_path.stem}.npz"
        
        # Save as compressed npz
        np.savez_compressed(output_file, pts=pts)
        stats[frame_id] = len(pts)
    
    logging.info(f"Saved {len(stats)} label files for sequence '{sequence_name}'")
    return stats


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def generate_labels(config_path: str):
    """
    Main function to generate SuperPoint labels from InCrowd-VI dataset.
    
    Parameters:
    -----------
    config_path : str
        Path to YAML configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    mps_root = Path(config['data']['mps_root'])
    frames_root = Path(config['data']['frames_root'])
    output_root = Path(config['data']['output_labels_root'])
    
    # Validate paths
    if not mps_root.exists():
        raise FileNotFoundError(f"MPS root not found: {mps_root}")
    if not frames_root.exists():
        raise FileNotFoundError(f"Frames root not found: {frames_root}")
    
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Discover all scenes and sequences
    all_stats = {}
    
    for scene_dir in sorted(mps_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        
        scene_name = scene_dir.name
        logging.info(f"=" * 80)
        logging.info(f"Processing scene: {scene_name}")
        logging.info(f"=" * 80)
        
        # Discover frames for this scene
        frame_map = discover_frames(frames_root, scene_name)
        
        if not frame_map:
            logging.warning(f"No frames found for scene '{scene_name}', skipping")
            continue
        
        # Process each MPS sequence in this scene
        for mps_seq_dir in sorted(scene_dir.iterdir()):
            if not mps_seq_dir.is_dir() or not mps_seq_dir.name.startswith('mps_'):
                continue
            
            sequence_name = mps_seq_dir.name.replace('mps_', '').replace('_vrs', '')
            logging.info(f"Processing sequence: {sequence_name}")
            
            # Filter frame_map to only include frames from this sequence
            seq_frame_map = {
                fid: fpath for fid, fpath in frame_map.items()
                if sequence_name in fpath.name
            }
            
            if not seq_frame_map:
                logging.warning(f"No matching frames for sequence '{sequence_name}'")
                continue
            
            stats = process_sequence(
                mps_seq_dir,
                sequence_name,
                scene_name,
                seq_frame_map,
                output_root,
                config
            )
            
            all_stats[f"{scene_name}/{sequence_name}"] = stats
    
    # Generate report
    if config.get('output', {}).get('generate_report', True):
        generate_report(all_stats, output_root, config)
    
    logging.info("Label generation complete!")


def generate_report(all_stats: Dict[str, Dict[str, int]], 
                   output_root: Path,
                   config: dict):
    """
    Generate a summary report of label generation.
    
    Parameters:
    -----------
    all_stats : dict
        Statistics from all processed sequences
    output_root : Path
        Output directory for report
    config : dict
        Configuration dictionary
    """
    report_file = config.get('output', {}).get('report_file', 'label_generation_report.txt')
    report_path = output_root / report_file
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("InCrowd-VI SuperPoint Label Generation Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Detection threshold: {config['model']['detection_threshold']}\n")
        f.write(f"  Top-k keypoints: {config['model']['top_k']}\n")
        f.write(f"  Confidence method: {config['model']['confidence']['method']}\n")
        f.write("\n")
        
        total_frames = 0
        all_counts = []
        
        for seq_id, stats in sorted(all_stats.items()):
            if not stats:
                continue
            
            counts = list(stats.values())
            total_frames += len(counts)
            all_counts.extend(counts)
            
            f.write(f"Sequence: {seq_id}\n")
            f.write(f"  Frames processed: {len(counts)}\n")
            f.write(f"  Keypoints per frame:\n")
            f.write(f"    Min:  {min(counts)}\n")
            f.write(f"    Max:  {max(counts)}\n")
            f.write(f"    Mean: {np.mean(counts):.2f}\n")
            f.write(f"    Std:  {np.std(counts):.2f}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("Global Statistics\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total frames: {total_frames}\n")
        if all_counts:
            f.write(f"Keypoints per frame (global):\n")
            f.write(f"  Min:  {min(all_counts)}\n")
            f.write(f"  Max:  {max(all_counts)}\n")
            f.write(f"  Mean: {np.mean(all_counts):.2f}\n")
            f.write(f"  Std:  {np.std(all_counts):.2f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    logging.info(f"Report saved to: {report_path}")
    print(f"\nReport saved to: {report_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate SuperPoint training labels from InCrowd-VI dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python incrowdvi_label_generator.py config.yaml
  python incrowdvi_label_generator.py config.yaml --verbose

For detailed documentation, see module docstring.
        """
    )
    
    parser.add_argument(
        'config',
        type=str,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        format='[%(asctime)s %(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=log_level
    )
    
    # Run label generation
    try:
        generate_labels(args.config)
    except Exception as e:
        logging.error(f"Label generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
