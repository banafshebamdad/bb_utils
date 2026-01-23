#!/usr/bin/env python3
"""
Visual validation tool for InCrowd-VI SuperPoint labels.

This script helps validate generated labels by:
1. Visualizing keypoints overlaid on images
2. Computing spatial distribution metrics
3. Checking label quality indicators
4. Comparing with expected patterns

Usage:
    python validate_labels.py \
        --frames_dir /path/to/frames \
        --labels_dir /path/to/labels \
        --output_dir /path/to/validation_output \
        --num_samples 50
"""

import argparse
import logging
from pathlib import Path
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_label(label_file: Path) -> np.ndarray:
    """Load keypoints from .npz file."""
    data = np.load(label_file)
    return data['pts']  # Shape: (N, 3) - [x, y, confidence]


def visualize_keypoints(image_path: Path, 
                       label_path: Path, 
                       output_path: Path,
                       title: str = ""):
    """
    Visualize keypoints on image.
    
    Creates visualization with:
    - Keypoints colored by confidence (red=low, green=high)
    - Confidence histogram
    - Spatial distribution heatmap
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        logging.error(f"Failed to load image: {image_path}")
        return False
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    # Load keypoints
    pts = load_label(label_path)
    if len(pts) == 0:
        logging.warning(f"No keypoints in {label_path}")
        return False
    
    x, y, conf = pts[:, 0], pts[:, 1], pts[:, 2]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Keypoints on image (colored by confidence)
    ax = axes[0, 0]
    ax.imshow(img_rgb)
    scatter = ax.scatter(x, y, c=conf, cmap='RdYlGn', s=20, alpha=0.7, 
                        vmin=0, vmax=1, edgecolors='black', linewidths=0.5)
    ax.set_title(f'Keypoints (n={len(pts)}) - Colored by Confidence')
    ax.axis('off')
    plt.colorbar(scatter, ax=ax, label='Confidence')
    
    # 2. Confidence histogram
    ax = axes[0, 1]
    ax.hist(conf, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Count')
    ax.set_title(f'Confidence Distribution\nMean: {conf.mean():.3f}, Std: {conf.std():.3f}')
    ax.grid(True, alpha=0.3)
    ax.axvline(conf.mean(), color='r', linestyle='--', label=f'Mean: {conf.mean():.3f}')
    ax.axvline(conf.median(), color='g', linestyle='--', label=f'Median: {np.median(conf):.3f}')
    ax.legend()
    
    # 3. Spatial heatmap
    ax = axes[1, 0]
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=[32, 24], 
                                              range=[[0, w], [0, h]])
    im = ax.imshow(heatmap.T, origin='lower', cmap='hot', 
                   extent=[0, w, 0, h], aspect='auto', alpha=0.6)
    ax.imshow(img_rgb, alpha=0.4, extent=[0, w, 0, h], aspect='auto')
    ax.set_title('Spatial Distribution Heatmap')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    plt.colorbar(im, ax=ax, label='Keypoint Density')
    
    # 4. Statistics table
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"""
    Label Statistics
    {'='*40}
    
    Image: {image_path.name}
    Image size: {w} × {h}
    
    Keypoints:
      Total count: {len(pts)}
      Density: {len(pts)/(w*h)*10000:.2f} pts/10k pixels
    
    Confidence:
      Min:    {conf.min():.4f}
      Max:    {conf.max():.4f}
      Mean:   {conf.mean():.4f}
      Median: {np.median(conf):.4f}
      Std:    {conf.std():.4f}
    
    Spatial Coverage:
      X range: [{x.min():.1f}, {x.max():.1f}]
      Y range: [{y.min():.1f}, {y.max():.1f}]
      X spread (std): {x.std():.1f}
      Y spread (std): {y.std():.1f}
    
    Quality Indicators:
      High conf (>0.8): {(conf > 0.8).sum()} ({(conf > 0.8).sum()/len(pts)*100:.1f}%)
      Med conf (0.5-0.8): {((conf >= 0.5) & (conf <= 0.8)).sum()} ({((conf >= 0.5) & (conf <= 0.8)).sum()/len(pts)*100:.1f}%)
      Low conf (<0.5): {(conf < 0.5).sum()} ({(conf < 0.5).sum()/len(pts)*100:.1f}%)
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    
    fig.suptitle(title or f'Label Validation: {image_path.stem}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True


def compute_label_quality_metrics(label_path: Path) -> dict:
    """
    Compute quality metrics for a label file.
    
    Returns:
        dict with quality metrics
    """
    pts = load_label(label_path)
    
    if len(pts) == 0:
        return {'valid': False, 'num_points': 0}
    
    x, y, conf = pts[:, 0], pts[:, 1], pts[:, 2]
    
    # Check for duplicate points (potential issue)
    coords = pts[:, :2]
    unique_coords = np.unique(coords, axis=0)
    has_duplicates = len(unique_coords) < len(coords)
    
    # Spatial uniformity (using standard deviation of distances to mean)
    center_x, center_y = x.mean(), y.mean()
    distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    spatial_uniformity = distances.std() / distances.mean() if len(distances) > 0 else 0
    
    return {
        'valid': True,
        'num_points': len(pts),
        'conf_mean': conf.mean(),
        'conf_std': conf.std(),
        'conf_min': conf.min(),
        'conf_max': conf.max(),
        'high_conf_ratio': (conf > 0.8).sum() / len(pts),
        'low_conf_ratio': (conf < 0.5).sum() / len(pts),
        'has_duplicates': has_duplicates,
        'spatial_uniformity': spatial_uniformity,
        'x_coverage': (x.max() - x.min()),
        'y_coverage': (y.max() - y.min()),
    }


def validate_dataset(frames_dir: Path, 
                    labels_dir: Path, 
                    output_dir: Path,
                    num_samples: int = 50):
    """
    Validate entire dataset.
    
    1. Random sample visualization
    2. Compute quality metrics for all labels
    3. Generate summary report
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    # Find all label files
    label_files = sorted(labels_dir.glob('*.npz'))
    logging.info(f"Found {len(label_files)} label files")
    
    if len(label_files) == 0:
        logging.error("No label files found!")
        return
    
    # 1. Random sample for visualization
    sample_size = min(num_samples, len(label_files))
    sample_labels = random.sample(label_files, sample_size)
    
    logging.info(f"Visualizing {sample_size} random samples...")
    successful_vis = 0
    for label_path in tqdm(sample_labels, desc="Creating visualizations"):
        # Find corresponding image
        img_path = frames_dir / f"{label_path.stem}.png"
        
        if not img_path.exists():
            logging.warning(f"Image not found for {label_path.name}")
            continue
        
        output_path = vis_dir / f"{label_path.stem}_validation.png"
        
        if visualize_keypoints(img_path, label_path, output_path):
            successful_vis += 1
    
    logging.info(f"Created {successful_vis} visualizations in {vis_dir}")
    
    # 2. Compute metrics for all labels
    logging.info("Computing quality metrics for all labels...")
    all_metrics = []
    
    for label_path in tqdm(label_files, desc="Computing metrics"):
        metrics = compute_label_quality_metrics(label_path)
        metrics['filename'] = label_path.name
        all_metrics.append(metrics)
    
    # Filter valid metrics
    valid_metrics = [m for m in all_metrics if m['valid']]
    
    # 3. Generate summary report
    report_path = output_dir / 'validation_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("InCrowd-VI Label Validation Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total labels: {len(label_files)}\n")
        f.write(f"Valid labels: {len(valid_metrics)}\n")
        f.write(f"Visualizations created: {successful_vis}\n\n")
        
        if valid_metrics:
            f.write("=" * 80 + "\n")
            f.write("Quality Metrics Summary\n")
            f.write("=" * 80 + "\n\n")
            
            metrics_arrays = {
                k: np.array([m[k] for m in valid_metrics if k in m and isinstance(m[k], (int, float))])
                for k in ['num_points', 'conf_mean', 'conf_std', 'high_conf_ratio', 
                         'low_conf_ratio', 'spatial_uniformity']
            }
            
            for metric_name, values in metrics_arrays.items():
                if len(values) > 0:
                    f.write(f"{metric_name}:\n")
                    f.write(f"  Mean:   {values.mean():.4f}\n")
                    f.write(f"  Median: {np.median(values):.4f}\n")
                    f.write(f"  Std:    {values.std():.4f}\n")
                    f.write(f"  Min:    {values.min():.4f}\n")
                    f.write(f"  Max:    {values.max():.4f}\n\n")
            
            # Check for potential issues
            f.write("=" * 80 + "\n")
            f.write("Potential Issues\n")
            f.write("=" * 80 + "\n\n")
            
            files_with_duplicates = [m['filename'] for m in valid_metrics if m.get('has_duplicates')]
            low_point_files = [m['filename'] for m in valid_metrics if m.get('num_points', 1000) < 100]
            low_conf_files = [m['filename'] for m in valid_metrics if m.get('conf_mean', 1) < 0.3]
            
            f.write(f"Files with duplicate keypoints: {len(files_with_duplicates)}\n")
            if files_with_duplicates[:10]:
                f.write("  Examples: " + ", ".join(files_with_duplicates[:10]) + "\n")
            
            f.write(f"\nFiles with <100 keypoints: {len(low_point_files)}\n")
            if low_point_files[:10]:
                f.write("  Examples: " + ", ".join(low_point_files[:10]) + "\n")
            
            f.write(f"\nFiles with mean confidence <0.3: {len(low_conf_files)}\n")
            if low_conf_files[:10]:
                f.write("  Examples: " + ", ".join(low_conf_files[:10]) + "\n")
    
    logging.info(f"Validation report saved to: {report_path}")
    print(f"\n✓ Validation complete!")
    print(f"  Visualizations: {vis_dir}")
    print(f"  Report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate InCrowd-VI SuperPoint labels",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--frames_dir', type=Path, required=True,
                       help='Directory containing frame images')
    parser.add_argument('--labels_dir', type=Path, required=True,
                       help='Directory containing label .npz files')
    parser.add_argument('--output_dir', type=Path, required=True,
                       help='Output directory for validation results')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of random samples to visualize')
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
    
    validate_dataset(args.frames_dir, args.labels_dir, args.output_dir, args.num_samples)


if __name__ == '__main__':
    main()
