#!/usr/bin/env python3
"""
Example usage script for InCrowd-VI SuperPoint label generator

This script demonstrates how to use the SuperPoint label generator programmatically
and provides a template for batch processing.

Author: Banafshe Bamdad
Date: January 21, 2026
"""

from pathlib import Path
from bb_utils.label_generation.incrowdvi_superpoint_labels import (
    generate_labels,
    compute_confidence,
    filter_and_sort_keypoints,
    CONFIDENCE_METHODS
)
import numpy as np


def example_basic_usage():
    """Basic usage: generate labels from config file."""
    config_path = "configs/incrowdvi_superpoint_labels.yaml"
    
    print("Generating labels from config...")
    generate_labels(config_path)
    print("Done!")


def example_confidence_computation():
    """Example: test different confidence computation methods."""
    # Simulate some uncertainty values
    inv_dist_std = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    max_inv_dist_std = 5.0
    
    print("\nConfidence Computation Examples:")
    print("=" * 60)
    print(f"inv_dist_std values: {inv_dist_std}")
    print()
    
    # Method 1: Inverse normalized
    conf_inv = compute_confidence(
        inv_dist_std,
        method='inverse_normalized',
        max_inv_dist_std=max_inv_dist_std
    )
    print(f"inverse_normalized: {conf_inv}")
    
    # Method 2: Exponential
    conf_exp = compute_confidence(
        inv_dist_std,
        method='exponential',
        scale=1.0
    )
    print(f"exponential (scale=1.0): {conf_exp}")
    
    conf_exp2 = compute_confidence(
        inv_dist_std,
        method='exponential',
        scale=0.5
    )
    print(f"exponential (scale=0.5): {conf_exp2}")


def example_keypoint_filtering():
    """Example: filter and sort keypoints."""
    # Create sample keypoints: [x, y, confidence]
    pts = np.array([
        [100, 200, 0.8],
        [150, 250, 0.3],
        [200, 300, 0.9],
        [250, 350, 0.01],  # Below threshold
        [300, 400, 0.7],
        [350, 450, 0.5],
    ])
    
    print("\nKeypoint Filtering Example:")
    print("=" * 60)
    print(f"Original keypoints ({len(pts)}):")
    print(pts)
    print()
    
    # Filter with threshold 0.015 and top-k 3
    filtered = filter_and_sort_keypoints(pts, confidence_threshold=0.015, top_k=3)
    
    print(f"After filtering (threshold=0.015, top_k=3): {len(filtered)} keypoints")
    print(filtered)
    print("\nNote: Keypoints are sorted by confidence (descending)")


def example_inspect_generated_labels():
    """Example: inspect a generated label file."""
    # This assumes you've run label generation
    # Replace with actual path to a generated file
    label_file = Path("output/labels/sequence_L_timestamp.npz")
    
    if not label_file.exists():
        print(f"\nLabel file not found: {label_file}")
        print("Run label generation first!")
        return
    
    print("\nInspecting Generated Label:")
    print("=" * 60)
    
    with np.load(label_file) as data:
        pts = data['pts']
        print(f"File: {label_file.name}")
        print(f"Shape: {pts.shape}")
        print(f"Keypoints: {len(pts)}")
        print(f"\nFirst 5 keypoints:")
        print(pts[:5])
        print(f"\nConfidence range: [{pts[:, 2].min():.4f}, {pts[:, 2].max():.4f}]")
        print(f"Mean confidence: {pts[:, 2].mean():.4f}")


def list_available_methods():
    """List all available confidence computation methods."""
    print("\nAvailable Confidence Methods:")
    print("=" * 60)
    for method_name in CONFIDENCE_METHODS.keys():
        print(f"  - {method_name}")


if __name__ == '__main__':
    print("InCrowd-VI SuperPoint Label Generator - Usage Examples")
    print("=" * 60)
    
    # List available methods
    list_available_methods()
    
    # Example: confidence computation
    example_confidence_computation()
    
    # Example: keypoint filtering
    example_keypoint_filtering()
    
    # Uncomment to run actual label generation
    # example_basic_usage()
    
    # Uncomment to inspect generated labels
    # example_inspect_generated_labels()
    
    print("\n" + "=" * 60)
    print("Examples complete!")
