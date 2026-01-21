"""
bb_utils - Utilities for SuperPoint training and dataset processing

This package provides tools for:
- NPZ file inspection
- InCrowd-VI dataset label generation for SuperPoint training
- Future: SP-SCore label generation
"""

from .incrowdvi_superpoint_labels import generate_labels

__all__ = ['generate_labels']
