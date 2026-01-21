"""
bb_utils - Utilities for SuperPoint training and dataset processing

This package provides tools for:
- NPZ file inspection (utils)
- InCrowd-VI dataset label generation for SuperPoint training (label_generation)
- Future: SP-SCore label generation (label_generation)
"""

from .label_generation import generate_labels
from .utils import inspect_npz_file

__all__ = ['generate_labels', 'inspect_npz_file']
