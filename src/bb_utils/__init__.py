"""
bb_utils - Utilities for SuperPoint training and dataset processing

This package provides tools for:
- NPZ file inspection (utils)
- InCrowd-VI dataset label generation for SuperPoint training (label_generation)
- 3D observation data organization (data_preparation)
- World-coordinate cache building from Aria MPS semidense point clouds (data_preparation)
- Future: SP-SCore label generation (label_generation)
"""

from .label_generation import generate_labels
from .utils import inspect_npz_file
from .data_preparation import organize_3d_observations
from .data_preparation import build_world_coords_cache, discover_sequences, run_split

__all__ = [
    'generate_labels',
    'inspect_npz_file',
    'organize_3d_observations',
    'build_world_coords_cache',
    'discover_sequences',
    'run_split',
]
