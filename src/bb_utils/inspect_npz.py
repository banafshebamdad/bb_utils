# 
# Banafshe Bamdad + GitHub Copilot (Claud Sonnet 4.5)
# Tue Jan 20, 2026 16:09 CET
# 
#!/usr/bin/env python3
"""
Comprehensive NPZ File Inspector

This script provides a detailed overview of the data structure and types
contained in a NumPy .npz file without modifying its contents.

Usage:
    python inspect_npz.py <path_to_npz_file>
"""

import argparse
import sys
import warnings
from pathlib import Path
import numpy as np


def format_bytes(num_bytes):
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def get_array_stats(array):
    """
    Compute statistics for numeric arrays.
    
    Returns a dictionary with min, max, mean, and count of non-finite values.
    Handles non-numeric arrays gracefully.
    For 2D arrays, also computes per-column statistics.
    """
    stats = {}
    
    # Check if array is numeric
    if np.issubdtype(array.dtype, np.number):
        # Handle complex numbers
        if np.issubdtype(array.dtype, np.complexfloating):
            stats['dtype_category'] = 'complex'
            stats['min'] = None
            stats['max'] = None
            stats['mean'] = None
            stats['note'] = 'Complex array - basic stats not computed'
        else:
            # For real numeric types
            stats['dtype_category'] = 'numeric'
            
            # Check for floating point types and handle non-finite values
            if np.issubdtype(array.dtype, np.floating):
                finite_mask = np.isfinite(array)
                num_nonfinite = np.sum(~finite_mask)
                stats['num_nonfinite'] = num_nonfinite
                
                if num_nonfinite > 0:
                    stats['num_nan'] = np.sum(np.isnan(array))
                    stats['num_inf'] = np.sum(np.isinf(array))
                
                # Compute stats only on finite values
                if np.any(finite_mask):
                    finite_values = array[finite_mask]
                    stats['min'] = float(np.min(finite_values))
                    stats['max'] = float(np.max(finite_values))
                    stats['mean'] = float(np.mean(finite_values))
                    
                    # Per-column statistics for 2D arrays
                    if array.ndim == 2 and array.shape[1] <= 10:
                        stats['per_column'] = []
                        for col_idx in range(array.shape[1]):
                            col_data = array[:, col_idx]
                            col_finite_mask = np.isfinite(col_data)
                            if np.any(col_finite_mask):
                                col_finite_values = col_data[col_finite_mask]
                                stats['per_column'].append({
                                    'index': col_idx,
                                    'min': float(np.min(col_finite_values)),
                                    'max': float(np.max(col_finite_values)),
                                    'mean': float(np.mean(col_finite_values))
                                })
                            else:
                                stats['per_column'].append({
                                    'index': col_idx,
                                    'min': None,
                                    'max': None,
                                    'mean': None
                                })
                else:
                    stats['min'] = None
                    stats['max'] = None
                    stats['mean'] = None
                    stats['note'] = 'All values are non-finite'
            else:
                # Integer types - no non-finite values possible
                stats['min'] = int(np.min(array)) if array.size > 0 else None
                stats['max'] = int(np.max(array)) if array.size > 0 else None
                stats['mean'] = float(np.mean(array)) if array.size > 0 else None
                
                # Per-column statistics for 2D arrays
                if array.ndim == 2 and array.shape[1] <= 10 and array.size > 0:
                    stats['per_column'] = []
                    for col_idx in range(array.shape[1]):
                        col_data = array[:, col_idx]
                        stats['per_column'].append({
                            'index': col_idx,
                            'min': int(np.min(col_data)) if col_data.size > 0 else None,
                            'max': int(np.max(col_data)) if col_data.size > 0 else None,
                            'mean': float(np.mean(col_data)) if col_data.size > 0 else None
                        })
    elif array.dtype == np.bool_:
        stats['dtype_category'] = 'boolean'
        stats['num_true'] = int(np.sum(array))
        stats['num_false'] = int(array.size - np.sum(array))
    elif array.dtype.kind in ['U', 'S']:  # Unicode or byte string
        stats['dtype_category'] = 'string'
        if array.size > 0:
            stats['sample_values'] = array.flat[0:min(3, array.size)].tolist()
    elif array.dtype == np.object_:
        stats['dtype_category'] = 'object'
        stats['warning'] = 'Object array detected - may require pickle loading'
        if array.size > 0:
            stats['sample_type'] = type(array.flat[0]).__name__
    else:
        stats['dtype_category'] = 'other'
    
    return stats


def inspect_npz_file(npz_path):
    """
    Inspect an NPZ file and print detailed information about its contents.
    
    Parameters:
    -----------
    npz_path : str or Path
        Path to the .npz file to inspect
    """
    npz_path = Path(npz_path)
    
    # Validate file exists
    if not npz_path.exists():
        print(f"Error: File '{npz_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    # Validate file extension
    if npz_path.suffix.lower() != '.npz':
        print(f"Warning: File '{npz_path}' does not have .npz extension.", file=sys.stderr)
    
    # Load the NPZ file
    try:
        with np.load(npz_path, allow_pickle=True) as npz_data:
            print("=" * 80)
            print(f"NPZ File Inspection: {npz_path}")
            print("=" * 80)
            print(f"File size: {format_bytes(npz_path.stat().st_size)}")
            print(f"Number of arrays: {len(npz_data.files)}")
            print(f"Keys: {', '.join(npz_data.files)}")
            print("=" * 80)
            print()
            
            # Iterate through each key in the archive
            for idx, key in enumerate(npz_data.files, 1):
                array = npz_data[key]
                
                print(f"[{idx}] Key: '{key}'")
                print("-" * 80)
                
                # Basic information
                print(f"  Data type:       {array.dtype}")
                print(f"  Shape:           {array.shape}")
                print(f"  Dimensions:      {array.ndim}")
                print(f"  Total elements:  {array.size:,}")
                print(f"  Memory usage:    {format_bytes(array.nbytes)}")
                
                # Get statistics based on array type
                stats = get_array_stats(array)
                
                # Print category-specific information
                if stats['dtype_category'] == 'numeric':
                    print(f"  Category:        Numeric")
                    
                    # Handle non-finite values
                    if 'num_nonfinite' in stats and stats['num_nonfinite'] > 0:
                        print(f"  Non-finite vals: {stats['num_nonfinite']:,}")
                        if 'num_nan' in stats:
                            print(f"    - NaN:         {stats['num_nan']:,}")
                        if 'num_inf' in stats:
                            print(f"    - Inf:         {stats['num_inf']:,}")
                    
                    # Print overall statistics
                    if stats['min'] is not None:
                        print(f"  Minimum:         {stats['min']}")
                        print(f"  Maximum:         {stats['max']}")
                        print(f"  Mean:            {stats['mean']}")
                        
                        # Print per-column statistics if available
                        if 'per_column' in stats:
                            print(f"\n  Per-column statistics:")
                            for col_stat in stats['per_column']:
                                col_idx = col_stat['index']
                                if col_stat['min'] is not None:
                                    print(f"    Column {col_idx}:")
                                    print(f"      Min:         {col_stat['min']}")
                                    print(f"      Max:         {col_stat['max']}")
                                    print(f"      Mean:        {col_stat['mean']}")
                                else:
                                    print(f"    Column {col_idx}:    All non-finite")
                    elif 'note' in stats:
                        print(f"  Note:            {stats['note']}")
                
                elif stats['dtype_category'] == 'complex':
                    print(f"  Category:        Complex")
                    print(f"  Note:            {stats['note']}")
                
                elif stats['dtype_category'] == 'boolean':
                    print(f"  Category:        Boolean")
                    print(f"  True values:     {stats['num_true']:,}")
                    print(f"  False values:    {stats['num_false']:,}")
                
                elif stats['dtype_category'] == 'string':
                    print(f"  Category:        String")
                    if 'sample_values' in stats:
                        print(f"  Sample values:   {stats['sample_values']}")
                
                elif stats['dtype_category'] == 'object':
                    print(f"  Category:        Object")
                    print(f"  ⚠ WARNING:       {stats['warning']}")
                    if 'sample_type' in stats:
                        print(f"  Sample type:     {stats['sample_type']}")
                
                else:
                    print(f"  Category:        {stats['dtype_category']}")
                
                print()
            
            print("=" * 80)
            print("Inspection complete.")
            print("=" * 80)
    
    except Exception as e:
        print(f"Error loading NPZ file: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Inspect the contents of a NumPy .npz file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inspect_npz.py data.npz
  python inspect_npz.py /path/to/dataset.npz
        """
    )
    
    parser.add_argument(
        'npz_file',
        type=str,
        help='Path to the .npz file to inspect'
    )
    
    parser.add_argument(
        '-v', '--version',
        action='version',
        version='NPZ Inspector 1.0.0'
    )
    
    args = parser.parse_args()
    
    # Inspect the file
    inspect_npz_file(args.npz_file)


if __name__ == '__main__':
    main()
