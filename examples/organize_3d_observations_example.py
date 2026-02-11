#
# File: organize_3d_observations_example.py
# Author: Banafshe Bamdad + GitHub Copilot (Claude Sonnet 4.5)
# Created: 2026-02-11 16:26 CET
#

#!/usr/bin/env python3
"""
Example script demonstrating how to use the organize_3d_observations function.

This script shows both programmatic usage and simulates what the CLI tool does.
"""

from pathlib import Path
from bb_utils.data_preparation import organize_3d_observations


def example_basic_usage():
    """Basic usage example."""
    print("=" * 70)
    print("Example 1: Basic Usage")
    print("=" * 70)
    
    data_root = Path("/path/to/ROOT")
    split_csv = Path("/path/to/split_crowd_density_with_frames.csv")
    output_dir = Path("/path/to/output")
    
    successful, failed, errors = organize_3d_observations(
        data_root_dir=data_root,
        split_csv_path=split_csv,
        output_dir=output_dir,
        dry_run=False,
        symlink=False,
        verbose=False
    )
    
    print(f"\nProcessing complete: {successful} successful, {failed} failed")


def example_dry_run():
    """Dry run example - preview operations without copying."""
    print("\n" + "=" * 70)
    print("Example 2: Dry Run (Preview Mode)")
    print("=" * 70)
    
    data_root = Path("/path/to/ROOT")
    split_csv = Path("/path/to/split_crowd_density_with_frames.csv")
    output_dir = Path("/path/to/output")
    
    successful, failed, errors = organize_3d_observations(
        data_root_dir=data_root,
        split_csv_path=split_csv,
        output_dir=output_dir,
        dry_run=True,  # Preview mode
        symlink=False,
        verbose=True   # Verbose output
    )
    
    print(f"\n[DRY RUN] Would process: {successful} successful, {failed} would fail")


def example_symlink_mode():
    """Symlink example - create symlinks instead of copying to save space."""
    print("\n" + "=" * 70)
    print("Example 3: Symlink Mode (Save Disk Space)")
    print("=" * 70)
    
    data_root = Path("/path/to/ROOT")
    split_csv = Path("/path/to/split_crowd_density_with_frames.csv")
    output_dir = Path("/path/to/output")
    
    successful, failed, errors = organize_3d_observations(
        data_root_dir=data_root,
        split_csv_path=split_csv,
        output_dir=output_dir,
        dry_run=False,
        symlink=True,  # Create symlinks instead of copies
        verbose=True
    )
    
    print(f"\nSymlinking complete: {successful} successful, {failed} failed")


def example_error_handling():
    """Example showing error handling."""
    print("\n" + "=" * 70)
    print("Example 4: Error Handling")
    print("=" * 70)
    
    data_root = Path("/path/to/ROOT")
    split_csv = Path("/path/to/split_crowd_density_with_frames.csv")
    output_dir = Path("/path/to/output")
    
    try:
        successful, failed, errors = organize_3d_observations(
            data_root_dir=data_root,
            split_csv_path=split_csv,
            output_dir=output_dir,
            dry_run=False,
            symlink=False,
            verbose=True
        )
        
        # Check results
        if failed > 0:
            print(f"\n⚠️  Warning: {failed} sequences failed to process")
            print("First few errors:")
            for error in errors[:5]:
                print(f"  - {error}")
        else:
            print(f"\n✓ Success! All {successful} sequences processed")
            
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("3D OBSERVATION ORGANIZER - USAGE EXAMPLES")
    print("=" * 70)
    print("\nNOTE: Update the paths in this script before running!\n")
    
    # Uncomment the example you want to run:
    
    # example_basic_usage()
    # example_dry_run()
    # example_symlink_mode()
    # example_error_handling()
    
    print("\n" + "=" * 70)
    print("Equivalent CLI commands:")
    print("=" * 70)
    print("""
# Basic usage:
bb-organize-3d-obs \\
  --data-root /path/to/ROOT \\
  --split-csv /path/to/split.csv \\
  --output-dir /path/to/output

# Dry run:
bb-organize-3d-obs \\
  --data-root /path/to/ROOT \\
  --split-csv /path/to/split.csv \\
  --output-dir /path/to/output \\
  --dry-run \\
  --verbose

# Symlink mode:
bb-organize-3d-obs \\
  --data-root /path/to/ROOT \\
  --split-csv /path/to/split.csv \\
  --output-dir /path/to/output \\
  --symlink \\
  --verbose
    """)


if __name__ == "__main__":
    main()
