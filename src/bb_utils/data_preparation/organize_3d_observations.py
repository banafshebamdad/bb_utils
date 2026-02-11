#
# File: organize_3d_observations.py
# Author: Banafshe Bamdad + GitHub Copilot (Claude Sonnet 4.5)
# Created: 2026-02-11 16:25 CET
#

#!/usr/bin/env python3
"""
Organize 3D observation data by train/val/test split.

This tool reads a CSV file containing scene/sequence metadata with split assignments,
and organizes the corresponding semidense point/observation files into train/val/test
directories with proper naming conventions.
"""

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def read_split_csv(csv_path: Path, logger: logging.Logger) -> List[Dict[str, str]]:
    """
    Read the split CSV file and return list of sequence metadata.
    
    Args:
        csv_path: Path to split CSV file
        logger: Logger instance
        
    Returns:
        List of dictionaries containing sequence metadata
    """
    logger.info(f"Reading split CSV from: {csv_path}")
    
    sequences = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip whitespace from all fields
            sequences.append({k.strip(): v.strip() for k, v in row.items()})
    
    logger.info(f"Found {len(sequences)} sequences in CSV")
    return sequences


def create_output_structure(output_dir: Path, logger: logging.Logger) -> Dict[str, Path]:
    """
    Create output directory structure for train/val/test splits.
    
    Args:
        output_dir: Base output directory
        logger: Logger instance
        
    Returns:
        Dictionary mapping split names to their paths
    """
    base_dir = output_dir / "raw_3d_observations"
    split_dirs = {}
    
    for split in ['train', 'val', 'test']:
        split_path = base_dir / split
        split_path.mkdir(parents=True, exist_ok=True)
        split_dirs[split] = split_path
        logger.info(f"Created directory: {split_path}")
    
    return split_dirs


def organize_3d_observations(
    data_root_dir: Path,
    split_csv_path: Path,
    output_dir: Path,
    dry_run: bool = False,
    symlink: bool = False,
    verbose: bool = False
) -> Tuple[int, int, List[str]]:
    """
    Organize 3D observation files into train/val/test directories.
    
    Args:
        data_root_dir: Root directory containing scene folders
        split_csv_path: Path to CSV file with split information
        output_dir: Base output directory
        dry_run: If True, only simulate operations without copying files
        symlink: If True, create symlinks instead of copying files
        verbose: Enable verbose logging
        
    Returns:
        Tuple of (successful_copies, failed_copies, error_messages)
    """
    logger = setup_logging(verbose)
    
    # Validate inputs
    if not data_root_dir.exists():
        raise FileNotFoundError(f"Data root directory not found: {data_root_dir}")
    if not split_csv_path.exists():
        raise FileNotFoundError(f"Split CSV file not found: {split_csv_path}")
    
    # Read split information
    sequences = read_split_csv(split_csv_path, logger)
    
    # Create output structure
    if not dry_run:
        split_dirs = create_output_structure(output_dir, logger)
    else:
        logger.info("[DRY RUN] Would create output directories")
        split_dirs = {
            'train': output_dir / "raw_3d_observations" / "train",
            'val': output_dir / "raw_3d_observations" / "val",
            'test': output_dir / "raw_3d_observations" / "test"
        }
    
    # Process each sequence
    successful = 0
    failed = 0
    errors = []
    
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Processing sequences...")
    operation = "Would create symlink" if symlink else "Would copy" if dry_run else "Creating symlink" if symlink else "Copying"
    
    for seq_info in tqdm(sequences, desc=operation):
        scene = seq_info['scene']
        sequence = seq_info['sequence']
        mps = seq_info['mps']
        split = seq_info['split']
        
        # Construct source directory path
        source_dir = data_root_dir / scene / mps
        
        # Check if source directory exists
        if not source_dir.exists():
            error_msg = f"Source directory not found: {source_dir}"
            logger.warning(error_msg)
            errors.append(error_msg)
            failed += 1
            continue
        
        # Process both files
        files_to_process = [
            'semidense_points.csv.gz',
            'semidense_observations.csv.gz'
        ]
        
        sequence_failed = False
        for filename in files_to_process:
            source_file = source_dir / filename
            dest_filename = f"{sequence}_{filename}"
            dest_file = split_dirs[split] / dest_filename
            
            # Check if source file exists
            if not source_file.exists():
                error_msg = f"Source file not found: {source_file}"
                logger.warning(error_msg)
                errors.append(error_msg)
                sequence_failed = True
                continue
            
            # Copy or symlink the file
            if dry_run:
                logger.debug(f"Would copy: {source_file} -> {dest_file}")
            else:
                try:
                    if symlink:
                        # Create absolute symlink
                        if dest_file.exists() or dest_file.is_symlink():
                            dest_file.unlink()
                        dest_file.symlink_to(source_file.absolute())
                        logger.debug(f"Symlinked: {source_file} -> {dest_file}")
                    else:
                        shutil.copy2(source_file, dest_file)
                        logger.debug(f"Copied: {source_file} -> {dest_file}")
                except Exception as e:
                    error_msg = f"Failed to process {source_file}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    sequence_failed = True
        
        if sequence_failed:
            failed += 1
        else:
            successful += 1
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total sequences processed: {len(sequences)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    
    if errors:
        logger.info(f"\nErrors encountered: {len(errors)}")
        for error in errors[:10]:  # Show first 10 errors
            logger.warning(f"  - {error}")
        if len(errors) > 10:
            logger.warning(f"  ... and {len(errors) - 10} more errors")
    
    logger.info("=" * 70)
    
    return successful, failed, errors


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Organize 3D observation data by train/val/test split",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  bb-organize-3d-obs \\
    --data-root /path/to/ROOT \\
    --split-csv split_crowd_density_with_frames.csv \\
    --output-dir /path/to/output

  # Dry run to preview operations
  bb-organize-3d-obs \\
    --data-root /path/to/ROOT \\
    --split-csv split.csv \\
    --output-dir /path/to/output \\
    --dry-run

  # Create symlinks instead of copying (saves disk space)
  bb-organize-3d-obs \\
    --data-root /path/to/ROOT \\
    --split-csv split.csv \\
    --output-dir /path/to/output \\
    --symlink
        """
    )
    
    parser.add_argument(
        '--data-root',
        type=Path,
        required=True,
        help='Root directory containing scene folders'
    )
    parser.add_argument(
        '--split-csv',
        type=Path,
        required=True,
        help='Path to CSV file with split information (columns: scene, sequence, mps, density, frame_count, split)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Base output directory (will create raw_3d_observations/ subdirectory)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview operations without actually copying files'
    )
    parser.add_argument(
        '--symlink',
        action='store_true',
        help='Create symbolic links instead of copying files (saves disk space)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    try:
        successful, failed, errors = organize_3d_observations(
            data_root_dir=args.data_root,
            split_csv_path=args.split_csv,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            symlink=args.symlink,
            verbose=args.verbose
        )
        
        # Exit with error code if any failures
        if failed > 0:
            sys.exit(1)
        
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
