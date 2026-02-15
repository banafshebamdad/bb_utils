#
# File: cli.py
# Author: Banafshe Bamdad + GitHub Copilot (Claude Sonnet 4.5)
# Email: banafshebamdad@gmail.com
# Created: 2026-02-15
#

"""
Command-line interface for confidence decay analysis.
"""

import argparse
from pathlib import Path
import logging
import sys
from glob import glob

from .confidence_function import ConfidenceParameters
from .analyzer import ConfidenceDecayAnalyzer
from .streaming_stats import TDIGEST_AVAILABLE

logger = logging.getLogger(__name__)


def discover_files(input_paths: list) -> list:
    """Discover *_semidense_points.csv.gz files from input paths.
    
    Args:
        input_paths: List of file paths or directories
        
    Returns:
        list: List of Path objects for CSV.gz files
    """
    files = []
    for input_path in input_paths:
        p = Path(input_path)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            # Find all *_semidense_points.csv.gz files in directory
            found = list(p.glob('*_semidense_points.csv.gz'))
            files.extend(found)
        else:
            logger.warning(f"Path not found: {input_path}")
    
    return files


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze inv_dist_std distribution and recommend confidence parameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input', type=str, nargs='+', required=True,
                       help='Input CSV.gz files or directories containing *_semidense_points.csv.gz')
    parser.add_argument('--column', type=str, default='inv_dist_std',
                       help='Column name to analyze')
    parser.add_argument('--x-thr', type=float, default=0.005,
                       help='MPS nominal threshold')
    parser.add_argument('--lambda', type=float, default=3.0, dest='lambda_factor',
                       help='Safety factor (outlier cutoff = λ × x_thr)')
    parser.add_argument('--c-min', type=float, default=0.1, dest='c_min',
                       help='Minimum confidence at x_max')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--per-file', action='store_true',
                       help='Compute per-file statistics')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--percentiles', action='store_true',
                       help='Compute percentiles (requires tdigest package)')
    parser.add_argument('--chunk-size', type=int, default=100000,
                       help='Chunk size for reading large CSV files')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check tdigest availability if percentiles requested
    if args.percentiles and not TDIGEST_AVAILABLE:
        logger.error("Percentiles requested but tdigest not installed. Install with: pip install tdigest")
        sys.exit(1)
    
    # Check matplotlib availability if plots requested
    if args.plot:
        try:
            import matplotlib
        except ImportError:
            logger.error("Plotting requested but matplotlib not installed. Install with: pip install matplotlib")
            sys.exit(1)
    
    # Discover files
    logger.info(f"Discovering files from {len(args.input)} input path(s)...")
    file_paths = discover_files(args.input)
    
    if not file_paths:
        logger.error(f"No *_semidense_points.csv.gz files found in {args.input}")
        sys.exit(1)
    
    logger.info(f"Found {len(file_paths)} file(s) to process")
    
    # Create confidence parameters
    try:
        confidence_params = ConfidenceParameters(
            x_thr=args.x_thr,
            lambda_factor=args.lambda_factor,
            c_min=args.c_min
        )
    except AssertionError as e:
        logger.error(f"Invalid parameters: {e}")
        sys.exit(1)
    
    # Create analyzer
    analyzer = ConfidenceDecayAnalyzer(
        file_paths=file_paths,
        column_name=args.column,
        confidence_params=confidence_params,
        output_dir=Path(args.output_dir),
        per_file=args.per_file,
        generate_plots=args.plot,
        use_percentiles=args.percentiles,
        chunk_size=args.chunk_size,
        cli_args=vars(args)
    )
    
    # Run analysis
    try:
        logger.info("Starting analysis...")
        result = analyzer.run()
        
        # Print summary
        result.print_summary()
        
        logger.info(f"Analysis complete. Results saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
