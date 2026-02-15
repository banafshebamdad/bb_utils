#
# File: analyzer.py
# Author: Banafshe Bamdad + GitHub Copilot (Claude Sonnet 4.5)
# Email: banafshebamdad@gmail.com
# Created: 2026-02-15
#

"""
Core analysis orchestrator with streaming statistics and improved plotting.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional
import logging
import json

from .streaming_stats import (
    StreamingStatistics,
    StreamingFilteringStats,
    StreamingConfidenceStats,
    PerFileStatistics
)
from .confidence_function import ConfidenceParameters, MPSConfidenceFunction

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Complete analysis result.
    
    Attributes:
        confidence_params: Confidence function parameters
        files_processed: List of processed file paths
        column_name: Column analyzed
        global_statistics: Global distribution statistics
        per_file_statistics: Per-file statistics DataFrame (optional)
        filtering_report: Outlier filtering statistics
        confidence_distribution: Confidence value statistics
        sanity_checks: Confidence function sanity checks
        timestamp: Analysis timestamp
        cli_args: CLI arguments used
    """
    confidence_params: ConfidenceParameters
    files_processed: List[str]
    column_name: str
    global_statistics: dict
    per_file_statistics: Optional[pd.DataFrame]
    filtering_report: dict
    confidence_distribution: dict
    sanity_checks: dict
    timestamp: str
    cli_args: dict
    
    def to_json(self, filepath: Path) -> None:
        """Save complete analysis as JSON.
        
        Args:
            filepath: Path to save JSON file
        """
        data = {
            'timestamp': self.timestamp,
            'cli_args': self.cli_args,
            'confidence_parameters': self.confidence_params.to_dict(),
            'files_processed': self.files_processed,
            'column_name': self.column_name,
            'statistics': self.global_statistics,
            'filtering_report': self.filtering_report,
            'confidence_distribution': self.confidence_distribution,
            'sanity_checks': self.sanity_checks
        }
        
        if self.per_file_statistics is not None:
            data['per_file_statistics'] = self.per_file_statistics.to_dict('records')
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def to_summary_csv(self, filepath: Path) -> None:
        """Save statistics and confidence info as CSV.
        
        Args:
            filepath: Path to save CSV file
        """
        rows = []
        
        # Overall row
        overall = {'file': 'OVERALL'}
        overall.update(self._flatten_dict(self.global_statistics))
        overall.update(self.confidence_params.to_dict())
        overall.update(self.filtering_report)
        overall.update(self.confidence_distribution)
        overall.update(self._flatten_dict(self.sanity_checks))
        rows.append(overall)
        
        # Per-file rows (if available)
        if self.per_file_statistics is not None:
            for _, row in self.per_file_statistics.iterrows():
                rows.append(row.to_dict())
        
        df = pd.DataFrame(rows)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
    
    def print_summary(self) -> None:
        """Print human-readable summary to stdout."""
        print("\n" + "="*70)
        print("CONFIDENCE DECAY ANALYSIS REPORT")
        print("="*70)
        
        print("\n--- CONFIDENCE PARAMETERS ---")
        print(f"MPS Threshold (x_thr):     {self.confidence_params.x_thr:.6f}")
        print(f"Safety Factor (λ):         {self.confidence_params.lambda_factor:.2f}")
        print(f"Outlier Cutoff (x_max):    {self.confidence_params.x_max:.6f}")
        print(f"Minimum Confidence (c_min): {self.confidence_params.c_min:.3f}")
        print(f"Decay Rate (alpha):        {self.confidence_params.alpha:.4f}")
        
        print("\n--- DATA QUALITY ---")
        dq = self.global_statistics['data_quality']
        print(f"Total rows scanned:        {dq['n_total_rows_scanned']:,}")
        print(f"Valid points:              {dq['n_valid']:,}")
        print(f"NaN values:                {dq['n_nan']:,}")
        print(f"Inf values:                {dq['n_inf']:,}")
        print(f"Negative values:           {dq['n_negative']:,}")
        
        print("\n--- DISTRIBUTION (raw data) ---")
        dist = self.global_statistics['distribution']
        print(f"Min:     {dist['min']:.6f}")
        print(f"Max:     {dist['max']:.6f}")
        print(f"Mean:    {dist['mean']:.6f}")
        print(f"Median:  {dist.get('median', 'N/A')}")
        print(f"Std:     {dist['std']:.6f}")
        if 'percentiles' in dist and isinstance(dist['percentiles'], dict):
            if 'p95' in dist['percentiles']:
                print(f"P95:     {dist['percentiles']['p95']:.6f}")
            if 'p99' in dist['percentiles']:
                print(f"P99:     {dist['percentiles']['p99']:.6f}")
        
        print("\n--- FILTERING REPORT ---")
        fr = self.filtering_report
        print(f"Outliers removed (x > {self.confidence_params.x_max:.6f}): {fr['n_removed_outliers']:,}")
        print(f"Points kept:               {fr['n_kept']:,}")
        print(f"Percentage removed:        {fr['percentage_removed']:.2f}%")
        
        print("\n--- CONFIDENCE SANITY CHECKS ---")
        cp = self.sanity_checks['checkpoints']
        print(f"c(x_thr)            = {cp['c_at_x_thr']:.6f}  (should be 1.0)")
        print(f"c(x_thr + 25% Δx)   = {cp['c_at_25pct']:.6f}")
        print(f"c(x_thr + 50% Δx)   = {cp['c_at_50pct']:.6f}")
        print(f"c(x_thr + 75% Δx)   = {cp['c_at_75pct']:.6f}")
        print(f"c(x_max)            = {cp['c_at_x_max']:.6f}  (should be {self.confidence_params.c_min:.3f})")
        
        cd = self.confidence_distribution
        print(f"\nConfidence distribution (kept points):")
        print(f"  Min:     {cd.get('confidence_min', 'N/A')}")
        print(f"  Median:  {cd.get('confidence_median', 'N/A')}")
        print(f"  Mean:    {cd.get('confidence_mean', 'N/A')}")
        print(f"  Max:     {cd.get('confidence_max', 'N/A')}")
        
        print(f"\nFiles processed: {len(self.files_processed)}")
        print("="*70 + "\n")
    
    @staticmethod
    def _flatten_dict(d: dict, parent_key: str = '', sep: str = '_') -> dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(AnalysisResult._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


class ReservoirSampler:
    """Reservoir sampling for unbiased data collection.
    
    Implements Algorithm R for uniform random sampling without replacement.
    Reference: Vitter, Jeffrey S. (1985). "Random sampling with a reservoir"
    
    Args:
        max_samples: Maximum number of samples to collect
    
    Example:
        >>> sampler = ReservoirSampler(max_samples=1000)
        >>> for chunk in data_chunks:
        ...     sampler.add_batch(chunk)
        >>> samples = sampler.get_sample()
    """
    
    def __init__(self, max_samples: int):
        self.max_samples = max_samples
        self.reservoir = []
        self.n_seen = 0
    
    def add_batch(self, values: np.ndarray) -> None:
        """Add a batch of values to reservoir with unbiased sampling.
        
        Args:
            values: Array of values to potentially add to reservoir
        """
        for value in values:
            self.n_seen += 1
            
            if len(self.reservoir) < self.max_samples:
                # Fill reservoir
                self.reservoir.append(value)
            else:
                # Random replacement
                j = np.random.randint(0, self.n_seen)
                if j < self.max_samples:
                    self.reservoir[j] = value
    
    def get_sample(self) -> np.ndarray:
        """Get the collected sample.
        
        Returns:
            np.ndarray: Sampled values
        """
        return np.array(self.reservoir)


class ConfidenceDecayAnalyzer:
    """Main analysis orchestrator - fully streaming with optional percentiles.
    
    Args:
        file_paths: List of CSV.gz file paths to analyze
        column_name: Column name to analyze (e.g., 'inv_dist_std')
        confidence_params: Confidence function parameters
        output_dir: Directory for output files
        per_file: Whether to compute per-file statistics
        generate_plots: Whether to generate visualization plots
        use_percentiles: Whether to compute percentiles (requires tdigest)
        chunk_size: Number of rows to read per chunk
        cli_args: CLI arguments for record keeping
    
    Example:
        >>> params = ConfidenceParameters(x_thr=0.005, lambda_factor=3.0, c_min=0.1)
        >>> analyzer = ConfidenceDecayAnalyzer(
        ...     file_paths=file_paths,
        ...     column_name='inv_dist_std',
        ...     confidence_params=params,
        ...     output_dir=Path('./output'),
        ...     per_file=True,
        ...     generate_plots=True,
        ...     use_percentiles=True
        ... )
        >>> result = analyzer.run()
        >>> result.print_summary()
    """
    
    def __init__(
        self,
        file_paths: List[Path],
        column_name: str,
        confidence_params: ConfidenceParameters,
        output_dir: Path,
        per_file: bool,
        generate_plots: bool,
        use_percentiles: bool = True,
        chunk_size: int = 100000,
        cli_args: dict = None
    ):
        self.file_paths = file_paths
        self.column_name = column_name
        self.confidence_params = confidence_params
        self.confidence_func = MPSConfidenceFunction(confidence_params)
        self.output_dir = output_dir
        self.per_file = per_file
        self.generate_plots = generate_plots
        self.use_percentiles = use_percentiles
        self.chunk_size = chunk_size
        self.cli_args = cli_args or {}
    
    def run(self) -> AnalysisResult:
        """Execute full analysis - all streaming, constant memory.
        
        Returns:
            AnalysisResult: Complete analysis results
        """
        
        logger.info("Initializing streaming accumulators...")
        
        # Determine percentiles to compute
        percentiles = [25, 50, 75, 90, 95, 99, 99.9] if self.use_percentiles else None
        
        # Global streaming accumulators
        global_stats = StreamingStatistics(percentiles=percentiles)
        filtering_stats = StreamingFilteringStats(self.confidence_params.x_max)
        confidence_stats = StreamingConfidenceStats(
            self.confidence_func, 
            use_percentiles=self.use_percentiles
        )
        
        # Per-file tracking (if requested)
        per_file_tracker = None
        if self.per_file:
            per_file_tracker = PerFileStatistics(
                percentiles=percentiles,
                confidence_func=self.confidence_func,
                use_percentiles=self.use_percentiles
            )
        
        # Process each file in streaming fashion
        logger.info(f"Processing {len(self.file_paths)} files...")
        for filepath in self.file_paths:
            logger.info(f"  Reading {filepath.name}...")
            
            # Read file in chunks
            for chunk in pd.read_csv(filepath, compression='gzip', chunksize=self.chunk_size):
                if self.column_name not in chunk.columns:
                    raise ValueError(f"Column '{self.column_name}' not found in {filepath}")
                
                values = chunk[self.column_name].values
                
                # Update all global accumulators with this chunk (vectorized)
                global_stats.update(values)
                filtering_stats.update(values)
                confidence_stats.update(values)
            
            # Per-file statistics (requires separate pass through file)
            if self.per_file:
                logger.info(f"  Computing per-file statistics for {filepath.name}...")
                per_file_tracker.process_file(filepath, self.column_name, self.chunk_size)
        
        # Finalize all statistics
        logger.info("Finalizing statistics...")
        global_statistics = global_stats.finalize()
        filtering_report = filtering_stats.finalize()
        confidence_distribution = confidence_stats.finalize()
        
        # Compute sanity checks
        logger.info("Computing sanity checks...")
        sanity_checks = self.confidence_func.sanity_checks(confidence_distribution)
        
        # Create result
        result = AnalysisResult(
            confidence_params=self.confidence_params,
            files_processed=[str(p) for p in self.file_paths],
            column_name=self.column_name,
            global_statistics=global_statistics,
            per_file_statistics=per_file_tracker.to_dataframe() if self.per_file else None,
            filtering_report=filtering_report,
            confidence_distribution=confidence_distribution,
            sanity_checks=sanity_checks,
            timestamp=datetime.now().isoformat(),
            cli_args=self.cli_args
        )
        
        # Save outputs
        logger.info("Saving outputs...")
        self._save_outputs(result)
        
        return result
    
    def _save_outputs(self, result: AnalysisResult) -> None:
        """Save JSON, CSV, and optional plots.
        
        Args:
            result: Analysis result to save
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON report
        result.to_json(self.output_dir / 'confidence_analysis.json')
        logger.info(f"Saved JSON report to {self.output_dir / 'confidence_analysis.json'}")
        
        # CSV summary
        result.to_summary_csv(self.output_dir / 'confidence_summary.csv')
        logger.info(f"Saved CSV summary to {self.output_dir / 'confidence_summary.csv'}")
        
        # Plots (if requested)
        if self.generate_plots:
            logger.info("Generating plots with reservoir sampling...")
            self._generate_plots_with_sampling(result)
    
    def _generate_plots_with_sampling(self, result: AnalysisResult, max_samples: int = 1_000_000) -> None:
        """Generate analysis plots using reservoir sampling for unbiased data collection.
        
        Args:
            result: Analysis result
            max_samples: Maximum number of samples to collect for plotting
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not available. Install with: pip install matplotlib")
            return
        
        # Use reservoir sampling for unbiased collection across all files
        logger.info(f"  Collecting up to {max_samples:,} samples via reservoir sampling...")
        raw_sampler = ReservoirSampler(max_samples)
        filtered_sampler = ReservoirSampler(max_samples)
        
        np.random.seed(42)  # For reproducibility
        
        for filepath in self.file_paths:
            for chunk in pd.read_csv(filepath, compression='gzip', chunksize=self.chunk_size):
                values = chunk[self.column_name].values
                valid = values[(np.isfinite(values)) & (values >= 0)]
                
                # Add to raw sampler
                raw_sampler.add_batch(valid)
                
                # Add kept values to filtered sampler
                kept = valid[valid <= self.confidence_params.x_max]
                filtered_sampler.add_batch(kept)
        
        raw_values = raw_sampler.get_sample()
        filtered_values = filtered_sampler.get_sample()
        
        logger.info(f"  Collected {len(raw_values):,} raw samples, {len(filtered_values):,} filtered samples")
        logger.info(f"  Creating plots...")
        
        # Plot 1: Histogram and CDF of inv_dist_std (log x-axis)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Histogram
        ax1.hist(raw_values, bins=100, alpha=0.7, edgecolor='black')
        ax1.axvline(self.confidence_params.x_thr, color='green', linestyle='--', 
                    linewidth=2, label=f'x_thr = {self.confidence_params.x_thr:.6f}')
        ax1.axvline(self.confidence_params.x_max, color='red', linestyle='--', 
                    linewidth=2, label=f'x_max = {self.confidence_params.x_max:.6f}')
        ax1.set_xscale('log')
        ax1.set_xlabel('inv_dist_std (log scale)')
        ax1.set_ylabel('Count')
        ax1.set_title(f'Distribution of inv_dist_std (n={len(raw_values):,} samples)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # CDF
        sorted_vals = np.sort(raw_values)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax2.plot(sorted_vals, cdf, linewidth=2)
        ax2.axvline(self.confidence_params.x_thr, color='green', linestyle='--', 
                    linewidth=2, label='x_thr')
        ax2.axvline(self.confidence_params.x_max, color='red', linestyle='--', 
                    linewidth=2, label='x_max')
        ax2.set_xscale('log')
        ax2.set_xlabel('inv_dist_std (log scale)')
        ax2.set_ylabel('CDF')
        ax2.set_title('Cumulative Distribution Function')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'inv_dist_std_distribution.png', dpi=150)
        plt.close()
        logger.info(f"  Saved {self.output_dir / 'inv_dist_std_distribution.png'}")
        
        # Plot 2: Confidence curve
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_range = np.linspace(0, self.confidence_params.x_max * 1.2, 1000)
        confidences = self.confidence_func.compute_confidence(x_range, strict=False)
        
        ax.plot(x_range, confidences, linewidth=2, label='c(x)')
        ax.axvline(self.confidence_params.x_thr, color='green', linestyle='--', 
                   linewidth=2, label='x_thr (c=1)')
        ax.axvline(self.confidence_params.x_max, color='red', linestyle='--', 
                   linewidth=2, label=f'x_max (c={self.confidence_params.c_min})')
        ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(self.confidence_params.c_min, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('inv_dist_std')
        ax.set_ylabel('Confidence c(x)')
        ax.set_title('MPS-Threshold-Based Confidence Function')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confidence_curve.png', dpi=150)
        plt.close()
        logger.info(f"  Saved {self.output_dir / 'confidence_curve.png'}")
        
        # Plot 3: Histogram of confidence values (for kept points)
        if len(filtered_values) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            confidences_kept = self.confidence_func.compute_confidence(filtered_values, strict=False)
            
            ax.hist(confidences_kept, bins=50, alpha=0.7, edgecolor='black')
            median_conf = np.median(confidences_kept)
            ax.axvline(median_conf, color='orange', linestyle='--', 
                       linewidth=2, label=f'Median = {median_conf:.3f}')
            ax.set_xlabel('Confidence')
            ax.set_ylabel('Count')
            ax.set_title(f'Distribution of Confidence Values (n={len(filtered_values):,} kept points)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'confidence_histogram.png', dpi=150)
            plt.close()
            logger.info(f"  Saved {self.output_dir / 'confidence_histogram.png'}")
