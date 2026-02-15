#
# File: streaming_stats.py
# Author: Banafshe Bamdad + GitHub Copilot (Claude Sonnet 4.5)
# Email: banafshebamdad@gmail.com
# Created: 2026-02-15
#

"""
Streaming statistics computation with constant memory usage.

Optional dependency: tdigest (only required if percentiles are requested)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import tdigest - only required if percentiles requested
try:
    from tdigest import TDigest
    TDIGEST_AVAILABLE = True
except ImportError:
    TDIGEST_AVAILABLE = False
    TDigest = None
    logger.debug("tdigest not available - percentiles will be disabled")


class StreamingStatistics:
    """Compute statistics in a single pass without storing all data.
    
    Uses vectorized Welford's algorithm for mean/variance.
    Optionally uses t-digest for approximate percentiles (requires tdigest package).
    
    Args:
        percentiles: List of percentiles to compute (e.g., [25, 50, 75, 90, 95, 99])
                     Requires tdigest package. If None or tdigest unavailable, skips percentiles.
    
    Example:
        >>> stats = StreamingStatistics(percentiles=[25, 50, 75, 95])
        >>> for chunk in data_chunks:
        ...     stats.update(chunk)
        >>> results = stats.finalize()
    """
    
    def __init__(self, percentiles: Optional[List[float]] = None):
        self.n_total = 0
        self.n_valid = 0
        self.n_nan = 0
        self.n_inf = 0
        self.n_negative = 0
        
        # Vectorized Welford's algorithm state
        self.n_for_mean = 0
        self.mean_accumulator = 0.0
        self.m2_accumulator = 0.0
        
        # Min/max tracking
        self.min_value = float('inf')
        self.max_value = float('-inf')
        
        # Optional percentiles
        self.percentiles = percentiles
        self.use_percentiles = (percentiles is not None and 
                               len(percentiles) > 0 and 
                               TDIGEST_AVAILABLE)
        
        if self.percentiles is not None and not TDIGEST_AVAILABLE:
            logger.warning(
                "Percentiles requested but tdigest not available. "
                "Install with: pip install tdigest"
            )
        
        self.digest = TDigest() if self.use_percentiles else None
    
    def update(self, values: np.ndarray) -> None:
        """Update statistics with new batch of values.
        
        Uses vectorized operations - no Python loops over elements.
        
        Args:
            values: Array of values to process
        """
        # Data quality checks (vectorized)
        self.n_total += len(values)
        self.n_nan += int(np.isnan(values).sum())
        self.n_inf += int(np.isinf(values).sum())
        self.n_negative += int((values < 0).sum())
        
        # Extract valid values
        valid = values[(np.isfinite(values)) & (values >= 0)]
        n_valid_batch = len(valid)
        
        if n_valid_batch == 0:
            return
        
        self.n_valid += n_valid_batch
        
        # Update min/max
        self.min_value = min(self.min_value, float(valid.min()))
        self.max_value = max(self.max_value, float(valid.max()))
        
        # Vectorized Welford's algorithm for running mean and variance
        # Reference: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        # Parallel algorithm variant for batches
        
        if self.n_for_mean == 0:
            # First batch - initialize
            self.mean_accumulator = float(valid.mean())
            self.m2_accumulator = float(((valid - self.mean_accumulator) ** 2).sum())
            self.n_for_mean = n_valid_batch
        else:
            # Combine with existing statistics using parallel Welford algorithm
            batch_mean = float(valid.mean())
            batch_m2 = float(((valid - batch_mean) ** 2).sum())
            
            # Combine
            n_combined = self.n_for_mean + n_valid_batch
            delta = batch_mean - self.mean_accumulator
            
            self.mean_accumulator = (self.n_for_mean * self.mean_accumulator + 
                                    n_valid_batch * batch_mean) / n_combined
            
            self.m2_accumulator = (self.m2_accumulator + batch_m2 + 
                                  delta ** 2 * self.n_for_mean * n_valid_batch / n_combined)
            
            self.n_for_mean = n_combined
        
        # Update percentile digest (if enabled)
        if self.use_percentiles:
            self.digest.batch_update(valid)
    
    def finalize(self) -> dict:
        """Compute final statistics.
        
        Returns:
            dict: Statistics dictionary with data_quality and distribution sections
        """
        if self.n_for_mean > 1:
            variance = self.m2_accumulator / (self.n_for_mean - 1)
            std = np.sqrt(variance)
        else:
            variance = 0.0
            std = 0.0
        
        result = {
            'data_quality': {
                'n_total_rows_scanned': int(self.n_total),
                'n_valid': int(self.n_valid),
                'n_nan': int(self.n_nan),
                'n_inf': int(self.n_inf),
                'n_negative': int(self.n_negative)
            },
            'distribution': {
                'min': float(self.min_value) if self.n_valid > 0 else np.nan,
                'max': float(self.max_value) if self.n_valid > 0 else np.nan,
                'mean': float(self.mean_accumulator) if self.n_valid > 0 else np.nan,
                'std': float(std)
            }
        }
        
        # Add percentiles if available
        if self.use_percentiles and self.n_valid > 0:
            result['distribution']['median'] = float(self.digest.percentile(50))
            result['distribution']['percentiles'] = {
                f'p{p}': float(self.digest.percentile(p))
                for p in self.percentiles
            }
        else:
            result['distribution']['median'] = np.nan
            result['distribution']['percentiles'] = {
                f'p{p}': np.nan for p in (self.percentiles or [])
            }
        
        return result


class StreamingFilteringStats:
    """Track filtering statistics in streaming fashion.
    
    Counts points filtered out (x > x_max) without storing data.
    
    Args:
        x_max: Outlier cutoff threshold
    
    Example:
        >>> filter_stats = StreamingFilteringStats(x_max=0.015)
        >>> for chunk in data_chunks:
        ...     filter_stats.update(chunk)
        >>> report = filter_stats.finalize()
    """
    
    def __init__(self, x_max: float):
        self.x_max = x_max
        self.n_total = 0
        self.n_removed = 0
        self.n_kept = 0
    
    def update(self, values: np.ndarray) -> None:
        """Update filtering counts with new batch of values (vectorized).
        
        Args:
            values: Raw inv_dist_std values (including NaN, inf, etc.)
        """
        # Only count valid values for filtering
        valid = values[(np.isfinite(values)) & (values >= 0)]
        n_valid = len(valid)
        self.n_total += n_valid
        
        # Count outliers (vectorized)
        n_outliers = int((valid > self.x_max).sum())
        self.n_removed += n_outliers
        self.n_kept += (n_valid - n_outliers)
    
    def finalize(self) -> dict:
        """Compute final filtering report.
        
        Returns:
            dict: Filtering statistics
        """
        percentage_removed = (100 * self.n_removed / self.n_total) if self.n_total > 0 else 0.0
        
        return {
            'n_removed_outliers': int(self.n_removed),
            'n_kept': int(self.n_kept),
            'percentage_removed': float(percentage_removed)
        }


class StreamingConfidenceStats:
    """Compute confidence distribution in streaming fashion.
    
    Given a confidence function, computes statistics on confidence values
    for kept points (x <= x_max) without storing all data.
    
    Uses vectorized operations for efficiency.
    
    Args:
        confidence_func: MPSConfidenceFunction instance
        use_percentiles: Whether to compute median via tdigest
    
    Example:
        >>> conf_stats = StreamingConfidenceStats(confidence_func, use_percentiles=True)
        >>> for chunk in data_chunks:
        ...     conf_stats.update(chunk)
        >>> distribution = conf_stats.finalize()
    """
    
    def __init__(self, confidence_func, use_percentiles: bool = False):
        self.confidence_func = confidence_func
        self.use_percentiles = use_percentiles and TDIGEST_AVAILABLE
        
        # Streaming stats for confidence values
        self.n_values = 0
        self.mean_accumulator = 0.0
        self.m2_accumulator = 0.0
        self.min_confidence = float('inf')
        self.max_confidence = float('-inf')
        
        # Optional: t-digest for median
        self.digest = TDigest() if self.use_percentiles else None
    
    def update(self, inv_dist_std_values: np.ndarray) -> None:
        """Update confidence statistics with new batch (vectorized).
        
        Args:
            inv_dist_std_values: Raw inv_dist_std values to process
        """
        # Only process valid values
        valid = inv_dist_std_values[(np.isfinite(inv_dist_std_values)) & 
                                     (inv_dist_std_values >= 0)]
        
        # Filter to kept points only (x <= x_max)
        kept = valid[valid <= self.confidence_func.params.x_max]
        
        if len(kept) == 0:
            return
        
        # Compute confidence for kept points (vectorized)
        confidences = self.confidence_func.compute_confidence(kept, strict=False)
        
        n_batch = len(confidences)
        
        # Update min/max
        self.min_confidence = min(self.min_confidence, float(confidences.min()))
        self.max_confidence = max(self.max_confidence, float(confidences.max()))
        
        # Vectorized Welford's algorithm for batched update
        if self.n_values == 0:
            # First batch
            self.mean_accumulator = float(confidences.mean())
            self.m2_accumulator = float(((confidences - self.mean_accumulator) ** 2).sum())
            self.n_values = n_batch
        else:
            # Combine with existing
            batch_mean = float(confidences.mean())
            batch_m2 = float(((confidences - batch_mean) ** 2).sum())
            
            n_combined = self.n_values + n_batch
            delta = batch_mean - self.mean_accumulator
            
            self.mean_accumulator = (self.n_values * self.mean_accumulator + 
                                    n_batch * batch_mean) / n_combined
            
            self.m2_accumulator = (self.m2_accumulator + batch_m2 + 
                                  delta ** 2 * self.n_values * n_batch / n_combined)
            
            self.n_values = n_combined
        
        # Update digest for median (if enabled)
        if self.use_percentiles:
            self.digest.batch_update(confidences)
    
    def finalize(self) -> dict:
        """Compute final confidence distribution.
        
        Returns:
            dict: Confidence distribution statistics
        """
        if self.n_values == 0:
            return {
                'confidence_min': np.nan,
                'confidence_max': np.nan,
                'confidence_mean': np.nan,
                'confidence_median': np.nan
            }
        
        result = {
            'confidence_min': float(self.min_confidence),
            'confidence_max': float(self.max_confidence),
            'confidence_mean': float(self.mean_accumulator)
        }
        
        if self.use_percentiles:
            result['confidence_median'] = float(self.digest.percentile(50))
        else:
            result['confidence_median'] = np.nan
        
        return result


class PerFileStatistics:
    """Track per-file statistics without storing all data.
    
    Args:
        percentiles: List of percentiles to compute (optional)
        confidence_func: Confidence function for filtering/confidence stats (optional)
        use_percentiles: Whether to compute percentiles
    
    Example:
        >>> per_file = PerFileStatistics(percentiles=[25, 50, 75, 95], 
        ...                              confidence_func=conf_func,
        ...                              use_percentiles=True)
        >>> for filepath in file_paths:
        ...     per_file.process_file(filepath, 'inv_dist_std', chunk_size=100000)
        >>> df = per_file.to_dataframe()
    """
    
    def __init__(self, percentiles: Optional[List[float]] = None, 
                 confidence_func=None,
                 use_percentiles: bool = False):
        self.file_stats = {}
        self.percentiles = percentiles
        self.confidence_func = confidence_func
        self.use_percentiles = use_percentiles
    
    def process_file(self, filepath: Path, column: str, chunk_size: int) -> None:
        """Process single file in chunks, accumulate statistics.
        
        Args:
            filepath: Path to CSV.gz file
            column: Column name to analyze
            chunk_size: Number of rows per chunk
        """
        # Initialize streaming accumulators for this file
        stats_tracker = StreamingStatistics(self.percentiles if self.use_percentiles else None)
        
        if self.confidence_func is not None:
            filtering_tracker = StreamingFilteringStats(self.confidence_func.params.x_max)
            confidence_tracker = StreamingConfidenceStats(self.confidence_func, self.use_percentiles)
        
        # Read CSV in chunks
        for chunk in pd.read_csv(filepath, compression='gzip', chunksize=chunk_size):
            if column not in chunk.columns:
                raise ValueError(f"Column '{column}' not found in {filepath}")
            
            values = chunk[column].values
            stats_tracker.update(values)
            
            if self.confidence_func is not None:
                filtering_tracker.update(values)
                confidence_tracker.update(values)
        
        # Finalize statistics for this file
        file_result = {
            'file': filepath.name,
            **self._flatten_dict(stats_tracker.finalize())
        }
        
        if self.confidence_func is not None:
            file_result.update(self._flatten_dict({
                'filtering': filtering_tracker.finalize(),
                'confidence_distribution': confidence_tracker.finalize()
            }))
        
        self.file_stats[filepath.name] = file_result
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for CSV export.
        
        Returns:
            pd.DataFrame: Per-file statistics
        """
        return pd.DataFrame(list(self.file_stats.values()))
    
    @staticmethod
    def _flatten_dict(d: dict, parent_key: str = '', sep: str = '_') -> dict:
        """Flatten nested dictionary.
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key for recursion
            sep: Separator for nested keys
            
        Returns:
            dict: Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(PerFileStatistics._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
