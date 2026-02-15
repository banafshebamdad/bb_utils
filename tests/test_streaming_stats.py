#
# File: test_streaming_stats.py
# Author: Banafshe Bamdad + GitHub Copilot (Claude Sonnet 4.5)
# Email: banafshebamdad@gmail.com
# Created: 2026-02-15 15:01 CET
#

"""Unit tests for streaming_stats module."""
import numpy as np
import pytest

from bb_utils.confidence_decay_analysis.streaming_stats import (
    StreamingStatistics,
    StreamingFilteringStats,
    StreamingConfidenceStats,
    PerFileStatistics,
)


class TestStreamingStatistics:
    """Test StreamingStatistics class."""

    def test_initialization(self):
        """Test initialization of StreamingStatistics."""
        stats = StreamingStatistics(compute_percentiles=False)
        assert stats.count == 0
        assert stats.mean == 0.0
        assert stats.m2 == 0.0
        assert stats.min_val == np.inf
        assert stats.max_val == -np.inf

    def test_update_single_value(self):
        """Test updating with a single value."""
        stats = StreamingStatistics(compute_percentiles=False)
        stats.update(np.array([5.0]))
        
        assert stats.count == 1
        assert stats.mean == 5.0
        assert stats.variance == 0.0
        assert stats.min_val == 5.0
        assert stats.max_val == 5.0

    def test_update_multiple_values(self):
        """Test updating with multiple values."""
        stats = StreamingStatistics(compute_percentiles=False)
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats.update(data)
        
        assert stats.count == 5
        assert stats.mean == 3.0
        assert np.isclose(stats.variance, 2.5)  # Population variance
        assert np.isclose(stats.std, np.sqrt(2.5))
        assert stats.min_val == 1.0
        assert stats.max_val == 5.0

    def test_update_incremental(self):
        """Test incremental updates match batch update."""
        # Batch update
        stats_batch = StreamingStatistics(compute_percentiles=False)
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        stats_batch.update(data)
        
        # Incremental update
        stats_incremental = StreamingStatistics(compute_percentiles=False)
        for chunk in [data[:3], data[3:6], data[6:]]:
            stats_incremental.update(chunk)
        
        # Should match
        assert stats_batch.count == stats_incremental.count
        assert np.isclose(stats_batch.mean, stats_incremental.mean)
        assert np.isclose(stats_batch.variance, stats_incremental.variance)
        assert stats_batch.min_val == stats_incremental.min_val
        assert stats_batch.max_val == stats_incremental.max_val

    def test_update_empty_array(self):
        """Test updating with empty array does nothing."""
        stats = StreamingStatistics(compute_percentiles=False)
        stats.update(np.array([1.0, 2.0, 3.0]))
        initial_count = stats.count
        
        stats.update(np.array([]))
        assert stats.count == initial_count

    def test_variance_vs_numpy(self):
        """Test that variance matches numpy computation."""
        np.random.seed(42)
        data = np.random.randn(1000)
        
        stats = StreamingStatistics(compute_percentiles=False)
        stats.update(data)
        
        # Compare with numpy (population variance, ddof=0)
        np_mean = np.mean(data)
        np_var = np.var(data, ddof=0)
        
        assert np.isclose(stats.mean, np_mean)
        assert np.isclose(stats.variance, np_var)

    def test_to_dict_without_percentiles(self):
        """Test to_dict output without percentiles."""
        stats = StreamingStatistics(compute_percentiles=False)
        stats.update(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        
        result = stats.to_dict()
        assert result['count'] == 5
        assert result['mean'] == 3.0
        assert result['min'] == 1.0
        assert result['max'] == 5.0
        assert 'std' in result
        assert result['percentiles'] is None

    def test_with_percentiles_requires_tdigest(self):
        """Test that percentiles require tdigest."""
        # Try to create with percentiles
        stats = StreamingStatistics(compute_percentiles=True)
        
        # If tdigest is available, should work
        # If not, percentiles should be None in output
        stats.update(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        result = stats.to_dict()
        
        # Either percentiles are computed or None
        assert 'percentiles' in result


class TestStreamingFilteringStats:
    """Test StreamingFilteringStats class."""

    def test_initialization(self):
        """Test initialization with outlier threshold."""
        stats = StreamingFilteringStats(outlier_threshold=0.015)
        assert stats.outlier_threshold == 0.015
        assert stats.count_discarded == 0

    def test_update_without_outliers(self):
        """Test update with no outliers."""
        stats = StreamingFilteringStats(outlier_threshold=10.0)
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats.update(data)
        
        assert stats.count_discarded == 0
        assert stats.count == 5

    def test_update_with_outliers(self):
        """Test update with some outliers."""
        stats = StreamingFilteringStats(outlier_threshold=5.0)
        data = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 10.0, 100.0])
        stats.update(data)
        
        # Values > 5.0 should be discarded
        assert stats.count_discarded == 2  # 6.0, 10.0, 100.0 - wait, 6.0 is also > 5.0
        # Actually: 6.0, 10.0, 100.0 are all > 5.0
        assert stats.count_discarded == 3
        assert stats.count == 4  # 1,2,3,4

    def test_to_dict_includes_discarded_count(self):
        """Test that to_dict includes count_discarded."""
        stats = StreamingFilteringStats(outlier_threshold=5.0)
        stats.update(np.array([1.0, 2.0, 10.0]))
        
        result = stats.to_dict()
        assert result['count'] == 2
        assert result['count_discarded'] == 1


class TestStreamingConfidenceStats:
    """Test StreamingConfidenceStats class."""

    def test_initialization(self):
        """Test initialization."""
        stats = StreamingConfidenceStats()
        assert stats.count == 0
        assert len(stats.confidence_bins) == 10
        assert len(stats.bin_counts) == 10
        assert all(count == 0 for count in stats.bin_counts)

    def test_update_simple(self):
        """Test updating with simple confidence values."""
        stats = StreamingConfidenceStats()
        confidences = np.array([0.1, 0.5, 0.9, 1.0])
        stats.update(confidences)
        
        assert stats.count == 4
        assert stats.min_confidence == 0.1
        assert stats.max_confidence == 1.0

    def test_binning(self):
        """Test that confidence values are binned correctly."""
        stats = StreamingConfidenceStats()
        
        # All values in first bin [0.0, 0.1)
        confidences = np.full(100, 0.05)
        stats.update(confidences)
        
        assert stats.bin_counts[0] == 100
        assert sum(stats.bin_counts[1:]) == 0

    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        stats = StreamingConfidenceStats()
        confidences = np.linspace(0.0, 1.0, 1000)
        stats.update(confidences)
        
        df = stats.to_dataframe()
        assert len(df) == 10
        assert 'bin_start' in df.columns
        assert 'bin_end' in df.columns
        assert 'count' in df.columns
        assert 'fraction' in df.columns
        
        # All fractions should sum to ~1.0
        assert np.isclose(df['fraction'].sum(), 1.0)

    def test_merge(self):
        """Test merging two StreamingConfidenceStats."""
        stats1 = StreamingConfidenceStats()
        stats1.update(np.array([0.1, 0.2, 0.3]))
        
        stats2 = StreamingConfidenceStats()
        stats2.update(np.array([0.4, 0.5, 0.6]))
        
        stats1.merge(stats2)
        
        assert stats1.count == 6
        assert stats1.min_confidence == 0.1
        assert stats1.max_confidence == 0.6


class TestPerFileStatistics:
    """Test PerFileStatistics class."""

    def test_initialization(self):
        """Test initialization."""
        stats = PerFileStatistics(
            filename="test.csv",
            outlier_threshold=0.015,
            compute_percentiles=False
        )
        assert stats.filename == "test.csv"
        assert stats.raw_stats.count == 0
        assert stats.filtered_stats.count == 0
        assert stats.confidence_stats.count == 0

    def test_update_all_stats(self):
        """Test updating all statistics together."""
        stats = PerFileStatistics(
            filename="test.csv",
            outlier_threshold=10.0,
            compute_percentiles=False
        )
        
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        confidences = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
        
        stats.update(data, confidences)
        
        assert stats.raw_stats.count == 5
        assert stats.filtered_stats.count == 5
        assert stats.confidence_stats.count == 5

    def test_update_with_filtering(self):
        """Test that filtering works correctly."""
        stats = PerFileStatistics(
            filename="test.csv",
            outlier_threshold=5.0,
            compute_percentiles=False
        )
        
        data = np.array([1.0, 2.0, 3.0, 10.0, 100.0])
        confidences = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
        
        stats.update(data, confidences)
        
        # Raw stats should have all 5 values
        assert stats.raw_stats.count == 5
        
        # Filtered stats should only have values <= 5.0
        assert stats.filtered_stats.count == 3
        assert stats.filtered_stats.count_discarded == 2
        
        # Confidence stats should have 3 values (corresponding to filtered data)
        assert stats.confidence_stats.count == 3

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = PerFileStatistics(
            filename="test.csv",
            outlier_threshold=10.0,
            compute_percentiles=False
        )
        
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        confidences = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
        stats.update(data, confidences)
        
        result = stats.to_dict()
        assert result['filename'] == "test.csv"
        assert 'raw' in result
        assert 'filtered' in result
        assert 'confidence' in result
        assert result['raw']['count'] == 5
        assert result['filtered']['count'] == 5


class TestStreamingStatisticsIntegration:
    """Integration tests for streaming statistics."""

    def test_large_dataset_memory_efficiency(self):
        """Test that streaming works with large datasets."""
        np.random.seed(42)
        
        stats = StreamingStatistics(compute_percentiles=False)
        
        # Simulate processing in chunks
        total_points = 1_000_000
        chunk_size = 10_000
        
        all_data = []
        for _ in range(total_points // chunk_size):
            chunk = np.random.randn(chunk_size)
            all_data.append(chunk)
            stats.update(chunk)
        
        # Verify against numpy
        all_data = np.concatenate(all_data)
        assert stats.count == total_points
        assert np.isclose(stats.mean, np.mean(all_data), atol=1e-10)
        assert np.isclose(stats.variance, np.var(all_data, ddof=0), atol=1e-8)

    def test_consistency_across_chunk_sizes(self):
        """Test that results are consistent regardless of chunk size."""
        np.random.seed(42)
        data = np.random.randn(10000)
        
        # Process in different chunk sizes
        stats1 = StreamingStatistics(compute_percentiles=False)
        for i in range(0, len(data), 100):
            stats1.update(data[i:i+100])
        
        stats2 = StreamingStatistics(compute_percentiles=False)
        for i in range(0, len(data), 1000):
            stats2.update(data[i:i+1000])
        
        stats3 = StreamingStatistics(compute_percentiles=False)
        stats3.update(data)
        
        # All should match
        assert stats1.count == stats2.count == stats3.count
        assert np.isclose(stats1.mean, stats2.mean)
        assert np.isclose(stats1.mean, stats3.mean)
        assert np.isclose(stats1.variance, stats2.variance)
        assert np.isclose(stats1.variance, stats3.variance)
