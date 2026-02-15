#
# File: test_analyzer.py
# Author: Banafshe Bamdad + GitHub Copilot (Claude Sonnet 4.5)
# Email: banafshebamdad@gmail.com
# Created: 2026-02-15 15:01 CET
#

"""Unit tests for analyzer module."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from bb_utils.confidence_decay_analysis.analyzer import (
    AnalysisResult,
    ReservoirSampler,
    ConfidenceDecayAnalyzer,
)
from bb_utils.confidence_decay_analysis.confidence_function import ConfidenceParameters


class TestAnalysisResult:
    """Test AnalysisResult dataclass."""

    def test_initialization(self):
        """Test initialization of AnalysisResult."""
        result = AnalysisResult(
            parameters=ConfidenceParameters(),
            global_stats={'count': 100},
            filtered_stats={'count': 95},
            confidence_stats={'min_confidence': 0.1},
            per_file_stats=[],
            sample_points=None,
            sample_confidences=None,
        )
        assert result.global_stats['count'] == 100
        assert result.filtered_stats['count'] == 95
        assert result.confidence_stats['min_confidence'] == 0.1

    def test_to_json(self):
        """Test JSON serialization."""
        result = AnalysisResult(
            parameters=ConfidenceParameters(x_thr=0.005, lambda_factor=3.0, c_min=0.1),
            global_stats={'count': 100, 'mean': 0.003},
            filtered_stats={'count': 95, 'mean': 0.0025},
            confidence_stats={'min_confidence': 0.1, 'max_confidence': 1.0},
            per_file_stats=[],
            sample_points=np.array([0.001, 0.002, 0.003]),
            sample_confidences=np.array([1.0, 1.0, 1.0]),
        )
        
        json_str = result.to_json()
        data = json.loads(json_str)
        
        assert 'parameters' in data
        assert data['parameters']['x_thr'] == 0.005
        assert data['global_stats']['count'] == 100
        assert len(data['sample_points']) == 3

    def test_to_json_without_samples(self):
        """Test JSON serialization without sample data."""
        result = AnalysisResult(
            parameters=ConfidenceParameters(),
            global_stats={'count': 100},
            filtered_stats={'count': 95},
            confidence_stats={'min_confidence': 0.1},
            per_file_stats=[],
            sample_points=None,
            sample_confidences=None,
        )
        
        json_str = result.to_json()
        data = json.loads(json_str)
        
        assert data['sample_points'] is None
        assert data['sample_confidences'] is None

    def test_to_summary_csv(self):
        """Test CSV summary generation."""
        result = AnalysisResult(
            parameters=ConfidenceParameters(),
            global_stats={'count': 100, 'mean': 0.003, 'std': 0.001},
            filtered_stats={'count': 95, 'mean': 0.0025, 'count_discarded': 5},
            confidence_stats={'min_confidence': 0.1, 'max_confidence': 1.0},
            per_file_stats=[],
            sample_points=None,
            sample_confidences=None,
        )
        
        csv_str = result.to_summary_csv()
        
        # Should be CSV formatted
        lines = csv_str.strip().split('\n')
        assert len(lines) == 2  # Header + data row
        
        # Check header
        header = lines[0].split(',')
        assert 'x_thr' in header
        assert 'global_count' in header
        assert 'filtered_count' in header

    def test_print_summary(self, capsys):
        """Test summary printing."""
        result = AnalysisResult(
            parameters=ConfidenceParameters(),
            global_stats={'count': 100, 'mean': 0.003},
            filtered_stats={'count': 95, 'count_discarded': 5},
            confidence_stats={'min_confidence': 0.1},
            per_file_stats=[],
            sample_points=None,
            sample_confidences=None,
        )
        
        result.print_summary()
        captured = capsys.readouterr()
        
        assert 'Confidence Decay Rate Analysis Summary' in captured.out
        assert 'Parameters:' in captured.out
        assert 'Global Statistics:' in captured.out


class TestReservoirSampler:
    """Test ReservoirSampler class."""

    def test_initialization(self):
        """Test initialization."""
        sampler = ReservoirSampler(max_samples=100)
        assert sampler.max_samples == 100
        assert sampler.count == 0
        assert len(sampler.reservoir_x) == 0
        assert len(sampler.reservoir_c) == 0

    def test_add_first_samples(self):
        """Test adding samples when reservoir not full."""
        sampler = ReservoirSampler(max_samples=5)
        
        x = np.array([1.0, 2.0, 3.0])
        c = np.array([0.9, 0.8, 0.7])
        
        sampler.add(x, c)
        
        assert sampler.count == 3
        assert len(sampler.reservoir_x) == 3
        assert len(sampler.reservoir_c) == 3

    def test_fill_reservoir(self):
        """Test filling reservoir to capacity."""
        sampler = ReservoirSampler(max_samples=5)
        
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        c = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        
        sampler.add(x, c)
        
        assert sampler.count == 5
        assert len(sampler.reservoir_x) == 5
        assert len(sampler.reservoir_c) == 5

    def test_reservoir_overflow(self):
        """Test that reservoir size is maintained on overflow."""
        np.random.seed(42)
        sampler = ReservoirSampler(max_samples=100)
        
        # Add more samples than capacity
        for _ in range(10):
            x = np.random.randn(50)
            c = np.random.rand(50)
            sampler.add(x, c)
        
        # Should still be at max capacity
        assert len(sampler.reservoir_x) == 100
        assert len(sampler.reservoir_c) == 100
        assert sampler.count == 500  # Total count

    def test_get_samples_empty(self):
        """Test getting samples from empty reservoir."""
        sampler = ReservoirSampler(max_samples=100)
        x, c = sampler.get_samples()
        
        assert x is None
        assert c is None

    def test_get_samples_with_data(self):
        """Test getting samples with data."""
        sampler = ReservoirSampler(max_samples=100)
        
        x = np.array([1.0, 2.0, 3.0])
        c = np.array([0.9, 0.8, 0.7])
        sampler.add(x, c)
        
        x_samples, c_samples = sampler.get_samples()
        
        assert len(x_samples) == 3
        assert len(c_samples) == 3
        np.testing.assert_array_equal(x_samples, x)
        np.testing.assert_array_equal(c_samples, c)

    def test_reservoir_sampling_distribution(self):
        """Test that reservoir sampling is approximately uniform."""
        np.random.seed(42)
        sampler = ReservoirSampler(max_samples=1000)
        
        # Add many samples in chunks
        for i in range(100):
            x = np.full(1000, i)  # Each chunk has unique value
            c = np.random.rand(1000)
            sampler.add(x, c)
        
        x_samples, c_samples = sampler.get_samples()
        
        # Should have samples from various chunks
        unique_values = np.unique(x_samples)
        
        # Should have representation from multiple chunks
        # (not all from first 1000 samples)
        assert len(unique_values) > 10


class TestConfidenceDecayAnalyzer:
    """Test ConfidenceDecayAnalyzer class."""

    @pytest.fixture
    def temp_csv_file(self):
        """Create a temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write header
            f.write('frame_id,inv_dist_std,other_column\n')
            # Write data
            for i in range(1000):
                inv_dist_std = 0.001 + i * 0.00001  # Range from 0.001 to 0.011
                f.write(f'{i},{inv_dist_std},dummy\n')
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        Path(temp_path).unlink()

    def test_initialization(self):
        """Test analyzer initialization."""
        params = ConfidenceParameters()
        analyzer = ConfidenceDecayAnalyzer(
            parameters=params,
            column_name='inv_dist_std',
            compute_percentiles=False,
            collect_samples=False,
        )
        
        assert analyzer.parameters == params
        assert analyzer.column_name == 'inv_dist_std'
        assert not analyzer.compute_percentiles
        assert not analyzer.collect_samples

    def test_analyze_files_single_file(self, temp_csv_file):
        """Test analyzing a single CSV file."""
        params = ConfidenceParameters(x_thr=0.005, lambda_factor=3.0, c_min=0.1)
        analyzer = ConfidenceDecayAnalyzer(
            parameters=params,
            column_name='inv_dist_std',
            compute_percentiles=False,
            collect_samples=False,
        )
        
        result = analyzer.analyze_files([temp_csv_file], per_file=False)
        
        assert isinstance(result, AnalysisResult)
        assert result.global_stats['count'] == 1000
        assert result.filtered_stats['count'] <= 1000
        assert result.sample_points is None  # collect_samples=False

    def test_analyze_files_with_per_file_stats(self, temp_csv_file):
        """Test analyzing with per-file statistics."""
        params = ConfidenceParameters()
        analyzer = ConfidenceDecayAnalyzer(
            parameters=params,
            column_name='inv_dist_std',
            compute_percentiles=False,
            collect_samples=False,
        )
        
        result = analyzer.analyze_files([temp_csv_file], per_file=True)
        
        assert len(result.per_file_stats) == 1
        assert result.per_file_stats[0]['filename'] == temp_csv_file

    def test_analyze_files_with_samples(self, temp_csv_file):
        """Test analyzing with sample collection."""
        params = ConfidenceParameters()
        analyzer = ConfidenceDecayAnalyzer(
            parameters=params,
            column_name='inv_dist_std',
            compute_percentiles=False,
            collect_samples=True,
            max_plot_samples=100,
        )
        
        result = analyzer.analyze_files([temp_csv_file], per_file=False)
        
        assert result.sample_points is not None
        assert result.sample_confidences is not None
        assert len(result.sample_points) <= 100

    def test_analyze_files_filtering(self, temp_csv_file):
        """Test that outlier filtering works."""
        # Use very small x_max to filter most data
        params = ConfidenceParameters(x_thr=0.001, lambda_factor=2.0, c_min=0.1)
        analyzer = ConfidenceDecayAnalyzer(
            parameters=params,
            column_name='inv_dist_std',
            compute_percentiles=False,
            collect_samples=False,
        )
        
        result = analyzer.analyze_files([temp_csv_file], per_file=False)
        
        # Some points should be filtered out (inv_dist_std > x_max)
        assert result.filtered_stats['count'] < result.global_stats['count']
        assert result.filtered_stats['count_discarded'] > 0

    def test_analyze_files_nonexistent(self):
        """Test error handling for nonexistent file."""
        params = ConfidenceParameters()
        analyzer = ConfidenceDecayAnalyzer(
            parameters=params,
            column_name='inv_dist_std',
        )
        
        with pytest.raises(Exception):  # Could be FileNotFoundError or pd error
            analyzer.analyze_files(['/nonexistent/file.csv'], per_file=False)

    def test_analyze_files_missing_column(self):
        """Test error handling for missing column."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('frame_id,other_column\n')
            f.write('1,dummy\n')
            temp_path = f.name
        
        try:
            params = ConfidenceParameters()
            analyzer = ConfidenceDecayAnalyzer(
                parameters=params,
                column_name='inv_dist_std',  # This column doesn't exist
            )
            
            with pytest.raises(KeyError):
                analyzer.analyze_files([temp_path], per_file=False)
        finally:
            Path(temp_path).unlink()

    def test_analyze_files_chunking(self, temp_csv_file):
        """Test that chunked reading produces same results."""
        params = ConfidenceParameters()
        
        # Analyze with small chunk size
        analyzer1 = ConfidenceDecayAnalyzer(
            parameters=params,
            column_name='inv_dist_std',
            compute_percentiles=False,
            collect_samples=False,
        )
        result1 = analyzer1.analyze_files([temp_csv_file], per_file=False, chunk_size=100)
        
        # Analyze with large chunk size
        analyzer2 = ConfidenceDecayAnalyzer(
            parameters=params,
            column_name='inv_dist_std',
            compute_percentiles=False,
            collect_samples=False,
        )
        result2 = analyzer2.analyze_files([temp_csv_file], per_file=False, chunk_size=10000)
        
        # Results should match
        assert result1.global_stats['count'] == result2.global_stats['count']
        assert np.isclose(result1.global_stats['mean'], result2.global_stats['mean'])


class TestAnalyzerIntegration:
    """Integration tests for the analyzer."""

    def test_full_workflow(self):
        """Test complete analysis workflow."""
        # Create test data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('frame_id,inv_dist_std\n')
            np.random.seed(42)
            data = np.abs(np.random.randn(10000) * 0.003)
            for i, val in enumerate(data):
                f.write(f'{i},{val}\n')
            temp_path = f.name
        
        try:
            # Analyze
            params = ConfidenceParameters(x_thr=0.005, lambda_factor=3.0, c_min=0.1)
            analyzer = ConfidenceDecayAnalyzer(
                parameters=params,
                column_name='inv_dist_std',
                compute_percentiles=False,
                collect_samples=True,
                max_plot_samples=1000,
            )
            
            result = analyzer.analyze_files([temp_path], per_file=True)
            
            # Verify result
            assert result.global_stats['count'] == 10000
            assert result.sample_points is not None
            assert len(result.per_file_stats) == 1
            
            # Test JSON export
            json_str = result.to_json()
            data = json.loads(json_str)
            assert data['global_stats']['count'] == 10000
            
            # Test CSV summary
            csv_str = result.to_summary_csv()
            assert 'x_thr' in csv_str
            assert 'global_count' in csv_str
            
        finally:
            Path(temp_path).unlink()
