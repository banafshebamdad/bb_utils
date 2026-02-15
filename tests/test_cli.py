#
# File: test_cli.py
# Author: Banafshe Bamdad + GitHub Copilot (Claude Sonnet 4.5)
# Email: banafshebamdad@gmail.com
# Created: 2026-02-15 15:01 CET
#

"""Unit tests for CLI module."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import sys

import pytest

from bb_utils.confidence_decay_analysis.cli import (
    discover_files,
    main,
)
from bb_utils.confidence_decay_analysis.confidence_function import ConfidenceParameters


class TestDiscoverFiles:
    """Test file discovery function."""

    @pytest.fixture
    def temp_dir_with_files(self):
        """Create temporary directory with test files."""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        # Create some test files
        (temp_path / 'file1_semidense_points.csv.gz').touch()
        (temp_path / 'file2_semidense_points.csv.gz').touch()
        (temp_path / 'other_file.csv').touch()
        
        # Create subdirectory with file
        sub_dir = temp_path / 'subdir'
        sub_dir.mkdir()
        (sub_dir / 'file3_semidense_points.csv.gz').touch()
        
        yield temp_path
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

    def test_discover_single_file(self):
        """Test discovering a single file."""
        with tempfile.NamedTemporaryFile(suffix='_semidense_points.csv.gz', delete=False) as f:
            temp_path = f.name
        
        try:
            files = discover_files([temp_path])
            assert len(files) == 1
            assert files[0] == temp_path
        finally:
            Path(temp_path).unlink()

    def test_discover_directory(self, temp_dir_with_files):
        """Test discovering files in a directory."""
        files = discover_files([str(temp_dir_with_files)])
        
        # Should find the 3 semidense_points files (including subdirectory)
        assert len(files) == 3
        assert all('semidense_points.csv.gz' in str(f) for f in files)

    def test_discover_mixed_inputs(self, temp_dir_with_files):
        """Test discovering with mixed file and directory inputs."""
        with tempfile.NamedTemporaryFile(suffix='_semidense_points.csv.gz', delete=False) as f:
            temp_file = f.name
        
        try:
            files = discover_files([temp_file, str(temp_dir_with_files)])
            
            # Should find the explicit file + 3 from directory
            assert len(files) == 4
        finally:
            Path(temp_file).unlink()

    def test_discover_nonexistent_raises_error(self):
        """Test that nonexistent paths raise an error."""
        with pytest.raises(FileNotFoundError):
            discover_files(['/nonexistent/path'])

    def test_discover_no_matching_files(self):
        """Test directory with no matching files raises error."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create a file that doesn't match the pattern
            Path(temp_dir) / 'other.csv'.touch()
            
            with pytest.raises(ValueError, match="No.*semidense_points.csv.gz"):
                discover_files([temp_dir])
        finally:
            import shutil
            shutil.rmtree(temp_dir)


class TestParameterValidation:
    """Test parameter validation through ConfidenceParameters."""

    def test_valid_parameters(self):
        """Test that valid parameters pass validation."""
        # Should not raise
        params = ConfidenceParameters(x_thr=0.005, lambda_factor=3.0, c_min=0.1)
        assert params is not None

    def test_invalid_x_thr(self):
        """Test that invalid x_thr raises error."""
        with pytest.raises(ValueError, match="x_thr must be positive"):
            ConfidenceParameters(x_thr=0.0, lambda_factor=3.0, c_min=0.1)
        
        with pytest.raises(ValueError, match="x_thr must be positive"):
            ConfidenceParameters(x_thr=-0.01, lambda_factor=3.0, c_min=0.1)

    def test_invalid_lambda_factor(self):
        """Test that invalid lambda_factor raises error."""
        with pytest.raises(ValueError, match="lambda_factor must be > 1"):
            ConfidenceParameters(x_thr=0.005, lambda_factor=1.0, c_min=0.1)
        
        with pytest.raises(ValueError, match="lambda_factor must be > 1"):
            ConfidenceParameters(x_thr=0.005, lambda_factor=0.5, c_min=0.1)

    def test_invalid_c_min(self):
        """Test that invalid c_min raises error."""
        with pytest.raises(ValueError, match="c_min must be in"):
            ConfidenceParameters(x_thr=0.005, lambda_factor=3.0, c_min=0.0)
        
        with pytest.raises(ValueError, match="c_min must be in"):
            ConfidenceParameters(x_thr=0.005, lambda_factor=3.0, c_min=1.0)


class TestMainFunction:
    """Test main CLI function."""

    @pytest.fixture
    def temp_csv_file(self):
        """Create a temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='_semidense_points.csv.gz', delete=False) as f:
            f.write('frame_id,inv_dist_std\n')
            for i in range(100):
                f.write(f'{i},{0.001 + i * 0.0001}\n')
            temp_path = f.name
        
        yield temp_path
        
        Path(temp_path).unlink()

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        
        import shutil
        shutil.rmtree(temp_dir)

    def test_main_basic_execution(self, temp_csv_file, temp_output_dir):
        """Test basic CLI execution."""
        args = [
            'bb-analyze-decay-rate',
            '--input', temp_csv_file,
            '--output-dir', temp_output_dir,
        ]
        
        with patch.object(sys, 'argv', args):
            # Should not raise
            exit_code = main()
            assert exit_code == 0
        
        # Check output files were created
        output_path = Path(temp_output_dir)
        assert (output_path / 'analysis_results.json').exists()
        assert (output_path / 'summary.csv').exists()

    def test_main_with_custom_parameters(self, temp_csv_file, temp_output_dir):
        """Test CLI with custom parameters."""
        args = [
            'bb-analyze-decay-rate',
            '--input', temp_csv_file,
            '--output-dir', temp_output_dir,
            '--x-thr', '0.01',
            '--lambda', '2.5',
            '--c-min', '0.15',
        ]
        
        with patch.object(sys, 'argv', args):
            exit_code = main()
            assert exit_code == 0
        
        # Verify parameters were used
        with open(Path(temp_output_dir) / 'analysis_results.json') as f:
            data = json.load(f)
            assert data['parameters']['x_thr'] == 0.01
            assert data['parameters']['lambda_factor'] == 2.5
            assert data['parameters']['c_min'] == 0.15

    def test_main_with_per_file(self, temp_csv_file, temp_output_dir):
        """Test CLI with per-file statistics."""
        args = [
            'bb-analyze-decay-rate',
            '--input', temp_csv_file,
            '--output-dir', temp_output_dir,
            '--per-file',
        ]
        
        with patch.object(sys, 'argv', args):
            exit_code = main()
            assert exit_code == 0
        
        # Check per-file results
        with open(Path(temp_output_dir) / 'analysis_results.json') as f:
            data = json.load(f)
            assert 'per_file_stats' in data
            assert len(data['per_file_stats']) > 0

    def test_main_with_verbose(self, temp_csv_file, temp_output_dir, capsys):
        """Test CLI with verbose output."""
        args = [
            'bb-analyze-decay-rate',
            '--input', temp_csv_file,
            '--output-dir', temp_output_dir,
            '--verbose',
        ]
        
        with patch.object(sys, 'argv', args):
            exit_code = main()
            assert exit_code == 0
        
        captured = capsys.readouterr()
        # Should have verbose logging output
        assert len(captured.out) > 0 or len(captured.err) > 0

    def test_main_invalid_parameters(self, temp_csv_file, temp_output_dir):
        """Test that invalid parameters cause early exit."""
        args = [
            'bb-analyze-decay-rate',
            '--input', temp_csv_file,
            '--output-dir', temp_output_dir,
            '--x-thr', '-0.01',  # Invalid (negative)
        ]
        
        with patch.object(sys, 'argv', args):
            exit_code = main()
            assert exit_code == 1

    def test_main_nonexistent_input(self, temp_output_dir):
        """Test that nonexistent input files cause error."""
        args = [
            'bb-analyze-decay-rate',
            '--input', '/nonexistent/file.csv.gz',
            '--output-dir', temp_output_dir,
        ]
        
        with patch.object(sys, 'argv', args):
            exit_code = main()
            assert exit_code == 1

    def test_main_creates_output_directory(self, temp_csv_file):
        """Test that output directory is created if it doesn't exist."""
        temp_dir = tempfile.mkdtemp()
        import shutil
        shutil.rmtree(temp_dir)  # Remove it
        
        output_dir = Path(temp_dir) / 'new_output'
        
        args = [
            'bb-analyze-decay-rate',
            '--input', temp_csv_file,
            '--output-dir', str(output_dir),
        ]
        
        try:
            with patch.object(sys, 'argv', args):
                exit_code = main()
                assert exit_code == 0
            
            assert output_dir.exists()
            assert (output_dir / 'analysis_results.json').exists()
        finally:
            if output_dir.exists():
                shutil.rmtree(output_dir)

    def test_main_percentiles_without_tdigest(self, temp_csv_file, temp_output_dir):
        """Test that --percentiles flag works (may warn if tdigest not available)."""
        args = [
            'bb-analyze-decay-rate',
            '--input', temp_csv_file,
            '--output-dir', temp_output_dir,
            '--percentiles',
        ]
        
        with patch.object(sys, 'argv', args):
            # Should not crash even if tdigest not available
            exit_code = main()
            assert exit_code == 0

    def test_main_plot_without_matplotlib(self, temp_csv_file, temp_output_dir):
        """Test that --plot flag works (may warn if matplotlib not available)."""
        args = [
            'bb-analyze-decay-rate',
            '--input', temp_csv_file,
            '--output-dir', temp_output_dir,
            '--plot',
        ]
        
        with patch.object(sys, 'argv', args):
            # Should not crash even if matplotlib not available
            exit_code = main()
            assert exit_code == 0


class TestCLIIntegration:
    """Integration tests for CLI."""

    def test_full_cli_workflow(self):
        """Test complete CLI workflow from end to end."""
        # Create test data
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        # Create multiple CSV files
        for i in range(3):
            with open(temp_path / f'file{i}_semidense_points.csv.gz', 'w') as f:
                f.write('frame_id,inv_dist_std\n')
                for j in range(100):
                    f.write(f'{j},{0.001 + j * 0.0001}\n')
        
        output_dir = temp_path / 'output'
        
        args = [
            'bb-analyze-decay-rate',
            '--input', str(temp_path),
            '--output-dir', str(output_dir),
            '--per-file',
            '--verbose',
        ]
        
        try:
            with patch.object(sys, 'argv', args):
                exit_code = main()
                assert exit_code == 0
            
            # Verify all outputs
            assert (output_dir / 'analysis_results.json').exists()
            assert (output_dir / 'summary.csv').exists()
            
            # Check JSON content
            with open(output_dir / 'analysis_results.json') as f:
                data = json.load(f)
                assert data['global_stats']['count'] == 300  # 3 files × 100 rows
                assert len(data['per_file_stats']) == 3
            
            # Check CSV content
            with open(output_dir / 'summary.csv') as f:
                csv_content = f.read()
                assert 'x_thr' in csv_content
                assert 'global_count' in csv_content
        
        finally:
            import shutil
            shutil.rmtree(temp_dir)
