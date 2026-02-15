#
# File: test_confidence_function.py
# Author: Banafshe Bamdad + GitHub Copilot (Claude Sonnet 4.5)
# Email: banafshebamdad@gmail.com
# Created: 2026-02-15 15:01 CET
#

"""Unit tests for confidence_function module."""
import numpy as np
import pytest

from bb_utils.confidence_decay_analysis.confidence_function import (
    ConfidenceParameters,
    MPSConfidenceFunction,
)


class TestConfidenceParameters:
    """Test ConfidenceParameters dataclass."""

    def test_default_parameters(self):
        """Test default parameter values."""
        params = ConfidenceParameters()
        assert params.x_thr == 0.005
        assert params.lambda_factor == 3.0
        assert params.c_min == 0.1
        assert params.x_max == 0.015  # 3.0 * 0.005
        # alpha = -ln(0.1) / (0.015 - 0.005) = -ln(0.1) / 0.01
        expected_alpha = -np.log(0.1) / 0.01
        assert np.isclose(params.alpha, expected_alpha)

    def test_custom_parameters(self):
        """Test custom parameter values."""
        params = ConfidenceParameters(x_thr=0.01, lambda_factor=2.0, c_min=0.2)
        assert params.x_thr == 0.01
        assert params.lambda_factor == 2.0
        assert params.c_min == 0.2
        assert params.x_max == 0.02  # 2.0 * 0.01
        expected_alpha = -np.log(0.2) / 0.01
        assert np.isclose(params.alpha, expected_alpha)

    def test_invalid_x_thr(self):
        """Test that invalid x_thr raises ValueError."""
        with pytest.raises(ValueError, match="x_thr must be positive"):
            ConfidenceParameters(x_thr=0.0)
        with pytest.raises(ValueError, match="x_thr must be positive"):
            ConfidenceParameters(x_thr=-0.01)

    def test_invalid_lambda_factor(self):
        """Test that invalid lambda_factor raises ValueError."""
        with pytest.raises(ValueError, match="lambda_factor must be > 1"):
            ConfidenceParameters(lambda_factor=1.0)
        with pytest.raises(ValueError, match="lambda_factor must be > 1"):
            ConfidenceParameters(lambda_factor=0.5)

    def test_invalid_c_min(self):
        """Test that invalid c_min raises ValueError."""
        with pytest.raises(ValueError, match="c_min must be in"):
            ConfidenceParameters(c_min=0.0)
        with pytest.raises(ValueError, match="c_min must be in"):
            ConfidenceParameters(c_min=1.0)
        with pytest.raises(ValueError, match="c_min must be in"):
            ConfidenceParameters(c_min=-0.1)


class TestMPSConfidenceFunction:
    """Test MPSConfidenceFunction class."""

    def test_compute_confidence_below_threshold(self):
        """Test confidence computation for values below x_thr."""
        params = ConfidenceParameters(x_thr=0.005, lambda_factor=3.0, c_min=0.1)
        func = MPSConfidenceFunction(params)
        
        x = np.array([0.0, 0.001, 0.003, 0.005])
        confidences = func.compute_confidence(x)
        
        # All values <= x_thr should have confidence 1.0
        np.testing.assert_array_equal(confidences, np.ones_like(x))

    def test_compute_confidence_in_decay_region(self):
        """Test confidence computation in the exponential decay region."""
        params = ConfidenceParameters(x_thr=0.005, lambda_factor=3.0, c_min=0.1)
        func = MPSConfidenceFunction(params)
        
        # Middle of decay region
        x = np.array([0.01])
        confidences = func.compute_confidence(x)
        
        # c(0.01) = exp(-alpha * (0.01 - 0.005))
        expected = np.exp(-params.alpha * (0.01 - 0.005))
        np.testing.assert_allclose(confidences, expected)
        
        # Should be between c_min and 1.0
        assert 0.1 < confidences[0] < 1.0

    def test_compute_confidence_at_x_max(self):
        """Test confidence computation at x_max."""
        params = ConfidenceParameters(x_thr=0.005, lambda_factor=3.0, c_min=0.1)
        func = MPSConfidenceFunction(params)
        
        x = np.array([params.x_max])
        confidences = func.compute_confidence(x)
        
        # At x_max, confidence should be exactly c_min
        np.testing.assert_allclose(confidences, params.c_min, rtol=1e-10)

    def test_compute_confidence_above_x_max_nonstrict(self):
        """Test confidence computation for values above x_max (non-strict mode)."""
        params = ConfidenceParameters(x_thr=0.005, lambda_factor=3.0, c_min=0.1)
        func = MPSConfidenceFunction(params)
        
        x = np.array([0.02, 0.1, 1.0])
        confidences = func.compute_confidence(x, strict=False)
        
        # All values > x_max should be clipped to c_min in non-strict mode
        np.testing.assert_array_equal(confidences, np.full_like(x, params.c_min))

    def test_compute_confidence_above_x_max_strict(self):
        """Test that values above x_max raise AssertionError in strict mode."""
        params = ConfidenceParameters(x_thr=0.005, lambda_factor=3.0, c_min=0.1)
        func = MPSConfidenceFunction(params)
        
        x = np.array([0.02, 0.1, 1.0])
        with pytest.raises(AssertionError, match="All x values must be"):
            func.compute_confidence(x, strict=True)

    def test_compute_confidence_mixed_regions(self):
        """Test confidence computation with values in all regions."""
        params = ConfidenceParameters(x_thr=0.005, lambda_factor=3.0, c_min=0.1)
        func = MPSConfidenceFunction(params)
        
        x = np.array([0.001, 0.005, 0.01, 0.015])
        confidences = func.compute_confidence(x, strict=False)
        
        # First two should be 1.0
        assert confidences[0] == 1.0
        assert confidences[1] == 1.0
        
        # Third should be in decay region
        expected_2 = np.exp(-params.alpha * (0.01 - 0.005))
        np.testing.assert_allclose(confidences[2], expected_2)
        
        # Fourth should be at x_max (c_min)
        np.testing.assert_allclose(confidences[3], params.c_min, rtol=1e-10)

    def test_compute_confidence_edge_cases(self):
        """Test edge cases for confidence computation."""
        params = ConfidenceParameters(x_thr=0.005, lambda_factor=3.0, c_min=0.1)
        func = MPSConfidenceFunction(params)
        
        # Empty array
        x = np.array([])
        confidences = func.compute_confidence(x)
        assert len(confidences) == 0
        
        # Single value
        x = np.array([0.01])
        confidences = func.compute_confidence(x)
        assert len(confidences) == 1
        assert 0.1 < confidences[0] < 1.0

    def test_compute_confidence_vectorization(self):
        """Test that confidence computation is properly vectorized."""
        params = ConfidenceParameters(x_thr=0.005, lambda_factor=3.0, c_min=0.1)
        func = MPSConfidenceFunction(params)
        
        # Large array to test vectorization
        x = np.linspace(0.0, 0.015, 1000)
        confidences = func.compute_confidence(x, strict=False)
        
        # Should be monotonically decreasing
        assert len(confidences) == 1000
        assert np.all(confidences[:-1] >= confidences[1:])
        
        # First value should be 1.0, last should be c_min
        assert confidences[0] == 1.0
        np.testing.assert_allclose(confidences[-1], params.c_min, rtol=1e-10)

    def test_compute_confidence_different_parameters(self):
        """Test confidence computation with different parameter sets."""
        # More aggressive decay (smaller lambda)
        params1 = ConfidenceParameters(x_thr=0.005, lambda_factor=2.0, c_min=0.1)
        func1 = MPSConfidenceFunction(params1)
        
        # Less aggressive decay (larger lambda)
        params2 = ConfidenceParameters(x_thr=0.005, lambda_factor=4.0, c_min=0.1)
        func2 = MPSConfidenceFunction(params2)
        
        x = np.array([0.0075])
        conf1 = func1.compute_confidence(x)
        conf2 = func2.compute_confidence(x)
        
        # More aggressive decay should give lower confidence at same x
        # (assuming x is between x_thr and x_max for both)
        assert conf1[0] < conf2[0]
