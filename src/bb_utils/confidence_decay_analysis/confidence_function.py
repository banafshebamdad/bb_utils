#
# File: confidence_function.py
# Author: Banafshe Bamdad + GitHub Copilot (Claude Sonnet 4.5)
# Email: banafshebamdad@gmail.com
# Created: 2026-02-15
#

"""
MPS-threshold-based confidence function with safety checks.
"""

import numpy as np
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceParameters:
    """Parameters for MPS-threshold-based confidence function.
    
    Attributes:
        x_thr: MPS nominal threshold (default: 0.005)
        lambda_factor: Safety factor λ > 1 (default: 3.0)
        c_min: Minimum confidence at x_max (default: 0.1)
        x_max: Outlier cutoff = λ * x_thr (computed automatically)
        alpha: Decay rate = -ln(c_min) / (x_max - x_thr) (computed automatically)
    
    Example:
        >>> params = ConfidenceParameters(x_thr=0.005, lambda_factor=3.0, c_min=0.1)
        >>> print(f"x_max: {params.x_max}, alpha: {params.alpha}")
    """
    x_thr: float           # MPS nominal threshold
    lambda_factor: float   # Safety factor λ > 1
    c_min: float          # Minimum confidence at x_max
    
    # Derived parameters (computed automatically)
    x_max: float = field(init=False)
    alpha: float = field(init=False)
    
    def __post_init__(self):
        """Validate and compute derived parameters."""
        assert self.x_thr > 0, "x_thr must be positive"
        assert self.lambda_factor > 1, "λ must be > 1"
        assert 0 < self.c_min < 1, "c_min must be in (0, 1)"
        
        self.x_max = self.lambda_factor * self.x_thr
        self.alpha = -np.log(self.c_min) / (self.x_max - self.x_thr)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization.
        
        Returns:
            dict: Parameters as dictionary
        """
        return {
            'x_thr': float(self.x_thr),
            'lambda_factor': float(self.lambda_factor),
            'c_min': float(self.c_min),
            'x_max': float(self.x_max),
            'alpha': float(self.alpha)
        }


class MPSConfidenceFunction:
    """MPS-threshold-based confidence mapping with safety modes.
    
    Confidence function:
        c(x) = 1                              if x <= x_thr
        c(x) = exp(-alpha * (x - x_thr))      if x_thr < x <= x_max
    
    Safety modes:
        strict=True:  Assert all x <= x_max (raises error if violated)
        strict=False: Clip values > x_max to x_max (returns c_min for outliers)
    
    Args:
        params: ConfidenceParameters instance
    
    Example:
        >>> params = ConfidenceParameters(x_thr=0.005, lambda_factor=3.0, c_min=0.1)
        >>> conf_func = MPSConfidenceFunction(params)
        >>> x = np.array([0.001, 0.007, 0.012])
        >>> confidences = conf_func.compute_confidence(x, strict=False)
    """
    
    def __init__(self, params: ConfidenceParameters):
        self.params = params
    
    def compute_confidence(self, x: np.ndarray, strict: bool = True) -> np.ndarray:
        """Compute confidence for inv_dist_std values.
        
        Args:
            x: Array of inv_dist_std values
            strict: If True, assert all x <= x_max. If False, clip to x_max.
            
        Returns:
            confidence: Array of confidence values in (c_min, 1]
            
        Raises:
            ValueError: If strict=True and any x > x_max
        """
        if strict:
            # Strict mode: verify all values are within bounds
            if np.any(x > self.params.x_max):
                n_violations = int((x > self.params.x_max).sum())
                max_violation = float(x[x > self.params.x_max].max())
                raise ValueError(
                    f"Strict mode violation: {n_violations} values exceed x_max={self.params.x_max:.6f}. "
                    f"Maximum value: {max_violation:.6f}. "
                    f"Set strict=False to clip values or filter outliers before calling."
                )
        else:
            # Non-strict mode: clip to valid range
            x = np.clip(x, 0, self.params.x_max)
        
        # Compute confidence (vectorized)
        confidence = np.ones_like(x, dtype=float)
        
        # For x > x_thr, apply exponential decay
        mask_decay = x > self.params.x_thr
        confidence[mask_decay] = np.exp(
            -self.params.alpha * (x[mask_decay] - self.params.x_thr)
        )
        
        return confidence
    
    def sanity_checks(self, confidence_dist: dict) -> dict:
        """Compute confidence at key checkpoints.
        
        Args:
            confidence_dist: Dictionary with confidence distribution statistics
            
        Returns:
            dict: Sanity check results with checkpoints and distribution
        """
        x_thr = self.params.x_thr
        x_max = self.params.x_max
        delta = x_max - x_thr
        
        # Confidence at key points
        checkpoints = {
            'c_at_x_thr': float(self.compute_confidence(np.array([x_thr]), strict=False)[0]),
            'c_at_25pct': float(self.compute_confidence(np.array([x_thr + 0.25 * delta]), strict=False)[0]),
            'c_at_50pct': float(self.compute_confidence(np.array([x_thr + 0.50 * delta]), strict=False)[0]),
            'c_at_75pct': float(self.compute_confidence(np.array([x_thr + 0.75 * delta]), strict=False)[0]),
            'c_at_x_max': float(self.compute_confidence(np.array([x_max]), strict=False)[0])
        }
        
        return {
            'checkpoints': checkpoints,
            'confidence_distribution': confidence_dist
        }
