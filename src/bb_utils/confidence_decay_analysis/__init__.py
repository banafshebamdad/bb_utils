#
# File: __init__.py
# Author: Banafshe Bamdad + GitHub Copilot (Claude Sonnet 4.5)
# Email: banafshebamdad@gmail.com
# Created: 2026-02-15
#

"""
Confidence decay rate analysis for InCrowd-VI semi-dense SLAM data.

This package provides tools to analyze inv_dist_std distributions and recommend
MPS-threshold-based confidence parameters for keypoint detection.
"""

from .confidence_function import ConfidenceParameters, MPSConfidenceFunction
from .analyzer import ConfidenceDecayAnalyzer, AnalysisResult

__all__ = [
    'ConfidenceParameters',
    'MPSConfidenceFunction',
    'ConfidenceDecayAnalyzer',
    'AnalysisResult',
]

__version__ = '1.0.0'
