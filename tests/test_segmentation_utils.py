#
# File: test_segmentation_utils.py
# Author: Banafshe Bamdad + Claude Code
# Created: 2026-04-21 CET
#

"""
Unit tests for bb_utils.segmentation.utils.

Tests cover:
- dilate_mask: zero radius = identity, positive radius expands mask
- union_masks: OR logic, empty list raises, shape mismatch raises
- resize_mask: shape change, binary values preserved
- SegmentationBackend._validate_output: shape mismatch raises, values clipped
"""

import numpy as np
import pytest

from bb_utils.segmentation.base import SegmentationBackend
from bb_utils.segmentation.utils import dilate_mask, resize_mask, union_masks


# ---------------------------------------------------------------------------
# dilate_mask
# ---------------------------------------------------------------------------

class TestDilateMask:

    def test_zero_radius_identity(self):
        mask = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)
        result = dilate_mask(mask, 0)
        np.testing.assert_array_equal(result, mask)

    def test_single_pixel_expands(self):
        mask = np.zeros((7, 7), dtype=np.uint8)
        mask[3, 3] = 1
        result = dilate_mask(mask, 1)
        # Centre and its 4-neighbours should be 1
        assert result[3, 3] == 1
        assert result[2, 3] == 1
        assert result[4, 3] == 1
        assert result[3, 2] == 1
        assert result[3, 4] == 1

    def test_all_zero_stays_zero(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        result = dilate_mask(mask, 3)
        np.testing.assert_array_equal(result, 0)

    def test_output_dtype_uint8(self):
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, 2] = 1
        result = dilate_mask(mask, 1)
        assert result.dtype == np.uint8

    def test_output_values_in_0_1(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[5, 5] = 1
        result = dilate_mask(mask, 2)
        assert set(np.unique(result)).issubset({0, 1})

    def test_negative_radius_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            dilate_mask(np.zeros((4, 4), dtype=np.uint8), -1)


# ---------------------------------------------------------------------------
# union_masks
# ---------------------------------------------------------------------------

class TestUnionMasks:

    def test_union_two_masks(self):
        a = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        b = np.array([[0, 0], [0, 1]], dtype=np.uint8)
        result = union_masks([a, b])
        expected = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_single_mask_passthrough(self):
        mask = np.array([[1, 0, 1]], dtype=np.uint8)
        result = union_masks([mask])
        np.testing.assert_array_equal(result, mask)

    def test_union_idempotent(self):
        mask = np.ones((5, 5), dtype=np.uint8)
        result = union_masks([mask, mask, mask])
        np.testing.assert_array_equal(result, mask)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            union_masks([])

    def test_shape_mismatch_raises(self):
        a = np.zeros((4, 4), dtype=np.uint8)
        b = np.zeros((5, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="same shape"):
            union_masks([a, b])

    def test_output_dtype_uint8(self):
        result = union_masks([np.zeros((3, 3), dtype=np.uint8)])
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# resize_mask
# ---------------------------------------------------------------------------

class TestResizeMask:

    def test_upsample_preserves_binary_values(self):
        mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        result = resize_mask(mask, (4, 4))
        assert result.shape == (4, 4)
        assert set(np.unique(result)).issubset({0, 1})

    def test_downsample_preserves_binary_values(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[5:, 5:] = 1
        result = resize_mask(mask, (5, 5))
        assert result.shape == (5, 5)
        assert set(np.unique(result)).issubset({0, 1})

    def test_same_shape_returns_copy(self):
        mask = np.ones((6, 6), dtype=np.uint8)
        result = resize_mask(mask, (6, 6))
        assert result.shape == (6, 6)
        np.testing.assert_array_equal(result, mask)

    def test_output_dtype_uint8(self):
        result = resize_mask(np.zeros((4, 4), dtype=np.uint8), (8, 8))
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# SegmentationBackend._validate_output
# ---------------------------------------------------------------------------

class _ConcreteBackend(SegmentationBackend):
    """Minimal concrete implementation for testing the base class helper."""
    def segment(self, image, target_classes):
        return np.zeros(image.shape[:2], dtype=np.uint8)


class TestValidateOutput:
    _b = _ConcreteBackend()

    def test_correct_shape_passes(self):
        mask = np.zeros((5, 5), dtype=np.uint8)
        result = self._b._validate_output(mask, (5, 5))
        assert result.shape == (5, 5)

    def test_shape_mismatch_raises(self):
        mask = np.zeros((4, 4), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="shape"):
            self._b._validate_output(mask, (5, 5))

    def test_values_clipped_to_0_1(self):
        mask = np.array([[0, 2, 255]], dtype=np.uint8)
        result = self._b._validate_output(mask, (1, 3))
        assert set(np.unique(result)).issubset({0, 1})

    def test_output_dtype_uint8(self):
        mask = np.zeros((3, 3), dtype=bool)
        result = self._b._validate_output(mask, (3, 3))
        assert result.dtype == np.uint8
