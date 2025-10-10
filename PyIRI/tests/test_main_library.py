"""Unit tests for PyIRI's main_library."""

import numpy as np
import pytest

from PyIRI.main_library import nearest_element
from PyIRI.main_library import to_numpy_array


class TestNearestElement:
    """Test TestNearestElement."""

    def test_nearest_basic(self):
        """Test TestNearestElement."""

        arr = np.array([0, 1, 2, 3, 4, 5])
        assert nearest_element(arr, 3.1) == 3
        assert nearest_element(arr, 2.6) == 3
        assert nearest_element(arr, 0.4) == 0

    def test_negative_values(self):
        """Test TestNearestElement with negative numbers."""

        arr = np.array([-10, -5, 0, 5, 10])
        assert nearest_element(arr, -3) == 1
        assert nearest_element(arr, 6) == 3

    def test_multiple_equal_distance(self):
        """Test TestNearestElement with first last numbers."""

        arr = np.array([0, 2, 4, 6])
        # 3 is equally close to 2 and 4, should return first (index 1)
        assert nearest_element(arr, 3) == 1

    def test_empty_array_raises(self):
        """Test TestNearestElement empty."""

        with pytest.raises(ValueError):
            nearest_element([], 1)


class TestToNumpyArray:
    """Test TestToNumpyArray."""

    def test_scalar_input(self):
        """Test test_scalar_input 1-element."""

        result = to_numpy_array(5.5)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert result[0] == 5.5

    def test_list_input(self):
        """Test test_scalar_input float."""

        result = to_numpy_array([1, 2, 3])
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))
        assert result.dtype == float

    def test_array_input(self):
        """Test test_scalar_input int."""

        arr = np.array([1, 2, 3], dtype=int)
        result = to_numpy_array(arr)
        np.testing.assert_array_equal(result, arr.astype(float))

    def test_nested_list(self):
        """Test test_scalar_input 2d."""

        result = to_numpy_array([[1, 2], [3, 4]])
        assert result.shape == (2, 2)
        assert result.dtype == float
