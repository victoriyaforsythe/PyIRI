"""Unit tests for PyIRI's main_library."""

import numpy as np

from PyIRI.main_library import to_numpy_array


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
