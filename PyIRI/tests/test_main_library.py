"""Unit tests for PyIRI's main_library."""

import numpy as np

from PyIRI.main_library import to_numpy_array
import PyIRI.main_library as main
import PyIRI


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


def test_fo_1day_interpolation():
    """Test linear interpolation of foF2."""
    F2, F1, E, Es, *_ = main.IRI_density_1day(2024,
                                              3,
                                              8,
                                              np.array([0]),
                                              np.array([20]),
                                              np.array([40]),
                                              np.array([0]),
                                              main.IG12_2_F107(40),
                                              PyIRI.coeff_dir,
                                              ccir_or_ursi=0)

    fo_1day = []
    true_fo = []
    for layer in [F2, F1, E, Es]:
        fo_1day.append(layer['fo'][0, 0])
        Nm_1day = layer['Nm'][0, 0]
        true_fo.append(main.den2freq(Nm_1day))

    assert fo_1day == true_fo, ("fo 1day interpolation error: fo = "
                                f"{fo_1day}, should be {true_fo}")
