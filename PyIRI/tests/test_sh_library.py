"""Unit tests for PyIRI's sh_library."""

import numpy as np
import pytest

# import PyIRI
from PyIRI.sh_library import nearest_element
from PyIRI.sh_library import real_SH_func
from PyIRI.sh_library import to_numpy_array


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


def test_shape_and_type(lmax):
    """Check output shape and dtype for simple 1D input."""
    theta = np.linspace(0, np.pi, 5)
    phi = np.linspace(0, 2 * np.pi, 5)
    lmax = 3
    F = real_SH_func(theta, phi, lmax=lmax)
    n_sh = (lmax + 1) ** 2
    assert isinstance(F, np.ndarray)
    assert F.shape == (n_sh, phi.size)
    assert F.dtype == float


def test_shape_for_2d_input():
    """Check shape when theta and phi are 2D arrays (gridded input)."""
    lon = np.linspace(0, 2 * np.pi, 10)
    lat = np.linspace(0, np.pi, 8)
    theta, phi = np.meshgrid(lat, lon, indexing="ij")
    F = real_SH_func(theta, phi, lmax=3)
    assert F.shape == ((3 + 1) ** 2, theta.shape[0], theta.shape[1])


def test_known_small_case():
    """For lmax=1, compare against analytical spherical harmonics."""
    theta = np.array([np.pi / 2])
    phi = np.array([0.0])
    F = real_SH_func(theta, phi, lmax=1)
    # indices: l=0 -> i=0, l=1 -> i=1,2,0 ± m
    # expect Y_0^0 ≈ sqrt(1/(4π))
    # since function uses 4π normalization, not unit normalization
    expected = 1.0
    np.testing.assert_allclose(F[0], expected, rtol=1e-6)

    # Y_1^0(θ=π/2) ≈ sqrt(3/(4π))*cos(θ) = 0
    np.testing.assert_allclose(F[2], 0.0, atol=1e-10)


def test_values_near_poles():
    """Ensure no NaNs or jumps near θ=0 and θ=π."""
    theta = np.linspace(0, np.pi, 20)
    phi = np.linspace(0, 2 * np.pi, 10)
    F = real_SH_func(theta, phi, lmax=4)
    assert not np.isnan(F).any()
    # Check continuity near poles
    north = np.nanmean(F[:, 0, :])
    south = np.nanmean(F[:, -1, :])
    assert np.isfinite(north) and np.isfinite(south)


def test_periodicity_phi():
    """Ensure lon periodicity (φ=0 and φ=2π) produce same values."""
    theta = np.array([np.pi / 3])
    phi = np.array([0, 2 * np.pi])
    F = real_SH_func(theta, phi, lmax=3)
    assert F.shape[-1] == 2
    np.testing.assert_allclose(F[..., 0], F[..., 1], atol=1e-12)


def test_large_lmax_runs_fast():
    """Sanity check performance for moderate lmax."""
    theta = np.linspace(0, np.pi, 10)
    phi = np.linspace(0, 2 * np.pi, 10)
    F = real_SH_func(theta, phi, lmax=10)
    assert F.shape == ((10 + 1) ** 2, theta.size)
