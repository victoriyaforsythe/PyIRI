"""Unit tests for PyIRI's sh_library."""

import numpy as np
import pytest
from unittest import mock

from PyIRI.sh_library import Apex
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


def test_shape_and_type():
    """Check output shape and dtype for simple 1D input."""
    theta = np.linspace(0, np.pi, 5)
    phi = np.linspace(0, 2 * np.pi, 5)
    lmax = 3
    F = real_SH_func(theta, phi, lmax=lmax)
    assert isinstance(F, np.ndarray)
    assert F.dtype == float
    # After squeeze, shape is (N_SH, N_G)
    assert F.shape == ((lmax + 1) ** 2, phi.size)


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
    expected = 1.0  # uses 4π normalization
    np.testing.assert_allclose(F[0], expected, rtol=1e-6)
    np.testing.assert_allclose(F[2], 0.0, atol=1e-10)


def test_large_lmax_runs_fast():
    """Sanity check performance for moderate lmax."""
    theta = np.linspace(0, np.pi, 10)
    phi = np.linspace(0, 2 * np.pi, 10)
    F = real_SH_func(theta, phi, lmax=10)
    assert F.shape == ((10 + 1) ** 2, theta.size)

@pytest.fixture
def fake_dataset():
    """Create a mock NetCDF dataset with minimal structure."""
    ds = mock.MagicMock()
    # Fake year array
    ds.__enter__.return_value = ds  # so it works with 'with' context
    ds['Year'].__getitem__.return_value = np.array([2000, 2005, 2010])

    # Suppose each coefficient field is (N_years, N_SH)
    n_years, n_sh = 3, 4
    shape = (n_years, n_sh)
    data = np.arange(n_years * n_sh).reshape(shape).astype(float)

    ds['QDLat'].__getitem__.return_value = data
    ds['QDLon_cos'].__getitem__.return_value = data + 1
    ds['QDLon_sin'].__getitem__.return_value = data + 2
    ds['GeoLat'].__getitem__.return_value = data + 3
    ds['GeoLon_cos'].__getitem__.return_value = data + 4
    ds['GeoLon_sin'].__getitem__.return_value = data + 5

    # For attribute access
    ds['QDLat'].shape = shape
    ds['GeoLat'].shape = shape
    return ds


@pytest.fixture
def patch_environment(fake_dataset):
    """Patch dependencies of Apex."""
    with mock.patch("PyIRI.sh_library.nc.Dataset",
                    return_value=fake_dataset), \
         mock.patch("PyIRI.sh_library.real_SH_func",
                    return_value=np.ones((4, 5))), \
         mock.patch("PyIRI.sh_library.oe.contract",
                    side_effect=lambda expr, a, b: np.ones(5)), \
         mock.patch("PyIRI.sh_library.PyIRI.main_library.adjust_longitude",
                    ide_effect=lambda x, mode: x), \
         mock.patch("PyIRI.sh_library.logger"):
        yield


def test_geo_to_qd(patch_environment):
    """Test GEO_2_QD transformation returns arrays of correct shape."""
    Lat = np.array([[10, 20], [30, 40]])
    Lon = np.array([[100, 120], [140, 160]])
    dtime = dt.datetime(2005, 1, 1)
    QDLat, QDLon = Apex(Lat, Lon, dtime, type="GEO_2_QD")
    assert QDLat.shape == Lat.shape
    assert QDLon.shape == Lon.shape
    assert np.all(np.isfinite(QDLat))
    assert np.all(np.isfinite(QDLon))


def test_qd_to_geo(patch_environment):
    """Test QD_2_GEO transformation returns arrays of correct shape."""
    Lat = np.array([10, 20, 30])
    Lon = np.array([100, 150, 200])
    dtime = dt.datetime(2010, 6, 1)

    GeoLat, GeoLon = Apex(Lat, Lon, dtime, type="QD_2_GEO")

    assert GeoLat.shape == Lat.shape
    assert GeoLon.shape == Lon.shape
    assert np.all(np.isfinite(GeoLat))
    assert np.all(np.isfinite(GeoLon))


def test_invalid_type_raises(patch_environment):
    """Ensure invalid 'type' argument raises ValueError."""
    Lat = np.array([0])
    Lon = np.array([0])
    dtime = dt.datetime(2005, 1, 1)

    with pytest.raises(ValueError):
        Apex(Lat, Lon, dtime, type="INVALID")
