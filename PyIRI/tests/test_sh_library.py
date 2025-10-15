"""Unit tests for PyIRI's sh_library."""

import datetime as dt

import numpy as np
import pytest

from PyIRI.sh_library import Apex
from PyIRI.sh_library import real_FS_func
from PyIRI.sh_library import real_SH_func


@pytest.mark.parametrize(
    "inp, c, exp_shape",
    [
        (np.array([0., 12.]), 2, (2, 3)),
        ([1, 2, 3, 4], 3, (4, 5)),
        ((5.5, 6.6), 4, (2, 7)),
        (12.0, 1, (1, 1)),
    ],
    ids=["array", "list", "tuple", "scalar"]
)
def test_shape_dtype_FS(inp, c, exp_shape):
    """Check the shape and dtype of output of function FS."""
    out = real_FS_func(inp, N_FS_c=c)
    assert isinstance(out, np.ndarray)
    assert np.issubdtype(out.dtype, np.floating)
    assert out.shape == exp_shape


def test_empty_FS():
    """Check the shape and dtype of output of function FS given empty input."""
    out = real_FS_func(np.array([]), N_FS_c=3)
    assert out.shape == (0, 5)
    assert out.size == 0


@pytest.mark.parametrize("c", [0, -2])
def test_Nc_negative_FS(c):
    """Check that an error is raised for negative/null coeff number in FS."""
    with pytest.raises(ValueError):
        real_FS_func([0, 1], N_FS_c=c)


@pytest.mark.parametrize("c", [1.5, "3", None])
def test_Nc_type_FS(c):
    """Check that an error is raised for inavlid input types in FS."""
    with pytest.raises(TypeError):
        real_FS_func([0, 1], N_FS_c=c)


def test_periodic_FS():
    """Check periodicity of function FS."""
    a = np.array([0.5, 5.2, 13.7])
    np.testing.assert_allclose(real_FS_func(a, 4), real_FS_func(a + 24, 4),
                               atol=1e-15)


@pytest.mark.parametrize("UT", [1, 12.3, 23.9999])
def test_known_small_case_FS(UT):
    """For small N_FS_c, compare against analytical Fourier series."""
    F = real_FS_func(UT, N_FS_c=2)
    np.testing.assert_allclose(F[0, 0], 1.0, rtol=1e-6)
    np.testing.assert_allclose(F[0, 1], np.cos(2 * np.pi / 24 * UT), atol=1e-10)
    np.testing.assert_allclose(F[0, 2], np.sin(2 * np.pi / 24 * UT), atol=1e-10)


def test_shape_and_type_SH():
    """Check output shape and dtype for simple 1D input for SH function."""
    theta = np.linspace(0, np.pi, 5)
    phi = np.linspace(0, 2 * np.pi, 5)
    lmax = 3
    F = real_SH_func(theta, phi, lmax=lmax)
    assert isinstance(F, np.ndarray)
    assert F.dtype == float
    # After squeeze, shape is (N_SH, N_G)
    assert F.shape == ((lmax + 1) ** 2, phi.size)


def test_shape_for_2d_input_SH():
    """Check shape when theta and phi are 2D arrays for SH function."""
    lon = np.linspace(0, 2 * np.pi, 10)
    lat = np.linspace(0, np.pi, 8)
    theta, phi = np.meshgrid(lat, lon, indexing="ij")
    F = real_SH_func(theta, phi, lmax=3)
    assert F.shape == ((3 + 1) ** 2, theta.shape[0], theta.shape[1])


def test_known_small_case_SH():
    """For lmax=1, compare against analytical spherical harmonics."""
    theta = np.array([np.pi / 2])
    phi = np.array([0.0])
    F = real_SH_func(theta, phi, lmax=1)
    expected = 1.0  # uses 4π normalization
    np.testing.assert_allclose(F[0], expected, rtol=1e-6)
    np.testing.assert_allclose(F[2], 0.0, atol=1e-10)


def test_large_lmax_runs_fast_SH():
    """Sanity check performance for moderate lmax."""
    theta = np.linspace(0, np.pi, 10)
    phi = np.linspace(0, 2 * np.pi, 10)
    F = real_SH_func(theta, phi, lmax=10)
    assert F.shape == ((10 + 1) ** 2, theta.size)


def test_geo_to_qd():
    """Test GEO_2_QD transformation returns arrays of correct shape."""
    Lat = np.array([[10, 20], [30, 40]])
    Lon = np.array([[100, 120], [140, 160]])
    dtime = dt.datetime(2005, 1, 1)
    QDLat, QDLon = Apex(Lat, Lon, dtime, transform_type="GEO_2_QD")
    assert QDLat.shape == Lat.shape
    assert QDLon.shape == Lon.shape
    assert np.all(np.isfinite(QDLat))
    assert np.all(np.isfinite(QDLon))


def test_qd_to_geo():
    """Test QD_2_GEO transformation returns arrays of correct shape."""
    Lat = np.array([10, 20, 30])
    Lon = np.array([100, 150, 200])
    dtime = dt.datetime(2010, 6, 1)
    GeoLat, GeoLon = Apex(Lat, Lon, dtime, transform_type="QD_2_GEO")
    assert GeoLat.shape == Lat.shape
    assert GeoLon.shape == Lon.shape
    assert np.all(np.isfinite(GeoLat))
    assert np.all(np.isfinite(GeoLon))


def test_invalid_type_raises():
    """Ensure invalid 'type' argument raises ValueError."""
    Lat = np.array([0])
    Lon = np.array([0])
    dtime = dt.datetime(2005, 1, 1)
    with pytest.raises(ValueError):
        Apex(Lat, Lon, dtime, transform_type="INVALID")
