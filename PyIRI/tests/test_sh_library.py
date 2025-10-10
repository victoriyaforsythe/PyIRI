"""Unit tests for PyIRI's sh_library."""

import datetime as dt

import numpy as np
import pytest

from PyIRI.sh_library import Apex
from PyIRI.sh_library import real_SH_func


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


def test_geo_to_qd():
    """Test GEO_2_QD transformation returns arrays of correct shape."""

    Lat = np.array([[10, 20], [30, 40]])
    Lon = np.array([[100, 120], [140, 160]])
    dtime = dt.datetime(2005, 1, 1)
    QDLat, QDLon = Apex(Lat, Lon, dtime, type="GEO_2_QD")
    assert QDLat.shape == Lat.shape
    assert QDLon.shape == Lon.shape
    assert np.all(np.isfinite(QDLat))
    assert np.all(np.isfinite(QDLon))


def test_qd_to_geo():
    """Test QD_2_GEO transformation returns arrays of correct shape."""

    Lat = np.array([10, 20, 30])
    Lon = np.array([100, 150, 200])
    dtime = dt.datetime(2010, 6, 1)
    GeoLat, GeoLon = Apex(Lat, Lon, dtime, type="QD_2_GEO")
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
        Apex(Lat, Lon, dtime, type="INVALID")
