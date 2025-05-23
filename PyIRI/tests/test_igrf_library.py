# -*- coding: utf-8 -*-
"""Test the PyIRI IGRF Library functions.

Notes
-----
Functions that rely on the IGRF coefficients are expected have outputs that
change when the IGRF coefficients are updated.

"""

import pytest
import numpy as np
import datetime as dt
from PyIRI import coeff_dir
from PyIRI.igrf_library import (
    verify_inclination,
    inc2modip,
    inc2magnetic_dip_latitude,
    decimal_year
)

def test_verify_inclination_valid():
    assert verify_inclination(45) == 45
    assert np.allclose(verify_inclination([0, -30, 60]), [0, -30, 60])

def test_verify_inclination_tolerance_adjustment():
    result = verify_inclination([90.05, -90.05])
    assert np.allclose(result, [90.0, -90.0])

def test_verify_inclination_invalid():
    with pytest.raises(ValueError):
        verify_inclination(100)
    with pytest.raises(ValueError):
        verify_inclination([-95])

def test_inc2modip_typical():
    inc = np.array([45])
    alat = np.array([30])
    modip = inc2modip(inc, alat)
    assert np.isfinite(modip).all()

def test_inc2magnetic_dip_latitude_typical():
    inc = np.array([60])
    dip_lat = inc2magnetic_dip_latitude(inc)
    expected = np.rad2deg(np.arctan(0.5 * np.tan(np.deg2rad(60))))
    assert np.isclose(dip_lat, expected, atol=1e-6)

def test_decimal_year_mid_year():
    dt_obj = dt.datetime(2020, 7, 2)  # 2020 is a leap year
    assert np.isclose(decimal_year(dt_obj), 2020.5)
