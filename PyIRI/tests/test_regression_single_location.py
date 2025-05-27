"""
Regression test for IRI_density_1day for single-location input and ref output.
"""

import json
import numpy as np
import os
import pathlib

import PyIRI
from PyIRI import edp_update as ml

test_dir = os.path.join(str(pathlib.Path(PyIRI.tests.__file__).resolve(
    ).parent), "testdata")

def test_IRI_density_1day_regression():
    """Compare IRI_density_1day output to known reference, single location."""
    year = 2020
    month = 4
    day = 1
    ahr = np.arange(0, 24, 0.25)
    alon = np.array([10.])
    alat = np.array([20.])
    aalt = np.arange(90, 700, 10)
    f107 = 100
    coeff_dir = PyIRI.coeff_dir
    ccir_or_ursi = 0

    f2, f1, e_peak, es_peak, sun, mag, edp = ml.IRI_density_1day(
        year, month, day, ahr, alon, alat, aalt, f107, coeff_dir, ccir_or_ursi)

    test_file = os.path.join(test_dir,
                             'Test_Output.json')

    with open(test_file, "r") as f:
        ref = json.load(f)

    def assert_close(actual, expected, key, atol=1e-2):
        assert np.allclose(actual, np.array(expected), atol=atol), (
            f"Mismatch in {key}")

    assert_close(f2["Nm"], ref["NmF2"], "NmF2")
    assert_close(f2["hm"], ref["hmF2"], "hmF2")
    assert_close(f2["B_top"], ref["B_F2_top"], "B_F2_top")
    assert_close(f2["B_bot"], ref["B_F2_bot"], "B_F2_bot")

    assert_close(f1["Nm"], ref["NmF1"], "NmF1")
    assert_close(f1["hm"], ref["hmF1"], "hmF1")
    assert_close(f1["B_bot"], ref["B_F1_bot"], "B_F1_bot")

    assert_close(e_peak["Nm"], ref["NmE"], "NmE")
    assert_close(e_peak["hm"], ref["hmE"], "hmE")
    assert_close(e_peak["B_top"], ref["B_E_top"], "B_E_top")
    assert_close(e_peak["B_bot"], ref["B_E_bot"], "B_E_bot")

    assert_close(es_peak["Nm"], ref["NmEs"], "NmEs")
    assert_close(es_peak["hm"], ref["hmEs"], "hmEs")
    assert_close(es_peak["B_top"], ref["B_Es_top"], "B_Es_top")
    assert_close(es_peak["B_bot"], ref["B_Es_bot"], "B_Es_bot")

    assert_close(sun["lon"], ref["sun_lon"], "sun_lon")
    assert_close(sun["lat"], ref["sun_lat"], "sun_lat")

    assert_close(mag["inc"], ref["inc"], "inc")
    assert_close(mag["modip"], ref["modip"], "modip")
    assert_close(mag["mag_dip_lat"], ref["mag_dip_lat"], "mag_dip_lat")
