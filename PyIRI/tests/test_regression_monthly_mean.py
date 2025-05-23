import numpy as np
import json
import pytest
import PyIRI
from PyIRI import edp_update as ml

def test_IRI_monthly_mean_par_coarse_regression():
    year = 2020
    month = 4
    f107 = 100
    ccir_or_ursi = 0
    lon_res = 180
    lat_res = 90
    hr_res = 24
    ahr = np.arange(0, 24, hr_res)

    alon_2d, alat_2d = np.mgrid[-180:180 + lon_res:lon_res, -90:90 + lat_res:lat_res]
    alon = np.reshape(alon_2d, alon_2d.size)
    alat = np.reshape(alat_2d, alat_2d.size)

    coeff_dir = PyIRI.coeff_dir
    f2, f1, e_peak, es_peak, sun, mag = ml.IRI_monthly_mean_par(
        year, month, ahr, alon, alat, coeff_dir, ccir_or_ursi)

    aalt = np.arange(200, 300, 10)
    edens_prof = ml.reconstruct_density_from_parameters(f2, f1, e_peak, aalt)

    with open("Test_Output_Mean_Monthly_Parameters.json", "r") as f:
        ref = json.load(f)

    def assert_close(actual, expected, key, atol=1e-2):
        assert np.allclose(actual, np.array(expected), atol=atol), f"Mismatch in {key}"

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

    assert_close(edens_prof, ref["edens_prof"], "edens_prof")
