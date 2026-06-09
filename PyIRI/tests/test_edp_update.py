"""Unit tests for PyIRI's edp_update module."""

import numpy as np

import PyIRI
from PyIRI import edp_update
from PyIRI.main_library import freq2den
from PyIRI.main_library import IG12_2_F107
from PyIRI.main_library import R12_2_F107


def test_fo_1day_interpolation():
    """Test linear interpolation of foF2.

    Tests whether IRI_density_1day handles the linear interpolation of fo
    correctly. In the IRI_density_1day function, Nm should be computed directly
    from the interpolated value of fo. Nm should NOT be linearly interpolated.
    If Nm is linearly interpolated, this test will fail.
    """
    F2, F1, E, Es, *_ = edp_update.IRI_density_1day(2024,
                                                    3,
                                                    8,
                                                    np.array([0]),
                                                    np.array([20]),
                                                    np.array([40]),
                                                    np.array([0]),
                                                    IG12_2_F107(40),
                                                    PyIRI.coeff_dir)

    Nm_fn_output = []
    Nm_from_fo = []
    for layer in [F2, F1, E, Es]:
        Nm_fn_output.append(layer['Nm'][0, 0])
        fo_fn_output = layer['fo'][0, 0]
        Nm_from_fo.append(freq2den(fo_fn_output))

    np.testing.assert_array_almost_equal(Nm_fn_output, Nm_from_fo, decimal=4,
                                         err_msg=("Nm/fo interpolation error: "
                                                  "Nm from function is "
                                                  f"{Nm_fn_output}, should be "
                                                  f"{Nm_from_fo}. Verify that "
                                                  "fo is linearly interpolated,"
                                                  " not Nm."))


def test_IRI_density_1day_runs():
    """Test that IRI_density_1day runs and returns expected shape.

    IRI_density_1day must return dictionaries of shape (N_T, N_G) for
    ionospheric parameters, shape (N_T,) for solar and magnetic parameters, and
    shape (N_T, N_G, N_V) for the EDP array.
    """
    year = 2024
    mth = 6
    day = 21
    aUT = np.array([12.0, 13.0])      # N_T = 2
    alon = np.array([0.0])            # N_G = 1
    alat = np.array([0.0])
    aalt = np.linspace(100, 600, 10)  # N_V = 10
    F107 = 100.0
    coeff_dir = PyIRI.coeff_dir

    N_T = len(aUT)
    N_G = len(alon)
    N_V = len(aalt)

    F2, F1, E, Es, sun, mag, EDP = edp_update.IRI_density_1day(
        year, mth, day, aUT, alon, alat, aalt, F107, coeff_dir)

    assert EDP.shape == (N_T, N_V, N_G)
    assert F2['Nm'].shape == (N_T, N_G)
    assert sun['lat'].shape == (N_T,)


def test_IRI_monthly_mean_par_runs():
    """Test that IRI_monthly_mean_par runs and returns expected shape.

    IRI_monthly_mean_par must return dictionaries of shape (N_T, N_G, 2) for
    ionospheric parameters, and shape (N_T,) for solar and magnetic parameters.
    """
    year = 2024
    mth = 6
    aUT = np.array([12.0])
    alon = np.array([0.0])
    alat = np.array([0.0])
    coeff_dir = PyIRI.coeff_dir

    N_T = len(aUT)
    N_G = len(alon)

    F2, F1, E, Es, sun, mag = edp_update.IRI_monthly_mean_par(
        year, mth, aUT, alon, alat, coeff_dir)

    assert F2['Nm'].shape == (N_T, N_G, 2)
    assert sun['lat'].shape == (N_T,)


def test_IRI_density_1day_for_monthly_mean_values():
    """Test that IRI_density_1day returns expected values for IG12=0/100.

    IRI_density_1day calculates iono paramaters using IRI_monthly_mean_par and
    interpolates to the solar activity value requested from the IG12=0 and
    IG12=100 parameters given by IRI_monthly_mean_par. Given IG12=0 or 100 as
    input, IRI_density_1day should return the same parameter values as
    IRI_monthly_mean_par.
    """
    year = 2024
    mth = 6
    day = 15
    aUT = np.array([12.0])
    alon = np.array([0.0])
    alat = np.array([0.0])
    aalt = np.array([100])
    coeff_dir = PyIRI.coeff_dir

    # Giving IG12=0 as input to IRI_density_1day
    F2min, F1min, Emin, _, sunmin, magmin, _ = edp_update.IRI_density_1day(
        year, mth, day, aUT, alon, alat, aalt, IG12_2_F107(0), coeff_dir)

    # Giving IG12=100 as input to IRI_density_1day
    F2max, F1max, Emax, _, sunmax, magmax, _ = edp_update.IRI_density_1day(
        year, mth, day, aUT, alon, alat, aalt, IG12_2_F107(100), coeff_dir)

    # Reference values for IG12=0/100 obtained from IRI_monthly_mean_par
    F2m, F1m, Em, _, sunm, magm = edp_update.IRI_monthly_mean_par(
        year, mth, aUT, alon, alat, coeff_dir)

    # Check that IRI_density_1day with IG12=0/100 as input returns the same
    # parameter values as IRI_monthly_mean_par
    groups = [
        (F2m, F2min, F2max, 'F2'), (F1m, F1min, F1max, 'F1'),
        (Em, Emin, Emax, 'E'),
    ]  # Skip Es because interpolated for R12=10-180

    for monthly, dmin, dmax, name in groups:
        for key in monthly:
            arr = monthly[key]

            np.testing.assert_array_almost_equal(
                arr[..., 0], dmin[key], decimal=3,
                err_msg=f"{name}.{key} min mismatch (IG12=0)"
            )
            np.testing.assert_array_almost_equal(
                arr[..., 1], dmax[key], decimal=3,
                err_msg=f"{name}.{key} max mismatch (IG12=100)"
            )

    # Check that sun and mag are the same regardless of solar activity
    groups = [
        (sunm, sunmin, sunmax, 'sun'), (magm, magmin, magmax, 'mag')
    ]

    # Check that IRI_density_1day with IG12=0/100 as input returns the same
    # parameter values as IRI_monthly_mean_par
    for monthly, dmin, dmax, name in groups:
        for key in monthly:

            np.testing.assert_array_almost_equal(
                monthly[key], dmin[key], decimal=3,
                err_msg=f"{name}.{key} min mismatch (IG12=0)"
            )
            np.testing.assert_array_almost_equal(
                monthly[key], dmax[key], decimal=3,
                err_msg=f"{name}.{key} max mismatch (IG12=100)"
            )


def test_IRI_density_1day_for_monthly_mean_values_Es():
    """Test that IRI_density_1day returns expected Es values for IG12=0/100.

    IRI_density_1day calculates the sporadic E layer using IRI_monthly_mean_par
    and interpolates to the solar activity value requested from the R12=10 and
    R12=180 parameters given by IRI_monthly_mean_par for Es specifically. Given
    R12=10 or 180 as input, IRI_density_1day should return the same Es as
    IRI_monthly_mean_par.
    """
    year = 2024
    mth = 6
    day = 15
    aUT = np.array([12.0])
    alon = np.array([0.0])
    alat = np.array([0.0])
    aalt = np.array([100])
    coeff_dir = PyIRI.coeff_dir

    # Giving R12=10 as input to IRI_density_1day
    _, _, _, Esmin, _, _, _ = edp_update.IRI_density_1day(
        year, mth, day, aUT, alon, alat, aalt, R12_2_F107(10), coeff_dir)

    # Giving R12=180 as input to IRI_density_1day
    _, _, _, Esmax, _, _, _ = edp_update.IRI_density_1day(
        year, mth, day, aUT, alon, alat, aalt, R12_2_F107(180), coeff_dir)

    # Reference values for R12=10/180 obtained from IRI_monthly_mean_par
    _, _, _, Esm, _, _ = edp_update.IRI_monthly_mean_par(
        year, mth, aUT, alon, alat, coeff_dir)

    # Check that IRI_density_1day with R12=10/180 as input returns the same
    # Es values as IRI_monthly_mean_par
    for key in Esm:
        arr = Esm[key]

        np.testing.assert_array_almost_equal(
            arr[..., 0], Esmin[key], decimal=3,
            err_msg=f"{key} min mismatch (R12=10)"
        )
        np.testing.assert_array_almost_equal(
            arr[..., 1], Esmax[key], decimal=3,
            err_msg=f"{key} max mismatch (R12=180)"
        )


def test_Probability_F1_output():
    """Test output of Probability_F1 against expected values.

    Checks that Probability_F1 returns expected values and expected shape.
    """
    year = 2020
    month = 4
    aUT = np.arange(0, 24, 24)
    alon = np.array([10.])
    alat = np.array([20.])
    expected = np.array([[[0.00210238, 0.00210238]]])

    result = edp_update.Probability_F1(year, month, aUT, alon, alat)
    assert result.shape == expected.shape, (
        "Shape mismatch in Probability_F1 output")
    assert np.allclose(result, expected, atol=1e-8), (
        "Value mismatch in Probability_F1 output")


def test_drop_up_output():
    """Test drop_up returns expected value.

    Checks that drop_up returns expected values.
    """
    z_E = 230.0
    z_F = 110.0
    z_top = 250.0
    expected = 0.2245381938521092

    result = edp_update.drop_up(z_E, z_F, z_top, drop_fraction=0.2)
    assert abs(result - expected) < 1e-8, "drop_up output mismatch"


def test_logistic_curve_output():
    """Test logistic_curve returns expected value.

    Checks that logistic_curve returns expected values.
    """
    z = 240.0
    z0 = 200.0
    scale = 50.0
    expected = 0.6899744811276125

    result = edp_update.logistic_curve(z, z0, scale)
    assert abs(result - expected) < 1e-8, "logistic_curve output mismatch"


def test_derive_dependent_F1_parameters_output():
    """Test derive_dependent_F1_parameters output accuracy.

    Checks that derive_dependent_F1_parameters returns expected values.
    """
    p_F1 = np.array([0.5])
    NmF2 = np.array([1.0e11])
    hmF2 = np.array([350.])
    B_F2_bot = np.array([60.])
    z_E = np.array([110.])
    expected_NmF1 = np.array([5.54030525e+10])
    expected_B_F1_bot = np.array([2.1137616])
    expected_hmF1 = np.array([254.])
    expected_thickness = np.array([72.])

    NmF1, B_F1_bot, hmF1, thickness = edp_update.derive_dependent_F1_parameters(
        p_F1, NmF2, hmF2, B_F2_bot, z_E)

    assert np.allclose(NmF1, expected_NmF1, atol=1e-6), "NmF1 mismatch"
    assert np.allclose(B_F1_bot, expected_B_F1_bot,
                       atol=1e-6), "B_F1_bot mismatch"
    assert np.allclose(hmF1, expected_hmF1, atol=1e-6), "hmF1 mismatch"
    assert np.allclose(thickness, expected_thickness,
                       atol=1e-6), "thickness mismatch"
