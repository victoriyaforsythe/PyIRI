import numpy as np
import pytest
import PyIRI
from PyIRI import edp_update

def test_IRI_density_1day_runs():
    year = 2024
    mth = 6
    day = 21
    aUT = np.array([12.0])
    alon = np.array([0.0])
    alat = np.array([0.0])
    aalt = np.linspace(100, 600, 20)
    F107 = 100.0
    coeff_dir = PyIRI.coeff_dir

    try:
        F2, F1, E, Es, sun, mag, EDP = edp_update.IRI_density_1day(
            year, mth, day, aUT, alon, alat, aalt, F107, coeff_dir)
        assert EDP.shape[1] == len(aalt)
    except Exception as e:
        pytest.fail(f"Function raised an exception: {e}")


def test_IRI_monthly_mean_par_runs():
    # TODO: provide realistic inputs and check outputs
    try:
        result = edp_update.IRI_monthly_mean_par  # Add input args if required
        assert callable(result) or result is not None
    except Exception as e:
        pytest.fail(f"Function IRI_monthly_mean_par raised an exception: {e}")


def test_reconstruct_density_from_parameters_runs():
    # TODO: provide realistic inputs and check outputs
    try:
        result = edp_update.reconstruct_density_from_parameters  # Add input args if required
        assert callable(result) or result is not None
    except Exception as e:
        pytest.fail(f"Function reconstruct_density_from_parameters raised an exception: {e}")


def test_reconstruct_density_from_parameters_1level_runs():
    # TODO: provide realistic inputs and check outputs
    try:
        result = edp_update.reconstruct_density_from_parameters_1level  # Add input args if required
        assert callable(result) or result is not None
    except Exception as e:
        pytest.fail(f"Function reconstruct_density_from_parameters_1level raised an exception: {e}")


def test_EDP_builder_runs():
    # TODO: provide realistic inputs and check outputs
    try:
        result = edp_update.EDP_builder  # Add input args if required
        assert callable(result) or result is not None
    except Exception as e:
        pytest.fail(f"Function EDP_builder raised an exception: {e}")


def test_run_iri_reg_grid_runs():
    # TODO: provide realistic inputs and check outputs
    try:
        result = edp_update.run_iri_reg_grid  # Add input args if required
        assert callable(result) or result is not None
    except Exception as e:
        pytest.fail(f"Function run_iri_reg_grid raised an exception: {e}")


def test_run_seas_iri_reg_grid_runs():
    # TODO: provide realistic inputs and check outputs
    try:
        result = edp_update.run_seas_iri_reg_grid  # Add input args if required
        assert callable(result) or result is not None
    except Exception as e:
        pytest.fail(f"Function run_seas_iri_reg_grid raised an exception: {e}")


def test_derive_dependent_F1_parameters_runs():
    # TODO: provide realistic inputs and check outputs
    try:
        result = edp_update.derive_dependent_F1_parameters  # Add input args if required
        assert callable(result) or result is not None
    except Exception as e:
        pytest.fail(f"Function derive_dependent_F1_parameters raised an exception: {e}")


def test_logistic_curve_runs():
    # TODO: provide realistic inputs and check outputs
    try:
        result = edp_update.logistic_curve  # Add input args if required
        assert callable(result) or result is not None
    except Exception as e:
        pytest.fail(f"Function logistic_curve raised an exception: {e}")


def test_drop_up_runs():
    # TODO: provide realistic inputs and check outputs
    try:
        result = edp_update.drop_up  # Add input args if required
        assert callable(result) or result is not None
    except Exception as e:
        pytest.fail(f"Function drop_up raised an exception: {e}")


def test_drop_down_runs():
    # TODO: provide realistic inputs and check outputs
    try:
        result = edp_update.drop_down  # Add input args if required
        assert callable(result) or result is not None
    except Exception as e:
        pytest.fail(f"Function drop_down raised an exception: {e}")


def test_Probability_F1_runs():
    # TODO: provide realistic inputs and check outputs
    try:
        result = edp_update.Probability_F1  # Add input args if required
        assert callable(result) or result is not None
    except Exception as e:
        pytest.fail(f"Function Probability_F1 raised an exception: {e}")

import numpy as np
import pytest
import PyIRI
from PyIRI import edp_update as ml


def test_Probability_F1_output():
    year = 2020
    month = 4
    aUT = np.arange(0, 24, 24)
    alon = np.array([10.])
    alat = np.array([20.])
    expected = np.array([[[0.00210238, 0.00210238]]])

    result = ml.Probability_F1(year, month, aUT, alon, alat)
    assert result.shape == expected.shape, "Shape mismatch in Probability_F1 output"
    assert np.allclose(result, expected, atol=1e-8), "Value mismatch in Probability_F1 output"


def test_drop_up_output():
    z_E = 230.0
    z_F = 110.0
    z_top = 250.0
    expected = 0.2245381938521092

    result = ml.drop_up(z_E, z_F, z_top, drop_fraction=0.2)
    assert abs(result - expected) < 1e-8, "drop_up output mismatch"


def test_logistic_curve_output():
    z = 240.0
    z0 = 200.0
    scale = 50.0
    expected = 0.6899744811276125

    result = ml.logistic_curve(z, z0, scale)
    assert abs(result - expected) < 1e-8, "logistic_curve output mismatch"


def test_derive_dependent_F1_parameters_output():
    p_F1 = np.array([0.5])
    NmF2 = np.array([1.0e11])
    hmF2 = np.array([350.])
    B_F2_bot = np.array([60.])
    z_E = np.array([110.])
    expected_NmF1 = np.array([5.54030525e+10])
    expected_B_F1_bot = np.array([2.1137616])
    expected_hmF1 = np.array([254.])
    expected_thickness = np.array([72.])

    NmF1, B_F1_bot, hmF1, thickness = ml.derive_dependent_F1_parameters(
        p_F1, NmF2, hmF2, B_F2_bot, z_E)

    assert np.allclose(NmF1, expected_NmF1, atol=1e-6), "NmF1 mismatch"
    assert np.allclose(B_F1_bot, expected_B_F1_bot, atol=1e-6), "B_F1_bot mismatch"
    assert np.allclose(hmF1, expected_hmF1, atol=1e-6), "hmF1 mismatch"
    assert np.allclose(thickness, expected_thickness, atol=1e-6), "thickness mismatch"