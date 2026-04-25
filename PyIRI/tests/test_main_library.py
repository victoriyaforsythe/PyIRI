"""Unit tests for PyIRI's main_library."""

import numpy as np
import pytest

import PyIRI
import PyIRI.main_library as main
from PyIRI.main_library import to_numpy_array


def test_scalar_input():
    """Test to_numpy_array with a scalar input."""
    result = to_numpy_array(5.5)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)
    assert result[0] == 5.5


def test_list_input():
    """Test to_numpy_array with a list input."""
    result = to_numpy_array([1, 2, 3])
    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))
    assert result.dtype == float


def test_array_input():
    """Test to_numpy_array with an int array input."""
    arr = np.array([1, 2, 3], dtype=int)
    result = to_numpy_array(arr)
    np.testing.assert_array_equal(result, arr.astype(float))


def test_nested_list():
    """Test to_numpy_array with a 2d array input."""
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


def test_IRI_density_1day_runs():
    """Test that IRI_density_1day runs and returns expected shape."""
    year = 2024
    mth = 6
    day = 21
    aUT = np.array([12.0])
    alon = np.array([0.0])
    alat = np.array([0.0])
    aalt = np.linspace(100, 600, 20)
    F107 = 100.0
    coeff_dir = PyIRI.coeff_dir

    F2, F1, E, Es, sun, mag, EDP = main.IRI_density_1day(
        year, mth, day, aUT, alon, alat, aalt, F107, coeff_dir)
    assert EDP.shape[1] == len(aalt)
    assert np.ndim(F2['Nm']) == 2


def test_IRI_monthly_mean_par_runs():
    """Test that IRI_monthly_mean_par runs and returns expected shape."""
    year = 2024
    mth = 6
    aUT = np.array([12.0])
    alon = np.array([0.0])
    alat = np.array([0.0])
    coeff_dir = PyIRI.coeff_dir

    F2, F1, E, Es, sun, mag = main.IRI_monthly_mean_par(
        year, mth, aUT, alon, alat, coeff_dir)
    assert np.ndim(F2['Nm']) == 3


def test_IRI_density_1day_for_monthly_mean_values():
    """Test that IRI_density_1day returns expected values for IG12=0/100."""
    year = 2024
    mth = 6
    day = 15
    aUT = np.array([12.0])
    alon = np.array([0.0])
    alat = np.array([0.0])
    aalt = np.array([100])
    coeff_dir = PyIRI.coeff_dir

    F2min, F1min, Emin, _, sunmin, magmin, _ = main.IRI_density_1day(
        year, mth, day, aUT, alon, alat, aalt, main.IG12_2_F107(0), coeff_dir)

    F2m, F1m, Em, _, sunm, magm = main.IRI_monthly_mean_par(
        year, mth, aUT, alon, alat, coeff_dir)

    F2max, F1max, Emax, _, sunmax, magmax, _ = main.IRI_density_1day(
        year, mth, day, aUT, alon, alat, aalt, main.IG12_2_F107(100), coeff_dir)

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

    groups = [
        (sunm, sunmin, sunmax, 'sun'), (magm, magmin, magmax, 'mag')
    ]  # Check that sun and mag are the same regardless of solar activity

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
    """Test that IRI_density_1day returns expected Es values for IG12=0/100."""
    year = 2024
    mth = 6
    day = 15
    aUT = np.array([12.0])
    alon = np.array([0.0])
    alat = np.array([0.0])
    aalt = np.array([100])
    coeff_dir = PyIRI.coeff_dir

    _, _, _, Esmin, _, _, _ = main.IRI_density_1day(
        year, mth, day, aUT, alon, alat, aalt, main.R12_2_F107(10), coeff_dir)

    _, _, _, Esm, _, _ = main.IRI_monthly_mean_par(
        year, mth, aUT, alon, alat, coeff_dir)

    _, _, _, Esmax, _, _, _ = main.IRI_density_1day(
        year, mth, day, aUT, alon, alat, aalt, main.R12_2_F107(180), coeff_dir)

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


@pytest.mark.parametrize(
    "solidx, F107",
    [
        ('IG12', main.IG12_2_F107(0)),
        ('R12', main.R12_2_F107(0)),
        ('F107', 0)
    ]
)
def test_solar_interpolate_args(solidx, F107):
    """Test all solidx arguments for solar_interpolate."""
    F_min = np.array([0., 100.])
    F_max = np.array([100., 200.])
    result = main.solar_interpolate(F_min, F_max, F107, solidx=solidx)
    np.testing.assert_array_almost_equal(result, F_min, decimal=4)


def test_solar_interpolate_invalid_solidx_raises():
    """Test invalid argument for solar_interpolate."""
    with pytest.raises(ValueError):
        main.solar_interpolate([1, 2], [3, 4], 100, solidx='INVALID')


def test_solar_interpolate_list_input():
    """Test list input for solar_interpolate."""
    list_input = main.solar_interpolate([1], [3], 100.0)
    arr_input = main.solar_interpolate(np.array([1]), np.array([3]), 100.0)
    np.testing.assert_array_almost_equal(list_input, arr_input, decimal=4)


def test_deprecated_function_warns():
    """Test deprecated solar_interpolate_R12."""
    with pytest.warns(DeprecationWarning):
        main.solar_interpolate_R12([1, 2], [3, 4], 50.0)


def test_deprecated_argument_warns():
    """Test deprecation in solar_interpolation_of_dictionary."""
    F = {'a': np.ones((3, 1, 2))}
    with pytest.warns(DeprecationWarning):
        main.solar_interpolation_of_dictionary(F, 100, use_R12=True)


def test_solar_interpolate_ig12_midpoint():
    """Test solar_interpolate using IG12."""
    F_min = np.array([0., 100.])
    F_max = np.array([100., 200.])
    F107 = main.IG12_2_F107(50)
    result = main.solar_interpolate(F_min, F_max, F107, solidx='IG12')
    np.testing.assert_array_almost_equal(result, [50., 150.], decimal=4)


def test_solar_interpolate_r12_midpoint():
    """Test solar_interpolate using R12."""
    F_min = np.array([0., 100.])
    F_max = np.array([100., 200.])
    F107 = main.R12_2_F107(50)
    result = main.solar_interpolate(F_min, F_max, F107, solidx='R12')
    np.testing.assert_array_almost_equal(result, [50., 150.], decimal=4)


def test_solar_interpolate_bounds():
    """Test solar_interpolate for F107=solmin or solmax."""
    F_min = np.array([10.])
    F_max = np.array([20.])
    low = main.solar_interpolate(F_min, F_max, F107=main.IG12_2_F107(0),
                                 solmin=0, solmax=100)
    high = main.solar_interpolate(F_min, F_max, F107=main.IG12_2_F107(100),
                                  solmin=0, solmax=100)
    np.testing.assert_almost_equal(low[0], 10.)
    np.testing.assert_almost_equal(high[0], 20.)


def test_solar_interpolation_of_dictionary_structure():
    """Testsolar_interpolation_of_dictionary for structure and validity."""
    F = {
        'param1': np.array([[[3, 9], [4, 12], [5, 15]]]),
        'param2': np.array([[[1.5, 4.5], [-1, 1], [0, 3]]])
    }  # shape (N_T, N_G, 2) = (1, 3, 2)
    res = main.solar_interpolation_of_dictionary(F, F107=main.IG12_2_F107(50))
    assert set(res.keys()) == {'param1', 'param2'}
    assert res['param1'].shape == (1, 3)
    np.testing.assert_almost_equal(res['param1'], [[6, 8, 10]], decimal=4)
    np.testing.assert_almost_equal(res['param2'], [[3, 0, 1.5]], decimal=4)


def test_solar_interpolation_of_dictionary_F2_shu2015_uses_ig12_for_hm():
    """Test solar_interpolation_of_dictionary_F2 for SHU2015 hmF2 model."""
    F = {
        'hm': np.array([[[3, 9], [4, 12], [5, 15]]])
    }
    F_copy = F.copy()
    result = main.solar_interpolation_of_dictionary_F2(F, F107=50,
                                                       hmF2_model='SHU2015')
    truth = main.solar_interpolate(F_copy['hm'][:, :, 0], F_copy['hm'][:, :, 1],
                                   F107=50)
    assert all(k in result for k in F)
    assert result['hm'].shape == (1, 3)
    assert np.all(result['hm'] == truth)


def test_solar_interpolation_of_dictionary_F2_amtb2013_uses_r12_for_hm():
    """Test solar_interpolation_of_dictionary_F2 for AMTB2013 hmF2 model."""
    F = {
        'hm': np.array([[[3, 9], [4, 12], [5, 15]]])
    }
    F_copy = F.copy()
    result = main.solar_interpolation_of_dictionary_F2(F, F107=50,
                                                       hmF2_model='AMTB2013')
    truth = main.solar_interpolate(F_copy['hm'][:, :, 0], F_copy['hm'][:, :, 1],
                                   F107=50, solidx='R12')
    assert all(k in result for k in F)
    assert result['hm'].shape == (1, 3)
    assert np.all(result['hm'] == truth)


def test_freq2den():
    """Test freq2den."""
    freq = 3e6  # MHz
    truth = 1.24e10 * freq**2
    np.testing.assert_array_almost_equal(main.freq2den(freq), truth)


def test_den2freq():
    """Test den2freq."""
    Nm = 1e12  # m-3
    truth = np.sqrt(Nm / 1.24e10)
    np.testing.assert_array_almost_equal(main.den2freq(Nm), truth)


def test_R12_F107_roundtrip():
    """R12 -> F107 -> R12 returns approximately original value."""
    original = np.array([0., 50., 100., 200.])
    f107 = main.R12_2_F107(original)
    recovered = main.F107_2_R12(f107)
    np.testing.assert_array_almost_equal(original, recovered, decimal=4)


def test_F107_2_R12_known_value():
    """Known conversion: R12=0 -> F107=63.75."""
    result = main.R12_2_F107(0)
    np.testing.assert_array_almost_equal(result, 63.75)


def test_IG12_R12_v2_vs_v1():
    """Version 1 and 2 give different results."""
    r12 = 100
    ig12_v1 = main.R12_2_IG12(r12, version=1)
    ig12_v2 = main.R12_2_IG12(r12, version=2)
    assert ig12_v1 != ig12_v2


def test_IG12_R12_roundtrip_v1():
    """IG12 <-> R12 roundtrip for version 1."""
    original = np.array([10., 50., 100.])
    ig12 = main.R12_2_IG12(original, version=1)
    recovered = main.IG12_2_R12(ig12, version=1)
    np.testing.assert_array_almost_equal(original, recovered, decimal=4)


def test_IG12_R12_roundtrip_v2():
    """IG12 <-> R12 roundtrip for version 2."""
    original = np.array([10., 50., 100.])
    ig12 = main.R12_2_IG12(original, version=2)
    recovered = main.IG12_2_R12(ig12, version=2)
    np.testing.assert_array_almost_equal(original, recovered, decimal=4)


def test_F107_IG12_chain_consistency():
    """F107 -> R12 -> IG12 matches direct F107_2_IG12."""
    f107 = 100.
    r12 = main.F107_2_R12(f107)
    ig12_manual = main.R12_2_IG12(r12, version=2)
    ig12_direct = main.F107_2_IG12(f107, version=2)
    assert ig12_manual == ig12_direct
