"""Unit tests for PyIRI's main_library."""

import numpy as np
import pytest

import PyIRI
import PyIRI.main_library as main
from PyIRI.main_library import to_numpy_array


def test_scalar_input():
    """Test to_numpy_array with a scalar input.

    Tests whether to_numpy_array correctly converts a scalar input into a float
    numpy array.
    """
    result = to_numpy_array(5.5)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)
    assert result[0] == 5.5


def test_list_input():
    """Test to_numpy_array with a list input.

    Tests whether to_numpy_array correctly converts a list input into a float
    numpy array.
    """
    result = to_numpy_array([1, 2, 3])
    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))
    assert result.dtype == float


def test_array_input():
    """Test to_numpy_array with an int array input.

    Tests whether to_numpy_array correctly converts an int numpy array input
    into a float numpy array.
    """
    arr = np.array([1, 2, 3], dtype=int)
    result = to_numpy_array(arr)
    np.testing.assert_array_equal(result, arr.astype(float))


def test_nested_list():
    """Test to_numpy_array with a 2d input.

    Tests whether to_numpy_array works as expected with a nested list input.
    """
    result = to_numpy_array([[1, 2], [3, 4]])
    assert result.shape == (2, 2)
    assert result.dtype == float


def test_fo_1day_interpolation():
    """Test linear interpolation of fo.

    Tests whether IRI_density_1day handles the linear interpolation of fo
    correctly. In the IRI_density_1day function, Nm should be computed directly
    from the interpolated value of fo. Nm should NOT be linearly interpolated.
    If Nm is linearly interpolated, this test will fail.
    """
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

    Nm_fn_output = []
    Nm_from_fo = []
    for layer in [F2, F1, E, Es]:
        Nm_fn_output.append(layer['Nm'][0, 0])
        fo_fn_output = layer['fo'][0, 0]
        Nm_from_fo.append(main.freq2den(fo_fn_output))

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

    F2, F1, E, Es, sun, mag, EDP = main.IRI_density_1day(
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

    F2, F1, E, Es, sun, mag = main.IRI_monthly_mean_par(
        year, mth, aUT, alon, alat, coeff_dir)

    assert F2['Nm'].shape == (N_T, N_G, 2)
    assert sun['lat'].shape == (N_T,)


@pytest.mark.parametrize('mth', range(1, 13))
def test_read_ccir_ursi_coeff_all_months(mth):
    """Test read_ccir_ursi_coeff for every month.

    Coefficients must be fully populated finite floats of the shape set by
    highest_power_of_extension, for every month's CCIR/URSI/Es file,
    including the CCIR files' final, partially filled record.
    """
    coeff_dir = PyIRI.coeff_dir
    coef = main.highest_power_of_extension()

    F_CCIR, F_URSI, F_M3000, F_Es = main.read_ccir_ursi_coeff(
        mth, coeff_dir)

    assert F_CCIR.shape == (coef['nj']['F0F2'], coef['nk']['F0F2'], 2)
    assert F_URSI.shape == (coef['nj']['F0F2'], coef['nk']['F0F2'], 2)
    assert F_M3000.shape == (coef['nj']['M3000'], coef['nk']['M3000'], 2)
    for arr in (F_CCIR, F_URSI, F_M3000, F_Es):
        # CCIR/M3000 arrays are dtype=object (a pre-existing quirk from a
        # None-padded partial line in the .asc files), so cast before
        # checking finiteness.
        assert np.isfinite(arr.astype(float)).all()


def test_read_ccir_ursi_coeff_cache_does_not_leak_mutations():
    """Test that caching in read_ccir_ursi_coeff does not alias arrays.

    read_ccir_ursi_coeff caches the parsed coefficient arrays internally for
    performance. Each call must still return an independent array so that a
    caller mutating one result cannot corrupt another call's result.
    """
    coeff_dir = PyIRI.coeff_dir

    first = main.read_ccir_ursi_coeff(3, coeff_dir)
    first[0][:] = np.nan

    second = main.read_ccir_ursi_coeff(3, coeff_dir)

    assert not np.isnan(second[0].astype(float)).any()


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
    F2min, F1min, Emin, _, sunmin, magmin, _ = main.IRI_density_1day(
        year, mth, day, aUT, alon, alat, aalt, main.IG12_2_F107(0), coeff_dir)

    # Giving IG12=100 as input to IRI_density_1day
    F2max, F1max, Emax, _, sunmax, magmax, _ = main.IRI_density_1day(
        year, mth, day, aUT, alon, alat, aalt, main.IG12_2_F107(100), coeff_dir)

    # Reference values for IG12=0/100 obtained from IRI_monthly_mean_par
    F2m, F1m, Em, _, sunm, magm = main.IRI_monthly_mean_par(
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
    _, _, _, Esmin, _, _, _ = main.IRI_density_1day(
        year, mth, day, aUT, alon, alat, aalt, main.R12_2_F107(10), coeff_dir)

    # Giving R12=180 as input to IRI_density_1day
    _, _, _, Esmax, _, _, _ = main.IRI_density_1day(
        year, mth, day, aUT, alon, alat, aalt, main.R12_2_F107(180), coeff_dir)

    # Reference values for R12=10/180 obtained from IRI_monthly_mean_par
    _, _, _, Esm, _, _ = main.IRI_monthly_mean_par(
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


@pytest.mark.parametrize(
    "solidx, F107",
    [
        ('IG12', main.IG12_2_F107(0)),
        ('R12', main.R12_2_F107(0)),
        ('F107', 0)
    ]
)
def test_solar_interpolate_args(solidx, F107):
    """Test all solar index inputs for solar_interpolate.

    The solar_interpolate function takes two ionospheric parameter arrays F_min
    and F_max and interpolates linearly to a chosen value of F10.7 assuming that
    F_min corresponds to solar min and F_max to solar max. The solar index used
    for characterizing min and max is given by solidx (default='IG12'), the min
    and max values are given by solmin (default=0) and solmax (default=100).

    Using default solmin=0 and solmax=100, this test checks:
        If given F107=IG12_2_F107(0) and solidx='IG12', must return F_min.
        If given F107=R12_2_F107(0) and solidx='R12', must return F_min.
        If given F107=0 and solidx='F107', must return F_min.

    Parameters
    ------------
    solidx : str
        Solar index name, IG12, R12, or F107.
    F107 : int or float
        F10.7 solar index value.
    """
    F_min = np.array([0., 100.])
    F_max = np.array([100., 200.])
    result = main.solar_interpolate(F_min, F_max, F107, solidx=solidx,
                                    solmin=0, solmax=100)
    np.testing.assert_array_almost_equal(result, F_min, decimal=4)


def test_solar_interpolate_invalid_solidx_raises():
    """Test invalid argument for solar_interpolate.

    The solar_interpolate function only accepts 'IG12', 'R12', and 'F107' for
    solidx inputs. Must return value error.
    """
    with pytest.raises(ValueError, match="Solar index unknown"):
        main.solar_interpolate([1, 2], [3, 4], 100, solidx='INVALID')


def test_solar_interpolate_input_type():
    """Test list input for solar_interpolate.

    The solar_interpolate function can accept scalars, lists, or arrays.
    Conversion to numpy float arrays is done internally.
    """
    int_input = main.solar_interpolate(1, 3, 100.0)
    list_input = main.solar_interpolate([1], [3], 100.0)
    arr_input = main.solar_interpolate(np.array([1]), np.array([3]), 100.0)
    np.testing.assert_array_almost_equal(list_input, arr_input, decimal=4)
    np.testing.assert_array_almost_equal(int_input, arr_input, decimal=4)


def test_deprecated_function_warns():
    """Test deprecated solar_interpolate_R12.

    The function solar_interpolate_R12 is deprecated. Usage must return a
    deprecation warning.
    """
    with pytest.warns(FutureWarning, match="Use solar_interpolate"):
        main.solar_interpolate_R12([1, 2], [3, 4], 50.0)


def test_deprecated_argument_warns():
    """Test deprecation in solar_interpolation_of_dictionary.

    The argument use_R12 (default=False) in the function
    solar_interpolation_of_dictionary is deprecated. Usage must return a
    deprecation warning.
    """
    F = {'a': np.ones((3, 1, 2))}
    with pytest.warns(FutureWarning,
                      match="use_R12 is deprecated"):
        main.solar_interpolation_of_dictionary(F, 100, use_R12=True)


def test_solar_interpolate_ig12_midpoint():
    """Test solar_interpolate using IG12.

    The solar_interpolate function takes two ionospheric parameter arrays F_min
    and F_max and interpolates linearly to a chosen value of F10.7 assuming that
    F_min corresponds to solar min and F_max to solar max. The solar index used
    for characterizing min and max is given by solidx (default='IG12'), the min
    and max values are given by solmin (default=0) and solmax (default=100).

    Using solmin=0, solmax=100, and solidx='IG12', this test verifies that with
    F107=IG12_2_F107(50) we obtain the average of F_min and F_max.
    """
    F_min = np.array([0., 100.])
    F_max = np.array([100., 200.])
    F107 = main.IG12_2_F107(50)

    result = main.solar_interpolate(F_min, F_max, F107, solidx='IG12',
                                    solmin=0, solmax=100)
    np.testing.assert_array_almost_equal(result, [50., 150.], decimal=4)


def test_solar_interpolate_r12_midpoint():
    """Test solar_interpolate using R12.

    The solar_interpolate function takes two ionospheric parameter arrays F_min
    and F_max and interpolates linearly to a chosen value of F10.7 assuming that
    F_min corresponds to solar min and F_max to solar max. The solar index used
    for characterizing min and max is given by solidx (default='IG12'), the min
    and max values are given by solmin (default=0) and solmax (default=100).

    Using solmin=0, solmax=100, and solidx='R12', this test verifies that with
    F107=R12_2_F107(50) we obtain the average of F_min and F_max.
    """
    F_min = np.array([0., 100.])
    F_max = np.array([100., 200.])
    F107 = main.R12_2_F107(50)

    result = main.solar_interpolate(F_min, F_max, F107, solidx='R12',
                                    solmin=0, solmax=100)
    np.testing.assert_array_almost_equal(result, [50., 150.], decimal=4)


def test_solar_interpolate_bounds():
    """Test solar_interpolate for F107=solmin or solmax.

    The solar_interpolate function takes two ionospheric parameter arrays F_min
    and F_max and interpolates linearly to a chosen value of F10.7 assuming that
    F_min corresponds to solar min and F_max to solar max. The solar index used
    for characterizing min and max is given by solidx (default='IG12'), the min
    and max values are given by solmin (default=0) and solmax (default=100).

    Using solmin=0, solmax=100, and solidx='F107', this test verifies that with
    F107=solmin/solmax we obtain the F_min/F_max.
    """
    F_min = np.array([10.])
    F_max = np.array([20.])
    low = main.solar_interpolate(F_min, F_max, F107=0, solidx='F107',
                                 solmin=0, solmax=100)
    high = main.solar_interpolate(F_min, F_max, F107=100, solidx='F107',
                                  solmin=0, solmax=100)
    np.testing.assert_almost_equal(low[0], 10.)
    np.testing.assert_almost_equal(high[0], 20.)


def test_solar_interpolation_of_dictionary_structure():
    """Test solar_interpolation_of_dictionary for structure and validity.

    The solar_interpolation_of_doctionary function takes a dictionary of
    ionospheric parameters for both min and max solar activity and interpolates
    linearly to a chosen value of F10.7. The solar index used for characterizing
    min and max is given by solidx (default='IG12'), the min and max values are
    given by solmin (default=0) and solmax (default=100).

    Using a dictionary of shape (N_T, N_G, 2), solidx='IG12', solmin=0,
    solmax=100, and F107=IG12_2_F107(50), this test verifies that the resulting
    dictionary has the shape (N_T, N_G) with the same keys as the output and
    returns the average values of solar min/max.
    """
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
    """Test solar_interpolation_of_dictionary_F2 for SHU2015 hmF2 model.

    The solar_interpolation_of_doctionary_F2 function takes the F2 dictionary of
    ionospheric parameters for both min and max solar activity and interpolates
    linearly to a chosen value of F10.7 using the approriate solar index and
    solar min/max bounds for each parameter.

    This test verifies that if using the Shu-2015 model for hmF2, IG12 is used
    as the solar index to interpolate with, along with solmin=0 and solmax=100.
    """
    F = {
        'hm': np.array([[[3, 9], [4, 12], [5, 15]]])
    }
    F_copy = F.copy()
    result = main.solar_interpolation_of_dictionary_F2(F, F107=50,
                                                       hmF2_model='SHU2015')
    truth = main.solar_interpolate(F_copy['hm'][:, :, 0], F_copy['hm'][:, :, 1],
                                   F107=50, solidx='IG12', solmin=0, solmax=100)
    assert all(k in result for k in F)
    assert result['hm'].shape == (1, 3)
    assert np.all(result['hm'] == truth)


def test_solar_interpolation_of_dictionary_F2_amtb2013_uses_r12_for_hm():
    """Test solar_interpolation_of_dictionary_F2 for AMTB2013 hmF2 model.

    The solar_interpolation_of_doctionary_F2 function takes the F2 dictionary of
    ionospheric parameters for both min and max solar activity and interpolates
    linearly to a chosen value of F10.7 using the approriate solar index and
    solar min/max bounds for each parameter.

    This test verifies that if using the AMTB-2013 model for hmF2, R12 is used
    as the solar index to interpolate with, along with solmin=0 and solmax=100.
    """
    F = {
        'hm': np.array([[[3, 9], [4, 12], [5, 15]]])
    }
    F_copy = F.copy()
    result = main.solar_interpolation_of_dictionary_F2(F, F107=50,
                                                       hmF2_model='AMTB2013')
    truth = main.solar_interpolate(F_copy['hm'][:, :, 0], F_copy['hm'][:, :, 1],
                                   F107=50, solidx='R12', solmin=0, solmax=100)
    assert all(k in result for k in F)
    assert result['hm'].shape == (1, 3)
    assert np.all(result['hm'] == truth)


def test_freq2den():
    """Test freq2den.

    Checks whether the freq2den conversion is implemented correctly.
    """
    freq = 3e6  # MHz
    truth = 1.24e10 * freq**2
    np.testing.assert_array_almost_equal(main.freq2den(freq), truth)


def test_den2freq():
    """Test den2freq.

    Checks whether the den2freq conversion is implemented correctly.
    """
    Nm = 1e12  # m-3
    truth = np.sqrt(Nm / 1.24e10)
    np.testing.assert_array_almost_equal(main.den2freq(Nm), truth)


def test_R12_F107_roundtrip():
    """R12 -> F107 -> R12 returns approximately original value.

    Checks whether the R12-F107 conversions are consistent.
    """
    original = np.array([0., 50., 100., 200.])
    f107 = main.R12_2_F107(original)
    recovered = main.F107_2_R12(f107)
    np.testing.assert_array_almost_equal(original, recovered, decimal=4)


def test_F107_2_R12_known_value():
    """Known conversion: R12=0 -> F107=63.75.

    Checks whether the F10.7 to R12 conversion is implemented correctly.
    """
    result = main.R12_2_F107(0)
    np.testing.assert_array_almost_equal(result, 63.75)


def test_IG12_R12_v2_vs_v1():
    """Version 1 and 2 give different results.

    Checks whether version=1 and version=2 return different results in the R12
    to IG12 conversion.
    """
    r12 = 100
    ig12_v1 = main.R12_2_IG12(r12, version=1)
    ig12_v2 = main.R12_2_IG12(r12, version=2)
    assert ig12_v1 != ig12_v2


def test_IG12_R12_roundtrip_v1():
    """IG12 <-> R12 roundtrip for version 1.

    Checks whether the IG12-R12 conversions are consistent for version 1.
    """
    original = np.array([10., 50., 100.])
    ig12 = main.R12_2_IG12(original, version=1)
    recovered = main.IG12_2_R12(ig12, version=1)
    np.testing.assert_array_almost_equal(original, recovered, decimal=4)


def test_IG12_R12_roundtrip_v2():
    """IG12 <-> R12 roundtrip for version 2.

    Checks whether the IG12-R12 conversions are consistent for version 2.
    """
    original = np.array([10., 50., 100.])
    ig12 = main.R12_2_IG12(original, version=2)
    recovered = main.IG12_2_R12(ig12, version=2)
    np.testing.assert_array_almost_equal(original, recovered, decimal=4)


def test_F107_IG12_chain_consistency():
    """F107 -> R12 -> IG12 matches direct F107_2_IG12.

    Checks whether the F107 to R12 to IG12 conversion is consistent with the
    direct F107 to IG12 conversion.
    """
    f107 = 100.
    r12 = main.F107_2_R12(f107)
    ig12_manual = main.R12_2_IG12(r12, version=2)
    ig12_direct = main.F107_2_IG12(f107, version=2)
    assert ig12_manual == ig12_direct
