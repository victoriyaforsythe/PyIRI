"""Unit tests for PyIRI's sh_library."""

import datetime as dt

import numpy as np
import pytest

import PyIRI
from PyIRI.main_library import freq2den
from PyIRI.main_library import IG12_2_F107
import PyIRI.sh_library as sh


def test_deprecated_argument_default_warns():
    """Test deprecation of default argument old_output.

    The argument old_output will change default from True to False in version
    0.2+. Usage of related functions must return a warning.
    """
    year = 2020
    month = 12
    day = 10
    aUT = 0
    alon = 10
    alat = 10
    aalt = 100
    F107 = 124

    hr_res = 24
    lat_res = 90
    lon_res = 180
    alt_res = 100
    alt_min = 100
    alt_max = 200

    with pytest.warns(FutureWarning,
                      match="IRI_monthly_mean_par"):
        sh.IRI_monthly_mean_par(year, month, aUT, alon, alat)
    with pytest.warns(FutureWarning,
                      match="IRI_density_1day"):
        sh.IRI_density_1day(year, month, day, aUT, alon, alat, aalt, F107)
    with pytest.warns(FutureWarning,
                      match="run_iri_reg_grid"):
        sh.run_iri_reg_grid(year, month, day, F107, hr_res=hr_res,
                            lat_res=lat_res, lon_res=lon_res, alt_res=alt_res,
                            alt_min=alt_min, alt_max=alt_max)
    with pytest.warns(FutureWarning,
                      match="run_seas_iri_reg_grid"):
        sh.run_seas_iri_reg_grid(year, month, hr_res=hr_res, lat_res=lat_res,
                                 lon_res=lon_res, alt_res=alt_res,
                                 alt_min=alt_min, alt_max=alt_max)


def test_sporadic_E_monthly_mean_deprecated_function_warns():
    """Test deprecated sporadic_E_monthly_mean.

    The function sporadic_E_monthly_mean is deprecated. Usage must return a
    deprecation warning.
    """
    with pytest.warns(FutureWarning, match="IRI_monthly_mean_par"):
        sh.sporadic_E_monthly_mean(2020, 12, 0, 10, 10)


def test_sporadic_E_1day_deprecated_function_warns():
    """Test deprecated sporadic_E_density_1day.

    The function sporadic_E_density_1day is deprecated. Usage must return a
    deprecation warning.
    """
    with pytest.warns(FutureWarning, match="IRI_density_1day"):
        sh.sporadic_E_1day(2020, 12, 10, 0, 10, 10, 124)


def test_fo_1day_interpolation():
    """Test linear interpolation of fo.

    Tests whether IRI_density_1day handles the linear interpolation of fo
    correctly. In the IRI_density_1day function, Nm should be computed directly
    from the interpolated value of fo. Nm should NOT be linearly interpolated.
    If Nm is linearly interpolated, this test will fail.
    """
    F2, F1, E, Es, *_ = sh.IRI_density_1day(2024,
                                            3,
                                            8,
                                            0,
                                            20,
                                            40,
                                            0,
                                            IG12_2_F107(40),
                                            old_output=False)

    Nm_fn_output = []
    Nm_from_fo = []
    for layer in [F2, F1, E, Es]:
        Nm_fn_output.append(layer['Nm'][0, 0])
        fo_fn_output = layer['fo'][0, 0]
        Nm_from_fo.append(freq2den(fo_fn_output))

    np.testing.assert_array_almost_equal(
        Nm_fn_output, Nm_from_fo, decimal=4,
        err_msg=("Nm/fo interpolation error: Nm from function is "
                 + f"{Nm_fn_output}, should be {Nm_from_fo}. Verify that fo is "
                 + "linearly interpolated, not Nm.")
    )


def test_IRI_density_1day_runs():
    """Test that IRI_density_1day runs and returns expected shape.

    IRI_density_1day must return dictionaries of shape (N_T, N_G) for
    ionospheric parameters, shape (N_T,) for solar and magnetic parameters, and
    shape (N_T, N_G, N_V) for the EDP array.
    """
    year = 2024
    mth = 6
    day = 21
    aUT = [12.0, 13.0]                # N_T = 2
    alon = [0.0]                      # N_G = 1
    alat = [0.0]
    aalt = np.linspace(100, 600, 10)  # N_V = 10
    F107 = 100.0

    N_T = len(aUT)
    N_G = len(alon)
    N_V = len(aalt)

    F2, F1, E, Es, sun, mag, EDP = sh.IRI_density_1day(
        year, mth, day, aUT, alon, alat, aalt, F107, old_output=False)

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
    aUT = [12.0, 13.0]                # N_T = 2
    alon = [0.0]                      # N_G = 1
    alat = [0.0]

    N_T = len(aUT)
    N_G = len(alon)

    F2, F1, E, Es, sun, mag = sh.IRI_monthly_mean_par(
        year, mth, aUT, alon, alat, old_output=False)

    assert F2['Nm'].shape == (N_T, N_G, 2)
    assert sun['lat'].shape == (N_T,)


def test_IRI_density_1day_for_monthly_mean_values():
    """Test that IRI_density_1day returns expected values for IG12=0/100.

    Unlike in main_library or edp_update, IRI_density_1day in sh_library
    calculates iono paramaters to the desired solar activity directly without
    calling IRI_monthly_mean_par. IRI_monthly_mean_par then calls
    IRI_density_1day twice to generate iono parameters for one solar min and one
    solar max. This is because not all parameters use the same solar index and
    solar min/max values in implementation of IRI_density_1day.

    Still, given IG12=0 or 100 as input, IRI_density_1day should return the same
    parameter values as IRI_monthly_mean_par using solidx='IG12', solmin=0, and
    solmax=100.
    """
    year = 2024
    mth = 6
    day = 15
    aUT = 12
    alon = 30
    alat = 20
    aalt = 100

    # Giving IG12=0 as input to IRI_density_1day
    F2min, F1min, Emin, _, sunmin, magmin, _ = sh.IRI_density_1day(
        year, mth, day, aUT, alon, alat, aalt, IG12_2_F107(0),
        old_output=False)

    # Giving IG12=100 as input to IRI_density_1day
    F2max, F1max, Emax, _, sunmax, magmax, _ = sh.IRI_density_1day(
        year, mth, day, aUT, alon, alat, aalt, IG12_2_F107(100),
        old_output=False)

    # Reference values for IG12=0/100 obtained from IRI_monthly_mean_par
    F2m, F1m, Em, _, sunm, magm = sh.IRI_monthly_mean_par(
        year, mth, aUT, alon, alat, old_output=False)

    # Check that IRI_density_1day with IG12=0/100 as input returns the same
    # parameter values as IRI_monthly_mean_par
    groups = [(F2m, F2min, F2max, 'F2'), (F1m, F1min, F1max, 'F1'),
              (Em, Emin, Emax, 'E'),]
    # Skip Es because interpolated for R12=10-180

    for monthly, dmin, dmax, name in groups:
        for key in monthly:
            arr = monthly[key]

            np.testing.assert_array_almost_equal(arr[..., 0], dmin[key],
                                                 decimal=3, err_msg=(f"{name}."
                                                 + f"{key} min mismatch "
                                                 + "(IG12=0)"))
            np.testing.assert_array_almost_equal(arr[..., 1], dmax[key],
                                                 decimal=3, err_msg=(f"{name}."
                                                 + f"{key} max mismatch "
                                                 + "(IG12=100)"))

    # Check that sun and mag are the same regardless of solar activity
    groups = [(sunm, sunmin, sunmax, 'sun'), (magm, magmin, magmax, 'mag')]

    # Check that IRI_density_1day with IG12=0/100 as input returns the same
    # parameter values as IRI_monthly_mean_par
    for monthly, dmin, dmax, name in groups:
        for key in monthly:

            np.testing.assert_array_almost_equal(monthly[key], dmin[key],
                                                 decimal=3, err_msg=(f"{name}."
                                                 + f"{key} min mismatch "
                                                 + "(IG12=0)"))
            np.testing.assert_array_almost_equal(monthly[key], dmax[key],
                                                 decimal=3, err_msg=(f"{name}."
                                                 + f"{key} max mismatch "
                                                 + "(IG12=100)"))


def test_EDP_builder_continuous():
    """Test EDP_builder_continuous output shape.

    Checks that the EDP_builder_continuous function outputs the correct shape.
    """
    N_T = 3
    N_G = 2
    N_V = 4
    shape = (N_T, N_G)

    F2 = {'Nm': np.full(shape, 1.2e12),
          'hm': np.full(shape, 350),
          'B0': np.full(shape, 120),
          'B1': np.full(shape, 2),
          'B_top': np.full(shape, 38)}
    F1 = {'Nm': np.full(shape, 2e11),
          'hm': np.full(shape, 200),
          'B_bot': np.full(shape, 17)}
    E = {'Nm': np.full(shape, 5e10),
         'hm': np.full(shape, 110),
         'B_top': np.full(shape, 5),
         'B_bot': np.full(shape, 7)}

    aalt = np.linspace(600, 700, N_V)

    EDP = sh.EDP_builder_continuous(F2, F1, E, aalt)

    assert EDP.shape == (N_T, N_V, N_G), ("EDP shape mismatch:"
                                          f" expected {(N_T, N_V, N_G)},"
                                          f" got {EDP.shape}")


def test_Ramakrishnan_Rawer_function():
    """Tests Ramakrishnan_Rawer_function output shape.

    Checks that the Ramakrishnan_Rawer_function outputs the correct shape.
    """
    N_T = 3
    N_G = 2
    N_V = 4
    shape = (N_V, N_T, N_G)

    NmF2 = np.full(shape, 1.2e12)
    hmF2 = np.full(shape, 350)
    B0 = np.full(shape, 120)
    B1 = np.full(shape, 2)
    h = np.full(shape, 200)

    den = sh.Ramakrishnan_Rawer_function(NmF2, hmF2, B0, B1, h)

    assert den.shape == shape, (f"Density shape mismatch: expected {shape},"
                                f" got {den.shape}")


def test_find_subsolar():
    """Tests find_subsolar.

    Checks that the find_subsolar function returns valid lat/lon outputs.
    """
    dtime = dt.datetime(2003, 4, 5)

    slon, slat = sh.find_subsolar(dtime, adjust_type='to360')

    assert (slon >= 0) and (slon <= 360), (f"slon value error: {slon}")
    assert (slat >= -90) and (slat <= 90), (f"slat value error: {slat}")


@pytest.mark.parametrize("transform_type",
                         ['GEO_2_QD',
                          'QD_2_GEO',
                          'MLT_2_QD',
                          'QD_2_MLT',
                          'GEO_2_MLT',
                          'MLT_2_GEO'])
@pytest.mark.parametrize("ll", [np.array([0]), np.array([[0], [0]])])
def test_Apex(transform_type, ll):
    """Test Apex output shape.

    Checks that the Apex function returns outputs of the correct shape.

    Parameters
    ------------
    transform_type : str
        Coordinate transform type between GEO, QD, and MLT.
    ll : array-like
        Argument used as latitude and longitude placeholders.
    """
    dtime = dt.datetime(2018, 1, 3)

    Lat, Lon = sh.Apex(ll, ll, dtime, transform_type=transform_type)

    assert Lat.shape == ll.shape, (f"Lat shape mismatch: expected {ll.shape},"
                                   f" got {Lat.shape}")


def test_Probability_F1_with_solzen():
    """Tests Probability_F1_with_solzen output shape.

    Checks that the Probability_F1_with_solzen function returns outputs of the
    correct shape.
    """
    solzen = np.ones((1, 3, 2))

    a_P = sh.Probability_F1_with_solzen(solzen)

    assert a_P.shape == solzen.shape, ("a_P shape mismatch: expected "
                                       f"{solzen.shape},"
                                       f" got {a_P.shape}")


@pytest.mark.parametrize("coord", ['GEO', 'QD', 'MLT'])
def test_gammaE_dynamic(coord):
    """Tests gammaE_dynamic output shape.

    Checks that the gammaE_dynamic function returns outputs of the correct
    shape.

    Parameters
    ------------
    coord : str
        Coordinate system used.
    """
    year = 2009
    month = 11
    day = 1
    aUT = [0, 1, 3]
    alon = np.array([0])
    alat = np.array([0])
    F107 = 100

    gamma_E, solzen_out, solzen_eff_out, slon, slat = sh.gammaE_dynamic(
        year, month, day, aUT, alon, alat, F107, coord=coord)

    N_T = len(aUT)
    N_G = len(alon)

    assert gamma_E.shape == (N_T, N_G), ("gamma_E shape mismatch: expected "
                                         f"{(N_T, N_G)},"
                                         f" got {gamma_E.shape}")
    assert solzen_out.shape == (N_T, N_G), ("solzen_out shape mismatch:"
                                            "expected "
                                            f"{(N_T, N_G)}, got "
                                            f"{solzen_out.shape}")
    assert solzen_eff_out.shape == (N_T, N_G), ("solzen_eff_out shape "
                                                "mismatch: expected "
                                                f"{(N_T, N_G)}, got "
                                                f"{solzen_eff_out.shape}")
    assert slon.shape == (N_T,), ("slon shape mismatch: expected "
                                  f"{(N_T,)},"
                                  f" got {slon.shape}")
    assert slat.shape == (N_T,), ("slat shape mismatch: expected "
                                  f"{(N_T,)},"
                                  f" got {slat.shape}")


@pytest.mark.parametrize("foF2_coeff", ['URSI', 'CCIR'])
@pytest.mark.parametrize("hmF2_model", ['SHU2015', 'AMTB2013', 'BSE1979'])
def test_load_coeff_matrices(foF2_coeff, hmF2_model):
    """Test load_coeff_matrices output shape.

    Checks that the load_coeff_matrices function returns outputs of the correct
    shape.

    Parameters
    ------------
    foF2_coeff : str
        foF2 coefficients to use between URSI and CCIR.
    hmF2_model : str
        hmF2 model to use between Shu-2015, AMTB-2013, and BSE-1979.
    """
    coeff_dir = PyIRI.coeff_dir
    month = 11

    C = sh.load_coeff_matrices(month, coeff_dir=coeff_dir,
                               foF2_coeff=foF2_coeff, hmF2_model=hmF2_model)

    if hmF2_model == 'BSE1979':
        N_P = 5
    else:
        N_P = 6
    N_IG = 2
    N_FS = 11
    N_SH = 900

    assert C.shape == (N_P, N_IG, N_FS, N_SH), ("C shape mismatch: expected "
                                                f"{(N_P, N_IG, N_FS, N_SH)},"
                                                f" got {C.shape}")


def test_run_iri_reg_grid():
    """Tests run_iri_reg_grid output shape.

    Checks that the run_iri_reg_grid function returns outputs of the correct
    shape.
    """
    hr_res = 4
    lat_res = 90
    lon_res = 90
    alt_res = 10
    alt_min = 90
    alt_max = 100
    coeff_dir = PyIRI.coeff_dir
    year = 2023
    month = 11
    day = 19
    F107 = 123
    foF2_coeff = 'URSI'
    hmF2_model = 'SHU2015'
    coord = 'MLT'

    (alon, alat, alon_2d, alat_2d, aalt,
        aUT, F2, F1, E, Es, sun, mag, EDP) = sh.run_iri_reg_grid(
            year, month, day, F107,
            hr_res=hr_res, lat_res=lat_res, lon_res=lon_res, alt_res=alt_res,
            alt_min=alt_min, alt_max=alt_max, coord=coord, coeff_dir=coeff_dir,
            foF2_coeff=foF2_coeff, hmF2_model=hmF2_model,
            old_output=False)

    N_lat = int(180 / lat_res + 1)
    N_lon = int(360 / lon_res + 1)
    N_hr = int(24 / hr_res)
    N_alt = int((alt_max - alt_min) / alt_res + 1)
    N_G = N_lat * N_lon

    assert alon.shape == (N_G,), ("alon shape mismatch:"
                                  f" expected {(N_G,)}, got {alon.shape}")
    assert alat.shape == (N_G,), ("alat shape mismatch:"
                                  f" expected {(N_G,)}, got {alat.shape}")
    assert aalt.shape == (N_alt,), ("aalt shape mismatch:"
                                    f" expected {(N_alt,)}, got {aalt.shape}")
    assert aUT.shape == (N_hr,), ("aUT shape mismatch:"
                                  f" expected {(N_hr,)}, got {aUT.shape}")
    assert alon_2d.shape == (N_lat, N_lon), ("alon_2d shape mismatch:"
                                             f" expected {(N_lat, N_lon)}, "
                                             f"got {alon_2d.shape}")
    assert F2['fo'].shape == (N_hr, N_G), ("foF2 shape mismatch:"
                                           f" expected {(N_hr, N_G)},"
                                           f" got {F2['fo'].shape}")
    assert EDP.shape == (N_hr, N_alt, N_G), ("EDP shape mismatch:"
                                             f" expected {(N_hr, N_alt, N_G)},"
                                             f" got {EDP.shape}")


def test_run_seas_iri_reg_grid():
    """Tests run_seas_iri_reg_grid output shape.

    Checks that the run_seas_iri_reg_grid function returns outputs of the
    correct shape.
    """
    hr_res = 4
    lat_res = 90
    lon_res = 90
    alt_res = 10
    alt_min = 90
    alt_max = 100
    coeff_dir = PyIRI.coeff_dir
    year = 2023
    month = 11
    foF2_coeff = 'URSI'
    hmF2_model = 'SHU2015'
    coord = 'MLT'

    (alon, alat, alon_2d, alat_2d, aalt,
        aUT, F2, F1, E, Es, sun, mag) = sh.run_seas_iri_reg_grid(
            year, month, solidx='IG12',
            hr_res=hr_res, lat_res=lat_res, lon_res=lon_res, alt_res=alt_res,
            alt_min=alt_min, alt_max=alt_max, coord=coord, coeff_dir=coeff_dir,
            foF2_coeff=foF2_coeff, hmF2_model=hmF2_model,
            old_output=False)

    N_lat = int(180 / lat_res + 1)
    N_lon = int(360 / lon_res + 1)
    N_hr = int(24 / hr_res)
    N_alt = int((alt_max - alt_min) / alt_res + 1)
    N_G = N_lat * N_lon

    assert alon.shape == (N_G,), ("alon shape mismatch:"
                                  f" expected {(N_G,)}, got {alon.shape}")
    assert alat.shape == (N_G,), ("alat shape mismatch:"
                                  f" expected {(N_G,)}, got {alat.shape}")
    assert aalt.shape == (N_alt,), ("aalt shape mismatch:"
                                    f" expected {(N_alt,)}, got {aalt.shape}")
    assert aUT.shape == (N_hr,), ("aUT shape mismatch:"
                                  f" expected {(N_hr,)}, got {aUT.shape}")
    assert alon_2d.shape == (N_lat, N_lon), ("alon_2d shape mismatch:"
                                             f" expected {(N_lat, N_lon)}, "
                                             f"got {alon_2d.shape}")
    assert F2['fo'].shape == (N_hr, N_G, 2), ("foF2 shape mismatch:"
                                              f" expected {(N_hr, N_G, 2)},"
                                              f" got {F2['fo'].shape}")


@pytest.mark.parametrize("coord", ['GEO', 'QD', 'MLT'])
def test_create_reg_grid_geo_or_mag(coord):
    """Tests create_reg_grid_geo_or_mag output shape.

    Checks that the create_reg_grid_geo_or_mag function returns outputs of the
    correct shape.

    Parameters
    ------------
    coord : str
        Coordinate system to use between GEO, QD, and MLT.
    """
    hr_res = 4
    lat_res = 90
    lon_res = 90
    alt_res = 10
    alt_min = 90
    alt_max = 100

    alon, alat, alon_2d, alat_2d, aalt, aUT = sh.create_reg_grid_geo_or_mag(
        hr_res, lat_res, lon_res, alt_res, alt_min, alt_max, coord=coord)

    N_lat = int(180 / lat_res + 1)
    N_lon = int(360 / lon_res + 1)
    N_hr = int(24 / hr_res)
    N_alt = int((alt_max - alt_min) / alt_res + 1)
    N_G = N_lat * N_lon

    assert alon.shape == (N_G,), ("alon shape mismatch:"
                                  f" expected {(N_G,)}, got {alon.shape}")
    assert alat.shape == (N_G,), ("alat shape mismatch:"
                                  f" expected {(N_G,)}, got {alat.shape}")
    assert aalt.shape == (N_alt,), ("aalt shape mismatch:"
                                    f" expected {(N_alt,)}, got {aalt.shape}")
    assert aUT.shape == (N_hr,), ("aUT shape mismatch:"
                                  f" expected {(N_hr,)}, got {aUT.shape}")
    assert alon_2d.shape == (N_lat, N_lon), ("alon_2d shape mismatch:"
                                             f" expected {(N_lat, N_lon)}, "
                                             f"got {alon_2d.shape}")


@pytest.mark.parametrize("foF2_coeff", ['URSI', 'CCIR'])
@pytest.mark.parametrize("hmF2_model", ['SHU2015', 'AMTB2013', 'BSE1979'])
def test_IRI_density_1day_runs_GEO(foF2_coeff, hmF2_model):
    """Tests IRI_density_1day output shape with GEO coordinate input.

    Checks that the IRI_density_1day function returns outputs of the correct
    shape given GEO coordinate input.

    Parameters
    ------------
    foF2_coeff : str
        foF2 coefficients to use between URSI and CCIR.
    hmF2_model : str
        hmF2 model to use between Shu-2015, AMTB-2013, and BSE-1979.
    """
    year = 2024
    mth = 6
    day = 21
    aUT = np.array([12.0, 13.0, 14.0])
    alon = [0]
    alat = 0
    coeff_dir = PyIRI.coeff_dir
    F107 = 125
    aalt = np.arange(90, 100, 10)
    coord = 'GEO'

    F2, F1, E, Es, sun, mag, EDP = sh.IRI_density_1day(
        year, mth, day, aUT, alon, alat, aalt, F107,
        coeff_dir=coeff_dir,
        foF2_coeff=foF2_coeff,
        hmF2_model=hmF2_model,
        coord=coord,
        old_output=False
    )

    expected_shape = (len(aUT), len(alon))
    actual_shape = F2['fo'].shape

    assert actual_shape == expected_shape, ("foF2 shape mismatch: expected "
                                            + f"{expected_shape}, got "
                                            + f"{actual_shape}")

    expected_shape = (len(aUT), len(aalt), len(alon))
    actual_shape = EDP.shape

    assert actual_shape == expected_shape, ("EDP shape mismatch: expected "
                                            + f"{expected_shape}, got "
                                            + f"{actual_shape}")


@pytest.mark.parametrize("foF2_coeff", ['URSI', 'CCIR'])
@pytest.mark.parametrize("hmF2_model", ['SHU2015', 'AMTB2013', 'BSE1979'])
def test_IRI_density_1day_runs_MLT(foF2_coeff, hmF2_model):
    """Tests IRI_density_1day output shape with MLT coordinate input.

    Checks that the IRI_density_1day function returns outputs of the correct
    shape given MLT coordinate input.

    Parameters
    ------------
    foF2_coeff : str
        foF2 coefficients to use between URSI and CCIR.
    hmF2_model : str
        hmF2 model to use between Shu-2015, AMTB-2013, and BSE-1979.
    """
    year = 2024
    mth = 6
    day = 21
    aUT = np.array([12.0, 13.0, 14.0])
    alon = [0]
    alat = 0
    coeff_dir = PyIRI.coeff_dir
    F107 = 125
    aalt = np.arange(90, 100, 10)
    coord = 'MLT'

    F2, F1, E, Es, sun, mag, EDP = sh.IRI_density_1day(
        year, mth, day, aUT, alon, alat, aalt, F107,
        coeff_dir=coeff_dir,
        foF2_coeff=foF2_coeff,
        hmF2_model=hmF2_model,
        coord=coord,
        old_output=False
    )

    expected_shape = (len(aUT), len(alon))
    actual_shape = F2['fo'].shape

    assert actual_shape == expected_shape, ("foF2 shape mismatch: expected "
                                            + f"{expected_shape}, got "
                                            + f"{actual_shape}")

    expected_shape = (len(aUT), len(aalt), len(alon))
    actual_shape = EDP.shape

    assert actual_shape == expected_shape, ("EDP shape mismatch: expected "
                                            + f"{expected_shape}, got "
                                            + f"{actual_shape}")


@pytest.mark.parametrize(
    "inp, c, exp_shape",
    [(np.array([0., 12.]), 2, (2, 3)),
     ([1, 2, 3, 4], 3, (4, 5)),
     ((5.5, 6.6), 4, (2, 7)),
     (12.0, 1, (1, 1)),],
    ids=["array", "list", "tuple", "scalar"]
)
def test_shape_dtype_FS(inp, c, exp_shape):
    """Tests real_FS_func output shape.

    Checks that the real_FS_func function returns outputs of the correct shape
    given different input types and shapes.

    Parameters
    ------------
    inp : array-like
        Input values.
    c : int
        Number of complex Fourier coefficients to use as a truncation level.
    exp_shape : tuple
        Expected output shape.
    """
    out = sh.real_FS_func(inp, N_FS_c=c)
    assert isinstance(out, np.ndarray)
    assert np.issubdtype(out.dtype, np.floating)
    assert out.shape == exp_shape


def test_empty_FS():
    """Tests real_FS_func output shape for empty input.

    Checks that the real_FS_func function returns empty outputs given empty
    inputs.
    """
    out = sh.real_FS_func(np.array([]), N_FS_c=3)
    assert out.shape == (0, 5)
    assert out.size == 0


@pytest.mark.parametrize("c", [0, -2])
def test_Nc_negative_FS(c):
    """Tests real_FS_func error for negative/null coefficient number input.

    Checks that the real_FS_func function returns a value error for negative or
    null coefficient input.
    """
    with pytest.raises(ValueError, match='must be >0'):
        sh.real_FS_func([0, 1], N_FS_c=c)


@pytest.mark.parametrize("c", [1.5, "3", None])
def test_Nc_type_FS(c):
    """Tests real_FS_func error for invalid dtype coefficient number input.

    Checks that the real_FS_func function returns a type error for invalid dtype
    input.
    """
    with pytest.raises(TypeError, match='must be an integer'):
        sh.real_FS_func([0, 1], N_FS_c=c)


def test_periodic_FS():
    """Tests periodicity of real_FS_func.

    Checks that the real_FS_func function is 24-hour periodic.
    """
    a = np.array([0.5, 5.2, 13.7])
    np.testing.assert_allclose(sh.real_FS_func(a, 4),
                               sh.real_FS_func(a + 24, 4),
                               atol=1e-15)


@pytest.mark.parametrize("UT", [1, 12.3, 23.9999])
def test_known_small_case_FS(UT):
    """Tests real_FS_func validity against known cases.

    Checks that the real_FS_func function returns accurate output values for
    small coefficient numbers as compared to analytical Fourier series.

    Parameters
    ------------
    UT : array-like
        Input time values in hours.
    """
    F = sh.real_FS_func(UT, N_FS_c=2)
    np.testing.assert_allclose(F[0, 0], 1.0, rtol=1e-6)
    np.testing.assert_allclose(F[0, 1], np.cos(2 * np.pi / 24 * UT), atol=1e-10)
    np.testing.assert_allclose(F[0, 2], np.sin(2 * np.pi / 24 * UT), atol=1e-10)


def test_shape_and_type_SH():
    """Tests real_SH_func output shape and type for 1D input.

    Checks that the real_SH_func function returns accurate output type and
    shape for a 1D input.
    """
    theta = np.linspace(0, np.pi, 5)
    phi = np.linspace(0, 2 * np.pi, 5)
    lmax = 3
    F = sh.real_SH_func(theta, phi, lmax=lmax)
    assert isinstance(F, np.ndarray)
    assert F.dtype == float
    # After squeeze, shape is (N_SH, N_G)
    assert F.shape == ((lmax + 1) ** 2, phi.size)


def test_shape_for_2d_input_SH():
    """Tests real_SH_func output shape for 2D input.

    Checks that the real_SH_func function returns accurate output shape for a 2D
    input.
    """
    lon = np.linspace(0, 2 * np.pi, 10)
    lat = np.linspace(0, np.pi, 8)
    theta, phi = np.meshgrid(lat, lon, indexing="ij")
    F = sh.real_SH_func(theta, phi, lmax=3)
    assert F.shape == ((3 + 1) ** 2, theta.shape[0], theta.shape[1])


def test_known_small_case_SH():
    """Tests real_SH_func validity against known cases.

    Checks that the real_SH_func function returns accurate output values for
    small coefficient numbers as compared to analytical spherical harmonics.
    """
    theta = np.array([np.pi / 2])
    phi = np.array([0.0])
    F = sh.real_SH_func(theta, phi, lmax=1)
    expected = 1.0  # uses 4π normalization
    np.testing.assert_allclose(F[0], expected, rtol=1e-6)
    np.testing.assert_allclose(F[2], 0.0, atol=1e-10)


@pytest.mark.parametrize("transform_type", ['GEO_2_QD', 'QD_2_GEO'])
def test_geo_to_qd(transform_type):
    """Tests Apex_geo_qd output shape and finiteness.

    Checks that the Apex_geo_qd outputs the correct shape and is finite.

    Parameters
    ------------
    trnasform_type : str
        Coord transform type between GEO and QD.
    """
    inLat = np.array([[10, 20], [30, 40]])
    inLon = np.array([[100, 120], [140, 160]])
    dtime = dt.datetime(2005, 1, 1)
    outLat, outLon = sh.Apex_geo_qd(inLat, inLon, dtime,
                                    transform_type=transform_type)
    assert inLat.shape == outLat.shape
    assert inLon.shape == outLon.shape
    assert np.all(np.isfinite(outLat))
    assert np.all(np.isfinite(outLon))


def test_invalid_transform_raises():
    """Tests Apex_geo_qd value error for invalid trnasform type.

    Checks that an invalid transform type input raises a value error.
    """
    Lat = np.array([0])
    Lon = np.array([0])
    dtime = dt.datetime(2005, 1, 1)
    with pytest.raises(ValueError, match='Transform type must be'):
        sh.Apex_geo_qd(Lat, Lon, dtime, transform_type="INVALID")


def test_F1_hmF1_clamped_to_180():
    """Tests derive_dependent_F1_parameters function for hmF1 clamping.

    The derive_dependent_F1_parameters function clamps any output hmF1 <180 km
    to 180 km.
    """
    P = np.array([0.5])
    NmF2 = np.array([1e12])
    hmF2 = np.array([200.0])  # 200 - 30 = 170
    B0 = np.array([30.0])
    B1 = np.array([1.0])
    hmE = np.array([100.0])

    _, _, hmF1, _ = sh.derive_dependent_F1_parameters(P, NmF2, hmF2, B0, B1,
                                                      hmE)
    assert hmF1[0] == 180.0


def test_F1_NmF1_minimum_one():
    """Tests derive_dependent_F1_parameters function for NmF1 clamping.

    The derive_dependent_F1_parameters function clamps any output NmF1 <=0 m-3
    to 1 m-3 to avoid log10(0).
    """
    P = np.array([0.0])
    NmF2 = np.array([0])
    hmF2 = np.array([300.0])
    B0 = np.array([50.0])
    B1 = np.array([1.0])
    hmE = np.array([100.0])

    NmF1, _, _, _ = sh.derive_dependent_F1_parameters(P, NmF2, hmF2, B0, B1,
                                                      hmE)
    assert np.all(NmF1 >= 1.0)


def test_BSE_ratio_clamped_below_1_7():
    """Tests BSE_1979_model function for foF2/foE clamping.

    The BSE_1979_model function clamps any output foF2/foE ratio < 1.7 to 1.7.
    Clamping to 1.7 is to avoid unreasonably low hmF2 values. Refer to Bilitza
    et al. (2022), The International Reference Ionosphere model: A review and
    description of an ionospheric benchmark, Reviews of Geophysics, 60.
    """
    M3000 = np.array([3.0])
    foF2 = np.array([1.5])
    foE = np.array([1.0])  # ratio = 1.5 < 1.7
    modip = np.array([0.0])
    F107 = 100

    hmF2 = sh.BSE_1979_model(M3000, foF2, foE, modip, F107)
    # Verify finite; exact value depends on clamped ratio=1.7 path
    assert np.isfinite(hmF2[0])


def test_thickness_F2_positive_output():
    """Tests thickness_F2 function for valid output values and shape.

    The thickness_F2 function must output positive values for B_F2_bot and
    B_F2_top.
    """
    foF2 = np.array([5.0, 10.0])
    M3000 = np.array([3.0, 4.0])
    hmF2 = np.array([300.0, 400.0])
    NmF2 = np.array([1e12, 2e12])
    F107 = 150

    B_top, B_bot = sh.thickness_F2(NmF2, foF2, M3000, hmF2, F107)
    assert B_top.shape == foF2.shape
    assert np.all(B_top > 0)
    assert np.all(B_bot > 0)


def test_invalid_coordinate_raises_error():
    """Tests IRI_sh_params function for coord system input value error.

    The IRI_sh_params function is the core behind IRI_density_1day and
    IRI_monthly_mean_par. Any invalid coordinate system input to this function
    will result in the same value error raised as when input in higher level
    functions.
    """
    with pytest.raises(ValueError, match="Coordinate system must be"):
        sh.IRI_sh_params(2024, 6, 12.0, 0.0, 45.0, coord='INVALID')


@pytest.mark.parametrize("hmF2_model", ['SHU2015', 'AMTB2013', 'BSE1979'])
def test_array_input_preserves_shape(hmF2_model):
    """Tests IRI_sh_params function for output shape.

    The IRI_sh_params function is the core behind IRI_density_1day and
    IRI_monthly_mean_par. It must output numerical maps of shape
    (n_params, N_T, N_G, 2), from which IRI_density_1day interpolates in solar
    activity. If option BSE1979 is chosen, a nan map is returned for hmF2.

    Parameters
    ------------
    hmF2_model : str
        hmF2 model to use.
    """
    N_T, N_G = 4, 10
    aUT = np.linspace(0, 24, N_T)
    alon = np.linspace(-180, 180, N_G)
    alat = np.linspace(-90, 90, N_G)

    result = sh.IRI_sh_params(2024, 6, aUT, alon, alat, hmF2_model=hmF2_model,
                              coord='GEO')

    # Shape: (n_params, N_T, N_G, 2)
    assert result.shape == (6, N_T, N_G, 2)


@pytest.mark.parametrize("coord", ['MLT', 'QD', 'GEO'])
def test_IRI_sh_params_coord_shape(coord):
    """Tests IRI_sh_params function for output shape depending on coord system.

    The IRI_sh_params function is the core behind IRI_density_1day and
    IRI_monthly_mean_par. It must output numerical maps of shape
    (n_params, N_T, N_G, 2).

    Parameters
    ------------
    coord : str
        Coordinate system to use.
    """
    result = sh.IRI_sh_params(2024, 6, [0.0, 12.0, 13.0], [0.0], [60.0],
                              coord=coord)
    assert result.shape == (6, 3, 1, 2)


def test_IRI_sh_params_empty_hmF2():
    """Tests IRI_sh_params function for output shape.

    The IRI_sh_params function is the core behind IRI_density_1day and
    IRI_monthly_mean_par. It must output numerical maps of shape
    (n_params, N_T, N_G, 2), from which IRI_density_1day interpolates in solar
    activity. If option BSE1979 is chosen, a nan map is returned for hmF2.
    """
    result = sh.IRI_sh_params(2024, 6, [12.0], [120.0], [-30.0],
                              hmF2_model='BSE1979')
    assert np.all(np.isnan(result[1]))


@pytest.mark.parametrize(
    "coord, exp_shape",
    [('MLT', (2, 3)),
     ('QD', (3,)),
     ('GEO', (3,)),]
)
def test_IRI_monthly_mean_par_dimensions(coord, exp_shape):
    """Tests IRI_monthly_mean_par function for output shape depending on coord.

    The IRI_monthly_mean_par calls on the IRI_density_1day function twice to
    generate ionospheric parameters for two solar activity levels requested,
    for the 15th of the month requested. For all coordinate systems, it should
    output shape (N_T, N_G, 2) for iono parameters and (N_G,) for sun
    parameters. For mag parameters, it outputs (N_T, N_G) if coordinate system
    is MLT, (N_G,) otherwise.

    Parameters
    ------------
    coord : str
        Coordinate system to use.
    exp_shape : tuple
        Expected output shape.
    """
    N_T, N_G = 2, 3

    F2, F1, E, Es, sun, mag = sh.IRI_monthly_mean_par(
        year=2024,
        month=6,
        aUT=np.linspace(0, 12, N_T),
        alon=np.linspace(-180, 180, N_G),
        alat=np.linspace(-60, 60, N_G),
        solidx='R12',
        solmin=10,
        solmax=100,
        coord=coord,
        old_output=False
    )

    # Ionospheric parameters stacked: (N_T, N_G, 2)
    assert F2['fo'].shape == (N_T, N_G, 2)
    assert F2['hm'].shape == (N_T, N_G, 2)
    assert F1['fo'].shape == (N_T, N_G, 2)
    assert E['fo'].shape == (N_T, N_G, 2)
    assert Es['fo'].shape == (N_T, N_G, 2)

    # Sun position: (N_T,)
    assert sun['lon'].shape == (N_T,)
    assert sun['lat'].shape == (N_T,)

    # Magnetic field: (N_G,) for GEO and QD, (N_T, N_G) for MLT
    assert mag['modip'].shape == exp_shape
    assert mag['inc'].shape == exp_shape
