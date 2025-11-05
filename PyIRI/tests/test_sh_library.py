"""Unit tests for PyIRI's sh_library."""

import datetime as dt

import numpy as np
import pytest

import PyIRI
from PyIRI.main_library import den2freq
from PyIRI.main_library import IG12_2_F107
import PyIRI.sh_library as sh


def test_fo_1day_interpolation():
    """Test linear interpolation of foF2."""
    F2, F1, E, *_ = sh.IRI_density_1day(2024,
                                        3,
                                        8,
                                        np.array([0]),
                                        np.array([20]),
                                        np.array([40]),
                                        np.array([0]),
                                        IG12_2_F107(40))

    Es = sh.sporadic_E_1day(2024,
                            3,
                            8,
                            np.array([0]),
                            np.array([20]),
                            np.array([40]),
                            IG12_2_F107(40))

    fo_1day = []
    true_fo = []
    for layer in [F2, F1, E, Es]:
        fo_1day.append(layer['fo'][0, 0])
        Nm_1day = layer['Nm'][0, 0]
        true_fo.append(den2freq(Nm_1day))

    assert fo_1day == true_fo, ("fo 1day interpolation error: fo = "
                                f"{fo_1day}, should be {true_fo}")


# EDP_builder_continuous
def test_EDP_builder_continuous():
    """Exercise EDP_builder_continuous."""
    N_T = 3
    N_G = 2
    N_V = 4

    F2 = {'Nm': np.ones((N_T, N_G)) * 1.2e12,
          'hm': np.ones((N_T, N_G)) * 350,
          'B0': np.ones((N_T, N_G)) * 120,
          'B1': np.ones((N_T, N_G)) * 2,
          'B_top': np.ones((N_T, N_G)) * 38}
    F1 = {'Nm': np.ones((N_T, N_G)) * 2e11,
          'hm': np.ones((N_T, N_G)) * 200,
          'B_bot': np.ones((N_T, N_G)) * 17}
    E = {'Nm': np.ones((N_T, N_G)) * 5e10,
         'hm': np.ones((N_T, N_G)) * 110,
         'B_top': np.ones((N_T, N_G)) * 5,
         'B_bot': np.ones((N_T, N_G)) * 7}

    aalt = np.linspace(600, 700, N_V)

    EDP = sh.EDP_builder_continuous(F2, F1, E, aalt)

    assert EDP.shape == (N_T, N_V, N_G), ("EDP shape mismatch:"
                                          f" expected {(N_T, N_V, N_G)},"
                                          f" got {EDP.shape}")


# Ramakrishnan_Rawer_function
def test_Ramakrishnan_Rawer_function():
    """Exercise Ramakrishnan_Rawer_function."""
    NmF2 = np.ones((2, 3, 4)) * 1.2e12
    hmF2 = np.ones((2, 3, 4)) * 350
    B0 = np.ones((2, 3, 4)) * 120
    B1 = np.ones((2, 3, 4)) * 2
    h = np.ones((2, 3, 4)) * 100

    den = sh.Ramakrishnan_Rawer_function(NmF2, hmF2, B0, B1, h)

    assert den.shape == (2, 3, 4), ("Density shape mismatch:"
                                    f" expected (2, 3, 4),"
                                    f" got {den.shape}")


# find_subsolar
def test_find_subsolar():
    """Exercise find_subsolar."""
    dtime = dt.datetime(2003, 4, 5)

    slon, slat = sh.find_subsolar(dtime, adjust_type='to360')

    assert (slon >= 0) and (slon <= 360), (f"slon value error: {slon}")
    assert (slat >= -90) and (slat <= 90), (f"slat value error: {slat}")


# Apex
@pytest.mark.parametrize("transform_type",
                         ['GEO_2_QD',
                          'QD_2_GEO',
                          'MLT_2_QD',
                          'QD_2_MLT',
                          'GEO_2_MLT',
                          'MLT_2_GEO'])
@pytest.mark.parametrize("alon", [np.array([0]), np.array([[0], [0]])])
def test_Apex(transform_type, alon):
    """Exercise Apex."""
    dtime = dt.datetime(2018, 1, 3)

    Lat, Lon = sh.Apex(alon, alon, dtime, transform_type=transform_type)

    assert Lat.shape == alon.shape, ("Lat shape mismatch: expected "
                                     f"{alon.shape},"
                                     f" got {Lat.shape}")


# Probability_F1_with_solzen
def test_Probability_F1_with_solzen():
    """Exercise Probability_F1_with_solzen."""
    solzen = np.ones((1, 3, 2))

    a_P = sh.Probability_F1_with_solzen(solzen)

    assert a_P.shape == solzen.shape, ("a_P shape mismatch: expected "
                                       f"{solzen.shape},"
                                       f" got {a_P.shape}")


# gammaE_dynamic
@pytest.mark.parametrize("coord", ['GEO', 'QD', 'MLT'])
def test_gammaE_dynamic(coord):
    """Exercise gammaE_dynamic."""
    year = 2009
    month = 11
    aUT = [0, 1, 3]
    alon = np.array([0])
    alat = np.array([0])
    aIG = [0, 100]

    gamma_E, solzen_out, solzen_eff_out, slon, slat = sh.gammaE_dynamic(
        year, month, aUT, alon, alat, aIG, coord=coord)

    N_T = len(aUT)
    N_G = len(alon)

    assert gamma_E.shape == (N_T, N_G, 2), ("gamma_E shape mismatch: expected "
                                            f"{(N_T, N_G, 2)},"
                                            f" got {gamma_E.shape}")
    assert solzen_out.shape == (N_T, N_G, 2), ("solzen_out shape mismatch:"
                                               "expected "
                                               f"{(N_T, N_G, 2)}, got "
                                               f"{solzen_out.shape}")
    assert solzen_eff_out.shape == (N_T, N_G, 2), ("solzen_eff_out shape "
                                                   "mismatch: expected "
                                                   f"{(N_T, N_G, 2)}, got "
                                                   f"{solzen_eff_out.shape}")
    assert slon.shape == (N_T,), ("slon shape mismatch: expected "
                                  f"{(N_T,)},"
                                  f" got {slon.shape}")
    assert slat.shape == (N_T,), ("slat shape mismatch: expected "
                                  f"{(N_T,)},"
                                  f" got {slat.shape}")


# load_Es_coeff_matrix
def test_load_Es_coeff_matrix():
    """Exercise load_Es_coeff_matrix."""
    coeff_dir = PyIRI.coeff_dir
    month = 11

    C = sh.load_Es_coeff_matrix(month, coeff_dir=coeff_dir)

    N_IG = 2
    N_FS = 11
    N_SH = 8100

    assert C.shape == (N_IG, N_FS, N_SH), ("C shape mismatch: expected "
                                           f"{(N_IG, N_FS, N_SH)},"
                                           f" got {C.shape}")


# load_coeff_matrices
@pytest.mark.parametrize("foF2_coeff", ['URSI', 'CCIR'])
@pytest.mark.parametrize("hmF2_model", ['SHU2015', 'AMTB2013', 'BSE1979'])
def test_load_coeff_matrices(foF2_coeff, hmF2_model):
    """Exercise load_coeff_matrices."""
    coeff_dir = PyIRI.coeff_dir
    month = 11

    C = sh.load_coeff_matrices(month, coeff_dir=coeff_dir,
                               foF2_coeff=foF2_coeff, hmF2_model=hmF2_model)

    if hmF2_model == 'BSE1979':
        N_P = 4
    else:
        N_P = 5
    N_IG = 2
    N_FS = 11
    N_SH = 900

    assert C.shape == (N_P, N_IG, N_FS, N_SH), ("C shape mismatch: expected "
                                                f"{(N_P, N_IG, N_FS, N_SH)},"
                                                f" got {C.shape}")


# run_seas_iri_reg_grid
def test_run_seas_iri_reg_grid():
    """Exercise run_seas_iri_reg_grid."""
    hr_res = 4
    lat_res = 90
    lon_res = 90
    alt_res = 10
    alt_min = 80
    alt_max = 100
    coeff_dir = PyIRI.coeff_dir
    year = 2023
    month = 11
    foF2_coeff = 'URSI'
    hmF2_model = 'SHU2015'
    coord = 'MLT'

    (alon, alat, alon_2d, alat_2d, aalt,
        aUT, F2, F1, E, sun, mag) = sh.run_seas_iri_reg_grid(
            year, month, hr_res=hr_res, lat_res=lat_res, lon_res=lon_res,
            alt_res=alt_res, alt_min=alt_min, alt_max=alt_max,
            coord=coord, coeff_dir=coeff_dir,
            foF2_coeff=foF2_coeff, hmF2_model=hmF2_model)

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


# run_iri_reg_grid
def test_run_iri_reg_grid():
    """Exercise run_iri_reg_grid."""
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
        aUT, F2, F1, E, sun, mag, EDP) = sh.run_iri_reg_grid(
            year, month, day, F107,
            hr_res=hr_res, lat_res=lat_res, lon_res=lon_res, alt_res=alt_res,
            alt_min=alt_min, alt_max=alt_max, coord=coord, coeff_dir=coeff_dir,
            foF2_coeff=foF2_coeff, hmF2_model=hmF2_model)

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


# create_reg_grid_geo_or_mag
@pytest.mark.parametrize("coord", ['GEO', 'QD', 'MLT'])
def test_create_reg_grid_geo_or_mag(coord):
    """Exercise create_reg_grid_geo_or_mag."""
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


# sporadic_E_1day
def test_Es_1day_runs():
    """Exercise sporadic_E_1day."""
    year = 2024
    mth = 6
    day = 21
    aUT = np.array([12.0, 13.0, 14.0])
    alon = [0]
    alat = 0
    coeff_dir = PyIRI.coeff_dir
    F107 = 120
    coord = 'MLT'

    Es = sh.sporadic_E_1day(
        year, mth, day, aUT, alon, alat, F107,
        coeff_dir=coeff_dir, coord=coord)

    expected_shape = (len(aUT), len(alon))
    actual_shape = Es['fo'].shape

    assert actual_shape == expected_shape, (
        f"foEs shape mismatch: expected {expected_shape}, got {actual_shape}"
    )


# sporadic_E_monthly_mean
@pytest.mark.parametrize("coord", ['GEO', 'QD', 'MLT'])
def test_Es_monthly_mean_runs(coord):
    """Exercise sporadic_E_monthly_mean."""
    year = 2024
    mth = 6
    aUT = np.array([12.0, 13.0, 14.0])
    alon = [0]
    alat = 0
    coeff_dir = PyIRI.coeff_dir

    Es = sh.sporadic_E_monthly_mean(
        year, mth, aUT, alon, alat,
        coeff_dir=coeff_dir, coord=coord)

    expected_shape = (len(aUT), len(alon), 2)
    actual_shape = Es['fo'].shape

    assert actual_shape == expected_shape, (
        f"foEs shape mismatch: expected {expected_shape}, got {actual_shape}"
    )


# IRI_monthly_mean_par
@pytest.mark.parametrize("foF2_coeff", ['URSI', 'CCIR'])
@pytest.mark.parametrize("hmF2_model", ['SHU2015', 'AMTB2013', 'BSE1979'])
@pytest.mark.parametrize("coord", ['GEO', 'QD', 'MLT'])
def test_IRI_monthly_mean_par_runs(foF2_coeff, hmF2_model, coord):
    """Exercise IRI_monthly_mean_par with supported foF2 and hmF2 options."""
    year = 2024
    mth = 6
    aUT = np.array([12.0, 13.0, 14.0])
    alon = [0]
    alat = 0
    coeff_dir = PyIRI.coeff_dir

    F2, F1, E, sun, mag = sh.IRI_monthly_mean_par(
        year, mth, aUT, alon, alat,
        coeff_dir=coeff_dir,
        foF2_coeff=foF2_coeff,
        hmF2_model=hmF2_model,
        coord=coord
    )

    expected_shape = (len(aUT), len(alon), 2)
    actual_shape = F2['fo'].shape

    assert actual_shape == expected_shape, (
        f"foF2 shape mismatch: expected {expected_shape}, got {actual_shape}"
    )


# IRI_density_1day
def test_IRI_density_1day_runs():
    """Exercise IRI_density_1day with supported foF2 and hmF2 options."""
    year = 2024
    mth = 6
    day = 21
    aUT = np.array([12.0, 13.0, 14.0])
    alon = [0]
    alat = 0
    coeff_dir = PyIRI.coeff_dir
    F107 = 125
    aalt = np.arange(90, 100, 10)
    foF2_coeff = 'URSI'
    hmF2_model = 'SHU2015'
    coord = 'MLT'

    F2, F1, E, sun, mag, EDP = sh.IRI_density_1day(
        year, mth, day, aUT, alon, alat, aalt, F107,
        coeff_dir=coeff_dir,
        foF2_coeff=foF2_coeff,
        hmF2_model=hmF2_model,
        coord=coord
    )

    expected_shape = (len(aUT), len(alon))
    actual_shape = F2['fo'].shape

    assert actual_shape == expected_shape, (
        f"foF2 shape mismatch: expected {expected_shape}, got {actual_shape}"
    )

    expected_shape = (len(aUT), len(aalt), len(alon))
    actual_shape = EDP.shape

    assert actual_shape == expected_shape, (
        f"EDP shape mismatch: expected {expected_shape}, got {actual_shape}"
    )


# real_FS_func
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
    out = sh.real_FS_func(inp, N_FS_c=c)
    assert isinstance(out, np.ndarray)
    assert np.issubdtype(out.dtype, np.floating)
    assert out.shape == exp_shape


def test_empty_FS():
    """Check the shape and dtype of output of function FS given empty input."""
    out = sh.real_FS_func(np.array([]), N_FS_c=3)
    assert out.shape == (0, 5)
    assert out.size == 0


@pytest.mark.parametrize("c", [0, -2])
def test_Nc_negative_FS(c):
    """Check that an error is raised for negative/null coeff number in FS."""
    with pytest.raises(ValueError):
        sh.real_FS_func([0, 1], N_FS_c=c)


@pytest.mark.parametrize("c", [1.5, "3", None])
def test_Nc_type_FS(c):
    """Check that an error is raised for inavlid input types in FS."""
    with pytest.raises(TypeError):
        sh.real_FS_func([0, 1], N_FS_c=c)


def test_periodic_FS():
    """Check periodicity of function FS."""
    a = np.array([0.5, 5.2, 13.7])
    np.testing.assert_allclose(sh.real_FS_func(a, 4),
                               sh.real_FS_func(a + 24, 4),
                               atol=1e-15)


@pytest.mark.parametrize("UT", [1, 12.3, 23.9999])
def test_known_small_case_FS(UT):
    """For small N_FS_c, compare against analytical Fourier series."""
    F = sh.real_FS_func(UT, N_FS_c=2)
    np.testing.assert_allclose(F[0, 0], 1.0, rtol=1e-6)
    np.testing.assert_allclose(F[0, 1], np.cos(2 * np.pi / 24 * UT), atol=1e-10)
    np.testing.assert_allclose(F[0, 2], np.sin(2 * np.pi / 24 * UT), atol=1e-10)


# real_SH_func
def test_shape_and_type_SH():
    """Check output shape and dtype for simple 1D input for SH function."""
    theta = np.linspace(0, np.pi, 5)
    phi = np.linspace(0, 2 * np.pi, 5)
    lmax = 3
    F = sh.real_SH_func(theta, phi, lmax=lmax)
    assert isinstance(F, np.ndarray)
    assert F.dtype == float
    # After squeeze, shape is (N_SH, N_G)
    assert F.shape == ((lmax + 1) ** 2, phi.size)


def test_shape_for_2d_input_SH():
    """Check shape when theta and phi are 2D arrays for SH function."""
    lon = np.linspace(0, 2 * np.pi, 10)
    lat = np.linspace(0, np.pi, 8)
    theta, phi = np.meshgrid(lat, lon, indexing="ij")
    F = sh.real_SH_func(theta, phi, lmax=3)
    assert F.shape == ((3 + 1) ** 2, theta.shape[0], theta.shape[1])


def test_known_small_case_SH():
    """For lmax=1, compare against analytical spherical harmonics."""
    theta = np.array([np.pi / 2])
    phi = np.array([0.0])
    F = sh.real_SH_func(theta, phi, lmax=1)
    expected = 1.0  # uses 4π normalization
    np.testing.assert_allclose(F[0], expected, rtol=1e-6)
    np.testing.assert_allclose(F[2], 0.0, atol=1e-10)


def test_large_lmax_runs_fast_SH():
    """Sanity check performance for moderate lmax."""
    theta = np.linspace(0, np.pi, 10)
    phi = np.linspace(0, 2 * np.pi, 10)
    F = sh.real_SH_func(theta, phi, lmax=10)
    assert F.shape == ((10 + 1) ** 2, theta.size)


# Apex_geo_qd
def test_geo_to_qd():
    """Test GEO_2_QD transformation returns arrays of correct shape."""
    Lat = np.array([[10, 20], [30, 40]])
    Lon = np.array([[100, 120], [140, 160]])
    dtime = dt.datetime(2005, 1, 1)
    QDLat, QDLon = sh.Apex_geo_qd(Lat, Lon, dtime, transform_type="GEO_2_QD")
    assert QDLat.shape == Lat.shape
    assert QDLon.shape == Lon.shape
    assert np.all(np.isfinite(QDLat))
    assert np.all(np.isfinite(QDLon))


def test_qd_to_geo():
    """Test QD_2_GEO transformation returns arrays of correct shape."""
    Lat = np.array([10, 20, 30])
    Lon = np.array([100, 150, 200])
    dtime = dt.datetime(2010, 6, 1)
    GeoLat, GeoLon = sh.Apex_geo_qd(Lat, Lon, dtime, transform_type="QD_2_GEO")
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
        sh.Apex_geo_qd(Lat, Lon, dtime, transform_type="INVALID")
