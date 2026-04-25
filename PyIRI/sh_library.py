#!/usr/bin/env python
# --------------------------------------------------------
# Distribution statement A. Approved for public release.
# Distribution is unlimited.
# This work was supported by the Office of Naval Research.
# --------------------------------------------------------
"""This library contains Spherical Harmonics and Apex features for PyIRI.

The Apex coordinates were estimated for each year from 1900 to 2030 using
apexpy. These results were converted to the spherical harmonic coefficients up
to lmax=20 and saved in the .nc file PyIRI.coeff_dir / 'Apex' / 'Apex.nc'.

Some core ionospheric parameters of F2, F1, E, and Es are computed using
spherical harmonics based on the IRI (and PyIRI) model. The other parameters are
derived from the core parameters.

References
----------
Emmert et al. (2010), A computationally compact representation of
Magnetic-Apex and Quasi-Dipole coordinates with smooth base vectors,
J. Geophys. Res., 115(A8), A08322, doi:10.1029/2010JA015326.

Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
International Reference Ionosphere Modeling Implemented in Python,
Space Weather, ESS Open Archive, September 28, 2023,
doi:10.22541/essoar.169592556.61105365/v1.

Servan-Schreiber et al. (2026), A Major Updateto the PyIRI Model,
Space Weather.

"""

import datetime as dt
import netCDF4 as nc
import numpy as np
import opt_einsum as oe
import os
import pandas as pd

import PyIRI
import PyIRI.edp_update as edpup
import PyIRI.igrf_library as igrf
from PyIRI import logger
import PyIRI.main_library as ml

import scipy.special as ss


def IRI_sh_params(year, month, aUT, alon, alat,
                  coeff_dir=None, foF2_coeff='URSI',
                  hmF2_model='SHU2015', coord='GEO'):
    """Output monthly mean ionospheric parameters using spherical harmonics.

    Parameters
    ----------
    year : int
        Year.
    month : int
        Month of the year.
    aUT : float, int, or array-like
        UT time [hour]. Scalar inputs will be converted to a Numpy array.
        Shape (N_T,)
    alon : float, list, or array-like
        Flattened array of longitude (geographic or quasi-dipole) [deg] or
        magnetic local time [hour]. Scalar inputs will be converted to a Numpy
        array.
        Shape (N_G,)
    alat : float, list, or array-like
        Flattened array of latitude (geographic or quasi-dipole) [deg]. Scalar
        inputs will be converted to a Numpy array.
        Shape (N_G,)
    coeff_dir: str
        Directory where the coefficient files are stored. If None, uses the
        default coefficient files stored in PyIRI.coeff_dir. (default=None)
    foF2_coeff : str
        Coefficients to use for foF2. Options are 'URSI' and 'CCIR'.
        (default='URSI')
    hmF2_model : str
        Model to use for hmF2. Options are 'SHU2015', 'AMTB2013', and
        'BSE1979'. (default='SHU2015')
    coord : str
        Coordinate system. Options are 'GEO' for geographic, 'QD' for quasi-
        dipole, and 'MLT' for magnetic local time. (default='GEO')

    Returns
    -------
    num_maps : np.ndarray
        Array containing numerical maps for foF2, hmF2, B0, B1, M3000, and foEs
        at each UT timestamp, grid location, and for min and max solar activity.
        Shape (N_T, N_G, 2)

    Notes
    -----
    This function returns monthly mean ionospheric parameters for min and max
    levels of solar activity for the parameters computed using spherical
    harmonics:
    - foF2 (either CCIR or URSI) using IG12=0-100
    - hmF2 (either SHU2015 using IG12=0-100 or AMTB2013 using R12=0-100)
    - M3000, B0, B1, and foEs using R12=0-100
    If hmF2_model='BSE1979', an array of NaN is returned, and hmF2 is
    calculated form M3000 in IRI_monthly_mean_par or IRI_density_1day

    References
    ----------
    Servan-Schreiber et al. (2026), A Major Update to the PyIRI Model, Space
    Weather.

    """
    # Set coefficient file path if none given
    if coeff_dir is None:
        coeff_dir = PyIRI.coeff_dir

    # Convert inputs to Numpy arrays
    aUT = ml.to_numpy_array(aUT)
    alon = ml.to_numpy_array(alon)
    alat = ml.to_numpy_array(alat)
    apdtime = (pd.to_datetime(dt.datetime(year, month, 15))
               + pd.to_timedelta(aUT, 'hours'))

    # Coordinate system conversion to geographic, quasi-dipole, and magnetic
    # local time
    if coord == 'GEO':
        aglat, aglon = alat, alon
        aqdlat, amlt = np.zeros((2, aUT.size, alat.size))
        for iUT in range(len(aUT)):
            pdtime = apdtime[iUT]
            aqdlat[iUT, :], amlt[iUT, :] = Apex(aglat, aglon, pdtime,
                                                'GEO_2_MLT')

    elif coord == 'MLT':
        aqdlat, amlt = alat, alon
        aglat, aglon = np.zeros((2, aUT.size, alat.size))
        for iUT in range(len(aUT)):
            pdtime = apdtime[iUT]
            aglat[iUT, :], aglon[iUT, :] = Apex(aqdlat, amlt, pdtime,
                                                'MLT_2_GEO')

    elif coord == 'QD':
        aqdlat, aqdlon = alat, alon
        aglat, aglon = Apex(aqdlat, aqdlon, apdtime[0], 'QD_2_GEO')
        amlt = np.zeros((aUT.size, alat.size))
        for iUT in range(len(aUT)):
            pdtime = apdtime[iUT]
            _, amlt[iUT, :] = Apex(aqdlat, aqdlon, pdtime, 'QD_2_MLT')
        aqdlat = np.broadcast_to(aqdlat, (aUT.size, aqdlat.size))

    else:
        raise ValueError("Coordinate system must be 'GEO', 'QD', or 'MLT'.")

    # Extract coefficient matrices and resonstruct ionospheric parameters
    C = load_coeff_matrices(month, coeff_dir, foF2_coeff, hmF2_model)

    n_FS_r = C.shape[2]
    n_FS_c = n_FS_r // 2 + 1
    F_FS = real_FS_func(aUT, n_FS_c)

    n_SH = C.shape[3]
    lmax = int(np.sqrt(n_SH)) - 1
    atheta = np.deg2rad(-(aqdlat - 90))
    aphi = np.deg2rad(amlt * 15)

    F_SH = real_SH_func(atheta, aphi, lmax=lmax)
    if coord == 'MLT':
        Params = oe.contract('ij,opjk,kl->oilp', F_FS, C, F_SH)
    else:
        Params = oe.contract('ij,opjk,kil->oilp', F_FS, C, F_SH)

    foF2 = Params[0, :, :, :]
    B0 = Params[1, :, :, :]
    B1 = Params[2, :, :, :]
    M3000 = Params[3, :, :, :]
    foEs = Params[4, :, :, :]
    if hmF2_model != 'BSE1979':
        hmF2 = Params[5, :, :, :]
    else:
        hmF2 = np.full(shape=foEs.shape, fill_value=np.nan)

    num_maps = np.array([foF2, hmF2, B0, B1, M3000, foEs])

    return num_maps


def IRI_density_1day(year, month, day, aUT, alon, alat, aalt, F107,
                     coeff_dir=None, foF2_coeff='URSI',
                     hmF2_model='SHU2015', coord='GEO'):
    """Output ionospheric parameters for a particular day.

    Parameters
    ----------
    year : int
        Year.
    month : int
        Month of the year.
    day : int
        Day of the month.
    aUT : float, int, or array-like
        UT time in [hour]. Scalar inputs will be converted to a Numpy array.
        Shape (N_T,)
    alon : float, list, or array-like
        Flattened array of longitude (geographic or quasi-dipole) [deg] or
        magnetic local time [hour]. Scalar inputs will be converted to a Numpy
        array.
        Shape (N_G,)
    alat : float, list, or array-like
        Flattened array of latitude (geographic or quasi-dipole) [deg]. Scalar
        inputs will be converted to a Numpy array.
        Shape (N_G,)
    aalt : array-like
        Altitude [km]. Scalar inputs will be converted to a Numpy array.
        Shape (N_V,)
    F107 : int or float
        User provided F10.7 solar flux index [SFU].
    coeff_dir: str
        Directory where the coefficient files are stored. If None, uses the
        default coefficient files stored in PyIRI.coeff_dir. (default=None)
    foF2_coeff : str
        Coefficients to use for foF2. Options are 'URSI' and 'CCIR'.
        (default='URSI')
    hmF2_model : str
        Model to use for hmF2. Options are 'SHU2015', 'AMTB2013', and
        'BSE1979'. (default='SHU2015')
    coord : str
        Coordinate system. Options are 'GEO' for geographic, 'QD' for quasi-
        dipole, and 'MLT' for magnetic local time. (default='GEO')

    Returns
    -------
    F2 : dict
        'Nm': Peak density of F2 region [m-3].
        'fo' : Critical frequency of F2 region [MHz].
        'M3000' : Obliquity factor for a distance of 3,000 km. Defined as
        refracted in the ionosphere, can be received at a distance of 3,000 km
        [unitless].
        'hm' : Height of the F2 peak [km].
        'B_top' : PyIRI top thickness of the F2 region [km].
        'B_bot' : PyIRI bottom thickness of the F2 region in [km].
        'B0' : IRI ABT-2009 bottom thickness parameter of the F2 region [km].
        'B1' : IRI ABT-2009 bottom shape parameter of the F2 region [unitless].
        Shape (N_T, N_G)
    F1 : dict
        'Nm' : Peak density of F1 region [m-3].
        'fo' : Critical frequency of F1 region [MHz].
        'P' : Probability occurrence of F1 region [unitless].
        'hm' : Height of the F1 peak [km].
        'B_bot' : Bottom thickness of the F1 region [km].
        Shape (N_T, N_G)
    E : dict
        'Nm' : Peak density of E region [m-3].
        'fo' : Critical frequency of E region [MHz].
        'hm' : Height of the E peak [km].
        'B_top' : Bottom thickness of the E region [km].
        'B_bot' : Bottom thickness of the E region [km].
        Shape (N_T, N_G)
    Es : dict
        'Nm' : Peak density of Es region [m-3].
        'fo' : Critical frequency of Es region [MHz].
        'hm' : Height of the Es peak [km].
        'B_top' : Bottom thickness of the Es region [km].
        'B_bot' : Bottom thickness of the Es region [km].
        Shape (N_T, N_G)
    sun : dict
        'lon' : Subsolar point longitude (geographic or quasi-dipole) [deg] or
        magnetic local time [hour].
        'lat' : Subsolar point latitude (geographic or quasi-dipole) [deg].
        Shape (N_T,)
    mag : dict
        'inc' : Inclination of the magnetic field [deg].
        'modip' : Modified dip angle [deg].
        'mag_dip_lat' : Magnetic dip latitude [deg].
        Shape (N_G,) if coord='GEO' or 'QD', (N_T, N_G) if coord='MLT'
    EDP : numpy.ndarray
        Electron density profiles [m-3].
        Shape (N_T, N_V, N_G)

    Notes
    -----
    This function returns ionospheric parameters and 3-D electron density for a
    given day and provided F10.7 solar flux index.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    Servan-Schreiber et al. (2026), A Major Update to the PyIRI Model, Space
    Weather.

    """
    # Set coefficient file path if none given
    if coeff_dir is None:
        coeff_dir = PyIRI.coeff_dir

    # Convert inputs to Numpy arrays
    aUT = ml.to_numpy_array(aUT)
    alon = ml.to_numpy_array(alon)
    alat = ml.to_numpy_array(alat)
    aalt = ml.to_numpy_array(aalt)
    apdtime = (pd.to_datetime(dt.datetime(year, month, 15))
               + pd.to_timedelta(aUT, 'hours'))

    if coord == 'GEO':
        aglat, aglon = alat, alon

    elif coord == 'MLT':
        aqdlat, amlt = alat, alon
        aglat, aglon = np.zeros((2, aUT.size, alat.size))
        for iUT in range(len(aUT)):
            pdtime = apdtime[iUT]
            aglat[iUT, :], aglon[iUT, :] = Apex(aqdlat, amlt, pdtime,
                                                'MLT_2_GEO')

    elif coord == 'QD':
        aqdlat, aqdlon = alat, alon
        aglat, aglon = Apex(aqdlat, aqdlon, apdtime[0], 'QD_2_GEO')

    else:
        raise ValueError("Coordinate system must be 'GEO', 'QD', or 'MLT'.")

    # Calculate required monhtly means and associated weights
    t_before, t_after, fr1, fr2 = ml.day_of_the_month_corr(year, month, day)

    foF2, hmF2, B0, B1, M3000, foEs = (
        IRI_sh_params(
            t_before.year,
            t_before.month,
            aUT,
            alon,
            alat,
            coeff_dir,
            foF2_coeff,
            hmF2_model,
            coord) * fr1
        + IRI_sh_params(
            t_after.year,
            t_after.month,
            aUT,
            alon,
            alat,
            coeff_dir,
            foF2_coeff,
            hmF2_model,
            coord) * fr2
    )

    # Interpolate parameters in solar activity
    foF2 = ml.solar_interpolate(foF2[:, :, 0], foF2[:, :, 1], F107,
                                solidx='IG12', solmin=0, solmax=100)
    if hmF2_model == 'SHU2015':
        hmF2 = ml.solar_interpolate(hmF2[:, :, 0], hmF2[:, :, 1], F107,
                                    solidx='IG12', solmin=0, solmax=100)
    elif hmF2_model == 'AMTB2013':
        hmF2 = ml.solar_interpolate(hmF2[:, :, 0], hmF2[:, :, 1], F107,
                                    solidx='R12', solmin=0, solmax=100)
    M3000 = ml.solar_interpolate(M3000[:, :, 0], M3000[:, :, 1], F107,
                                 solidx='R12', solmin=0, solmax=100)
    B0 = ml.solar_interpolate(B0[:, :, 0], B0[:, :, 1], F107, solidx='R12',
                              solmin=0, solmax=100)
    B1 = ml.solar_interpolate(B1[:, :, 0], B1[:, :, 1], F107, solidx='R12',
                              solmin=0, solmax=100)
    foEs = ml.solar_interpolate(foEs[:, :, 0], foEs[:, :, 1], F107,
                                solidx='R12', solmin=0, solmax=100)

    # Define constant parameters
    hmEs = 100. + np.zeros(foEs.shape)
    hmE = 110. + np.zeros(foEs.shape)
    B_E_top = 7. + np.zeros((foEs.shape))
    B_E_bot = 5. + np.zeros((foEs.shape))
    B_Es_top = 1. + np.zeros((foEs.shape))
    B_Es_bot = 1. + np.zeros((foEs.shape))

    # Find magnetic inclination
    decimal_year = ml.decimal_year(apdtime[0])

    # Calculate magnetic inclination, modified dip angle, and magnetic dip
    # latitude using IGRF at 300 km altitude
    inc = igrf.inclination(coeff_dir,
                           decimal_year,
                           aglon, aglat, 300., only_inc=True)
    modip = igrf.inc2modip(inc, aglat)
    mag_dip_lat = igrf.inc2magnetic_dip_latitude(inc)

    # foE, solar zenith angle, subsolar point coordinates
    foE, solzen, solzen_eff, slon, slat = gammaE_dynamic(year, month, day, aUT,
                                                         aglon, aglat, F107,
                                                         coord)

    # BSE1979 model for hmF2
    if hmF2_model == 'BSE1979':
        hmF2 = BSE_1979_model(M3000, foE, foF2, modip, F107)

    # Correct for linear interpolation
    # fo is interpolated linearly, and Nm is then found from fo
    NmF2 = ml.freq2den(foF2)
    NmE = ml.freq2den(foE)
    NmEs = ml.freq2den(foEs)

    # Introduce a minimum limit for the peaks to avoid negative density (for
    # high F10.7, extrapolation can cause NmF2 to go negative)
    NmF2 = ml.limit_Nm(NmF2)
    NmE = ml.limit_Nm(NmE)
    NmEs = ml.limit_Nm(NmEs)

    # Probability of F1 layer appearance based on solar zenith angle
    P_F1 = Probability_F1_with_solzen(solzen)

    # Derive dependent F1 parameters after the interpolation so that the F1
    # location does not carry the little errors caused by the interpolation
    NmF1, foF1, hmF1, B_F1_bot = derive_dependent_F1_parameters(
        P_F1,
        NmF2,
        hmF2,
        B0,
        B1,
        hmE)

    B_F2_top, B_F2_bot = thickness_F2(NmF2, foF2, M3000, hmF2, F107)

    # --------------------------------------------------------------------------
    # Add all parameters to dictionaries:
    F2 = {'Nm': NmF2,
          'fo': foF2,
          'M3000': M3000,
          'hm': hmF2,
          'B0': B0,
          'B1': B1,
          'B_top': B_F2_top,
          'B_bot': B_F2_bot}
    F1 = {'Nm': NmF1,
          'fo': foF1,
          'P': P_F1,
          'hm': hmF1,
          'B_bot': B_F1_bot}
    E = {'Nm': NmE,
         'fo': foE,
         'hm': hmE,
         'B_bot': B_E_bot,
         'B_top': B_E_top,
         'solzen': solzen,
         'solzen_eff': solzen_eff}
    Es = {'Nm': NmEs,
          'fo': foEs,
          'hm': hmEs,
          'B_bot': B_Es_bot,
          'B_top': B_Es_top}
    sun = {'lon': slon,
           'lat': slat}
    mag = {'inc': inc,
           'modip': modip,
           'mag_dip_lat': mag_dip_lat}

    # Construct density
    EDP = EDP_builder_continuous(F2, F1, E, aalt)

    return F2, F1, E, Es, sun, mag, EDP


def IRI_monthly_mean_par(year, month, aUT, alon, alat, solidx='IG12',
                         solmin=0, solmax=100, coeff_dir=None,
                         foF2_coeff='URSI', hmF2_model='SHU2015', coord='GEO'):
    """Output ionospheric parameters for a particular day.

    Parameters
    ----------
    year : int
        Year.
    month : int
        Month of the year.
    day : int
        Day of the month.
    aUT : float, int, or array-like
        UT time in [hour]. Scalar inputs will be converted to a Numpy array.
        Shape (N_T,)
    alon : float, list, or array-like
        Flattened array of longitude (geographic or quasi-dipole) [deg] or
        magnetic local time [hour]. Scalar inputs will be converted to a Numpy
        array.
        Shape (N_G,)
    alat : float, list, or array-like
        Flattened array of latitude (geographic or quasi-dipole) [deg]. Scalar
        inputs will be converted to a Numpy array.
        Shape (N_G,)
    solidx : str
        User choice of solar index (F107, IG12, or R12).
    solmin : int or float
        User choice of solar minimum.
    solmax : int or float
        User choice of solar maximum.
    coeff_dir: str
        Directory where the coefficient files are stored. If None, uses the
        default coefficient files stored in PyIRI.coeff_dir. (default=None)
    foF2_coeff : str
        Coefficients to use for foF2. Options are 'URSI' and 'CCIR'.
        (default='URSI')
    hmF2_model : str
        Model to use for hmF2. Options are 'SHU2015', 'AMTB2013', and
        'BSE1979'. (default='SHU2015')
    coord : str
        Coordinate system. Options are 'GEO' for geographic, 'QD' for quasi-
        dipole, and 'MLT' for magnetic local time. (default='GEO')

    Returns
    -------
    F2 : dict
        'Nm': Peak density of F2 region [m-3].
        'fo' : Critical frequency of F2 region [MHz].
        'M3000' : Obliquity factor for a distance of 3,000 km. Defined as
        refracted in the ionosphere, can be received at a distance of 3,000 km
        [unitless].
        'hm' : Height of the F2 peak [km].
        'B_top' : PyIRI top thickness of the F2 region [km].
        'B_bot' : PyIRI bottom thickness of the F2 region in [km].
        'B0' : IRI ABT-2009 bottom thickness parameter of the F2 region [km].
        'B1' : IRI ABT-2009 bottom shape parameter of the F2 region [unitless].
        Shape (N_T, N_G, 2)
    F1 : dict
        'Nm' : Peak density of F1 region [m-3].
        'fo' : Critical frequency of F1 region [MHz].
        'P' : Probability occurrence of F1 region [unitless].
        'hm' : Height of the F1 peak [km].
        'B_bot' : Bottom thickness of the F1 region [km].
        Shape (N_T, N_G, 2)
    E : dict
        'Nm' : Peak density of E region [m-3].
        'fo' : Critical frequency of E region [MHz].
        'hm' : Height of the E peak [km].
        'B_top' : Bottom thickness of the E region [km].
        'B_bot' : Bottom thickness of the E region [km].
        Shape (N_T, N_G, 2)
    Es : dict
        'Nm' : Peak density of Es region [m-3].
        'fo' : Critical frequency of Es region [MHz].
        'hm' : Height of the Es peak [km].
        'B_top' : Bottom thickness of the Es region [km].
        'B_bot' : Bottom thickness of the Es region [km].
        Shape (N_T, N_G, 2)
    sun : dict
        'lon' : Subsolar point longitude (geographic or quasi-dipole) [deg] or
        magnetic local time [hour].
        'lat' : Subsolar point latitude (geographic or quasi-dipole) [deg].
        Shape (N_T,)
    mag : dict
        'inc' : Inclination of the magnetic field [deg].
        'modip' : Modified dip angle [deg].
        'mag_dip_lat' : Magnetic dip latitude [deg].
        Shape (N_G,) if coord='GEO' or 'QD', (N_T, N_G) if coord='MLT'

    Notes
    -----
    This function returns monthly mean ionospheric parameters between a
    user-selected solar min and solar max.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    Servan-Schreiber et al. (2026), A Major Update to the PyIRI Model, Space
    Weather.

    """

    # Set coefficient file path if none given
    if coeff_dir is None:
        coeff_dir = PyIRI.coeff_dir

    # Convert inputs to Numpy arrays
    aUT = ml.to_numpy_array(aUT)
    alon = ml.to_numpy_array(alon)
    alat = ml.to_numpy_array(alat)

    # Compute ionospheric parameters for solar min and solar max
    if solidx == 'IG12':
        F107min = ml.IG12_2_F107(solmin)
        F107max = ml.IG12_2_F107(solmax)
    elif solidx == 'R12':
        F107min = ml.R12_2_F107(solmin)
        F107max = ml.R12_2_F107(solmax)

    F2min, F1min, Emin, Esmin, sun, mag, _ = IRI_density_1day(
        year, month, 15, aUT, alon,
        alat, 0, F107min,
        coeff_dir=coeff_dir,
        foF2_coeff=foF2_coeff,
        hmF2_model=hmF2_model,
        coord=coord
    )

    F2max, F1max, Emax, Esmax, sun, mag, _ = IRI_density_1day(
        year, month, 15, aUT, alon,
        alat, 0, F107max,
        coeff_dir=coeff_dir,
        foF2_coeff=foF2_coeff,
        hmF2_model=hmF2_model,
        coord=coord
    )

    F2 = {k: np.stack([F2min[k], F2max[k]], axis=-1) for k in F2min}
    F1 = {k: np.stack([F1min[k], F1max[k]], axis=-1) for k in F1min}
    E = {k: np.stack([Emin[k], Emax[k]], axis=-1) for k in Emin}
    Es = {k: np.stack([Esmin[k], Esmax[k]], axis=-1) for k in Esmin}

    return F2, F1, E, Es, sun, mag


def create_reg_grid_geo_or_mag(hr_res=1, lat_res=1, lon_res=1, alt_res=10,
                               alt_min=0, alt_max=700, coord='GEO'):
    """Create a regular grid in geographic or magnetic coordinates.

    Parameters
    ----------
    hr_res : int or float
        Time resolution [hour]. (default=1)
    lat_res : int or float
        Latitude resolution (geographic or quasi-dipole) [deg]. (default=1)
    lon_res : int or float
        Longitude (geographic or quasi-dipole) [deg] or magnetic
        local time [1/15 hour] resolution. (default=1)
    alt_res : int or float
        Altitude resolution [km]. (default=10)
    alt_min : int or float
        Altitude minimum [km]. (default=0)
    alt_max : int or float
        Altitude maximum [km]. (default=700)
    coord : str
        Coordinate system. Options are 'GEO' for geographic, 'QD' for quasi-
        dipole, and 'MLT' for magnetic local time. (default='GEO')

    Returns
    -------
    alon : numpy.ndarray
        Flattened array of longitude (geographic or quasi-dipole) [deg] or
        magnetic local time [hour].
        Shape ((180 / lat_res + 1) * (360 / lon_res + 1),)
    alat : numpy.ndarray
        Flattened array of latitude (geographic or quasi-dipole) [deg].
        Shape ((180 / lat_res + 1) * (360 / lon_res + 1),)
    alon_2d : numpy.ndarray
        2-D array of longitude (geographic or quasi-dipole) [deg] or
        magnetic local time [hour].
        Shape (180 / lat_res + 1, 360 / lon_res + 1)
    alat_2d : numpy.ndarray
        2-D array of latitude (geographic or quasi-dipole) [deg].
        Shape (180 / lat_res + 1, 360 / lon_res + 1)
    aalt : numpy.ndarray
        Array of altitudes [km].
        Shape (24 / hour_res,)
    aUT : numpy.ndarray
        Array of UT times [hour].
        Shape ((alt_max - alt_min) / alt_res + 1,)

    Notes
    -----
    This function creates a regular global grid in geographic, quasi-dipole, or
    magnetic local time coordinates using the given spatial resolution lon_res,
    lat_res; an array of altitudes using the given altitude resolution alt_res
    for the vertical dimension of electron density profiles; and a time array
    using the given temporal resolution hr_res.

    """
    # Check lat_res, lon_res, alt_res, and hr_res validity
    lat_res_check = 180 % lat_res != 0
    lon_res_check = 360 % lon_res != 0
    alt_res_check = (alt_max - alt_min) % alt_res != 0
    hr_res_check = 24 % hr_res != 0

    if lat_res_check:
        raise ValueError("Latitude resolution does not evenly divide 180.")
    if lon_res_check:
        raise ValueError("Longitude resolution does not evenly divide 360.")
    if alt_res_check:
        raise ValueError("Altitude resolution does not evenly divide "
                         f"{alt_max - alt_min} (alt_max - alt_mmin).")
    if hr_res_check:
        raise ValueError("Time resolution does not evenly divide 24.")

    # Create latitude and longitude 2-D grids
    alon = np.arange(0, 360 + lon_res, lon_res)
    if coord == 'MLT':
        alon = alon / 15
    alat = np.arange(90, -90 - lat_res, -lat_res)
    alon_2d, alat_2d = np.meshgrid(alon, alat)

    # Create flattened latitude and longitude grids
    alon = alon_2d.reshape(alon_2d.size)
    alat = alat_2d.reshape(alat_2d.size)

    # Create altitude array
    aalt = np.arange(alt_min, alt_max + alt_res, alt_res)

    # Create time array (24 excluded)
    aUT = np.arange(0, 24, hr_res)

    return alon, alat, alon_2d, alat_2d, aalt, aUT


def run_iri_reg_grid(year, month, day, F107, coeff_dir=None, hr_res=1,
                     lat_res=1, lon_res=1, alt_res=10, alt_min=0, alt_max=700,
                     foF2_coeff='URSI', hmF2_model='SHU2015', coord='GEO'):
    """Run IRI for a single day on a regular grid.

    Parameters
    ----------
    year : int
        Year.
    month : int
        Month of the year.
    day : int
        Day of the month.
    F107 : int or float
        User provided F10.7 solar flux index [sfu].
    coeff_dir: str
        Directory where the coefficient files are stored. If None, uses the
        default coefficient files stored in PyIRI.coeff_dir. (default=None)
    hr_res : int or float
        Time resolution [hour]. (default=1)
    lat_res : int or float
        Latitude resolution (geographic or quasi-dipole) [deg]. (default=1)
    lon_res : int or float
        Longitude (geographic or quasi-dipole) [deg] or magnetic
        local time [1/15 hour] resolution. (default=1)
    alt_res : int or float
        Altitude resolution [km]. (default=10)
    alt_min : int or float
        Altitude minimum [km]. (default=0)
    alt_max : int or float
        Altitude maximum [km]. (default=700)
    foF2_coeff : str
        Coefficients to use for foF2. Options are 'URSI' and 'CCIR'.
        (default='URSI')
    hmF2_model : str
        Model to use for hmF2. Options are 'SHU2015', 'AMTB2013', and
        'BSE1979'. (default='SHU2015')
    coord : str
        Coordinate system. Options are 'GEO' for geographic, 'QD' for quasi-
        dipole, and 'MLT' for magnetic local time. (default='GEO')

    Returns
    -------
    alon : numpy.ndarray
        Flattened array of longitude (geographic or quasi-dipole) [deg] or
        magnetic local time [hour].
        Shape (N_G,)
    alat : numpy.ndarray
        Flattened array of latitude (geographic or quasi-dipole) [deg].
        Shape (N_G,)
    alon_2d : numpy.ndarray
        2-D array of longitude (geographic or quasi-dipole) [deg] or
        magnetic local time [hour].
        Shape (180 // lat_res + 1, 360 // lon_res + 1)
    alat_2d : numpy.ndarray
        2-D array of latitude (geographic or quasi-dipole) [deg].
        Shape (180 // lat_res + 1, 360 // lon_res + 1)
    aalt : numpy.ndarray
        Array of altitudes [km].
        Shape (N_V,)
    aUT : numpy.ndarray
        Array of UT times [hour].
        Shape (N_T,)
    F2 : dict
        'Nm': Peak density of F2 region [m-3].
        'fo' : Critical frequency of F2 region [MHz].
        'M3000' : Obliquity factor for a distance of 3,000 km. Defined as
        refracted in the ionosphere, can be received at a distance of 3,000 km
        [unitless].
        'hm' : Height of the F2 peak [km].
        'B_top' : PyIRI top thickness of the F2 region [km].
        'B_bot' : PyIRI bottom thickness of the F2 region in [km].
        'B0' : IRI ABT-2009 bottom thickness parameter of the F2 region [km].
        'B1' : IRI ABT-2009 bottom shape parameter of the F2 region [unitless].
        Shape (N_T, N_G)
    F1 : dict
        'Nm' : Peak density of F1 region [m-3].
        'fo' : Critical frequency of F1 region [MHz].
        'P' : Probability occurrence of F1 region [unitless].
        'hm' : Height of the F1 peak [km].
        'B_bot' : Bottom thickness of the F1 region [km].
        Shape (N_T, N_G)
    E : dict
        'Nm' : Peak density of E region [m-3].
        'fo' : Critical frequency of E region [MHz].
        'hm' : Height of the E peak [km].
        'B_top' : Bottom thickness of the E region [km].
        'B_bot' : Bottom thickness of the E region [km].
        Shape (N_T, N_G)
    Es : dict
        'Nm' : Peak density of Es region [m-3].
        'fo' : Critical frequency of Es region [MHz].
        'hm' : Height of the Es peak [km].
        'B_top' : Bottom thickness of the Es region [km].
        'B_bot' : Bottom thickness of the Es region [km].
        Shape (N_T, N_G)
    sun : dict
        'lon' : Longitude of subsolar point [deg].
        'lat' : Latitude of subsolar point [deg].
        Shape (N_T,)
    mag : dict
        'inc' : Inclination of the magnetic field [deg].
        'modip' : Modified dip angle [deg].
        'mag_dip_lat' : Magnetic dip latitude [deg].
        Shape (N_G,) if coord='GEO' or 'QD', (N_T, N_G) if coord='MLT'
    EDP : numpy.ndarray
        Electron density profiles [m-3].
        Shape (N_T, N_V, N_G)

    See Also
    --------
    create_reg_grid_geo_or_mag

    """
    # Set coefficient file path if none given
    if coeff_dir is None:
        coeff_dir = PyIRI.coeff_dir

    # Create the grids
    alon, alat, alon_2d, alat_2d, aalt, aUT = create_reg_grid_geo_or_mag(
        hr_res=hr_res, lat_res=lat_res, lon_res=lon_res, alt_res=alt_res,
        alt_min=alt_min, alt_max=alt_max, coord=coord)

    # Run IRI for one day
    F2, F1, E, Es, sun, mag, EDP = IRI_density_1day(
        year, month, day, aUT, alon, alat, aalt, F107, coeff_dir,
        foF2_coeff, hmF2_model, coord)

    return alon, alat, alon_2d, alat_2d, aalt, aUT, F2, F1, E, Es, sun, mag, EDP


def run_seas_iri_reg_grid(year, month, coeff_dir=None, hr_res=1, lat_res=1,
                          lon_res=1, alt_res=10, alt_min=0, alt_max=700,
                          foF2_coeff='URSI', hmF2_model='SHU2015',
                          coord='GEO', solidx='IG12', solmin=0, solmax=100):
    """Run IRI for monthly mean parameters on a regular grid.

    Parameters
    ----------
    year : int
        Year.
    month : int
        Month of the year.
    coeff_dir: str
        Directory where the coefficient files are stored. If None, uses the
        default coefficient files stored in PyIRI.coeff_dir. (default=None)
    hr_res : int or float
        Time resolution [hour]. (default=1)
    lat_res : int or float
        Latitude resolution (geographic or quasi-dipole) [deg]. (default=1)
    lon_res : int or float
        Longitude (geographic or quasi-dipole) [deg] or magnetic
        local time [1/15 hour] resolution. (default=1)
    alt_res : int or float
        Altitude resolution [km]. (default=10)
    alt_min : int or float
        Altitude minimum [km]. (default=0)
    alt_max : int or float
        Altitude maximum [km]. (default=700)
    foF2_coeff : str
        Coefficients to use for foF2. Options are 'URSI' and 'CCIR'.
        (default='URSI')
    hmF2_model : str
        Model to use for hmF2. Options are 'SHU2015', 'AMTB2013', and
        'BSE1979'. (default='SHU2015')
    coord : str
        Coordinate system. Options are 'GEO' for geographic, 'QD' for quasi-
        dipole, and 'MLT' for magnetic local time. (default='GEO')
    solidx : str
        User selected solar index (F107, IG12 or R12).
    solmin : int or float
        User selected solar min.
    solmax : int or float
        User selected solar max.

    Returns
    -------
    alon : numpy.ndarray
        Flattened array of longitude (geographic or quasi-dipole) [deg] or
        magnetic local time [hour].
        Shape ((180 / lat_res + 1) * (360 / lon_res + 1),)
    alat : numpy.ndarray
        Flattened array of latitude (geographic or quasi-dipole) [deg].
        Shape ((180 / lat_res + 1) * (360 / lon_res + 1),)
    alon_2d : numpy.ndarray
        2-D array of longitude (geographic or quasi-dipole) [deg] or
        magnetic local time [hour].
        Shape (180 / lat_res + 1, 360 / lon_res + 1)
    alat_2d : numpy.ndarray
        2-D array of latitude (geographic or quasi-dipole) [deg].
        Shape (180 / lat_res + 1, 360 / lon_res + 1)
    aalt : numpy.ndarray
        Array of altitudes [km].
        Shape (24 / hour_res,)
    aUT : numpy.ndarray
        Array of UT times [hour].
        Shape ((alt_max - alt_min) / alt_res + 1,)
    F2 : dict
        'Nm': Peak density of F2 region [m-3].
        'fo' : Critical frequency of F2 region [MHz].
        'M3000' : Obliquity factor for a distance of 3,000 km. Defined as
        refracted in the ionosphere, can be received at a distance of 3,000 km
        [unitless].
        'hm' : Height of the F2 peak [km].
        'B_top' : PyIRI top thickness of the F2 region [km].
        'B_bot' : PyIRI bottom thickness of the F2 region in [km].
        'B0' : IRI ABT-2009 bottom thickness parameter of the F2 region [km].
        'B1' : IRI ABT-2009 bottom shape parameter of the F2 region [unitless].
        Shape (N_T, N_G, 2)
    F1 : dict
        'Nm' : Peak density of F1 region [m-3].
        'fo' : Critical frequency of F1 region [MHz].
        'P' : Probability occurrence of F1 region [unitless].
        'hm' : Height of the F1 peak [km].
        'B_bot' : Bottom thickness of the F1 region [km].
        Shape (N_T, N_G, 2)
    E : dict
        'Nm' : Peak density of E region [m-3].
        'fo' : Critical frequency of E region [MHz].
        'hm' : Height of the E peak [km].
        'B_top' : Bottom thickness of the E region [km].
        'B_bot' : Bottom thickness of the E region [km].
        Shape (N_T, N_G, 2)
    Es : dict
        'Nm' : Peak density of Es region [m-3].
        'fo' : Critical frequency of Es region [MHz].
        'hm' : Height of the Es peak [km].
        'B_top' : Bottom thickness of the Es region [km].
        'B_bot' : Bottom thickness of the Es region [km].
        Shape (N_T, N_G, 2)
    sun : dict
        'lon' : Longitude of subsolar point [deg].
        'lat' : Latitude of subsolar point [deg].
        Shape (N_T)
    mag : dict
        'inc' : Inclination of the magnetic field [deg].
        'modip' : Modified dip angle [deg].
        'mag_dip_lat' : Magnetic dip latitude [deg].
        Shape (N_G) if coord='GEO' or 'QD', (N_T, N_G) if coord='MLT'

    See Also
    --------
    create_reg_grid_geo_or_mag

    """
    # Set coefficient file path if none given
    if coeff_dir is None:
        coeff_dir = PyIRI.coeff_dir

    # Create coordinate, altitude, and time arrays
    alon, alat, alon_2d, alat_2d, aalt, aUT = create_reg_grid_geo_or_mag(
        hr_res=hr_res, lat_res=lat_res, lon_res=lon_res, alt_res=alt_res,
        alt_min=alt_min, alt_max=alt_max, coord=coord)

    # Run IRI for monthly mean parameters
    F2, F1, E, Es, sun, mag = IRI_monthly_mean_par(year, month, aUT, alon, alat,
                                                   coeff_dir=coeff_dir,
                                                   foF2_coeff=foF2_coeff,
                                                   hmF2_model=hmF2_model,
                                                   coord=coord,
                                                   solidx=solidx,
                                                   solmin=solmin,
                                                   solmax=solmax)

    return alon, alat, alon_2d, alat_2d, aalt, aUT, F2, F1, E, Es, sun, mag


def load_coeff_matrices(month, coeff_dir=None, foF2_coeff='URSI',
                        hmF2_model='SHU2015'):
    """Load ionospheric model coefficient matrices from NetCDF files.

    Parameters
    ----------
    month : int
        Month of the year.
    coeff_dir: str
        Directory where the coefficient files are stored. If None, uses the
        default coefficient files stored in PyIRI.coeff_dir. (default=None)
    foF2_coeff : str
        Coefficients to use for foF2. Options are 'URSI' and 'CCIR'.
        (default='URSI')
    hmF2_model : str
        Model to use for hmF2. Options are 'SHU2015', 'AMTB2013', and
        'BSE1979'. (default='SHU2015')

    Returns
    -------
    C : numpy.ndarray
        Coefficient matrix. N_P is the number of parameters (N_P=5 if
        hmF2_model == 'BSE1979' else N_P=6), N_sol=2 is the number of solar
        index values stored (IG12/R12=0 and IG12/R12=100), N_FS=9 is the number
        of real FS coefficients used, and N_SH=900 is the number of real SH
        coefficients used.
        Shape (N_P, N_sol, N_FS, N_SH)

    """
    # Set coefficient file path if none given
    if coeff_dir is None:
        coeff_dir = PyIRI.coeff_dir

    # Load foF2, B0, B1, and M3000F2 coefficients
    filenames = [f'foF2_{foF2_coeff}.nc', 'B0.nc', 'B1.nc', 'M3000F2.nc',
                 'foEs.nc']

    path = os.path.join(coeff_dir, 'SH', filenames[0])
    with nc.Dataset(path) as ds:
        C_month = ds['Coefficients'][:, month - 1, :, :]
        N_sol = C_month.shape[0]
        N_FS = C_month.shape[1]
        N_SH = C_month.shape[2]
        C = np.zeros((len(filenames), N_sol, N_FS, N_SH))

    for ids in range(len(filenames)):
        fname = filenames[ids]
        path = os.path.join(coeff_dir, 'SH', fname)
        with nc.Dataset(path) as ds:
            C_month = ds['Coefficients'][:, month - 1, :, :]
            C[ids, :, :, :] = C_month

    # Optionally add hmF2 coefficients
    if hmF2_model != 'BSE1979':
        hmF2_path = os.path.join(coeff_dir, 'SH', f'hmF2_{hmF2_model}.nc')
        with nc.Dataset(hmF2_path) as ds:
            C_month = ds['Coefficients'][:, month - 1, :, :]
            C_month = C_month[np.newaxis, :, :, :]
            C = np.concatenate([C, C_month], axis=0)

    return C


def gammaE_dynamic(year, month, day, aUT, alon, alat, F107, coord='GEO'):
    """Calculate numerical maps for critical frequency of E region.

    Parameters
    ----------
    year : int
        Year.
    month : int
        Month.
    day : int
        Day.
    aUT : int, float, or array-like
        UT time [hour]. Scalar inputs will be converted to a Numpy array.
        Shape (N_T,)
    alon : int, float, or array-like
        Flattened array of geographic longitudes [deg]. Scalar inputs will be
        converted to a Numpy array.
        Shape (N_G,) if coord='GEO', else (N_T, N_G)
    alat : int, float, or array-like
        Flattened array of geographic latitudes [deg]. Scalar inputs will be
        converted to a Numpy array.
        Shape (N_G,) if coord='GEO', else (N_T, N_G)
    F107 : int or float
        F10.7 solar input [sfu].

    Returns
    -------
    gamma_E : numpy.ndarray
        Critical frequency of the E region [MHz].
        Shape (N_T, N_G)
    solzen : numpy.ndarray
        Solar zenith angle [deg].
        Shape (N_T, N_G)
    solzen_eff : numpy.ndarray
        Effective solar zenith angle [deg].
        Shape (N_T, N_G)
    slon : numpy.ndarray
        Subsolar point longitude (geographic or quasi-dipole) [deg] or
        magnetic local time [hour].
        Shape (N_T,)
    slat : numpy.ndarray
        Subsolar point latitude (geographic or quasi-dipole) [deg].
        Shape (N_T,)

    Notes
    -----
    This function calculates numerical maps for foE for two levels of solar
    activity.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    """
    # Convert inputs to Numpy arrays
    aUT = ml.to_numpy_array(aUT)
    alon = ml.to_numpy_array(alon)
    alat = ml.to_numpy_array(alat)

    # Determine which coordinates are used as inputs
    # (N_G, N_G) = geographic coordinates
    # (N_T, N_G) = MLT or quasi-dipole coordinates
    if len(alon.shape) > 1:
        N_G = alon.shape[1]
        N_T = alon.shape[0]
    else:
        N_G = alon.shape[0]
        N_T = aUT.shape[0]

    # Initialize numerical map arrays for 2 levels of solar activity
    gamma_E = np.zeros((N_T, N_G))
    solzen_out = np.zeros((N_T, N_G))
    solzen_eff_out = np.zeros((N_T, N_G))

    # Initialize subsolar point location arrays
    slon = np.zeros((N_T))
    slat = np.zeros((N_T))

    # Loop to select the correct grid at each UT time
    for iIG in range(0, 2):
        for iUT in range(0, N_T):
            if len(alon.shape) > 1:
                alon_iUT = alon[iUT, :]
                alat_iUT = alat[iUT, :]
            else:
                alon_iUT = alon
                alat_iUT = alat

            solzen_t, slon_t, slat_t = ml.solzen_timearray_grid(
                year,
                month,
                day,
                np.array([aUT[iUT]]),
                alon_iUT,
                alat_iUT)

            # Convert subsolar point geographic coordinates back to magnetic if
            # user choice of coordinates is magnetic
            if coord != 'GEO':
                pdtime = (pd.to_datetime(dt.datetime(year, month, day)
                          + pd.to_timedelta(aUT[iUT], 'hours')))
                if coord == 'MLT':
                    slat_t, slon_t = Apex(slat_t, slon_t, pdtime, 'GEO_2_MLT')
                elif coord == 'QD':
                    slat_t, slon_t = Apex(slat_t, slon_t, pdtime, 'GEO_2_QD')

            # Convert [-180,180] to [0,360] degrees geographic longitude if
            # user choice is [0,360]
            elif np.nanmin(alon) >= 0:
                slon_t += 180

            slon[iUT] = slon_t[0]
            slat[iUT] = slat_t[0]
            solzen = solzen_t[0, :]

            # Effective solar zenith angle
            solzen_eff = ml.solzen_effective(solzen)

            # Save output arrays
            gamma_E[iUT, :] = ml.foE(month, solzen_eff, alat_iUT, F107)
            solzen_out[iUT, :] = solzen
            solzen_eff_out[iUT, :] = solzen_eff

    return gamma_E, solzen_out, solzen_eff_out, slon, slat


def Probability_F1_with_solzen(solzen):
    """Calculate probability occurrence of F1 layer.

    Parameters
    ----------
    solzen : array-like
        Array of solar zenith angles [deg].
        Shape (N_T, N_G)

    Returns
    -------
    a_P : numpy.ndarray
        Probability occurrence of F1 layer.
        Shape (N_T, N_G)

    Notes
    -----
    This function calculates the probability of an F1 layer as a numerical map.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    Bilitza et al. (2022), The International Reference Ionosphere
    model: A review and description of an ionospheric benchmark, Reviews
    of Geophysics, 60.

    """
    # Simplified constant value from Bilitza et al. (2022)
    gamma = 2.36

    a_P = (0.5 + 0.5 * np.cos(np.deg2rad(solzen)))**gamma

    return a_P


def Apex_geo_qd(Lat, Lon, dtime, transform_type):
    """Convert between geographic (GEO) and quasi-dipole (QD) coordinates.

    Parameters
    ----------
    Lat : array-like
        Geographic or quasi-dipole latitude grid [deg].
    Lon : array-like
        Geographic or quasi-dipole longitude grid [deg].
    dtime : datetime.datetime
        UT time specifying the target year for Apex coefficients.
    transform_type : str
        Transformation type:
            'GEO_2_QD' — convert from geographic to quasi-dipole
            'QD_2_GEO' — convert from quasi-dipole to geographic

    Returns
    -------
    tuple of numpy.ndarray
        (Latitude, Longitude) in the transformed coordinate system, reshaped
        to match the input grid dimensions.

    Notes
    -----
    This function converts between geographic (GEO) and quasi-dipole (QD)
    coordinates using precomputed spherical harmonic (SH) coefficients stored
    in Apex.nc.
    Coefficients are 4π-normalized real spherical harmonics.
    If the requested year is outside the available range, the nearest available
    year is used and an error is logged.
    Requires the file: PyIRI.coeff_dir / Apex / Apex.nc
    To see how the coefficient file 'Apex.nc' is computed, see the notebook
    docs / tutorials / Generate_Apex_Coefficients.ipynb (requires ApexPy and
    pyshtools).

    """
    # Convert Lat and Lon to numpy arrays
    Lat = ml.to_numpy_array(Lat)
    Lon = ml.to_numpy_array(Lon)

    # Flatten to 1D
    Lat_1d = Lat.reshape(Lat.size)
    Lon_1d = Lon.reshape(Lon.size)

    # Convert to spherical coordinates
    atheta = np.deg2rad(-(Lat_1d - 90.0))
    aphi = np.deg2rad(Lon_1d)

    # Open coefficient file safely
    filename = os.path.join(PyIRI.coeff_dir, 'Apex', 'Apex.nc')
    with nc.Dataset(filename, "r") as ds:
        ayear = ds['Year'][:]

        # Find the closest available year
        ind_year = np.abs(ayear - np.mean(dtime.year)).argmin()
        if (np.mean(dtime.year) < 1900) | (np.mean(dtime.year) > np.max(ayear)):
            msg = ("Apex coefficients are available only "
                   f"from {np.nanmax(ayear)} to {np.nanmax(ayear)}. "
                   f"Using {ayear[ind_year]} for input year {dtime.year}.")
            logger.error(msg)

        # Determine lmax from SH length
        lmax = int(np.sqrt(ds['QDLat'].shape[1]) - 1)

        # Build spherical harmonic basis
        F_SH = real_SH_func(atheta, aphi, lmax=lmax)

        # GEO to QD
        if transform_type == 'GEO_2_QD':
            C_QDLat = ds['QDLat'][ind_year, :]
            C_QDLon_cos = ds['QDLon_cos'][ind_year, :]
            C_QDLon_sin = ds['QDLon_sin'][ind_year, :]

            QDLat = oe.contract('ij,i->j', F_SH, C_QDLat)
            QDLon_cos = oe.contract('ij,i->j', F_SH, C_QDLon_cos)
            QDLon_sin = oe.contract('ij,i->j', F_SH, C_QDLon_sin)

            QDLon = PyIRI.main_library.adjust_longitude(
                np.degrees(np.arctan2(QDLon_sin, QDLon_cos)), 'to360'
            )
            return np.reshape(QDLat, Lat.shape), np.reshape(QDLon, Lon.shape)

        # QD to GEO
        elif transform_type == 'QD_2_GEO':
            C_GeoLat = ds['GeoLat'][ind_year, :]
            C_GeoLon_cos = ds['GeoLon_cos'][ind_year, :]
            C_GeoLon_sin = ds['GeoLon_sin'][ind_year, :]

            GeoLat = oe.contract('ij,i->j', F_SH, C_GeoLat)
            GeoLon_cos = oe.contract('ij,i->j', F_SH, C_GeoLon_cos)
            GeoLon_sin = oe.contract('ij,i->j', F_SH, C_GeoLon_sin)

            GeoLon = PyIRI.main_library.adjust_longitude(
                np.degrees(np.arctan2(GeoLon_sin, GeoLon_cos)), 'to360'
            )
            return np.reshape(GeoLat, Lat.shape), np.reshape(GeoLon, Lon.shape)

        else:
            raise ValueError("Parameter 'transform_type' must be either "
                             "'GEO_2_QD' or 'QD_2_GEO'.")


def Apex(Lat, Lon, dtime, transform_type):
    """Convert between GEO, QD, and MLT coordinates.

    Convert between geographic (GEO), quasi-dipole (QD), and magnetic local
    time (MLT) coordinates suing spherical harmoncis.

    Parameters
    ----------
    Lat : array-like
        Geographic or quasi-dipole latitude grid [deg].
    Lon : array-like
        Geographic or quasi-dipole longitude [deg] or magnetic local time grid
        [hour].
    dtime : datetime object
        UT time specifying the target year for Apex coefficients.
    transform_type : str
        Transformation type:
            'GEO_2_QD' — convert from geographic to quasi-dipole
            'QD_2_GEO' — convert from quasi-dipole to geographic
            'GEO_2_MLT' — convert from geographic to magnetic local time
            'MLT_2_GEO' — convert from magnetic local time to geographic
            'QD_2_MLT' — convert from quasi-dipole to magnetic local time
            'MLT_2_QD' — convert from magnetic local time to quasi-dipole

    Returns
    -------
    tuple of numpy.ndarray
        (Latitude, Longitude) in the transformed coordinate system, reshaped
        to match the input grid dimensions.

    See Also
    --------
    Apex_geo_qd

    """
    # Convert Lat and Lon to numpy arrays
    Lat = ml.to_numpy_array(Lat)
    Lon = ml.to_numpy_array(Lon)

    # Convert to selected coordinate system
    if transform_type in ['GEO_2_MLT', 'MLT_2_GEO', 'QD_2_MLT', 'MLT_2_QD']:
        sLon, sLat = find_subsolar(dtime)
        sQDLat, sQDLon = Apex_geo_qd(sLat, sLon, dtime, 'GEO_2_QD')

        if transform_type == 'GEO_2_MLT':
            QDLat, QDLon = Apex_geo_qd(Lat, Lon, dtime, 'GEO_2_QD')
            MLT = ((QDLon - sQDLon + 180) / 15) % 24
            convLat, convLon = QDLat, MLT

        elif transform_type == 'MLT_2_GEO':
            Lon = (Lon * 15 + sQDLon - 180) % 360
            convLat, convLon = Apex_geo_qd(Lat, Lon, dtime, 'QD_2_GEO')

        elif transform_type == 'QD_2_MLT':
            MLT = ((Lon - sQDLon + 180) / 15) % 24
            convLat, convLon = Lat, MLT

        elif transform_type == 'MLT_2_QD':
            QDLon = (Lon * 15 + sQDLon - 180) % 360
            convLat, convLon = Lat, QDLon

    elif transform_type == 'GEO_2_QD' or transform_type == 'QD_2_GEO':
        convLat, convLon = Apex_geo_qd(Lat, Lon, dtime, transform_type)

    else:
        raise ValueError("Parameter 'transform_type' must be"
                         "'GEO_2_QD', 'QD_2_GEO', "
                         "'MLT_2_QD', 'QD_2_MLT', "
                         "'GEO_2_MLT', or 'MLT_2_GEO'.")

    return convLat, convLon


def find_subsolar(dtime, adjust_type='to360'):
    """Calculate the geographic coordinates of the subsolar point.

    Parameters
    ----------
    dtime: datetime object
        UT time at which to calculate the coordinates.

    Returns
    ----------
    slon: float
        Geographic longitude of the subsolar point [deg].
    slat: float
        Geographic latitude of the subsolar point [deg].

    """
    # Calculate Julian time
    jday = ml.juldat(dtime)

    # Find subsolar point coordinates
    slon, slat = ml.subsolar_point(jday)

    # Adjust longitude to [-180:180] or [0:360]
    slon = ml.adjust_longitude(slon, adjust_type)

    return slon, slat


def real_SH_func(theta, phi, lmax=29):
    """Generate real-valued spherical harmonic basis functions.

    Generate real-valued spherical harmonic basis functions up to degree
    lmax (4pi-normalized). Includes Condon-Shortley phase. To use with SH
    coefficients created with pyshtools, set cspase=-1 and normalization='4pi'
    in pyshtools.

    Parameters
    ----------
    theta : array-like
        User input colatitudes [0-π] [rad].
        Shape (N_T, N_G) if grids are converted to magnetic local time from
        geographic or quasi-dipole coordinates, (N_G,) if grids are originally
        in magnetic local time coordinates
    phi : array-like
        User input longitudes [0-2π) [rad].
        Shape (N_T, N_G) grids are converted to magnetic local time from
        geographic or quasi-dipole coordinates, (N_G,) if grids are originally
        in magnetic local time coordinates
        coordinates
    lmax : int
        Maximum spherical harmonic degree (and order). (default=29)

    Returns
    -------
    F_SH : numpy.ndarray
        Real-valued spherical harmonic basis matrix, with N_SH=(lmax+1)**2.
        Each column corresponds to a pair of coordinates, each row to an SH
        mode. The SH mode of degree l and order m is stored in row i=l*(l+1)+m.
        Shape (N_SH, N_T, N_G) if input grids were converted to magnetic local
        time from geographic or quasi-dipole coordinates, (N_SH, N_G) if grids
        were originally in magnetic local time coordinates

    """
    # Convert inputs to arrays if lists or int or float
    theta = ml.to_numpy_array(theta)
    phi = ml.to_numpy_array(phi)

    # If theta has shape (N_G,), i.e., if user input coord='MLT', theta and
    # phi arrays must be artificially expanded to shape (1, N_G)
    mlt_flag = 0
    if len(theta.shape) == 1:
        mlt_flag = 1
        theta = theta[np.newaxis, :]
        # Ensure phi has same shape as z for broadcasting
        phi = phi[np.newaxis, :]

    # Argument for the associated Legendre polynomials
    z = np.cos(theta)  # shape (N_T, N_G)
    N_T, N_G = z.shape
    N_SH = (lmax + 1) ** 2

    # Broadcast phi to match z
    if phi.shape != z.shape:
        phi = np.broadcast_to(phi, z.shape)

    # Preallocate F_SH array
    F_SH = np.empty((N_SH, N_T, N_G), dtype=float)

    # Correct a bug in Scipy which causes P(l, m=0, z=-1.0) to be =1.0 instead
    # of =(-1.0)**l
    mask_pole = np.where(z == -1.0)
    mask_pole_time = mask_pole[0]
    mask_pole_pos = mask_pole[1]

    # Fill basis using index mapping i = l*(l+1)+m (with m in [-l..l])
    for L in range(lmax + 1):
        base = L * (L + 1)  # center index for this degree

        # m = 0
        P_l0 = ss.assoc_legendre_p(L, 0, z)[0]
        if (mask_pole_time.size > 0) or (mask_pole_pos.size > 0):
            P_l0 = P_l0.copy()
            P_l0[mask_pole_time, mask_pole_pos] = (-1.0) ** L

        # 4π normalization: sqrt((2 - δ_{m0}) * (2l + 1) * (l-m)! / (l+m)!)
        norm0 = np.sqrt((2 - 1) * (2 * L + 1) * ss.factorial(L - 0)
                        / ss.factorial(L + 0))
        F_SH[base, :, :] = P_l0 * norm0

        # m = 1..l
        for m in range(1, L + 1):
            P_lm = ss.assoc_legendre_p(L, m, z)[0]

            norm = np.sqrt((2 - 0) * (2 * L + 1) * ss.factorial(L - m)
                           / ss.factorial(L + m))
            P_lm *= norm

            # Correct index mapping:
            #   positive m: i_pos = l*(l+1) + m
            #   negative m: i_neg = l*(l+1) - m
            i_pos = base + m
            i_neg = base - m

            F_SH[i_pos, :, :] = P_lm * np.cos(m * phi)
            F_SH[i_neg, :, :] = P_lm * np.sin(m * phi)

    # If N_T=1, i.e., user input coord='MLT', then we can get rid of the N_T
    # dimension
    if mlt_flag:
        F_SH = F_SH.squeeze(1)

    return F_SH


def real_FS_func(aUT, N_FS_c=5):
    """Generate a real-valued Fourier Series (FS) basis matrix.

    Parameters
    ----------
    aUT : int, float, or array-like
        User input time values in Universal Time (UT) [0-24) [hour]. Scalar
        inputs will be converted to a Numpy array.
        Shape (N_T,)

    N_FS_c : int
        Number of complex Fourier coefficients to use as a truncation level.
        (default=5) The associated number of real Fourier coefficients is
        N_FS_r = 2 * N_FS_c - 1.

    Returns
    -------
    F_FS : numpy.ndarray
        Real-valued FS basis matrix.
        Shape (N_T, N_FS_r)

    """
    # Convert inputs to Numpy arrays
    aUT = ml.to_numpy_array(aUT)

    # Number of time samples
    N_T = aUT.size

    # Number of real Fourier coefficients
    # (constant term + sine + cosine pairs)
    N_FS_r = 2 * N_FS_c - 1

    # Initialize the output matrix for the Fourier Series basis
    F_FS = np.empty((N_T, N_FS_r))

    # Indices for harmonic terms (excluding the constant term)
    k_vals = np.arange(1, N_FS_c)

    # Angular frequencies (convert 24-hour period to radians)
    omega = 2 * np.pi * k_vals / 24

    # Compute the phase angles for each time and harmonic
    phase = np.outer(aUT, omega)

    # First column: constant (DC) term
    F_FS[:, 0] = 1

    # Even-indexed columns: cosine terms for each harmonic
    F_FS[:, 1::2] = np.cos(phase)

    # Odd-indexed columns: sine terms for each harmonic
    F_FS[:, 2::2] = np.sin(phase)

    # Return the complete real-valued Fourier Series basis matrix
    return F_FS


def EDP_builder_continuous(F2, F1, E, aalt):
    """Construct vertical EDP with continuous F1 layer.

    Parameters
    ----------
    F2 : dict
        Dictionary of parameters for the F2 layer.
        Shape (N_T, N_G)
    F1 : dict
        Dictionary of parameters for the F1 layer.
        Shape (N_T, N_G)
    E : dict
        Dictionary of parameters for the E layer.
        Shape (N_T, N_G)
    aalt : array-like
        1-D array of altitudes [km].
        Shape (N_V,)

    Returns
    -------
    density : numpy.ndarray
        3-D electron density profiles [m-3].
        Shape (N_T, N_V, N_G)

    Notes
    -----
    This function builds the EDP from the provided parameters for all time
    frames, all vertical and all horizontal points.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    """
    # Number of elements in horizontal dimension of grid
    N_G = F2['Nm'].shape[1]

    # Vertical dimension
    N_V = aalt.size

    # Time dimension
    N_T = F2['Nm'].shape[0]

    # Empty arrays
    density_F2 = np.zeros((N_V, N_T, N_G))
    full_F1 = np.zeros((N_V, N_T, N_G))
    density_F1 = np.zeros((N_V, N_T, N_G))
    density_E = np.zeros((N_V, N_T, N_G))

    # Grid altitude array to compare with hmFs later
    a_alt = np.full((N_G, N_T, N_V), aalt)
    a_alt = np.swapaxes(a_alt, 0, 2)

    # Import parameter data
    NmF2 = F2['Nm']
    NmF1 = F1['Nm']
    NmE = E['Nm']
    hmF2 = F2['hm']
    hmF1 = F1['hm']
    hmE = E['hm']
    B0 = F2['B0']
    B1 = F2['B1']
    B_F2_top = F2['B_top']
    B_F1_bot = F1['B_bot']
    B_E_bot = E['B_bot']
    B_E_top = E['B_top']

    # Set to some parameters if zero or lower:
    B_F2_top[np.where(B_F2_top <= 0)] = 10

    # Fill arrays with parameters to add height dimension and populate it
    # with same values, this is important to keep all operations in matrix
    # form
    shape = (N_V, N_T, N_G)
    a_NmF2 = np.full(shape, NmF2)
    a_NmF1 = np.full(shape, NmF1)
    a_NmE = np.full(shape, NmE)
    a_hmF2 = np.full(shape, hmF2)
    a_hmF1 = np.full(shape, hmF1)
    a_hmE = np.full(shape, hmE)
    a_B_F2_top = np.full(shape, B_F2_top)
    a_B0 = np.full(shape, B0)
    a_B1 = np.full(shape, B1)
    a_B_F1_bot = np.full(shape, B_F1_bot)
    a_B_E_top = np.full(shape, B_E_top)
    a_B_E_bot = np.full(shape, B_E_bot)

    # Drop functions to reduce contributions of the layers when adding them up
    multiplier_down_F2 = edpup.drop_down(a_alt, a_hmF2, a_hmE)
    multiplier_down_F1 = edpup.drop_down(a_alt, a_hmF1, a_hmE)
    multiplier_up = edpup.drop_up(a_alt, a_hmE, a_hmF2)

    # ------F2 region------
    # F2 top
    a = np.where(a_alt >= a_hmF2)
    density_F2[a] = ml.epstein_function_top_array(4. * a_NmF2[a], a_hmF2[a],
                                                  a_B_F2_top[a], a_alt[a])
    # F2 bottom down to E
    a = np.where((a_alt < a_hmF2) & (a_alt >= a_hmE))
    density_F2[a] = (Ramakrishnan_Rawer_function(a_NmF2[a], a_hmF2[a],
                                                 a_B0[a], a_B1[a], a_alt[a])
                     * multiplier_down_F2[a])

    # ------E region-------
    a = np.where((a_alt >= a_hmE) & (a_alt < a_hmF2))
    density_E[a] = ml.epstein_function_array(4. * a_NmE[a],
                                             a_hmE[a],
                                             a_B_E_top[a],
                                             a_alt[a]) * multiplier_up[a]
    a = np.where(a_alt < a_hmE)
    density_E[a] = ml.epstein_function_array(4. * a_NmE[a],
                                             a_hmE[a],
                                             a_B_E_bot[a],
                                             a_alt[a])

    # Add F2 and E layers
    density = density_F2 + density_E

    # ------F1 region------
    a = np.where((a_alt > a_hmE) & (a_alt < a_hmF1))
    full_F1[a] = ml.epstein_function_array(4. * a_NmF1[a],
                                           a_hmF1[a],
                                           a_B_F1_bot[a],
                                           a_alt[a]) * multiplier_down_F1[a]
    # Find the difference between the EDP and the F1 layer and add to the EDP
    # the positive part
    density_F1 = full_F1 - density
    density_F1[density_F1 < 0] = 0.

    density = density + density_F1 / 2.

    # Make 1 everything that is <= 0 (just in case)
    density[np.where(density <= 1.0)] = 1.0

    # Reshape to correct shape
    density = np.swapaxes(density, 0, 1)

    return density


def Ramakrishnan_Rawer_function(NmF2, hmF2, B0, B1, h):
    """Construct density of the Ramakrishnan & Rawer F2 bottomside.

    Parameters
    ----------
    NmF2 : array-like
        F2 region peak electron density [m-3].
    hmF2 : array-like
        F2 region peak height [km].
    B0 : array-like
        F2 region thickness parameter [km].
    B1 : array-like
        F2 region thickness parameter [km].
    h : array-like
        Altitude [km].

    Returns
    -------
    den : numpy.ndarray
        Constructed density [m-3]. Same shape as inputs.

    Notes
    -----
    This function constructs bottomside of F2 layer using
    Ramakrishnan & Rawer equation (as in IRI). All inputs are supposed
    to have same size.

    References
    ----------
    Bilitza et al. (2022), The International Reference Ionosphere
    model: A review and description of an ionospheric benchmark, Reviews
    of Geophysics, 60.

    """
    x = (hmF2 - h) / B0
    den = NmF2 * ml.fexp(-(np.sign(x) * (np.abs(x)**B1))) / np.cosh(x)

    return den


def derive_dependent_F1_parameters(P, NmF2, hmF2, B0, B1, hmE, threshold=0.1,
                                   thickness_fraction=0.75):
    """Combine DA with background F1 region.

    Parameters
    ----------
    P : array-like
        Probability of F1 to occurre from PyIRI.
    NmF2 : array-like
        NmF2 parameter peak density of F2 layer [m-3].
    hmF2 : array-like
        hmF2 parameter height of the peak of F2 [km].
    B0 : array-like
        B0 parameter thickness of F2 [km].
    B1 : array-like
        B1 parameter shape of F2.
    hmE : array-like
        hmE parameter height of E layer [km].
    threshold : flt
        Cuts the probability P at this threshhold.
        Default is 0.1.
    thickness_fraction : flt
        F1 thickness as a fraction of hmF1 - hmE.
        Default is 0.75.

    Returns
    -------
    NmF1 : array_like
        NmF1 parameter peak of F1 layer [m-3].
    foF1 : array_like
        foF1 parameter peak of F1 layer [MHz].
    hmF1 : array_like
        hmF1 parameter peak height of F1 layer [km].
    B_F1_bot : array_like
        B_F1_bot thickness of F1 layer [km].

    Notes
    -----
    This function derives F1 from F2 fields.

    """
    # Compute B_F1_bot using normalized probability P with a flexible
    # threshold.
    P_clipped = np.clip(P, threshold, 1)
    norm_shift = P_clipped - threshold
    max_shift = np.max(norm_shift)

    # Prevent division by zero
    norm_shifted = np.divide(norm_shift,
                             max_shift,
                             out=np.zeros_like(norm_shift),
                             where=max_shift != 0)

    # Map to [0.5, 1], then to [0, 1]
    norm_P = (np.clip(norm_shifted + 0.5, 0.5, 1) - 0.5) / 0.5

    # Estimate the F1 layer peak height (hmF1) using the B0 information
    hmF1 = hmF2 - B0

    # Don't let it go below 180 km.
    hmF1[hmF1 <= 180.] = 180.

    # Estimate bottom-side F1 thickness
    B_F1_bot = (hmF1 - hmE) * thickness_fraction * norm_P

    # Find the exact NmF1 at the hmF1 using F2 bottom function with
    # the drop down function
    NmF1 = Ramakrishnan_Rawer_function(NmF2,
                                       hmF2,
                                       B0,
                                       B1,
                                       hmF1) * edpup.drop_down(hmF1, hmF2, hmE)

    # Clip NmF1 to minimum of 1 to avoid crashing a later log10(NmF1)
    # with zeros.
    NmF1[NmF1 <= 0.] = 1.

    # Convert plasma density to freqeuncy
    foF1 = ml.den2freq(NmF1)

    return NmF1, foF1, hmF1, B_F1_bot


def BSE_1979_model(M3000, foF2, foE, modip, F107):
    """Return hmF2 for the BSE-1979 model for a specific F10.7.

    Parameters
    ----------
    M3000 : array-like
        Propagation parameter for F2 region.
    foE : array-like
        Critical frequency of E region [MHz].
    foF2 : array-like
        Critical frequency of F2 region [MHz].
    modip : array-like
        Modified dip angle [deg].
    F107: int or float
        F10.7 solar activity input [sfu].

    Returns
    -------
    hmF2 : array-like
        Height of F2 layer [km].

    Notes
    -----
    This function returns height of the F2 layer following the BSE-1979 model.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    Bilitza et al. (2022), The International Reference Ionosphere
    model: A review and description of an ionospheric benchmark, Reviews
    of Geophysics, 60.

    """
    R12 = ml.F107_2_R12(F107)

    ratio = foF2 / foE
    f1 = 0.00232 * R12 + 0.222
    f2 = 1.0 - R12 / 150.0 * np.exp(-(modip / 40.)**2)
    f3 = 1.2 - 0.0116 * np.exp(R12 / 41.84)
    f4 = 0.096 * (R12 - 25.) / 150.

    a = np.where(ratio < 1.7)
    ratio[a] = 1.7

    DM = f1 * f2 / (ratio - f3) + f4
    hmF2 = 1490.0 / (M3000 + DM) - 176.0

    return hmF2


def thickness_F2(NmF2, foF2, M3000, hmF2, F107):
    """Return thicknesses of the F2 layer.

    Parameters
    ----------
    foF2 : array-like
        Critical frequency of F2 region [MHz].
    M3000 : array-like
        Propagation parameter for F2 region related to hmF2.
    hmF2 : array-like
        Height of the F2 layer [km].
    F107 : array-like
        F10.7 value [sfu].

    Returns
    -------
    B_F2_top : array-like
        Thickness of F2 top [km].
    B_F2_bot : array-like
        Thickness of F2 bottom [km].

    Notes
    -----
    This function returns the thickness of the F2 top layer for an F10.7 solar
    activity input.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    """

    # In the actual NeQuick_2 code there is a typo, missing 0.01 which makes
    # the B 100 times smaller. It took me a long time to find this mistake,
    # while comparing with my results. The printed guide doesn't have this
    # typo.
    dNdHmx = -3.467 + 1.714 * np.log(foF2) + 2.02 * np.log(M3000)
    dNdHmx = 0.01 * ml.fexp(dNdHmx)
    B_F2_bot = 3.85e-12 * NmF2 / dNdHmx

    # B_F2_top..................................................................
    # set empty array
    k = foF2 * 0.

    # shape parameter depends on solar activity:
    R12 = ml.F107_2_R12(F107)
    k = (3.22 - 0.0538 * foF2 - 0.00664 * hmF2
         + (0.113 * hmF2 / B_F2_bot) + 0.00257 * R12)

    # auxiliary parameters x and v:
    x = (k * B_F2_bot - 150.) / 100.
    # thickness
    B_F2_top = (100. * x + 150.) / (0.041163 * x**2 - 0.183981 * x + 1.424472)

    return B_F2_top, B_F2_bot
