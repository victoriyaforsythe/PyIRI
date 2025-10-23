#!/usr/bin/env python
# --------------------------------------------------------
# Distribution statement A. Approved for public release.
# Distribution is unlimited.
# This work was supported by the Office of Naval Research.
# --------------------------------------------------------
"""This library contains Spherical Harmonics and Apex features for PyIRI.

The Apex coordinates were estimated for each year from 1900 to 2025 using
apexpy. These results were converted to the spherical harmonic coefficients up
to lmax=20 and saved in the .nc file PyIRI.coeff_dir / 'Apex' / 'Apex.nc'.

The ionospheric parameters of F2, F1, E, and Es are computed using spherical
harmonics based on the IRI (and PyIRI) model.

References
----------
Emmert et al. (2010), A computationally compact representation of
Magnetic-Apex and Quasi-Dipole coordinates with smooth base vectors,
J. Geophys. Res., 115(A8), A08322, doi:10.1029/2010JA015326.

Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
International Reference Ionosphere Modeling Implemented in Python,
Space Weather, ESS Open Archive, September 28, 2023,
doi:10.22541/essoar.169592556.61105365/v1.

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


def IRI_monthly_mean_par(year, month, aUT, alon, alat,
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
    This function returns monthly mean ionospheric parameters for min and max
    levels of solar activity, i.e., 12-month running mean of the Global
    Ionosonde Index IG12 of value 0 and 100.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

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

    # Set limits for solar driver based on IG12 = 0 - 100.
    aIG = np.array([0., 100.])

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

    # Find magnetic inclination
    decimal_year = ml.decimal_year(apdtime[0])

    # Calculate magnetic inclination, modified dip angle, and magnetic dip
    # latitude using IGRF at 300 km altitude
    inc = igrf.inclination(coeff_dir,
                           decimal_year,
                           aglon, aglat, 300., only_inc=True)
    modip = igrf.inc2modip(inc, aglat)
    mag_dip_lat = igrf.inc2magnetic_dip_latitude(inc)

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
    if hmF2_model != 'BSE1979':
        hmF2 = Params[4, :, :, :]

    NmF2 = ml.freq2den(foF2)  # Previously: * 1e11

    # Solar driven E region and locations of subsolar points
    foE, solzen, solzen_eff, slon, slat = gammaE_dynamic(year, month, aUT,
                                                         aglon, aglat,
                                                         aIG, coord)
    NmE = ml.freq2den(foE)  # Previously: * 1e11

    # Probability of F1 layer appearance based on solar zenith angle
    P_F1 = Probability_F1_with_solzen(solzen)

    # Introduce a minimum limit for the peaks to avoid negative density
    NmE = ml.limit_Nm(NmE)

    # Find heights of the F2 and E ionospheric layers
    if hmF2_model == 'BSE1979':
        if coord == 'MLT':
            modip_mod = modip.copy()[:, np.newaxis, :]
        else:
            modip_mod = modip.copy()
        hmF2, hmE, _ = ml.hm_IRI(M3000, foE, foF2, modip_mod, aIG)
    else:
        hmE = 110. + np.zeros(M3000.shape)

    # Find thicknesses of the F2 and E ionospheric layers
    B_F2_bot, B_F2_top, B_E_bot, B_E_top, B_Es_bot, B_Es_top = ml.thickness(
        foF2,
        M3000,
        hmF2,
        hmE,
        month,
        aIG)

    # Find height of the F1 layer based on the P and F2
    NmF1, foF1, hmF1, B_F1_bot = edpup.derive_dependent_F1_parameters(
        P_F1,
        NmF2,
        hmF2,
        B_F2_bot,
        hmE)

    # Add all parameters to dictionaries:
    F2 = {'Nm': NmF2,
          'fo': foF2,
          'M3000': M3000,
          'hm': hmF2,
          'B_top': B_F2_top,
          'B_bot': B_F2_bot,
          'B0': B0,
          'B1': B1}

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

    sun = {'lon': slon,
           'lat': slat}

    mag = {'inc': inc,
           'modip': modip,
           'mag_dip_lat': mag_dip_lat}

    return F2, F1, E, sun, mag


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

    """
    # Set coefficient file path if none given
    if coeff_dir is None:
        coeff_dir = PyIRI.coeff_dir

    # Convert inputs to Numpy arrays
    aUT = ml.to_numpy_array(aUT)
    alon = ml.to_numpy_array(alon)
    alat = ml.to_numpy_array(alat)
    aalt = ml.to_numpy_array(aalt)

    # Calculate required monhtly means and associated weights
    t_before, t_after, fr1, fr2 = ml.day_of_the_month_corr(year, month, day)

    F2_1, F1_1, E_1, sun_1, mag_1 = IRI_monthly_mean_par(t_before.year,
                                                         t_before.month,
                                                         aUT,
                                                         alon,
                                                         alat,
                                                         coeff_dir,
                                                         foF2_coeff,
                                                         hmF2_model,
                                                         coord)
    F2_2, F1_2, E_2, sun_2, mag_2 = IRI_monthly_mean_par(t_after.year,
                                                         t_after.month,
                                                         aUT,
                                                         alon,
                                                         alat,
                                                         coeff_dir,
                                                         foF2_coeff,
                                                         hmF2_model,
                                                         coord)

    F2 = ml.fractional_correction_of_dictionary(fr1, fr2, F2_1, F2_2)
    F1 = ml.fractional_correction_of_dictionary(fr1, fr2, F1_1, F1_2)
    E = ml.fractional_correction_of_dictionary(fr1, fr2, E_1, E_2)
    sun = ml.fractional_correction_of_dictionary(fr1, fr2, sun_1, sun_2)
    mag = ml.fractional_correction_of_dictionary(fr1, fr2, mag_1, mag_2)

    # Interpolate parameters in solar activity
    F2 = ml.solar_interpolation_of_dictionary(F2, F107)
    F1 = ml.solar_interpolation_of_dictionary(F1, F107)
    E = ml.solar_interpolation_of_dictionary(E, F107)

    # Introduce a minimum limit for the peaks to avoid negative density (for
    # high F10.7, extrapolation can cause NmF2 to go negative)
    F2['Nm'] = ml.limit_Nm(F2['Nm'])
    E['Nm'] = ml.limit_Nm(E['Nm'])

    # Derive dependent F1 parameters after the interpolation so that the F1
    # location does not carry the little errors caused by the interpolation
    NmF1, foF1, hmF1, B_F1_bot = edpup.derive_dependent_F1_parameters(
        F1['P'],
        F2['Nm'],
        F2['hm'],
        F2['B_bot'],
        E['hm'])

    # Update the F1 dictionary with the re-derived parameters
    F1['Nm'] = NmF1
    F1['hm'] = hmF1
    F1['fo'] = foF1
    F1['B_bot'] = B_F1_bot

    # Construct density
    EDP = EDP_builder_continuous(F2, F1, E, aalt)

    return F2, F1, E, sun, mag, EDP


def sporadic_E_monthly_mean(year, month, aUT, alon, alat, coeff_dir=None,
                            coord='GEO'):
    """Output monthly mean sporadic E layer using spherical harmonics.

    Parameters
    ----------
    year : int
        Year.
    month : int
        Month of the year.
    aUT : float, int, or array-like of float or int
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
    coord : str
        Coordinate system. Options are 'GEO' for geographic, 'QD' for quasi-
        dipole, and 'MLT' for magnetic local time. (default='GEO')

    Returns
    -------
    Es : dict
        'Nm' : Peak density of Es region [m-3].
        'fo' : Critical frequency of Es region [MHz].
        'hm' : Height of the Es peak [km].
        'B_top' : Bottom thickness of the Es region [km].
        'B_bot' : Bottom thickness of the Es region [km].
        Shape (N_T, N_G, 2)

    Notes
    -----
    This function returns monthly mean ionospheric parameters for min and max
    levels of solar activity, i.e., 12-month running mean of the Global
    Ionosonde Index IG12 of value 0 and 100.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

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
    C = load_Es_coeff_matrix(month, coeff_dir)

    n_FS_r = C.shape[1]
    n_FS_c = n_FS_r // 2 + 1
    F_FS = real_FS_func(aUT, n_FS_c)

    n_SH = C.shape[2]
    lmax = int(np.sqrt(n_SH)) - 1
    atheta = np.deg2rad(-(aqdlat - 90))
    aphi = np.deg2rad(amlt * 15)

    F_SH = real_SH_func(atheta, aphi, lmax=lmax)
    if coord == 'MLT':
        foEs = oe.contract('ij,pjk,kl->ilp', F_FS, C, F_SH)
    else:
        foEs = oe.contract('ij,pjk,kil->ilp', F_FS, C, F_SH)

    NmEs = ml.freq2den(foEs)

    hmEs = 110. + np.zeros(NmEs.shape)

    B_Es_top = 1. + np.zeros((hmEs.shape))
    B_Es_bot = 1. + np.zeros((hmEs.shape))

    Es = {'Nm': NmEs,
          'fo': foEs,
          'hm': hmEs,
          'B_top': B_Es_top,
          'B_bot': B_Es_bot}

    return Es


def sporadic_E_1day(year, month, day, aUT, alon, alat, F107, coeff_dir=None,
                    coord='GEO'):
    """Output sporadic E layer parameters for a particular day.

    Parameters
    ----------
    year : int
        Year.
    month : int
        Month of the year.
    day : int
        Day of the month.
    aUT : float, int, or array-like of float or int
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
    F107 : int or float
        User provided F10.7 solar flux index [SFU].
    coeff_dir: str
        Directory where the coefficient files are stored. If None, uses the
        default coefficient files stored in PyIRI.coeff_dir. (default=None)
    coord : str
        Coordinate system. Options are 'GEO' for geographic, 'QD' for quasi-
        dipole, and 'MLT' for magnetic local time. (default='GEO')

    Returns
    -------
    Es : dict
        'Nm' : Peak density of Es region [m-3].
        'fo' : Critical frequency of Es region [MHz].
        'hm' : Height of the Es peak [km].
        'B_top' : Bottom thickness of the Es region [km].
        'B_bot' : Bottom thickness of the Es region [km].
        Shape (N_T, N_G)

    Notes
    -----
    This function returns ionospheric parameters of the sporadic E layer for a
    given day and solar activity input.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    """
    # Set coefficient file path if none given
    if coeff_dir is None:
        coeff_dir = PyIRI.coeff_dir

    # Convert inputs to Numpy arrays
    aUT = ml.to_numpy_array(aUT)
    alon = ml.to_numpy_array(alon)
    alat = ml.to_numpy_array(alat)

    # Calculate required monhtly means and associated weights
    t_before, t_after, fr1, fr2 = ml.day_of_the_month_corr(year, month, day)

    Es_1 = sporadic_E_monthly_mean(t_before.year, t_before.month, aUT, alon,
                                   alat, coeff_dir, coord)
    Es_2 = sporadic_E_monthly_mean(t_after.year, t_after.month, aUT, alon,
                                   alat, coeff_dir, coord)

    Es = ml.fractional_correction_of_dictionary(fr1, fr2, Es_1, Es_2)

    # Interpolate parameters in solar activity
    Es = ml.solar_interpolation_of_dictionary(Es, F107)

    # Introduce a minimum limit for the peaks to avoid negative density (for
    # high F10.7, extrapolation can cause NmF2 to go negative)
    Es['Nm'] = ml.limit_Nm(Es['Nm'])

    return Es


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
        User provided F10.7 solar flux index in SFU.
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
    F2, F1, E, sun, mag, EDP = IRI_density_1day(
        year, month, day, aUT, alon, alat, aalt, F107, coeff_dir,
        foF2_coeff, hmF2_model, coord)

    return alon, alat, alon_2d, alat_2d, aalt, aUT, F2, F1, E, sun, mag, EDP


def run_seas_iri_reg_grid(year, month, coeff_dir=None, hr_res=1, lat_res=1,
                          lon_res=1, alt_res=10, alt_min=0, alt_max=700,
                          foF2_coeff='URSI', hmF2_model='SHU2015',
                          coord='GEO'):
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
    F2, F1, E, sun, mag = IRI_monthly_mean_par(year, month, aUT, alon, alat,
                                               coeff_dir, foF2_coeff,
                                               hmF2_model, coord)

    return alon, alat, alon_2d, alat_2d, aalt, aUT, F2, F1, E, sun, mag


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
        Coefficient matrix. N_P is the number of parameters (N_P=4 if
        hmF2_model == 'BSE1979' else N_P=5), N_IG=2 is the number of IG12
        values stored (IG12=0 and IG12=100), N_FS=9 is the number of real FS
        coefficients used, and N_SH=900 is the number of real SH coefficients
        used.
        Shape (N_P, N_IG, N_FS, N_SH)

    """
    # Set coefficient file path if none given
    if coeff_dir is None:
        coeff_dir = PyIRI.coeff_dir

    # Load foF2, B0, B1, and M3000F2 coefficients
    filenames = [f'foF2_{foF2_coeff}.nc', 'B0.nc', 'B1.nc', 'M3000F2.nc']

    path = os.path.join(coeff_dir, 'SH', filenames[0])
    with nc.Dataset(path) as ds:
        C_month = ds['Coefficients'][:, month - 1, :, :]
        N_IG = C_month.shape[0]
        N_FS = C_month.shape[1]
        N_SH = C_month.shape[2]
        C = np.zeros((len(filenames), N_IG, N_FS, N_SH))

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


def load_Es_coeff_matrix(month, coeff_dir=None):
    """Load sporadic E layer coefficient matrix from its NetCDF file.

    Parameters
    ----------
    month : int
        Month of the year.
    coeff_dir: str
        Directory where the coefficient files are stored. If None, uses the
        default coefficient files stored in PyIRI.coeff_dir. (default=None)

    Returns
    -------
    C : numpy.ndarray
        Coefficient matrix. N_IG=2 is the number of IG12 values stored (IG12=0
        and IG12=100), N_FS=9 is the number of real FS coefficients used, and
        N_SH=8100 is the number of real SH coefficients used.
        Shape (N_IG, N_FS, N_SH)

    """
    # Set coefficient file path if none given
    if coeff_dir is None:
        coeff_dir = PyIRI.coeff_dir

    # Load Es coefficients
    filename = 'foEs.nc'

    path = os.path.join(coeff_dir, 'SH', filename)
    with nc.Dataset(path) as ds:
        C = ds['Coefficients'][:, month - 1, :, :]

    return C


def gammaE_dynamic(year, month, aUT, alon, alat, aIG, coord='GEO'):
    """Calculate numerical maps for critical frequency of E region.

    Parameters
    ----------
    year : int
        Year.
    month : int
        Month.
    aUT : int, float, or array-like
        UT time [hour]. Scalar inputs will be converted to a Numpy array.
        Shape (N_T,)
    alon : int, float, or array-like
        Flattened array of geographic longitudes [deg]. Scalar inputs will be
        converted to a Numpy array.
        Shape (N_G,)
    alat : int, float, or array-like
        Flattened array of geographic latitudes [deg]. Scalar inputs will be
        converted to a Numpy array.
        Shape (N_G,)
    aIG : array-like
        Min and max of IG12 index.
        Shape (2,)

    Returns
    -------
    gamma_E : numpy.ndarray
        Critical frequency of the E region [MHz].
        Shape (N_T, N_G, 2)
    solzen : numpy.ndarray
        Solar zenith angle [deg].
        Shape (N_T, N_G, 2)
    solzen_eff : numpy.ndarray
        Effective solar zenith angle [deg].
        Shape (N_T, N_G, 2)
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
    # (N_G,) = MLT coordinates
    # (N_T, N_G) = geographic or quasi-dipole coordinates
    if len(alon.shape) > 1:
        N_G = alon.shape[1]
        N_T = alon.shape[0]
    else:
        N_G = alon.shape[0]
        N_T = aUT.shape[0]

    # Min and max of solar activity
    aF107_min_max = np.array([ml.IG12_2_F107(aIG[0]),
                              ml.IG12_2_F107(aIG[1])])

    # Initialize numerical map arrays for 2 levels of solar activity
    gamma_E = np.zeros((N_T, N_G, 2))
    solzen_out = np.zeros((N_T, N_G, 2))
    solzen_eff_out = np.zeros((N_T, N_G, 2))

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
                15,
                np.array([aUT[iUT]]),
                alon_iUT,
                alat_iUT)

            # Convert subsolar point geographic coordinates back to magnetic if
            # user choice of coordinates is magnetic
            if coord != 'GEO':
                pdtime = (pd.to_datetime(dt.datetime(year, month, 15)
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
            gamma_E[iUT, :, iIG] = ml.foE(month, solzen_eff, alat_iUT,
                                          aF107_min_max[iIG])
            solzen_out[iUT, :, iIG] = solzen
            solzen_eff_out[iUT, :, iIG] = solzen_eff

    return gamma_E, solzen_out, solzen_eff_out, slon, slat


def Probability_F1_with_solzen(solzen):
    """Calculate probability occurrence of F1 layer.

    Parameters
    ----------
    solzen : array-like
        Array of solar zenith angles [deg].
        Shape (N_T, N_G, 2)

    Returns
    -------
    a_P : numpy.ndarray
        Probability occurrence of F1 layer.
        Shape (N_T, N_G, 2)

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
    Requires the file: PyIRI.coeff_dir / 'Apex' / 'Apex.nc'

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
        if np.any(mask_pole):
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

    density = density + density_F1

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
