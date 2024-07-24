#!/usr/bin/env python
# --------------------------------------------------------
# Distribution statement A. Approved for public release.
# Distribution is unlimited.
# This work was supported by the Office of Naval Research.
# --------------------------------------------------------
"""This library contains components for PyIRI software.

References
----------
Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
International Reference Ionosphere Modeling Implemented in Python,
Space Weather, ESS Open Archive, September 28, 2023,
doi:10.22541/essoar.169592556.61105365/v1.

Bilitza et al. (2022), The International Reference Ionosphere
model: A review and description of an ionospheric benchmark, Reviews
of Geophysics, 60.

Nava et al. (2008). A new version of the NeQuick ionosphere
electron density model. J. Atmos. Sol. Terr. Phys., 70 (15),
doi:10.1016/j.jastp.2008.01.015.

Jones, W. B., Graham, R. P., & Leftin, M. (1966). Advances
in ionospheric mapping by numerical methods.

"""

import datetime as dt
from fortranformat import FortranRecordReader
import math
import numpy as np
import os

import PyIRI
import PyIRI.igrf_library as igrf
from PyIRI import logger


def IRI_monthly_mean_par(year, mth, aUT, alon, alat, coeff_dir, ccir_or_ursi=0):
    """Output monthly mean ionospheric parameters.

    Parameters
    ----------
    year : int
        Year.
    mth : int
        Month of year.
    aUT : array-like
        Array of universal time (UT) in hours. Must be Numpy array of any size
        [N_T].
    alon : array-like
        Flattened array of geographic longitudes in degrees. Must be Numpy
        array of any size [N_G].
    alat : array-like
        Flattened array of geographic latitudes in degrees. Must be Numpy array
        of any size [N_G].
    coeff_dir : str
        Place where coefficients are.
    ccir_or_ursi : int
        If 0 is given CCIR will be used for F2 critical frequency, if 1 then
        URSI. (default=0)

    Returns
    -------
    F2 : dict
        'Nm' is peak density of F2 region in m-3.
        'fo' is critical frequency of F2 region in MHz.
        'M3000' is the obliquity factor for a distance of 3,000 km.
        Defined as refracted in the ionosphere, can be received at a
        distance of 3,000 km, unitless.
        'hm' is height of the F2 peak in km.
        'B_topi is top thickness of the F2 region in km.
        'B_bot' is bottom thickness of the F2 region in km.
        Shape [N_T, N_G, 2].
    F1 : dict
        'Nm' is peak density of F1 region in m-3.
        'fo' is critical frequency of F1 region in MHz.
        'P' is the probability occurrence of F1 region, unitless.
        'hm' is height of the F1 peak in km.
        'B_bot' is bottom thickness of the F1 region in km.
        Shape [N_T, N_G, 2].
    E : dict
        'Nm' is peak density of E region in m-3.
        'fo' is critical frequency of E region in MHz.
        'hm' is height of the E peak in km.
        'B_top' is bottom thickness of the E region in km.
        'B_bot' is bottom thickness of the E region in km.
        Shape [N_T, N_G, 2].
    Es : dict
        'Nm' is peak density of Es region in m-3.
        'fo' is critical frequency of Es region in MHz.
        'hm' is height of the Es peak in km.
        'B_top' is bottom thickness of the Es region in km.
        'B_bot' is bottom thickness of the Es region in km.
        Shape [N_T, N_G, 2].
    sun : dict
        'lon' is longitude of subsolar point in degrees.
        'lat' is latitude of subsolar point in degrees.
        Shape [N_G]
    mag : dict
        'inc' is inclination of the magnetic field in degrees.
        'modip' is modified dip angle in degrees.
        'mag_dip_lat' is magnetic dip latitude in degrees.
        Shape [N_G]

    Notes
    -----
    This function returns monthly mean ionospheric parameters for min
    and max levels of solar activity that correspond to the Ionosonde
    Index IG12 of 0 and 100.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    """
    # Set limits for solar driver based of IG12 = 0 - 100.
    aIG = np.array([0., 100.])

    # Date and time for the middle of the month (day=15) that will be used to
    # find magnetic inclination
    dtime = dt.datetime(year, mth, 15)
    date_decimal = decimal_year(dtime)

    # -------------------------------------------------------------------------
    # Calculating magnetic inclanation, modified dip angle, and magnetic dip
    # latitude using IGRF at 300 km of altitude
    inc = igrf.inclination(coeff_dir, date_decimal, alon, alat)
    modip = igrf.inc2modip(inc, alat)
    mag_dip_lat = igrf.inc2magnetic_dip_latitude(inc)

    # --------------------------------------------------------------------------
    # Calculate diurnal Fourier functions F_D for the given time array aUT
    D_f0f2, D_M3000, D_Es_med = diurnal_functions(aUT)

    # --------------------------------------------------------------------------
    # Calculate geographic Jones and Gallet (JG) functions F_G for the given
    # grid and corresponding map of modip angles
    G_fof2, G_M3000, G_Es_med = set_gl_G(alon, alat, modip)

    # --------------------------------------------------------------------------
    # Read CCIR coefficients and form matrix U
    F_c_CCIR, F_c_URSI, F_M3000_c, F_Es_med = read_ccir_ursi_coeff(mth,
                                                                   coeff_dir)
    if ccir_or_ursi == 0:
        F_fof2_c = F_c_CCIR
    if ccir_or_ursi == 1:
        F_fof2_c = F_c_URSI

    # --------------------------------------------------------------------------
    # Multiply matrices (F_D U)F_G
    foF2, M3000, foEs = gamma(D_f0f2, D_M3000, D_Es_med, G_fof2, G_M3000,
                              G_Es_med, F_fof2_c, F_M3000_c, F_Es_med)

    # --------------------------------------------------------------------------
    # Probability of F1 layer to appear
    P_F1, foF1 = Probability_F1(year, mth, aUT, alon, alat, mag_dip_lat, aIG)

    # --------------------------------------------------------------------------
    # Solar driven E region and locations of subsolar points
    foE, slon, slat = gammaE(year, mth, aUT, alon, alat, aIG)

    # --------------------------------------------------------------------------
    # Convert critical frequency to the electron density (m-3)
    NmF2, NmF1, NmE, NmEs = freq_to_Nm(foF2, foF1, foE, foEs)

    # --------------------------------------------------------------------------
    # Find heights of the F2 and E ionospheric layers
    hmF2, hmE, hmEs = hm_IRI(M3000, foE, foF2, modip, aIG)

    # --------------------------------------------------------------------------
    # Find thicknesses of the F2 and E ionospheric layers
    B_F2_bot, B_F2_top, B_E_bot, B_E_top, B_Es_bot, B_Es_top = thickness(foF2,
                                                                         M3000,
                                                                         hmF2,
                                                                         hmE,
                                                                         mth,
                                                                         aIG)

    # --------------------------------------------------------------------------
    # Find height of the F1 layer based on the location and thickness of the
    # F2 layer
    hmF1 = hmF1_from_F2(NmF2, NmF1, hmF2, B_F2_bot)

    # --------------------------------------------------------------------------
    # Find thickness of the F1 layer
    B_F1_bot = find_B_F1_bot(hmF1, hmE, P_F1)

    # --------------------------------------------------------------------------
    # Add all parameters to dictionaries:
    F2 = {'Nm': NmF2,
          'fo': foF2,
          'M3000': M3000,
          'hm': hmF2,
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
         'B_top': B_E_top}
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

    return F2, F1, E, Es, sun, mag


def IRI_density_1day(year, mth, day, aUT, alon, alat, aalt, F107, coeff_dir,
                     ccir_or_ursi=0):
    """Output ionospheric parameters for a particular day.

    Parameters
    ----------
    year : int
        Year.
    mth : int
        Month of year.
    day : int
        Day of month.
    aUT : array-like
        Array of universal time (UT) in hours. Must be Numpy array of any size
        [N_T].
    alon : array-like
        Flattened array of geographic longitudes in degrees. Must be Numpy array
        of any size [N_G].
    alat : array-like
        Flattened array of geographic latitudes in degrees. Must be Numpy array
        of any size [N_G].
    aalt : array-like
        Array of altitudes in km. Must be Numpy array of any size [N_V].
    F107 : float
        User provided F10.7 solar flux index in SFU.
    coeff_dir : str
        Place where coefficients are located.
    ccir_or_ursi : int
        If 0 is given CCIR will be used for F2 critical frequency. If 1 then
        URSI coefficients. (default=0)

    Returns
    -------
    F2 : dict
        'Nm' is peak density of F2 region in m-3.
        'fo' is critical frequency of F2 region in MHz.
        'M3000' is the obliquity factor for a distance of 3,000 km.
        Defined as refracted in the ionosphere, can be received at a distance
        of 3,000 km, unitless.
        'hm' is height of the F2 peak in km.
        'B_topi is top thickness of the F2 region in km.
        'B_bot' is bottom thickness of the F2 region in km.
        Shape [N_T, N_G, 2].
    F1 : dict
        'Nm' is peak density of F1 region in m-3.
        'fo' is critical frequency of F1 region in MHz.
        'P' is the probability occurrence of F1 region, unitless.
        'hm' is height of the F1 peak in km.
        'B_bot' is bottom thickness of the F1 region in km.
        Shape [N_T, N_G, 2].
    E : dict
        'Nm' is peak density of E region in m-3.
        'fo' is critical frequency of E region in MHz.
        'hm' is height of the E peak in km.
        'B_top' is bottom thickness of the E region in km.
        'B_bot' is bottom thickness of the E region in km.
        Shape [N_T, N_G, 2].
    Es : dict
        'Nm' is peak density of Es region in m-3.
        'fo' is critical frequency of Es region in MHz.
        'hm' is height of the Es peak in km.
        'B_top' is bottom thickness of the Es region in km.
        'B_bot' is bottom thickness of the Es region in km.
        Shape [N_T, N_G, 2].
    sun : dict
        'lon' is longitude of subsolar point in degrees.
        'lat' is latitude of subsolar point in degrees.
        Shape [N_G].
    mag : dict
        'inc' is inclination of the magnetic field in degrees.
        'modip' is modified dip angle in degrees.
        'mag_dip_lat' is magnetic dip latitude in degrees.
        Shape [N_G].
    EDP : array-like
        Electron density profiles in m-3 with shape [N_T, N_V, N_G]

    Notes
    -----
    This function returns ionospheric parameters and 3-D electron density
    for a given day and provided F10.7 solar flux index.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    """

    # find out what monthly means are needed first and what their weights
    # will be
    t_before, t_after, fr1, fr2 = day_of_the_month_corr(year, mth, day)

    F2_1, F1_1, E_1, Es_1, sun_1, mag_1 = IRI_monthly_mean_par(t_before.year,
                                                               t_before.month,
                                                               aUT,
                                                               alon,
                                                               alat,
                                                               coeff_dir,
                                                               ccir_or_ursi)
    F2_2, F1_2, E_2, Es_2, sun_2, mag_2 = IRI_monthly_mean_par(t_after.year,
                                                               t_after.month,
                                                               aUT,
                                                               alon,
                                                               alat,
                                                               coeff_dir,
                                                               ccir_or_ursi)

    F2 = fractional_correction_of_dictionary(fr1, fr2, F2_1, F2_2)
    F1 = fractional_correction_of_dictionary(fr1, fr2, F1_1, F1_2)
    E = fractional_correction_of_dictionary(fr1, fr2, E_1, E_2)
    Es = fractional_correction_of_dictionary(fr1, fr2, Es_1, Es_2)
    sun = fractional_correction_of_dictionary(fr1, fr2, sun_1, sun_2)
    mag = fractional_correction_of_dictionary(fr1, fr2, mag_1, mag_2)

    # interpolate parameters in solar activity
    F2 = solar_interpolation_of_dictionary(F2, F107)
    F1 = solar_interpolation_of_dictionary(F1, F107)
    E = solar_interpolation_of_dictionary(E, F107)
    Es = solar_interpolation_of_dictionary(Es, F107)

    # construct density
    EDP = reconstruct_density_from_parameters_1level(F2, F1, E, aalt)

    return F2, F1, E, Es, sun, mag, EDP


def read_ccir_ursi_coeff(mth, coeff_dir, output_quartiles=False):
    """Read coefficients from CCIR, URSI, and Es.

    Parameters
    ----------
    mth : int
        Month.
    coeff_dir : str
        Place where the coefficient files are.
    output_quartiles : bool
        Return an additional output, the upper and lower quartiles of the
        Bradley coefficients for Es (default=False)

    Returns
    -------
    F_fof2_CCIR : array-like
        CCIR coefficients for F2 frequency.
    F_fof2_URSI : array-like
        URSI coefficients for F2 frequency.
    F_M3000 : array-like
        CCIR coefficients for M3000.
    F_Es_median : array-like
        Bradley coefficients for Es.
    F_Es_low : array-like
        Optional output only included if `output_quartiles` is True.
    F_Es_upper : array-like
        Optional output only included if `output_quartiles` is True.

    Notes
    -----
    This function sets the combination of sin and cos functions that define
    the diurnal distribution of the parameters and that can be further
    multiplied by the coefficients U_jk (from CCIR, URSI, and Es maps).
    The desired equation can be found in the Technical Note Advances in
    Ionospheric Mapping by Numerical Methods, Jones & Graham 1966. Equation
    (c) page 38.
    Acknowledgement for Es coefficients:
    Mrs. Estelle D. Powell and Mrs. Gladys I. Waggoner in supervising the
    collection, keypunching and processing of the foEs data.
    This work was sponsored by U.S. Navy as part of the SS-267 program.
    The final development work and production of the foEs maps was supported
    by the U.S Information Agency.
    Acknowledgments to Doug Drob (NRL) for giving me these coefficients.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    Jones, W. B., Graham, R. P., & Leftin, M. (1966). Advances
    in ionospheric mapping 476 by numerical methods.

    Bradley, P. A. (2003). Ingesting a sporadic-e model to iri.
    Adv. Space Res., 31(3), 577-588.

    """
    # pull the predefined sizes of the function extensions
    coef = highest_power_of_extension()

    # check that month goes from 1 to 12
    if (mth < 1) | (mth > 12):
        logger.error("Error: month is out of 1-12 range")

    # add 10 to the month because the file numeration goes from 11 to 22.
    cm = str(mth + 10)

    # F region coefficients:
    file_F = open(os.path.join(coeff_dir, 'CCIR', 'ccir' + cm + '.asc'),
                  mode='r')
    fmt = FortranRecordReader('(1X,4E15.8)')
    full_array = []
    for line in file_F:
        line_vals = fmt.read(line)
        full_array = np.concatenate((full_array, line_vals), axis=None)
    array0_F_CCIR = full_array[0:-2]
    file_F.close()

    # F region URSI coefficients:
    file_F = open(os.path.join(coeff_dir, 'URSI', 'ursi' + cm + '.asc'),
                  mode='r')
    fmt = FortranRecordReader('(1X,4E15.8)')
    full_array = []
    for line in file_F:
        line_vals = fmt.read(line)
        full_array = np.concatenate((full_array, line_vals), axis=None)
    array0_F_URSI = full_array
    file_F.close()

    # Sporadic E coefficients:
    file_E = open(os.path.join(coeff_dir, 'Es', 'Es' + cm + '.asc'),
                  mode='r')
    array0_E = np.fromfile(file_E, sep=' ')
    file_E.close()

    # for FoF2 CCIR: reshape array to [nj, nk, 2] shape
    F = np.zeros((coef['nj']['F0F2'], coef['nk']['F0F2'], 2))
    array1 = array0_F_CCIR[0:F.size]
    F_fof2_2_CCIR = np.reshape(array1, F.shape, order='F')

    # for FoF2 URSI: reshape array to [nj, nk, 2] shape
    F = np.zeros((coef['nj']['F0F2'], coef['nk']['F0F2'], 2))
    array1 = array0_F_URSI[0:F.size]
    F_fof2_2_URSI = np.reshape(array1, F.shape, order='F')

    # for M3000: reshape array to [nj, nk, 2] shape
    F = np.zeros((coef['nj']['M3000'], coef['nk']['M3000'], 2))
    array2 = array0_F_CCIR[(F_fof2_2_CCIR.size)::]
    F_M3000_2 = np.reshape(array2, F.shape, order='F')

    # for Es:
    # these number are the exact format for the files, even though some
    # numbers will be zeroes.
    nk = 76
    H = 17
    ns = 6

    F = np.zeros((H, nk, ns))
    # Each file starts with 60 indexes, we will not need them, therefore
    # skip it
    skip_coeff = np.zeros((6, 10))
    array1 = array0_E[skip_coeff.size:(skip_coeff.size + F.size)]
    F_E = np.reshape(array1, F.shape, order='F')

    F_fof2_CCIR = F_fof2_2_CCIR
    F_fof2_URSI = F_fof2_2_URSI
    F_M3000 = F_M3000_2
    F_Es_median = F_E[:, :, 2:4]

    # Trim E-region arrays to the exact shape of the functions
    F_Es_median = F_Es_median[0:coef['nj']['Es_median'],
                              0:coef['nk']['Es_median'], :]

    if output_quartiles:
        F_Es_upper = F_E[:, :, 0:2]
        F_Es_low = F_E[:, :, 4:6]

        # Trim E-region arrays to the exact shape of the functions
        F_Es_upper = F_Es_upper[0:coef['nj']['Es_upper'],
                                0:coef['nk']['Es_upper'], :]
        F_Es_low = F_Es_low[0:coef['nj']['Es_lower'],
                            0:coef['nk']['Es_lower'], :]

        # Define the output
        output = (F_fof2_CCIR, F_fof2_URSI, F_M3000, F_Es_median, F_Es_low,
                  F_Es_upper)
    else:
        # Define the output
        output = (F_fof2_CCIR, F_fof2_URSI, F_M3000, F_Es_median)

    return output


def set_diurnal_functions(nj, time_array):
    """Calculate diurnal Fourier function components.

    Parameters
    ----------
    nj : array-like
        The highest order of diurnal variation.
    time_array : array-like
        Array of UTs in hours.

    Returns
    -------
    D : array-like
        Diurnal functions.

    Notes
    -----
    This function sets the combination of sin and cos functions that define
    the diurnal distribution of the parameters and that can be further
    multiplied by the coefficients U_jk (from CCIR, URSI, and Es maps).
    The desired equation can be found in the Technical Note Advances in
    Ionospheric Mapping by Numerical Methods, Jones & Graham 1966. Equation
    (c) page 38.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    Jones, W. B., Graham, R. P., & Leftin, M. (1966). Advances
    in ionospheric mapping 476 by numerical methods.

    """
    # check that time array goes from 0-24:
    if (np.min(time_array) < 0) | (np.max(time_array) > 24):
        flag = 'Error: in set_diurnal_functions time array min < 0 or max > 24'
        logger.error(flag)

    D = np.zeros((nj, time_array.size))

    # convert time array to -pi/2 to pi/2
    hou = np.deg2rad((15.0E0 * time_array - 180.0E0))

    # construct the array of evaluated functions that can be further multiplied
    # by the coefficients
    ii = 0
    for j in range(0, nj):
        for i in range(0, 2):
            if np.mod(i, 2) == 0:
                f = np.cos(hou * (i + j))
            else:
                f = np.sin(hou * (i + j))
            if ii < nj:
                D[ii, :] = f
                ii = ii + 1
            else:
                break

    return D


def diurnal_functions(time_array):
    """Set diurnal functions for F2, M3000, and Es.

    Parameters
    ----------
    time_array : array-like
        Array of UTs in hours.

    Returns
    -------
    D_f0f2 : array-like
        Diurnal functions for foF2.
    D_M3000 : array-like
        Diurnal functions for M3000.
    D_Es_median : array-like
        Diurnal functions for Es.

    Notes
    -----
    This function calculates diurnal functions for F0F2, M3000, and Es
    coefficients

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    Jones, W. B., Graham, R. P., & Leftin, M. (1966). Advances
    in ionospheric mapping 476 by numerical methods.

    """
    # nj is the highest order of the expansion
    coef = highest_power_of_extension()

    D_f0f2 = set_diurnal_functions(coef['nj']['F0F2'], time_array)
    D_M3000 = set_diurnal_functions(coef['nj']['M3000'], time_array)
    D_Es_median = set_diurnal_functions(coef['nj']['Es_median'], time_array)

    # transpose array so that multiplication can be done later
    D_f0f2 = np.transpose(D_f0f2)
    D_M3000 = np.transpose(D_M3000)
    D_Es_median = np.transpose(D_Es_median)

    return D_f0f2, D_M3000, D_Es_median


def set_global_functions(Q, nk, alon, alat, modip):
    """Set global functions.

    Parameters
    ----------
    Q : array-like
        Vector of highest order of sin(x).
    nk : array-like
        Highest order of geographic extension, or how many functions are there
        (e.g., there are 76 functions in Table 3 on page 18 of Jones & Graham
        1966).
    alon : array-like
        Flattened array of geographic longitudes in degrees.
    alat : array-like
        Flattened array of geographic latitudes in degrees.
    modip : array-like
        Modified dip angle in degrees.

    Returns
    -------
    Gk : array-like
        Global functions

    Notes
    -----
    This function sets Geographic Coordinate Functions G_k(position) page
    # 18 of Jones & Graham 1965

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    Jones, W. B., Graham, R. P., & Leftin, M. (1966). Advances
    in ionospheric mapping 476 by numerical methods.

    """
    Gk = np.zeros((nk, alon.size))
    k = 0
    for j in range(0, len(Q)):
        for i in range(0, Q[j]):
            for m in range(0, 2):
                if m == 0:
                    fun3_st = 'cos(lon*' + str(j) + ')'
                    fun3 = np.cos(np.deg2rad(alon * j))
                else:
                    fun3_st = 'sin(lon*' + str(j) + ')'
                    fun3 = np.sin(np.deg2rad(alon * j))
                if (j != 0) | (fun3_st != 'sin(lon*' + str(j) + ')'):
                    fun1 = (np.sin(np.deg2rad(modip)))**i
                    fun2 = (np.cos(np.deg2rad(alat)))**j

                    Gk[k, :] = fun1 * fun2 * fun3
                    k = k + 1

    return Gk


def set_gl_G(alon, alat, modip):
    """Calculate global functions.

    Parameters
    ----------
    alon : array-like
        Flattened array of geographic longitudes in degrees.
    alat : array-like
        Flattened array of geographic latitudes in degrees.
    modip : array-like
        Modified dip angle in degrees.

    Returns
    -------
    G_fof2 : array-like
        Global functions for F2 region.
    G_M3000 : array-like
        Global functions for M3000 propagation parameter.
    G_Es_median : array-like
        Global functions for Es region.

    Notes
    -----
    This function sets Geographic Coordinate Functions G_k(position) page
    # 18 of Jones & Graham 1965 for F0F2, M3000, and Es coefficients

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    Jones, W. B., & Gallet, R. M. (1965). Representation of diurnal
    and geographic variations of ionospheric data by numerical methods,
    control of instability, ITU Telecommunication Journal , 32 (1), 18–28.

    """
    coef = highest_power_of_extension()

    G_fof2 = set_global_functions(coef['QM']['F0F2'],
                                  coef['nk']['F0F2'], alon, alat, modip)
    G_M3000 = set_global_functions(coef['QM']['M3000'],
                                   coef['nk']['M3000'], alon, alat, modip)
    G_Es_median = set_global_functions(coef['QM']['Es_median'],
                                       coef['nk']['Es_median'],
                                       alon, alat, modip)

    return G_fof2, G_M3000, G_Es_median


def gamma(D_f0f2, D_M3000, D_Es_median, G_fof2, G_M3000, G_Es_median,
          F_fof2_coeff, F_M3000_coeff, F_Es_median):
    """Calculate foF2, M3000 propagation parameter, and foEs.

    Parameters
    ----------
    D_f0f2 : array-like
        Diurnal functions for F2 region.
    D_M3000 : array-like
        Diurnal functions for M3000 propagation parameter.
    D_Es_median : array-like
        Diurnal functions for Es region.
    G_fof2 : array-like
        Global functions for F2 region.
    G_M3000 : array-like
        Global functions for M3000 propagation parameter.
    G_Es_median : array-like
        Global functions for Es region.
    F_fof2_coeff : array-like
        CCIR or URCI coefficients.
    F_M3000_coeff : array-like
        CCIR coefficients.
    F_Es_median : array-like
        Bradley Es coefficients.

    Returns
    -------
    gamma_f0f2 : array-like
        Critical frequency of F2 layer.
    gamma_M3000 : array-like
        M3000 propagation parameter.
    gamma_Es_median : array-like
        Critical frequency of Es layer.

    Notes
    -----
    This function calculates numerical maps for F0F2, M3000, and Es for 2
    levels of solar activity (min, max) using matrix multiplication

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    """
    Nd = D_f0f2.shape[0]
    Ng = G_fof2.shape[1]

    gamma_f0f2 = np.zeros((Nd, Ng, 2))
    gamma_M3000 = np.zeros((Nd, Ng, 2))
    gamma_Es_median = np.zeros((Nd, Ng, 2))

    # find numerical maps for 2 levels of solar activity
    for isol in range(0, 2):
        mult1 = np.matmul(D_f0f2, F_fof2_coeff[:, :, isol])
        gamma_f0f2[:, :, isol] = np.matmul(mult1, G_fof2)

        mult2 = np.matmul(D_M3000, F_M3000_coeff[:, :, isol])
        gamma_M3000[:, :, isol] = np.matmul(mult2, G_M3000)

        mult3 = np.matmul(D_Es_median, F_Es_median[:, :, isol])
        gamma_Es_median[:, :, isol] = np.matmul(mult3, G_Es_median)

    return gamma_f0f2, gamma_M3000, gamma_Es_median


def highest_power_of_extension():
    """Provide the highest power of extension.

    Returns
    -------
    const : dict
        Dictionary that has QM, nk, and nj parameters.

    Notes
    -----
    This function sets a common set of constants that define the power of
    expansions.
    QM = array of highest power of sin(x).
    nk = highest order of geographic extension.
    e.g. there are 76 functions in Table 3 on page 18 in Jones & Graham 1965.
    nj = highest order in diurnal variation.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    Jones, W. B., & Gallet, R. M. (1965). Representation of diurnal
    and geographic variations of ionospheric data by numerical methods,
    control of instability, ITU Telecommunication Journal , 32 (1), 18–28.

    """
    # degree of extension
    QM_F0F2 = [12, 12, 9, 5, 2, 1, 1, 1, 1]
    QM_M3000 = [7, 8, 6, 3, 2, 1, 1]
    QM_Es_upper = [11, 12, 6, 3, 1]
    QM_Es_median = [11, 13, 7, 3, 1, 1]
    QM_Es_lower = [11, 13, 7, 1, 1]
    QM = {'F0F2': QM_F0F2, 'M3000': QM_M3000,
          'Es_upper': QM_Es_upper, 'Es_median': QM_Es_median,
          'Es_lower': QM_Es_lower}

    # geographic
    nk_F0F2 = 76
    nk_M3000 = 49
    nk_Es_upper = 55
    nk_Es_median = 61
    nk_Es_lower = 55
    nk = {'F0F2': nk_F0F2, 'M3000': nk_M3000,
          'Es_upper': nk_Es_upper, 'Es_median': nk_Es_median,
          'Es_lower': nk_Es_lower}

    # diurnal
    nj_F0F2 = 13
    nj_M3000 = 9
    nj_Es_upper = 5
    nj_Es_median = 7
    nj_Es_lower = 5

    nj = {'F0F2': nj_F0F2, 'M3000': nj_M3000,
          'Es_upper': nj_Es_upper, 'Es_median': nj_Es_median,
          'Es_lower': nj_Es_lower}

    const = {'QM': QM, 'nk': nk, 'nj': nj}

    return const


def juldat(times):
    """Calculate the Julian time given calendar date and time.

    Parameters
    ----------
    times : class:`dt.datetime`
        Julian time in days.

    Returns
    -------
    julian_datetime : float
        Julian date.

    Raises
    ------
    ValueError
        For input of the wrong type.

    Notes
    -----
    This function calculates the Julian time given calendar date and time
    in np.datetime64 array. This function is consistent with NOVAS, The
    US Navy Observatory Astronomy Software Library and algorithm by Fliegel
    and Van Flander.

    """
    if isinstance(times, dt.datetime):
        Y = times.year
        M = times.month
        D = times.day
        h = times.hour
        m = times.minute
        s = times.second

        julian_datetime = (367. * Y
                           - int((7. * (Y + int((M + 9.) / 12.))) / 4.)
                           + int((275. * M) / 9.) + D + 1721013.5
                           + (h + m / 60. + s / math.pow(60, 2)) / 24.
                           - 0.5 * math.copysign(1, 100. * Y + M - 190002.5)
                           + 0.5)

    else:
        raise ValueError("input must be datetime.datetime")

    return julian_datetime


def subsolar_point(juliantime):
    """Find location of subsolar point.

    Parameters
    ----------
    juliantime : float
        Julian time in days.

    Returns
    -------
    lonsun : float
        Longitude of the sun in degrees.
    latsun : float
        Latitude of the sun in degrees.

    Notes
    -----
    This function returns the lon and lat of subsolar point for a given
    Juliantime Latitude of subsolar point is same as solar declination angle.

    """
    # number of centuries from J2000
    t = (juliantime - 2451545.) / 36525.

    # mean longitude corrected for aberration (deg)
    lms = np.mod(280.460 + 36000.771 * t, 360.)

    # mean anomaly (deg)
    solanom = np.mod(357.527723 + 35999.05034 * t, 360.)

    # ecliptic longitude (deg)
    ecplon = (lms + 1.914666471 * np.sin(np.deg2rad(solanom))
              + 0.01999464 * np.sin(np.deg2rad(2. * solanom)))

    # obliquity of the ecliptic (deg)
    epsilon = (23.439291 - 0.0130042 * t)

    # solar declination in radians
    decsun = np.rad2deg(np.arcsin(np.sin(np.deg2rad(epsilon))
                                  * np.sin(np.deg2rad(ecplon))))

    # greenwich mean sidereal time in seconds
    tgmst = (67310.54841
             + (876600. * 3600. + 8640184.812866) * t
             + 0.93104 * t**2
             - 6.2e-6 * t**3)

    # greenwich mean sidereal time in degrees
    tgmst = np.mod(tgmst, 86400.) / 240.

    # right ascension
    num = (np.cos(np.deg2rad(epsilon))
           * np.sin(np.deg2rad(ecplon))
           / np.cos(np.deg2rad(decsun)))

    den = np.cos(np.deg2rad(ecplon)) / np.cos(np.deg2rad(decsun))
    sunra = np.rad2deg(np.arctan2(num, den))

    # longitude of the sun in degrees
    lonsun = -(tgmst - sunra)

    # lonsun=adjust_longitude(lonsun, 'to180')
    lonsun = adjust_longitude(lonsun, 'to180')

    return lonsun, decsun


def solar_zenith(lon_sun, lat_sun, lon_observer, lat_observer):
    """Calculate solar zenith angle from known location of the sun.

    Parameters
    ----------
    lon_sun : array-like
        Longitude of the sun in degrees.
    lat_sun : array-like
        Latitude of the sun in degrees.
    lon_observer : array-like
        Longitude of the observer in degrees.
    lat_observer : array-like
        Latitude of the observer in degrees.

    Returns
    -------
    azenith : array-like
        Solar zenith angle.

    Raises
    ------
    ValueError
        If the solar longitude and latitude inputs aren't the same size

    Notes
    -----
    This function takes lon and lat of the subsolar point and lon and lat
    of the observer, and calculates solar zenith angle.

    """
    if (isinstance(lon_sun, int)) or (isinstance(lon_sun, float)):
        # cosine of solar zenith angle
        cos_zenith = (np.sin(np.deg2rad(lat_sun))
                      * np.sin(np.deg2rad(lat_observer))
                      + np.cos(np.deg2rad(lat_sun))
                      * np.cos(np.deg2rad(lat_observer))
                      * np.cos(np.deg2rad((lon_sun - lon_observer))))

        # solar zenith angle
        azenith = np.rad2deg(np.arccos(cos_zenith))

    if isinstance(lon_sun, np.ndarray):
        if lon_sun.size != lat_sun.size:
            raise ValueError('`lon_sun` and `lat_sun` lengths are not equal')

        # make array to hold zenith angles based of size of lon_sun and
        # type of lon_observer
        if (isinstance(lon_observer, int)) or (isinstance(lon_observer, float)):
            azenith = np.zeros((lon_sun.size, 1))
        if isinstance(lon_observer, np.ndarray):
            azenith = np.zeros((lon_sun.size, lon_observer.size))

        for i in range(0, lon_sun.size):
            # cosine of solar zenith angle
            cos_zenith = (np.sin(np.deg2rad(lat_sun[i]))
                          * np.sin(np.deg2rad(lat_observer))
                          + np.cos(np.deg2rad(lat_sun[i]))
                          * np.cos(np.deg2rad(lat_observer))
                          * np.cos(np.deg2rad((lon_sun[i] - lon_observer))))

            # solar zenith angle
            azenith[i, :] = np.rad2deg(np.arccos(cos_zenith))

    return azenith


def solzen_timearray_grid(year, mth, day, T0, alon, alat):
    """Calculate solar zenith angle.

    Parameters
    ----------
    year : int
        Year.
    mth : int
        Month.
    day : int
        Day.
    T0 : array-like
        Array of UTs in hours.
    alon : array-like
        Flattened array of longitudes in degrees.
    alat : array-like
        Flattened array of latitudes in degrees.

    Returns
    -------
    solzen : array-like
        Solar zenith angle.
    aslon : array-like
        Longitude of subsolar point in degrees.
    aslat : array-like
        Latitude of subsolar point in degrees.

    Raises
    ------
    ValueError
        If the input arrays are not the same shape

    Notes
    -----
    This function returns solar zenith angle for the given year, month,
    day, array of UT, and arrays of lon and lat of the grid.

    """
    # check size of the grid arrays
    if alon.size != alat.size:
        raise ValueError('`alon` and `alat` sizes are  not the same')

    aslon = np.zeros((T0.size))
    aslat = np.zeros((T0.size))
    for i in range(0, T0.size):
        UT = T0[i]
        hour = np.fix(UT)
        minute = np.fix((UT - hour) * 60.)
        date_sd = dt.datetime(int(year), int(mth), int(day),
                              int(hour), int(minute))
        jday = juldat(date_sd)
        slon, slat = subsolar_point(jday)
        aslon[i] = slon
        aslat[i] = slat

    solzen = solar_zenith(aslon, aslat, alon, alat)

    return solzen, aslon, aslat


def solzen_effective(chi):
    """Calculate effective solar zenith angle.

    Parameters
    ----------
    chi : array-like
        Solar zenith angle (deg).

    Returns
    -------
    chi_eff : array-like
        Effective solar zenith angle (deg).

    Notes
    -----
    This function calculates effective solar zenith angle as a function of
    solar zenith angle and solar zenith angle at day-night transition,
    according to the Titheridge model. f2 is used during daytime, and f1
    during nighttime, and the exponential day-night transition is used to
    ensure the continuity of foE and its first derivative at solar terminator
    [Nava et al, 2008 "A new version of the NeQuick ionosphere electron
    density model"]

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    Nava et al. (2008). A new version of the NeQuick ionosphere
    electron density model. J. Atmos. Sol. Terr. Phys., 70 (15),
    490 doi: 10.1016/j.jastp.2008.01.015

    """
    # solar zenith angle at day-night transition (deg), number
    chi0 = 86.23292796211615E0
    alpha = 12.0

    x = chi - chi0
    f1 = 90.0 - 0.24 * fexp(20. - 0.2 * chi)
    f2 = chi
    ee = fexp(alpha * x)
    chi_eff = (f1 * ee + f2) / (ee + 1.0)

    # replace nan with 0
    chi_eff = np.nan_to_num(chi_eff)

    return chi_eff


def foE(mth, solzen_effective, alat, f107):
    """Calculate critical frequency of E region.

    Parameters
    ----------
    mth : int
        Month.
    solzen_effective : array-like
        Effective solar zenith angle with shape [ntime].
    alat : array-like
        Flattened array of latitudes in degrees with shape [ngrid].
    f107 : float
        F10.7 solar flux in SFU.

    Returns
    -------
    foE : array-like
        Critical frequency of E region in MHz with shape [ntime, ngrid].

    Notes
    -----
    This function caclulates foE for a given effective solar zenith angle
    and level of solar activity. This routine is based on the Ionospheric
    Correction Algorithm for Galileo Single Frequency Users that describes
    the NeQuick Model.

    Result is:
    foE = critical frequency of the E region (MHz), np.array,

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    Nava et al. (2008). A new version of the NeQuick ionosphere
    electron density model. J. Atmos. Sol. Terr. Phys., 70 (15),
    490 doi: 10.1016/j.jastp.2008.01.015

    """
    # define the seas parameter as a function of the month of the year as
    # follows
    if (mth == 1) or (mth == 2) or (mth == 11) or (mth == 12):
        seas = -1
    if (mth == 3) or (mth == 4) or (mth == 9) or (mth == 10):
        seas = 0
    if (mth == 5) or (mth == 6) or (mth == 7) or (mth == 8):
        seas = 1

    # introduce the latitudinal dependence
    ee = fexp(np.deg2rad(0.3 * alat))

    # combine seasonal and latitudinal dependence
    seasp = seas * (ee - 1.0) / (ee + 1.0)

    # critical frequency
    foE = np.sqrt(0.49 + (1.112 - 0.019 * seasp)**2 * np.sqrt(f107)
                  * (np.cos(np.deg2rad(solzen_effective)))**0.6)

    # turn nans into zeros
    foE = np.nan_to_num(foE)

    return foE


def gammaE(year, mth, utime, alon, alat, aIG):
    """Calculate numerical maps for critical frequency of E region.

    Parameters
    ----------
    year : int
        Year.
    mth : int
        Month.
    utime : array-like
        Array of UTs in hours.
    alon : array-like
        Flattened array of longitudes in degrees.
    alat : array-like
        Flattened array of latitudes in degrees.
    aIG : array-like
        Min and max of IG12 index.

    Returns
    -------
    gamma_E : array-like
        critical frequency of E region in MHz.
    slon : array-like
        Longitude of subsolar point in degrees.
    slat : array-like
        Latitude of subsolar point in degrees.

    Notes
    -----
    This function calculates numerical maps for FoE for 2 levels of solar
    activity.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    """
    # solar zenith angle for day 15 in the month of interest
    solzen, slon, slat = solzen_timearray_grid(year, mth, 15, utime, alon, alat)

    # solar zenith angle for day 15 in the month of interest at noon
    solzen_noon, slon_noon, slat_noon = solzen_timearray_grid(year, mth, 15,
                                                              utime * 0 + 12.,
                                                              alon, alat)
    # effective solar zenith angle
    solzen_eff = solzen_effective(solzen)

    # make arrays to hold numerical maps for 2 levels of solar activity
    gamma_E = np.zeros((utime.size, alon.size, 2))

    # min and max of solar activity
    aF107_min_max = np.array([IG12_2_F107(aIG[0]), IG12_2_F107(aIG[1])])

    # find numerical maps for 2 levels of solar activity
    for isol in range(0, 2):
        gamma_E[:, :, isol] = foE(mth, solzen_eff, alat, aF107_min_max[isol])

    return gamma_E, slon, slat


def Probability_F1(year, mth, utime, alon, alat, mag_dip_lat, aIG):
    """Calculate probability occurrence of F1 layer.

    Parameters
    ----------
    year : int
        Year.
    mth : int
        Month.
    time : array-like
        Array of UTs in hours.
    alon : array-like
        Flattened array of longitudes in degrees.
    alat : array-like
        Flattened array of latitudes in degrees.
    mag_dip_lat : array-like
        Flattened array of magnetic dip latitudes in degrees.
    aIG : array-like
        Min and Max of IG12.

    Returns
    -------
    a_P : array-like
        Probability occurrence of F1 layer.
    a_foF1 : array-like
        Critical frequency of F1 layer in MHz.

    Notes
    -----
    This function calculates numerical maps probability of F1 layer.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    Bilitza et al. (2022), The International Reference Ionosphere
    model: A review and description of an ionospheric benchmark, Reviews
    of Geophysics, 60.

    """
    # make arrays to hold numerical maps for 2 levels of solar activity
    a_P = np.zeros((utime.size, alon.size, 2))
    a_foF1 = np.zeros((utime.size, alon.size, 2))
    a_R12 = np.zeros((utime.size, alon.size, 2))
    a_mag_dip_lat_abs = np.zeros((utime.size, alon.size, 2))
    a_solzen = np.zeros((utime.size, alon.size, 2))

    # add 2 levels of solar activity for mag_dip_lat, by using same elements
    # empty array, swap axises before filling with same elements of modip,
    # swap back to match M3000 array shape
    a_mag_dip_lat_abs = np.swapaxes(a_mag_dip_lat_abs, 1, 2)
    a_mag_dip_lat_abs = np.full(a_mag_dip_lat_abs.shape, np.abs(mag_dip_lat))
    a_mag_dip_lat_abs = np.swapaxes(a_mag_dip_lat_abs, 1, 2)

    # simplified constant value from Bilitza review 2022
    gamma = 2.36

    # solar zenith angle for day 15 in the month of interest
    solzen, aslon, aslat = solzen_timearray_grid(year, mth, 15, utime, alon,
                                                 alat)

    # min and max of solar activity
    R12_min_max = np.array([IG12_2_R12(aIG[0]), IG12_2_R12(aIG[1])])

    for isol in range(0, 2):
        a_R12[:, :, isol] = np.full((utime.size, alon.size), R12_min_max[isol])
        a_P[:, :, isol] = (0.5 + 0.5 * np.cos(np.deg2rad(solzen)))**gamma
        a_solzen[:, :, isol] = solzen

    n = (0.093
         + 0.0046 * a_mag_dip_lat_abs
         - 0.000054 * a_mag_dip_lat_abs**2
         + 0.0003 * a_R12)

    f_100 = (5.348
             + 0.011 * a_mag_dip_lat_abs
             - 0.00023 * a_mag_dip_lat_abs**2)

    f_0 = (4.35
           + 0.0058 * a_mag_dip_lat_abs
           - 0.00012 * a_mag_dip_lat_abs**2)

    f_s = f_0 + (f_100 - f_0) * a_R12 / 100.

    arg = (np.cos(np.deg2rad(a_solzen)))
    ind = np.where(arg > 0)
    a_foF1[ind] = f_s[ind] * arg[ind]**n[ind]

    a_foF1[np.where(a_P < 0.5)] = np.nan

    return a_P, a_foF1


def fexp(x):
    """Calculate exponent without overflow.

    Parameters
    ----------
    x : array-like
        Any input.

    Returns
    -------
    y : array-like
        Exponent of x.

    Notes
    -----
    This function function calculates exp(x) with restrictions to not cause
    overflow.

    """
    if (isinstance(x, float)) or (isinstance(x, int)):
        if x > 80:
            y = 5.5406E34
        if x < -80:
            y = 1.8049E-35
        else:
            y = np.exp(x)

    if isinstance(x, np.ndarray):
        y = np.zeros((x.shape))
        a = np.where(x > 80.)
        b = np.where(x < -80.)
        c = np.where((x >= -80.) & (x <= 80.))
        y[a] = 5.5406E34
        y[b] = 1.8049E-35
        y[c] = np.exp(x[c])

    return y


def freq_to_Nm(foF2, foF1, foE, foEs):
    """Convert critical frequency to plasma density.

    Parameters
    ----------
    foF2 : array-like
        Critical frequency of F2 layer in MHz.
    foF1 : array-like
        Critical frequency of F1 layer in MHz.
    foE : array-like
        Critical frequency of E layer in MHz.
    foEs : array-like
        Critical frequency of Es layer in MHz.

    Returns
    -------
    NmF2 : array-like
       Peak density of F2 layer in m-3.
    NmF1 : array-like
        Peak density of F1 layer in m-3.
    NmE : array-like
        Peak density of E layer in m-3.
    NmEs : array-like
        Peak density of Es layer in m-3.

    Notes
    -----
    This function returns maximum density for the given critical frequency and
    limits it to 1 if it is below zero.

    """
    # F2 peak
    NmF2 = freq2den(foF2) * 1e11

    # F1 peak
    NmF1 = freq2den(foF1) * 1e11

    # E
    NmE = freq2den(foE) * 1e11

    # Es
    NmEs = freq2den(foEs) * 1e11

    # exclude negative values, in case there are any
    NmF2[np.where(NmF2 <= 0)] = 1.
    NmF1[np.where(NmF1 <= 0)] = 1.
    NmE[np.where(NmE <= 0)] = 1.
    NmEs[np.where(NmEs <= 0)] = 1.

    return NmF2, NmF1, NmE, NmEs


def hmF1_from_F2(NmF2, NmF1, hmF2, B_F2_bot):
    """Determine the height of F1 layer.

    Parameters
    ----------
    NmF2 : array-like
        Peak density of F2 layer in m-3.
    NmF1 : array-like
        Peak density of F1 layer in m-3.
    hmF2 : array-like
        Height of F2 layer in km.
    B_F2_bot : array-like
        Thickness of F2 bottom layer in km.

    Returns
    -------
    hmF1 : array-like
        Height of F1 layer in km.

    Notes
    -----
    This function calculates hmF1 from known shape of F2 bottom side, where
    it drops to NmF1.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    """
    hmF1 = NmF2 * 0 + np.nan
    a = NmF2 * 0
    b = NmF2 * 0
    c = NmF2 * 0
    d = NmF2 * 0
    x = NmF2 * 0 + np.nan

    a[:, :, :] = 1.
    ind_finite = np.isfinite(NmF1)
    b[ind_finite] = (2. - 4. * NmF2[ind_finite] / NmF1[ind_finite])
    c[:, :, :] = 1.

    d = (b**2) - (4 * a * c)
    ind_positive = np.where(d >= 0)

    # take second root of the quadratic equation (it is below hmF2)
    x[ind_positive] = ((-b[ind_positive] - np.sqrt(d[ind_positive]))
                       / (2 * a[ind_positive]))

    ind_g0 = np.where(x > 0)
    hmF1[ind_g0] = B_F2_bot[ind_g0] * np.log(x[ind_g0]) + hmF2[ind_g0]

    return hmF1


def find_B_F1_bot(hmF1, hmE, P_F1):
    """Determine the thickness of F1 layer.

    Parameters
    ----------
    hmF1 : array-like
        Height of F1 layer in km.
    hmE : array-like
        Height of E layer in km.
    P_F1 : array-like
        Probability of observing F1 layer.

    Returns
    -------
    B_F1_bot : array-like
        Thickness of F1 layer in km.

    Notes
    -----
    This function returns thickness of F1 layer in km. This is done using hmF1
    and hmE, as described in NeQuick Eq 87

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    """
    B_F1_bot = 0.5 * (hmF1 - hmE)
    B_F1_bot[np.where(P_F1 < 0.5)] = np.nan

    return B_F1_bot


def hm_IRI(M3000, foE, foF2, modip, aIG):
    """Return height of the ionospheric layers.

    Parameters
    ----------
    M3000 : array-like
        Propagation parameter for F2 region related to hmF2.
    foE : array-like
        Critical frequency of E region in MHz.
    foF2 : array-like
        Critical frequency of F2 region in MHz.
    modip : array-like
        Modified dip angle in degrees.
    aIG : array-like
        Min and max of IG12 index.

    Returns
    -------
    hmF2 : array-like
        Height of F2 layer in km.
    hmE : array-like
        Height of E layer in km.
    hmEs : array-like
        Height of Es layer in km.

    Notes
    -----
    This function returns height of ionospheric layers like it is
    done in IRI.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    Bilitza et al. (2022), The International Reference Ionosphere
    model: A review and description of an ionospheric benchmark, Reviews
    of Geophysics, 60.

    """
    s = M3000.shape
    time_dim = s[0]
    grid_dim = s[1]

    R12_min_max = np.array([IG12_2_R12(aIG[0]), IG12_2_R12(aIG[1])])

    # E
    hmE = 110. + np.zeros((M3000.shape))

    # F2
    # based on BSE-1979 IRI Option developed by Bilitza et al. (1979)
    # see 3.3.1 of IRI 2020 Review by Bilitza et al. 2022
    # add 2 levels of solar activity for modip, by using same elements
    # empty array, swap axises before filling with same elements of modip,
    # swap back to match M3000 array shape
    modip_2levels = M3000 * 0.
    modip_2levels = np.swapaxes(modip_2levels, 1, 2)
    modip_2levels = np.full(modip_2levels.shape, modip)
    modip_2levels = np.swapaxes(modip_2levels, 1, 2)

    # make R12 same as input arrays to account for 2 levels of solar activity
    aR12 = M3000 * 0.
    for isol in range(0, 2):
        aR12[:, :, isol] = np.full((time_dim, grid_dim), R12_min_max[isol])

    # empty array
    hmF2 = np.zeros((M3000.shape))
    ratio = foF2 / foE
    f1 = 0.00232 * aR12 + 0.222
    f2 = 1.0 - aR12 / 150.0 * np.exp(-(modip_2levels / 40.)**2)
    f3 = 1.2 - 0.0116 * np.exp(aR12 / 41.84)
    f4 = 0.096 * (aR12 - 25.) / 150.

    a = np.where(ratio < 1.7)
    ratio[a] = 1.7

    DM = f1 * f2 / (ratio - f3) + f4
    hmF2 = 1490.0 / (M3000 + DM) - 176.0

    # Es
    hmEs = 100. + np.zeros((M3000.shape))

    return hmF2, hmE, hmEs


def thickness(foF2, M3000, hmF2, hmE, mth, aIG):
    """Return thicknesses of ionospheric layers.

    Parameters
    ----------
    foF2 : array-like
        Critical frequency of F2 region in MHz.
    M3000 : array-like
        Propagation parameter for F2 region related to hmF2.
    hmF2 : array-like
        Height of the F2 layer.
    hmE : array-like
        Height of the E layer.
    mth : int
        Month of the year.
    aIG : array-like
        Min and max of IG12 index.

    Returns
    -------
    B_F2_bot : array-like
        Thickness of F2 bottom in km.
    B_F2_top : array-like
        Thickness of F2 top in km.
    B_E_bot : array-like
        Thickness of E bottom in km.
    B_E_top : array-like
        Thickness of E top in km.
    B_Es_bot : array-like
        Thickness of Es bottom in km.
    B_Es_top : array-like
        Thickness of Es top in km.

    Notes
    -----
    This function returns thicknesses of ionospheric layers.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    """
    # B_F2_bot..................................................................
    NmF2 = freq2den(foF2)

    # In the actual NeQuick_2 code there is a typo, missing 0.01 which makes
    # the B 100 times smaller. It took me a long time to find this mistake,
    # while comparing with my results. The printed guide doesn't have this
    # typo.
    dNdHmx = -3.467 + 1.714 * np.log(foF2) + 2.02 * np.log(M3000)
    dNdHmx = 0.01 * fexp(dNdHmx)
    B_F2_bot = 0.385 * NmF2 / dNdHmx

    # B_F2_top..................................................................
    # set empty arrays
    k = foF2 * 0.

    # shape parameter depends on solar activity:
    for isol in range(0, 2):
        # Effective sunspot number
        R12 = IG12_2_R12(aIG[isol])
        k[:, :, isol] = (3.22
                         - 0.0538 * foF2[:, :, isol]
                         - 0.00664 * hmF2[:, :, isol]
                         + (0.113 * hmF2[:, :, isol] / B_F2_bot[:, :, isol])
                         + 0.00257 * R12)

    # auxiliary parameters x and v:
    x = (k * B_F2_bot - 150.) / 100.
    # thickness
    B_F2_top = (100. * x + 150.) / (0.041163 * x**2 - 0.183981 * x + 1.424472)

    # B_E_top..................................................................
    B_E_top = 7. + np.zeros((NmF2.shape))

    # B_E_bot..................................................................
    B_E_bot = 5. + np.zeros((NmF2.shape))

    # B_Es_top.................................................................
    B_Es_top = 1. + np.zeros((NmF2.shape))

    # B_Es_bot.................................................................
    B_Es_bot = 1. + np.zeros((NmF2.shape))

    return B_F2_bot, B_F2_top, B_E_bot, B_E_top, B_Es_bot, B_Es_top


def epstein(Nm, hm, B, alt):
    """Calculate Epstein function for given parameters.

    Parameters
    ----------
    Nm : array-like
        Peak density in m-3.
    hm : array-like
        Height of peak density in km.
    B : array-like
        Thickness of the layer in km.
    alt : array-like
        Altitude array in km.

    Returns
    -------
    res : array-like
        Constructed Epstein profile in m-3.

    Notes
    -----
    This function returns Epstein function for given parameters.
    In a typical Epstein function: X = Nm, Y = hm, Z = B, and W = alt.

    """
    aexp = fexp((alt - hm) / B)
    res = Nm * aexp / (1 + aexp)**2

    return res


def decimal_year(dtime):
    """Determine the decimal year.

    Parameters
    ----------
    dtime : class:`dt.datetime`
        Given datetime.

    Returns
    -------
    date_decimal : float
        Decimal year.

    Notes
    -----
    This function returns decimal year. For example, middle of the year
    is 2020.5.

    """
    # day of the year
    doy = dtime.timetuple().tm_yday

    # decimal, day of year devided by number of days in year
    days_of_year = int(dt.datetime(dtime.year, 12, 31).strftime('%j'))
    decimal = (doy - 1) / days_of_year

    # year plus decimal
    date_decimal = dtime.year + decimal

    return date_decimal


def set_geo_grid(dlon, dlat):
    """Set geographical grid for given horizontal resolution.

    Parameters
    ----------
    dlon : float
        Longitudinal step size in degrees.
    dlat : float
        Latitudinal step size in degrees.

    Returns
    -------
    alon : array-like
        Flattened coordinates of longitudes in degrees.
    alat : array-like
        Flattened coordinates of latitudes in degrees.
    alon_2d : array-like
        Reshaped 2-D array of longitudes in degrees.
    alat_2d : array-like
        Reshaped 2-D array of latitudes in degrees.

    Notes
    -----
    This function makes global grid arrays for given resolution.

    """
    alon_2d, alat_2d = np.mgrid[-180:180 + dlon:dlon, -90:90 + dlat:dlat]
    alon = np.reshape(alon_2d, alon_2d.size)
    alat = np.reshape(alat_2d, alat_2d.size)

    return alon, alat, alon_2d, alat_2d


def set_alt_grid(dalt):
    """Set an altitude array with given vertical resolution.

    Parameters
    ----------
    dalt : float
        Vertical step in km.

    Returns
    -------
    aalt : array-like
        Altitude array in km.

    Notes
    -----
    This function makes altitude array from 90 to 1000 km for given resolution.

    """
    aalt = np.mgrid[90:1000 + dalt:dalt]

    return aalt


def set_temporal_array(dUT):
    """Set a time array with given time step.

    Parameters
    ----------
    dUT : float
        Time step in hours.

    Returns
    -------
    aUT : array-like
        Universal time array in hours.
    ahour : array-like
        int array of hours.
    aminute : array-like
        int array of minutes.
    asecond : array-like
        int array of seconds.
    atime_frame_strings : array-like
        String array of time stamps HHMM.

    Notes
    -----
    This function converts ionospheric frequency to plasma density.

    """
    aUT = np.arange(0, 24, dUT)
    ahour = np.fix(aUT).astype(int)
    aminute = ((aUT - ahour) * 60.).astype(int)
    asecond = (aUT * 0).astype(int)
    atime_frame_strings = [str(ahour[it]).zfill(2) + str(aminute[it]).zfill(2)
                           for it in range(0, aUT.size)]

    return aUT, ahour, aminute, asecond, atime_frame_strings


def freq2den(freq):
    """Convert ionospheric frequency to plasma density.

    Parameters
    ----------
    freq : array-like
        ionospheric frequency in MHz.

    Returns
    -------
    dens : array-like
        plasma density in m-3.

    Notes
    -----
    This function converts ionospheric frequency to plasma density.

    """
    dens = 0.124 * freq**2

    return dens


def R12_2_F107(R12):
    """Convert R12 to F10.7 coefficients.

    Parameters
    ----------
    R12 : float or array-like
        12-month sunspot number.

    Returns
    -------
    F107 : float or array-like
        Solar flux at 10.7 in SFU.

    Notes
    -----
    This function converts R12 to F10.7.

    """
    F107 = 63.7 + 0.728 * R12 + 8.9E-4 * R12**2

    return F107


def F107_2_R12(F107):
    """Convert F10.7 to R12 coefficients.

    Parameters
    ----------
    F107 : float or array-like
        Solar flux at 10.7 in SFU.

    Returns
    -------
    R12 : float or array-like
        12-month sunspot number.

    Notes
    -----
    This function converts F10.7 to R12.

    """
    a = 8.9E-4
    b = 0.728
    c = 63.7 - F107
    x = quadratic([a, b, c])[0]

    return x


def R12_2_IG12(R12):
    """Convert R12 to IG12 coefficients.

    Parameters
    ----------
    R12 : float or array-like
        Sunspot number coefficient R12.

    Returns
    -------
    IG12 : float or array-like
        Ionosonde Global Coefficient.

    Notes
    -----
    This function converts R12 to IG12.

    """
    IG12 = 12.349 + 1.468 * R12 - 0.00268 * R12**2

    return IG12


def IG12_2_R12(IG12):
    """Convert IG12 to R12 coefficients.

    Parameters
    ----------
    IG12 : float or array-like
        Ionosonde Global coefficient.

    Returns
    -------
    R12 : float or array-like
        Sunspot number coefficient R12.

    Notes
    -----
    This function converts IG12 to R12.

    References
    ----------
    Bilitza et al. (2022), The International Reference Ionosphere
    model: A review and description of an ionospheric benchmark, Reviews
    of Geophysics, 60.

    """
    a = -0.00268
    b = 1.468
    c = 12.349 - IG12

    x = quadratic([a, b, c])[0]
    return x


def F107_2_IG12(F107):
    """Convert F10.7 to IG12 coefficients.

    Parameters
    ----------
    F107 : float or array-like
        Solar flux F10.7 coefficient in SFU.

    Returns
    -------
    IG12 : float or array-like
        Ionosonde Global coefficient.

    Notes
    -----
    This function converts F10.7 to IG12.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    Bilitza et al. (2022), The International Reference Ionosphere
    model: A review and description of an ionospheric benchmark, Reviews
    of Geophysics, 60.

    """
    R12 = F107_2_R12(F107)
    IG12 = R12_2_IG12(R12)

    return IG12


def IG12_2_F107(IG12):
    """Convert IG12 to F10.7 coefficients.

    Parameters
    ----------
    IG12 : float or array-like
        Ionosonde Global coefficient.

    Returns
    -------
    F107 : float or array-like
        Solar flux F10.7 coefficient in SFU.

    Notes
    -----
    This function converts IG12 to F10.7.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    Bilitza et al. (2022), The International Reference Ionosphere
    model: A review and description of an ionospheric benchmark, Reviews
    of Geophysics, 60.

    """
    R12 = IG12_2_R12(IG12)
    F107 = R12_2_F107(R12)

    return F107


def quadratic(coeff):
    """Solve quadratic equation for given coefficients a, b, c.

    Parameters
    ----------
    coeff : array-like
        Array with a, b, and c coefficients as the first three elements,
        which may be floats or arrays.

    Returns
    -------
    [root1, root2] : list
        List of two root solutions. The first solution, uses addition and the
        second solution uses subtraction.

    Notes
    -----
    This function solves quadratic equation and returns 2 roots.

    """
    a = coeff[0]
    b = coeff[1]
    c = coeff[2]

    d = (b**2) - (4. * a * c)

    # find two solutions
    root1 = (-b + np.sqrt(d)) / (2. * a)
    root2 = (-b - np.sqrt(d)) / (2. * a)

    return [root1, root2]


def epstein_function_array(A1, hm, B, x):
    """Construct density Epstein profile for any layer (except topside of F2).

    Parameters
    ----------
    A1 : array-like
        Amplitude of layer in m-3.
    hm : array-like
        Height of layer in km.
    B : array-like
        Thickness in km.
    x : array-like
        Altitude in km.

    Returns
    -------
    density : array-like
        Constructed density in m-3.

    Notes
    -----
    This function constructs density Epstein profile for any layer.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    """
    density = np.zeros((A1.shape))
    alpha = (x - hm) / B
    exp = fexp(alpha)

    a = np.where(alpha <= 25)
    b = np.where(alpha > 25)
    density[a] = A1[a] * exp[a] / (1. + exp[a])**2
    density[b] = 0.

    return density


def epstein_function_top_array(A1, hmF2, B_F2_top, x):
    """Construct density Epstein profile for the topside of F2 layer.

    Parameters
    ----------
    A1 : array-like
        Amplitude of F2 layer in m-3.
    hmF2 : array-like
        Height of F2 layer in km.
    B_F2_top : array-like
        Thickness of topside F2 layer in km.
    x : array-like
        Altitude in km.

    Returns
    -------
    density : array-like
        Constructed density in m-3.

    Notes
    -----
    This function constructs density Epstein profile for the topside of F2
    layer.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    """
    g = 0.125
    r = 100.
    dh = x - hmF2
    z = dh / (B_F2_top * (1 + (r * g * dh / (r * B_F2_top + g * dh))))
    exp = fexp(z)

    density = np.zeros((exp.size))
    density[exp > 1e11] = A1[exp > 1e11] / exp[exp > 1e11]
    density[exp <= 1e11] = (A1[exp <= 1e11]
                            * exp[exp <= 1e11]
                            / (exp[exp <= 1e11] + 1)**2)
    return density


def drop_function(x):
    """Calculate drop function from a simple family of curve.

    Parameters
    ----------
    x : array-like
        Portion of the altitude array.

    Returns
    -------
    y : array-like
        Function that can be multiplied with x to reduce the influence of x.

    Notes
    -----
    This is a drop function from a simple family of curve. It is used to
    reduce the F1_top contribution for the F2_bot region, so that when the
    summation of Epstein functions is performed, the presence of F1 region
    would not mess up with the value of NmF2.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    """
    # degree of the drop (4 give a good result)
    n = 4
    nelem = x.size

    if nelem > 1:
        y = 1. - (x / (nelem - 1.))**n
    else:
        y = np.zeros((x.size)) + 1.

    return y


def reconstruct_density_from_parameters(F2, F1, E, alt):
    """Construct vertical EDP for 2 levels of solar activity.

    Parameters
    ----------
    F2 : dict
        Dictionary of parameters for F2 layer.
    F1 : dict
        Dictionary of parameters for F1 layer.
    E : dict
        Dictionary of parameters for E layer.
    alt : array-like
        1-D array of altitudes [N_V] in km.

    Returns
    -------
    x_out : array-like
        Electron density for two levels of solar activity [2, N_T, N_V, N_G]
        in m-3.

    Notes
    -----
    This function calculates 3-D density from given dictionaries of
    the parameters for 2 levels of solar activity.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    """
    s = F2['Nm'].shape
    N_G = s[1]
    N_T = s[0]
    N_V = alt.size

    x_out = np.full((2, N_T, N_V, N_G), np.nan)

    for isolar in range(0, 2):
        x = np.full((11, N_T, N_G), np.nan)

        x[0, :, :] = F2['Nm'][:, :, isolar]
        x[1, :, :] = F1['Nm'][:, :, isolar]
        x[2, :, :] = E['Nm'][:, :, isolar]
        x[3, :, :] = F2['hm'][:, :, isolar]
        x[4, :, :] = F1['hm'][:, :, isolar]
        x[5, :, :] = E['hm'][:, :, isolar]
        x[6, :, :] = F2['B_bot'][:, :, isolar]
        x[7, :, :] = F2['B_top'][:, :, isolar]
        x[8, :, :] = F1['B_bot'][:, :, isolar]
        x[9, :, :] = E['B_bot'][:, :, isolar]
        x[10, :, :] = E['B_top'][:, :, isolar]

        EDP = EDP_builder(x, alt)
        x_out[isolar, :, :, :] = EDP

    return x_out


def reconstruct_density_from_parameters_1level(F2, F1, E, alt):
    """Construct vertical EDP for 1 level of solar activity.

    Parameters
    ----------
    F2 : dict
        Dictionary of parameters for F2 layer.
    F1 : dict
        Dictionary of parameters for F1 layer.
    E : dict
        Dictionary of parameters for E layer.
    alt : array-like
        1-D array of altitudes [N_V] in km.

    Returns
    -------
    x_out : array-like
        Electron density for two levels of solar activity [N_T, N_V, N_G]
        in m-3.

    Notes
    -----
    This function calculates 3-D density from given dictionaries of
    the parameters for 1 level of solar activity.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    """
    s = F2['Nm'].shape

    N_T = s[0]
    N_G = s[1]

    x = np.full((11, N_T, N_G), np.nan)

    x[0, :, :] = F2['Nm'][:, :]
    x[1, :, :] = F1['Nm'][:, :]
    x[2, :, :] = E['Nm'][:, :]
    x[3, :, :] = F2['hm'][:, :]
    x[4, :, :] = F1['hm'][:, :]
    x[5, :, :] = E['hm'][:, :]
    x[6, :, :] = F2['B_bot'][:, :]
    x[7, :, :] = F2['B_top'][:, :]
    x[8, :, :] = F1['B_bot'][:, :]
    x[9, :, :] = E['B_bot'][:, :]
    x[10, :, :] = E['B_top'][:, :]

    EDP = EDP_builder(x, alt)

    return EDP


def EDP_builder(x, aalt):
    """Construct vertical EDP.

    Parameters
    ----------
    x : array-like
        Array where 1st dimention indicates the parameter (total 11
        parameters), second dimension is time, and third is horizontal grid
        [11, N_T, N_G].
    aalt : array-like
        1-D array of altitudes [N_V] in km.

    Returns
    -------
    density_out : array-like
        3-D electron density [N_T, N_V, N_G] in m-3.

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
    # number of elements in time dimention
    nUT = x.shape[1]

    # number of elements in horizontal dimention of grid
    nhor = x.shape[2]

    # time and horisontal grid dimention
    ngrid = nhor * nUT

    # vertical dimention
    nalt = aalt.size

    # empty arrays
    density_out = np.zeros((nalt, ngrid))
    density_F2 = np.zeros((nalt, ngrid))
    density_F1 = np.zeros((nalt, ngrid))
    density_E = np.zeros((nalt, ngrid))
    drop_1 = np.zeros((nalt, ngrid))
    drop_2 = np.zeros((nalt, ngrid))

    # shapes:
    # for filling with altitudes because the last dimentions should match
    # the source
    shape1 = (ngrid, nalt)

    # for filling with horizontal maps because the last dimentions should
    # match the source
    shape2 = (nalt, ngrid)

    order = 'F'
    NmF2 = np.reshape(x[0, :, :], ngrid, order=order)
    NmF1 = np.reshape(x[1, :, :], ngrid, order=order)
    NmE = np.reshape(x[2, :, :], ngrid, order=order)
    hmF2 = np.reshape(x[3, :, :], ngrid, order=order)
    hmF1 = np.reshape(x[4, :, :], ngrid, order=order)
    hmE = np.reshape(x[5, :, :], ngrid, order=order)
    B_F2_bot = np.reshape(x[6, :, :], ngrid, order=order)
    B_F2_top = np.reshape(x[7, :, :], ngrid, order=order)
    B_F1_bot = np.reshape(x[8, :, :], ngrid, order=order)
    B_E_bot = np.reshape(x[9, :, :], ngrid, order=order)
    B_E_top = np.reshape(x[10, :, :], ngrid, order=order)

    # set to some parameters if zero or lower:
    B_F1_bot[np.where(B_F1_bot <= 0)] = 10
    B_F2_top[np.where(B_F2_top <= 0)] = 30

    # array of hmFs with same dimensions as result, to later search
    # using argwhere
    a_alt = np.full(shape1, aalt, order='F')
    a_alt = np.swapaxes(a_alt, 0, 1)

    a_NmF2 = np.full(shape2, NmF2)
    a_NmF1 = np.full(shape2, NmF1)
    a_NmE = np.full(shape2, NmE)

    a_hmF2 = np.full(shape2, hmF2)
    a_hmF1 = np.full(shape2, hmF1)
    a_hmE = np.full(shape2, hmE)
    a_B_F2_top = np.full(shape2, B_F2_top)
    a_B_F2_bot = np.full(shape2, B_F2_bot)
    a_B_F1_bot = np.full(shape2, B_F1_bot)
    a_B_E_top = np.full(shape2, B_E_top)
    a_B_E_bot = np.full(shape2, B_E_bot)

    a_A1 = 4. * a_NmF2
    a_A2 = 4. * a_NmF1
    a_A3 = 4. * a_NmE

    # !!! do not use a[0], because all 3 dimensions are needed. this is
    # the same as density[a]= density[a[0], a[1], a[2]]

    # F2 top (same for yes F1 and no F1)
    a = np.where(a_alt >= a_hmF2)
    density_F2[a] = epstein_function_top_array(a_A1[a], a_hmF2[a],
                                               a_B_F2_top[a], a_alt[a])

    # E bottom (same for yes F1 and no F1)
    a = np.where(a_alt <= a_hmE)
    density_E[a] = epstein_function_array(a_A3[a], a_hmE[a], a_B_E_bot[a],
                                          a_alt[a])

    # when F1 is present-----------------------------------------
    # F2 bottom down to F1
    a = np.where((np.isfinite(a_NmF1))
                 & (a_alt < a_hmF2)
                 & (a_alt >= a_hmF1))
    density_F2[a] = epstein_function_array(a_A1[a], a_hmF2[a],
                                           a_B_F2_bot[a], a_alt[a])

    # E top plus F1 bottom (hard boundaries)
    a = np.where((a_alt > a_hmE) & (a_alt < a_hmF1))
    drop_1[a] = 1. - ((a_alt[a] - a_hmE[a]) / (a_hmF1[a] - a_hmE[a]))**4.
    drop_2[a] = 1. - ((a_hmF1[a] - a_alt[a]) / (a_hmF1[a] - a_hmE[a]))**4.

    density_E[a] = epstein_function_array(a_A3[a],
                                          a_hmE[a],
                                          a_B_E_top[a],
                                          a_alt[a]) * drop_1[a]
    density_F1[a] = epstein_function_array(a_A2[a],
                                           a_hmF1[a],
                                           a_B_F1_bot[a],
                                           a_alt[a]) * drop_2[a]

    # when F1 is not present(hard boundaries)--------------------
    a = np.where((np.isnan(a_NmF1))
                 & (a_alt < a_hmF2)
                 & (a_alt > a_hmE))
    drop_1[a] = 1. - ((a_alt[a] - a_hmE[a]) / (a_hmF2[a] - a_hmE[a]))**4.
    drop_2[a] = 1. - ((a_hmF2[a] - a_alt[a]) / (a_hmF2[a] - a_hmE[a]))**4.
    density_E[a] = epstein_function_array(a_A3[a],
                                          a_hmE[a],
                                          a_B_E_top[a],
                                          a_alt[a]) * drop_1[a]
    density_F2[a] = epstein_function_array(a_A1[a],
                                           a_hmF2[a],
                                           a_B_F2_bot[a],
                                           a_alt[a]) * drop_2[a]

    density = density_F2 + density_F1 + density_E

    density_out = np.reshape(density, (nalt, nUT, nhor), order='F')
    density_out = np.swapaxes(density_out, 0, 1)

    # make 1 everything that is <= 0
    density_out[np.where(density_out <= 1)] = 1.

    return density_out


def day_of_the_month_corr(year, month, day):
    """Find fractions of influence of months "before" and "after".

    Parameters
    ----------
    year : int
        Given year.
    month : int
        Given month.
    day : day
        Given day.

    Returns
    -------
    t_before : class:`dt.datetime`
        Consider mean values from this month as month "before".
    t_after : class:`dt.datetime`
        Consider mean values from this month as month "after".
    fraction1 : float
        Fractional influence of month "before".
    fraction2 : float
        Fractional influence of month "after".

    Notes
    -----
    This function finds two months around the given day and calculates
    fractions of influence for previous and following months.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    Bilitza et al. (2022), The International Reference Ionosphere
    model: A review and description of an ionospheric benchmark, Reviews
    of Geophysics, 60.

    """
    # middles of the months around
    delta_month = dt.timedelta(days=+30)
    dtime0 = dt.datetime(year, month, 15)
    dtime1 = dtime0 - delta_month
    dtime2 = dtime0 + delta_month

    # day of interest
    time_event = dt.datetime(year, month, day)

    # chose month before or after
    if day >= 15:
        t_before = dtime0
        t_after = dt.datetime(dtime2.year, dtime2.month, 15)

    if day < 15:
        t_before = dt.datetime(dtime1.year, dtime1.month, 15)
        t_after = dtime0

    # how many days after middle of previous month and to the middle of
    # next month
    dt1 = time_event - t_before
    dt2 = t_after - time_event
    dt3 = t_after - t_before

    # fractions of the influence for the interpolation
    fraction1 = dt2.days / dt3.days
    fraction2 = dt1.days / dt3.days

    return t_before, t_after, fraction1, fraction2


def fractional_correction_of_dictionary(fraction1, fraction2, F_before,
                                        F_after):
    """Interpolate btw 2 middles of consequent months to the given day.

    Parameters
    ----------
    fraction1 : float
        Fractional influence of month "before".
    fraction2 : float
        Fractional influence of month "after".
    F_before : dict
        Dictionary of mean parameters for month "before".
    F_after : dict
        Dictionary of mean parameters for month "after".

    Returns
    -------
    F_new : dict
        Parameters interpolated according to given fractions.

    Notes
    -----
    This function interpolates between 2 middles of consequent months to the
    specified day by using provided fractions previously calculated by
    function "day_of_the_month_corr".

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    Bilitza et al. (2022), The International Reference Ionosphere
    model: A review and description of an ionospheric benchmark, Reviews
    of Geophysics, 60.

    """
    F_new = F_before
    for key in F_before:
        F_new[key] = F_before[key] * fraction1 + F_after[key] * fraction2

    return F_new


def solar_interpolate(F_min, F_max, F107):
    """Interpolate given array to provided F10.7 level.

    Parameters
    ----------
    F_min : array-like
        Any given array of parameters that corresponds to solar min.
    F_max : array-like
        Any given array of parameters that corresponds to solar max.
    F107 : float
        Given solar flux index in SFU.

    Returns
    -------
    F : array-like
        Parameters interpolated to the given F10.7.

    Notes
    -----
    This function interpolates it between to a given F10.7. The
    reference points are set in terms of IG12 coefficients of 0 and 100.
    The F10.7 is first converted to IG12 and then the interpolation is
    occurred.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    Bilitza et al. (2022), The International Reference Ionosphere
    model: A review and description of an ionospheric benchmark, Reviews
    of Geophysics, 60.

    """
    # min and max of IG12 Ionospheric Global Index
    IG12_min = 0.
    IG12_max = 100.

    IG12 = F107_2_IG12(F107)

    # linear interpolation of the whole matrix:
    # https://en.wikipedia.org/wiki/Linear_interpolation
    F = (F_min * (IG12_max - IG12) / (IG12_max - IG12_min)
         + F_max * (IG12 - IG12_min) / (IG12_max - IG12_min))

    return F


def solar_interpolation_of_dictionary(F, F107):
    """Interpolate given dictionary to provided F10.7 level.

    Parameters
    ----------
    F : dict
        Dictionary of parameters with 2 levels of solar activity
        specified as 1st dimension.
    F107 : float
        Interpolate to this particular level of F10.7.

    Returns
    -------
    F_new : dict
        Parameters interpolated to the given F10.7.

    Notes
    -----
    This function looks at each key in the dictionary and interpolates
    it between to a given F10.7. The reference points are set in terms
    of IG12 coefficients of 0 and 100. The F10.7 is first converted to
    IG12 and then the interpolation is occurred.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    Bilitza et al. (2022), The International Reference Ionosphere
    model: A review and description of an ionospheric benchmark, Reviews
    of Geophysics, 60.

    """
    # Make dictionary with same elements as initial array
    F_new = F

    for key in F:
        F_key = F[key]
        F_key = np.swapaxes(F_key, 0, 2)
        F_new[key] = solar_interpolate(F_key[0, :], F_key[1, :], F107)
        F_new[key] = np.swapaxes(F_new[key], 0, 1)

    return F_new


def adjust_longitude(lon, type):
    """Adjust longitudes from 180 to 360 and back.

    Parameters
    ----------
    lon : array-like
        Longitudes in degrees.
    type : str
        Indicates the type of adjustment.

    Returns
    -------
    lon : array-like
        Adjusted longitude.

    Notes
    -----
    This function adjusts the array of longitudes to go from -180-180 or
    from 0-360.

    """
    if isinstance(lon, np.ndarray):
        if type == 'to360':
            # check that values in the array don't go over 360
            multiple = np.floor_divide(np.abs(lon), 360)
            lon = lon - multiple * 360 * np.sign(lon)

            a = np.where(lon < 0.)
            inda = a[0]
            lon[inda] = lon[inda] + 360.

            b = np.where(lon > 360.)
            indb = b[0]
            lon[indb] = lon[indb] - 360.

        if type == 'to180':
            # check that values in the array don't go over 360
            multiple = np.floor_divide(np.abs(lon), 360)
            lon = lon - multiple * 360 * np.sign(lon)

            a = np.where(lon > 180.)
            inda = a[0]
            lon[inda] = lon[inda] - 360.

            b = np.where(lon < -180.)
            indb = b[0]
            lon[indb] = lon[indb] + 360.

        if type == 'to24':
            # check that values in the array don't go over 24
            multiple = np.floor_divide(np.abs(lon), 24)
            lon = lon - multiple * 24. * np.sign(lon)

            a = np.where(lon < 0.)
            inda = a[0]
            lon[inda] = lon[inda] + 24.

            b = np.where(lon > 24.)
            indb = b[0]
            lon[indb] = lon[indb] - 24.

    if (isinstance(lon, int)) or (isinstance(lon, float)):
        if type == 'to360':
            # check that values in the array don't go over 360
            multiple = np.floor_divide(np.abs(lon), 360)
            lon = lon - multiple * 360 * np.sign(lon)

            if lon < 0.:
                lon = lon + 360.
            if lon > 360.:
                lon = lon - 360.

        if type == 'to180':
            # check that values in the array don't go over 360
            multiple = np.floor_divide(np.abs(lon), 360)
            lon = lon - multiple * 360 * np.sign(lon)

            if lon > 180.:
                lon = lon - 360.
            if lon < -180.:
                lon = lon + 360.

        if type == 'to24':
            # check that values in the array don't go over 24
            multiple = np.floor_divide(np.abs(lon), 24)
            lon = lon - multiple * 24. * np.sign(lon)

            if lon < 0.:
                lon = lon + 24.
            if lon > 24.:
                lon = lon - 24.

    return lon


def create_reg_grid(hr_res=1, lat_res=1, lon_res=1, alt_res=10, alt_min=0,
                    alt_max=700):
    """Run IRI for a single day on a regular grid.

    Parameters
    ----------
    hr_res : int or float
        Time resolution in hours (default=1)
    lat_res : int or float
        Latitude resolution in degrees (default=1)
    lon_res : int or float
        Longitude resolution in degrees (default=1)
    alt_res : int or float
        Altitude resolution in km (default=10)
    alt_min : int or float
        Altitude minimum in km (default=0)
    alt_max : int or float
        Altitude maximum in km (default=700)

    Returns
    -------
    alon : array-like
        1D longitude grid
    alat : array-like
        1D latitude grid
    alon_2d : array-like
        2D longitude grid
    alat_2d : array-like
        2D latitude grid
    aalt : array-like
        Altitude grid
    ahr : array-like
        UT grid

    Notes
    -----
    This function creates a regular global grid in Geographic coordinates
    using given spatial resolution lon_res, lat_res. In case you want to run
    IRI in magnetic coordinates, you can obtain a regular grid in magnetic
    coordinates, then convert it to geographic coordinates using your own
    methods, and then use these 1-D alon and alat arrays. Similarly, if you
    are interested in regional grid, then use your own alon and alat 1-D
    arrays. alon and alat can also be irregular arrays. In case you need to
    run IRI for 1 grid point, define alon and alat as NumPy arrays that have
    only 1 element. Size of alon and alat is [N_G].
    This function creates creates an array of altitudes for the vertical
    dimension of electron density profiles. Any 1-D Numpy array in [km] would
    work, regularly or irregularly spaced.
    This function also creates time array aUT using a given temporal
    resolution hr_res in hours.
    E.g. if hr_res = 0.25 it will lead to 15-minutes time
    resolution. You can define aUT your own way, just keep it 1-D and
    expressed in hours. It can be regularly or irregularly spaced array. If
    you want to run PyIRI for just 1 time frame, then define aUT as NumPy
    arrays that have only 1 element. Size of aUT is [N_T].

    """

    alon, alat, alon_2d, alat_2d = set_geo_grid(lon_res, lat_res)

    aalt = np.arange(alt_min, alt_max, alt_res)

    ahr, _, _, _, _ = set_temporal_array(hr_res)

    return alon, alat, alon_2d, alat_2d, aalt, ahr


def run_iri_reg_grid(year, month, day, f107, hr_res=1, lat_res=1, lon_res=1,
                     alt_res=10, alt_min=0, alt_max=700, ccir_or_ursi=0):
    """Run IRI for a single day on a regular grid.

    Parameters
    ----------
    year : int
        Four digit year in C.E.
    month : int
        Integer month (range 1-12)
    day : int
        Integer day of month (range 1-31)
    f107 : int or float
        F10.7 index for the given day
    hr_res : int or float
        Time resolution in hours (default=1)
    lat_res : int or float
        Latitude resolution in degrees (default=1)
    lon_res : int or float
        Longitude resolution in degrees (default=1)
    alt_res : int or float
        Altitude resolution in km (default=10)
    alt_min : int or float
        Altitude minimum in km (default=0)
    alt_max : int or float
        Altitude maximum in km (default=700)
    ccir_or_ursi : int
        If 0 use CCIR coefficients, if 1 use URSI coefficients

    Returns
    -------
    alon : array-like
        1D longitude grid
    alat : array-like
        1D latitude grid
    alon_2d : array-like
        2D longitude grid
    alat_2d : array-like
        2D latitude grid
    aalt : array-like
        Altitude grid
    ahr : array-like
        UT grid
    f2 : array-like
        F2 peak
    f1 : array-like
        F1 peak
    epeak : array-like
        E peak
    es_peak : array-like
        Sporadic E (Es) peak
    sun : array-like
        Solar zenith angle in degrees
    mag : array-like
        Magnetic inclination in degrees
    edens_prof : array-like
        Electron density profile in per cubic m

    See Also
    --------
    create_reg_grid

    """
    # Define the grids
    alon, alat, alon_2d, alat_2d, aalt, ahr = create_reg_grid(
        hr_res=hr_res, lat_res=lat_res, lon_res=lon_res,
        alt_res=alt_res, alt_min=alt_min,
        alt_max=alt_max)
    # -------------------------------------------------------------------------
    # Monthly mean density for min and max of solar activity:
    # -------------------------------------------------------------------------
    # The original IRI model further interpolates between 2 levels of solar
    # activity to estimate density for a particular level of F10.7.
    # Additionally, it interpolates between 2 consecutive months to make a
    # smooth seasonal transition. Here is an example of how this interpolation
    # can be done. If you need to run IRI for a particular day, you can just
    # use this function
    f2, f1, epeak, es_peak, sun, mag, edens_prof = IRI_density_1day(
        year, month, day, ahr, alon, alat, aalt, f107, PyIRI.coeff_dir,
        ccir_or_ursi)

    return (alon, alat, alon_2d, alat_2d, aalt, ahr, f2, f1, epeak,
            es_peak, sun, mag, edens_prof)


def run_seas_iri_reg_grid(year, month, hr_res=1, lat_res=1, lon_res=1,
                          alt_res=10, alt_min=0, alt_max=700, ccir_or_ursi=0):
    """Run IRI for monthly mean parameters on a regular grid.

    Parameters
    ----------
    year : int
        Four digit year in C.E.
    month : int
        Integer month (range 1-12)
    f107 : int or float
        F10.7 index for the given day
    hr_res : int or float
        Time resolution in hours (default=1)
    lat_res : int or float
        Latitude resolution in degrees (default=1)
    lon_res : int or float
        Longitude resolution in degrees (default=1)
    alt_res : int or float
        Altitude resolution in km (default=10)
    alt_min : int or float
        Altitude minimum in km (default=0)
    alt_max : int or float
        Altitude maximum in km (default=700)
    ccir_or_ursi : int
        If 0 use CCIR coefficients, if 1 use URSI coefficients

    Returns
    -------
    alon : array-like
        1D longitude grid
    alat : array-like
        1D latitude grid
    alon_2d : array-like
        2D longitude grid
    alat_2d : array-like
        2D latitude grid
    aalt : array-like
        Altitude grid
    ahr : array-like
        UT grid
    f2 : array-like
        F2 peak
    f1 : array-like
        F1 peak
    epeak : array-like
        E peak
    es_peak : array-like
        Sporadic E (Es) peak
    sun : array-like
        Solar zenith angle in degrees
    mag : array-like
        Magnetic inclination in degrees
    edens_prof : array-like
        Electron density profile in per cubic m

    See Also
    --------
    create_reg_grid

    """
    # Define the grids
    alon, alat, alon_2d, alat_2d, aalt, ahr = create_reg_grid(
        hr_res=hr_res, lat_res=lat_res, lon_res=lon_res,
        alt_res=alt_res, alt_min=alt_min,
        alt_max=alt_max)

    # -------------------------------------------------------------------------
    # Monthly mean ionospheric parameters for min and max of solar activity:
    # -------------------------------------------------------------------------
    # This is how PyIRI needs to be called to find monthly mean values for all
    # ionospheric parameters, for min and max conditions of solar activity:
    # year and month should be integers, and ahr and alon, alat should be 1-D
    # NumPy arrays. alon and alat should have the same size. coeff_dir is the
    # coefficient directory. Matrix size for all the output parameters is
    # [N_T, N_G, 2], where 2 indicates min and max of the solar activity that
    # corresponds to Ionospheric Global (IG) index levels of 0 and 100.

    f2, f1, epeak, es_peak, sun, mag = IRI_monthly_mean_par(
        year, month, ahr, alon, alat, PyIRI.coeff_dir, ccir_or_ursi)

    # -------------------------------------------------------------------------
    # Monthly mean density for min and max of solar activity:
    # -------------------------------------------------------------------------
    # Construct electron density profiles for min and max levels of solar
    # activity for monthly mean parameters.  The result will have the following
    # dimensions [2, N_T, N_V, N_G]
    edens_prof = reconstruct_density_from_parameters(f2, f1, epeak, aalt)

    return (alon, alat, alon_2d, alat_2d, aalt, ahr, f2, f1, epeak, es_peak,
            sun, mag, edens_prof)


def edp_to_vtec(edp, aalt, min_alt=0.0, max_alt=202000.0):
    """Calculate Vertical Total Electron Content (VTEC) from electron density.

    Parameters
    ----------
    edp : np.array
        Electron density profile in per cubic m with dimensions of
        [N_T, N_V, N_G]
    aalt : np.array
        Altitude array with altitude in km used to create the `edp`, has
        dimensions of [N_V].
    min_alt : float
        Minimum allowable altitude in km to include in TEC calculation
        (default=0.0)
    max_alt : float
        Maximum allowable altitude in km to include in TEC calculation
        (default=202000.0)

    Returns
    -------
    vtec : np.array
        Vertical Total Electron Content in TECU with dimensions of [N_T, N_G]

    Raises
    ------
    ValueError
        If `min_alt` and `max_alt` result in no density data to integrate.

    Notes
    -----
    Providing `aalt` allows irregular altitude grids to be used.

    Supplying `min_alt` and `max_alt` will not extend the electron density
    integration range, only limit it.  Current limits extend from the ground
    to the altitude of the GPS satellite constellation.

    1 TECU = 10^16 electrons per square meter

    """
    # Get the different dimensions
    num_t, num_v, num_g = edp.shape

    # Reshape the electron density information to be flat in the time-space dims
    edp_vert = edp.swapaxes(0, 1).reshape(num_v, num_t * num_g)

    # Select the good altitude range
    alt_mask = (aalt >= min_alt) & (aalt <= max_alt)
    new_v = alt_mask.sum()

    if new_v == 0:
        raise ValueError('Altitude range contains no values')

    # Get the distance between each electron density point in meters
    alt_res = (aalt[1:] - aalt[:-1]) * 1000.0  # km to meters
    uniq_res = np.unique(alt_res[alt_mask[:-1]])

    if uniq_res.size == 1:
        dist = np.full(shape=(new_v, num_t * num_g), fill_value=uniq_res[0])
    else:
        alt_res = list(alt_res[alt_mask[:-1]])
        if len(alt_res) < new_v:
            # If all values are good, the resolution array will be one short.
            # Pad with the last value.
            alt_res.append(alt_res[-1])

        dist = np.full(shape=(num_t * num_g, new_v),
                       fill_value=alt_res).transpose()

    # Calculate the VTEC by summing the electron density over the desired
    # altitude range at each time-space point.  At every altitude, the electron
    # density is multiplied by the height covered by that density value
    vtec = (edp_vert[alt_mask] * dist).sum(axis=0)

    # Convert to TECU and reshape to have unflattened time-space dims
    vtec = vtec.reshape((num_t, num_g)) * 1.0e-16

    return vtec
