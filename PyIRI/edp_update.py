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
import PyIRI.main_library as main
from PyIRI import logger


def IRI_monthly_mean_par(year, mth, aUT, alon, alat, coeff_dir, ccir_or_ursi=0):
    """Output monthly mean ionospheric parameters using new EDP construction.

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

    # -------------------------------------------------------------------------
    # Calculating magnetic inclanation, modified dip angle, and magnetic dip
    # latitude using IGRF at 300 km of altitude
    _, _, _, _, _, inc, _ = igrf.inclination(coeff_dir,
                                             dtime,
                                             alon, alat, 300.0)
    modip = igrf.inc2modip(inc, alat)
    mag_dip_lat = igrf.inc2magnetic_dip_latitude(inc)

    # --------------------------------------------------------------------------
    # Calculate diurnal Fourier functions F_D for the given time array aUT
    D_f0f2, D_M3000, D_Es_med = main.diurnal_functions(aUT)

    # --------------------------------------------------------------------------
    # Calculate geographic Jones and Gallet (JG) functions F_G for the given
    # grid and corresponding map of modip angles
    G_fof2, G_M3000, G_Es_med = main.set_gl_G(alon, alat, modip)

    # --------------------------------------------------------------------------
    # Read CCIR coefficients and form matrix U
    (F_c_CCIR,
     F_c_URSI,
     F_M3000_c,
     F_Es_med) = main.read_ccir_ursi_coeff(mth, coeff_dir)

    if ccir_or_ursi == 0:
        F_fof2_c = F_c_CCIR
    if ccir_or_ursi == 1:
        F_fof2_c = F_c_URSI

    # --------------------------------------------------------------------------
    # Multiply matrices (F_D U)F_G
    foF2, M3000, foEs = main.gamma(D_f0f2, D_M3000, D_Es_med, G_fof2, G_M3000,
                                   G_Es_med, F_fof2_c, F_M3000_c, F_Es_med)

    # --------------------------------------------------------------------------
    # Solar driven E region and locations of subsolar points
    foE, solzen, solzen_eff, slon, slat = main.gammaE(year, mth, aUT, alon,
                                                      alat, aIG)

    # --------------------------------------------------------------------------
    # Probability of F1 layer to appear based on SZA
    P_F1 = Probability_F1(year, mth, aUT, alon, alat)

    # --------------------------------------------------------------------------
    # Convert critical frequency to the electron density (m-3)
    # we don't yet have the foF1, therefore we are passing an array of zeros
    NmF2, _, NmE, NmEs = main.freq_to_Nm(foF2,
                                         np.zeros((foF2.size)),
                                         foE,
                                         foEs)

    # Limit NmF2 to 1e8
    # Apparently, sometimes PyIRI gives nan at night
    # E.g. 20240722 04:15 with F107 = 191
    # Here check that no NaNs go further, because we hate NaNs
    # Do the same for the E region, also just in case
    NmF2 = np.nan_to_num(NmF2)
    NmF2[NmF2 < 1e8] = 1e8
    NmE = np.nan_to_num(NmE)
    NmE[NmE < 1e8] = 1e8

    # --------------------------------------------------------------------------
    # Find heights of the F2 and E ionospheric layers
    hmF2, hmE, hmEs = main.hm_IRI(M3000, foE, foF2, modip, aIG)

    # --------------------------------------------------------------------------
    # Find thicknesses of the F2 and E ionospheric layers
    (B_F2_bot,
     B_F2_top,
     B_E_bot,
     B_E_top,
     B_Es_bot,
     B_Es_top) = main.thickness(foF2,
                                M3000,
                                hmF2,
                                hmE,
                                mth,
                                aIG)

    # --------------------------------------------------------------------------
    # Find height of the F1 layer based on the P and F2
    NmF1, foF1, hmF1, B_F1_bot = derive_dependent_F1_parameters(P_F1,
                                                                NmF2,
                                                                hmF2,
                                                                B_F2_bot,
                                                                hmE)

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
    t_before, t_after, fr1, fr2 = main.day_of_the_month_corr(year, mth, day)

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

    F2 = main.fractional_correction_of_dictionary(fr1, fr2, F2_1, F2_2)
    F1 = main.fractional_correction_of_dictionary(fr1, fr2, F1_1, F1_2)
    E = main.fractional_correction_of_dictionary(fr1, fr2, E_1, E_2)
    Es = main.fractional_correction_of_dictionary(fr1, fr2, Es_1, Es_2)
    sun = main.fractional_correction_of_dictionary(fr1, fr2, sun_1, sun_2)
    mag = main.fractional_correction_of_dictionary(fr1, fr2, mag_1, mag_2)

    # interpolate parameters in solar activity
    F2 = main.solar_interpolation_of_dictionary(F2, F107)
    F1 = main.solar_interpolation_of_dictionary(F1, F107)
    E = main.solar_interpolation_of_dictionary(E, F107)
    Es = main.solar_interpolation_of_dictionary(Es, F107)

    # derive_dependent_F1_parameters one more time after the interpolation
    # so that the F1 location does not carry the little errors caused by the
    # interpolation
    NmF1, foF1, hmF1, B_F1_bot = derive_dependent_F1_parameters(F1['P'],
                                                                F2['Nm'],
                                                                F2['hm'],
                                                                F2['B_bot'],
                                                                E['hm'])
    # Update the F1 dictionary with the re-derived parameters
    F1['Nm'] = NmF1
    F1['hm'] = hmF1
    F1['fo'] = foF1
    F1['B_bot'] = B_F1_bot

    # construct density
    EDP = reconstruct_density_from_parameters_1level(F2, F1, E, aalt)

    return F2, F1, E, Es, sun, mag, EDP


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
    full_F1 = np.zeros((nalt, ngrid))
    density_F1 = np.zeros((nalt, ngrid))
    density_E = np.zeros((nalt, ngrid))

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

    # Set to some parameters if zero or lower (just in case)
    B_F2_top[np.where(B_F2_top <= 0)] = 10.

    # Array of hmFs, importantly, with same dimensions as result, to
    # later search for regions using argwhere
    a_alt = np.full(shape1, aalt, order='F')
    a_alt = np.swapaxes(a_alt, 0, 1)

    # Fill arrays with parameters to add height dimension and populate it
    # with same values, this is important to keep all operations in matrix
    # form
    a_NmF2 = np.full(shape2, NmF2)
    a_NmF1 = np.full(shape2, NmF1)
    a_NmE = np.full(shape2, NmE)
    a_hmF2 = np.full(shape2, hmF2)
    a_hmF1 = np.full(shape2, hmF1)
    a_hmE = np.full(shape2, hmE)
    a_B_F2_bot = np.full(shape2, B_F2_bot)
    a_B_F2_top = np.full(shape2, B_F2_top)
    a_B_F1_bot = np.full(shape2, B_F1_bot)
    a_B_E_top = np.full(shape2, B_E_top)
    a_B_E_bot = np.full(shape2, B_E_bot)

    # Drop functions to reduce contributions of the layers when adding them up
    multiplier_down_F2 = drop_down(a_alt, a_hmF2, a_hmE)
    multiplier_down_F1 = drop_down(a_alt, a_hmF1, a_hmE)
    multiplier_up = drop_up(a_alt, a_hmE, a_hmF2)

    # In the where statements all 3 dimensions are needed.
    # This is the same as density[a]= density[a[0], a[1], a[2]].

    # ------F2 region------
    a = np.where(a_alt >= a_hmF2)
    density_F2[a] = main.epstein_function_top_array(4. * a_NmF2[a], a_hmF2[a],
                                                    a_B_F2_top[a], a_alt[a])
    a = np.where((a_alt < a_hmF2) & (a_alt >= a_hmE))
    density_F2[a] = (main.epstein_function_array(4. * a_NmF2[a],
                                                a_hmF2[a],
                                                a_B_F2_bot[a],
                                                a_alt[a])
                     * multiplier_down_F2[a])

    # ------E region-------
    a = np.where((a_alt >= a_hmE) & (a_alt < a_hmF2))
    density_E[a] = main.epstein_function_array(4. * a_NmE[a],
                                               a_hmE[a],
                                               a_B_E_top[a],
                                               a_alt[a]) * multiplier_up[a]
    a = np.where(a_alt < a_hmE)
    density_E[a] = main.epstein_function_array(4. * a_NmE[a],
                                               a_hmE[a],
                                               a_B_E_bot[a],
                                               a_alt[a])

    # Add F2 and E layers
    density = density_F2 + density_E

    # ------F1 region------
    a = np.where((a_alt > a_hmE) & (a_alt < a_hmF1))
    full_F1[a] = main.epstein_function_array(4. * a_NmF1[a],
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

    # Reshape to the [N_T, N_V, N_G]
    density_out = np.reshape(density, (nalt, nUT, nhor), order='F')
    density_out = np.swapaxes(density_out, 0, 1)

    return density_out


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
    alon, alat, alon_2d, alat_2d, aalt, ahr = main.create_reg_grid(
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
    alon, alat, alon_2d, alat_2d, aalt, ahr = main.create_reg_grid(
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


def derive_dependent_F1_parameters(P, NmF2, hmF2, B_F2_bot, hmE):
    """Combine DA with background F1 region.

    Parameters
    ----------
    P : array-like
        Probability of F1 to occurre from PyIRI.
    NmF2 : array-like
        NmF2 parameter - peak density of F2 layer.
    hmF2 : array-like
        hmF2 parameter - height of the peak of F2.
    B_F2_bot : array-like
        B_F2_bot parameter - thickness of F2.
    hmE : array-like
        hmE parameter - height of E layer.

    Returns
    -------
    NmF1 : array_like
        NmF1 parameter - peak of F1 layer.
    foF1 : array_like
        foF1 parameter - critical freqeuncy of F1 layer.
    hmF1 : array_like
        hmF1 parameter - peak height of F1 layer.
    B_F1_bot : array_like
        B_F1_bot - thickness of F1 layer.

    Notes
    -----
    This function derives F1 from F2 fields.

    """

    # Estimate the F1 layer peak height (hmF1) as 0.4 between the F2 peak
    # height (hmF2) and the E layer peak height (hmE)
    hmF1 = hmF2 - (hmF2 - hmE) * 0.4

    # Compute B_F1_bot using normalized probability P with a flexible
    # threshold.
    threshold = 0.1
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
    B_F1_bot = (hmF1 - hmE) * 0.5 * norm_P

    # Find the exact NmF1 at the hmF1 using F2 bottom function with the drop
    # down function
    NmF1 = (main.epstein_function_array(4. * NmF2, hmF2, B_F2_bot, hmF1)
            * drop_down(hmF1, hmF2, hmE))

    # Convert plasma density to freqeuncy
    foF1 = main.den2freq(NmF1)

    return NmF1, foF1, hmF1, B_F1_bot


def logistic_curve(h, h0, B):
    """Logistic function centered at h0 with width parameter B.

    Parameters
    ----------
    h : float or array-like
        Input value(s), e.g., altitude in km.
    h0 : float
        Center point of the logistic curve (where it equals 0.5).
    B : float
        Width parameter controlling the steepness of the transition.

    Returns
    -------
    f : float or array
        Logistic function value(s) in the range (0, 1).
    """
    h = np.asarray(h)

    return 1. / (1. + np.exp(-(h - h0) / B))


def drop_up(h, hmE, hmF2, drop_fraction=0.2):
    """Smooth function of altitude with a sharp drop.

    Parameters
    ----------
    h : float or array-like
        Altitude in km.
    hmE : float
        Altitude where function = 1.
    hmF2 : float
        Altitude where function = 0.
    drop_fraction : float, optional
        Fraction of the distance (hmF2 - hmE) over which the drop occurs.

    Returns
    -------
    f : float or array
        Function value at altitude h.
    """
    h = np.asarray(h)
    D = hmF2 - hmE
    # if D <= 0:
    #     raise ValueError("hmF2 must be greater than hmE.")
    B = drop_fraction * D
    ht = hmF2 - B  # center of the drop, near hmF2
    sigma_h = logistic_curve(h, ht, B)
    sigma_E = logistic_curve(hmE, ht, B)
    sigma_F2 = logistic_curve(hmF2, ht, B)

    return (sigma_F2 - sigma_h) / (sigma_F2 - sigma_E)


def drop_down(h, hmF2, hmE, drop_fraction=0.1):
    """Smooth function of altitude with a sharp rise.

    Parameters
    ----------
    h : float or array-like
        Altitude in km.
    hmF2 : float
        Altitude where function = 1.
    hmE : float
        Altitude where function = 0.
    drop_fraction : float, optional
        Fraction of the total distance (hmF2 - hmE) over which the transition
        occurs.

    Returns
    -------
    f : float or array
        Function value at altitude h.
    """
    h = np.asarray(h)
    D = hmF2 - hmE
    # if D <= 0:
    #     raise ValueError("hmF2 must be greater than hmE.")
    B = drop_fraction * D
    ht = hmE + B  # center of logistic curve
    sigma_h = logistic_curve(h, ht, B)
    sigma_E = logistic_curve(hmE, ht, B)
    sigma_F2 = logistic_curve(hmF2, ht, B)

    return (sigma_h - sigma_E) / (sigma_F2 - sigma_E)


def Probability_F1(year, mth, utime, alon, alat):
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

    Returns
    -------
    a_P : array-like
        Probability occurrence of F1 layer.

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

    # simplified constant value from Bilitza review 2022
    gamma = 2.36

    # solar zenith angle for day 15 in the month of interest
    solzen, _, _ = main.solzen_timearray_grid(year, mth, 15,
                                              utime, alon,
                                              alat)

    for isol in range(0, 2):
        a_P[:, :, isol] = (0.5 + 0.5 * np.cos(np.deg2rad(solzen)))**gamma

    return a_P