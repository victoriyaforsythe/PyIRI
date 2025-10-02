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

# from apexpy import Apex
import datetime as dt
import netCDF4 as nc
import numpy as np
import opt_einsum as oe
import os
import pandas as pd
import PyIRI
import PyIRI.edp_update as edpup
import PyIRI.igrf_library as igrf
import PyIRI.main_library as main
import scipy.special as ss
# def IRI_monthly_mean_par(year, month, aUT, alon, alat,
#                          coeff_dir=None, foF2_coeff='URSI',
#                          hmF2_model='SHU2015', coord='geo'):
#     """Output monthly mean ionospheric parameters using spherical harmonics.

#     Parameters
#     ----------
#     year : int
#         Year.
#     month : int
#         Month of the year.
#     aUT : float, int, or array-like of float or int
#         Universal Time (UT) in hours. Scalar inputs will be converted to a
#         Numpy array.
#         Shape (N_T,)
#     alon : float, list, or array-like
#         Flattened array of geographic longitude in degrees or magnetic local
#         time in hours. Scalar inputs will be converted to a Numpy array.
#         Shape (N_G,)
#     alat : float, list, or array-like
#         Flattened array of geographic or magnetic quasi-dipole latitude in
#         degrees. Scalar inputs will be converted to a Numpy array.
#         Shape (N_G,)
#     coeff_dir: str
#         Directory where the coefficient files are stored. If None, uses the
#         default coefficient files stored in PyIRI.coeff_dir.
#         (default=None)
#     foF2_coeff : str
#         Coefficients to use for foF2. Options are 'URSI' and 'CCIR'.
#         (default='URSI')
#     hmF2_model : str
#         Model to use for hmF2. Options are 'SHU2015', 'AMTB2013', and
#         'BSE1979'. (default='SHU2015')
#     coord : str
#         Coordinate system. Options are 'geo' for geographic and 'mlt' for
#         quasi-dipole latitude and magnetic local time. (default='geo')

#     Returns
#     -------
#     F2 : dict
#         'Nm': Peak density of F2 region [m-3].
#         'fo' : Critical frequency of F2 region [MHz].
#         'M3000' : Obliquity factor for a distance of 3,000 km. Defined as
#         refracted in the ionosphere, can be received at a distance of 3,000 km
#         [unitless].
#         'hm' : Height of the F2 peak [km].
#         'B_top' : PyIRI top thickness of the F2 region [km].
#         'B_bot' : PyIRI bottom thickness of the F2 region in [km].
#         'B0' : IRI ABT-2009 bottom thickness parameter of the F2 region [km].
#         'B1' : IRI ABT-2009 bottom shape parameter of the F2 region [unitless].
#         Shape (N_T, N_G, 2)
#     F1 : dict
#         'Nm' : Peak density of F1 region [m-3].
#         'fo' : Critical frequency of F1 region [MHz].
#         'P' : Probability occurrence of F1 region [unitless].
#         'hm' : Height of the F1 peak [km].
#         'B_bot' : Bottom thickness of the F1 region [km].
#         Shape (N_T, N_G, 2)
#     E : dict
#         'Nm' : Peak density of E region [m-3].
#         'fo' : Critical frequency of E region [MHz].
#         'hm' : Height of the E peak [km].
#         'B_top' : Bottom thickness of the E region [km].
#         'B_bot' : Bottom thickness of the E region [km].
#         Shape (N_T, N_G, 2)
#     sun : dict
#         'lon' : Longitude of subsolar point [degree].
#         'lat' : Latitude of subsolar point [degree].
#         Shape (N_T)
#     mag : dict
#         'inc' : Inclination of the magnetic field [degree].
#         'modip' : Modified dip angle [degree].
#         'mag_dip_lat' : Magnetic dip latitude [degree].
#         Shape (N_G) if coord='geo', (N_T, N_G) if coord='mlt'

#     Notes
#     -----
#     This function returns monthly mean ionospheric parameters for min and max
#     levels of solar activity, i.e., 12-month running mean of the Global
#     Ionosonde Index IG12 of value 0 and 100.

#     References
#     ----------
#     Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
#     International Reference Ionosphere Modeling Implemented in Python,
#     Space Weather.

#     """
#     # -------------------------------------------------------------------------
#     # Set coefficient file path if none given
#     if coeff_dir is None:
#         coeff_dir = PyIRI.coeff_dir

#     # -------------------------------------------------------------------------
#     # Convert inputs to Numpy arrays
#     aUT = main.to_numpy_array(aUT)
#     alon = main.to_numpy_array(alon)
#     alat = main.to_numpy_array(alat)
#     apdtime = (pd.to_datetime(dt.datetime(year, month, 15))
#                + pd.to_timedelta(aUT, 'hours'))

#     # -------------------------------------------------------------------------
#     # Coordinate system conversion to geographic and mlt
#     if coord == 'geo':
#         aglat, aglon = alat, alon
#         aqdlat, amlt = np.zeros((2, aUT.size, alat.size))
#         for iUT in range(len(aUT)):
#             pdtime = apdtime[iUT]
#             A = Apex(pdtime)
#             aqdlat[iUT, :], amlt[iUT, :] = A.convert(alat, alon,
#                                                      'geo', 'mlt', height=0,
#                                                      datetime=pdtime)
#     elif coord == 'mlt':
#         aqdlat, amlt = alat, alon
#         aglat, aglon = np.zeros((2, aUT.size, alat.size))
#         for iUT in range(len(aUT)):
#             pdtime = apdtime[iUT]
#             A = Apex(pdtime)
#             aglat[iUT, :], aglon[iUT, :] = A.convert(aqdlat, amlt,
#                                                      'mlt', 'geo', height=0,
#                                                      datetime=pdtime)
#     else:
#         raise ValueError('Coordinate system not yet implemented.')

#     # -------------------------------------------------------------------------
#     # Set limits for solar driver based on IG12 = 0 - 100.
#     aIG = np.array([0., 100.])

#     # -------------------------------------------------------------------------
#     # Find magnetic inclination
#     decimal_year = main.decimal_year(apdtime[0])

#     # -------------------------------------------------------------------------
#     # Calculate magnetic inclination, modified dip angle, and magnetic dip
#     # latitude using IGRF at 300 km altitude
#     inc = igrf.inclination(coeff_dir,
#                            decimal_year,
#                            aglon, aglat, 300., only_inc=True)
#     modip = igrf.inc2modip(inc, aglat)
#     mag_dip_lat = igrf.inc2magnetic_dip_latitude(inc)

#     # -------------------------------------------------------------------------
#     # Extract coefficient matrices and resonstruct ionospheric parameters
#     C = load_coeff_matrices(month, coeff_dir, foF2_coeff, hmF2_model)

#     n_FS_r = C.shape[2]
#     n_FS_c = n_FS_r // 2 + 1
#     F_FS = real_FS_func(aUT, n_FS_c)

#     n_SH = C.shape[3]
#     lmax = int(np.sqrt(n_SH)) - 1
#     atheta = np.deg2rad(-(aqdlat - 90))
#     aphi = np.deg2rad(amlt * 15)

#     F_SH = real_SH_func(atheta, aphi, lmax=lmax)
#     if coord == 'geo':
#         Params = oe.contract('ij,opjk,kil->oilp', F_FS, C, F_SH)
#     elif coord == 'mlt':
#         Params = oe.contract('ij,opjk,kl->oilp', F_FS, C, F_SH)

#     foF2 = Params[0, :, :, :]
#     B0 = Params[1, :, :, :]
#     B1 = Params[2, :, :, :]
#     M3000 = Params[3, :, :, :]
#     if hmF2_model != 'BSE1979':
#         hmF2 = Params[4, :, :, :]

#     NmF2 = main.freq2den(foF2)  # Previously: * 1e11

#     # -------------------------------------------------------------------------
#     # Solar driven E region and locations of subsolar points
#     foE, solzen, solzen_eff, slon, slat = gammaE_dynamic(year, month, aUT,
#                                                          aglon, aglat,
#                                                          aIG)
#     NmE = main.freq2den(foE)  # Previously: * 1e11

#     # -------------------------------------------------------------------------
#     # Probability of F1 layer appearance based on solar zenith angle
#     P_F1 = Probability_F1_with_solzen(solzen)

#     # -------------------------------------------------------------------------
#     # Introduce a minimum limit for the peaks to avoid negative density
#     NmE = main.limit_Nm(NmE)

#     # -------------------------------------------------------------------------
#     # Find heights of the F2 and E ionospheric layers
#     if hmF2_model == 'BSE1979':
#         if coord == 'mlt':
#             modip_mod = modip.copy()[:, np.newaxis, :]
#         else:
#             modip_mod = modip.copy()
#         hmF2, hmE, _ = main.hm_IRI(M3000, foE, foF2, modip_mod, aIG)
#     else:
#         hmE = 110. + np.zeros(M3000.shape)

#     # -------------------------------------------------------------------------
#     # Find thicknesses of the F2 and E ionospheric layers
#     B_F2_bot, B_F2_top, B_E_bot, B_E_top, B_Es_bot, B_Es_top = main.thickness(
#         foF2,
#         M3000,
#         hmF2,
#         hmE,
#         month,
#         aIG)

#     # -------------------------------------------------------------------------
#     # Find height of the F1 layer based on the P and F2
#     NmF1, foF1, hmF1, B_F1_bot = edpup.derive_dependent_F1_parameters(
#         P_F1,
#         NmF2,
#         hmF2,
#         B_F2_bot,
#         hmE)

#     # -------------------------------------------------------------------------
#     # Add all parameters to dictionaries:
#     F2 = {'Nm': NmF2,
#           'fo': foF2,
#           'M3000': M3000,
#           'hm': hmF2,
#           'B_top': B_F2_top,
#           'B_bot': B_F2_bot,
#           'B0': B0,
#           'B1': B1}

#     F1 = {'Nm': NmF1,
#           'fo': foF1,
#           'P': P_F1,
#           'hm': hmF1,
#           'B_bot': B_F1_bot}

#     E = {'Nm': NmE,
#          'fo': foE,
#          'hm': hmE,
#          'B_bot': B_E_bot,
#          'B_top': B_E_top,
#          'solzen': solzen,
#          'solzen_eff': solzen_eff}

#     sun = {'lon': slon,
#            'lat': slat}

#     mag = {'inc': inc,
#            'modip': modip,
#            'mag_dip_lat': mag_dip_lat}

#     return F2, F1, E, sun, mag


# def IRI_density_1day(year, month, day, aUT, alon, alat, aalt, F107,
#                      coeff_dir=None, foF2_coeff='URSI',
#                      hmF2_model='SHU2015', coord='geo'):
#     """Output ionospheric parameters for a particular day.

#     Parameters
#     ----------
#     year : int
#         Year.
#     month : int
#         Month of the year.
#     day : int
#         Day of the month.
#     aUT : float, int, or array-like of float or int
#         Universal Time (UT) in hours. Scalar inputs will be converted to a
#         Numpy array.
#         Shape (N_T,)
#     alon : float, list, or array-like
#         Flattened array of geographic longitude in degrees or magnetic local
#         time in hours. Scalar inputs will be converted to a Numpy array.
#         Shape (N_G,)
#     alat : float, list, or array-like
#         Flattened array of geographic or magnetic quasi-dipole latitude in
#         degrees. Scalar inputs will be converted to a Numpy array.
#         Shape (N_G,)
#     aalt : array-like
#         Array of altitudes in km. Scalar inputs will be converted to a Numpy
#         array.
#         Shape (N_V,)
#     F107 : int or float
#         User provided F10.7 solar flux index in SFU.
#     coeff_dir: str
#         Directory where the coefficient files are stored. If None, uses the
#         default coefficient files stored in PyIRI.coeff_dir.
#         (default=None)
#     foF2_coeff : str
#         Coefficients to use for foF2. Options are 'URSI' and 'CCIR'.
#         (default='URSI')
#     hmF2_model : str
#         Model to use for hmF2. Options are 'SHU2015', 'AMTB2013', and
#         'BSE1979'. (default='SHU2015')
#     coord : str
#         Coordinate system. Options are 'geo' for geographic and 'mlt' for
#         quasi-dipole latitude and magnetic local time. (default='geo')

#     Returns
#     -------
#     F2 : dict
#         'Nm': Peak density of F2 region [m-3].
#         'fo' : Critical frequency of F2 region [MHz].
#         'M3000' : Obliquity factor for a distance of 3,000 km. Defined as
#         refracted in the ionosphere, can be received at a distance of 3,000 km
#         [unitless].
#         'hm' : Height of the F2 peak [km].
#         'B_top' : PyIRI top thickness of the F2 region [km].
#         'B_bot' : PyIRI bottom thickness of the F2 region in [km].
#         'B0' : IRI ABT-2009 bottom thickness parameter of the F2 region [km].
#         'B1' : IRI ABT-2009 bottom shape parameter of the F2 region [unitless].
#         Shape (N_T, N_G)
#     F1 : dict
#         'Nm' : Peak density of F1 region [m-3].
#         'fo' : Critical frequency of F1 region [MHz].
#         'P' : Probability occurrence of F1 region [unitless].
#         'hm' : Height of the F1 peak [km].
#         'B_bot' : Bottom thickness of the F1 region [km].
#         Shape (N_T, N_G)
#     E : dict
#         'Nm' : Peak density of E region [m-3].
#         'fo' : Critical frequency of E region [MHz].
#         'hm' : Height of the E peak [km].
#         'B_top' : Bottom thickness of the E region [km].
#         'B_bot' : Bottom thickness of the E region [km].
#         Shape (N_T, N_G)
#     sun : dict
#         'lon' : Longitude of subsolar point [degree].
#         'lat' : Latitude of subsolar point [degree].
#         Shape (N_T)
#     mag : dict
#         'inc' : Inclination of the magnetic field [degree].
#         'modip' : Modified dip angle [degree].
#         'mag_dip_lat' : Magnetic dip latitude [degree].
#         Shape (N_G) if coord='geo', (N_T, N_G) if coord='mlt'
#     EDP : nd.array
#         Electron density profiles [m-3].
#         Shape (N_T, N_V, N_G)

#     Notes
#     -----
#     This function returns ionospheric parameters and 3-D electron density for a
#     given day and provided F10.7 solar flux index.

#     References
#     ----------
#     Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
#     International Reference Ionosphere Modeling Implemented in Python,
#     Space Weather.

#     """
#     # -------------------------------------------------------------------------
#     # Set coefficient file path if none given
#     if coeff_dir is None:
#         coeff_dir = PyIRI.coeff_dir

#     # -------------------------------------------------------------------------
#     # Convert inputs to Numpy arrays
#     aUT = main.to_numpy_array(aUT)
#     alon = main.to_numpy_array(alon)
#     alat = main.to_numpy_array(alat)
#     aalt = main.to_numpy_array(aalt)

#     # -------------------------------------------------------------------------
#     # Calculate required monhtly means and associated weights
#     t_before, t_after, fr1, fr2 = main.day_of_the_month_corr(year, month, day)

#     F2_1, F1_1, E_1, sun_1, mag_1 = IRI_monthly_mean_par(t_before.year,
#                                                          t_before.month,
#                                                          aUT,
#                                                          alon,
#                                                          alat,
#                                                          coeff_dir,
#                                                          foF2_coeff,
#                                                          hmF2_model,
#                                                          coord)
#     F2_2, F1_2, E_2, sun_2, mag_2 = IRI_monthly_mean_par(t_after.year,
#                                                          t_after.month,
#                                                          aUT,
#                                                          alon,
#                                                          alat,
#                                                          coeff_dir,
#                                                          foF2_coeff,
#                                                          hmF2_model,
#                                                          coord)

#     F2 = main.fractional_correction_of_dictionary(fr1, fr2, F2_1, F2_2)
#     F1 = main.fractional_correction_of_dictionary(fr1, fr2, F1_1, F1_2)
#     E = main.fractional_correction_of_dictionary(fr1, fr2, E_1, E_2)
#     sun = main.fractional_correction_of_dictionary(fr1, fr2, sun_1, sun_2)
#     mag = main.fractional_correction_of_dictionary(fr1, fr2, mag_1, mag_2)

#     # -------------------------------------------------------------------------
#     # Interpolate parameters in solar activity
#     F2 = main.solar_interpolation_of_dictionary(F2, F107)
#     F1 = main.solar_interpolation_of_dictionary(F1, F107)
#     E = main.solar_interpolation_of_dictionary(E, F107)

#     # -------------------------------------------------------------------------
#     # Introduce a minimum limit for the peaks to avoid negative density (for
#     # high F10.7, extrapolation can cause NmF2 to go negative)
#     F2['Nm'] = main.limit_Nm(F2['Nm'])
#     E['Nm'] = main.limit_Nm(E['Nm'])

#     # -------------------------------------------------------------------------
#     # Derive dependent F1 parameters after the interpolation so that the F1
#     # location does not carry the little errors caused by the interpolation
#     NmF1, foF1, hmF1, B_F1_bot = edpup.derive_dependent_F1_parameters(
#         F1['P'],
#         F2['Nm'],
#         F2['hm'],
#         F2['B_bot'],
#         E['hm'])

#     # -------------------------------------------------------------------------
#     # Update the F1 dictionary with the re-derived parameters
#     F1['Nm'] = NmF1
#     F1['hm'] = hmF1
#     F1['fo'] = foF1
#     F1['B_bot'] = B_F1_bot

#     # -------------------------------------------------------------------------
#     # Construct density
#     EDP = edpup.reconstruct_density_from_parameters_1level(F2, F1, E, aalt)

#     return F2, F1, E, sun, mag, EDP


# def sporadic_E_monthly_mean(year, month, aUT, alon, alat, coeff_dir=None,
#                             coord='geo'):
#     """Output monthly mean sporadic E layer using spherical harmonics.

#     Parameters
#     ----------
#     year : int
#         Year.
#     month : int
#         Month of the year.
#     aUT : float, int, or array-like of float or int
#         Universal Time (UT) in hours. Scalar inputs will be converted to a
#         Numpy array.
#         Shape (N_T,)
#     alon : float, list, or array-like
#         Flattened array of geographic longitude in degrees or magnetic local
#         time in hours. Scalar inputs will be converted to a Numpy array.
#         Shape (N_G,)
#     alat : float, list, or array-like
#         Flattened array of geographic or magnetic quasi-dipole latitude in
#         degrees. Scalar inputs will be converted to a Numpy array.
#         Shape (N_G,)
#     coeff_dir: str
#         Directory where the coefficient files are stored. If None, uses the
#         default coefficient files stored in PyIRI.coeff_dir.
#         (default=None)
#     coord : str
#         Coordinate system. Options are 'geo' for geographic and 'mlt' for
#         quasi-dipole latitude and magnetic local time. (default='geo')

#     Returns
#     -------
#     Es : dict
#         'Nm' : Peak density of Es region [m-3].
#         'fo' : Critical frequency of Es region [MHz].
#         'hm' : Height of the Es peak [km].
#         'B_top' : Bottom thickness of the Es region [km].
#         'B_bot' : Bottom thickness of the Es region [km].
#         Shape (N_T, N_G, 2)

#     Notes
#     -----
#     This function returns monthly mean ionospheric parameters for min and max
#     levels of solar activity, i.e., 12-month running mean of the Global
#     Ionosonde Index IG12 of value 0 and 100.

#     References
#     ----------
#     Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
#     International Reference Ionosphere Modeling Implemented in Python,
#     Space Weather.

#     """
#     # -------------------------------------------------------------------------
#     # Set coefficient file path if none given
#     if coeff_dir is None:
#         coeff_dir = PyIRI.coeff_dir

#     # -------------------------------------------------------------------------
#     # Convert inputs to Numpy arrays
#     aUT = main.to_numpy_array(aUT)
#     alon = main.to_numpy_array(alon)
#     alat = main.to_numpy_array(alat)
#     apdtime = (pd.to_datetime(dt.datetime(year, month, 15))
#                + pd.to_timedelta(aUT, 'hours'))

#     # -------------------------------------------------------------------------
#     # Coordinate system conversion to geographic and mlt
#     if coord == 'geo':
#         aglat, aglon = alat, alon
#         aqdlat, amlt = np.zeros((2, aUT.size, alat.size))
#         for iUT in range(len(aUT)):
#             pdtime = apdtime[iUT]
#             A = Apex(pdtime)
#             aqdlat[iUT, :], amlt[iUT, :] = A.convert(aglat, aglon,
#                                                      'geo', 'mlt', height=0,
#                                                      datetime=pdtime)
#     elif coord == 'mlt':
#         aqdlat, amlt = alat, alon
#     else:
#         raise ValueError('Coordinate system not yet implemented.')

#     # -------------------------------------------------------------------------
#     # Extract coefficient matrices and resonstruct ionospheric parameters
#     C = load_Es_coeff_matrix(month, coeff_dir)

#     n_FS_r = C.shape[1]
#     n_FS_c = n_FS_r // 2 + 1
#     F_FS = real_FS_func(aUT, n_FS_c)

#     n_SH = C.shape[2]
#     lmax = int(np.sqrt(n_SH)) - 1
#     atheta = np.deg2rad(-(aqdlat - 90))
#     aphi = np.deg2rad(amlt * 15)

#     F_SH = real_SH_func(atheta, aphi, lmax=lmax)
#     if coord == 'geo':
#         foEs = oe.contract('ij,pjk,kil->ilp', F_FS, C, F_SH)
#     elif coord == 'mlt':
#         foEs = oe.contract('ij,pjk,kl->ilp', F_FS, C, F_SH)

#     NmEs = main.freq2den(foEs)

#     hmEs = 110. + np.zeros(NmEs.shape)

#     B_Es_top = 1. + np.zeros((hmEs.shape))
#     B_Es_bot = 1. + np.zeros((hmEs.shape))

#     Es = {'Nm': NmEs,
#           'fo': foEs,
#           'hm': hmEs,
#           'B_top': B_Es_top,
#           'B_bot': B_Es_bot}

#     return Es


# def sporadic_E_1day(year, month, day, aUT, alon, alat, aalt, F107,
#                     coeff_dir=None, coord='geo'):
#     """Output sporadic E layer parameters for a particular day.

#     Parameters
#     ----------
#     year : int
#         Year.
#     month : int
#         Month of the year.
#     day : int
#         Day of the month.
#     aUT : float, int, or array-like of float or int
#         Universal Time (UT) in hours. Scalar inputs will be converted to a
#         Numpy array.
#         Shape (N_T,)
#     alon : float, list, or array-like
#         Flattened array of geographic longitude in degrees or magnetic local
#         time in hours. Scalar inputs will be converted to a Numpy array.
#         Shape (N_G,)
#     alat : float, list, or array-like
#         Flattened array of geographic or magnetic quasi-dipole latitude in
#         degrees. Scalar inputs will be converted to a Numpy array.
#         Shape (N_G,)
#     aalt : array-like
#         Array of altitudes in km. Scalar inputs will be converted to a Numpy
#         array.
#         Shape (N_V,)
#     F107 : int or float
#         User provided F10.7 solar flux index in SFU.
#     coeff_dir: str
#         Directory where the coefficient files are stored. If None, uses the
#         default coefficient files stored in PyIRI.coeff_dir.
#         (default=None)
#     coord : str
#         Coordinate system. Options are 'geo' for geographic and 'mlt' for
#         quasi-dipole latitude and magnetic local time. (default='geo')

#     Returns
#     -------
#     Es : dict
#         'Nm' : Peak density of Es region [m-3].
#         'fo' : Critical frequency of Es region [MHz].
#         'hm' : Height of the Es peak [km].
#         'B_top' : Bottom thickness of the Es region [km].
#         'B_bot' : Bottom thickness of the Es region [km].
#         Shape (N_T, N_G)

#     Notes
#     -----
#     This function returns ionospheric parameters of the sporadic E layer for a
#     given day and solar activity input.

#     References
#     ----------
#     Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
#     International Reference Ionosphere Modeling Implemented in Python,
#     Space Weather.

#     """
#     # -------------------------------------------------------------------------
#     # Set coefficient file path if none given
#     if coeff_dir is None:
#         coeff_dir = PyIRI.coeff_dir

#     # -------------------------------------------------------------------------
#     # Convert inputs to Numpy arrays
#     aUT = main.to_numpy_array(aUT)
#     alon = main.to_numpy_array(alon)
#     alat = main.to_numpy_array(alat)
#     aalt = main.to_numpy_array(aalt)

#     # -------------------------------------------------------------------------
#     # Calculate required monhtly means and associated weights
#     t_before, t_after, fr1, fr2 = main.day_of_the_month_corr(year, month, day)

#     Es_1 = sporadic_E_monthly_mean(t_before.year, t_before.month, aUT, alon,
#                                    alat, coeff_dir, coord)
#     Es_2 = sporadic_E_monthly_mean(t_after.year, t_after.month, aUT, alon,
#                                    alat, coeff_dir, coord)

#     Es = main.fractional_correction_of_dictionary(fr1, fr2, Es_1, Es_2)

#     # -------------------------------------------------------------------------
#     # Interpolate parameters in solar activity
#     Es = main.solar_interpolation_of_dictionary(Es, F107)

#     # -------------------------------------------------------------------------
#     # Introduce a minimum limit for the peaks to avoid negative density (for
#     # high F10.7, extrapolation can cause NmF2 to go negative)
#     Es['Nm'] = main.limit_Nm(Es['Nm'])

#     return Es


# def run_iri_reg_grid(year, month, day, F107, coeff_dir=None, hr_res=1,
#                      lat_res=1, lon_res=1, alt_res=10, alt_min=0, alt_max=700,
#                      foF2_coeff='URSI', hmF2_model='SHU2015', coord='geo'):
#     """Run IRI for a single day on a regular grid.

#     Parameters
#     ----------
#     year : int
#         Year.
#     month : int
#         Month of the year.
#     day : int
#         Day of the month.
#     F107 : int or float
#         User provided F10.7 solar flux index in SFU.
#     coeff_dir: str
#         Directory where the coefficient files are stored. If None, uses the
#         default coefficient files stored in PyIRI.coeff_dir.
#         (default=None)
#     hr_res : int or float
#         Time resolution in hours. (default=1)
#     lat_res : int or float
#         Latitude resolution (geographic or quasi-dipole) in degrees.
#         (default=1)
#     lon_res : int or float
#         Longitude resolution (geographic in degrees or magnetic local time
#         in units of 1/15 hours). (default=1)
#     alt_res : int or float
#         Altitude resolution in km. (default=10)
#     alt_min : int or float
#         Altitude minimum in km. (default=0)
#     alt_max : int or float
#         Altitude maximum in km. (default=700)
#     foF2_coeff : str
#         Coefficients to use for foF2. Options are 'URSI' and 'CCIR'.
#         (default='URSI')
#     hmF2_model : str
#         Model to use for hmF2. Options are 'SHU2015', 'AMTB2013', and
#         'BSE1979'. (default='SHU2015')
#     coord : str
#         Coordinate system. Options are 'geo' for geographic and 'mlt' for
#         quasi-dipole latitude and magnetic local time. (default='geo')

#     Returns
#     -------
#     alon : nd.arrray
#         Flattened array of geographic longitude in degrees or magnetic local
#         time [hour].
#         Shape (N_G,)
#     alat : array-like
#         Flattened array of geographic or magnetic quasi-dipole latitude
#         [degree].
#         Shape (N_G,)
#     alon_2d : array-like
#         2-D array of geographic longitude in degrees or magnetic local time
#         [hour].
#         Shape (N_G,)
#     alat_2d : array-like
#         2-D array of geographic or magnetic quasi-dipole latitude [degree].
#         Shape (N_G,)
#     aalt : array-like
#         Array of altitudes [km].
#         Shape (N_V,)
#     aUT : array-like
#         Array of UT times [hour].
#         Shape (N_T,)
#     F2 : dict
#         'Nm': Peak density of F2 region [m-3].
#         'fo' : Critical frequency of F2 region [MHz].
#         'M3000' : Obliquity factor for a distance of 3,000 km. Defined as
#         refracted in the ionosphere, can be received at a distance of 3,000 km
#         [unitless].
#         'hm' : Height of the F2 peak [km].
#         'B_top' : PyIRI top thickness of the F2 region [km].
#         'B_bot' : PyIRI bottom thickness of the F2 region in [km].
#         'B0' : IRI ABT-2009 bottom thickness parameter of the F2 region [km].
#         'B1' : IRI ABT-2009 bottom shape parameter of the F2 region [unitless].
#         Shape (N_T, N_G)
#     F1 : dict
#         'Nm' : Peak density of F1 region [m-3].
#         'fo' : Critical frequency of F1 region [MHz].
#         'P' : Probability occurrence of F1 region [unitless].
#         'hm' : Height of the F1 peak [km].
#         'B_bot' : Bottom thickness of the F1 region [km].
#         Shape (N_T, N_G)
#     E : dict
#         'Nm' : Peak density of E region [m-3].
#         'fo' : Critical frequency of E region [MHz].
#         'hm' : Height of the E peak [km].
#         'B_top' : Bottom thickness of the E region [km].
#         'B_bot' : Bottom thickness of the E region [km].
#         Shape (N_T, N_G)
#     sun : dict
#         'lon' : Longitude of subsolar point [degree].
#         'lat' : Latitude of subsolar point [degree].
#         Shape (N_T)
#     mag : dict
#         'inc' : Inclination of the magnetic field [degree].
#         'modip' : Modified dip angle [degree].
#         'mag_dip_lat' : Magnetic dip latitude [degree].
#         Shape (N_G) if coord='geo', (N_T, N_G) if coord='mlt'
#     EDP : nd.array
#         Electron density profiles [m-3].
#         Shape (N_T, N_V, N_G)

#     See Also
#     --------
#     create_reg_grid_geo_or_mag

#     """
#     # -------------------------------------------------------------------------
#     # Set coefficient file path if none given
#     if coeff_dir is None:
#         coeff_dir = PyIRI.coeff_dir

#     # -------------------------------------------------------------------------
#     # Create the grids
#     alon, alat, alon_2d, alat_2d, aalt, aUT = create_reg_grid_geo_or_mag(
#         hr_res=hr_res, lat_res=lat_res, lon_res=lon_res, alt_res=alt_res,
#         alt_min=alt_min, alt_max=alt_max, coord=coord)

#     # -------------------------------------------------------------------------
#     # Run IRI for one day
#     F2, F1, E, sun, mag, EDP = IRI_density_1day(
#         year, month, day, aUT, alon, alat, aalt, F107, coeff_dir,
#         foF2_coeff, hmF2_model, coord)

#     return alon, alat, alon_2d, alat_2d, aalt, aUT, F2, F1, E, sun, mag, EDP


# def create_reg_grid_geo_or_mag(hr_res=1, lat_res=1, lon_res=1, alt_res=10,
#                                alt_min=0, alt_max=700, coord='geo'):
#     """Create a regular grid in geographic or magnetic coordinates.

#     Parameters
#     ----------
#     hr_res : int or float
#         Time resolution in hours (default=1)
#     lat_res : int or float
#         Latitude resolution (geographic or quasi-dipole) in degrees (default=1)
#     lon_res : int or float
#         Longitude resolution (geographic in degrees or magnetic local time in
#         units of 1/15 hours) (default=1)
#     alt_res : int or float
#         Altitude resolution in km (default=10)
#     alt_min : int or float
#         Altitude minimum in km (default=0)
#     alt_max : int or float
#         Altitude maximum in km (default=700)
#     coord : str
#         Coordinate system. Options are 'geo' for geographic and 'mlt' for
#         quasi-dipole latitude and magnetic local time. (default='geo')

#     Returns
#     -------
#     alon : nd.arrray
#         Flattened array of geographic longitude in degrees or magnetic local
#         time [hour].
#         Shape (N_G,)
#     alat : array-like
#         Flattened array of geographic or magnetic quasi-dipole latitude
#         [degree].
#         Shape (N_G,)
#     alon_2d : array-like
#         2-D array of geographic longitude in degrees or magnetic local time
#         [hour].
#         Shape (N_G,)
#     alat_2d : array-like
#         2-D array of geographic or magnetic quasi-dipole latitude [degree].
#         Shape (N_G,)
#     aalt : array-like
#         Array of altitudes [km].
#         Shape (N_V,)
#     aUT : array-like
#         Array of UT times [hour].
#         Shape (N_T,)

#     Notes
#     -----
#     This function creates a regular global grid in geographic or magnetic
#     coordinates using the given spatial resolution lon_res, lat_res; an array
#     of altitudes using the given altitude resolution alt_res for the vertical
#     dimension of electron density profiles; and a time array using the given
#     temporal resolution hr_res in hours.

#     """
#     alon = np.arange(0, 360 + lon_res, lon_res)
#     if coord == 'mlt':
#         alon = alon / 15
#     alat = np.arange(90, -90 - lat_res, -lat_res)
#     alon_2d, alat_2d = np.meshgrid(alon, alat)

#     alon = alon_2d.reshape(alon_2d.size)
#     alat = alat_2d.reshape(alat_2d.size)

#     aalt = np.arange(alt_min, alt_max, alt_res)

#     aUT = np.arange(0, 24, hr_res)

#     return alon, alat, alon_2d, alat_2d, aalt, aUT


# def run_seas_iri_reg_grid(year, month, coeff_dir=None, hr_res=1, lat_res=1,
#                           lon_res=1, alt_res=10, alt_min=0, alt_max=700,
#                           foF2_coeff='URSI', hmF2_model='SHU2015',
#                           coord='geo'):
#     """Run IRI for monthly mean parameters on a regular grid.

#     Parameters
#     ----------
#     year : int
#         Year.
#     month : int
#         Month of the year.
#     coeff_dir: str
#         Directory where the coefficient files are stored. If None, uses the
#         default coefficient files stored in PyIRI.coeff_dir.
#         (default=None)
#     hr_res : int or float
#         Time resolution in hours. (default=1)
#     lat_res : int or float
#         Latitude resolution (geographic or quasi-dipole) in degrees.
#         (default=1)
#     lon_res : int or float
#         Longitude resolution (geographic in degrees or magnetic local time
#         in units of 1/15 hours). (default=1)
#     alt_res : int or float
#         Altitude resolution in km. (default=10)
#     alt_min : int or float
#         Altitude minimum in km. (default=0)
#     alt_max : int or float
#         Altitude maximum in km. (default=700)
#     foF2_coeff : str
#         Coefficients to use for foF2. Options are 'URSI' and 'CCIR'.
#         (default='URSI')
#     hmF2_model : str
#         Model to use for hmF2. Options are 'SHU2015', 'AMTB2013', and
#         'BSE1979'. (default='SHU2015')
#     coord : str
#         Coordinate system. Options are 'geo' for geographic and 'mlt' for
#         quasi-dipole latitude and magnetic local time. (default='geo')

#     Returns
#     -------
#     alon : nd.arrray
#         Flattened array of geographic longitude in degrees or magnetic local
#         time in hours.
#         Shape (N_G,)
#     alat : array-like
#         Flattened array of geographic or magnetic quasi-dipole latitude in
#         degrees.
#         Shape (N_G,)
#     alon_2d : array-like
#         2-D array of geographic longitude in degrees or magnetic local time in
#         hours.
#         Shape (N_G,)
#     alat_2d : array-like
#         2-D array of geographic or magnetic quasi-dipole latitude in degrees.
#         Shape (N_G,)
#     aalt : array-like
#         Array of altitudes [km].
#         Shape (N_V,)
#     aUT : array-like
#         Array of UT times [hour].
#         Shape (N_T,)
#     F2 : dict
#         'Nm': Peak density of F2 region [m-3].
#         'fo' : Critical frequency of F2 region [MHz].
#         'M3000' : Obliquity factor for a distance of 3,000 km. Defined as
#         refracted in the ionosphere, can be received at a distance of 3,000 km
#         [unitless].
#         'hm' : Height of the F2 peak [km].
#         'B_top' : PyIRI top thickness of the F2 region [km].
#         'B_bot' : PyIRI bottom thickness of the F2 region in [km].
#         'B0' : IRI ABT-2009 bottom thickness parameter of the F2 region [km].
#         'B1' : IRI ABT-2009 bottom shape parameter of the F2 region [unitless].
#         Shape (N_T, N_G, 2)
#     F1 : dict
#         'Nm' : Peak density of F1 region [m-3].
#         'fo' : Critical frequency of F1 region [MHz].
#         'P' : Probability occurrence of F1 region [unitless].
#         'hm' : Height of the F1 peak [km].
#         'B_bot' : Bottom thickness of the F1 region [km].
#         Shape (N_T, N_G, 2)
#     E : dict
#         'Nm' : Peak density of E region [m-3].
#         'fo' : Critical frequency of E region [MHz].
#         'hm' : Height of the E peak [km].
#         'B_top' : Bottom thickness of the E region [km].
#         'B_bot' : Bottom thickness of the E region [km].
#         Shape (N_T, N_G, 2)
#     sun : dict
#         'lon' : Longitude of subsolar point [degree].
#         'lat' : Latitude of subsolar point [degree].
#         Shape (N_T)
#     mag : dict
#         'inc' : Inclination of the magnetic field [degree].
#         'modip' : Modified dip angle [degree].
#         'mag_dip_lat' : Magnetic dip latitude [degree].
#         Shape (N_G) if coord='geo', (N_T, N_G) if coord='mlt'
#     EDP : nd.array
#         Electron density profiles [m-3].
#         Shape (N_T, N_V, N_G, 2)

#     See Also
#     --------
#     create_reg_grid_geo_or_mag

#     """
#     # -------------------------------------------------------------------------
#     # Set coefficient file path if none given
#     if coeff_dir is None:
#         coeff_dir = PyIRI.coeff_dir

#     alon, alat, alon_2d, alat_2d, aalt, aUT = create_reg_grid_geo_or_mag(
#         hr_res=hr_res, lat_res=lat_res, lon_res=lon_res, alt_res=alt_res,
#         alt_min=alt_min, alt_max=alt_max, coord=coord)

#     F2, F1, E, sun, mag = IRI_monthly_mean_par(year, month, aUT, alon, alat,
#                                                coeff_dir, foF2_coeff,
#                                                hmF2_model, coord)

#     EDP = edpup.reconstruct_density_from_parameters(F2, F1, E, aalt)

#     return alon, alat, alon_2d, alat_2d, aalt, aUT, F2, F1, E, sun, mag, EDP


# def real_SH_func(theta, phi, lmax=29):
#     """Generate real-valued spherical harmonic basis functions.

#     Generate real-valued spherical harmonic basis functions up to degree
#     lmax (4pi-normalized). Includes Condon-Shortley phase. To use with SH
#     coefficients created with pyshtools, set cspase=-1 and normalization='4pi'
#     in pyshtools.

#     Parameters
#     ----------
#     theta : array-like
#         User input colatitudes [0-π] [rad].
#         Shape (N_T, N_G) if grids are converted to magnetic from geographic
#         coordinates, (N_G,) if grids are originally in magnetic coordinates
#     phi : array-like
#         User input longitudes [0-2π) [rad].
#         Shape (N_T, N_G) if input grids were converted to magnetic from
#         geographic coordinates, (N_G,) if grids were originally in magnetic
#         coordinates
#     lmax : int
#         Maximum spherical harmonic degree (and order). (default=29)

#     Returns
#     -------
#     F_SH : array-like
#         Real-valued spherical harmonic basis matrix, with N_SH=(lmax+1)**2.
#         Each column corresponds to a pair of coordinates, each row to an SH
#         mode. The SH mode of degree l and order m is stored in row i=l*(l+1)+m.
#         Shape (N_SH, N_T, N_G) if input grids were converted to magnetic from
#         geographic coordinates, (N_SH, N_G) if grids were originally in
#         magnetic coordinates

#     """
#     # -------------------------------------------------------------------------
#     # Convert inputs to arrays if lists or int or float
#     theta = main.to_numpy_array(theta)
#     phi = main.to_numpy_array(phi)

#     # -------------------------------------------------------------------------
#     # If theta has shape (n_pos,), i.e., if user input coord='mlt', theta and
#     # phi arrays must be artificially expanded to shape (1, n_pos)
#     mlt_flag = 0
#     if len(theta.shape) == 1:
#         mlt_flag = 1
#         theta = theta[np.newaxis, :]
#         phi = phi[np.newaxis, :]

#     # -------------------------------------------------------------------------
#     # Argument for the associated Legendre polynomials
#     z = np.cos(theta)

#     # -------------------------------------------------------------------------
#     # Determine the unnormalized associated Legendre polynomials
#     P_all = ss.assoc_legendre_p_all(lmax, lmax, z, norm=False).squeeze(0)
#     # Only need the positive orders to build real SH
#     P_cropped = P_all[:, :lmax + 1, :, :]

#     # -------------------------------------------------------------------------
#     # Scipy bug fix; we should have P(-1)=(-1)**l for m=0 else P(-1)=0, but
#     # scipy gives P(-1)=1 for m=0 else P(-1)=0
#     if -1.0 in z:
#         idx_180_time = np.where(z == -1.0)[0]
#         idx_180_pos = np.where(z == -1.0)[1]
#         P_cropped[1::2, 0, idx_180_time, idx_180_pos] *= -1.0

#     # -------------------------------------------------------------------------
#     # 4-pi normalization using matrix multiplication
#     l_mat = np.ones_like(P_cropped, dtype=float)
#     l_range = np.arange(0, lmax + 1)[:, np.newaxis, np.newaxis, np.newaxis]
#     l_mat *= l_range
#     m_mat = np.ones_like(P_cropped, dtype=float)
#     m_range = np.arange(0, lmax + 1)[np.newaxis, :, np.newaxis, np.newaxis]
#     m_mat *= np.abs(m_range)

#     del_m0 = np.array(m_mat == 0, dtype=float)
#     Norm = np.ones_like(P_cropped, dtype=float)
#     Norm *= np.sqrt((2 - del_m0) * (2 * l_mat + 1)
#                     * ss.factorial(l_mat - m_mat)
#                     / (ss.factorial(l_mat + m_mat)))

#     P_cropped *= Norm

#     # -------------------------------------------------------------------------
#     # Create cos(m*phi) and sin(m*phi) matrices
#     phi = phi[np.newaxis, :]
#     m_range = np.arange(0, lmax + 1)[:, np.newaxis, np.newaxis]

#     cos_phi = np.cos(m_range * phi)[np.newaxis, :, :]
#     sin_phi = np.sin(m_range * phi)[np.newaxis, :, :]

#     # -------------------------------------------------------------------------
#     # Create the positive and negative order real SH
#     Y_all_pos = P_cropped * cos_phi
#     Y_all_neg = P_cropped * sin_phi

#     # -------------------------------------------------------------------------
#     # Flatten the real SH matrix from (lmax+1, lmax+1, N_T, N_G) to
#     # ((lmax+1)**2, N_T, N_G)
#     N_G = Y_all_pos.shape[3]
#     N_T = Y_all_pos.shape[2]
#     lmax = Y_all_pos.shape[0] - 1

#     mask = np.triu(np.ones((lmax + 1, lmax + 1), dtype=bool), k=1)

#     Y_all_pos[mask] = np.nan
#     Y_all_neg[mask] = np.nan

#     Y_all_neg_flip = np.flip(Y_all_neg, axis=1)

#     Y_all_recombined = np.concatenate((Y_all_neg_flip[:, :-1, :, :],
#                                        Y_all_pos), axis=1)
#     Y_all_flat = Y_all_recombined.reshape(((lmax + 1) * (2 * lmax + 1),
#                                            N_T, N_G))
#     mask_flat = np.isfinite(Y_all_flat[:, 0, 0])
#     F_SH = Y_all_flat[mask_flat, :, :]

#     # -------------------------------------------------------------------------
#     # If N_T=1, i.e., user input coord='mlt', then we can get rid of the N_T
#     # dimension
#     if mlt_flag:
#         F_SH = F_SH.squeeze(1)

#     return F_SH


# def real_FS_func(aUT, N_FS_c=5):
#     """Generate a real-valued Fourier Series (FS) basis matrix.

#     Parameters
#     ----------
#     aUT : int, float, or array-like
#         User input time values in Universal Time (UT) [0-24) [hour]. Scalar
#         inputs will be converted to a Numpy array.
#         Shape (N_T,)

#     N_FS_c : int
#         Number of complex Fourier coefficients to use (i.e., truncation level).
#         (default=5) The associated number of real Fourier coefficients is
#         N_FS_r=2*N_FS_c-1.

#     Returns
#     -------
#     F_FS : array-like
#         Real-valued FS basis matrix.
#         Shape (N_T, N_FS_r)

#     """
#     # -------------------------------------------------------------------------
#     # Convert inputs to Numpy arrays
#     aUT = main.to_numpy_array(aUT)

#     N_T = aUT.size
#     N_FS_r = 2 * N_FS_c - 1
#     F_FS = np.empty((N_T, N_FS_r))

#     k_vals = np.arange(1, N_FS_c)
#     omega = 2 * np.pi * k_vals / 24
#     phase = np.outer(aUT, omega)

#     F_FS[:, 0] = 1
#     F_FS[:, 1::2] = np.cos(phase)
#     F_FS[:, 2::2] = np.sin(phase)

#     return F_FS


# def load_coeff_matrices(month, coeff_dir=None, foF2_coeff='URSI',
#                         hmF2_model='SHU2015'):
#     """Load ionospheric model coefficient matrices from NetCDF files.

#     Parameters
#     ----------
#     month : int
#         Month of the year.
#     coeff_dir: str
#         Directory where the coefficient files are stored. If None, uses the
#         default coefficient files stored in PyIRI.coeff_dir.
#         (default=None)
#     foF2_coeff : str
#         Coefficients to use for foF2. Options are 'URSI' and 'CCIR'.
#         (default='URSI')
#     hmF2_model : str
#         Model to use for hmF2. Options are 'SHU2015', 'AMTB2013', and
#         'BSE1979'. (default='SHU2015')

#     Returns
#     -------
#     C : array-like
#         Coefficient matrix. N_P is the number of parameters (N_P=5 if
#         hmF2_model != 'BSE1979' else N_P=6), N_IG=2 is the number of IG12
#         values stored (IG12=0 and IG12=100), N_FS=9 is the number of real FS
#         coefficients used, and N_SH=900 is the number of real SH coefficients
#         used.
#         Shape (N_P, N_IG, N_FS, N_SH)

#     """
#     # -------------------------------------------------------------------------
#     # Set coefficient file path if none given
#     if coeff_dir is None:
#         coeff_dir = PyIRI.coeff_dir

#     # -------------------------------------------------------------------------
#     # Load foF2, B0, B1, and M3000F2 coefficients
#     filenames = [f'foF2_{foF2_coeff}.nc', 'B0.nc', 'B1.nc', 'M3000F2.nc']

#     path = os.path.join(coeff_dir, 'SH', filenames[0])
#     with nc.Dataset(path) as ds:
#         C_month = ds['Coefficients'][:, month - 1, :, :]
#         N_IG = C_month.shape[0]
#         N_FS = C_month.shape[1]
#         N_SH = C_month.shape[2]
#         C = np.zeros((len(filenames), N_IG, N_FS, N_SH))

#     for ids in range(len(filenames)):
#         fname = filenames[ids]
#         path = os.path.join(coeff_dir, 'SH', fname)
#         with nc.Dataset(path) as ds:
#             C_month = ds['Coefficients'][:, month - 1, :, :]
#             C[ids, :, :, :] = C_month

#     # -------------------------------------------------------------------------
#     # Optionally add hmF2 coefficients
#     if hmF2_model != 'BSE1979':
#         hmF2_path = os.path.join(coeff_dir, 'SH', f'hmF2_{hmF2_model}.nc')
#         with nc.Dataset(hmF2_path) as ds:
#             C_month = ds['Coefficients'][:, month - 1, :, :]
#             C_month = C_month[np.newaxis, :, :, :]
#             C = np.concatenate([C, C_month], axis=0)

#     return C


# def load_Es_coeff_matrix(month, coeff_dir=None):
#     """Load sporadic E layer coefficient matrix from its NetCDF file.

#     Parameters
#     ----------
#     month : int
#         Month of the year.
#     coeff_dir: str
#         Directory where the coefficient files are stored. If None, uses the
#         default coefficient files stored in PyIRI.coeff_dir.
#         (default=None)

#     Returns
#     -------
#     C : array-like
#         Coefficient matrix. N_IG=2 is the number of IG12 values stored (IG12=0
#         and IG12=100), N_FS=9 is the number of real FS coefficients used, and
#         N_SH=8100 is the number of real SH coefficients used.
#         Shape (N_IG, N_FS, N_SH)

#     """
#     # -------------------------------------------------------------------------
#     # Set coefficient file path if none given
#     if coeff_dir is None:
#         coeff_dir = PyIRI.coeff_dir

#     # -------------------------------------------------------------------------
#     # Load Es coefficients
#     filename = 'foEs.nc'

#     path = os.path.join(coeff_dir, 'SH', filename)
#     with nc.Dataset(path) as ds:
#         C = ds['Coefficients'][:, month - 1, :, :]

#     return C


# def gammaE_dynamic(year, month, aUT, alon, alat, aIG):
#     """Calculate numerical maps for critical frequency of E region.

#     Parameters
#     ----------
#     year : int
#         Year.
#     month : int
#         Month.
#     aUT : int, float, or array-like
#         Array of UT Time in hours.
#     alon : int, float, or array-like
#         Flattened array of geographic longitudes in degrees.
#     alat : int, float, or array-like
#         Flattened array of geographic latitudes in degrees.
#     aIG : array-like
#         Min and max of IG12 index.

#     Returns
#     -------
#     gamma_E : array-like
#         Critical frequency of the E region [MHz].
#     solzen : array-like
#         Solar zenith angle [degree].
#     solzen_eff : array-like
#         Effective solar zenith angle [degree].
#     slon : array-like
#         Geographic longitude [degree] or magnetic local time [hour] of the
#         subsolar point.
#     slat : array-like
#         Geographic or quasi-dipole latitude of the subsolar point [degree].

#     Notes
#     -----
#     This function calculates numerical maps for foE for two levels of solar
#     activity.

#     References
#     ----------
#     Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
#     International Reference Ionosphere Modeling Implemented in Python,
#     Space Weather.

#     """
#     # -------------------------------------------------------------------------
#     # Convert inputs to Numpy arrays
#     aUT = main.to_numpy_array(aUT)
#     alon = main.to_numpy_array(alon)
#     alat = main.to_numpy_array(alat)

#     # -------------------------------------------------------------------------
#     # Determine which coordinates are used as inputs
#     # (N_G,) = magnetic coordinates
#     # (N_T, N_G) = geographic coordinates
#     if len(alon.shape) > 1:
#         N_G = alon.shape[1]
#         N_T = alon.shape[0]
#     else:
#         N_G = alon.shape[0]
#         N_T = aUT.shape[0]

#     # -------------------------------------------------------------------------
#     # Min and max of solar activity
#     aF107_min_max = np.array([main.IG12_2_F107(aIG[0]),
#                               main.IG12_2_F107(aIG[1])])

#     # -------------------------------------------------------------------------
#     # Initialize numerical map arrays for 2 levels of solar activity
#     gamma_E = np.zeros((N_T, N_G, 2))
#     solzen_out = np.zeros((N_T, N_G, 2))
#     solzen_eff_out = np.zeros((N_T, N_G, 2))

#     # -------------------------------------------------------------------------
#     # Initialize subsolar point location arrays
#     slon = np.zeros((N_T))
#     slat = np.zeros((N_T))

#     # -------------------------------------------------------------------------
#     # Loop to select the correct grid at each UT time
#     for iIG in range(0, 2):
#         for iUT in range(0, N_T):
#             if len(alon.shape) > 1:
#                 alon_iUT = alon[iUT, :]
#                 alat_iUT = alat[iUT, :]
#             else:
#                 alon_iUT = alon
#                 alat_iUT = alat

#             solzen_t, slon_t, slat_t = main.solzen_timearray_grid(
#                 year,
#                 month,
#                 15,
#                 np.array([aUT[iUT]]),
#                 alon_iUT,
#                 alat_iUT)

#             # -----------------------------------------------------------------
#             # Convert subsolar point geographic coordinates back to magnetic if
#             # user choice of coordinates is magnetic
#             if len(alon.shape) > 1:
#                 pdtime = (pd.to_datetime(dt.datetime(year, month, 15)
#                           + pd.to_timedelta(aUT[iUT], 'hours')))
#                 A = Apex(pdtime)
#                 slat_t, slon_t = A.convert(slat_t, slon_t, 'geo', 'mlt',
#                                            height=0, datetime=pdtime)

#             # -----------------------------------------------------------------
#             # Convert [-180,180] to [0,360] degrees geographic longitude if
#             # user choice is [0,360]
#             elif np.nanmin(alon) >= 0:
#                 slon_t += 180

#             slon[iUT] = slon_t
#             slat[iUT] = slat_t
#             solzen = solzen_t[0, :]

#             # -----------------------------------------------------------------
#             # Effective solar zenith angle
#             solzen_eff = main.solzen_effective(solzen)

#             # -----------------------------------------------------------------
#             # Save output arrays
#             gamma_E[iUT, :, iIG] = main.foE(month, solzen_eff, alat_iUT,
#                                             aF107_min_max[iIG])
#             solzen_out[iUT, :, iIG] = solzen
#             solzen_eff_out[iUT, :, iIG] = solzen_eff

#     return gamma_E, solzen_out, solzen_eff_out, slon, slat


def Probability_F1_with_solzen(solzen):
    """Calculate probability occurrence of F1 layer.

    Parameters
    ----------
    solzen : array-like
        Array of solar zenith angles [degree].
        Shape (N_T, N_G, 2)

    Returns
    -------
    a_P : array-like
        Probability occurrence of F1 layer.

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
    # -------------------------------------------------------------------------
    # Simplified constant value from Bilitza et al. (2022)
    gamma = 2.36

    a_P = (0.5 + 0.5 * np.cos(np.deg2rad(solzen)))**gamma

    return a_P
