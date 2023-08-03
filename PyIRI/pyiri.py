#!/usr/bin/env python
# --------------------------------------------------------
# Distribution statement A. Approved for public release.
# Distribution is unlimited.
# This work was supported by the Office of Naval Research.
# --------------------------------------------------------
"""General interface with examples of how to use PyIRI.

References
----------
.. [1] Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
International Reference Ionosphere Modeling Implemented in Python,
Space Weather.
.. [2] Bilitza et al. (2022), The International Reference Ionosphere
model: A review and description of an ionospheric benchmark, Reviews
of Geophysics, 60, e2022RG000792. https://doi.org/10.1029/2022RG000792
.. [3] Nava et al. (2008). A new version of the nequick ionosphere
electron density model. J. Atmos. Sol. Terr. Phys., 70 (15),
490 doi: 10.1016/j.jastp.2008.01.015
.. [4] Jones, W. B., Graham, R. P., & Leftin, M. (1966). Advances
in ionospheric mapping 476 by numerical methods.

"""

import numpy as np

from PyIRI import coeff_dir
import PyIRI.main_library as ml
import PyIRI.plotting as plot


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
    grid_loc : tuple
        Tuple of grid locations, which are numpy arrays
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
    create_reg_grid output is `grid_loc`

    """
    # Define the grids
    grid_loc = create_reg_grid(hr_res=hr_res, lat_res=lat_res, lon_res=lon_res,
                               alt_res=alt_res, alt_min=alt_min,
                               alt_max=alt_max)
    alon, alat, _, _, aalt, ahr = grid_loc

    # -------------------------------------------------------------------------
    # Monthly mean density for min and max of solar activity:
    # -------------------------------------------------------------------------
    # The original IRI model further interpolates between 2 levels of solar
    # activity to estimate density for a particular level of F10.7.
    # Additionally, it interpolates between 2 consecutive months to make a
    # smooth seasonal transition. Here is an example of how this interpolation
    # can be done. If you need to run IRI for a particular day, you can just
    # use this function
    f2, f1, epeak, es_peak, sun, mag, edens_prof = ml.IRI_density_1day(
        year, month, day, ahr, alon, alat, aalt, f107, coeff_dir, ccir_or_ursi)

    return grid_loc, f2, f1, epeak, es_peak, sun, mag, edens_prof


def run_seas_iri_reg_grid(year, month, day, hr_res=1, lat_res=1, lon_res=1,
                          alt_res=10, alt_min=0, alt_max=700, ccir_or_ursi=0):
    """Run IRI for a single day on a regular grid with no solar activity.

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
    grid_loc : tuple
        Tuple of grid locations, which are numpy arrays
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
    create_reg_grid output is `grid_loc`

    """
    # Define the grids
    grid_loc = create_reg_grid(hr_res=hr_res, lat_res=lat_res, lon_res=lon_res,
                               alt_res=alt_res, alt_min=alt_min,
                               alt_max=alt_max)
    alon, alat, _, _, aalt, ahr = grid_loc

    # -------------------------------------------------------------------------
    # Monthly mean ionospheric parameters for min and max of solar activity:
    # -------------------------------------------------------------------------
    # This is how PyIRI needs to be called to obtain montly mean values for all
    # ionospheric parameters, for min and max conditions of solar activity:
    # year and month should be integers, and ahr and alon, alat should be 1-D
    # NumPy arrays. alon and alat should have the same size. coeff_dir is the
    # coefficient directory. Matrix size for all the output parameters is
    # [N_T, N_G, 2], where 2 indicates min and max of the solar activity that
    # corresponds to Ionospheric Global (IG) index levels of 0 and 100.

    f2, f1, epeak, es_peak, sun, mag = ml.IRI_monthly_mean_par(
        year, month, ahr, alon, alat, coeff_dir, ccir_or_ursi)

    # -------------------------------------------------------------------------
    # Montly mean density for min and max of solar activity:
    # -------------------------------------------------------------------------
    # Construct electron dnesity profiles for min and max levels of solar
    # activity for monthly mean parameters.  The result will have the following
    # dimensions [2, N_T, N_V, N_G]
    edens_prof = ml.reconstruct_density_from_parameters(f2, f1, epeak, aalt)

    return grid_loc, f2, f1, epeak, es_peak, sun, mag, edens_prof


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

    """
    # ------------------------------------------------------------------------
    # Define spatial horizontal grid:
    # ------------------------------------------------------------------------
    # This function creates a regular global grid in Geographic coordinates
    # using given spatial resolution lon_res, lat_res. In case you want to run
    # IRI in magnetic coordinates, you can obtain a regular grid in magnetic
    # coordinates, then convert it to geographic coordinates using your own
    # methods, and then use these 1-D alon and alat arrays. Similarly, if you
    # are interested in regional grid, then use your own alon and alat 1-D
    # arrays. alon and alat can also be irregular arrays. In case you need to
    # run IRI for 1 grid point, define alon and alat as NumPy arrays that have
    # only 1 element. Size of alon and alat is [N_G]
    alon, alat, alon_2d, alat_2d = ml.set_geo_grid(lon_res, lat_res)

    # -------------------------------------------------------------------------
    # Altitudinal array to build electron density:
    # -------------------------------------------------------------------------
    # Create an array of altitudes for the veritical dimension of electron
    # density profiles. Any 1-D Numpy array in [km] would work, regularly or
    # irregularly spaced.
    aalt = np.arange(alt_min, alt_max, alt_res)

    # -------------------------------------------------------------------------
    # Define time arrays:
    # -------------------------------------------------------------------------
    # This function creates time array aUT using a given temporal resolution
    # hr_res in hours. E.g. if hr_res = 0.25 it will lead to 15-minutes time
    # resolution. You can define aUT your own way, just keep it 1-D and
    # expressed in hours. It can be regularly or irregularly spaced array. If
    # you want to run PyIRI for just 1 time frame, then define aUT as NumPy
    # arrays that have only 1 element. Size of aUT is [N_T]
    ahr, _, _, _, _ = ml.set_temporal_array(hr_res)

    return alon, alat, alon_2d, alat_2d, aalt, ahr


def run_diagnostic_plots(plot_dir, plot_hr, alon, alat, alon_2d, alat_2d, aalt,
                         ahr, f2, f1, epeak, es_peak, sun, mag, edens_prof):
    """Create a standard set of plots useful for understanding IRI.

    Parameters
    ----------
    plot_dir : str
        Output plot directory (standard plot names used, will be overwritten)
    plot_hr : int or float
        UT hour to plot
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

    """
    # Plot all the parameters using PyIRI_Plotting_Library
    plot.PyIRI_plot_mag_dip_lat(mag, alon, alat, alon_2d, alat_2d, plot_dir)
    plot.PyIRI_plot_modip(mag, alon, alat, alon_2d, alat_2d, plot_dir)
    plot.PyIRI_plot_inc(mag, alon, alat, alon_2d, alat_2d, plot_dir)
    plot.PyIRI_plot_B_F1_bot_min_max(f1, ahr, alon, alat, alon_2d, alat_2d,
                                     sun, plot_hr, plot_dir)
    plot.PyIRI_plot_B_F2_bot_min_max(f2, ahr, alon, alat, alon_2d, alat_2d,
                                     sun, plot_hr, plot_dir)
    plot.PyIRI_plot_B_F2_top_min_max(f2, ahr, alon, alat, alon_2d, alat_2d,
                                     sun, plot_hr, plot_dir)
    plot.PyIRI_plot_M3000_min_max(f2, ahr, alon, alat, alon_2d, alat_2d,
                                  sun, plot_hr, plot_dir)
    plot.PyIRI_plot_hmF2_min_max(f2, ahr, alon, alat, alon_2d, alat_2d,
                                 sun, plot_hr, plot_dir)
    plot.PyIRI_plot_hmF1_min_max(f1, ahr, alon, alat, alon_2d, alat_2d,
                                 sun, plot_hr, plot_dir)
    plot.PyIRI_plot_foF1_min_max(f1, ahr, alon, alat, alon_2d, alat_2d,
                                 sun, plot_hr, plot_dir)
    plot.PyIRI_plot_foE_min_max(epeak, ahr, alon, alat, alon_2d, alat_2d,
                                sun, plot_hr, plot_dir)
    plot.PyIRI_plot_foEs_min_max(es_peak, ahr, alon, alat, alon_2d, alat_2d,
                                 sun, plot_hr, plot_dir)
    plot.PyIRI_plot_foF2_min_max(f2, ahr, alon, alat, alon_2d, alat_2d,
                                 sun, plot_hr, plot_dir)
    plot.PyIRI_plot_NmF2_min_max(f2, ahr, alon, alat, alon_2d, alat_2d,
                                 sun, plot_hr, plot_dir)

    # Plot EDP at one location using PyIRI_Plotting_Library
    plot.PyIRI_EDP_sample(edens_prof, ahr, alon, alat, alon_2d, alat_2d, aalt,
                          plot_hr, plot_dir)
    return
