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
# import datetime as dt
# import netCDF4 as nc
# import numpy as np
# import opt_einsum as oe
# import os
# import pandas as pd
# import PyIRI
# import PyIRI.edp_update as edpup
# import PyIRI.igrf_library as igrf
# import PyIRI.main_library as main
# import scipy.special as ss


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
