#!/usr/bin/env python
# --------------------------------------------------------
# Distribution statement A. Approved for public release.
# Distribution is unlimited.
# This work was supported by the Office of Naval Research.
# --------------------------------------------------------
"""This library contains Apex components for PyIRI software.

The Apex coordinates were estimated for each year from
1900 to 2025 using ApexPy. These results were converted
to the spherical harmonic coefficients up to lmax=20 and
saved in the .nc file PyIRI.coeff_dir / 'Apex' / 'Apex.nc'.

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

import numpy as np
import os

import netCDF4 as nc
import opt_einsum as oe
import scipy.special as ss

import PyIRI
from PyIRI import logger
from PyIRI import main_library as ml


def Apex(Lat, Lon, dtime, transform_type):
    """Convert between geographic (GEO) and quasi-dipole (QD) coordinates.

    Parameters
    ----------
    Lat : array-like
        Geographic or quasi-dipole latitude grid [deg].
    Lon : array-like
        Geographic or quasi-dipole longitude grid [deg].
    dtime : datetime.datetime
        Datetime object specifying the target year for Apex coefficients.
    transform_type : str
        Transformation type:
            'GEO_2_QD' — convert from geographic to quasi-dipole
            'QD_2_GEO' — convert from quasi-dipole to geographic

    Returns
    -------
    tuple of np.ndarray
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

        # GEO → QD
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

        # QD → GEO
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
        Shape (N_T, N_G) if grids are converted to magnetic from geographic
        coordinates, (N_G,) if grids are originally in magnetic coordinates
    phi : array-like
        User input longitudes [0-2π) [rad].
        Shape (N_T, N_G) if input grids were converted to magnetic from
        geographic coordinates, (N_G,) if grids were originally in magnetic
        coordinates
    lmax : int
        Maximum spherical harmonic degree (and order). (default=29)

    Returns
    -------
    F_SH : array-like
        Real-valued spherical harmonic basis matrix, with N_SH=(lmax+1)**2.
        Each column corresponds to a pair of coordinates, each row to an SH
        mode. The SH mode of degree l and order m is stored in row i=l*(l+1)+m.
        Shape (N_SH, N_T, N_G) if input grids were converted to magnetic from
        geographic coordinates, (N_SH, N_G) if grids were originally in
        magnetic coordinates

    """
    # Convert inputs to arrays if lists or int or float
    theta = ml.to_numpy_array(theta)
    phi = ml.to_numpy_array(phi)

    # If theta has shape (n_pos,), i.e., if user input coord='mlt', theta and
    # phi arrays must be artificially expanded to shape (1, n_pos)
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

    # If N_T=1, i.e., user input coord='mlt', then we can get rid of the N_T
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
        Number of complex Fourier coefficients to use (i.e., truncation level).
        (default=5) The associated number of real Fourier coefficients is
        N_FS_r=2*N_FS_c-1.

    Returns
    -------
    F_FS : array-like
        Real-valued FS basis matrix.
        Shape (N_T, N_FS_r)

    """
    # -------------------------------------------------------------------------
    # Convert inputs to Numpy arrays
    aUT = ml.to_numpy_array(aUT)

    N_T = aUT.size
    N_FS_r = 2 * N_FS_c - 1
    F_FS = np.empty((N_T, N_FS_r))

    k_vals = np.arange(1, N_FS_c)
    omega = 2 * np.pi * k_vals / 24
    phase = np.outer(aUT, omega)

    F_FS[:, 0] = 1
    F_FS[:, 1::2] = np.cos(phase)
    F_FS[:, 2::2] = np.sin(phase)

    return F_FS
