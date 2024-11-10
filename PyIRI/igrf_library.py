#!/usr/bin/env python
# ----------------------------------------------------------
# Distribution statement A. Approved for public release.
# Distribution is unlimited.
# This work was supported by the Office of Naval Research.
# ----------------------------------------------------------
"""This library contains components for IGRF.

References
----------
Alken et al. (2021). International geomagnetic reference field: the
thirteenth generation. Earth, Planets and Space, 73(49),
doi:10.1186/s40623-020-01288-x.

"""

import numpy as np
import os
from scipy import interpolate


def verify_inclination(inc):
    """Test the validity of an inclination input.

    Parameters
    ----------
    inc : int, float, or array-like
        Magnetic inclination in degrees.

    Returns
    -------
    inc : int, float, or array-like
        Magnetic inclination in degrees

    Raises
    ------
    ValueError
        If inclination is not between -90 and 90 degrees

    """
    # Ensure the inclination to be tested is a numpy array
    test_inc = np.asarray(inc)
    if test_inc.shape == ():
        test_inc = np.asarray([inc])

    # Get a mask that specifies if each inclination is realistic
    good_inc = (test_inc <= 90.0) & (test_inc >= -90.0)

    # Return original input if all are good, correct or raise error otherwise
    if good_inc.all():
        return inc
    else:
        # Adjust values within a small tolerance
        test_inc[(test_inc > 90.0) & (test_inc <= 90.1)] = 90.0
        test_inc[(test_inc < -90.0) & (test_inc >= -90.1)] = -90.0
        good_inc = (test_inc <= 90.0) & (test_inc >= -90.0)

        if good_inc.all():
            # Return the adjusted input using the original type
            if isinstance(inc, (float, int, np.floating, np.integer)):
                return test_inc[0].astype(type(inc))
            elif isinstance(inc, list):
                return list(test_inc)
            else:
                return test_inc
        else:
            raise ValueError('unrealistic magnetic inclination value(s)')


def inc2modip(inc, alat):
    """Calculate modified dip angle from magnetic inclination.

    Parameters
    ----------
    inc : array-like
        Magnetic inclination in degrees.
    alat : array-like
        Flattened array of latitudes in degrees.

    Returns
    -------
    modip_deg : array-like
        Modified dip angle in degrees.

    Notes
    -----
    This function calculates modified dip angle for a given inclination and
    geographic latitude.

    """
    inc = verify_inclination(inc)

    alat_rad = np.deg2rad(alat)
    rad_arg = np.deg2rad((inc / np.sqrt(np.cos(alat_rad))))
    modip_rad = np.arctan(rad_arg)
    modip_deg = np.rad2deg(modip_rad)

    return modip_deg


def inc2magnetic_dip_latitude(inc):
    """Calculate magnetic dip latitude from magnetic inclination.

    Parameters
    ----------
    inc : array-like
        Magnetic inclination in degrees.

    Returns
    -------
    magnetic_dip_latitude : array-like
        Magnetic dip latitude in degrees.

    Notes
    -----
    This function calculates magnetic dip latitude.

    """
    inc = verify_inclination(inc)
    arg = 0.5 * (np.tan(np.deg2rad(inc)))
    magnetic_dip_latitude = np.rad2deg(np.arctan(arg))

    return magnetic_dip_latitude


def inclination(coeff_dir, date_decimal, alon, alat):
    """Calculate magnetic inclination using IGRF13.

    Parameters
    ----------
    coeff_dir : str
        Where IGRF13 coefficients are located.
    date_decimal : float
        Decimal year
    alon : array-like
        Flattened array of geographic longitudes in degrees.
    alat : array-like
        Flattened array of geographic latitudes in degrees.

    Returns
    -------
    inc : array-like
        Magnetic inclination in degrees.

    Raises
    ------
    IOError
        If unable to find IGRF coefficient file

    Notes
    -----
    This code is a slight modification of the IGRF13 pyIGRF release,
    https://www.ngdc.noaa.gov/IAGA/vmod/igrf.html
    The reading of the IGRF coefficient file was modified to speded up the
    process, and the main code was simlified to oly focus on the iclination
    of magnetic field output for the given grid.

    """
    # Set altitude to 300 km
    aalt = np.full(shape=alon.shape, fill_value=300.0)

    # Open IGRF file and read to array the main table
    igrf_file = os.path.join(coeff_dir, 'IGRF', 'IGRF13.shc')
    if not os.path.isfile(igrf_file):
        raise IOError('unable to find IGRF coefficient file: {:}'.format(
            igrf_file))

    with open(igrf_file, mode='r') as fopen:
        file_array = np.genfromtxt(fopen, delimiter='', skip_header=5)

    # Create the time array of years that match the years in the file
    # (check if file is change to the newer version)
    igrf_time = np.arange(1900, 2025 + 5, 5)

    # Exclude first 2 columns, these are the m and n indecies
    igrf_coeffs = file_array[:, 2:]

    # Maximum degree of polynomials
    nmax = 13

    # Colatitude (from 0 to 180)
    colat = 90.0 - alat

    # Compute geocentric colatitude and radius from geodetic colatitude
    # and height
    alt, colat, bx_rot, bz_rot = gg_to_geo(aalt, colat)

    # Interpolate coefficients from 5 year slots to given decimal year
    igrf_extrap = interpolate.interp1d(igrf_time, igrf_coeffs,
                                       fill_value='extrapolate')
    coeffs = igrf_extrap(date_decimal)

    # Compute the main field B_r, B_theta and B_phi value for the location(s)
    br, bt, bp = synth_values(coeffs.T, alt, colat, alon, nmax)

    # Rearrange to X, Y, Z components
    x_comp = -bt
    y_comp = bp
    z_comp = -br

    # Rotate back to geodetic coords if needed
    t_comp = x_comp
    x_comp = x_comp * bz_rot + z_comp * bx_rot
    z_comp = z_comp * bz_rot - t_comp * bx_rot

    # Compute the four non-linear components
    dec, hoz, inc, eff = xyz2dhif(x_comp, y_comp, z_comp)

    # Return only inclination because that is what we need for PyIRI
    return inc


def gg_to_geo(h, gdcolat):
    """Compute geocentric colatitude and radius from geodetic colat and height.

    Parameters
    ----------
    h : array-like
        Altitude in kilometers.
    gdcolat : array-like
        Geodetic colatitude in degrees.

    Returns
    -------
    rad : array-like
        Geocentric radius in kilometers.
    thc : array-like
        Geocentric colatitude in degrees.
    sd : array-like
        Rotate B_X to gd_lat.
    cd : array-like
        Rotate B_Z to gd_lat.

    Notes
    -----
    IGRF-13 Alken et al., 2021.

    References
    ----------
    Equations (51)-(53) from "The main field" (chapter 4) by Langel,
    R. A. in: "Geomagnetism", Volume 1, Jacobs, J. A., Academic Press,
    1987.

    Malin, S.R.C. and Barraclough, D.R., 1981. An algorithm for
    synthesizing the geomagnetic field. Computers & Geosciences, 7(4),
    pp. 401-405.

    """
    # Use WGS-84 ellipsoid parameters
    eqrad = 6378.137  # equatorial radius
    flat = 1. / 298.257223563
    plrad = eqrad * (1. - flat)  # polar radius
    ctgd = np.cos(np.deg2rad(gdcolat))
    stgd = np.sin(np.deg2rad(gdcolat))
    a2 = eqrad * eqrad
    a4 = a2 * a2
    b2 = plrad * plrad
    b4 = b2 * b2
    c2 = ctgd * ctgd
    s2 = 1. - c2
    rho = np.sqrt(a2 * s2 + b2 * c2)
    rad = np.sqrt(h * (h + 2 * rho) + (a4 * s2 + b4 * c2) / rho**2)
    cd = (h + rho) / rad
    sd = (a2 - b2) * ctgd * stgd / (rho * rad)
    cthc = ctgd * cd - stgd * sd
    thc = np.rad2deg(np.arccos(cthc))  # arccos returns values in [0, pi]

    return rad, thc, sd, cd


def geo_to_gg(radius, theta):
    """Compute geodetic colatitude and vertical height above the ellipsoid.

    Parameters
    ----------
    radius : array-like
        Geocentric radius in kilometers.
    theta : array-like
        Geocentric colatitude in degrees.

    Returns
    -------
    height : array-like
        Altitude in kilometers.
    beta : array-like
        Geodetic colatitude.

    Notes
    -----
    IGRF-13 Alken et al., 2021.
    Compute geodetic colatitude and vertical height above the ellipsoid from
    geocentric radius and colatitude. Round-off errors might lead to a
    failure of the algorithm especially but not exclusively for points close
    to the geographic poles. Corresponding geodetic coordinates are returned
    as NaN.

    References
    ----------
    Zhu, J., "Conversion of Earth-centered Earth-fixed coordinates to
    geodetic coordinates", IEEE Transactions on Aerospace and Electronic
    Systems}, 1994, vol. 30, num. 3, pp. 957-961.

    """
    # Use WGS-84 ellipsoid parameters
    a = 6378.137  # equatorial radius
    b = 6356.752  # polar radius
    a2 = a**2
    b2 = b**2
    e2 = (a2 - b2) / a2  # squared eccentricity
    e4 = e2 * e2
    ep2 = (a2 - b2) / b2  # squared primed eccentricity
    r = radius * np.sin(np.radians(theta))
    z = radius * np.cos(np.radians(theta))
    r2 = r**2
    z2 = z**2
    F = 54. * b2 * z2
    G = r2 + (1. - e2) * z2 - e2 * (a2 - b2)
    c = e4 * F * r2 / G**3
    s = (1. + c + np.sqrt(c**2 + 2 * c))**(1. / 3.)
    P = F / (3 * (s + 1. / s + 1.)**2 * G**2)
    Q = np.sqrt(1. + 2 * e4 * P)
    r0 = (-P * e2 * r / (1. + Q)
          + np.sqrt(0.5 * a2 * (1. + 1. / Q) - P * (1. - e2) * z2
                    / (Q * (1. + Q)) - 0.5 * P * r2))
    U = np.sqrt((r - e2 * r0)**2 + z2)
    V = np.sqrt((r - e2 * r0)**2 + (1. - e2) * z2)
    z0 = b2 * z / (a * V)
    height = U * (1. - b2 / (a * V))
    beta = 90. - np.degrees(np.arctan2(z + ep2 * z0, r))

    return height, beta


def synth_values(coeffs, radius, theta, phi, nmax=None, nmin=1, grid=False):
    """Compute radial, colatitude and azimuthal field components.

    Parameters
    ----------
    coeffs : array-like
        Coefficients of the spherical harmonic expansion. The last
        dimension is equal to the number of coefficients, `N` at the grid
        points.
    radius : array-like
        Array containing the radius in km.
    theta : array-like
        Array containing the colatitude in degrees.
    phi : array-like
        Array containing the longitude in degrees.
    nmax : int or NoneType
        Maximum degree up to which expansion is to be used, if None it will be
        specified by the ``coeffs`` variable.  However, smaller values are
        also valid if specified. (default=None)
    nmin : int
        Minimum degree from which expansion is to be used. Note that it will
        just skip the degrees smaller than ``nmin``, the whole sequence of
        coefficients 1 through ``nmax`` must still be given in ``coeffs``.
        (default=1)
    grid : bool
        If ``True``, field components are computed on a regular grid. Arrays
        ``theta`` and ``phi`` must have one dimension less than the output
        grid since the grid will be created as their outer product
        (default=False).

    Returns
    -------
    B_radius : array-like
        Radial field components in km.
    B_theta : array-like
        Colatitude field components in degrees.
    B_phi : array-like
        Azimuthal field components in degrees.

    Raises
    ------
    ValueError
        If an inappropriate input value is supplied

    Notes
    -----
    by IGRF-13 Alken et al., 2021
    Based on chaosmagpy from Clemens Kloss (DTU Space, Copenhagen)
    Computes radial, colatitude and azimuthal field components from the
    magnetic potential field in terms of spherical harmonic coefficients.
    A reduced version of the DTU synth_values chaosmagpy code.

    References
    ----------
    Zhu, J., "Conversion of Earth-centered Earth-fixed coordinates to
    geodetic coordinates", IEEE Transactions on Aerospace and Electronic
    Systems}, 1994, vol. 30, num. 3, pp. 957-961.

    """
    # Ensure ndarray inputs
    coeffs = np.array(coeffs, dtype=float)
    radius = np.array(radius, dtype=float) / 6371.2  # Earth's average radius
    theta = np.array(theta, dtype=float)
    phi = np.array(phi, dtype=float)
    if (np.amin(theta) < 0.0) | (np.amax(theta) > 180.0):
        raise ValueError('Colatitude outside bounds [0, 180].')

    if nmin < 1:
        raise ValueError('Only positive nmin allowed.')

    # Handle optional argument: nmax
    nmax_coeffs = int(np.sqrt(coeffs.shape[-1] + 1) - 1)  # degrees
    if nmax is None:
        nmax = nmax_coeffs
    elif nmax <= 0:
        raise ValueError('Only positive, non-zero nmax allowed.')

    if nmax > nmax_coeffs:
        nmax = nmax_coeffs

    if nmax < nmin:
        raise ValueError(f'Nothing to compute: nmax < nmin ({nmax} < {nmin}.)')

    # Manually broadcast input grid on surface
    if grid:
        theta = theta[..., None]  # First dimension is theta
        phi = phi[None, ...]  # Second dimension is phi

    # Get shape of broadcasted result
    try:
        broad = np.broadcast(radius, theta, phi,
                             np.broadcast_to(0, coeffs.shape[:-1]))
    except ValueError:
        raise ValueError(''.join(['Cannot broadcast grid shapes (excl. last ',
                                  'dimension of coeffs):\nradius: ',
                                  repr(radius.shape), '\ntheta: ',
                                  repr(theta.shape), '\nphi: ', repr(phi.shape),
                                  '\ncoeffs: ', repr(coeffs.shape)]))
    grid_shape = broad.shape

    # Initialize radial dependence given the source
    r_n = radius**(-(nmin + 2))

    # Compute associated Legendre polynomials as (n, m, theta-points)-array
    Pnm = legendre_poly(nmax, theta)

    # Save sinth for fast access
    sinth = Pnm[1, 1]

    # Calculate cos(m*phi) and sin(m*phi) as (m, phi-points)-array
    phi = np.radians(phi)
    cos_mp = np.cos(np.multiply.outer(np.arange(nmax + 1), phi))
    sin_mp = np.sin(np.multiply.outer(np.arange(nmax + 1), phi))

    # allocate arrays in memory
    B_radius = np.zeros(grid_shape)
    B_theta = np.zeros(grid_shape)
    B_phi = np.zeros(grid_shape)
    num = nmin**2 - 1
    for n in range(nmin, nmax + 1):
        B_radius += (n + 1) * Pnm[n, 0] * r_n * coeffs[..., num]
        B_theta += -Pnm[0, n + 1] * r_n * coeffs[..., num]
        num += 1
        for m in range(1, n + 1):
            B_radius += ((n + 1) * Pnm[n, m] * r_n
                         * (coeffs[..., num] * cos_mp[m]
                            + coeffs[..., num + 1] * sin_mp[m]))
            B_theta += (-Pnm[m, n + 1] * r_n
                        * (coeffs[..., num] * cos_mp[m]
                           + coeffs[..., num + 1] * sin_mp[m]))
            with np.errstate(divide='ignore', invalid='ignore'):
                # Handle poles using L'Hopital's rule
                div_Pnm = np.where(theta == 0., Pnm[m, n + 1],
                                   Pnm[n, m] / sinth)
                div_Pnm = np.where(theta == np.degrees(np.pi),
                                   -Pnm[m, n + 1], div_Pnm)
            B_phi += (m * div_Pnm * r_n
                      * (coeffs[..., num] * sin_mp[m]
                         - coeffs[..., num + 1] * cos_mp[m]))
            num += 2
        r_n = r_n / radius  # Equivalent to r_n = radius**(-(n+2))

    return B_radius, B_theta, B_phi


def legendre_poly(nmax, theta):
    r"""Calculate associated Legendre polynomials `P(n,m)`.

    Parameters
    ----------
    nmax : int
        Maximum degree up to which expansion, with a minimum value of one.
    theta : array-like
        Array containing the colatitude in degrees.

    Returns
    -------
    Pnm : array-like
        Evaluated values and derivatives.

    Raises
    ------
    IndexError
        If nmax is less than 1

    Notes
    -----
    by IGRF-13 Alken et al., 2021
    Returns associated Legendre polynomials `P(n,m)` (Schmidt
    quasi-normalized), and the derivative :math:`dP(n,m)/d\\theta` evaluated
    at :math:`\\theta`.

    References
    ----------
    Langel, R. A. "Chapter four: Main field." Geomagnetism, edited by JA Jacobs
    1 (1987).
    Zhu, J., "Conversion of Earth-centered Earth-fixed coordinates to
    geodetic coordinates", IEEE Transactions on Aerospace and Electronic
    Systems}, 1994, vol. 30, num. 3, pp. 957-961.

    """
    costh = np.cos(np.radians(theta))
    sinth = np.sqrt(1 - costh**2)
    Pnm = np.zeros((nmax + 1, nmax + 2) + costh.shape)
    Pnm[0, 0] = 1  # is copied into trailing dimenions
    Pnm[1, 1] = sinth  # write theta into trailing dimenions via broadcasting
    rootn = np.sqrt(np.arange(2 * nmax**2 + 1))

    # Recursion relations after Langel "The Main Field" (1987),
    # eq. (27) and Table 2 (p. 256)
    for m in range(nmax):
        Pnm_tmp = rootn[m + m + 1] * Pnm[m, m]
        Pnm[m + 1, m] = costh * Pnm_tmp
        if m > 0:
            Pnm[m + 1, m + 1] = sinth * Pnm_tmp / rootn[m + m + 2]
        for n in np.arange(m + 2, nmax + 1):
            d = n * n - m * m
            e = n + n - 1
            Pnm[n, m] = ((e * costh * Pnm[n - 1, m]
                          - rootn[d - e] * Pnm[n - 2, m]) / rootn[d])

    # dP(n,m) = Pnm(m,n+1) is the derivative of P(n,m) vrt. theta
    Pnm[0, 2] = -Pnm[1, 1]
    Pnm[1, 2] = Pnm[1, 0]
    for n in range(2, nmax + 1):
        n = int(n)
        Pnm[0, n + 1] = - np.sqrt((n * n + n) / 2.) * Pnm[n, 1]
        Pnm[1, n + 1] = ((np.sqrt(2.0 * (n * n + n)) * Pnm[n, 0]
                          - np.sqrt((n * n + n - 2)) * Pnm[n, 2]) / 2.0)
        for m in np.arange(2, n):
            m = int(m)
            Pnm_part1 = np.sqrt((n + m) * (n - m + 1)) * Pnm[n, m - 1]
            Pnm_part2 = np.sqrt((n + m + 1.0) * (n - m)) * Pnm[n, m + 1]
            Pnm[m, n + 1] = 0.5 * Pnm_part1 - Pnm_part2
        Pnm[n, n + 1] = np.sqrt(2. * n) * Pnm[n, n - 1] / 2.0

    return Pnm


def xyz2dhif(x, y, z):
    """Calculate declination, intensity, inclination of mag field.

    Parameters
    ----------
    x : array-like
        North component of the magnetic field in nT.
    y : array-like
        East component of the magnetic field in nT.
    y : array-like
        Vertical component of the magnetic field in nT.

    Returns
    -------
    dec : array-like
        Declination of the magnetic field in degrees.
    hoz : array-like
        Horizontal intensity of the magnetic field in nT.
    inc : array-like
        Inclination of the magnetic field in degrees.
    eff : array-like
        Total intensity of the magnetic filed in nT.

    Notes
    -----
    by IGRF-13 Alken et al., 2021
    Calculate D, H, I and F from (X, Y, Z)
    Based on code from D. Kerridge, 2019.

    """
    hsq = x * x + y * y
    hoz = np.sqrt(hsq)
    eff = np.sqrt(hsq + z * z)
    dec = np.rad2deg(np.arctan2(y, x))
    inc = np.rad2deg(np.arctan2(z, hoz))

    return dec, hoz, inc, eff
