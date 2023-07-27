###############################################################################
# Distribution statement A. Approved for public release. Distribution is
# unlimited.
#
#      This work was supported by the Office of Naval Research
###############################################################################

import numpy as np
from scipy import interpolate

#*******************************************************************************
# by Victoriya V Forsythe Makarevich
#
# Naval Research Laboratory 
# Space Physics 
#
# Date: 11.07.2022
#
# This code calculates latitude for a given inclination and modip
#
# Variables: inc = inclination of the magnetic field, [ngrid], result of inclination(dir, date_decimal, alon, alat, aalt)
#            modip = modified dip anle [ngrid]
#
# Result:    alat = np array of latitudes [ngrid]
#-------------------------------------------------------------------------------
def modip2lat(inc, modip):
#-------------------------------------------------------------------------------
    #degree to radians convertion 
    DR=np.pi/180. #degree to radian conversion factor
    RD=1./DR #radian to dergree conversion factor
    
    # Modified dip angle
    alat = np.arccos((inc/np.tan(modip*DR))**2*DR)*RD
#------------------------------------------------------------------------------------
    return(alat)
#------------------------------------------------------------------------------------
#************************************************************************************
#************************************************************************************
# by Victoriya V Forsythe Makarevich
#
# Naval Research Laboratory 
# Space Physics 
#
# Date: 11.07.2022
#
# This code calculates modified dip angle for a given inclination and latitude
#
# Variables: inc = inclination of the magnetic field, [ngrid], result of inclination(dir, date_decimal, alon, alat, aalt)
#            alat = np array of latitudes [ngrid]
#
# Result:    modip = modified dip anle [ngrid]
#------------------------------------------------------------------------------------
def inc2modip(inc, alat):
#------------------------------------------------------------------------------------    
    #degree to radians convertion 
    DR=np.pi/180. #degree to radian conversion factor
    RD=1./DR #radian to dergree conversion factor
    
    # Modified dip angle
    modip = np.arctan((inc/np.sqrt(np.cos(alat*DR)))*DR)*RD
#------------------------------------------------------------------------------------
    return(modip)
#------------------------------------------------------------------------------------
#************************************************************************************
#************************************************************************************
# by Victoriya V Forsythe Makarevich
#
# Naval Research Laboratory 
# Space Physics 
#
# Date: 07.05.2023
#
# This code calculates magnetic dip latitude, from Bilitza review 2022, Equation 41 
#
# Variables: inc = inclination of the magnetic field, [ngrid], result of inclination(dir, date_decimal, alon, alat, aalt)
#
# Result:    magnetic_dip_latitude  = magnetic dip latitude  [ngrid]
#------------------------------------------------------------------------------------
def inc2magnetic_dip_latitude(inc):
#------------------------------------------------------------------------------------    
    magnetic_dip_latitude = np.rad2deg(np.arctan(0.5*(np.tan(np.deg2rad(inc)))))
#------------------------------------------------------------------------------------
    return(magnetic_dip_latitude)
#------------------------------------------------------------------------------------
#************************************************************************************
#************************************************************************************
# by Victoriya V Forsythe Makarevich
#
# Naval Research Laboratory 
# Space Physics 
#
# Date: 11.07.2022
#
# This code is a slight modification of the IGRF13 pyIGRF release, https://www.ngdc.noaa.gov/IAGA/vmod/igrf.html
# The reading of the IGRF coefficient file was modified to speded up the process, and the main code was simlified to 
# oly focus on the iclination of magnetic field output for the given grid
#
# Variables: date_decimal = decimal year, e.g. 2020.11 
#            alon = np array of longitudes [ngrid]
#            alat = np array of latitudes [ngrid]
#            aalt = np array of altitudes [ngrid]
#
# Result:    inc = inclination of the magnetic field in degrees [ngrid]
#------------------------------------------------------------------------------------
def inclination(dir, date_decimal, alon, alat):
#------------------------------------------------------------------------------------
    #300 km altitude
    aalt=np.zeros((alon.shape))+300.
    
    #open IGRF file and read to array the main table
    IGRF_FILE = dir+'IGRF/IGRF13.shc'
    f=open(IGRF_FILE, mode='r')
    file_array = np.genfromtxt(f, delimiter='', skip_header=5)
    f.close()

    #create the time array of years that match the years in the file (check if file is change to the newer version)
    igrf_time=np.arange(1900, 2025+5, 5)

    #exclude first 2 columns, these are the m & n indecies
    igrf_coeffs=file_array[:,2:]

    #maximum degree of polynomials
    nmax=13

    #colatitude (from 0 to 180)
    colat=90-alat

    #Compute geocentric colatitude and radius from geodetic colatitude and height
    alt, colat, sd, cd = gg_to_geo(aalt, colat)

    #interpolate coefficients from 5 year slots to given decimal year
    f = interpolate.interp1d(igrf_time, igrf_coeffs, fill_value='extrapolate')
    coeffs = f(date_decimal)

    # Compute the main field B_r, B_theta and B_phi value for the location(s) 
    Br, Bt, Bp = synth_values(coeffs.T, alt, colat, alon, nmax)

    # Rearrange to X, Y, Z components 
    X = -Bt; Y = Bp; Z = -Br

    # Rotate back to geodetic coords if needed
    t = X; X = X*cd + Z*sd;  Z = Z*cd - t*sd

    # Compute the four non-linear components 
    dec, hoz, inc, eff = xyz2dhif(X,Y,Z)
    
    #return only inclanation because that is what we need for NIDA
    return(inc)

#************************************************************************************
#************************************************************************************
def gg_to_geo(h, gdcolat):
    """
    Compute geocentric colatitude and radius from geodetic colatitude and
    height.

    Parameters
    ----------
    h : ndarray, shape (...)
        Altitude in kilometers.
    gdcolat : ndarray, shape (...)
        Geodetic colatitude

    Returns
    -------
    radius : ndarray, shape (...)
        Geocentric radius in kilometers.
    theta : ndarray, shape (...)
        Geocentric colatitude in degrees.
    
    sd : ndarray shape (...) 
        rotate B_X to gd_lat 
    cd :  ndarray shape (...) 
        rotate B_Z to gd_lat 

    References
    ----------
    Equations (51)-(53) from "The main field" (chapter 4) by Langel, R. A. in:
    "Geomagnetism", Volume 1, Jacobs, J. A., Academic Press, 1987.
    
    Malin, S.R.C. and Barraclough, D.R., 1981. An algorithm for synthesizing 
    the geomagnetic field. Computers & Geosciences, 7(4), pp.401-405.

    """
    # Use WGS-84 ellipsoid parameters

    eqrad = 6378.137 # equatorial radius
    flat  = 1/298.257223563
    plrad = eqrad*(1-flat) # polar radius
    ctgd  = np.cos(np.deg2rad(gdcolat))
    stgd  = np.sin(np.deg2rad(gdcolat))
    a2    = eqrad*eqrad
    a4    = a2*a2
    b2    = plrad*plrad
    b4    = b2*b2
    c2    = ctgd*ctgd
    s2    = 1-c2
    rho   = np.sqrt(a2*s2 + b2*c2)
    
    rad   = np.sqrt(h*(h+2*rho) + (a4*s2+b4*c2)/rho**2)

    cd    = (h+rho)/rad
    sd    = (a2-b2)*ctgd*stgd/(rho*rad)
    
    cthc  = ctgd*cd - stgd*sd           # Also: sthc = stgd*cd + ctgd*sd
    thc   = np.rad2deg(np.arccos(cthc)) # arccos returns values in [0, pi]
    
    return(rad, thc, sd, cd)

#************************************************************************************
#************************************************************************************
def geo_to_gg(radius, theta):
    """
    Compute geodetic colatitude and vertical height above the ellipsoid from
    geocentric radius and colatitude.

    Parameters
    ----------
    radius : ndarray, shape (...)
        Geocentric radius in kilometers.
    theta : ndarray, shape (...)
        Geocentric colatitude in degrees.

    Returns
    -------
    height : ndarray, shape (...)
        Altitude in kilometers.
    beta : ndarray, shape (...)
        Geodetic colatitude

    Notes
    -----
    Round-off errors might lead to a failure of the algorithm especially but
    not exclusively for points close to the geographic poles. Corresponding
    geodetic coordinates are returned as NaN.

    References
    ----------
    Function uses Heikkinen's algorithm taken from:

    Zhu, J., "Conversion of Earth-centered Earth-fixed coordinates to geodetic
    coordinates", IEEE Transactions on Aerospace and Electronic Systems}, 1994,
    vol. 30, num. 3, pp. 957-961

    """
    
    # Use WGS-84 ellipsoid parameters
    a =  6378.137  # equatorial radius
    b =  6356.752  # polar radius
    
    a2 = a**2
    b2 = b**2

    e2 = (a2 - b2) / a2  # squared eccentricity
    e4 = e2*e2
    ep2 = (a2 - b2) / b2  # squared primed eccentricity

    r = radius * np.sin(np.radians(theta))
    z = radius * np.cos(np.radians(theta))

    r2 = r**2
    z2 = z**2

    F = 54*b2*z2

    G = r2 + (1. - e2)*z2 - e2*(a2 - b2)

    c = e4*F*r2 / G**3

    s = (1. + c + np.sqrt(c**2 + 2*c))**(1./3)

    P = F / (3*(s + 1./s + 1.)**2 * G**2)

    Q = np.sqrt(1. + 2*e4*P)

    r0 = -P*e2*r / (1. + Q) + np.sqrt(
        0.5*a2*(1. + 1./Q) - P*(1. - e2)*z2 / (Q*(1. + Q)) - 0.5*P*r2)

    U = np.sqrt((r - e2*r0)**2 + z2)

    V = np.sqrt((r - e2*r0)**2 + (1. - e2)*z2)

    z0 = b2*z/(a*V)

    height = U*(1. - b2 / (a*V))

    beta = 90. - np.degrees(np.arctan2(z + ep2*z0, r))

    return(height, beta)

#************************************************************************************
#************************************************************************************
def synth_values(coeffs, radius, theta, phi, \
                 nmax=None, nmin=None, grid=None):
    """
    Based on chaosmagpy from Clemens Kloss (DTU Space, Copenhagen)
    Computes radial, colatitude and azimuthal field components from the
    magnetic potential field in terms of spherical harmonic coefficients.
    A reduced version of the DTU synth_values chaosmagpy code

    Parameters
    ----------

    coeffs : ndarray, shape (..., N)
        Coefficients of the spherical harmonic expansion. The last dimension is
        equal to the number of coefficients, `N` at the grid points.
    radius : float or ndarray, shape (...)
        Array containing the radius in kilometers.
    theta : float or ndarray, shape (...)
        Array containing the colatitude in degrees
        :math:`[0^\\circ,180^\\circ]`.
    phi : float or ndarray, shape (...)
        Array containing the longitude in degrees.
    nmax : int, positive, optional
        Maximum degree up to which expansion is to be used (default is given by
        the ``coeffs``, but can also be smaller if specified
        :math:`N` :math:`\\geq` ``nmax`` (``nmax`` + 2)
    nmin : int, positive, optional
        Minimum degree from which expansion is to be used (defaults to 1).
        Note that it will just skip the degrees smaller than ``nmin``, the
        whole sequence of coefficients 1 through ``nmax`` must still be given
        in ``coeffs``.
        Magnetic field source (default is an internal source).
    grid : bool, optional
        If ``True``, field components are computed on a regular grid. Arrays
        ``theta`` and ``phi`` must have one dimension less than the output grid
        since the grid will be created as their outer product (defaults to
        ``False``).

    Returns
    -------
    B_radius, B_theta, B_phi : ndarray, shape (...)
        Radial, colatitude and azimuthal field components.

    Notes
    -----
    The function can work with different grid shapes, but the inputs have to
    satisfy NumPy's `broadcasting rules \\
    <https://docs.scipy.org/doc/numpy-1.15.0/user/basics.broadcasting.html>`_
    (``grid=False``, default). This also applies to the dimension of the
    coefficients ``coeffs`` excluding the last dimension.

    The optional parameter ``grid`` is for convenience. If set to ``True``,
    a singleton dimension is appended (prepended) to ``theta`` (``phi``)
    for broadcasting to a regular grid. The other inputs ``radius`` and
    ``coeffs`` must then be broadcastable as before but now with the resulting
    regular grid.

    Examples
    --------
    The most straight forward computation uses a fully specified grid. For
    example, compute the magnetic field at :math:`N=50` grid points on the
    surface.

    .. code-block:: python

      import igrf_utils as iut
      import numpy as np

      N = 13
      coeffs = np.ones((3,))  # degree 1 coefficients for all points
      radius = 6371.2 * np.ones((N,))  # radius of 50 points in km
      phi = np.linspace(-180., 180., num=N)  # azimuth of 50 points in deg.
      theta = np.linspace(0., 180., num=N)  # colatitude of 50 points in deg.

      B = iut.synth_values(coeffs, radius, theta, phi)
      print([B[num].shape for num in range(3)])  # (N,) shaped output

    Instead of `N` points, compute the field on a regular
    :math:`N\\times N`-grid in azimuth and colatitude (slow).

    .. code-block:: python

      radius_grid = 6371.2 * np.ones((N, N))
      phi_grid, theta_grid = np.meshgrid(phi, theta)  # regular NxN grid

      B = iut.synth_values(coeffs, radius_grid, theta_grid, phi_grid)
      print([B[num].shape for num in range(3)])  # NxN output

    But this is slow since some computations on the grid are executed several
    times. The preferred method is to use NumPy's broadcasting rules (fast).

    .. code-block:: python

      radius_grid = 6371.2  # float, () or (1,)-shaped array broadcasted to NxN
      phi_grid = phi[None, ...]  # prepend singleton: 1xN
      theta_grid = theta[..., None]  # append singleton: Nx1

      B = iut.synth_values(coeffs, radius_grid, theta_grid, phi_grid)
      print([B[num].shape for num in range(3)])  # NxN output

    For convenience, you can do the same by using ``grid=True`` option.

    .. code-block:: python

      B = iut.synth_values(coeffs, radius_grid, theta, phi, grid=True)
      print([B[num].shape for num in range(3)])  # NxN output

    Remember that ``grid=False`` (or left out completely) will result in
    (N,)-shaped outputs as in the first example.

    """

    # ensure ndarray inputs
    coeffs = np.array(coeffs, dtype=float)
    radius = np.array(radius, dtype=float) / 6371.2  # Earth's average radius
    theta = np.array(theta, dtype=float)
    phi = np.array(phi, dtype=float)

    if np.amin(theta) <= 0.0 or np.amax(theta) >= 180.0:
        if np.amin(theta) == 0.0 or np.amax(theta) == 180.0:
            included='yes'
        else:
            raise ValueError('Colatitude outside bounds [0, 180].')

    if nmin is None:
        nmin = 1
    else:
        assert nmin > 0, 'Only positive nmin allowed.'

    # handle optional argument: nmax
    nmax_coeffs = int(np.sqrt(coeffs.shape[-1] + 1) - 1)  # degree
    if nmax is None:
        nmax = nmax_coeffs
    else:
        assert nmax > 0, 'Only positive nmax allowed.'

    if nmax > nmax_coeffs:
        print('Supplied nmax = {0} and nmin = {1} is '
                      'incompatible with number of model coefficients. '
                      'Using nmax = {2} instead.'.format(
                        nmax, nmin, nmax_coeffs))
        nmax = nmax_coeffs

    if nmax < nmin:
        raise ValueError(f'Nothing to compute: nmax < nmin ({nmax} < {nmin}.)')

    # handle grid option
    grid = False if grid is None else grid

    # manually broadcast input grid on surface
    if grid:
        theta = theta[..., None]  # first dimension is theta
        phi = phi[None, ...]  # second dimension is phi

    # get shape of broadcasted result
    try:
        b = np.broadcast(radius, theta, phi,
                         np.broadcast_to(0, coeffs.shape[:-1]))
    except ValueError:
        print('Cannot broadcast grid shapes (excl. last dimension of coeffs):')
        print(f'radius: {radius.shape}')
        print(f'theta:  {theta.shape}')
        print(f'phi:    {phi.shape}')
        print(f'coeffs: {coeffs.shape[:-1]}')
        raise

    grid_shape = b.shape

    # initialize radial dependence given the source
    r_n = radius**(-(nmin+2))

    # compute associated Legendre polynomials as (n, m, theta-points)-array
    Pnm = legendre_poly(nmax, theta)

    # save sinth for fast access
    sinth = Pnm[1, 1]

    # calculate cos(m*phi) and sin(m*phi) as (m, phi-points)-array
    phi = np.radians(phi)
    cos_mp = np.cos(np.multiply.outer(np.arange(nmax+1), phi))
    sin_mp = np.sin(np.multiply.outer(np.arange(nmax+1), phi))

    # allocate arrays in memory
    B_radius = np.zeros(grid_shape)
    B_theta = np.zeros(grid_shape)
    B_phi = np.zeros(grid_shape)

    num = nmin**2 - 1
    for n in range(nmin, nmax+1):
        B_radius += (n+1) * Pnm[n, 0] * r_n * coeffs[..., num]

        B_theta += -Pnm[0, n+1] * r_n * coeffs[..., num]

        num += 1

        for m in range(1, n+1):
            B_radius += ((n+1) * Pnm[n, m] * r_n
                             * (coeffs[..., num] * cos_mp[m]
                                + coeffs[..., num+1] * sin_mp[m]))
            
            B_theta += (-Pnm[m, n+1] * r_n
                        * (coeffs[..., num] * cos_mp[m]
                           + coeffs[..., num+1] * sin_mp[m]))

            with np.errstate(divide='ignore', invalid='ignore'):
                # handle poles using L'Hopital's rule
                div_Pnm = np.where(theta == 0., Pnm[m, n+1], Pnm[n, m] / sinth)
                div_Pnm = np.where(theta == np.degrees(np.pi), -Pnm[m, n+1], div_Pnm)

            B_phi += (m * div_Pnm * r_n
                      * (coeffs[..., num] * sin_mp[m]
                         - coeffs[..., num+1] * cos_mp[m]))

            num += 2

        r_n = r_n / radius  # equivalent to r_n = radius**(-(n+2))
 
    return B_radius, B_theta, B_phi


#************************************************************************************
#************************************************************************************
def legendre_poly(nmax, theta):
    """
    Returns associated Legendre polynomials `P(n,m)` (Schmidt quasi-normalized)
    and the derivative :math:`dP(n,m)/d\\theta` evaluated at :math:`\\theta`.

    Parameters
    ----------
    nmax : int, positive
        Maximum degree of the spherical expansion.
    theta : ndarray, shape (...)
        Colatitude in degrees :math:`[0^\\circ, 180^\\circ]`
        of arbitrary shape.

    Returns
    -------
    Pnm : ndarray, shape (n, m, ...)
          Evaluated values and derivatives, grid shape is appended as trailing
          dimensions. `P(n,m)` := ``Pnm[n, m, ...]`` and `dP(n,m)` :=
          ``Pnm[m, n+1, ...]``

    """

    costh = np.cos(np.radians(theta))
    sinth = np.sqrt(1-costh**2)

    Pnm = np.zeros((nmax+1, nmax+2) + costh.shape)
    Pnm[0, 0] = 1  # is copied into trailing dimenions
    Pnm[1, 1] = sinth  # write theta into trailing dimenions via broadcasting

    rootn = np.sqrt(np.arange(2 * nmax**2 + 1))

    # Recursion relations after Langel "The Main Field" (1987),
    # eq. (27) and Table 2 (p. 256)
    for m in range(nmax):
        Pnm_tmp = rootn[m+m+1] * Pnm[m, m]
        Pnm[m+1, m] = costh * Pnm_tmp

        if m > 0:
            Pnm[m+1, m+1] = sinth*Pnm_tmp / rootn[m+m+2]

        for n in np.arange(m+2, nmax+1):
            d = n * n - m * m
            e = n + n - 1
            Pnm[n, m] = ((e * costh * Pnm[n-1, m] - rootn[d-e] * Pnm[n-2, m])
                         / rootn[d])

    # dP(n,m) = Pnm(m,n+1) is the derivative of P(n,m) vrt. theta
    Pnm[0, 2] = -Pnm[1, 1]
    Pnm[1, 2] = Pnm[1, 0]
    for n in range(2, nmax+1):
        Pnm[0, n+1] = -np.sqrt((n*n + n) / 2) * Pnm[n, 1]
        Pnm[1, n+1] = ((np.sqrt(2 * (n*n + n)) * Pnm[n, 0]
                       - np.sqrt((n*n + n - 2)) * Pnm[n, 2]) / 2)

        for m in np.arange(2, n):
            Pnm[m, n+1] = (0.5*(np.sqrt((n + m) * (n - m + 1)) * Pnm[n, m-1]
                           - np.sqrt((n + m + 1) * (n - m)) * Pnm[n, m+1]))

        Pnm[n, n+1] = np.sqrt(2 * n) * Pnm[n, n-1] / 2

    return Pnm

#************************************************************************************
#************************************************************************************
def xyz2dhif(x, y, z):
    """Calculate D, H, I and F from (X, Y, Z)
      
    Based on code from D. Kerridge, 2019
    
    Parameters
    ---------------
    X: north component (nT) : float
    Y: east component (nT) : float
    Z: vertical component (nT) : float
    
    Returns
    ------
    A tuple: (D, H, I, F) : float
    D: declination (degrees) : float
    H: horizontal intensity (nT) : float
    I: inclination (degrees) : float
    F: total intensity (nT) : float
    


    """
    r2d = np.rad2deg
    d2r = np.deg2rad
    hsq = x*x + y*y
    hoz  = np.sqrt(hsq)
    eff = np.sqrt(hsq + z*z)
    dec = np.arctan2(y,x)
    inc = np.arctan2(z,hoz)
    
    return(r2d(dec), hoz, r2d(inc), eff)
