# #############################################################################
# Distribution statement A. Approved for public release. Distribution is
# unlimited.
#      This work was supported by the Office of Naval Research
# #############################################################################


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

import datetime as dt
import math
import numpy as np
import os
from fortranformat import FortranRecordReader
import igrf_library as igrf


def IRI_monthly_mean_par(year, mth, aUT, alon, alat, coeff_dir, ccir_or_ursi):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 07.13.2023
    #
    # This is the main function that runs PyIRI and outputs all the parameters
    #
    # Variables:----------------------------------------------------------------
    # year = year of interest (integer)
    #
    # month = month of interest (integer)
    #
    # aUT   = 1-D array of UT time frames, (must be NumPy array), {hours},
    # [N_T]
    # In case only 1 time frame is needed, just pass it as an array with 1
    # element
    #
    # alon  = 1-D array of geographic longitudes, (must be NumPy array),
    # {degrees}, [N_G]
    # In case only 1 location is needed, just pass it as an array with
    # 1 element
    #
    # alat  = 1-D array of geographic latitudes, (must be NumPy array),
    # {degrees}, [N_G]
    # In case only 1 location is needed, just pass it as an array with 1
    # element
    #
    # dir   = direction to the root folder in your computer where PyIRI is
    # placed, (must be string)
    #
    # ccir_or_ursi = 0=CCIR option, 1=URSI option (integer)
    #
    # Result:--------------------------------------------------------------------
    # F2 = dictionary that contains:
    # Nm: peak density of F2 region, (NumPy array), {m-3}, [N_T, N_G, 2]
    # fo: critical frequency of F2 region, (NumPy array), {MHz}, [N_T, N_G, 2]
    # M3000: the obliquity factor for a distance of 3,000 km. Defined as
    # refracted in the ionosphere, can be received at a distance of
    # 3,000 km, (NumPy array), {unitless}, [N_T, N_G, 2]
    # hm: height of the F2 peak, (NumPy array), {km}, [N_T, N_G, 2]
    # B_top: top thickness of the F2 region, (NumPy array), {km}, [N_T, N_G, 2]
    # B_bot: bottom thickness of the F2 region, (NumPy array), {km},
    # [N_T, N_G, 2]
    #
    # F1 = dictionary that contains:
    # Nm: peak density of F1 region, (NumPy array, can have NaNs), {m-3},
    # [N_T, N_G, 2]
    # fo: critical frequency of F1 region, (NumPy array, can have NaNs),
    # {MHz}, [N_T, N_G, 2]
    # P: probability of F1 region to appear, (NumPy array), {unitless},
    # [N_T, N_G, 2]
    # hm: height of the F1 peak, (NumPy array, can have NaNs), {km},
    # [N_T, N_G, 2]
    # B_bot: bottom thickness of the F1 region, (NumPy array, can have NaNs),
    # {km}, [N_T, N_G, 2]
    #
    # E = dictionary that contains:
    # Nm: peak density of E region, (NumPy array), {m-3}, [N_T, N_G, 2]
    # fo: critical frequency of E region, (NumPy array), {MHz}, [N_T, N_G, 2]
    # hm: height of the E peak, (NumPy array), {km}, [N_T, N_G, 2]
    # B_top: top thickness of the E region, (NumPy array), {km}, [N_T, N_G, 2]
    # B_bot:    bottom thickness of the E region, (NumPy array), {km},
    # [N_T, N_G, 2]
    #
    # Es = dictionary that contains:
    # Nm: peak density of sporadic E region Es, (NumPy array), {m-3},
    # [N_T, N_G, 2]
    # fo: critical frequency of Es, (NumPy array), {MHz}, [N_T, N_G, 2]
    # hm: height of the Es peak, (NumPy array), {km}, [N_T, N_G, 2]
    # B_top: top thickness of the Es, (NumPy array), {km}, [N_T, N_G, 2]
    # B_bot: bottom thickness of the Es, (NumPy array), {km}, [N_T, N_G, 2]
    #
    # sun = dictionary that contains:
    # lon: geographic longitude of subsolar point, (NumPy array), {degrees},
    # [N_T]
    # lat: geographic latitude of subsolar point, (NumPy array), {degrees},
    # [N_T]
    #
    # mag = dictionary that contains:
    # inc: inclination of the Earth magnetic field at 300 km altitude,
    # (NumPy array),
    # {degrees}, [N_G]
    # modip: modified dip angle of the Earth magnetic field, (NumPy array),
    # {degrees}, [N_G]
    # mag_dip_lat: magnetic dip latitude, (NumPy array), {degrees}, [N_G]
    # **************************************************************************
    # **************************************************************************

    # Set limits for solar driver based of IG12=0-100.
    IG_min = 0.
    IG_max = 100.
    F107_min = IG12_2_F107(IG_min)
    F107_max = IG12_2_F107(IG_max)
    print('Model parameters are being determinaed for 2 levels of solar'
          'activity at F10.7=69.3 and 135.2 SFU')
    acoeff = ['CCIR', 'URSI']
    print('For NmF2 determination '+acoeff[ccir_or_ursi]+' used.')

    # defining this F107 limits to use throughout the code
    F107 = np.array([F107_min, F107_max])

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
    # -------------------------------------------------------------------------
    # Calculate diurnal Fourier functions F_D for the given time array aUT
    D_f0f2, D_M3000, D_Es_med = diurnal_functions(aUT)
    # -------------------------------------------------------------------------
    # Calculate geographic Jones and Gallet (JG) functions F_G for the given
    # grid and corresponding map of modip angles
    G_fof2, G_M3000, G_Es_med = set_gl_G(alon, alat, modip)
    # -------------------------------------------------------------------------
    # Read CCIR coefficients and form matrix U
    F_c_CCIR, F_c_URSI, F_M3000_c, F_Es_med = read_ccir_coeff(mth,
                                                              coeff_dir)
    if ccir_or_ursi == 0:
        F_fof2_c = F_c_CCIR
    if ccir_or_ursi == 1:
        F_fof2_c = F_c_URSI
    # -------------------------------------------------------------------------
    # Multiply matricies (F_D U)F_G
    foF2, M3000, foEs = gamma(aUT, D_f0f2, D_M3000, D_Es_med, G_fof2, G_M3000,
                              G_Es_med, F_fof2_c, F_M3000_c, F_Es_med)
    # -------------------------------------------------------------------------
    # Probability of F1 layer to apear
    P_F1, foF1 = Probability_F1(year, mth, aUT, alon, alat, mag_dip_lat, F107)
    # -------------------------------------------------------------------------
    # Solar driven E region and locations of subsolar points
    foE, slon, slat = gammaE(year, mth, aUT, alon, alat, F107)
    # -------------------------------------------------------------------------
    # Convert critical frequency to the electron density (m-3)
    NmF2, NmF1, NmE, NmEs = freq_to_Nm(foF2, foF1, foE, foEs)
    # -------------------------------------------------------------------------
    # Find heights of the F2 and E ionospheric layers
    hmF2, hmE, hmEs = hm_IRI(M3000, foE, foF2, modip, F107)
    # -------------------------------------------------------------------------
    # Find thicknesses of the F2 and E ionospheric layers
    B_F2_bot, B_F2_top, B_E_bot, B_E_top, B_Es_bot, B_Es_top = thickness(foF2,
                                                                         M3000,
                                                                         hmF2,
                                                                         hmE,
                                                                         mth,
                                                                         F107)
    # -------------------------------------------------------------------------
    # Find height of the F1 layer based on the location and thickness of the
    # F2 layer
    hmF1 = hmF1_from_F2(NmF2, NmF1, hmF2, B_F2_bot)
    # -------------------------------------------------------------------------
    # Find thickness of the F1 layer
    B_F1_bot = find_B_F1_bot(hmF1, hmE, P_F1)
    # -------------------------------------------------------------------------
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
    # -------------------------------------------------------------------------
    return(F2, F1, E, Es, sun, mag)
    # -------------------------------------------------------------------------


def IRI_density_1day(year, month, day, aUT, alon, alat, aalt, F107, coeff_dir,
                     driver_dir, ccir_or_ursi):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 07.13.2023
    #
    # This is the main function that runs PyIRI for a particular day of
    # interest. It uses F10.7 from a provided text file. If this text
    # file needs to be updated, then daily averaged data from
    # https://omniweb.gsfc.nasa.gov/form/dx1.html     and add it to the file.
    # Use a different routine to run PyIRI for your own F10.7 index.
    #
    # Variables:----------------------------------------------------------------
    # year  = year of interest (integer)
    #
    # month = month of interest (integer)
    #
    # aUT   = 1-D array of UT time frames, (must be NumPy array), {hours},
    # [N_T], In case only 1 time frame is needed, just pass it as an array
    # with 1 element
    #
    # alon  = 1-D array of geographic longitudes, (must be NumPy array),
    # {degrees}, [N_G], In case only 1 location is needed, just pass it as an
    # array with 1 element
    #
    # alat  = 1-D array of geographic latitudes, (must be NumPy array),
    # {degrees}, [N_G], In case only 1 location is needed, just pass it as an
    # array with 1 element
    #
    # aalt  = 1-D array of altitudes, (must be NumPy array), {km}, [N_V]
    # In case only 1 height is needed, just pass it as an array with 1 element
    #
    # dir   = direction to the root folder in your computer where PyIRI is
    # placed, (must be string)
    #
    # ccir_or_ursi = 0=CCIR option, 1=URSI option (integer)
    #
    # Result:-------------------------------------------------------------------
    # F2 = dictionary that contains:
    # NmF2: peak density of F2 region, (NumPy array), {m-3},
    # [N_T, N_G, 2]
    #
    # foF2: critical frequency of F2 region, (NumPy array), {MHz},
    # [N_T, N_G, 2]
    #
    # M3000: the obliquity factor for a distance of 3,000 km. Defined as
    # MUF(3000)F2/foF2, where MUF(3000) F2 is the highest frequency that,
    # refracted in the ionosphere, can be received at a distance of
    # 3,000 km, (NumPy array), {unitless}, [N_T, N_G, 2]
    #
    # hmF2: height of the F2 peak, (NumPy array), {km}, [N_T, N_G, 2]
    #
    # B_F2_top: top thickness of the F2 region, (NumPy array), {km},
    # [N_T, N_G, 2]
    #
    # B_F2_bot: bottom thickness of the F2 region, (NumPy array), {km},
    # [N_T, N_G, 2]
    #
    # F1 = dictionary that contains:
    # NmF1: peak density of F1 region, (NumPy array, can have NaNs), {m-3},
    # [N_T, N_G, 2]
    #
    # foF1: critical frequency of F1 region, (NumPy array, can have NaNs),
    # {MHz}, [N_T, N_G, 2]
    #
    # P_F1: probability of F1 region to appear, (NumPy array), {unitless},
    # [N_T, N_G, 2]
    #
    # hmF1: height of the F1 peak, (NumPy array, can have NaNs), {km},
    # [N_T, N_G, 2]
    #
    # B_F1_bot: bottom thickness of the F1 region, (NumPy array,
    # can have NaNs), {km}, [N_T, N_G, 2]
    #
    # E = dictionary that contains:
    # NmE: peak density of E region, (NumPy array), {m-3},
    # [N_T, N_G, 2]
    #
    # foE: critical frequency of E region, (NumPy array), {MHz},
    # [N_T, N_G, 2]
    #
    # hmE: height of the E peak, (NumPy array), {km}, [N_T, N_G, 2]
    #
    # B_E_top: top thickness of the E region, (NumPy array), {km},
    # [N_T, N_G, 2]
    #
    # B_E_bot: bottom thickness of the E region, (NumPy array), {km},
    # [N_T, N_G, 2]
    #
    # Es = dictionary that contains:
    # NmEs: peak density of sporadic E region Es, (NumPy array),
    # {m-3}, [N_T, N_G, 2]
    #
    # foEs: critical frequency of Es, (NumPy array), {MHz},
    # [N_T, N_G, 2]
    #
    # hmEs: height of the Es peak, (NumPy array), {km},
    # [N_T, N_G, 2]
    #
    # B_Es_top: top thickness of the Es, (NumPy array), {km},
    # [N_T, N_G, 2]
    #
    # B_Es_bot: bottom thickness of the Es, (NumPy array), {km},
    # [N_T, N_G, 2]
    #
    # sun = dictionary that contains:
    # lon: geographic longitude of subsolar point, (NumPy array),
    # {degrees}, [N_T]
    #
    # lat: geographic latitude of subsolar point, (NumPy array),
    # {degrees}, [N_T]
    #
    # mag = dictionary that contains:
    # inc: inclination of the Earth magnetic field at 300 km altitude,
    # (NumPy array), {degrees}, [N_G]
    #
    # EDP = electron density for the day of interest, (NumPy array), {m-3},
    # [N_T, N_V, N_G]
    # **************************************************************************
    print('PyIRI: IRI_density_1day:------------------------------------------')
    print('Determining parameters and electron density for 1 day: \
           year='+str(year)+', month='+str(month)+', day='+str(day))
    print('For UT = ', aUT)
    print('Longitude = ', alon)
    print('Latitude = ', alat)

    # date and time python object for the time of interest
    dtime = dt.datetime(year, month, day)

    # solar parameters for the day of interest, or provide your own F10.7
    if np.isfinite(F107) != 1:
        F107 = solar_parameter(dtime, driver_dir)
        print('F10.7 value was taken from OMNIWeb data file: \
               PyIRI/solar_drivers/Solar_Driver_F107.txt')
        print('F10.7='+str(F107))
    else:
        print('Provided F10.7='+str(F107))

    acoeff = ['CCIR', 'URSI']
    print('For NmF2 determination '+acoeff[ccir_or_ursi]+'  used.')

    # find out what monthly means are needed first and what their weights
    # will be
    t_before, t_after, fr1, fr2 = day_of_the_month_corr(year, month, day)

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

    print('Mean monthly parameters are calculated')
    F2 = fractional_correction_of_dictionary(fr1, fr2, F2_1, F2_2)
    F1 = fractional_correction_of_dictionary(fr1, fr2, F1_1, F1_2)
    E = fractional_correction_of_dictionary(fr1, fr2, E_1, E_2)
    Es = fractional_correction_of_dictionary(fr1, fr2, Es_1, Es_2)
    sun = fractional_correction_of_dictionary(fr1, fr2, sun_1, sun_2)
    mag = fractional_correction_of_dictionary(fr1, fr2, mag_1, mag_2)

    # construct density
    EDP_min_max = reconstruct_density_from_parameters(F2, F1, E, aalt)

    # interpolate in solar activity
    EDP = solar_interpolate(EDP_min_max[0, :], EDP_min_max[1, :], F107)

    # interpolate parameters in solar activity
    F2 = solar_interpolation_of_dictionary(F2, F107)
    F1 = solar_interpolation_of_dictionary(F1, F107)
    E = solar_interpolation_of_dictionary(E, F107)
    Es = solar_interpolation_of_dictionary(Es, F107)

    print('Completed')
    print('-----------------------------------------------------------------')
    # -----------------------------------------------------------------------
    return(F2, F1, E, Es, sun, mag, EDP)
    # -----------------------------------------------------------------------


def adjust_longitude(lon, type):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 10.21.2022
    #
    # This function adjustst the array of longitudes to go from -180-180 or
    # from 0-360.
    #
    # Variables: lon = numpy array of longitudes
    #            type = string of 'to360' or 'to180'
    #
    # Result:    adjusted array of longitudes in the same format as the input
    # **************************************************************************
    # **************************************************************************
    if isinstance(lon, np.ndarray):

        if type == 'to360':
            # check that values in the array don't go over 360
            multiple = np.floor_divide(np.abs(lon), 360)
            lon = lon - multiple*360*np.sign(lon)

            a = np.where(lon < 0.)
            inda = a[0]
            lon[inda] = lon[inda] + 360.

            b = np.where(lon > 360.)
            indb = b[0]
            lon[indb] = lon[indb] - 360.

        if type == 'to180':
            # check that values in the array don't go over 360
            multiple = np.floor_divide(np.abs(lon), 360)
            lon = lon - multiple*360*np.sign(lon)

            a = np.where(lon > 180.)
            inda = a[0]
            lon[inda] = lon[inda] - 360.

            b = np.where(lon < -180.)
            indb = b[0]
            lon[indb] = lon[indb] + 360.

        if type == 'to24':
            # check that values in the array don't go over 24
            multiple = np.floor_divide(np.abs(lon), 24)
            lon = lon - multiple*24.*np.sign(lon)

            a = np.where(lon < 0.)
            inda = a[0]
            lon[inda] = lon[inda]+24.

            b = np.where(lon > 24.)
            indb = b[0]
            lon[indb] = lon[indb]-24.

    if (isinstance(lon, int)) | (isinstance(lon, float)):
        if type == 'to360':
            # check that values in the array don't go over 360
            multiple = np.floor_divide(np.abs(lon), 360)
            lon = lon - multiple*360*np.sign(lon)

            if lon < 0.:
                lon = lon + 360.
            if lon > 360.:
                lon = lon - 360.

        if type == 'to180':
            # check that values in the array don't go over 360
            multiple = np.floor_divide(np.abs(lon), 360)
            lon = lon - multiple*360*np.sign(lon)

            if lon > 180.:
                lon = lon - 360.
            if lon < -180.:
                lon = lon + 360.

        if type == 'to24':
            # check that values in the array don't go over 24
            multiple = np.floor_divide(np.abs(lon), 24)
            lon = lon - multiple*24.*np.sign(lon)

            if lon < 0.:
                lon = lon + 24.
            if lon > 24.:
                lon = lon - 24.

# -----------------------------------------------------------------------
    return(lon)
# -----------------------------------------------------------------------


def read_ccir_coeff(mth, coeff_dir):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 10.21.2022
    #
    # This function reads CCIR coefficients and coefficients for sporadic E
    # (Es)
    #
    # Variables: mth  = the month of interest
    #            dir  = direction of the folder
    #
    # Result:    F_fof2, F_M3000, F_Es_upper, F_Es_median, F_Es_low:
    #            Coefficients for F0F2, M3000, upper decile Es, median Es,
    # lower decile Es
    #            for 2 levels of solar activity
    #            Shape: [nj, nk, 2]
    #
    # Acknowledgement for Es coefficients:
    # Mrs. Estelle D. Powell and Mrs. Gladys I. Waggoner in supervising the
    # collection, keypunching and processing of the foEs data.
    # This work was sponsored by U.S. Navy as part of the SS-267 program.
    # The final development work and production of the foEs maps was supported
    # by the U.S> Information Agency.
    # Acknowledgemets to Doug Drob (NRL) for giving me these coefficients.
    # **************************************************************************
    # **************************************************************************

    # pull the predefined sizes of the function extensions
    coef = highest_power_of_extension()

    # check that month goes from 1 to 12
    if (mth < 1) | (mth > 12):
        flag = 'Error: In read_ccir_coeff_and_interpolate month \
                is < 1 or > 12!'
        print(flag)

    # add 10 to the month becasue the file numeration goes from 11 to 22.
    cm = str(mth+10)

    # F region coefficients:
    file_F = open(os.path.join(coeff_dir, 'CCIR', 'ccir'+cm+'.asc'), mode='r')
    fmt = FortranRecordReader('(1X,4E15.8)')
    full_array = []
    for line in file_F:
        line_vals = fmt.read(line)
        full_array = np.concatenate((full_array, line_vals), axis=None)
    array0_F_CCIR = full_array[0:-2]
    file_F.close()

    # F region URSI coefficients:
    file_F = open(os.path.join(coeff_dir, 'URSI', 'ursi'+cm+'.asc'), mode='r')
    fmt = FortranRecordReader('(1X,4E15.8)')
    full_array = []
    for line in file_F:
        line_vals = fmt.read(line)
        full_array = np.concatenate((full_array, line_vals), axis=None)
    array0_F_URSI = full_array
    file_F.close()

    # Sporadic E coefficients:
    file_E = open(os.path.join(coeff_dir, 'Es', 'Es'+cm+'.asc'), mode='r')
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
    # these number are the excact format for the files, even though some
    # numbers will be zeroes.
    nk = 76
    H = 17
    ns = 6

    F = np.zeros((H, nk, ns))
    # Each file starts with 60 indexies, we will not need them, therefore
    # skip it
    skip_coeff = np.zeros((6, 10))
    array1 = array0_E[skip_coeff.size:(skip_coeff.size+F.size)]
    F_E = np.reshape(array1, F.shape, order='F')

    F_fof2_CCIR = F_fof2_2_CCIR
    F_fof2_URSI = F_fof2_2_URSI
    F_M3000 = F_M3000_2
    F_Es_upper = F_E[:, :, 0:2]
    F_Es_median = F_E[:, :, 2:4]
    F_Es_low = F_E[:, :, 4:6]

    # Trim E-region arrays to the exact shape of the functions
    F_Es_upper = F_Es_upper[0:coef['nj']['Es_upper'],
                            0:coef['nk']['Es_upper'],
                            :]
    F_Es_median = F_Es_median[0:coef['nj']['Es_median'],
                              0:coef['nk']['Es_median'],
                              :]
    F_Es_low = F_Es_low[0:coef['nj']['Es_lower'],
                        0:coef['nk']['Es_lower'],
                        :]
# -----------------------------------------------------------------------
    return(F_fof2_CCIR, F_fof2_URSI, F_M3000, F_Es_upper, F_Es_median,
           F_Es_low)
# -----------------------------------------------------------------------


def set_diurnal_functions(nj, time_array):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 10.21.2022
    #
    # This function sets the combination of sin and cos functions that define
    # the diurnal destribution of the parameters and that can be further
    # multiplied by the coefficients U_jk (from CCIR and Es maps).
    # The desired eequation can be found in the Technical Note Advances in
    # Ionospheric Mapping by Numerical Methods, Jones & Graham 1966. Equation
    # (c) page 38.
    #
    # Variables: nj = the highest order of diurnal variation
    #            time_array = array of UT in hours (0-24)
    #
    # Result:    D = Dk(T0) Equation (c) page 38 in Jones & Graham 1966
    # (except the U_jk coefficients)
    # **************************************************************************
    # **************************************************************************

    # check that time array goes from 0-24:
    if (np.min(time_array) < 0) | (np.max(time_array) > 24):
        flag = 'Error: in set_diurnal_functions time array min < 0 or max > 24'
        print(flag)

    D = np.zeros((nj, time_array.size))

    # convert time array to -pi/2 to pi/2
    hou = np.deg2rad((15.0E0*time_array - 180.0E0))

    # construct the array of evaluated functions that can be further multiplied
    # by the coefficients
    ii = 0
    for j in range(0, nj):
        for i in range(0, 2):
            if np.mod(i, 2) == 0:
                f = np.cos(hou*(i + j))
            else:
                f = np.sin(hou*(i + j))
            if ii < nj:
                D[ii, :] = f
                ii = ii + 1
            else:
                break
# -----------------------------------------------------------------------
    return(D)
# -----------------------------------------------------------------------


def diurnal_functions(time_array):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 10.21.2022
    #
    # This function calculates diurnal functions for F0F2, M3000, and Es
    # coefficients
    #
    # Variables:
    # time_array = array of UT in hours (0-24)
    #
    # Result:
    # D_f0f2, D_M3000, D_Es_upper, D_Es_median, D_Es_low
    # D = Dk(T0) Equation (c) page 38 in Jones & Graham 1966 (except the U_jk
    # coefficients)
    # **************************************************************************
    # **************************************************************************
    # nj is the highest order of the expansion

    coef = highest_power_of_extension()

    D_f0f2 = set_diurnal_functions(coef['nj']['F0F2'], time_array)
    D_M3000 = set_diurnal_functions(coef['nj']['M3000'], time_array)
    D_Es_median = set_diurnal_functions(coef['nj']['Es_median'], time_array)

    # transpose array so that multiplication can be done later
    D_f0f2 = np.transpose(D_f0f2)
    D_M3000 = np.transpose(D_M3000)
    D_Es_median = np.transpose(D_Es_median)
# -----------------------------------------------------------------------
    return(D_f0f2, D_M3000, D_Es_median)
# -----------------------------------------------------------------------


def set_global_functions(Q, nk, alon, alat, x):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 10.21.2022
    #
    # This function sets Geographic Coodrdinate Functions G_k(position) page
    # 18 in Jones & Graham 1966
    #
    # Variables: Q = vector of highest order of sin(x)
    #            nk = highest order of geographic extension, or how many
    # functions are there e.g. there are 76 functions in Table 3 on page 18
    # in Jones & Graham 1966
    #
    # Variables:
    # alon = 1D numpy array of geographic longitudes in degrees [ngrid]
    # alat = 1D numpy array of geographic latitudes in degrees [ngrid]
    # x = modified dip angle [ngrid]
    #
    # Result:
    # Gk = Geographic Coodrdinate Functions G_k of size [k, ngrid]
    # **************************************************************************
    # **************************************************************************
    Gk = np.zeros((nk, alon.size))
    k = 0.
    for j in range(0, len(Q)):
        for i in range(0, Q[j]):
            for m in range(0, 2):
                if m == 0:
                    fun3_st = 'cos(lon*'+str(j)+')'
                    fun3 = np.cos(np.deg2rad(alon*j))
                else:
                    fun3_st = 'sin(lon*'+str(j)+')'
                    # fun3 = np.sin(np.deg2rad(alon*j))
                if (j != 0) | (fun3_st != 'sin(lon*'+str(j)+')'):
                    fun1 = (np.sin(np.deg2rad(x)))**i
                    # fun1_st = 'sin(x)^'+str(i)

                    fun2 = (np.cos(np.deg2rad(alat)))**j
                    # fun2_st = 'cos(alat)^'+str(j)

                    Gk[k, :] = fun1*fun2*fun3
                    k = k + 1
# -----------------------------------------------------------------------
    return(Gk)
# -----------------------------------------------------------------------


def set_gl_G(alon, alat, x):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 10.21.2022
    #
    # This function sets Geographic Coodrdinate Functions G_k(position) page
    # 18 of Jones & Graham 1966 for F0F2, M3000, and Es coefficients
    #
    # Variables:
    # alon = 1D numpy array of geographic longitudes in degrees [ngrid]
    # alat = 1D numpy array of geographic latitudes in degrees [ngrid]
    # x = modified dip angle [ngrid]
    #
    # Result:
    # G_fof2, G_M3000, G_Es_upper, G_Es_median, G_Es_low =
    # Geographic Coodrdinate Functions G_k of size [k, ngrid]
    # **************************************************************************
    # **************************************************************************
    coef = highest_power_of_extension()

    G_fof2 = set_global_functions(coef['QM']['F0F2'],
                                  coef['nk']['F0F2'], alon, alat, x)
    G_M3000 = set_global_functions(coef['QM']['M3000'],
                                   coef['nk']['M3000'], alon, alat, x)
    G_Es_median = set_global_functions(coef['QM']['Es_median'],
                                       coef['nk']['Es_median'], alon, alat, x)
# -----------------------------------------------------------------------
    return(G_fof2, G_M3000, G_Es_median)
# -----------------------------------------------------------------------


def gamma(time, D_f0f2, D_M3000, D_Es_median, G_fof2, G_M3000, G_Es_median,
          F_fof2_coeff, F_M3000_coeff, F_Es_median):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 10.21.2022
    #
    # This function caclulates numerical maps for F0F2, M3000, and Es for 2
    # levels of solar activity (min, max)
    #
    # Variables:
    # D_f0f2, D_M3000, D_Es_upper, D_Es_median, D_Es_lower =
    # diurnal functions for given time array
    #
    # G_fof2, G_M3000, G_Es_upper, G_Es_median, G_Es_lower =
    # global functions for given grid
    #
    # F_fof2_coeff, F_M3000_coeff, F_Es_upper, F_Es_median, F_Es_lower =
    # CCIR and Es coefficients
    #
    # Result:
    # gamma_f0f2, gamma_M3000, gamma_Es_median = numerical maps for F0F2,
    # M3000, and Es (median) in [ntime, ngrid, 2]
    # format, where 0 =low, 1 = high solar activity
    # **************************************************************************
    # **************************************************************************
    Nd = D_f0f2.shape[0]
    Ng = G_fof2.shape[1]

    gamma_f0f2 = np.zeros((Nd, Ng, 2))
    gamma_M3000 = np.zeros((Nd, Ng, 2))
    gamma_Es_median = np.zeros((Nd, Ng, 2))

    # find numerical maps for 2 levels of solar activity
    for isol in range(0, 2):
        # matrix multiplication, where first operation calculates D = Dk(T0)
        # Equation (c) page 38 in Jones & Graham 1966
        # and second operation calculates Equation (f) page 39
        mult1 = np.matmul(D_f0f2, F_fof2_coeff[:, :, isol])
        gamma_f0f2[:, :, isol] = np.matmul(mult1, G_fof2)

        mult2 = np.matmul(D_M3000, F_M3000_coeff[:, :, isol])
        gamma_M3000[:, :, isol] = np.matmul(mult2, G_M3000)

        mult3 = np.matmul(D_Es_median, F_Es_median[:, :, isol])
        gamma_Es_median[:, :, isol] = np.matmul(mult3, G_Es_median)
# -----------------------------------------------------------------------
    return(gamma_f0f2, gamma_M3000, gamma_Es_median)
# -----------------------------------------------------------------------


def highest_power_of_extension():
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 10.21.2022
    #
    # This function sets a common set of constants that define the power of
    # etensions
    #
    # Variables:
    # none
    #
    # Result:
    # a dictionary where: QM = array of highest power of sin(x)
    # nk = highest order of geographic extension, or how many functions are
    # there
    # e.g. there are 76 functions in Table 3 on page 18 in Jones & Graham 1966
    # nj = highest order in diurnal variation
    # **************************************************************************
    # **************************************************************************
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
# -----------------------------------------------------------------------
    return(const)
# -----------------------------------------------------------------------


def subsol_apex(dtime):
    # **************************************************************************
    # **************************************************************************
    # Fortran code by A. D. Richmond, NCAR. Translated from IDL by K. Laundal.
    # Based on formulas in Astronomical Almanac for the year 1996, p. C24.
    # (U.S. Government Printing Office, 1994). Usable for years 1601-2100,
    # inclusive. According to the Almanac, results are good to at least 0.01
    # degree latitude and 0.025 degrees longitude between years 1950 and 2050.
    # Accuracy for other years has not been tested. Every day is assumed to
    # have exactly 86400 seconds; thus leap seconds that sometimes occur on
    # December 31 are ignored (their effect is below the accuracy threshold of
    # the algorithm).
    #
    # This function calculates coordinates of subsolar point
    #
    # Variables: datetime = class:`datetime.datetime`
    #
    # Result:   sslat = Latitude of subsolar point (=solar declination)
    #           sslon = Longitude of subsolar point (=solar declination)
    # **************************************************************************
    # **************************************************************************
    # Convert to year, day of year and seconds since midnight
    if isinstance(dtime, dt.datetime):
        year = np.asanyarray([dtime.year])
        doy = np.asanyarray([dtime.timetuple().tm_yday])
        ut = np.asanyarray([dtime.hour*3600. +
                            dtime.minute*60. +
                            dtime.second])
    elif isinstance(dtime, np.ndarray):
        # This conversion works for datetime of wrong precision or unit epoch
        times = dtime.astype('datetime64[s]')
        year_floor = times.astype('datetime64[Y]')
        day_floor = times.astype('datetime64[D]')
        year = year_floor.astype(int) + 1970.
        doy = (day_floor - year_floor).astype(int) + 1.
        ut = (times.astype('datetime64[s]') - day_floor).astype(float)
    else:
        raise ValueError("input must be datetime.datetime or numpy array")
    if not (np.all(1601. <= year) and np.all(year <= 2100.)):
        raise ValueError('Year must be in [1601, 2100]')

    yr = year - 2000.

    nleap = np.floor((year - 1601.) / 4.).astype(int)
    nleap -= 99
    mask_1900 = year <= 1900
    if np.any(mask_1900):
        ncent = np.floor((year[mask_1900] - 1601.)/100.).astype(int)
        ncent = 3. - ncent
        nleap[mask_1900] = nleap[mask_1900] + ncent

    l0 = -79.549 + (-0.238699 * (yr - 4.0 * nleap) + 3.08514e-2 * nleap)
    g0 = -2.472 + (-0.2558905 * (yr - 4.0 * nleap) - 3.79617e-2 * nleap)

    # Days (including fraction) since 12 UT on January 1 of IYR:
    df = (ut / 86400.0 - 1.5) + doy

    # Mean longitude of Sun:
    lmean = l0 + 0.9856474 * df

    # Mean anomaly in radians:
    grad = np.radians(g0 + 0.9856003 * df)

    # Ecliptic longitude:
    lmrad = (np.radians(lmean + 1.915 * np.sin(grad) +
                        0.020 * np.sin(2.0 * grad)))
    sinlm = np.sin(lmrad)

    # Obliquity of ecliptic in radians:
    epsrad = np.radians(23.439 - 4e-7 * (df + 365 * yr + nleap))

    # Right ascension:
    alpha = np.degrees(np.arctan2(np.cos(epsrad) * sinlm, np.cos(lmrad)))

    # Declination, which is also the subsolar latitude:
    sslat = np.degrees(np.arcsin(np.sin(epsrad) * sinlm))

    # Equation of time (degrees):
    etdeg = lmean - alpha
    nrot = np.round(etdeg / 360.)
    etdeg = etdeg - 360. * nrot

    # Subsolar longitude calculation. Earth rotates one degree every 240 s.
    sslon = 180. - (ut / 240. + etdeg)
    nrot = np.round(sslon / 360.)
    sslon = sslon - 360. * nrot

    # Return a single value from the output if the input was a single value
    if isinstance(dtime, dt.datetime):
        return sslon[0], sslat[0]
# -----------------------------------------------------------------------
    return(sslon, sslat)
# -----------------------------------------------------------------------


def juldat(times):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 10.26.2022
    #
    # This function calculates the Julian time given calendar date and time
    # in np.datetime64 array. This function is consistent with NOVAS, The
    # US Navy Observatory Astromony Software Library and algorythm by Fliegel
    # and Van Flander.
    #
    # Variables:
    # np.datetime64 array
    #
    # Result:
    # TJD = julian date
    # **************************************************************************
    # **************************************************************************
    if isinstance(times, dt.datetime):
        Y = times.year
        M = times.month
        D = times.day
        h = times.hour
        m = times.minute
        s = times.second

        julian_datetime = (367.*Y - int((7.*(Y + int((M + 9.) / 12.)))/4.) +
                           int((275.*M)/9.) + D + 1721013.5 +
                           (h + m/60. + s / math.pow(60, 2)) / 24. -
                           0.5*math.copysign(1, 100.*Y + M - 190002.5) + 0.5)

    else:
        raise ValueError("input must be datetime.datetime")
# -----------------------------------------------------------------------
    return(julian_datetime)
# -----------------------------------------------------------------------


def doy(dtime):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 10.26.2022
    #
    # This function returns day of year given datetime object
    #
    # Variables:
    # np.datetime64 array
    #
    # Result:
    # doy = day of year
    # **************************************************************************
    # **************************************************************************
    if isinstance(dtime, dt.datetime):
        doy = dtime.timetuple().tm_yday
    else:
        raise ValueError("input must be datetime.datetime")
# -----------------------------------------------------------------------
    return(doy)
# -----------------------------------------------------------------------


def subsolar_point(juliantime):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich after SAMI4 code part written in
    # Fortran by Doug Drob
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 10.26.2022
    #
    # This function returns the lon and lat of subsolar point for a given
    # Juliantime Latitude of subsolar point is same as solar declination angle
    #
    # Variables: Juliantime (in days)
    #
    # Result:  doy = day of year
    # **************************************************************************
    # **************************************************************************
    # number of centuries from J2000
    t = (juliantime - 2451545.)/36525.

    # mean longitude corrected for aberation (deg)
    lms = np.mod(280.460 + 36000.771*t, 360.)

    # mean anomaly (deg)
    solanom = np.mod(357.527723 + 35999.05034*t, 360.)

    # ecliptic longitude (deg)
    ecplon = (lms + 1.914666471*np.sin(np.deg2rad(solanom)) +
              0.01999464*np.sin(np.deg2rad(2.*solanom)))

    # obliquity of the ecliptic (deg)
    epsilon = (23.439291 - 0.0130042*t)

    # solar declination in radians
    decsun = np.rad2deg(np.arcsin(np.sin(np.deg2rad(epsilon)) *
                                  np.sin(np.deg2rad(ecplon))))

    # greenwich mean sidereal time in seconds
    tgmst = (67310.54841 +
             (876600.*3600.+8640184.812866)*t +
             0.93104*t**2 -
             6.2e-6*t**3)

    # greenwich mean sidereal time in degrees
    tgmst = np.mod(tgmst, 86400.)/240.

    # right ascension
    num = (np.cos(np.deg2rad(epsilon)) *
           np.sin(np.deg2rad(ecplon)) /
           np.cos(np.deg2rad(decsun)))

    den = np.cos(np.deg2rad(ecplon))/np.cos(np.deg2rad(decsun))
    sunra = np.rad2deg(np.arctan2(num, den))

    # longitude of the sun indegrees
    lonsun = -(tgmst - sunra)

    # lonsun=adjust_longitude(lonsun, 'to180')
    lonsun = adjust_longitude(lonsun, 'to180')
# -----------------------------------------------------------------------
    return(lonsun, decsun)
# -----------------------------------------------------------------------


def doy_2_date(year, day_num):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 10.27.2022
    #
    # This function takes year and doy to produce datetime
    #
    # Variables:
    # year = year
    # day_num = doy
    #
    # Result:
    # res_date = julian date
    # **************************************************************************
    # **************************************************************************
    # converting to date
    res_date = (dt.datetime(int(year), 1, 1) +
                dt.timedelta(days=int(day_num) - 1))
# -----------------------------------------------------------------------
    return(res_date)
# -----------------------------------------------------------------------


def solar_zenith(lon_sun, lat_sun, lon_observer, lat_observer):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 10.27.2022
    #
    # This function takes lon and lat of the subsolar point and lon and lat
    # of the observer, and calculates solar zenith angle.
    #
    # Variables:
    # lon_sun = longitude of the Sun (deg), int, float, or np.array
    # lat_sun = latitude of the Sun (deg), int, float, or np.array
    # lon_observer = longitude of the observer (deg), np.array
    # lat_observer = latitude of the observer (deg), np.array
    #
    # Result:
    # zenith = solar zenith angle (deg)
    # **************************************************************************
    # **************************************************************************

    if (isinstance(lon_sun, int)) | (isinstance(lon_sun, float)):
        # cosine of solar zenith angle
        cos_zenith = (np.sin(np.deg2rad(lat_sun)) *
                      np.sin(np.deg2rad(lat_observer)) +
                      np.cos(np.deg2rad(lat_sun)) *
                      np.cos(np.deg2rad(lat_observer)) *
                      np.cos(np.deg2rad((lon_sun-lon_observer))))

        # solar zenith angle
        azenith = np.rad2deg(np.arccos(cos_zenith))

    if isinstance(lon_sun, np.ndarray):
        if lon_sun.size != lat_sun.size:
            print('Error: in solar_zenith lon_sun and lat_sun lengths \
                   are not equal')

        # make array to hold zenith angles based of size of lon_sun and
        # type of lon_observer
        if (isinstance(lon_observer, int)) | (isinstance(lon_observer, float)):
            azenith = np.zeros((lon_sun.size, 1))
        if isinstance(lon_observer, np.ndarray):
            azenith = np.zeros((lon_sun.size, lon_observer.size))

        for i in range(0, lon_sun.size):
            # cosine of solar zenith angle
            cos_zenith = (np.sin(np.deg2rad(lat_sun[i])) *
                          np.sin(np.deg2rad(lat_observer)) +
                          np.cos(np.deg2rad(lat_sun[i])) *
                          np.cos(np.deg2rad(lat_observer)) *
                          np.cos(np.deg2rad((lon_sun[i]-lon_observer))))

            # solar zenith angle
            azenith[i, :] = np.rad2deg(np.arccos(cos_zenith))
# -----------------------------------------------------------------------
    return(azenith)
# -----------------------------------------------------------------------


def solzen_timearray_grid(year, mth, day, T0, alon, alat):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 10.31.2022
    #
    # This function returns solar zenith angle for the given year, month,
    # day, array of UT, and arrays of lon and lat of the grid
    #
    # Variables:
    # year = year [int]
    # mth = month [int]
    # day = day [int]
    # T0 = array of UTs [hours], np.array
    # alon = array of longitudes (deg), np.array
    # alat = array of latitudes (deg), np.array
    #
    # Result:
    # solzen = solar zenith angle (deg), np.array, [ntime, ngrid]
    # **************************************************************************
    # **************************************************************************
    # check size of the grid arrays
    if alon.size != alat.size:
        flag = 'Error: in solzen_timearray_grid alon and alat sizes are \
                not the same'
        print(flag)

    aslon = np.zeros((T0.size))
    aslat = np.zeros((T0.size))
    for i in range(0, T0.size):
        UT = T0[i]
        hour = np.fix(UT)
        minute = np.fix((UT - hour)*60.)
        date_sd = dt.datetime(int(year), int(mth), int(day),
                              int(hour), int(minute))
        jday = juldat(date_sd)
        slon, slat = subsolar_point(jday)
        aslon[i] = slon
        aslat[i] = slat

    solzen = solar_zenith(aslon, aslat, alon, alat)
# -----------------------------------------------------------------------
    return(solzen, aslon, aslat)
# -----------------------------------------------------------------------


def solzen_effective(chi):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 10.31.2022
    #
    # This function calculates effective solar zenith angle as a function of
    # solar zenith angle and solar zenith angle at day-night transition,
    # according to the Titheridge model. f2 is used during daytime, and f1
    # during nightime, and the exponential day-night transition is used to
    # ensure the continuity of foE and its first derivative at solar terminator
    # [Nava et al, 2008 "A new version of the NeQuick ionosphere electron
    # density model"]
    #
    # Variables:
    # chi = solar zenith angle (deg)
    #
    # Result:
    # chi_eff = effective solar zenith angle (deg)
    # **************************************************************************
    # **************************************************************************
    # solar zenith angle at day-night transition (deg), number
    chi0 = 86.23292796211615E0

    alpha = 12.

    x = chi - chi0

    f1 = 90. - 0.24*fexp(20. - 0.2*chi)

    f2 = chi

    ee = fexp(alpha*x)

    chi_eff = (f1*ee + f2)/(ee + 1.)

    # replace nan with 0
    chi_eff = np.nan_to_num(chi_eff)
# -----------------------------------------------------------------------
    return(chi_eff)
# -----------------------------------------------------------------------


def foE(mth, solzen_effective, alat, f107):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 11.01.2022
    #
    # This function caclulates numerical maps for FoE for a given effective
    # solar zenith angle and level of solar activity. This routine is based
    # on the Ionospheric Correction Algorithm for Galileo Single Frequency
    # Users that describes NeQuick Model
    #
    # Variables:
    # mth  = the month of interest
    # f107 = the F10.7 of interest in SFU
    # alat = 1D numpy array of geographic latitudes in degrees [ngrid]
    # solzen_effective = effective solar zenith angle [ntime, ngrid]
    #
    # Result:
    # foE = critical frequency of the E region (MHz), np.array, [ntime, ngrid]
    # **************************************************************************
    # **************************************************************************
    # define the seas parameter as a function of the month of the year as
    # follows
    if (mth == 1) | (mth == 2) | (mth == 11) | (mth == 12):
        seas = -1
    if (mth == 3) | (mth == 4) | (mth == 9) | (mth == 10):
        seas = 0
    if (mth == 5) | (mth == 6) | (mth == 7) | (mth == 8):
        seas = 1

    # introduce the latitudinal dependence
    ee = fexp(np.deg2rad(0.3*alat))

    # combine seasonal and latitudinal dependence
    seasp = seas*(ee - 1.)/(ee + 1.)

    # critical frequency
    foE = (np.sqrt(0.49 + (1.112 - 0.019*seasp)**2 *
                   np.sqrt(f107)*(np.cos(np.deg2rad(solzen_effective)))**0.6))

    # turn nans into zeros
    foE = np.nan_to_num(foE)
# -----------------------------------------------------------------------
    return(foE)
# -----------------------------------------------------------------------


def foE_IRI(solzen, solzen_noon, alat, f107):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 06.22.2023
    #
    # This function caclulates numerical maps for FoE for a given effective
    # solar zenith angle, solar angle at noon, geo latitude and level of solar
    # activity. This routine is based on the IRI-2020
    # [review Bilitza et al, 2022]
    #
    # Variables:
    # solzen  = solar zenith angle
    # solzen_noon = solar zenith angle at noon
    # alat = 1D numpy array of geographic latitudes in degrees [ngrid]
    # f107 = F10.7
    #
    # Result:
    # foE = critical frequency of the E region (MHz), np.array, [ntime, ngrid]
    # **************************************************************************
    # **************************************************************************
    # set arrays same size as alat
    m = alat*0.
    C = alat*0.
    n = alat*0.

    # variation with solar activity (factor A)
    A = 1. + 0.0094*(f107 - 66.)
    # variation with noon solar zenith angle (B) and with latitude (C)
    abs_alat = np.absolute(alat)
    sl = np.cos(np.deg2rad(abs_alat))

    a1 = np.where(abs_alat < 32.)
    a2 = np.where(abs_alat >= 32.)

    m[a1] = -1.93+1.92*sl[a1]
    C[a1] = 23.+116.*sl[a1]

    m[a2] = 0.11-0.49*sl[a2]
    C[a2] = 92.+35.*sl[a2]

    solzen_noon[np.where(solzen_noon >= 90.)] = 89.999
    B = (np.cos(np.deg2rad(solzen_noon)))**m

    # variation with solar zenith angle (D)
    a3 = np.where(abs_alat > 12.)
    a4 = np.where(abs_alat <= 12.)

    n[a3] = 1.2
    n[a4] = 1.31

    # adjusted solar zenith angle during nighttime (XHIC)
    solzen_adjusted = solzen - 3.*np.log(1. + np.exp((solzen - 89.98)/3.))

    D = np.cos(np.deg2rad(solzen_adjusted))**n

    foE4 = (A*B*C*D)

    # minimum allowable foE (foe_min=sqrt[SMIN])...............................
    SMIN2 = 0.121 + 0.0015*(f107 - 60.)
    SMIN4 = SMIN2**2
    foE4[np.where(foE4 < SMIN4)] = SMIN4

    foE = foE4**0.25
# -----------------------------------------------------------------------
    return(foE)
# -----------------------------------------------------------------------


def gammaE(year, mth, time, alon, alat, aF107):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 11.01.2022
    #
    # This function caclulates numerical maps for FoE for 2 levels of solar
    # activity
    #
    # Variables:
    # mth  = the month of interest
    # time = array of UT (hours), [ntime]
    # alon = 1D numpy array of geographic longitudes in degrees [ngrid]
    # alat = 1D numpy array of geographic latitudes in degrees [ngrid]
    #
    # Result:
    # gamma_E = numerical map for foE for 2 levels of solar activity,
    # np.array, [ntime, ngrid, 2]
    # **************************************************************************
    # **************************************************************************
    # solar zenith angle for day 15 in the month of interest
    solzen, slon, slat = solzen_timearray_grid(year, mth, 15, time, alon, alat)

    # solar zenith angle for day 15 in the month of interest at noon
    solzen_noon, slon_noon, slat_noon = solzen_timearray_grid(year, mth, 15,
                                                              time*0+12.,
                                                              alon, alat)

    # effective solar zenith angle
    solzen_eff = solzen_effective(solzen)

    # make arrays to hold numerical maps for 2 levels of solar activity
    gamma_E = np.zeros((time.size, alon.size, 2))

    # find numerical maps for 2 levels of solar activity
    for isol in range(0, 2):
        # gamma_E[:, :, isol]=foE_IRI(solzen, solzen_noon, alat, aF107[isol])
        gamma_E[:, :, isol] = foE(mth, solzen_eff, alat, aF107[isol])
# -----------------------------------------------------------------------
    return(gamma_E, slon, slat)
# -----------------------------------------------------------------------


def Probability_F1(year, mth, time, alon, alat, mag_dip_lat, aF107):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 07.05.2023
    #
    # This function caclulates numerical maps probability of F1 layer
    #
    # Variables:
    # mth  = the month of interest
    # time = array of UT (hours), [ntime]
    # alon = 1D numpy array of geographic longitudes in degrees [ngrid]
    # alat = 1D numpy array of geographic latitudes in degrees [ngrid]
    # mag_dip_lat = 1D array of magnetic dip latitude [ngrid]
    #
    # Result:
    # P = numerical map for probability of F1 layer to apear for 2 levels of
    # solar activity, np.array, [ntime, ngrid, 2]
    # foF1 = critical frequency of F1 layer, np.array, [ntime, ngrid, 2]
    # **************************************************************************
    # **************************************************************************
    # make arrays to hold numerical maps for 2 levels of solar activity
    a_P = np.zeros((time.size, alon.size, 2))
    a_foF1 = np.zeros((time.size, alon.size, 2))
    a_R12 = np.zeros((time.size, alon.size, 2))
    a_mag_dip_lat_abs = np.zeros((time.size, alon.size, 2))
    a_solzen = np.zeros((time.size, alon.size, 2))

    # add 2 levels of solar activity for mag_dip_lat, by using same elemetns
    # empty array, swap axises before filling with same elements of modip,
    # swap back to match M3000 array shape
    a_mag_dip_lat_abs = np.swapaxes(a_mag_dip_lat_abs, 1, 2)
    a_mag_dip_lat_abs = np.full(a_mag_dip_lat_abs.shape, np.abs(mag_dip_lat))
    a_mag_dip_lat_abs = np.swapaxes(a_mag_dip_lat_abs, 1, 2)

    # simplified constant value from Bilitza review 2022
    gamma = 2.36

    # solar zenith angle for day 15 in the month of interest
    solzen, aslon, aslat = solzen_timearray_grid(year, mth, 15, time, alon,
                                                 alat)

    # min and max of solar activity
    R12_min_max = [F107_2_R12(aF107[0]), F107_2_R12(aF107[1])]

    for isol in range(0, 2):
        a_R12[:, :, isol] = np.full((time.size, alon.size), R12_min_max[isol])
        a_P[:, :, isol] = (0.5 + 0.5*np.cos(np.deg2rad(solzen)))**gamma
        a_solzen[:, :, isol] = solzen

    n = (0.093 +
         0.0046*a_mag_dip_lat_abs -
         0.000054*a_mag_dip_lat_abs**2 +
         0.0003*a_R12)

    f_100 = (5.348 +
             0.011*a_mag_dip_lat_abs -
             0.00023*a_mag_dip_lat_abs**2)

    f_0 = (4.35 +
           0.0058*a_mag_dip_lat_abs -
           0.00012*a_mag_dip_lat_abs**2)

    f_s = f_0 + (f_100 - f_0)*a_R12/100.

    arg = (np.cos(np.deg2rad(a_solzen)))
    a_foF1[np.where(arg > 0)] = (f_s[np.where(arg > 0)] *
                                 arg[np.where(arg > 0)]
                                 ** n[np.where(arg > 0)])

    a_foF1[np.where(a_P < 0.5)] = np.nan
# -----------------------------------------------------------------------
    return(a_P, a_foF1)
# -----------------------------------------------------------------------


def gammaF1(foE, foF2):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 11.01.2022
    #
    # This function caclulates critical frequency of F1 layer based on E region
    # following recepie from Leitinger et al, 2005 (NeQuick 2 model)
    #
    # Variables:
    # foE  = critical freqeuncy of the E layer
    # foF2  = critical freqeuncy of the F2 layer
    #
    # Result:
    # gamma_F1 = critical freqeuncy of the F1 layer, np.array,
    # [ntime, ngrid, 2]
    # **************************************************************************
    # **************************************************************************
    # set the new array to be equal to the initial critical frequency of E
    # region
    foF1 = foE*0.
    # from Leitinger et al, 2005 (NeQuick 2 model):
    foF1[np.where(foE >= 2.)] = 1.4*foE[np.where(foE >= 2.)]
    foF1[np.where(foE < 2.)] = 0.
    foF1[np.where(1.4*foE >= 0.85*foF2)] = (foE[np.where(1.4*foE >=
                                                         0.85*foF2)]*1.4*0.85)
# -----------------------------------------------------------------------
    return(foF1)
# -----------------------------------------------------------------------


def fexp(x):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 11.01.2022
    #
    # This function caclulates exp(x) with restrictions to not cause overflow
    #
    # Variables:
    # x, np.array, any shape
    #
    # Result:
    # y = np.array, x.shape
    # **************************************************************************
    # **************************************************************************
    if (isinstance(x, float)) | (isinstance(x, int)):
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
# -----------------------------------------------------------------------
    return(y)
# -----------------------------------------------------------------------


def freq_to_Nm(foF2, foF1, foE, foEs):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 11.01.2022
    #
    # This function returns maximum density for the given critical frequency
    #
    # Variables:
    # foF2, foE, foEs
    #
    # Result:
    # NmF2, NmE, NmEs
    # **************************************************************************
    # **************************************************************************
    # F2 peak
    NmF2 = freq2den(foF2)*1e11

    # F1 peak
    NmF1 = freq2den(foF1)*1e11

    # E
    NmE = freq2den(foE)*1e11

    # Es
    NmEs = freq2den(foEs)*1e11

    # exclude negative values, in case there are any
    NmF2[np.where(NmF2 <= 0)] = 1.
    NmF1[np.where(NmF1 <= 0)] = 1.
    NmE[np.where(NmE <= 0)] = 1.
    NmEs[np.where(NmEs <= 0)] = 1.
# -----------------------------------------------------------------------
    return(NmF2, NmF1, NmE, NmEs)
# -----------------------------------------------------------------------


def hm(M3000, foE, foF2, B_F2_bot, hmF2):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 11.01.2022
    #
    # This function returns hmF for the given M3000 and critical frequencies
    # how it is done in NeQuick
    #
    # Variables:
    # M3000, foE, foF2
    #
    # Result:
    # hmF2, hmF1, hmE, hmEs
    # **************************************************************************
    # **************************************************************************
    # E
    hmE = 120.+np.zeros((M3000.shape))

    # F2
    M = M3000
    del_M = foE*0.
    del_M[np.where(foE < 1e-30)] = -0.012

    # based on the Ionospheric Correction Algorithm for Galileo Single
    # Frequency Users that describes NeQuick Model
    ratio = foF2/foE

    ee = fexp(20.*(ratio - 1.75))

    ro = (ratio*ee + 1.75)/(ee + 1.)

    del_M[np.where(foE >= 1e-30)] = ((0.253/(ro[np.where(foE >= 1e-30)] -
                                             1.215)) - 0.012)

    hmF2 = (((1490.*M*np.sqrt((0.0196*M**2 + 1.) /
                              (1.2967*M**2 - 1.)))/(M+del_M)) - 176.)

    # Es
    hmEs = 100. + np.zeros((M3000.shape))
# -----------------------------------------------------------------------
    return(hmF2, hmE, hmEs)
# -----------------------------------------------------------------------


def hmF1_from_F2(NmF2, NmF1, hmF2, B_F2_bot):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 11.01.2022
    #
    # This function calculates hmF1 from known shape of F2 bottom side, where
    # it drops to NmF1
    #
    # Variables:
    # NmF2, hmF2, B_F2_bot, NmF1
    #
    # Result:
    # hmF1
    # **************************************************************************
    # **************************************************************************
    hmF1 = NmF2*0 + np.nan
    a = NmF2*0
    b = NmF2*0
    c = NmF2*0
    d = NmF2*0
    x = NmF2*0 + np.nan

    a[:, :, :] = 1.
    ind_finite = np.isfinite(NmF1)
    b[ind_finite] = (2. - 4.*NmF2[ind_finite]/NmF1[ind_finite])
    c[:, :, :] = 1.

    d = (b**2) - (4*a*c)
    ind_positive = np.where(d >= 0)

    # take second root of the quadratic equation (it is below hmF2)
    x[ind_positive] = ((-b[ind_positive] - np.sqrt(d[ind_positive])) /
                       (2*a[ind_positive]))

    ind_g0 = np.where(x > 0)
    hmF1[ind_g0] = B_F2_bot[ind_g0]*np.log(x[ind_g0]) + hmF2[ind_g0]

# -----------------------------------------------------------------------
    return(hmF1)
# -----------------------------------------------------------------------


def find_B_F1_bot(hmF1, hmE, P_F1):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 07.06.2023
    #
    # This function calculates B_F1_bot from hmF1 and hmE from NeQuick Eq 87
    #
    # Variables:
    # hmF1, hmE
    #
    # Result:
    # B_F1_bot
    # **************************************************************************
    # **************************************************************************
    B_F1_bot = 0.5*(hmF1 - hmE)
    B_F1_bot[np.where(P_F1 < 0.5)] = np.nan
# -----------------------------------------------------------------------
    return(B_F1_bot)
# -----------------------------------------------------------------------


def hm_IRI(M3000, foE, foF2, modip, aF107):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 11.01.2022
    #
    # This function returns hmF for the given M3000 and critical frequencies
    # how it is done in IRI-2020
    #
    # Variables:
    # M3000, foE, foF2
    #
    # Result:
    # hmF2, hmF1, hmE, hmEs
    # **************************************************************************
    # **************************************************************************
    s = M3000.shape
    time_dim = s[0]
    grid_dim = s[1]

    R12_min_max = np.array([F107_2_R12(aF107[0]), F107_2_R12(aF107[1])])

    # E
    hmE = 120. + np.zeros((M3000.shape))

    # F2
    # based on BSE-1979 IRI Option developed by Bilitza et al. (1979)
    # see 3.3.1 of IRI 2020 Review by Bilitza et al. 2022

    # add 2 levels of solar activity for modip, by using same elemetns
    # empty array, swap axises before filling with same elements of modip,
    # swap back to match M3000 array shape
    modip_2levels = M3000*0.
    modip_2levels = np.swapaxes(modip_2levels, 1, 2)
    modip_2levels = np.full(modip_2levels.shape, modip)
    modip_2levels = np.swapaxes(modip_2levels, 1, 2)

    # make R12 same as input arrays to account for 2 levels of solar activity
    aR12 = M3000*0.
    for isol in range(0, 2):
        aR12[:, :, isol] = np.full((time_dim, grid_dim), R12_min_max[isol])

    # empty array
    hmF2 = np.zeros((M3000.shape))

    ratio = foF2/foE
    f1 = 0.00232*aR12+0.222
    f2 = 1.0-aR12/150.0*np.exp(-(modip_2levels/40.)**2)
    f3 = 1.2-0.0116*np.exp(aR12/41.84)
    f4 = 0.096*(aR12-25.)/150.

    a = np.where(ratio < 1.7)
    ratio[a] = 1.7

    DM = f1*f2/(ratio - f3)+f4
    hmF2 = 1490.0/(M3000+DM) - 176.0

    # Es
    hmEs = 100.+np.zeros((M3000.shape))
# -----------------------------------------------------------------------
    return(hmF2, hmE, hmEs)
# -----------------------------------------------------------------------


def thickness(foF2, M3000, hmF2, hmE, mth, aF107):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 11.01.2022
    #
    # This function returns thicknesses
    #
    # Variables:
    # NmF2, foF2, M3000, hmF2, hmF1, hmE,
    # mth = month
    #
    # Result:
    # B_F2_bot, B_F1_top, B_E_top, B_E_bot, B_Es_top, B_Es_bot
    # **************************************************************************
    # **************************************************************************

    # B_F2_bot..................................................................
    NmF2 = freq2den(foF2)
    # In the actual NeQuick_2 code there is a typo, missing 0.01 which makes
    # the B 100 times smaller. It took me a long time to find this mistake,
    # while comparing with my results. The printed guide doesn't have this
    # typo.
    dNdHmx = -3.467 + 1.714*np.log(foF2) + 2.02*np.log(M3000)
    dNdHmx = 0.01*fexp(dNdHmx)
    B_F2_bot = 0.385*NmF2/dNdHmx

    # B_F2_top..................................................................
    # set empty arrays
    k = foF2*0
    # shape parameter depends on solar activity:

    for isol in range(0, 2):
        # Effective sunspot number
        R12 = F107_2_R12(aF107[isol])
        k[:, :, isol] = (3.22 -
                         0.0538*foF2[:, :, isol] -
                         0.00664*hmF2[:, :, isol] +
                         (0.113*hmF2[:, :, isol]/B_F2_bot[:, :, isol]) +
                         0.00257*R12)

    # auxiliary parameters x and v:
    x = (k*B_F2_bot-150.)/100.
    # thickness
    B_F2_top = (100.*x + 150.)/(0.041163*x**2 - 0.183981*x + 1.424472)

    # B_E_top..................................................................
    B_E_top = 7.+np.zeros((NmF2.shape))

    # B_E_bot..................................................................
    B_E_bot = 5.+np.zeros((NmF2.shape))

    # B_Es_top.................................................................
    B_Es_top = 1.+np.zeros((NmF2.shape))

    # B_Es_bot.................................................................
    B_Es_bot = 1.+np.zeros((NmF2.shape))

# -----------------------------------------------------------------------
    return(B_F2_bot, B_F2_top, B_E_bot, B_E_top, B_Es_bot, B_Es_top)
# -----------------------------------------------------------------------


def epstein(Nm, hm, B, alt):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 01.10.2023
    #
    # This function returns epstein function
    #
    # Variables:
    # Nm = peak amplitude (np.array)
    # hm = peak heigh (np.array)
    # B = thickness (np.array)
    # alt = height dependent variable (np.array)
    #
    # Result:
    # res = Epstein function
    # **************************************************************************
    # **************************************************************************
    X = Nm
    Y = hm
    Z = B
    W = alt

    aexp = fexp((W - Y)/Z)

    res = X*aexp/(1+aexp)**2
# -----------------------------------------------------------------------
    return(res)
# -----------------------------------------------------------------------


def amplitude(NmF2, NmE, NmF1, hmF2, hmF1, hmE, B_E_top, B_F1_bot, B_F2_bot,
              foF1):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 11.01.2022
    #
    # This function returns thicknesses
    #
    # Variables:
    # NmF2, NmE, NmF1, hmF2, hmF1, hmE, B_E_top, B_F1_bot, B_F2_bot, foF1
    # (or gamma_F1)
    #
    # Result:
    # A1, A2, A3 = amplitudes of F2, F1, and E regions
    # **************************************************************************
    # **************************************************************************

    A1 = np.zeros((NmF2.size))
    A2 = np.zeros((NmF2.size))
    A3 = np.zeros((NmF2.size))

    # A1 = F2 layer amplitude
    A1 = 4.*NmF2

    A3a = 4.*NmE

    for i in range(0, 5):
        if (NmF1 != 0):
            A2a = 4.*((NmF1-epstein(A1, hmF2, B_F2_bot, hmF1) -
                       epstein(A3a, hmE, B_E_top, hmF1)))
        else:
            A2a = np.array([0])
        if (NmE != 0):
            A3a = 4.*((NmE - epstein(A2a, hmF1, B_F1_bot, hmE) -
                       epstein(A1, hmF2, B_F2_bot, hmE)))
        else:
            A3a = np.array([0])

    A2 = A2a
    exp = fexp(60*(A3a - 0.005))
    A3 = (A3a*exp+0.05)/(1+exp)
# -----------------------------------------------------------------------
    return(A1, A2, A3)
# -----------------------------------------------------------------------


def leap_year(year):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 11.01.2022
    #
    # This function returns 1 if the given year is a leap year and 0 is not.
    #
    # Variables:
    # year = year
    #
    # Result:
    # 1, 0 = yes, no
    # **************************************************************************
    # **************************************************************************
    # if it is a century year then it should be evenly devided by 400
    if (year % 400 == 0) and (year % 100 == 0):
        # ----------------------------------------------------------------------
        return(1)
        # ----------------------------------------------------------------------
    # not a century year should be evenly devided by 4
    elif (year % 4 == 0) and (year % 100 != 0):
        # ----------------------------------------------------------------------
        return(1)
        # ----------------------------------------------------------------------
    # else it is not a leap year
    else:
        # ----------------------------------------------------------------------
        return(0)
        # ----------------------------------------------------------------------


def decimal_year(dtime):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 11.08.2022
    #
    # This function returns decimal year. For example, middle of the year
    # is 2020.5.
    #
    # Variables:
    # dtime = datetime python object
    #
    # Result:
    # decimal year, like 2020.5
    # **************************************************************************
    # **************************************************************************
    # day of the year
    doy = dtime.timetuple().tm_yday

    # decimal, day of year devided by number of days in year, taking leap years
    # in account
    decimal = (doy-1)/(365 + leap_year(dtime.year))

    # year plus decimal
    date_decimal = dtime.year+decimal
# -----------------------------------------------------------------------
    return(date_decimal)
# -----------------------------------------------------------------------


def lon2MLT(alon, slon):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 11.08.2022
    #
    # This function calculates MLT from given geographic longitude and subsolar
    # point longitude
    #
    # Variables:
    # alon = np array of longitudes
    # slon = longitude of subsolar point
    #
    # Result:
    # MLT = magnetic local time
    # **************************************************************************
    # **************************************************************************
    MLT = (180. + alon - slon) / 15. % 24.
# -----------------------------------------------------------------------
    return(MLT)
# -----------------------------------------------------------------------


def MLT2lon(MLT, slon):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 11.08.2022
    #
    # This function calculates Longitude from given MLT and subsolar point
    # longitude
    #
    # Variables:
    # MLT = np array of MLT
    # slon = longitude of subsolar point
    #
    # Result:
    # alon = geographic longitude
    # **************************************************************************
    # **************************************************************************
    alon = (15. * MLT - 180. + slon + 360.) % 360.
# -----------------------------------------------------------------------
    return(alon)
# -----------------------------------------------------------------------


def strd(dtime):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 11.22.2022
    #
    # This function returns string of day stamp
    #
    # Variables:
    # datetime = datetime object
    #
    # Result:
    # date_out = string YYYYMMDD
    # **************************************************************************
    # **************************************************************************
    date_out = dtime.strftime('%Y%m%d')
# -----------------------------------------------------------------------
    return(date_out)
# -----------------------------------------------------------------------


def strdoy(dtime):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 11.22.2022
    #
    # This function returns string of day stamp
    #
    # Variables:
    # datetime = datetime object
    #
    # Result:
    # date_out = string YYYYDOY
    # **************************************************************************
    # **************************************************************************
    date_out = dtime.strftime('%Y%j')
# -----------------------------------------------------------------------
    return(date_out)
# -----------------------------------------------------------------------


def check_path(path):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 11.22.2022
    #
    # This function checks if the path exists, if not creates it, if yes then
    # deletes all files in the path
    #
    # Variables:
    # path = path (string)
    # **************************************************************************
    # **************************************************************************
    if os.path.exists(path) is False:
        print('Directory ', path, ' does no exist.')
        print('Creating directory: ', path)
        os.mkdir(path)
# -----------------------------------------------------------------------
    return()
# -----------------------------------------------------------------------


def set_grid(dMLT, dQlat):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 11.23.2022
    #
    # This function sets a grid for the given dMLT and dLat
    #
    # Variables:
    # dMLT = time step (hours) as longitudinal cell size, 1h = 15 deg
    # dQlat = latitudinal cell size in Quasi-Dipole Coordinates (degrees)
    #
    # Result:
    # aMLT = flattened MLT coordinates of the grid
    # aQDlat = flattened QDlat coordinates of the grid
    # aMLT_2d = 2d MLT coordinates
    # aQDlat_2d = 2d QDlat coordinates
    # **************************************************************************
    # **************************************************************************
    aMLT_2d, aQDlat_2d = np.mgrid[0:24+dMLT:dMLT, -90:90+dQlat:dQlat]
    aMLT = np.reshape(aMLT_2d, aMLT_2d.size)
    aQDlat = np.reshape(aQDlat_2d, aQDlat_2d.size)
# -----------------------------------------------------------------------
    return(aMLT, aQDlat, aMLT_2d, aQDlat_2d)
# -----------------------------------------------------------------------


def set_geo_grid(dlon, dlat):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 03.14.2022
    #
    # This function sets a grid for the given dlat and dlon
    #
    # Variables:
    # dlon = longitudinal cell size in Geographic Coordinates (degrees)
    # dlat = latitudinal cell size in Geographic Coordinates (degrees)
    #
    # Result:
    # alon = flattened lon coordinates of the grid
    # alat = flattened lat coordinates of the grid
    # alon_2d = 2d lon coordinates
    # alat_2d = 2d lat coordinates
    # **************************************************************************
    # **************************************************************************
    alon_2d, alat_2d = np.mgrid[-180:180+dlon:dlon, -90:90+dlat:dlat]
    alon = np.reshape(alon_2d, alon_2d.size)
    alat = np.reshape(alat_2d, alat_2d.size)
# -----------------------------------------------------------------------
    return(alon, alat, alon_2d, alat_2d)
# -----------------------------------------------------------------------


def set_alt_grid(dalt):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 03.14.2022
    #
    # This function sets alt grid
    #
    # Variables: dalt = alt cell size (km)
    #
    # Result:    aalt = alt grid
    # **************************************************************************
    # **************************************************************************
    aalt = np.mgrid[90:1000+dalt:dalt]
# -----------------------------------------------------------------------
    return(aalt)
# -----------------------------------------------------------------------


def set_temporal_array(dUT):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 11.23.2022
    #
    # This function sets a time array for data assimilation with given time
    # step
    #
    # Variables:
    # dUT = time step (hours)
    #
    # Result:
    # aUT = array of UTs (hours)
    # ahour = integer array of hours
    # aminute = integer array of minutes
    # asecond = integer array of seconds
    # atime_frame_strings = string array of time stamps HHMM
    # **************************************************************************
    # **************************************************************************
    aUT = np.arange(0, 24, dUT)
    ahour = np.fix(aUT).astype(int)
    aminute = ((aUT - ahour)*60.).astype(int)
    asecond = (aUT*0).astype(int)
    atime_frame_strings = [str(ahour[it]).zfill(2)+str(aminute[it]).zfill(2)
                           for it in range(0, aUT.size)]
# -----------------------------------------------------------------------
    return(aUT, ahour, aminute, asecond, atime_frame_strings)
# -----------------------------------------------------------------------


def solar_interpolate(F_min, F_max, F107):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 11.29.2022
    #
    # This function interpolates the filed between its min and max to the given
    # F10.7 value by converting it first to IG12 and using IG12 limits of 0 and
    # 100.
    #
    # Variables:
    # F_min = array for minimun level of solar activity
    # F_max = array for maximum level of solar activity
    # F107 = given F10.7
    #
    # Result:
    # F = interpolated field to the given F10.7 [ngrid]
    # **************************************************************************
    # **************************************************************************
    # min and max of IG12 Ionospheric Global Index
    IG12_min = 0
    IG12_max = 100

    IG12 = F107_2_IG12(F107)

    # linear interpolation of the whole matrix:
    # https://en.wikipedia.org/wiki/Linear_interpolation
    F = (F_min*(IG12_max-IG12)/(IG12_max-IG12_min) +
         F_max*(IG12-IG12_min)/(IG12_max-IG12_min))
# -----------------------------------------------------------------------
    return(F)
# -----------------------------------------------------------------------


def solar_parameter(dtime, driver_dir):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 01.06.2023
    #
    # This function reads F10.7 data file and saves and gives the value for
    # the day
    #
    # Variables:
    # dtime = dtime object
    #
    # Result:
    # F107 = index
    # **************************************************************************
    # **************************************************************************
    doy = dtime.timetuple().tm_yday
    year = dtime.year
    filenam = os.path.join(driver_dir, 'solar_drivers',
                           'Solar_Driver_F107.txt')
    f = open(filenam, mode='r')
    table = np.genfromtxt(f, delimiter='', skip_header=7)
    a = np.where((table[:, 0] == year) & (table[:, 1] == doy))
    F107 = float(table[a[0], 3])
    # -----------------------------------------------------------------------
    return(F107)
    # -----------------------------------------------------------------------


def freq2den(f):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 01.11.2023
    #
    # This function converts plasma frequency to density
    #
    # Variables:
    # f = frequency
    #
    # Result:
    # d = density
    # **************************************************************************
    # **************************************************************************
    d = 0.124*f**2
    # -----------------------------------------------------------------------
    return(d)
    # -----------------------------------------------------------------------


def R12_2_F107(R12):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 01.11.2023
    #
    # This function converts R12 (or RZ12 as in IRI) sunspot number to F10.7
    #
    # Variables:
    # R12 = sunspot number
    #
    # Result:
    # F10.7 = solar flux
    # **************************************************************************
    # **************************************************************************
    F107 = 63.7 + 0.728*R12 + 8.9E-4*R12**2
    # -----------------------------------------------------------------------
    return(F107)
    # -----------------------------------------------------------------------


def F107_2_R12(F107):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 01.11.2023
    #
    # This function converts F10.7 to R12 sunspot number
    #
    # Variables:
    # F10.7 = Solar flux
    #
    # Result:
    # R12 = sunspot number
    # **************************************************************************
    # **************************************************************************
    a = 8.9E-4
    b = 0.728
    c = 63.7-F107
    x = quadratic([a, b, c])[0]
    # -----------------------------------------------------------------------
    return(x)
    # -----------------------------------------------------------------------


def F107_2_AZR(f107):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 06.23.2023
    #
    # This function converts F10.7 to effective sunspot number
    # (NeQuick guide Eqn. 19)
    #
    # Variables:
    # F10.7 = Solar flux
    #
    # Result:
    # AZR = effective sunspot number
    # **************************************************************************
    # **************************************************************************
    x = np.sqrt(167273. + (f107 - 63.7)*1123.6) - 408.99
    # -----------------------------------------------------------------------
    return(x)
    # -----------------------------------------------------------------------


def R12_2_IG12(R12):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 05.03.2023
    #
    # This function converts R12 (or RZ12 as in IRI) to ionospheric global
    # index IG12
    #
    # Variables:
    # R12 = sunspot number
    #
    # Result:
    # IG12 = ionospheric global index
    # **************************************************************************
    # **************************************************************************
    IG12 = 12.349 + 1.468*R12 - 0.00268*R12**2
    # -----------------------------------------------------------------------
    return(IG12)
    # -----------------------------------------------------------------------


def IG12_2_R12(IG12):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich

    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 05.03.2023
    #
    # This function converts IG12 to R12
    #
    # Variables:
    # IG12 = ionospheric global index
    #
    # Result:
    # R12 = sunspot number
    # **************************************************************************
    # **************************************************************************
    a = -0.00268
    b = 1.468
    c = 12.349 - IG12

    x = quadratic([a, b, c])[0]
    # -----------------------------------------------------------------------
    return(x)
    # -----------------------------------------------------------------------


def F107_2_IG12(F107):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 05.03.2023
    #
    # This function converts F10.7 to ionospheric global index IG12
    # (as in IRI)
    #
    # Variables:
    # F107 = sunspot number
    #
    # Result:
    # IG12 = ionospheric global index
    # **************************************************************************
    # **************************************************************************
    R12 = F107_2_R12(F107)
    IG12 = R12_2_IG12(R12)
    # -----------------------------------------------------------------------
    return(IG12)
    # -----------------------------------------------------------------------


def IG12_2_F107(IG12):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 05.03.2023
    #
    # This function converts IG12 to F10.7
    #
    # Variables:
    # IG12 = ionospheric global index
    #
    # Result:
    # F107 = sunspot number
    # **************************************************************************
    # **************************************************************************
    R12 = IG12_2_R12(IG12)
    F107 = R12_2_F107(R12)
    # -----------------------------------------------------------------------
    return(F107)
    # -----------------------------------------------------------------------


def quadratic(coeff):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 05.03.2023
    #
    # This function solves quadratic equation from abc
    #
    # Variables:
    # [a,b,c] = coefficients to quadratic equation
    #
    # Result:
    # [root1, root2] = solutions
    # **************************************************************************
    # **************************************************************************
    a = coeff[0]
    b = coeff[1]
    c = coeff[2]

    d = (b**2) - (4*a*c)

    # find two solutions
    root1 = (-b+np.sqrt(d))/(2*a)
    root2 = (-b-np.sqrt(d))/(2*a)
    # -----------------------------------------------------------------------
    return([root1, root2])
    # -----------------------------------------------------------------------


def epstein_function_array(A1, hm, B_bot, x):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 02.23.2022
    #
    # This function constructs density epstein profile. It is used in ionosonde
    # and RO data aggregators.
    #
    # Variables:
    # A1, hm, B_bot
    # x = height array (km)
    #
    # Result:
    # density = density (m-3)
    # **************************************************************************
    # **************************************************************************

    density = np.zeros((A1.shape))

    alpha = (x-hm)/B_bot
    exp = fexp(alpha)

    a = np.where(alpha <= 25)
    b = np.where(alpha > 25)
    density[a] = A1[a]*exp[a]/(1+exp[a])**2
    density[b] = 0.
# -----------------------------------------------------------------------
    return(density)
# -----------------------------------------------------------------------


def epstein_function_top_array(A1, hmF2, B_F2_top, x):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 02.23.2022
    #
    # This function constructs density epstein profile for the topside of F2
    # layer. It is used in ionosonde and RO data aggregators.
    #
    # Variables:
    # A1, hmF2, B_F2_top
    # x = height array (km)
    #
    # Result:
    # density = density (m-3)
    # **************************************************************************
    # **************************************************************************
    g = 0.125
    r = 100
    dh = x - hmF2
    z = dh/(B_F2_top*(1 + (r*g*dh/(r*B_F2_top+g*dh))))
    exp = fexp(z)

    density = np.zeros((exp.size))
    density[exp > 1e11] = A1[exp > 1e11]/exp[exp > 1e11]
    density[exp <= 1e11] = (A1[exp <= 1e11] *
                            exp[exp <= 1e11] /
                            (exp[exp <= 1e11] + 1)**2)
# -----------------------------------------------------------------------
    return(density)
# -----------------------------------------------------------------------


def amplitudes_array(NmF2, NmF1, NmE, hmF2, hmF1, hmE, B_E_top, B_F1_bot,
                     B_F2_bot):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 04.06.2023
    #
    # This function returns amplitudes for epstein functions from given Nms
    #
    # Variables: NmF2, NmF1, NmE, hmF2, hmF1, hmE, B_E_top, B_F1_bot, B_F2_bot
    #
    # Result:
    # A1 = F2 region amplitude
    # A2 = F1 region amplitude
    # A4 = E region amplitude
    # **************************************************************************
    # **************************************************************************

    # Amplitude of Epstein function should be multiplyed by 4
    A1 = 4.*NmF2
    A2a = 4.*NmF1
    A3a = 4.*NmE

    for i in range(0, 5):
        F2_contribution = epstein_function_array(A1, hmF2, B_F2_bot, hmF1)
        E_contribution = epstein_function_array(A3a, hmE, B_E_top, hmF1)

        a = np.where(NmF1 > (F2_contribution + E_contribution))
        A2a[a] = NmF1[a] - F2_contribution[a]-E_contribution[a]

        F1_contribution = epstein_function_array(A2a, hmF1, B_F1_bot, hmE)
        F2_contribution = epstein_function_array(A1, hmF2, B_F2_bot, hmE)

        a = np.where(NmE > (F1_contribution + F2_contribution))
        A3a[a] = NmE[a] - F1_contribution[a] - F2_contribution[a]

    A2 = A2a
    A3 = A3a
# -----------------------------------------------------------------------
    return(A1, A2, A3)
# -----------------------------------------------------------------------


def drop_function(x):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 01.20.2023
    #
    # This is a drop function from a simple family of curve. It is used to
    # reduce the F1_top contribution for the F2_bot region, so that when the
    # summation of epsein functions is performed, the presence of F1 region
    # would not mess up with the value of NmF2
    #
    # Result: y
    # **************************************************************************
    # **************************************************************************

    # degree of the drop (4 give a good result)
    n = 4
    nelem = x.size

    if nelem > 1:
        y = 1-(x/(nelem-1))**n
    else:
        y = np.zeros((x.size))+1
    # -----------------------------------------------------------------------
    return(y)
    # -----------------------------------------------------------------------


def reconstruct_density_from_parameters(F2, F1, E, alt):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 07.13.2023
    #
    # This function calculates 3-D density field from given dictionaries of
    # the parameters for 2 levels of solar activity
    #
    # Variables:
    # dictionaries F2, F1, E
    # alt = 1-D array of altitudes (NumPy array) {km} [N_V]
    #
    # Result:
    # x_out = electron density for 2 levels of solar activity (NumPy array)
    # {m-3}, [2, N_T, N_V, N_G]
    # **************************************************************************
    # **************************************************************************

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
# -----------------------------------------------------------------------
    return(x_out)
# -----------------------------------------------------------------------


def EDP_builder(x, aalt):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 07.13.2023
    #
    # This function builds the EDP from the provided parameters
    #
    # Variables:
    # x = array where 1st dimention indicates the parameter, second dimension
    # is time, and third is horizontal grid (NumPy array) [11, N_T, N_G]
    #
    # aalt = 1-D array of altitudes (NumPy array) [N_V]
    #
    # Result:
    # density_out = (NumPy array) [N_T, N_V, N_G]
    # **************************************************************************
    # **************************************************************************
    # number of elements in time dimention
    nUT = x.shape[1]
    # number of elements in horizontal dimention of grid
    nhor = x.shape[2]
    # time and horisontal grid dimention
    ngrid = nhor*nUT
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

    # array of hmFs with same dimentions as result, to later search
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

    a_A1 = 4.*a_NmF2
    a_A2 = 4.*a_NmF1
    a_A3 = 4.*a_NmE

    # !!! do not use a[0], because all 3 dimentions are needed. this is
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
    a = np.where((np.isfinite(a_NmF1)) &
                 (a_alt < a_hmF2) &
                 (a_alt >= a_hmF1))
    density_F2[a] = epstein_function_array(a_A1[a], a_hmF2[a],
                                           a_B_F2_bot[a], a_alt[a])

    # E top plus F1 bottom (hard boundaries)
    a = np.where((a_alt > a_hmE) & (a_alt < a_hmF1))
    drop_1[a] = 1. - ((a_alt[a]-a_hmE[a])/(a_hmF1[a]-a_hmE[a]))**4.
    drop_2[a] = 1. - ((a_hmF1[a]-a_alt[a])/(a_hmF1[a]-a_hmE[a]))**4.

    density_E[a] = epstein_function_array(a_A3[a],
                                          a_hmE[a],
                                          a_B_E_top[a],
                                          a_alt[a])*drop_1[a]
    density_F1[a] = epstein_function_array(a_A2[a],
                                           a_hmF1[a],
                                           a_B_F1_bot[a],
                                           a_alt[a])*drop_2[a]

    # when F1 is not present(hard boundaries)--------------------
    a = np.where((np.isnan(a_NmF1)) &
                 (a_alt < a_hmF2) &
                 (a_alt > a_hmE))
    drop_1[a] = 1. - ((a_alt[a]-a_hmE[a])/(a_hmF2[a]-a_hmE[a]))**4.
    drop_2[a] = 1. - ((a_hmF2[a]-a_alt[a])/(a_hmF2[a]-a_hmE[a]))**4.
    density_E[a] = epstein_function_array(a_A3[a],
                                          a_hmE[a],
                                          a_B_E_top[a],
                                          a_alt[a])*drop_1[a]
    density_F2[a] = epstein_function_array(a_A1[a],
                                           a_hmF2[a],
                                           a_B_F2_bot[a],
                                           a_alt[a])*drop_2[a]

    density = density_F2 + density_F1 + density_E

    density_out = np.reshape(density, (nalt, nUT, nhor), order='F')
    density_out = np.swapaxes(density_out, 0, 1)

    # make 1 everything that is <= 0
    density_out[np.where(density_out <= 1)] = 1.
# -----------------------------------------------------------------------
    return(density_out)
# -----------------------------------------------------------------------


def day_of_the_month_corr(year, month, day):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 07.14.2023
    #
    # This function finds 2 months around the given day and calculates
    # fractions of influence
    #
    # Variables:
    # year = year (ineteger)
    # month = month (ineteger)
    # day = day (ineteger)
    #
    # Result:
    # month_before = month before given day (ineteger)
    # month_after = month after given day (ineteger)
    # fraction1 = fractional influence of month before (float)
    # fraction2 = fractional influency of month after (float)
    # **************************************************************************
    # **************************************************************************

    # middles of the months around
    delta_month = dt.timedelta(days=+30)
    dtime0 = dt.datetime(year, month, 15)
    dtime1 = dtime0-delta_month
    dtime2 = dtime0+delta_month

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
    dt1 = time_event-t_before
    dt2 = t_after-time_event
    dt3 = t_after-t_before

    # fractions of the influence for the interpolation
    fraction1 = dt2.days/dt3.days
    fraction2 = dt1.days/dt3.days
    # -----------------------------------------------------------------------
    return(t_before, t_after, fraction1, fraction2)
    # -----------------------------------------------------------------------


def fractional_correction_of_dictionary(fraction1, fraction2, F_before,
                                        F_after):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 07.14.2023
    #
    # This function interpolates between 2 middles of consequent months to the
    # specified day
    #
    # Variables:
    # fraction1 = fractional influence of month before (float)
    # fraction2 = fractional influency of month after (float)
    # F_before = dictionary of average parameters for a month before
    # F_after = dictionary of average parameters for a month after
    #
    # Result:
    # F_new = parameters adjusted by fractions
    # **************************************************************************
    # **************************************************************************
    F_new = F_before
    for key in F_before:
        F_new[key] = F_before[key]*fraction1 + F_after[key]*fraction2
    # -----------------------------------------------------------------------
    return(F_new)
    # -----------------------------------------------------------------------


def solar_interpolation_of_dictionary(F, F107):
    # **************************************************************************
    # **************************************************************************
    # by Victoriya V Forsythe Makarevich
    #
    # Naval Research Laboratory
    # Space Physics
    #
    # Date: 07.14.2023
    #
    # This function interpolates between 2 middles of consequent months to the
    # specified day
    #
    # Variables:
    # fraction1 = fractional influence of month before (float)
    # fraction2 = fractional influency of month after (float)
    # F_before = dictionary of average parameters for a month before
    # F_after = dictionary of average parameters for a month after
    #
    # Result:
    # F_new = parameters adjusted by fractions
    # **************************************************************************
    # **************************************************************************

    # min and max of IG12 Ionospheric Global Index
    IG12_min = 0
    IG12_max = 100
    IG12 = F107_2_IG12(F107)

    F_new = F

    for key in F:
        F_key = F[key]
        F_key = np.swapaxes(F_key, 0, 2)
        F_min = F_key[0, :]
        F_max = F_key[1, :]

        F_new[key] = (F_min*(IG12_max-IG12)/(IG12_max-IG12_min) +
                      F_max*(IG12-IG12_min)/(IG12_max-IG12_min))

        F_new[key] = np.swapaxes(F_new[key], 0, 1)
    # -----------------------------------------------------------------------
    return(F_new)
