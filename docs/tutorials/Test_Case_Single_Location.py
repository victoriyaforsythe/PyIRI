# # Daily IRI Parameters for given level of solar activity for a single location
# 
# In case you are interested in a single location (as opposed to the grid)
# PyIRI can evaluate parameters at this location if it is passed as a 1-element
# NumPy array. 
# 
# 1. Import libraries:
import numpy as np
import PyIRI
# Use this module for new EDP update
import PyIRI.edp_update as ml
# import PyIRI.main_library as ml
import PyIRI.plotting as plot

# 2. Specify a year, a month, and a day:

year = 2020
month = 4
day = 1

# 3. Specify solar flux index F10.7 in SFU:

f107 = 100

# 4. Specify what coefficients to use for the peak of F2 layer:
ccir_or_ursi = 0

# 5. Create a horizontal grid with a single location of interest.
# E.g. lon = 10 E, lat = 20 N. 

alon = np.array([10.])
alat = np.array([20.])

# 6. Create any temporal array expressed in decimal hours (regular or irregular).
# In case you are interested in only one time frame, it also needs to be presented as
# NumPy array with a single element. 
# For this example we are using regularly spaced time array:

hr_res = 0.25
ahr = np.arange(0, 24, hr_res)

# 12. Create height array. It can be regular or irregular.
# Here is an example for regularly spaced array:

alt_res = 10
alt_min = 90
alt_max = 700
aalt = np.arange(alt_min, alt_max, alt_res)

# 7. Find ionospheric parameters for F2, F1, E, and Es regions by
# calling IRI_density_1day function:
f2, f1, e_peak, es_peak, sun, mag, edp = ml.IRI_density_1day(year, month, day, ahr, alon, alat, aalt, f107, PyIRI.coeff_dir, ccir_or_ursi)

# 8. Plot results for F2 region at time stamp UT=10:

plot_dir = '/Users/vmakarevich/Documents/Science_VF2/PyIRI_Test/'

# Make sure alon and alat contains lon_plot and lat_plot
lon_plot = 10
lat_plot = 20
plot.PyIRI_plot_1location_diurnal_par(f2, f1, e_peak, es_peak, alon, alat,
lon_plot, lat_plot, ahr, plot_dir,
plot_name='PyIRI_diurnal.png')


test_data = {'NmF2': np.array(f2['Nm']).tolist(),
             'hmF2': np.array(f2['hm']).tolist(),
             'B_F2_top': np.array(f2['B_top']).tolist(),
             'B_F2_bot': np.array(f2['B_bot']).tolist(),
             'NmF1': np.array(f1['Nm']).tolist(),
             'hmF1': np.array(f1['hm']).tolist(),
             'B_F1_bot': np.array(f1['B_bot']).tolist(),
             'NmE': np.array(e_peak['Nm']).tolist(),
             'hmE': np.array(e_peak['hm']).tolist(),
             'B_E_top': np.array(e_peak['B_top']).tolist(),
             'B_E_bot': np.array(e_peak['B_bot']).tolist(),
             'NmEs': np.array(es_peak['Nm']).tolist(),
             'hmEs': np.array(es_peak['hm']).tolist(),
             'B_Es_top': np.array(es_peak['B_top']).tolist(),
             'B_Es_bot': np.array(es_peak['B_bot']).tolist(),
             'sun_lon': np.array(sun['lon']).tolist(),
             'sun_lat': np.array(sun['lat']).tolist(),
             'inc': np.array(mag['inc']).tolist(),
             'modip': np.array(mag['modip']).tolist(),
             'mag_dip_lat': np.array(mag['mag_dip_lat']).tolist()}

import json
with open("Test_Output.json", "w") as f:
    json.dump(test_data, f)



