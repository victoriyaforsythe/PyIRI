.. _exthree:

Example 3: Daily Ionospheric Parameters For a Single Location
=============================================================

In case you are interested in a single location (as opposed to the grid)
PyIRI can evaluate parameters at this location if it is passed as a 1-element
NumPy array. 

1. Import libraries:

::


   import numpy as np
   import PyIRI
   import PyIRI.main_library as ml
   import PyIRI.plotting as plot

2. Specify a year, a month, and a day:

::


   year = 2020
   month = 4
   day = 1

3. Specify solar flux index F10.7 in SFU:

::


   f107 = 100

4. Specify what coefficients to use for the peak of F2 layer:

   0 = CCIR, 1 = URSI

::


   ccir_or_ursi = 0

5. Create a horizontal grid with a single location of interest.
   E.g. lon = 10 E, lat = 20 N. 

::

   alon = np.array([10.])
   alat = np.array([20.])

6. Create any temporal array expressed in decimal hours (regular or irregular).
   For this example we use regularly spaced time array:

::

   hr_res = 1
   ahr = np.arange(0, 24, hr_res)

7. Create height array. It can be regular or irregular.
   Here is an example for regularly spaced array:

::

   alt_res = 10
   alt_min = 90
   alt_max = 700
   aalt = np.arange(alt_min, alt_max, alt_res)
   
8. Find ionospheric parameters for F2, F1, E, and Es regions by
   calling IRI_density_1day function:

::

   f2, f1, e_peak, es_peak, sun, mag, edp = ml.IRI_density_1day(year, month, day, ahr, alon, alat, aalt, f107, PyIRI.coeff_dir, ccir_or_ursi)

f2 dictionary contains:

-  peak density 'Nm' in m-3

-  critical frequency 'fo' in MHz

-  the obliquity factor for a distance of 3,000 km 'M3000'

-  height of the peak 'hm' in km

-  thickness of the topside 'B_top' in km

-  thickness of the bottomside 'B_bot' in km.


f1 dictionary contains:

-  peak density 'Nm' in m-3

-  critical frequency 'fo' in MHz

-  probability occurrence of F1 region 'P'

-  height of the peak 'hm' in km

-  thickness of the bottomside 'B_bot' in km.


e_peak dictionary contains:

-  peak density 'Nm' in m-3

-  critical frequency 'fo' in MHz

-  height of the peak 'hm' in km

-  thickness of the bottomside 'B_bot' in km.

-  thickness of the topside 'B_top' in km.


e_peak dictionary contains:

-  peak density 'Nm' in m-3

-  critical frequency 'fo' in MHz

-  height of the peak 'hm' in km

-  thickness of the bottomside 'B_bot' in km.

-  thickness of the topside 'B_top' in km.


sun dictionary contains:

-  longitude of subsolar point 'lon'

-  latitude of subsolar point 'lat'


mag dictionary contains:

-  magnetic field inclination in degrees 'inc'

-  modified dip angle in degrees 'modip'

-  magnetic dip latitude in degrees 'mag_dip_lat'


edp array:

-  3-D electron density with shape  [N_T, N_V, N_G]

9. Plot diurnal variation of the parameters for a single location:

::

   plot_dir = '/Users/Documents/MY_FOLDER/'

   # Make sure alon and alat contains lon_plot and lat_plot
   lon_plot = 10
   lat_plot = 20

   plot.PyIRI_plot_1location_diurnal_par(f2, f1, e_peak, es_peak, alon, alat,
   lon_plot, lat_plot, ahr, plot_dir, plot_name='PyIRI_diurnal.pdf')


.. image:: Figs/PyIRI_diurnal.pdf
    :width: 600px
    :align: center
    :alt: Diurnal variation of the ionospheric parameters.

10. Plot diurnal variation of electron density:

::

   plot.PyIRI_plot_1location_diurnal_density(edp, alon, alat, lon_plot, lat_plot,
   aalt, ahr, plot_dir, plot_name='PyIRI_EDP_diurnal.pdf')


.. image:: Figs/PyIRI_EDP_diurnal.pdf
    :width: 600px
    :align: center
    :alt: Diurnal variation of the ionospheric parameter.
