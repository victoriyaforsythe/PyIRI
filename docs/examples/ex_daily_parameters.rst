Example 2: Daily Ionospheric Parameters
=======================================

PyIRI can calculate daily ionospheric parameters for the user provided grid.
The estimation of the parameters occurs simultaneously at all grid points
and for all desired diurnal time frames. The parameters are obtained by
linearly interpolating between min and max levels of solar activity, and
by interpolation between median monthly values to the day of interest.

1. Import libraries:

::


   import numpy as np
   import PyIRI
   import PyIRI.edp_update as ml
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

5. Create any horizontal grid (regular or irregular, global or regional).
   The grid arrays (alon and alat) should be flattened to be 1-D arrays. 
   This is an example of a regular global grid:

::

   dlon = 5
   dlat = 5
   alon_2d, alat_2d = np.mgrid[-180:180 + dlon:dlon, -90:90 + dlat:dlat]
   alon = np.reshape(alon_2d, alon_2d.size)
   alat = np.reshape(alat_2d, alat_2d.size)

6. Create any temporal array expressed in decimal hours (regular or irregular).
   For this example we use regularly spaced time array:

::

   hr_res = 1
   ahr = np.arange(0, 24, hr_res)

8. Create height array. It can be regular or irregular.
   Here is an example for regularly spaced array:

::

   alt_res = 10
   alt_min = 90
   alt_max = 700
   aalt = np.arange(alt_min, alt_max, alt_res)
   
9. Find ionospheric parameters for F2, F1, E, and Es regions by
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


es_peak dictionary contains:

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


10. Plot results for F2 region at time stamp UT = 10:

::

   UT_plot = 10
   plot_dir = '/Users/Documents/MY_FOLDER/'
   
   plot.PyIRI_plot_NmF2(f2, ahr, alon, alat, alon_2d, alat_2d, sun,
   UT_plot, plot_dir, plot_name='PyIRI_NmF2.png')


.. image:: Figs/PyIRI_NmF2.png
    :width: 600px
    :align: center
    :alt: Global distribution of NmF2.

::

   plot.PyIRI_plot_foF2(f2, ahr, alon, alat, alon_2d, alat_2d, sun,
   UT_plot, plot_dir, plot_name='PyIRI_foF2.png')


.. image:: Figs/PyIRI_foF2.png
    :width: 600px
    :align: center
    :alt: Global distribution of foF2.

::

   plot.PyIRI_plot_hmF2(f2, ahr, alon, alat, alon_2d, alat_2d, sun,
   UT_plot, plot_dir, plot_name='PyIRI_hmF2.png')


.. image:: Figs/PyIRI_hmF2.png
    :width: 600px
    :align: center
    :alt: Global distribution of hmF2.

1.  Plot results for F1 region:

::

   plot.PyIRI_plot_NmF1(f2, ahr, alon, alat, alon_2d, alat_2d, sun,
   UT_plot, plot_dir, plot_name='PyIRI_NmF1.png')


.. image:: Figs/PyIRI_NmF1.png
    :width: 600px
    :align: center
    :alt: Global distribution of NmF1.

::

   plot.PyIRI_plot_foF1(f1, ahr, alon, alat, alon_2d, alat_2d, sun,
   UT_plot, plot_dir, plot_name='PyIRI_foF1.png')


.. image:: Figs/PyIRI_foF1.png
    :width: 600px
    :align: center
    :alt: Global distribution of foF1.

::

   plot.PyIRI_plot_hmF1(f1, ahr, alon, alat, alon_2d, alat_2d, sun,
   UT_plot, plot_dir, plot_name='PyIRI_hmF1.png')


.. image:: Figs/PyIRI_hmF1.png
    :width: 600px
    :align: center
    :alt: Global distribution of hmF1.

1.  Plot results for E region:

::

   plot.PyIRI_plot_foE(e_peak, ahr, alon, alat, alon_2d, alat_2d, sun,
   UT_plot, plot_dir, plot_name='PyIRI_foE.png')


.. image:: Figs/PyIRI_foE.png
    :width: 600px
    :align: center
    :alt: Global distribution of foE.

1.  Plot results for Es region:

::

   plot.PyIRI_plot_foEs(es_peak, ahr, alon, alat, alon_2d, alat_2d, sun,
   UT_plot, plot_dir, plot_name='PyIRI_foEs.png')


.. image:: Figs/PyIRI_foEs.png
    :width: 600px
    :align: center
    :alt: Global distribution of foEs for min and max levels of solar activity.

1.  Plot electron density vertical profiles from one location.
    Make sure this location belongs to alon and alat arrays.

::

   lon_plot = 0
   lat_plot = 0
   plot.PyIRI_EDP_sample_1day(edp, ahr, alon, alat, lon_plot, lat_plot, aalt,
   UT_plot, plot_dir, plot_name='PyIRI_EDP_sample_1day.png')



.. image:: Figs/PyIRI_EDP_sample_1day.png
    :width: 600px
    :align: center
    :alt: Electron density profile for 1 location.

