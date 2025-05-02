Example 1: Monthly Mean Ionospheric Parameters
==============================================

PyIRI can calculate monthly mean ionospheric parameters
for the user provided grid. The estimation of the parameters
occurs simultaneously at all grid points and for all
desired diurnal time frames.

1. Import libraries:

::


   import numpy as np
   import PyIRI
   import PyIRI.main_library as ml
   import PyIRI.plotting as plot

2. Specify a year and a month:

::


   year = 2020
   month = 4

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

7. Find ionospheric parameters for F2, F1, E, and Es regions by
   calling IRI_monthly_mean_par function:

::

   f2, f1, e_peak, es_peak, sun, mag = ml.IRI_monthly_mean_par(year, month, ahr, alon, alat, PyIRI.coeff_dir, ccir_or_ursi)

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


8. Plot results for F2 region at time stamp UT = 10:

::

   UT_plot = 10
   plot_dir = '/Users/Documents/MY_FOLDER/'
   
   plot.PyIRI_plot_NmF2_min_max(f2, ahr, alon, alat, alon_2d, alat_2d, sun,
   UT_plot, plot_dir, plot_name='PyIRI_NmF2_min_max.png')


.. image:: /docs/examples/Figs/PyIRI_NmF2_min_max.png
    :width: 600px
    :align: center
    :alt: Global distribution of NmF2 for min and max levels of solar activity.

::

   plot.PyIRI_plot_foF2_min_max(f2, ahr, alon, alat, alon_2d, alat_2d, sun,
   UT_plot, plot_dir, plot_name='PyIRI_foF2_min_max.png')


.. image:: /docs/examples/Figs/PyIRI_foF2_min_max.png
    :width: 600px
    :align: center
    :alt: Global distribution of foF2 for min and max levels of solar activity.

::

   plot.PyIRI_plot_M3000_min_max(f2, ahr, alon, alat, alon_2d, alat_2d, sun,
   UT_plot, plot_dir, plot_name='PyIRI_M3000_min_max.png')


.. image:: /docs/examples/Figs/PyIRI_M3000_min_max.png
    :width: 600px
    :align: center
    :alt: Global distribution of M3000 for min and max levels of solar activity.

::

   plot.PyIRI_plot_hmF2_min_max(f2, ahr, alon, alat, alon_2d, alat_2d, sun,
   UT_plot, plot_dir, plot_name='PyIRI_hmF2_min_max.png')


.. image:: /docs/examples/Figs/PyIRI_hmF2_min_max.png
    :width: 600px
    :align: center
    :alt: Global distribution of hmF2 for min and max levels of solar activity.

9. Plot results for F1 region:

::

   plot.PyIRI_plot_NmF1_min_max(f1, ahr, alon, alat, alon_2d, alat_2d, sun,
   UT_plot, plot_dir, plot_name='PyIRI_NmF1_min_max.png')


.. image:: /docs/examples/Figs/PyIRI_NmF1_min_max.png
    :width: 600px
    :align: center
    :alt: Global distribution of NmF1 for min and max levels of solar activity.

::

   plot.PyIRI_plot_foF1_min_max(f1, ahr, alon, alat, alon_2d, alat_2d, sun,
   UT_plot, plot_dir, plot_name='PyIRI_foF1_min_max.png')


.. image:: /docs/examples/Figs/PyIRI_foF1_min_max.png
    :width: 600px
    :align: center
    :alt: Global distribution of foF1 for min and max levels of solar activity.

::

   plot.PyIRI_plot_hmF1_min_max(f1, ahr, alon, alat, alon_2d, alat_2d, sun,
   UT_plot, plot_dir, plot_name='PyIRI_hmF1_min_max.png')


.. image:: /docs/examples/Figs/PyIRI_hmF1_min_max.png
    :width: 600px
    :align: center
    :alt: Global distribution of hmF1 for min and max levels of solar activity.

10. Plot results for E region:

::

   plot.PyIRI_plot_foE_min_max(e_peak, ahr, alon, alat, alon_2d, alat_2d, sun,
   UT_plot, plot_dir, plot_name='PyIRI_foE_min_max.png')


.. image:: /docs/examples/Figs/PyIRI_foE_min_max.png
    :width: 600px
    :align: center
    :alt: Global distribution of foE for min and max levels of solar activity.

11. Plot results for Es region:

::

   plot.PyIRI_plot_foEs_min_max(es_peak, ahr, alon, alat, alon_2d, alat_2d, sun,
   UT_plot, plot_dir, plot_name='PyIRI_foEs_min_max.png')


.. image:: /docs/examples/Figs/PyIRI_foEs_min_max.png
    :width: 600px
    :align: center
    :alt: Global distribution of foEs for min and max levels of solar activity.

12. Create height array. It can be regular or irregular.
Here is an example for regularly spaced array:

::

   alt_res = 10
   alt_min = 90
   alt_max = 700
   aalt = np.arange(alt_min, alt_max, alt_res)

13. Construct electron density form the parameters:

::

   edens_prof = ml.reconstruct_density_from_parameters(f2, f1, e_peak, aalt)
   print('edens_prof has shape: ', edens_prof.shape)

14. Plot electron density vertical profiles from one location.
    Make sure this location belongs to alon and alat arrays.

::

   lon_plot = 0
   lat_plot = 0
   plot.PyIRI_EDP_sample(edens_prof, ahr, alon, alat, lon_plot, lat_plot, aalt,
   UT_plot, plot_dir, plot_name='PyIRI_EDP_sample.png')



.. image:: /docs/examples/Figs/PyIRI_EDP_sample.png
    :width: 600px
    :align: center
    :alt: EDPs for min and max of solar activity.



