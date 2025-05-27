Example 4: Calculate Vertical TEC
=================================

The Vertical Total Electron Content (VTEC) is arguably the most common
ionospheric measurement.  It is the column integrated electron density, between
a transmitting satellite, most frequently one belonging to a Global Navigation
Satellite System (GNSS) constellation, and a receiver.  TEC is measured in
TEC Units (TECU), which are equal to 1.0e16 electrons per square meter.  PyIRI
provides a function to calculate VTEC from an electron density profile:
:py:func:`PyIRI.main_library.edp_to_vtec`.

This example is a follow-on using the electron density
profile and altitude grid from the previous example:

::
   TEC = main.edp_to_vtec(edp, aalt, min_alt=0.0, max_alt=202000.0)
   plot.PyIRI_plot_vTEC(TEC, ahr, alon, alat, alon_2d, alat_2d, sun, UT_plot, plot_dir, plot_name='PyIRI_vTEC.png')

This will yeild an array with VTEC in shape [N_time, N_grid].

.. image:: Figs/PyIRI_vTEC.png
    :width: 600px
    :align: center
    :alt: vTEC map.