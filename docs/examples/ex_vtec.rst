Example 4: Calculate Vertical TEC
=================================

The Vertical Total Electron Content (VTEC) is arguably the most common
ionospheric measurement.  It is the column integrated electron density, between
a transmitting satellite, most frequently one belonging to a Global Navigation
Satellite System (GNSS) constellation, and a receiver.  TEC is measured in
TEC Units (TECU), which are equal to 1.0e16 electrons per square meter.  PyIRI
provides a function to calculate VTEC from an electron density profile:
:py:func:`PyIRI.main_library.edp_to_vtec`.

This example is a follow-on from :ref:`exthree`. Using the electron density
profile and altitude grid from the previous example:

::
   vtec = ml.edp_to_vtec(edp, aalt)

   print(vtec.shape, vtec.min(), vtec.max())

This will yeild an array with VTEC at 24 times and one location, with values
between 4 and 45 TECU::

  (24, 1) 4.336832567501178 44.44731135732205

