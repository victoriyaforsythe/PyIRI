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

Make sure to use sufficient vertical resolution and the maximum height for
this calculation.

The following matrix approach can be implemented to calculate TEC for the
PyIRI output array of density edens_prof:

# Find dimentions:
N_time = edens_prof.shape[0]
N_alt = edens_prof.shape[1]
N_grid = edens_prof.shape[2]

# Array of distances between vertical grid cells in meters
dist = np.concatenate((np.diff(aalt), aalt[-1] - aalt[-2]), axis=None) * 1000.

# Convert dist to a matrix, to perform the array multipllication
dist_extended = np.reshape(np.full((N_time, N_grid, N_alt), dist), (N_time, N_alt, N_grid))

# Find TEC in shape [N_time, N_grid], result is in TECU
tec = np.sum(den * dist_extended, axis=1) / 1e16