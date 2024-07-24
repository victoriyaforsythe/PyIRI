Example 5: Terminal Friendly PyIRI Runs
=======================================

In case you want to use a regularly spaced global grid and to run
PyIRI from the terminal, you can use run_iri_reg_grid function.
Here is an example for year = 2020, month = 4, day =15, F10.7 = 150.

::


   import PyIRI.main_library as ml


   alon, alat, alon_2d, alat_2d, aalt, ahr, f2, f1, epeak, es_peak, \ 
       sun, mag, edens_prof = ml.run_iri_reg_grid(
       2020, 4, 15, 150, hr_res=1, lat_res=1, lon_res=1,
       alt_res=10, alt_min=0, alt_max=700, ccir_or_ursi=0)

The result "edens_prof" will have shape [12, 70, 2701]
