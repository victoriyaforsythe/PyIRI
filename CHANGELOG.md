# Change Log
All notable changes to this project are documented in this file. This project
adheres to [Semantic Versioning](https://semver.org/).

## 0.1.5 (11-25-2025)
* Fixed bug in F1 layer formula.

## 0.1.4 (11-20-2025)
* Updated formulation of F1 layer that is now consistent with SAMI-4.

## 0.1.3 (11-06-2025)
* Order of linear interpolation changed. Parameters are now interpolated in fo, and converted to Nm.
  
## 0.1.1 (11-05-2025)
* Added conversion to numpy array in Apex function
* Corrected for linear interpolation of foF2
* Removed examples
* Updated all tutorials
* Fixed bug in coordinate transform at the pole

## 0.1.0 (10-29-2025)
* Added an EDP builder derived from PyIRTAM to build the bottomside F2 layer using B0 and B1.
* Added a tutorial to generate SH EDP parameters.
* Added pandas to the list of dependencies.
* Added unit tests for the new spherical harmonics features.
* Added spherical harmonics reconstruction features for IRI EDP parameters (foF2, hmF2, B0, B1, foEs, M3000(F2)).
* Increased min python version to 3.10 (which is a requirement for some scipy functions).
* Added a new capability to do the Apex coordinate transformation without ApexPy.
* Added unit tests for Apex_library.
* Corrected foEs interpolation method and updated test data and example figures.
* Deprecated `output_quartiles` argument in `read_ccir_ursi_coeff()` and replaced it with `output_deciles` (functionality is unchanged).
* Typos are fixed in the Es section.
* Added new hmF1 derivation based on B0 parameter.

## 0.0.4 (06-04-2025)
* Limit to the peaks of density is set so it doesn't go negative anywhere

## 0.0.3 (05-27-2025)
* Added magnetic filed strength and orientation as an output for IGRF library.
* Added altitude as an optional input to the IGRF function.
* Updated the README to include more links and information.
* Updated the documentation to include more citation information, correct
  the installation instructions, and improve the examples.
* Added logger and fixed merge conflicts between @aburrell and
  @victoriyaforsythe develop branches
* Added a function to calculate vertical TEC
* New library edp_update is added that introduces a new approach for the F1
  region modeling, where all fields are now continuous. This is important for
  the raytracing application.
* The construction of the EDP in edp_update was modified to improve the
  reliability of the code.
* Coverage status is added.
* Tests are added for edp_update.
* Regression tests are added for monthly_mean function.
* Regression tests are added for single_location function.
* Test data files are added.
* Plots are refreshed in the Docs.

## 0.0.2 (11-15-2023)
* The order of interpolation was changed. Previously the EDPs were
  constructed for 2 levels of solar activity and then interpolated in between.
  This was changed to the interpolation between parameters and the
  construction of the EDP for the final result.
* Changed E-region height from 120 km to 110 km.
* Added a link in the docs to the journal publication.
* Print statements were deleted or replaced with logging messages.

## 0.0.1 (08-10-2023)
* Alpha release
