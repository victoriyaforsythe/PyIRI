# Change Log
All notable changes to this project are documented in this file. This project
adheres to [Semantic Versioning](https://semver.org/).

## 0.0.3 (XX-XX-2024)
* Added unit tests for the IGRF library functions.
* Added pytest and coveralls to main CI tests.
* Updated the README to include more links and information.
* Updated the documentation to include more citation information, correct
  the installation instructions, and improve the examples.
* Added logger and fixed merge conflicts between @aburrell and
  @victoriyaforsythe develop branches

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
