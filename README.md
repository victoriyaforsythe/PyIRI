<img width="200" height="200" src="https://raw.githubusercontent.com/victoriyaforsythe/PyIRI/main/docs/figures/PyIRI_logo.png" alt="Black circle with PyIRI logo of two snakes marking the EIA" title="PyIRI Logo" style="float:left;">

# PyIRI
[![PyPI Package latest release](https://img.shields.io/pypi/v/PyIRI.svg)](https://pypi.org/project/PyIRI/)
[![Build Status](https://github.com/victoriyaforsythe/PyIRI/actions/workflows/main.yml/badge.svg)](https://github.com/victoriyaforsythe/PyIRI/actions/workflows/main.yml)
[![Documentation Status](https://readthedocs.org/projects/pyiri/badge/?version=latest)](https://pyiri.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8235173.svg)](https://doi.org/10.5281/zenodo.8235173)

Python implementation of the International Reference Ionosphere (IRI) that
evaluates the electron density and associated ionospheric parameters on the
entire given global grid and for the entire day simultaneously. 

# Installation

PyIRI may be installed from PyPi, which will handle all dependencies.

```
pip install PyIRI
```

You may also clone and install PyIRI from the GitHub repository.

```
git clone https://github.com/victoriyaforsythe/PyIRI.git
cd /path/to/PyIRI
python -m build .
pip install .
```

For more details about the dependencies, see the documentation.

# Example

PyIRI currently provides the ionospheric electron density and associated profile
parameters on a user-specified grid for all times on a requested day.  For
example,

```
import PyIRI.edp_update as ml

# This example uses 15 April 2020, with an F10.7 of 150.0 sfu
(alon, alat, alon_2d, alat_2d, aalt, ahr, f2, f1, epeak, es_peak, sun, mag,
 edens_prof) = ml.run_iri_reg_grid(2020, 4, 15, 150, hr_res=1, lat_res=1,
                                   lon_res=1, alt_res=10, alt_min=0,
                                   alt_max=700, ccir_or_ursi=0)
print(edens_prof.shape)
```

The resulting electron density profile (`edens_prof`) will have a shape of
12 x 70 x 2701.  More examples are available in the documentation and Jupyter
Notebooks tutorials.


PyIRI does not calculate the Total Electron Content (TEC) automatically, as
the altitude array is provided by the user. If the user supplies an
array with low resolution, the resulting TEC values may be inaccurate.

The edp_to_vtec function can be used to calculate the vTEC from the edp output.

```
# Find TEC in shape [N_time, N_grid]
TEC = main.edp_to_vtec(edp, aalt, min_alt=0.0, max_alt=202000.0)

```
