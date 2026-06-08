"""Generating Apex Coordinate Spherical Harmonic Coefficients (1900-2030).

This notebook reproduces the Apex coordinate transformation coefficients used
by PyIRI for converting between geographic (GEO) and quasi-dipole (QD)
coordinates and back.

Apex coordinates are computed using the [ApexPy](https://apexpy.readthedocs.io)
library, then expanded into real spherical harmonic (SH) coefficients using
SHTOOLS (https://shtools.github.io/SHTOOLS/). The resulting coefficients are
saved in PyIRI.coeff_dir / 'Apex' / 'Apex.nc' and used internally by
PyIRI.sh_library.Apex() for high-performance coordinate transforms without
requiring ApexPy at runtime.

Purpose:
- Provide a transparent, reproducible process for regenerating the Apex SH
  coefficients.
- Ensure PyIRI remains up-to-date when new IGRF magnetic field updates are
  released.
- Maintain a 1-year temporal resolution of coefficients, from 1900 through 2030
  (extend as needed).

---

Update Schedule:
- Every year: Re-run this notebook to extend the coefficient set by one year.
- Every 5 years: Re-generate the entire 1900-present dataset after a new IGRF
  model release.

---

Workflow Overview:
1. Build a regular 1° GEO latitude-longitude grid (0-360° lon, 90° to -90° lat).
2. For each year:
 - Use ApexPy.Apex to convert between GEO and QD coordinates at 0 km altitude.
 - Store QD latitudes and longitudes (as cosine and sine components of
   longitude to avoid discontinuities at the 0°/360° seam).
3. Expand each coordinate field into spherical harmonic coefficients using
   SHTOOLS up to lmax = 20.
4. Write the full coefficient set to a NetCDF file (Apex.nc).

---

Output:
- Apex.npy: intermediate grid data (raw ApexPy results)
- Apex.nc: NetCDF file containing SH coefficients for:
- QDLat, QDLon_cos, QDLon_sin
- GeoLat, GeoLon_cos, GeoLon_sin

---

Notes:
- This script must be run in an environment with ApexPy, pyshtools, netCDF4,
  and numpy installed.
- Generated coefficients are 4π-normalized real SHs consistent with the
  conventions used in PyIRI.
"""

from apexpy import Apex
import datetime as dt
import os

import netCDF4 as nc  # future use
import numpy as np
import pandas as pd
import pyshtools as pysh
from tqdm import tqdm

# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
save_dir = None
if save_dir is None:
    raise ValueError("This script is not a typical tutorial. It is provided "
                     + "as a reference to extract Apex coefficients using "
                     + "ApexPy, which is a lengthy process. To run it, please "
                     + "provide proper paths for the save_dir directory "
                     + "wherein Apex coefficients will be saved.")

ddeg = 1  # grid resolution in degrees
ayear = np.arange(1900, 2026)

# ---------------------------------------------------------------------
# Build GEO latitude–longitude grid (1° resolution)
# ---------------------------------------------------------------------
aLon = np.arange(0, 360 + ddeg, ddeg)
aLat = np.arange(90, -90 - ddeg, -ddeg)
Lon_grid_2d, Lat_grid_2d = np.meshgrid(aLon, aLat)
Lat_grid_1d = Lat_grid_2d.ravel()
Lon_grid_1d = Lon_grid_2d.ravel()

# ---------------------------------------------------------------------
# Build yearly datetime array
# ---------------------------------------------------------------------
atime = pd.to_datetime([dt.datetime(int(y), 1, 1) for y in ayear])

# ---------------------------------------------------------------------
# Allocate output arrays
# ---------------------------------------------------------------------
n_times = atime.size
n_locs = Lat_grid_1d.size
shape = (n_times, n_locs)

aQDLat = np.zeros(shape)
aQDLon_cos = np.zeros(shape)
aQDLon_sin = np.zeros(shape)
aGeoLat = np.zeros(shape)
aGeoLon_cos = np.zeros(shape)
aGeoLon_sin = np.zeros(shape)

# ---------------------------------------------------------------------
# Compute QD and GEO coordinates
# cos/sin forms prevent seam at 0°/360° longitude
# ---------------------------------------------------------------------
for i, t in enumerate(tqdm(atime, desc="Computing Apex coordinates")):
    A = Apex(t)

    QDLat, QDLon = A.convert(Lat_grid_1d, Lon_grid_1d, "geo", "qd",
                             height=0.0, datetime=t)
    aQDLat[i] = QDLat
    aQDLon_cos[i] = np.cos(np.radians(QDLon))
    aQDLon_sin[i] = np.sin(np.radians(QDLon))

    GeoLat, GeoLon = A.convert(Lat_grid_1d, Lon_grid_1d, "qd", "geo",
                               height=0.0, datetime=t)
    aGeoLat[i] = GeoLat
    aGeoLon_cos[i] = np.cos(np.radians(GeoLon))
    aGeoLon_sin[i] = np.sin(np.radians(GeoLon))

# ---------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------
grid = {
    "QDLat": aQDLat,
    "QDLon_cos": aQDLon_cos,
    "QDLon_sin": aQDLon_sin,
    "GeoLat": aGeoLat,
    "GeoLon_cos": aGeoLon_cos,
    "GeoLon_sin": aGeoLon_sin,
}

out_path = os.path.join(save_dir, "Apex.npy")
np.save(out_path, grid)
print(f"Saved {out_path}", flush=True)


def flatten_SH_coeff(P, N):
    """Flatten real spherical harmonic (SH) coefficient arrays.

    Parameters
    ----------
    P : np.ndarray
        Positive-order SH coefficients of shape (n_time, lmax + 1, lmax + 1).
    N : np.ndarray
        Negative-order SH coefficients of the same shape.

    Returns
    -------
    np.ndarray
        Flattened SH coefficient array of shape (n_time, n_SH),
        where n_SH = (lmax + 1)**2.
    """
    n_time, lmax_p1, _ = P.shape

    # Mask upper triangle (m > l) which contains invalid entries
    mask = np.triu(np.ones((lmax_p1, lmax_p1), dtype=bool), k=1)
    P[:, mask] = np.nan
    N[:, mask] = np.nan

    # Flip negative orders, drop redundant m=0 column, and recombine
    N_flip = np.flip(N, axis=2)
    recombined = np.concatenate((N_flip[:, :, :-1], P), axis=2)

    # Flatten and remove NaNs
    flat = recombined.reshape(n_time, -1)
    mask_flat = np.isfinite(flat[0])
    return flat[:, mask_flat]


# ---------------------------------------------------------------------
# Compute and save spherical harmonic coefficients
# ---------------------------------------------------------------------
lmax = 20
n_SH = (lmax + 1) ** 2
C = np.empty((atime.size, 2, lmax + 1, lmax + 1))
res_dict = {}

for key in grid:
    # Expand each field into SH coefficients for all years
    for i, t in enumerate(atime):
        field = grid[key][i].reshape(Lat_grid_2d.shape)
        grid_SH = pysh.SHGrid.from_array(field, grid="DH")
        coeffs = grid_SH.expand(normalization="4pi", csphase=-1)
        C[i] = coeffs.to_array(lmax=lmax)

    # Separate positive and negative orders and flatten
    P, N = C[:, 0], C[:, 1]
    res_dict[key] = flatten_SH_coeff(P, N)

# ---------------------------------------------------------------------
# Write NetCDF output
# ---------------------------------------------------------------------
nc_path = os.path.join(save_dir, "Apex.nc")
with nc.Dataset(nc_path, "w") as data:
    # Dimensions
    data.createDimension("Year", atime.size)
    data.createDimension("j_SH", n_SH)

    # Variables
    j_SH = data.createVariable("j_SH", "i4", ("j_SH",))
    j_SH.units = "N/A"
    j_SH.description = (
        "Real spherical harmonic coefficient index (0 ... n_SH - 1)."
    )

    year = data.createVariable("Year", "i4", ("Year",))
    year.units = "N/A"
    year.description = "Year of the coefficient set."

    # Template for coordinate datasets
    var_info = {
        "QDLat": "QD latitude distribution",
        "QDLon_cos": "cosine of QD longitude",
        "QDLon_sin": "sine of QD longitude",
        "GeoLat": "geographic latitude distribution",
        "GeoLon_cos": "cosine of geographic longitude",
        "GeoLon_sin": "sine of geographic longitude",
    }

    for name, desc in var_info.items():
        v = data.createVariable(name, "f8", ("Year", "j_SH"))
        v.units = "N/A"
        v.description = (
            f"Spherical harmonic coefficients for {desc}. "
            "Flattened so mode of degree l and order m is at "
            "index j = l*(l+1) + m."
        )
        v[:, :] = res_dict[name]

    # Metadata
    year[:] = atime.year
    j_SH[:] = np.arange(n_SH)
    data.Title = "Apex spherical harmonic coefficients"
    data.History = "Generated using ApexPy and SHTOOLS; 4π normalization."

print(f"File written: {nc_path}")
