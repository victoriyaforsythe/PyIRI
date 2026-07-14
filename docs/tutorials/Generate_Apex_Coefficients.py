"""Generating Apex Coordinate Spherical Harmonic Coefficients (1900-2030).

---------------------------------------------------------------------------
IMPORTANT - READ BEFORE RUNNING
---------------------------------------------------------------------------
This script is not a PyIRI tutorial.

It is a reference routine for extracting Apex magnetic coordinate coefficients
via the ApexPy library. This process may take considerable time and should only
be run for the sake of a coefficient update.
---------------------------------------------------------------------------

This script reproduces the Apex coordinate transformation coefficients used
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
- Every year: Re-run this script to extend the coefficient set by one year.
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

import argparse
try:
    from apexpy import Apex
except ImportError:
    raise ImportError("This script requires the apexpy module, which is a "
                      "Python wrapper for the Apex Fortran library. The apexpy "
                      "module is only necessary to run this Apex coefficient "
                      "derivation routine; it is not necessary to run the PyIRI"
                      " model.")
import datetime as dt
import os

import netCDF4 as nc
import numpy as np
import pandas as pd
try:
    import pyshtools as pysh
except ImportError:
    raise ImportError("This script requires the pyshtools module, which is a "
                      "Python wrapper for the SHTOOLS Fortran library. The "
                      "pyshtools module is only necessary to run this Apex "
                      "coefficient derivation routine; it is not necessary to "
                      "run the PyIRI model.")
import sys
from tqdm import tqdm

# ---------------------------------------------------------------------
# Input saving directory
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description=(
        "************************************************************\n"
        "  WARNING: This is not a PyIRI tutorial.\n"
        "************************************************************\n\n"
        "This script regenerates Apex coefficient files using ApexPy.\n"
        "It should only be run for the sake of coefficient updates.\n\n"
    ),
    formatter_class=argparse.RawDescriptionHelpFormatter
)

parser.add_argument(
    "--save-dir",
    type=str,
    help="Path to the directory where coefficient files will be saved.",
)

parser.add_argument(
    "--end-year",
    type=int,
    help="Upper year range 1900-YYYY.",
    default=dt.datetime.now(dt.timezone.utc).year,
)

args = parser.parse_args()

if args.save_dir is None:
    print("\nYou did not provide a saving directory.\n")
    parser.print_help()
    sys.exit(1)

if not os.path.exists(args.save_dir):
    raise ValueError(f"Directory does not exist: {args.save_dir}")
if not os.path.isdir(args.save_dir):
    raise ValueError(f"This is not a directory: {args.save_dir}")

save_dir = os.path.normpath(args.save_dir)
end_year = args.end_year

# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
ddeg = 1  # grid resolution in degrees
ayear = np.arange(1900, end_year + 1)

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
        Positive-order SH coefficient array.
        Shape (n_time, lmax + 1, lmax + 1)
    N : np.ndarray
        Negative-order SH coefficient array.
        Shape (n_time, lmax + 1, lmax + 1)

    Returns
    -------
    np.ndarray
        Flattened SH coefficient array.
        Shape (n_time, (lmax + 1)**2)
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
nc_path = os.path.join(save_dir, f"Apex_1900_{end_year}.nc")
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
