"""Generating Spherical Harmonic Coefficients from IRI (or PyIRI) Grids.

This script details the SH coefficient extraction routine used by PyIRI for
expressing the foF2, hmF2, B0, B1, M3000F2, and foEs parameters.

Original IRI (or PyIRI in the case of foEs) grids are computed using the
iricore (https://github.com/MIST-Experiment/iricore/tree/master) IRI python
wrapper. They are created in Quasi-Dipole Magnetic Local Time (QD-MLT)
coordinates using the [ApexPy](https://apexpy.readthedocs.io) Apex python
wrapper. For a given year, IRI (or PyIRI) original parameter grids are computed
for the 15th day of every month over a 24-hour period (15-min increments). The
spherical harmonic and Fourier series coefficients are later extracted using
the pyshtools (https://github.com/SHTOOLS/SHTOOLS) SHTOOLS python wrapper.
These 3 libraries require a Fortran compiler and are therefore only called
outside the PyIRI model for coefficient extraction.

Purpose:
- Provide a transparent, reproducible process for regenerating the ionospheric
   parameter SH coefficients.
- Ensure PyIRI remains up-to-date when IRI model updates are released.
- Maintain a 10-year temporal resolution of coefficients (current coefficients
  are extracted for 2020).

---

Update Schedule:
- Every 10 years: The SH coefficients have been shown to remain accurate over
  decadal timescales despite magnetic pole drift, but re-derivation of
  coefficients may be necessary for years <2010 or >2030.

---

Workflow Overview:
1. Generate a regular 3° QD-MLT grid (0-24h MLT, -90°-90° QD lat).
2. For each timestamp (each UT of each month):
   - Use ApexPy.Apex to convert from QD-MLT to GEO coordinates at 0 km altitude.
   - Run the IRI (or PyIRI) model to evaluate the selected ionospheric
     parameter over each GEO grid for solar min AND solar max.
   - Save monthly grids of both solar activity levels.
3. Once a year's worth of ionospheric grids are generated, expand each month of
   each solar activity level into spherical harmonic and fourier series
   coefficients using SHTOOLS up to lmax = 29.
4. Write the full coefficient set to a NetCDF file (<parameter>.nc).

---

Output:
- <parameter>_<month:02d>.npy: IRI (or PyIRI) monthly grids, shape (2, 96, 7381)
  for 2 solar activity levels, 96 timestamps, 7381 grid points (3° resolution
  grid corresponding to lmax=29).
- <parameter>.nc: NetCDF file containing SH coefficients for the selected
  parameter over an entire year.

---

Notes:
- This script must be run in an environment with iricore, ApexPy, pyshtools,
  netCDF4, and numpy installed.
- Generated coefficients are 4π-normalized real SHs consistent with the
  conventions used in PyIRI.
- Solar min and max correspond to either IG12=0-100 for parameters foF2 (CCIR
  and URSI) and hmF2 using the Shubin model. For hmF2 using the AMTB model, B0,
  B1, M(3000)F2, and foEs, solar min and max should be R12=0-100.

Warning:
Generating IRI (or PyIRI) grids for an entire year takes a very long time if
done using this script (>12 hours). This code is provided for reference and
transparency; for proper running, it is recommended to use parallelization on a
cluster.
"""

from apexpy import Apex
import datetime as dt
import iricore
import netCDF4 as nc
import numpy as np
import pandas as pd
import PyIRI.main_library as ml
import pyshtools as pysh
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Create parameter grids using iricore

# ------------------------------------
# Set dir path to save IRI grids and dir path to save coeff files
# Set to None as default, will not run unless given proper paths by user
iri_grids_save_dir = None
coeffs_save_dir = None
if iri_grids_save_dir is None or coeffs_save_dir is None:
    raise ValueError("This script is not a typical tutorial. It is provided "
                     + "as a reference to extract SH coefficients from IRI "
                     + "grids, which is a lengthy process. To run it, please "
                     + "provide proper paths for the iri_grids_save_dir "
                     + "directory (wherein IRI grids will be stored) and for "
                     + "the coeffs_save_dir directory (wherein coefficient"
                     + "files will be saved).")

# ------------------------------------
# Create a regular QDLat-MLT grid
ddeg = 3
aMLT = np.arange(0, 24 * 15 + ddeg, ddeg) / 15
aQDLat = np.arange(90, -90 - ddeg, -ddeg)
MLT_grid_2d, QDLat_grid_2d = np.meshgrid(aMLT, aQDLat)

QDLat_grid_1d = QDLat_grid_2d.reshape(QDLat_grid_2d.size)
MLT_grid_1d = MLT_grid_2d.reshape(MLT_grid_2d.size)

# ------------------------------------
# Create UT arrays
year = 2020
months = np.arange(1, 13)
day = 15
dmin = 15

apdtime_12mo = pd.to_datetime([dt.datetime(year, month, day)
                               + dt.timedelta(minutes=i * dmin)
                               for month in months
                               for i in range(0, int(24 * 60 / dmin))])
aUT_12mo = (apdtime_12mo.hour + apdtime_12mo.minute / 60.
            + apdtime_12mo.second / 3600.)

# ------------------------------------
# Convert magnetic to geographic
'''WARNING: This section takes >7 minutes to run and is provided as a
reference.'''

aGLon_grid_1d = np.zeros((apdtime_12mo.size, QDLat_grid_1d.size))
aGLat_grid_1d = np.zeros((apdtime_12mo.size, QDLat_grid_1d.size))

for it in tqdm(range(0, apdtime_12mo.size)):
    A = Apex(apdtime_12mo[it])
    GLat_grid_1d, GLon_grid_1d = A.convert(QDLat_grid_1d,
                                           MLT_grid_1d,
                                           'mlt', 'geo', height=0.,
                                           datetime=apdtime_12mo[it])
    aGLat_grid_1d[it, :] = GLat_grid_1d
    aGLon_grid_1d[it, :] = GLon_grid_1d

# -----------------------------------------------------------------------------
# Evaluate iricore at each timestamp and geographic location
'''WARNING: This section takes 12 hours to run and is provided as a reference.
'''

# ------------------------------------
# Choose parameter to evaluate
# Choice of parameter is foF2_CCIR, foF2_URSI, hmF2_SHU2015, hmF2_AMTB2013,
# B0, B1, or M3000F2
# hmF2_BSE1979 is calculated from M3000F2 and therefore not evaluated
# for foEs, replace iricore with PyIRI.edp_update.IRI_density_1day()
param = 'foF2_CCIR'

# ------------------------------------
# Get correct solar index based on selected parameter
params_IG12 = ['foF2_CCIR', 'foF2_URSI', 'hmF2_SHU2015']
params_R12 = ['M3000F2', 'hmF2_AMTB2013', 'B0', 'B1', 'foEs']
if param in params_IG12:
    solar = 'IG12'
elif param in params_R12:
    solar = 'R12'

# ------------------------------------
# Get default IRI flags
jf = iricore.get_jf()

# ------------------------------------
# Initilialize parameter array
# We save monthly to avoid crashing memory
n_pos = QDLat_grid_1d.size
n_time = apdtime_12mo.size // 12
aParam = np.empty((2, apdtime_12mo.size // 12, n_pos))

# ------------------------------------
# Choose start month and select associated timestamp
# Start month selection is in case memory crashed halfway through the year
start_mo = 1
mo = start_mo
iut_mo = 0
iut_start = (start_mo - 1) * 24 * int(60 / dmin)

# ------------------------------------
# Run iricore

# ------------------------------------
# Loop over time
for iut in range(iut_start, apdtime_12mo.size):
    # ------------------------------------
    # Select timestamp-dependent geographic grid
    UT = aUT_12mo[iut]
    dtime = apdtime_12mo[iut]
    GLat_grid_1d = aGLat_grid_1d[iut, :]
    GLon_grid_1d = aGLon_grid_1d[iut, :]

    # ------------------------------------
    # Save at the end of each month
    if dtime.month > mo:
        np.save(f'{coeffs_save_dir}/{param}_{mo:02d}.npy', aParam)
        print(f'Saved {param}_{mo:02d}.npy')
        mo += 1
        iut_mo = 0
        aParam = np.empty((2, apdtime_12mo.size // 12, n_pos))

    # ------------------------------------
    # Adapt the model to the chosen parameter
    if param == 'foF2_CCIR':
        ioarr = 0
        jf[4] = True
    elif param == 'foF2_URSI':
        ioarr = 0
        jf[4] = False
    elif param == 'hmF2_SHU2015':
        ioarr = 1
        jf[38] = False
        jf[39] = False
    elif param == 'hmF2_AMTB2013':
        ioarr = 1
        jf[38] = False
        jf[39] = True
    elif param == 'B0':
        ioarr = 9
    elif param == 'B1':
        ioarr = 34
    elif param == 'M3000F2':
        ioarr = 35

    # ------------------------------------
    # Run the model twice for IG12/R12=0 and IG12/R12=100
    if solar == 'R12':
        iri_out_0 = iricore.iri(dtime, [0, 10, 10], GLat_grid_1d,
                                GLon_grid_1d, version=20, oarr32=0,
                                jf=jf)
        iri_out_100 = iricore.iri(dtime, [0, 10, 10], GLat_grid_1d,
                                  GLon_grid_1d, version=20, oarr32=100,
                                  jf=jf)

    elif solar == 'IG12':
        iri_out_0 = iricore.iri(dtime, [0, 10, 10], GLat_grid_1d,
                                GLon_grid_1d, version=20, oarr38=0,
                                jf=jf, oarr40=ml.IG12_2_F107(0))
        iri_out_100 = iricore.iri(dtime, [0, 10, 10], GLat_grid_1d,
                                  GLon_grid_1d, version=20, oarr38=100,
                                  jf=jf, oarr40=ml.IG12_2_F107(100))

    aParam[0, iut_mo] = iri_out_0.oarr[:, ioarr]
    aParam[1, iut_mo] = iri_out_100.oarr[:, ioarr]

    iut_mo += 1

# ------------------------------------
# Save the last month
np.save(f'{iri_grids_save_dir}/{param}_{mo:02d}.npy', aParam)


def flatten_SH_coeff(P, N):
    """Flatten the SH coefficients."""
    n_time = P.shape[0]
    lmax = P.shape[1] - 1

    mask_P = np.triu(np.ones((lmax + 1, lmax + 1), dtype=bool), k=1)

    P[:, mask_P] = np.nan
    N[:, mask_P] = np.nan

    N_flip = np.flip(N, axis=2)

    recombined = np.concatenate((N_flip[:, :, :-1], P), axis=2)
    flat = recombined.reshape((n_time, (lmax + 1) * (2 * lmax + 1)))
    mask_flat = np.isfinite(flat[0, :])
    flat = flat[:, mask_flat]

    return flat


def complex_to_real_FS_coeff(complex_coeffs):
    """Convert from complex to real FS coefficients."""
    n_freqs, n_pos = complex_coeffs.shape

    real_coeffs = np.empty((2 * n_freqs - 1, n_pos), dtype=float)

    real_coeffs[0, :] = complex_coeffs[0, :].real

    odd_real_coeffs = 2 * complex_coeffs[1:, :].real
    even_real_coeffs = - 2 * complex_coeffs[1:, :].imag

    real_coeffs[1::2, :] = odd_real_coeffs
    real_coeffs[2::2, :] = even_real_coeffs

    return real_coeffs


ddeg = 3
aMLT = np.arange(0, 24 * 15 + ddeg, ddeg) / 15
aQDLat = np.arange(90, -90 - ddeg, -ddeg)
MLT_grid_2d, QDLat_grid_2d = np.meshgrid(aMLT, aQDLat)

QDLat_grid_1d = QDLat_grid_2d.reshape(QDLat_grid_2d.size)
MLT_grid_1d = MLT_grid_2d.reshape(MLT_grid_2d.size)

year = 2020

solars = ['IG12', 'R12']
solars_verbose = ['Ionosonde Global index (IG)', 'Sunspot Number (R)']

# -----------------------------------------------------------------------------
# Extract the coefficients

# ------------------------------------
# Determine lmax
# Calculate the max degree of SH (lmax) given the original grid resolution
# The total number of SH coefficients will be (lmax + 1) * (lmax + 1)
n_qdlat = QDLat_grid_2d.shape[0]
n_mlt = MLT_grid_2d.shape[1]
lmax = int(n_qdlat / 2 - 1)
n_SH = (lmax + 1) * (lmax + 1)
n_pos = QDLat_grid_1d.size

# ------------------------------------
# Truncate the max degree of FS reconstruction (derived experimentally)
aMo = np.arange(1, 13)
n_FS_c = 6
n_FS_r = n_FS_c * 2 - 1

# ------------------------------------
# Initialize the coefficient matrix
C = np.empty((2, 12, n_FS_r, n_SH))

if param in ['foF2_CCIR', 'foF2_URSI', 'hmF2_SHU2015']:
    solar = solars[0]
    solar_verbose = solars_verbose[0]
else:
    solar = solars[1]
    solar_verbose = solars_verbose[1]

# ------------------------------------
# Loop over previously calculated monthly parameter files
for imo in tqdm(range(12)):

    mo = imo + 1

    aParam = np.load(f'{iri_grids_save_dir}/{param}_{mo:02d}.npy')

    # ------------------------------------
    # Loop over IG12 = 0 and IG12 = 100
    for iIG in range(2):

        aParam_IG = aParam[iIG]

        # ------------------------------------
        # Initialize FS coefficient matrix for selected IG12
        aFS_c = np.zeros((n_FS_c, n_pos), dtype=complex)

        # ------------------------------------
        # Calculate n_FS_c complex FS coefficients for each position
        for ipos in range(0, n_pos):
            FS_ipos = np.fft.rfft(aParam_IG[:, ipos].flatten(),
                                  norm='forward')
            FS_truncated_ipos = FS_ipos[:n_FS_c]
            aFS_c[:, ipos] = FS_truncated_ipos

        # ------------------------------------
        # Convert to n_FS_r real FS coefficients
        aFS_r = complex_to_real_FS_coeff(aFS_c)

        # ------------------------------------
        # Initilialize SH coefficient matrix for selected IG12
        aSH_of_FS = np.zeros((n_FS_r, 2, (lmax + 1), (lmax + 1)))

        # ------------------------------------
        # Calculate n_SH real SH coefficients using pyshtools
        # We reshape to (n_qdlat, n_mlt) instead of the original (n_mlt,
        # n_qdlat) to match pyshtools expectations
        # NOTE: normalization is '4pi', csphase is included!
        for iFS in range(n_FS_r):
            aFS_r_iFS = aFS_r[iFS]
            aFS_r_iFS_2d = aFS_r_iFS.reshape((n_qdlat, n_mlt))
            aFS_r_iFS_grid = pysh.SHGrid.from_array(aFS_r_iFS_2d, grid='DH')
            aSH_of_FS_iFS = aFS_r_iFS_grid.expand(normalization='4pi',
                                                  csphase=-1)
            aSH_of_FS_iFS = aSH_of_FS_iFS.to_array()
            aSH_of_FS[iFS, :, :, :] = aSH_of_FS_iFS

        # ------------------------------------
        # Separate in positive and negative orders and flatten all coefficients
        P = aSH_of_FS[:, 0, :, :]
        N = aSH_of_FS[:, 1, :, :]
        aSH_of_FS_flat = flatten_SH_coeff(P, N)

        C[iIG, imo, :, :] = aSH_of_FS_flat

# -----------------------------------------------------------------------------
# Save coefficients into NetCDF files

# ------------------------------------
# Open the output netcdf file
data = nc.Dataset(f'{coeffs_save_dir}/{param}.nc', 'w')

# ------------------------------------
# Set the dimensions
n_FS = n_FS_r
data.createDimension(f'{solar}', 2)
data.createDimension('Month', 12)
data.createDimension('i_FS', n_FS)
data.createDimension('j_SH', n_SH)

# ------------------------------------
# Set up variables
i_FS = data.createVariable('i_FS', 'int', ('i_FS'))
i_FS.units = 'N/A'
i_FS.description = ('Real Fourier Series (FS) coefficient number (0 to '
                    + 'n_FS - 1).')

j_SH = data.createVariable('j_SH', 'int', ('j_SH'))
j_SH.units = 'N/A'
j_SH.description = ('Real Spherical Harmonic (SH) coefficient number (0 to '
                    + 'n_SH - 1).')

solar_idx = data.createVariable(f'{solar}', 'float64', (f'{solar}'))
solar_idx.units = 'N/A'
solar_idx.description = (f'12-month running mean of the {solar_verbose}.')

Month = data.createVariable('Month', 'int', ('Month'))
Month.units = 'N/A'
Month.description = 'Month of the year.'

C_mat = data.createVariable('Coefficients', 'float64', (f'{solar}', 'Month',
                                                        'i_FS', 'j_SH'))
C_mat.units = 'N/A'
C_mat.description = ('Each row i contains the n_SH Spherical Harmonic (SH) '
                     + 'coefficients used to reconstruct the ith Fourier Series'
                     + '(FS) coefficient. The SH coefficients of row i are '
                     + 'flattened so that the SH mode of degree l and order m '
                     + 'is in column j = l * (l + 1) + m. To reconstruct the '
                     + 'ith FS coefficient a_i at coordinates (theta, phi) '
                     + 'from the corresponding row of SH coefficients {g_[i,0],'
                     + ' ..., g_[i,n_SH-1]}, use the SH reconstruction formula:'
                     + ' a_i(theta, phi) = sum(g_[i,j] * Y_j(theta, phi)) where'
                     + ' j=[0, n_SH-1] is the mode number (i.e., column) and '
                     + 'Y_j is the associated SH function. To reconstruct the '
                     + 'ionospheric parameter value P(t, theta, phi) at time t '
                     + 'and coordinates (theta, phi), use the FS reconstruction'
                     + ' formula: P(t, theta, phi) = a_0(theta, phi) + '
                     + 'sum(a_[2i-1](theta, phi) * cos(2pi/24 * i * t) + '
                     + 'a_[2i](theta, phi) * sin(2pi/24 * i * t)) where i=[0, '
                     + 'n_FS-1]. The coefficients were calculated over the year'
                     + f' {year}.')

# ------------------------------------
# Populate with the data
solar_idx[:] = [0, 100]
Month[:] = np.arange(1, 13)
i_FS[:] = np.arange(0, n_FS)
j_SH[:] = np.arange(0, n_SH)
C_mat[:, :, :, :] = C

# ------------------------------------
# Add title
param_name = param.replace('_', ' ')
data.Title = f'{param_name} FS/SH coefficients'

# ------------------------------------
# Close the file
data.close()
print(f'File {coeffs_save_dir}/{param}.nc was written.')
