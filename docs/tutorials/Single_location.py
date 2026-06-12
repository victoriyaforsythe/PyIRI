"""PyIRI Tutorial: Daily Ionospheric Electron Density Profile Generation.

This tutorial demonstrates how to use PyIRI, a modern, Python-native
reformulation of the International Reference Ionosphere (IRI), to:
- Generate ionospheric electron density profile (EDP) parameters at a single
  location and for an entire day
- Plot the evolution of peak plasma densities, heights, and thicknesses for F2,
  F1, E, and Es layers
- Visualize the full EDP (electron density vs. height and time)

Key Features:
- Uses updated IRI coefficients expressed in spherical harmonics and Fourier
  series
- Supports climatological generation without external dependencies
- Output is based on a specified solar flux index and date

Requirements:
Make sure you have the following PyIRI modules:
- PyIRI.sh_library
- PyIRI.edp_update
"""

# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import PyIRI
import PyIRI.edp_update as ml  # Legacy PyIRI formalism (Fourier + empirical)
import PyIRI.sh_library as sh  # Updated PyIRI using spherical harmonics

plot_dir = os.path.join(Path(PyIRI.__file__).parent.parent, "docs/figures")

# Specify date
year = 2020
month = 4
day = 1

# Specify solar activity index (F10.7 in SFU)
F107 = 100

# Select base coefficient set: 0 = CCIR, 1 = URSI
ccir_or_ursi = 0

# Location of interest: 10°E, 20°N
alon = np.array([10.])
alat = np.array([20.])

# Time grid: Universal Time from 0 to 24 in 15-minute steps
hr_res = 0.25
aUT = np.arange(0, 24, hr_res)

# Height grid: 90 km to 700 km in 1 km steps
alt_res = 1
alt_min = 90
alt_max = 700
aalt = np.arange(alt_min, alt_max, alt_res)

# Coefficient sources and model options
foF2_coeff = 'CCIR'       # Options: 'CCIR' or 'URSI'
hmF2_model = 'SHU2015'    # Options: 'SHU2015', 'AMTB2013', 'BSE1979'
coord = 'GEO'             # Coordinate system: 'GEO', 'QD', or 'MLT'
coeff_dir = None          # Use default coefficient path

# ----------------------------------------
# Run PyIRI (Spherical Harmonics version)
# ----------------------------------------
# Compute ionospheric parameters for F2, F1, and E layers
(F2,
 F1,
 E,
 Es,
 sun,
 mag,
 EDP) = sh.IRI_density_1day(year,
                            month,
                            day,
                            aUT,
                            alon,
                            alat,
                            aalt,
                            F107,
                            coeff_dir=coeff_dir,
                            foF2_coeff=foF2_coeff,
                            hmF2_model=hmF2_model,
                            coord=coord,
                            old_output=False)
# In version 0.1.6, the sh.IRI_density_1day function did not return the
# sporadic E layer dict. To return the Es dict along with the others, make sure
# to specify old_output=False. The default of the old_output argument will
# change from True to False in version 0.2+.

# Plot results for F2 region at time stamp UT=10
# Make sure alon and alat contains lon_plot and lat_plot
# Define location for plotting
lon_plot = 10
lat_plot = 20

fig, ax = plt.subplots(4, 1, sharex=True, figsize=(5, 7),
                       constrained_layout=True)
plt.xlim([0, 24])
plt.xticks(np.arange(0, 24 + 4, 4))
ind_grid = np.where((alon == lon_plot) & (alat == lat_plot))[0]

# Panel 1: Peak Electron Densities
ax_plot = ax[0]
ax_plot.set_facecolor('lightgrey')
ax_plot.set_ylabel('Peak Density (m$^{-3}$)')
ax_plot.plot(aUT, F2['Nm'][:, ind_grid], label='$Nm$F2', c='red')
ax_plot.plot(aUT, F1['Nm'][:, ind_grid], label='$Nm$F1', c='green')
ax_plot.plot(aUT, E['Nm'][:, ind_grid], label='$Nm$E', c='orange')
ax_plot.plot(aUT, Es['Nm'][:, ind_grid], label='$Nm$Es', c='blue')
ax_plot.legend(loc='upper left', prop={'size': 7})

# Panel 2: Peak Heights
ax_plot = ax[1]
ax_plot.set_facecolor('lightgrey')
ax_plot.set_ylabel('Peak Height (km)')
ax_plot.plot(aUT, F2['hm'][:, ind_grid], label='$hm$F2', c='red')
ax_plot.plot(aUT, F1['hm'][:, ind_grid], label='$hm$F1', c='green')
ax_plot.plot(aUT, E['hm'][:, ind_grid], label='$hm$E', c='orange')
ax_plot.plot(aUT, Es['hm'][:, ind_grid], label='$hm$Es', c='blue')
ax_plot.legend(loc='upper left', prop={'size': 7})

# Panel 3: Topside Thickness
ax_plot = ax[2]
ax_plot.set_facecolor('lightgrey')
ax_plot.set_ylabel('Top Thickness (km)')
ax_plot.plot(aUT, F2['B_top'][:, ind_grid], label='$B_{top}^{F2}$', c='red')
ax_plot.plot(aUT, E['B_top'][:, ind_grid], label='$B_{top}^{E}$', c='orange')
ax_plot.plot(aUT, Es['B_top'][:, ind_grid], label='$B_{top}^{Es}$', c='blue')
ax_plot.legend(loc='upper left', prop={'size': 7})

# Panel 4: Bottomside Thickness
ax_plot = ax[3]
ax_plot.set_facecolor('lightgrey')
ax_plot.set_xlabel('UT (hours)')
ax_plot.set_ylabel('Bottom Thickness (km)')
ax_plot.plot(aUT, F2['B0'][:, ind_grid], label='$B0$', c='red')
ax_plot.plot(aUT, F1['B_bot'][:, ind_grid], label='$B_{bot}^{F1}$', c='green')
ax_plot.plot(aUT, E['B_bot'][:, ind_grid], label='$B_{bot}^{E}$', c='orange')
ax_plot.plot(aUT, Es['B_top'][:, ind_grid], label='$B_{bot}^{Es}$', c='blue')
ax_plot.legend(loc='upper left', prop={'size': 7})

# Save figure
fig_name = os.path.join(plot_dir, "PyIRI_sh_diurnal.png")
plt.savefig(fig_name, format='png', bbox_inches='tight')
print(f"Figure saved at {fig_name}")

# Plot electron density as a function of time
fig, ax = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True)
plt.xlim([0, 24])
plt.ylim([90, 600])
plt.xticks(np.arange(0, 24 + 4, 4))
ax.set_facecolor('grey')
ax.set_xlabel('UT (hours)')
ax.set_ylabel('Altitude (km)')
ind_grid = np.where((alon == lon_plot) & (alat == lat_plot))[0]
z = np.transpose(np.reshape(EDP[:, :, ind_grid], (aUT.size, aalt.size)))
mesh = ax.pcolormesh(aUT, aalt, z)
cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label('Electron Density (m$^{-3}$)')

# Save figure
fig_name = os.path.join(plot_dir, "PyIRI_sh_EDP_diurnal.png")
plt.savefig(fig_name, format='png', bbox_inches='tight')
print(f"Figure saved at {fig_name}")

# ----------------------------------------
# Run PyIRI (Legacy version)
# ----------------------------------------
# Select base coefficient set: 0 = CCIR, 1 = URSI
ccir_or_ursi = 0

# Run legacy PyIRI model based on v0.0.4 to get ionospheric layers and electron
# density profile (EDP)
F2, F1, E, Es, sun, mag, EDP = ml.IRI_density_1day(
    year, month, day,
    aUT, alon, alat, aalt,
    F107, PyIRI.coeff_dir, ccir_or_ursi
)

# Plot results for F2 region at time stamp UT=10
# Make sure alon and alat contains lon_plot and lat_plot
# Define location for plotting
lon_plot = 10
lat_plot = 20

fig, ax = plt.subplots(4, 1, sharex=True, figsize=(5, 7),
                       constrained_layout=True)
plt.xlim([0, 24])
plt.xticks(np.arange(0, 24 + 4, 4))
ind_grid = np.where((alon == lon_plot) & (alat == lat_plot))[0]

# Panel 1: Peak Electron Densities
ax_plot = ax[0]
ax_plot.set_facecolor('lightgrey')
ax_plot.set_ylabel('Peak Density (m$^{-3}$)')
ax_plot.plot(aUT, F2['Nm'][:, ind_grid], label='$Nm$F2', c='red')
ax_plot.plot(aUT, F1['Nm'][:, ind_grid], label='$Nm$F1', c='green')
ax_plot.plot(aUT, E['Nm'][:, ind_grid], label='$Nm$E', c='orange')
ax_plot.plot(aUT, Es['Nm'][:, ind_grid], label='$Nm$Es', c='blue')
ax_plot.legend(loc='upper left', prop={'size': 7})

# Panel 2: Peak Heights
ax_plot = ax[1]
ax_plot.set_facecolor('lightgrey')
ax_plot.set_ylabel('Peak Height (km)')
ax_plot.plot(aUT, F2['hm'][:, ind_grid], label='$hm$F2', c='red')
ax_plot.plot(aUT, F1['hm'][:, ind_grid], label='$hm$F1', c='green')
ax_plot.plot(aUT, E['hm'][:, ind_grid], label='$hm$E', c='orange')
ax_plot.plot(aUT, Es['hm'][:, ind_grid], label='$hm$Es', c='blue')
ax_plot.legend(loc='upper left', prop={'size': 7})

# Panel 3: Topside Thickness
ax_plot = ax[2]
ax_plot.set_facecolor('lightgrey')
ax_plot.set_ylabel('Top Thickness (km)')
ax_plot.plot(aUT, F2['B_top'][:, ind_grid], label='$B_{top}^{F2}$', c='red')
ax_plot.plot(aUT, E['B_top'][:, ind_grid], label='$B_{top}^{E}$', c='orange')
ax_plot.plot(aUT, Es['B_top'][:, ind_grid], label='$B_{top}^{Es}$', c='blue')
ax_plot.legend(loc='upper left', prop={'size': 7})

# Panel 4: Bottomside Thickness
ax_plot = ax[3]
ax_plot.set_facecolor('lightgrey')
ax_plot.set_xlabel('UT (hours)')
ax_plot.set_ylabel('Bottom Thickness (km)')
ax_plot.plot(aUT, F2['B_bot'][:, ind_grid], label='$B_{bot}^{F2}$', c='red')
ax_plot.plot(aUT, F1['B_bot'][:, ind_grid], label='$B_{bot}^{F1}$', c='green')
ax_plot.plot(aUT, E['B_bot'][:, ind_grid], label='$B_{bot}^{E}$', c='orange')
ax_plot.plot(aUT, Es['B_top'][:, ind_grid], label='$B_{bot}^{Es}$', c='blue')
ax_plot.legend(loc='upper left', prop={'size': 7})

# Save figure
fig_name = os.path.join(plot_dir, "PyIRI_legacy_diurnal.png")
plt.savefig(fig_name, format='png', bbox_inches='tight')
print(f"Figure saved at {fig_name}")

# Plot electron density as a function of time
fig, ax = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True)
plt.xlim([0, 24])
plt.ylim([90, 600])
plt.xticks(np.arange(0, 24 + 4, 4))
ax.set_facecolor('grey')
ax.set_xlabel('UT (hours)')
ax.set_ylabel('Altitude (km)')
ind_grid = np.where((alon == lon_plot) & (alat == lat_plot))[0]
z = np.transpose(np.reshape(EDP[:, :, ind_grid], (aUT.size, aalt.size)))
mesh = ax.pcolormesh(aUT, aalt, z)
cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label('Electron Density (m$^{-3}$)')

# Save figure
fig_name = os.path.join(plot_dir, "PyIRI_legacy_EDP_diurnal.png")
plt.savefig(fig_name, format='png', bbox_inches='tight')
print(f"Figure saved at {fig_name}")
