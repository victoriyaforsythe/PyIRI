"""PyIRI Tutorial: Global Daily Ionospheric Parameters and Electron Density.

This tutorial demonstrates how to use the PyIRI model (with its spherical
harmonics architecture) to compute and visualize global maps of ionospheric
parameters for a specified month, year, day, and solar activity F10.7 index.

Specifically, this example shows how to:
- Configure the PyIRI model for a given date
- Specify the foF2 and hmF2 models (e.g., CCIR or URSI for foF2;
  SHU2015, AMTB2013, or BSE1979 for hmF2)
- Generate a horizontal global grid in geographic coordinates (GEO)
- Evaluate ionospheric parameters at each grid point over a full 24-hour UT
  cycle

Note: This example uses GEO input coordinates (longitude, latitude).
The PyIRI model also supports input in:
- Magnetic Local Time & Quasi-Dipole Latitude (MLT)
- Quasi-Dipole Longitude & Latitude (QD)
See the coordinate transformation tutorial for how to generate and use these
input formats.

---

Output:
The output consists of gridded global maps of key ionospheric parameters. For
example, the F2 dictionary includes:

- foF2 - Peak plasma frequency of the F2 layer
- hmF2 - Peak height of the F2 layer
- B0 and B1 - Thickness and shape parameters defining the electron density
  profile

All output maps have the shape (N_T, N_G), where:
- N_T is the number of time points (from the UT array)
- N_G is the number of horizontal grid locations

The EDP output has the shape (N_T, N_V, N_G), where:
- N_V is the number of vertical points (from the altitude array)
"""

# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import PyIRI
import PyIRI.sh_library as sh  # Updated PyIRI using spherical harmonics

plot_dir = os.path.join(Path(PyIRI.__file__).parent.parent, "docs", "figures")

# Specify date
year = 2020
month = 4
day = 1

# Specify solar activity index (F10.7 in SFU)
F107 = 100

# Create horizontal grid
lon_r = 5
lat_r = 5
alon_2d, alat_2d = np.mgrid[-180:180 + lon_r:lon_r, -90:90 + lat_r:lat_r]
alon = np.reshape(alon_2d, alon_2d.size)
alat = np.reshape(alat_2d, alat_2d.size)

# Time grid: Universal Time from 0 to 24 in 15-minute steps
hr_res = 1
aUT = np.arange(0, 24, hr_res)

# Height grid: 90 km to 700 km in 1 km steps
alt_res = 1
alt_min = 90
alt_max = 700
aalt = np.arange(alt_min, alt_max, alt_res)

# Coefficient sources and model options
foF2_coeff = 'CCIR'  # Options: 'CCIR' or 'URSI'
hmF2_model = 'SHU2015'  # Options: 'SHU2015', 'AMTB2013', 'BSE1979'
coord = 'GEO'  # Coordinate system: 'GEO', 'QD', or 'MLT'
coeff_dir = None  # Use default coefficient path

# ----------------------------------------
# Run PyIRI (Spherical Harmonics version)
# ----------------------------------------
# Compute ionospheric parameters for F2, F1, and E layers
F2, F1, E, Es, sun, mag, EDP = sh.IRI_density_1day(year,
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

# Select a time frame to plot
UT_plot = 10

ind_time = np.where(aUT == UT_plot)

# Plot foF2
fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(5, 3),
                       constrained_layout=True)
plt.xlim([-180, 180])
plt.ylim([-90, 90])
plt.xticks(np.arange(-180, 180 + 45, 90))
plt.yticks(np.arange(-90, 90 + 45, 45))
ax.set_facecolor('grey')
ax.set_xlabel(r'Geo Lon ($^\circ$)')
ax.set_ylabel(r'Geo Lat ($^\circ$)')
z = np.reshape(F2['fo'][ind_time, :], alon_2d.shape)
mesh = ax.pcolormesh(alon_2d, alat_2d, z)
ax.scatter(sun['lon'][ind_time], sun['lat'][ind_time],
           c='red', s=20, edgecolors="black", linewidths=0.5)
cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label('$fo$F2 (MHz)')
# Save figure
fig_name = os.path.join(plot_dir, "PyIRI_sh_foF2.png")
plt.savefig(fig_name, format='png', bbox_inches='tight')
print(f"Figure saved at {fig_name}")


# Plot hmF2
fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(5, 3),
                       constrained_layout=True)
plt.xlim([-180, 180])
plt.ylim([-90, 90])
plt.xticks(np.arange(-180, 180 + 45, 90))
plt.yticks(np.arange(-90, 90 + 45, 45))
ax.set_facecolor('grey')
ax.set_xlabel(r'Geo Lon ($^\circ$)')
ax.set_ylabel(r'Geo Lat ($^\circ$)')
z = np.reshape(F2['hm'][ind_time, :], alon_2d.shape)
mesh = ax.pcolormesh(alon_2d, alat_2d, z)
ax.scatter(sun['lon'][ind_time], sun['lat'][ind_time],
           c='red', s=20, edgecolors="black", linewidths=0.5)
cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label('$hm$F2 (km)')
# Save figure
fig_name = os.path.join(plot_dir, "PyIRI_sh_hmF2.png")
plt.savefig(fig_name, format='png', bbox_inches='tight')
print(f"Figure saved at {fig_name}")

# Plot B0
fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(5, 3),
                       constrained_layout=True)
plt.xlim([-180, 180])
plt.ylim([-90, 90])
plt.xticks(np.arange(-180, 180 + 45, 90))
plt.yticks(np.arange(-90, 90 + 45, 45))
ax.set_facecolor('grey')
ax.set_xlabel(r'Geo Lon ($^\circ$)')
ax.set_ylabel(r'Geo Lat ($^\circ$)')
z = np.reshape(F2['B0'][ind_time, :], alon_2d.shape)
mesh = ax.pcolormesh(alon_2d, alat_2d, z)
ax.scatter(sun['lon'][ind_time], sun['lat'][ind_time],
           c='red', s=20, edgecolors="black", linewidths=0.5)
cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label('$B0$ (km)')
# Save figure
fig_name = os.path.join(plot_dir, "PyIRI_sh_B0.png")
plt.savefig(fig_name, format='png', bbox_inches='tight')
print(f"Figure saved at {fig_name}")

# Plot B1
fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(5, 3),
                       constrained_layout=True)
plt.xlim([-180, 180])
plt.ylim([-90, 90])
plt.xticks(np.arange(-180, 180 + 45, 90))
plt.yticks(np.arange(-90, 90 + 45, 45))
ax.set_facecolor('grey')
ax.set_xlabel(r'Geo Lon ($^\circ$)')
ax.set_ylabel(r'Geo Lat ($^\circ$)')
z = np.reshape(F2['B1'][ind_time, :], alon_2d.shape)
mesh = ax.pcolormesh(alon_2d, alat_2d, z)
ax.scatter(sun['lon'][ind_time], sun['lat'][ind_time],
           c='red', s=20, edgecolors="black", linewidths=0.5)
cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label('$B1$ (unitless)')
# Save figure
fig_name = os.path.join(plot_dir, "PyIRI_sh_B1.png")
plt.savefig(fig_name, format='png', bbox_inches='tight')
print(f"Figure saved at {fig_name}")


# Plot B_top
fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(5, 3),
                       constrained_layout=True)
plt.xlim([-180, 180])
plt.ylim([-90, 90])
plt.xticks(np.arange(-180, 180 + 45, 90))
plt.yticks(np.arange(-90, 90 + 45, 45))
ax.set_facecolor('grey')
ax.set_xlabel(r'Geo Lon ($^\circ$)')
ax.set_ylabel(r'Geo Lat ($^\circ$)')
z = np.reshape(F2['B_top'][ind_time, :], alon_2d.shape)
mesh = ax.pcolormesh(alon_2d, alat_2d, z)
ax.scatter(sun['lon'][ind_time], sun['lat'][ind_time],
           c='red', s=20, edgecolors="black", linewidths=0.5)
cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label('$B_{top}$ (km)')
# Save figure
fig_name = os.path.join(plot_dir, "PyIRI_sh_B_top.png")
plt.savefig(fig_name, format='png', bbox_inches='tight')
print(f"Figure saved at {fig_name}")

# Plot probability of F1 to occurre
fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(5, 3),
                       constrained_layout=True)
plt.xlim([-180, 180])
plt.ylim([-90, 90])
plt.xticks(np.arange(-180, 180 + 45, 90))
plt.yticks(np.arange(-90, 90 + 45, 45))
ax.set_facecolor('grey')
ax.set_xlabel(r'Geo Lon ($^\circ$)')
ax.set_ylabel(r'Geo Lat ($^\circ$)')
z = np.reshape(F1['P'][ind_time, :], alon_2d.shape)
mesh = ax.pcolormesh(alon_2d, alat_2d, z)
ax.scatter(sun['lon'][ind_time], sun['lat'][ind_time],
           c='red', s=20, edgecolors="black", linewidths=0.5)
cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label('Probability of F1')
# Save figure
fig_name = os.path.join(plot_dir, "PyIRI_sh_P.png")
plt.savefig(fig_name, format='png', bbox_inches='tight')
print(f"Figure saved at {fig_name}")


# Plot thickness of F1
fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(5, 3),
                       constrained_layout=True)
plt.xlim([-180, 180])
plt.ylim([-90, 90])
plt.xticks(np.arange(-180, 180 + 45, 90))
plt.yticks(np.arange(-90, 90 + 45, 45))
ax.set_facecolor('grey')
ax.set_xlabel(r'Geo Lon ($^\circ$)')
ax.set_ylabel(r'Geo Lat ($^\circ$)')
z = np.reshape(F1['B_bot'][ind_time, :], alon_2d.shape)
mesh = ax.pcolormesh(alon_2d, alat_2d, z)
ax.scatter(sun['lon'][ind_time], sun['lat'][ind_time],
           c='red', s=20, edgecolors="black", linewidths=0.5)
cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label('B$_{bot}^{F1}$ (km)')
# Save figure
fig_name = os.path.join(plot_dir, "PyIRI_sh_B_F1_bot.png")
plt.savefig(fig_name, format='png', bbox_inches='tight')
print(f"Figure saved at {fig_name}")


# Plot foF1
fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(5, 3),
                       constrained_layout=True)
plt.xlim([-180, 180])
plt.ylim([-90, 90])
plt.xticks(np.arange(-180, 180 + 45, 90))
plt.yticks(np.arange(-90, 90 + 45, 45))
ax.set_facecolor('grey')
ax.set_xlabel(r'Geo Lon ($^\circ$)')
ax.set_ylabel(r'Geo Lat ($^\circ$)')
z = np.reshape(F1['fo'][ind_time, :], alon_2d.shape)
mesh = ax.pcolormesh(alon_2d, alat_2d, z)
ax.scatter(sun['lon'][ind_time], sun['lat'][ind_time],
           c='red', s=20, edgecolors="black", linewidths=0.5)
cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label('$fo$F1 (MHz)')
# Save figure
fig_name = os.path.join(plot_dir, "PyIRI_sh_foF1.png")
plt.savefig(fig_name, format='png', bbox_inches='tight')
print(f"Figure saved at {fig_name}")


# Plot hmF1
fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(5, 3),
                       constrained_layout=True)
plt.xlim([-180, 180])
plt.ylim([-90, 90])
plt.xticks(np.arange(-180, 180 + 45, 90))
plt.yticks(np.arange(-90, 90 + 45, 45))
ax.set_facecolor('grey')
ax.set_xlabel(r'Geo Lon ($^\circ$)')
ax.set_ylabel(r'Geo Lat ($^\circ$)')
z = np.reshape(F1['hm'][ind_time, :], alon_2d.shape)
mesh = ax.pcolormesh(alon_2d, alat_2d, z)
ax.scatter(sun['lon'][ind_time], sun['lat'][ind_time],
           c='red', s=20, edgecolors="black", linewidths=0.5)
cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label('$hm$F1 (km)')
# Save figure
fig_name = os.path.join(plot_dir, "PyIRI_sh_hmF1.png")
plt.savefig(fig_name, format='png', bbox_inches='tight')
print(f"Figure saved at {fig_name}")

# Plot foE
fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(5, 3),
                       constrained_layout=True)
plt.xlim([-180, 180])
plt.ylim([-90, 90])
plt.xticks(np.arange(-180, 180 + 45, 90))
plt.yticks(np.arange(-90, 90 + 45, 45))
ax.set_facecolor('grey')
ax.set_xlabel(r'Geo Lon ($^\circ$)')
ax.set_ylabel(r'Geo Lat ($^\circ$)')
z = np.reshape(E['fo'][ind_time, :], alon_2d.shape)
mesh = ax.pcolormesh(alon_2d, alat_2d, z)
ax.scatter(sun['lon'][ind_time], sun['lat'][ind_time],
           c='red', s=20, edgecolors="black", linewidths=0.5)
cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label('$fo$E (MHz)')
# Save figure
fig_name = os.path.join(plot_dir, "PyIRI_sh_foE.png")
plt.savefig(fig_name, format='png', bbox_inches='tight')
print(f"Figure saved at {fig_name}")

# Plot foEs
fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(5, 3),
                       constrained_layout=True)
plt.xlim([-180, 180])
plt.ylim([-90, 90])
plt.xticks(np.arange(-180, 180 + 45, 90))
plt.yticks(np.arange(-90, 90 + 45, 45))
ax.set_facecolor('grey')
ax.set_xlabel(r'Geo Lon ($^\circ$)')
ax.set_ylabel(r'Geo Lat ($^\circ$)')
z = np.reshape(Es['fo'][ind_time, :], alon_2d.shape)
mesh = ax.pcolormesh(alon_2d, alat_2d, z)
ax.scatter(sun['lon'][ind_time], sun['lat'][ind_time],
           c='red', s=20, edgecolors="black", linewidths=0.5)
cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label('$fo$Es (MHz)')
# Save figure
fig_name = os.path.join(plot_dir, "PyIRI_sh_foEs.png")
plt.savefig(fig_name, format='png', bbox_inches='tight')
print(f"Figure saved at {fig_name}")

# Calculate vTEC from EDP array
TEC = PyIRI.main_library.edp_to_vtec(EDP, aalt, min_alt=0.0, max_alt=202000.0)

# Plot vTEC
fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(5, 3),
                       constrained_layout=True)
plt.xlim([-180, 180])
plt.ylim([-90, 90])
plt.xticks(np.arange(-180, 180 + 45, 90))
plt.yticks(np.arange(-90, 90 + 45, 45))
ax.set_facecolor('grey')
ax.set_xlabel(r'Geo Lon ($^\circ$)')
ax.set_ylabel(r'Geo Lat ($^\circ$)')
z = np.reshape(TEC[ind_time, :], alon_2d.shape)
mesh = ax.pcolormesh(alon_2d, alat_2d, z)
ax.scatter(sun['lon'][ind_time], sun['lat'][ind_time],
           c='red', s=20, edgecolors="black", linewidths=0.5)
cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label('vTEC (TECU)')
# Save figure
fig_name = os.path.join(plot_dir, "PyIRI_sh_vTEC.png")
plt.savefig(fig_name, format='png', bbox_inches='tight')
print(f"Figure saved at {fig_name}")

# Select location to plot EDP
lon_plot = 10
lat_plot = 10

# Plot EDP
fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(4, 4),
                       constrained_layout=True)
ax.set_xlabel('Electron Density (m$^{-3}$)')
ax.set_ylabel('Altitude (km)')
ax.set_facecolor("lightgrey")
plt.xlim([0, 1.5e12])
plt.ylim([0, 700])
ind_grid = np.where((alon == lon_plot) & (alat == lat_plot))
ind_time = np.where(aUT == UT_plot)
ind_vert = np.where(aalt >= 0)
ind = ind_time, ind_vert, ind_grid
x = np.reshape(EDP[ind], aalt.shape)
ax.plot(x, aalt, c='black', linewidth=1)
plt.title(f'{lon_plot}° Lon, {lat_plot}° Lat, {UT_plot} UT')
# Save figure
fig_name = os.path.join(plot_dir, "PyIRI_sh_EDP.png")
plt.savefig(fig_name, format='png', bbox_inches='tight')
print(f"Figure saved at {fig_name}")
