#!/usr/bin/env python
# ---------------------------------------------------------
# Distribution statement A. Approved for public release.
# Distribution is unlimited.
# This work was supported by the Office of Naval Research.
# ---------------------------------------------------------
"""This library contains components visualization routines for PyIRI."""

import matplotlib.pyplot as plt
import numpy as np
import os


def PyIRI_plot_mag_dip_lat(mag, alon, alat, alon_2d, alat_2d, plot_dir,
                           plot_name='PyIRI_mag_dip_lat.png'):
    """Plot magnetic dip latitude.

    Parameters
    ----------
    mag : dict
        Dictionary output of IRI_monthly_mean_parameters.
    alon : array-like
        Flattened array of geo longitudes in degrees.
    alat : array-like
        Flattened array of geo latitudes in degrees.
    alon_2d : array-like
        2-D array of geo longitudes in degrees.
    alat_2d : array-like
        2-D array of geo latitudes in degrees.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Output name, without directory, for the saved figure
        (default='PyIRI_mag_dip_lat.png')

    """
    figname = os.path.join(plot_dir, plot_name)
    fig, ax = plt.subplots(1, 1)
    ax.set_facecolor('grey')
    ax.set_xlabel('Geo Lon (°)')
    ax.set_ylabel('Geo Lat (°)')
    z = np.reshape(mag['mag_dip_lat'], alon_2d.shape)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180 + 45, 45))
    plt.yticks(np.arange(-90, 90 + 45, 45))
    mesh = ax.pcolormesh(alon_2d, alat_2d, z)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label('Mag Dip Lat (°)')
    plt.title('Alt = 300 km')
    plt.savefig(figname)
    return


def PyIRI_plot_inc(mag, alon, alat, alon_2d, alat_2d, plot_dir,
                   plot_name='PyIRI_inc.png'):
    """Plot magnetic inclination.

    Parameters
    ----------
    mag : dict
        Dictionary output of IRI_monthly_mean_parameters.
    alon : array-like
        Flattened array of geo longitudes in degrees.
    alat : array-like
        Flattened array of geo latitudes in degrees.
    alon_2d : array-like
        2-D array of geo longitudes in degrees.
    alat_2d : array-like
        2-D array of geo latitudes in degrees.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Name for the output figure, without directory (default='PyIRI_inc.png')

    """
    figname = os.path.join(plot_dir, plot_name)
    fig, ax = plt.subplots(1, 1)
    ax.set_facecolor('grey')
    ax.set_xlabel('Geo Lon (°)')
    ax.set_ylabel('Geo Lat (°)')
    z = np.reshape(mag['inc'], alon_2d.shape)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180 + 45, 45))
    plt.yticks(np.arange(-90, 90 + 45, 45))
    mesh = ax.pcolormesh(alon_2d, alat_2d, z)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label('Inclination (°)')
    plt.title('Alt = 300 km')
    plt.savefig(figname)
    return


def PyIRI_plot_modip(mag, alon, alat, alon_2d, alat_2d, plot_dir,
                     plot_name='PyIRI_modip.png'):
    """Plot modified dip angle.

    Parameters
    ----------
    mag : dict
        Dictionary output of IRI_monthly_mean_parameters.
    alon : array-like
        Flattened array of geo longitudes in degrees.
    alat : array-like
        Flattened array of geo latitudes in degrees.
    alon_2d : array-like
        2-D array of geo longitudes in degrees.
    alat_2d : array-like
        2-D array of geo latitudes in degrees.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Name for the output figure, without directory
        (default='PyIRI_modip.png')

    """
    figname = os.path.join(plot_dir, 'PyIRI_modip.png')
    fig, ax = plt.subplots(1, 1)
    ax.set_facecolor('grey')
    ax.set_xlabel('Geo Lon (°)')
    ax.set_ylabel('Geo Lat (°)')
    z = np.reshape(mag['modip'], alon_2d.shape)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180 + 45, 45))
    plt.yticks(np.arange(-90, 90 + 45, 45))
    mesh = ax.pcolormesh(alon_2d, alat_2d, z)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(r'Modip ($^\circ$)')
    plt.title('Alt = 300 km')
    plt.savefig(figname)
    return


def PyIRI_plot_B_F1_bot_min_max(F1, aUT, alon, alat, alon_2d, alat_2d, sun,
                                UT, plot_dir,
                                plot_name='PyIRI_B_F1_bot_min_max.png'):
    """Plot thickness of F1 bottom side for solar min and max.

    Parameters
    ----------
    F1 : dict
        Dictionary output of IRI_monthly_mean_parameters.
    aUT : array-like
        Array of universal times in hours used in PyIRI.
    alon : array-like
        Flattened array of geo longitudes in degrees.
    alat : array-like
        Flattened array of geo latitudes in degrees.
    alon_2d : array-like
        2-D array of geo longitudes in degrees.
    alat_2d : array-like
        2-D array of geo latitudes in degrees.
    sun : dict
        Dictionary output of IRI_monthly_mean_parameters.
    UT : float
        UT time frame from array aUT to plot.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Name for the output figure, without directory
        (default='PyIRI_B_F1_bot_min_max.png')

    """
    ind_time = np.where(aUT == UT)
    ind_grid = np.where(np.isfinite(alon))
    figname = os.path.join(plot_dir, plot_name)
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 3),
                           constrained_layout=True)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180 + 45, 90))
    plt.yticks(np.arange(-90, 90 + 45, 45))
    abc = ['(a)', '(b)']
    atitle = ['Solar Min', 'Solar Max']
    for isol in range(0, 2):
        ax[isol].set_facecolor('grey')
        ax[isol].set_xlabel('Geo Lon (°)')
        ax[isol].set_ylabel('Geo Lat (°)')
        if isol == 1:
            ax[1].set_ylabel(' ')
        z = np.reshape(F1['B_bot'][ind_time, ind_grid, isol], alon_2d.shape)
        mesh = ax[isol].pcolormesh(alon_2d, alat_2d, z)
        cbar = fig.colorbar(mesh, ax=ax[isol])
        ax[isol].text(140, 70, abc[isol], c='white')
        ax[isol].title.set_text(atitle[isol])
        ax[isol].scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red',
                         s=20, edgecolors="black", linewidths=0.5)
    cbar.set_label('$B^{F1}_{bot}$ (km)')
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    return


def PyIRI_plot_B_F2_bot_min_max(F2, aUT, alon, alat, alon_2d, alat_2d, sun,
                                UT, plot_dir,
                                plot_name='PyIRI_B_F2_bot_min_max.png'):
    """Plot thickness of F2 bottom side for solar min and max.

    Parameters
    ----------
    F2 : dict
        Dictionary output of IRI_monthly_mean_parameters.
    aUT : array-like
        Array of universal times in hours used in PyIRI
    alon : array-like
        Flattened array of geo longitudes in degrees.
    alat : array-like
        Flattened array of geo latitudes in degrees.
    alon_2d : array-like
        2-D array of geo longitudes in degrees.
    alat_2d : array-like
        2-D array of geo latitudes in degrees.
    sun : dict
        Dictionary output of IRI_monthly_mean_parameters.
    UT : float
        UT time frame from array aUT to plot.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Name for the output figure, without directory
        (default='PyIRI_B_F2_bot_min_max.png')

    """
    ind_time = np.where(aUT == UT)
    ind_grid = np.where(np.isfinite(alon))
    figname = os.path.join(plot_dir, plot_name)
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 3),
                           constrained_layout=True)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180 + 45, 90))
    plt.yticks(np.arange(-90, 90 + 45, 45))
    abc = ['(a)', '(b)']
    atitle = ['Solar Min', 'Solar Max']
    for isol in range(0, 2):
        ax[isol].set_facecolor('grey')
        ax[isol].set_xlabel('Geo Lon (°)')
        ax[isol].set_ylabel('Geo Lat (°)')
        if isol == 1:
            ax[1].set_ylabel(' ')
        z = np.reshape(F2['B_bot'][ind_time, ind_grid, isol], alon_2d.shape)
        mesh = ax[isol].pcolormesh(alon_2d, alat_2d, z)
        cbar = fig.colorbar(mesh, ax=ax[isol])
        ax[isol].text(140, 70, abc[isol], c='white')
        ax[isol].title.set_text(atitle[isol])
        ax[isol].scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red',
                         s=20, edgecolors="black", linewidths=0.5)
    cbar.set_label('$B^{F2}_{bot}$ (km)')
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    return


def PyIRI_plot_B_F2_top_min_max(F2, aUT, alon, alat, alon_2d, alat_2d, sun,
                                UT, plot_dir,
                                plot_name='PyIRI_B_F2_top_min_max.png'):
    """Plot thickness of F2 topside for solar min and max.

    Parameters
    ----------
    F2 : dict
        Dictionary output of IRI_monthly_mean_parameters.
    aUT : array-like
        Array of universal times in hours used in PyIRI.
    alon : array-like
        Flattened array of geo longitudes in degrees.
    alat : array-like
        Flattened array of geo latitudes in degrees.
    alon_2d : array-like
        2-D array of geo longitudes in degrees.
    alat_2d : array-like
        2-D array of geo latitudes in degrees.
    sun : dict
        Dictionary output of IRI_monthly_mean_parameters.
    UT : float
        UT time frame from array aUT to plot.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Name for the output figure, without directory
        (default='PyIRI_B_F2_top_min_max.png')

    """
    ind_time = np.where(aUT == UT)
    ind_grid = np.where(np.isfinite(alon))
    figname = os.path.join(plot_dir, plot_name)
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 3),
                           constrained_layout=True)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180 + 45, 90))
    plt.yticks(np.arange(-90, 90 + 45, 45))
    abc = ['(a)', '(b)']
    atitle = ['Solar Min', 'Solar Max']
    for isol in range(0, 2):
        ax[isol].set_facecolor('grey')
        ax[isol].set_xlabel('Geo Lon (°)')
        ax[isol].set_ylabel('Geo Lat (°)')
        if isol == 1:
            ax[1].set_ylabel(' ')
        z = np.reshape(F2['B_top'][ind_time, ind_grid, isol], alon_2d.shape)
        mesh = ax[isol].pcolormesh(alon_2d, alat_2d, z)
        cbar = fig.colorbar(mesh, ax=ax[isol])
        ax[isol].text(140, 70, abc[isol], c='white')
        ax[isol].title.set_text(atitle[isol])
        ax[isol].scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red',
                         s=20, edgecolors="black", linewidths=0.5)
    cbar.set_label('$B^{F2}_{top}$ (km)')
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    return


def PyIRI_plot_M3000_min_max(F2, aUT, alon, alat, alon_2d, alat_2d, sun,
                             UT, plot_dir, plot_name='PyIRI_M3000_min_max.png'):
    """Plot M3000 propagation parameter for solar min and max.

    Parameters
    ----------
    F2 : dict
        Dictionary output of IRI_monthly_mean_parameters.
    aUT : array-like
        Array of universal times in hours used in PyIRI.
    alon : array-like
        Flattened array of geo longitudes in degrees.
    alat : array-like
        Flattened array of geo latitudes in degrees.
    alon_2d : array-like
        2-D array of geo longitudes in degrees.
    alat_2d : array-like
        2-D array of geo latitudes in degrees.
    sun : dict
        Dictionary output of IRI_monthly_mean_parameters.
    UT : float
        UT time frame from array aUT to plot.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Name for the output figure, without directory
        (default='PyIRI_M3000_min_max.png')

    """
    ind_time = np.where(aUT == UT)
    ind_grid = np.where(np.isfinite(alon))
    figname = os.path.join(plot_dir, plot_name)
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 3),
                           constrained_layout=True)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180 + 45, 90))
    plt.yticks(np.arange(-90, 90 + 45, 45))
    abc = ['(a)', '(b)']
    atitle = ['Solar Min', 'Solar Max']
    for isol in range(0, 2):
        ax[isol].set_facecolor('grey')
        ax[isol].set_xlabel('Geo Lon (°)')
        ax[isol].set_ylabel('Geo Lat (°)')
        if isol == 1:
            ax[1].set_ylabel(' ')
        z = np.reshape(F2['M3000'][ind_time, ind_grid, isol], alon_2d.shape)
        mesh = ax[isol].pcolormesh(alon_2d, alat_2d, z)
        cbar = fig.colorbar(mesh, ax=ax[isol])
        ax[isol].text(140, 70, abc[isol], c='white')
        ax[isol].title.set_text(atitle[isol])
        ax[isol].scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red',
                         s=20, edgecolors="black", linewidths=0.5)
    cbar.set_label('MUF(3000)F2/$fo$F2')
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    return


def PyIRI_plot_hmF2_min_max(F2, aUT, alon, alat, alon_2d, alat_2d, sun,
                            UT, plot_dir, plot_name='PyIRI_hmF2_min_max.png'):
    """Plot hmF2 for solar min and max.

    Parameters
    ----------
    F2 : dict
        Dictionary output of IRI_monthly_mean_parameters.
    aUT : array-like
        Array of universal times in hours used in PyIRI.
    alon : array-like
        Flattened array of geo longitudes in degrees.
    alat : array-like
        Flattened array of geo latitudes in degrees.
    alon_2d : array-like
        2-D array of geo longitudes in degrees.
    alat_2d : array-like
        2-D array of geo latitudes in degrees.
    sun : dict
        Dictionary output of IRI_monthly_mean_parameters.
    UT : float
        UT time frame from array aUT to plot.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Name for the output figure, without directory
        (default='PyIRI_hmF2_min_max.png')

    """
    ind_time = np.where(aUT == UT)
    ind_grid = np.where(np.isfinite(alon))
    figname = os.path.join(plot_dir, plot_name)
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 3),
                           constrained_layout=True)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180 + 45, 90))
    plt.yticks(np.arange(-90, 90 + 45, 45))
    abc = ['(a)', '(b)']
    atitle = ['Solar Min', 'Solar Max']
    for isol in range(0, 2):
        ax[isol].set_facecolor('grey')
        ax[isol].set_xlabel('Geo Lon (°)')
        ax[isol].set_ylabel('Geo Lat (°)')
        if isol == 1:
            ax[1].set_ylabel(' ')
        z = np.reshape(F2['hm'][ind_time, ind_grid, isol], alon_2d.shape)
        mesh = ax[isol].pcolormesh(alon_2d, alat_2d, z)
        cbar = fig.colorbar(mesh, ax=ax[isol])
        ax[isol].text(140, 70, abc[isol], c='white')
        ax[isol].title.set_text(atitle[isol])
        ax[isol].scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red',
                         s=20, edgecolors="black", linewidths=0.5)
    cbar.set_label('$hm$F2 (km)')
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    return


def PyIRI_plot_hmF1_min_max(F1, aUT, alon, alat, alon_2d, alat_2d, sun,
                            UT, plot_dir, plot_name='PyIRI_hmF1_min_max.png'):
    """Plot hmF1 for solar min and max.

    Parameters
    ----------
    F1 : dict
        Dictionary output of IRI_monthly_mean_parameters.
    aUT : array-like
        Array of universal times in hours used in PyIRI.
    alon : array-like
        Flattened array of geo longitudes in degrees.
    alat : array-like
        Flattened array of geo latitudes in degrees.
    alon_2d : array-like
        2-D array of geo longitudes in degrees.
    alat_2d : array-like
        2-D array of geo latitudes in degrees.
    sun : dict
        Dictionary output of IRI_monthly_mean_parameters.
    UT : float
        UT time frame from array aUT to plot.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Name for the output figure, without directory
        (default='PyIRI_hmF1_min_max.png')

    """
    ind_time = np.where(aUT == UT)
    ind_grid = np.where(np.isfinite(alon))
    figname = os.path.join(plot_dir, plot_name)
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 3),
                           constrained_layout=True)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180 + 45, 90))
    plt.yticks(np.arange(-90, 90 + 45, 45))
    abc = ['(a)', '(b)']
    atitle = ['Solar Min', 'Solar Max']
    for isol in range(0, 2):
        ax[isol].set_facecolor('grey')
        ax[isol].set_xlabel('Geo Lon (°)')
        ax[isol].set_ylabel('Geo Lat (°)')
        if isol == 1:
            ax[1].set_ylabel(' ')
        z = np.reshape(F1['hm'][ind_time, ind_grid, isol], alon_2d.shape)
        mesh = ax[isol].pcolormesh(alon_2d, alat_2d, z)
        cbar = fig.colorbar(mesh, ax=ax[isol])
        ax[isol].text(140, 70, abc[isol], c='white')
        ax[isol].title.set_text(atitle[isol])
        ax[isol].scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red',
                         s=20, edgecolors="black", linewidths=0.5)
    cbar.set_label('$hm$F1 (km)')
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    return


def PyIRI_plot_foEs_min_max(Es, aUT, alon, alat, alon_2d, alat_2d, sun,
                            UT, plot_dir, plot_name='PyIRI_foEs_min_max.png'):
    """Plot foEs for solar min and max.

    Parameters
    ----------
    Es : dict
        Dictionary output of IRI_monthly_mean_parameters.
    aUT : array-like
        Array of universal times in hours used in PyIRI.
    alon : array-like
        Flattened array of geo longitudes in degrees.
    alat : array-like
        Flattened array of geo latitudes in degrees.
    alon_2d : array-like
        2-D array of geo longitudes in degrees.
    alat_2d : array-like
        2-D array of geo latitudes in degrees.
    sun : dict
        Dictionary output of IRI_monthly_mean_parameters.
    UT : float
        UT time frame from array aUT to plot.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Name for the output figure, without directory
        (default='PyIRI_foEs_min_max.png')

    """
    ind_time = np.where(aUT == UT)
    ind_grid = np.where(np.isfinite(alon))
    figname = os.path.join(plot_dir, plot_name)
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 3),
                           constrained_layout=True)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180 + 45, 90))
    plt.yticks(np.arange(-90, 90 + 45, 45))
    abc = ['(a)', '(b)']
    atitle = ['Solar Min', 'Solar Max']
    for isol in range(0, 2):
        ax[isol].set_facecolor('grey')
        ax[isol].set_xlabel('Geo Lon (°)')
        ax[isol].set_ylabel('Geo Lat (°)')
        if isol == 1:
            ax[1].set_ylabel(' ')
        z = np.reshape(Es['fo'][ind_time, ind_grid, isol], alon_2d.shape)
        mesh = ax[isol].pcolormesh(alon_2d, alat_2d, z)
        cbar = fig.colorbar(mesh, ax=ax[isol])
        ax[isol].text(140, 70, abc[isol], c='white')
        ax[isol].title.set_text(atitle[isol])
        ax[isol].scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red',
                         s=20, edgecolors="black", linewidths=0.5)
    cbar.set_label('$fo$Es (MHz)')
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    return


def PyIRI_plot_foE_min_max(E, aUT, alon, alat, alon_2d, alat_2d, sun,
                           UT, plot_dir, plot_name='PyIRI_foE_min_max.png'):
    """Plot foE for solar min and max.

    Parameters
    ----------
    E : dict
        Dictionary output of IRI_monthly_mean_parameters.
    aUT : array-like
        Array of universal times in hours used in PyIRI.
    alon : array-like
        Flattened array of geo longitudes in degrees.
    alat : array-like
        Flattened array of geo latitudes in degrees.
    alon_2d : array-like
        2-D array of geo longitudes in degrees.
    alat_2d : array-like
        2-D array of geo latitudes in degrees.
    sun : dict
        Dictionary output of IRI_monthly_mean_parameters.
    UT : float
        UT time frame from array aUT to plot.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Name for the output figure, without directory
        (default='PyIRI_foE_min_max.png')

    """
    ind_time = np.where(aUT == UT)
    ind_grid = np.where(np.isfinite(alon))
    figname = os.path.join(plot_dir, plot_name)
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 3),
                           constrained_layout=True)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180 + 45, 90))
    plt.yticks(np.arange(-90, 90 + 45, 45))
    abc = ['(a)', '(b)']
    atitle = ['Solar Min', 'Solar Max']
    for isol in range(0, 2):
        ax[isol].set_facecolor('grey')
        ax[isol].set_xlabel('Geo Lon (°)')
        ax[isol].set_ylabel('Geo Lat (°)')
        if isol == 1:
            ax[1].set_ylabel(' ')
        z = np.reshape(E['fo'][ind_time, ind_grid, isol], alon_2d.shape)
        mesh = ax[isol].pcolormesh(alon_2d, alat_2d, z)
        cbar = fig.colorbar(mesh, ax=ax[isol])
        ax[isol].text(140, 70, abc[isol], c='white')
        ax[isol].title.set_text(atitle[isol])
        ax[isol].scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red',
                         s=20, edgecolors="black", linewidths=0.5)
    cbar.set_label('$fo$E (MHz)')
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    return


def PyIRI_plot_foF2_min_max(F2, aUT, alon, alat, alon_2d, alat_2d, sun,
                            UT, plot_dir, plot_name='PyIRI_foF2_min_max.png'):
    """Plot foF2 for solar min and max.

    Parameters
    ----------
    F2 : dict
        Dictionary output of IRI_monthly_mean_parameters.
    aUT : array-like
        Array of universal times in hours used in PyIRI.
    alon : array-like
        Flattened array of geo longitudes in degrees.
    alat : array-like
        Flattened array of geo latitudes in degrees.
    alon_2d : array-like
        2-D array of geo longitudes in degrees.
    alat_2d : array-like
        2-D array of geo latitudes in degrees.
    sun : dict
        Dictionary output of IRI_monthly_mean_parameters.
    UT : float
        UT time frame from array aUT to plot.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Name for the output figure, without directory
        (default='PyIRI_foF2_min_max.png')

    """
    ind_time = np.where(aUT == UT)
    ind_grid = np.where(np.isfinite(alon))
    figname = os.path.join(plot_dir, plot_name)
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 3),
                           constrained_layout=True)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180 + 45, 90))
    plt.yticks(np.arange(-90, 90 + 45, 45))
    abc = ['(a)', '(b)']
    atitle = ['Solar Min', 'Solar Max']
    for isol in range(0, 2):
        ax[isol].set_facecolor('grey')
        ax[isol].set_xlabel('Geo Lon (°)')
        ax[isol].set_ylabel('Geo Lat (°)')
        if isol == 1:
            ax[1].set_ylabel(' ')
        z = np.reshape(F2['fo'][ind_time, ind_grid, isol], alon_2d.shape)
        mesh = ax[isol].pcolormesh(alon_2d, alat_2d, z)
        cbar = fig.colorbar(mesh, ax=ax[isol])
        ax[isol].text(140, 70, abc[isol], c='white')
        ax[isol].title.set_text(atitle[isol])
        ax[isol].scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red',
                         s=20, edgecolors="black", linewidths=0.5)
    cbar.set_label('$fo$F2 (MHz)')
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    return


def PyIRI_plot_NmF2_min_max(F2, aUT, alon, alat, alon_2d, alat_2d, sun,
                            UT, plot_dir, plot_name='PyIRI_NmF2_min_max.png'):
    """Plot NmF2 for solar min and max.

    Parameters
    ----------
    F2 : dict
        Dictionary output of IRI_monthly_mean_parameters.
    aUT : array-like
        Array of universal times in hours used in PyIRI.
    alon : array-like
        Flattened array of geo longitudes in degrees.
    alat : array-like
        Flattened array of geo latitudes in degrees.
    alon_2d : array-like
        2-D array of geo longitudes in degrees.
    alat_2d : array-like
        2-D array of geo latitudes in degrees.
    sun : dict
        Dictionary output of IRI_monthly_mean_parameters.
    UT : float
        UT time frame from array aUT to plot.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Name for the output figure, without directory
        (default='PyIRI_NmF2_min_max.png')

    """
    ind_time = np.where(aUT == UT)
    ind_grid = np.where(np.isfinite(alon))
    figname = os.path.join(plot_dir, plot_name)
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 3),
                           constrained_layout=True)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180 + 45, 90))
    plt.yticks(np.arange(-90, 90 + 45, 45))
    abc = ['(a)', '(b)']
    atitle = ['Solar Min', 'Solar Max']
    for isol in range(0, 2):
        ax[isol].set_facecolor('grey')
        ax[isol].set_xlabel('Geo Lon (°)')
        ax[isol].set_ylabel('Geo Lat (°)')
        if isol == 1:
            ax[1].set_ylabel(' ')
        z = np.reshape(F2['Nm'][ind_time, ind_grid, isol], alon_2d.shape)
        mesh = ax[isol].pcolormesh(alon_2d, alat_2d, z)
        cbar = fig.colorbar(mesh, ax=ax[isol])
        ax[isol].text(140, 70, abc[isol], c='white')
        ax[isol].title.set_text(atitle[isol])
        ax[isol].scatter(sun['lon'][ind_time], sun['lat'][ind_time],
                         c='red', s=20, edgecolors="black", linewidths=0.5)
    cbar.set_label('$Nm$F2 (m$^{-3}$)')
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    return


def PyIRI_plot_NmF1_min_max(F1, aUT, alon, alat, alon_2d, alat_2d, sun,
                            UT, plot_dir, plot_name='PyIRI_NmF1_min_max.png'):
    """Plot NmF1 for solar min and max.

    Parameters
    ----------
    F1 : dict
        Dictionary output of IRI_monthly_mean_parameters.
    aUT : array-like
        Array of universal times in hours used in PyIRI.
    alon : array-like
        Flattened array of geo longitudes in degrees.
    alat : array-like
        Flattened array of geo latitudes in degrees.
    alon_2d : array-like
        2-D array of geo longitudes in degrees.
    alat_2d : array-like
        2-D array of geo latitudes in degrees.
    sun : dict
        Dictionary output of IRI_monthly_mean_parameters.
    UT : float
        UT time frame from array aUT to plot.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Name for the output figure, without directory
        (default='PyIRI_NmF1_min_max.png')

    """
    ind_time = np.where(aUT == UT)
    ind_grid = np.where(np.isfinite(alon))
    figname = os.path.join(plot_dir, plot_name)
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 3),
                           constrained_layout=True)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180 + 45, 90))
    plt.yticks(np.arange(-90, 90 + 45, 45))
    abc = ['(a)', '(b)']
    atitle = ['Solar Min', 'Solar Max']
    for isol in range(0, 2):
        ax[isol].set_facecolor('grey')
        ax[isol].set_xlabel('Geo Lon (°)')
        ax[isol].set_ylabel('Geo Lat (°)')
        if isol == 1:
            ax[1].set_ylabel(' ')
        z = np.reshape(F1['Nm'][ind_time, ind_grid, isol], alon_2d.shape)
        mesh = ax[isol].pcolormesh(alon_2d, alat_2d, z)
        cbar = fig.colorbar(mesh, ax=ax[isol])
        ax[isol].text(140, 70, abc[isol], c='white')
        ax[isol].title.set_text(atitle[isol])
        ax[isol].scatter(sun['lon'][ind_time], sun['lat'][ind_time],
                         c='red', s=20, edgecolors="black", linewidths=0.5)
    cbar.set_label('$Nm$F1 (m$^{-3}$)')
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    return


def PyIRI_plot_foF1_min_max(F1, aUT, alon, alat, alon_2d, alat_2d, sun,
                            UT, plot_dir, plot_name='PyIRI_foF1_min_max.png'):
    """Plot foF1 for solar min and max.

    Parameters
    ----------
    F1 : dict
        Dictionary output of IRI_monthly_mean_parameters.
    aUT : array-like
        Array of universal times in hours used in PyIRI.
    alon : array-like
        Flattened array of geo longitudes in degrees.
    alat : array-like
        Flattened array of geo latitudes in degrees.
    alon_2d : array-like
        2-D array of geo longitudes in degrees.
    alat_2d : array-like
        2-D array of geo latitudes in degrees.
    sun : dict
        Dictionary output of IRI_monthly_mean_parameters.
    UT : float
        UT time frame from array aUT to plot.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Name for the output figure, without directory
        (default='PyIRI_foF1_min_max.png')

    """
    ind_time = np.where(aUT == UT)
    ind_grid = np.where(np.isfinite(alon))
    figname = os.path.join(plot_dir, plot_name)
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 3),
                           constrained_layout=True)
    ax[0].set_facecolor('grey')
    ax[0].set_xlabel('Geo Lon (°)')
    ax[0].set_ylabel('Geo Lat (°)')
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180 + 45, 90))
    plt.yticks(np.arange(-90, 90 + 45, 45))
    abc = ['(a)', '(b)', '(c)']
    ax[0].text(130, 70, abc[0], c='white')
    P = F1['P'][ind_time, ind_grid, 0]
    foF1_min = F1['fo'][ind_time, ind_grid, 0]
    foF1_max = F1['fo'][ind_time, ind_grid, 1]
    # --------------------------------
    z = np.reshape(P, alon_2d.shape)
    mesh = ax[0].pcolormesh(alon_2d, alat_2d, z)
    cbar0 = fig.colorbar(mesh, ax=ax[0])
    ax[0].scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red', s=20,
                  edgecolors="black", linewidths=0.5, zorder=2)
    cbar0.set_label('Probability')
    ax[1].set_facecolor('grey')
    ax[1].set_xlabel('Geo Lon (°)')
    ax[1].set_ylabel(' ')
    ax[1].text(130, 70, abc[1], c='white')
    ax[1].scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red', s=20,
                  edgecolors="black", linewidths=0.5, zorder=2)
    # --------------------------------
    z = np.reshape(foF1_min, alon_2d.shape)
    mesh = ax[1].pcolormesh(alon_2d, alat_2d, z)
    cbar1 = fig.colorbar(mesh, ax=ax[1])
    cbar1.set_label('$fo$F1 (MHz)')
    ax[2].set_facecolor('grey')
    ax[2].set_xlabel('Geo Lon (°)')
    ax[2].set_ylabel(' ')
    ax[2].text(130, 70, abc[2], c='white')
    ax[2].scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red', s=20,
                  edgecolors="black", linewidths=0.5, zorder=2)
    # --------------------------------
    z2 = np.reshape(foF1_max, alon_2d.shape)
    mesh = ax[2].pcolormesh(alon_2d, alat_2d, z2)
    cbar = fig.colorbar(mesh, ax=ax[2])
    cbar.set_label('$fo$F1 (MHz)')
    ax[0].title.set_text('Probability')
    ax[1].title.set_text('Solar Min')
    ax[2].title.set_text('Solar Max')
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    return


def PyIRI_EDP_sample(EDP, aUT, alon, alat, lon_plot, lat_plot, aalt,
                     UT, plot_dir, plot_name='PyIRI_EDP_sample.png'):
    """Plot EDP for one location for solar min and max.

    Parameters
    ----------
    EDP : array-like
        3-D electron density array output of IRI_monthly_mean_parameters.
    aUT : array-like
        Array of universal times in hours used in PyIRI
    alon : array-like
        Flattened array of geo longitudes in degrees.
    alat : array-like
        Flattened array of geo latitudes in degrees.
    lon_plot : float
        Longitude location for EDP.
    lat_plot : array-like
        Latitude location for EDP.
    sun : dict
        Dictionary output of IRI_monthly_mean_parameters.
    UT : float
        UT time frame from array aUT to plot.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Name for the output figure, without directory
        (default='PyIRI_EDP_sample.png')

    """
    figname = os.path.join(plot_dir, plot_name)
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(5, 5),
                           constrained_layout=True)
    ax.set_xlabel('Electron Density (m$^{-3}$)')
    ax.set_ylabel('Altitude (km)')
    ax.set_facecolor("grey")
    ind_grid = np.where((alon == lon_plot) & (alat == lat_plot))
    ind_time = np.where(aUT == UT)
    ind_vert = np.where(aalt >= 0)
    ind_min = 0, ind_time, ind_vert, ind_grid
    x = np.reshape(EDP[ind_min], aalt.shape)
    ax.plot(x, aalt, c='black', label='Sol min', linewidth=1)
    ind_max = 1, ind_time, ind_vert, ind_grid
    x = np.reshape(EDP[ind_max], aalt.shape)
    ax.plot(x, aalt, c='white', label='Sol max', linewidth=1)
    ax.legend(loc='upper right', prop={'size': 8})
    plt.title(str(lon_plot) + ' Lon, ' + str(lat_plot) + ' Lat')
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    return


def PyIRI_EDP_sample_1day(EDP, aUT, alon, alat, lon_plot, lat_plot, aalt,
                          UT, plot_dir, plot_name='PyIRI_EDP_sample_1day.png'):
    """Plot EDP for one location.

    Parameters
    ----------
    EDP : array-like
        3-D electron density array output of IRI_density_1day.
    aUT : array-like
        Array of universal times in hours used in PyIRI
    alon : array-like
        Flattened array of geo longitudes in degrees.
    alat : array-like
        Flattened array of geo latitudes in degrees.
    lon_plot : float
        Longitude location for EDP.
    lat_plot : array-like
        Latitude location for EDP.
    sun : dict
        Dictionary output of IRI_monthly_mean_parameters.
    UT : float
        UT time frame from array aUT to plot.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Name for the output figure, without directory
        (default='PyIRI_EDP_sample_1day.png')

    """
    figname = os.path.join(plot_dir, plot_name)
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(5, 5),
                           constrained_layout=True)
    ax.set_xlabel('Electron Density (m$^{-3}$)')
    ax.set_ylabel('Altitude (km)')
    ax.set_facecolor("grey")
    ind_grid = np.where((alon == lon_plot) & (alat == lat_plot))
    ind_time = np.where(aUT == UT)
    ind_vert = np.where(aalt >= 0)
    ind = ind_time, ind_vert, ind_grid
    x = np.reshape(EDP[ind], aalt.shape)
    ax.plot(x, aalt, c='black', linewidth=1)
    plt.title(str(lon_plot) + ' Lon, ' + str(lat_plot) + ' Lat')
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    return


def PyIRI_plot_B_F1_bot(F1, aUT, alon, alat, alon_2d, alat_2d, sun,
                        UT, plot_dir, plot_name='PyIRI_B_F1_bot.png'):
    """Plot thickness of F1 bottom side.

    Parameters
    ----------
    F1 : dict
        Dictionary output of IRI_monthly_mean_parameters.
    aUT : array-like
        Array of universal times in hours used in PyIRI.
    alon : array-like
        Flattened array of geo longitudes in degrees.
    alat : array-like
        Flattened array of geo latitudes in degrees.
    alon_2d : array-like
        2-D array of geo longitudes in degrees.
    alat_2d : array-like
        2-D array of geo latitudes in degrees.
    sun : dict
        Dictionary output of IRI_monthly_mean_parameters.
    UT : float
        UT time frame from array aUT to plot.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Name for the output figure, without directory
        (default='PyIRI_B_F1_bot.png')

    """
    ind_time = np.where(aUT == UT)
    ind_grid = np.where(np.isfinite(alon))
    figname = os.path.join(plot_dir, plot_name)
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6, 3),
                           constrained_layout=True)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180 + 45, 90))
    plt.yticks(np.arange(-90, 90 + 45, 45))
    ax.set_facecolor('grey')
    ax.set_xlabel('Geo Lon (°)')
    ax.set_ylabel('Geo Lat (°)')
    z = np.reshape(F1['B_bot'][ind_time, ind_grid], alon_2d.shape)
    mesh = ax.pcolormesh(alon_2d, alat_2d, z)
    cbar = fig.colorbar(mesh, ax=ax)
    ax.scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red',
               s=20, edgecolors="black", linewidths=0.5)
    cbar.set_label('$B^{F1}_{bot}$ (km)')
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    return


def PyIRI_plot_B_F2_bot(F2, aUT, alon, alat, alon_2d, alat_2d, sun,
                        UT, plot_dir,
                        plot_name='PyIRI_B_F2_bot.png'):
    """Plot thickness of F2 bottom side.

    Parameters
    ----------
    F2 : dict
        Dictionary output of IRI_monthly_mean_parameters.
    aUT : array-like
        Array of universal times in hours used in PyIRI
    alon : array-like
        Flattened array of geo longitudes in degrees.
    alat : array-like
        Flattened array of geo latitudes in degrees.
    alon_2d : array-like
        2-D array of geo longitudes in degrees.
    alat_2d : array-like
        2-D array of geo latitudes in degrees.
    sun : dict
        Dictionary output of IRI_monthly_mean_parameters.
    UT : float
        UT time frame from array aUT to plot.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Name for the output figure, without directory
        (default='PyIRI_B_F2_bot.png')

    """
    ind_time = np.where(aUT == UT)
    ind_grid = np.where(np.isfinite(alon))
    figname = os.path.join(plot_dir, plot_name)
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6, 3),
                           constrained_layout=True)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180 + 45, 90))
    plt.yticks(np.arange(-90, 90 + 45, 45))
    ax.set_facecolor('grey')
    ax.set_xlabel('Geo Lon (°)')
    ax.set_ylabel('Geo Lat (°)')
    z = np.reshape(F2['B_bot'][ind_time, ind_grid], alon_2d.shape)
    mesh = ax.pcolormesh(alon_2d, alat_2d, z)
    cbar = fig.colorbar(mesh, ax=ax)
    ax.scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red',
               s=20, edgecolors="black", linewidths=0.5)
    cbar.set_label('$B^{F2}_{bot}$ (km)')
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    return


def PyIRI_plot_B_F2_top(F2, aUT, alon, alat, alon_2d, alat_2d, sun,
                        UT, plot_dir,
                        plot_name='PyIRI_B_F2_top.png'):
    """Plot thickness of F2 topside.

    Parameters
    ----------
    F2 : dict
        Dictionary output of IRI_monthly_mean_parameters.
    aUT : array-like
        Array of universal times in hours used in PyIRI.
    alon : array-like
        Flattened array of geo longitudes in degrees.
    alat : array-like
        Flattened array of geo latitudes in degrees.
    alon_2d : array-like
        2-D array of geo longitudes in degrees.
    alat_2d : array-like
        2-D array of geo latitudes in degrees.
    sun : dict
        Dictionary output of IRI_monthly_mean_parameters.
    UT : float
        UT time frame from array aUT to plot.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Name for the output figure, without directory
        (default='PyIRI_B_F2_top.png')

    """
    ind_time = np.where(aUT == UT)
    ind_grid = np.where(np.isfinite(alon))
    figname = os.path.join(plot_dir, plot_name)
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6, 3),
                           constrained_layout=True)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180 + 45, 90))
    plt.yticks(np.arange(-90, 90 + 45, 45))
    ax.set_facecolor('grey')
    ax.set_xlabel('Geo Lon (°)')
    ax.set_ylabel('Geo Lat (°)')
    z = np.reshape(F2['B_top'][ind_time, ind_grid], alon_2d.shape)
    mesh = ax.pcolormesh(alon_2d, alat_2d, z)
    cbar = fig.colorbar(mesh, ax=ax)
    ax.scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red',
               s=20, edgecolors="black", linewidths=0.5)
    cbar.set_label('$B^{F2}_{top}$ (km)')
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    return


def PyIRI_plot_M3000(F2, aUT, alon, alat, alon_2d, alat_2d, sun,
                     UT, plot_dir, plot_name='PyIRI_M3000.png'):
    """Plot M3000 propagation parameter.

    Parameters
    ----------
    F2 : dict
        Dictionary output of IRI_monthly_mean_parameters.
    aUT : array-like
        Array of universal times in hours used in PyIRI.
    alon : array-like
        Flattened array of geo longitudes in degrees.
    alat : array-like
        Flattened array of geo latitudes in degrees.
    alon_2d : array-like
        2-D array of geo longitudes in degrees.
    alat_2d : array-like
        2-D array of geo latitudes in degrees.
    sun : dict
        Dictionary output of IRI_monthly_mean_parameters.
    UT : float
        UT time frame from array aUT to plot.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Name for the output figure, without directory
        (default='PyIRI_M3000.png')

    """
    ind_time = np.where(aUT == UT)
    ind_grid = np.where(np.isfinite(alon))
    figname = os.path.join(plot_dir, plot_name)
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6, 3),
                           constrained_layout=True)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180 + 45, 90))
    plt.yticks(np.arange(-90, 90 + 45, 45))
    ax.set_facecolor('grey')
    ax.set_xlabel('Geo Lon (°)')
    ax.set_ylabel('Geo Lat (°)')
    z = np.reshape(F2['M3000'][ind_time, ind_grid], alon_2d.shape)
    mesh = ax.pcolormesh(alon_2d, alat_2d, z)
    cbar = fig.colorbar(mesh, ax=ax)
    ax.scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red',
               s=20, edgecolors="black", linewidths=0.5)
    cbar.set_label('MUF(3000)F2/$fo$F2')
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    return


def PyIRI_plot_hmF2(F2, aUT, alon, alat, alon_2d, alat_2d, sun,
                    UT, plot_dir, plot_name='PyIRI_hmF2.png'):
    """Plot hmF2.

    Parameters
    ----------
    F2 : dict
        Dictionary output of IRI_monthly_mean_parameters.
    aUT : array-like
        Array of universal times in hours used in PyIRI.
    alon : array-like
        Flattened array of geo longitudes in degrees.
    alat : array-like
        Flattened array of geo latitudes in degrees.
    alon_2d : array-like
        2-D array of geo longitudes in degrees.
    alat_2d : array-like
        2-D array of geo latitudes in degrees.
    sun : dict
        Dictionary output of IRI_monthly_mean_parameters.
    UT : float
        UT time frame from array aUT to plot.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Name for the output figure, without directory
        (default='PyIRI_hmF2.png')

    """
    ind_time = np.where(aUT == UT)
    ind_grid = np.where(np.isfinite(alon))
    figname = os.path.join(plot_dir, plot_name)
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6, 3),
                           constrained_layout=True)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180 + 45, 90))
    plt.yticks(np.arange(-90, 90 + 45, 45))
    ax.set_facecolor('grey')
    ax.set_xlabel('Geo Lon (°)')
    ax.set_ylabel('Geo Lat (°)')
    z = np.reshape(F2['hm'][ind_time, ind_grid], alon_2d.shape)
    mesh = ax.pcolormesh(alon_2d, alat_2d, z)
    cbar = fig.colorbar(mesh, ax=ax)
    ax.scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red',
               s=20, edgecolors="black", linewidths=0.5)
    cbar.set_label('$hm$F2 (km)')
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    return


def PyIRI_plot_hmF1(F1, aUT, alon, alat, alon_2d, alat_2d, sun,
                    UT, plot_dir, plot_name='PyIRI_hmF1.png'):
    """Plot hmF1 for solar min and max.

    Parameters
    ----------
    F1 : dict
        Dictionary output of IRI_monthly_mean_parameters.
    aUT : array-like
        Array of universal times in hours used in PyIRI.
    alon : array-like
        Flattened array of geo longitudes in degrees.
    alat : array-like
        Flattened array of geo latitudes in degrees.
    alon_2d : array-like
        2-D array of geo longitudes in degrees.
    alat_2d : array-like
        2-D array of geo latitudes in degrees.
    sun : dict
        Dictionary output of IRI_monthly_mean_parameters.
    UT : float
        UT time frame from array aUT to plot.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Name for the output figure, without directory
        (default='PyIRI_hmF1_min_max.png')

    """
    ind_time = np.where(aUT == UT)
    ind_grid = np.where(np.isfinite(alon))
    figname = os.path.join(plot_dir, plot_name)
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6, 3),
                           constrained_layout=True)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180 + 45, 90))
    plt.yticks(np.arange(-90, 90 + 45, 45))
    ax.set_facecolor('grey')
    ax.set_xlabel('Geo Lon (°)')
    ax.set_ylabel('Geo Lat (°)')
    z = np.reshape(F1['hm'][ind_time, ind_grid], alon_2d.shape)
    mesh = ax.pcolormesh(alon_2d, alat_2d, z)
    cbar = fig.colorbar(mesh, ax=ax)
    ax.scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red',
               s=20, edgecolors="black", linewidths=0.5)
    cbar.set_label('$hm$F1 (km)')
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    return


def PyIRI_plot_foEs(Es, aUT, alon, alat, alon_2d, alat_2d, sun,
                    UT, plot_dir, plot_name='PyIRI_foEs.png'):
    """Plot foEs.

    Parameters
    ----------
    Es : dict
        Dictionary output of IRI_monthly_mean_parameters.
    aUT : array-like
        Array of universal times in hours used in PyIRI.
    alon : array-like
        Flattened array of geo longitudes in degrees.
    alat : array-like
        Flattened array of geo latitudes in degrees.
    alon_2d : array-like
        2-D array of geo longitudes in degrees.
    alat_2d : array-like
        2-D array of geo latitudes in degrees.
    sun : dict
        Dictionary output of IRI_monthly_mean_parameters.
    UT : float
        UT time frame from array aUT to plot.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Name for the output figure, without directory
        (default='PyIRI_foEs_min_max.png')

    """
    ind_time = np.where(aUT == UT)
    ind_grid = np.where(np.isfinite(alon))
    figname = os.path.join(plot_dir, plot_name)
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6, 3),
                           constrained_layout=True)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180 + 45, 90))
    plt.yticks(np.arange(-90, 90 + 45, 45))
    ax.set_facecolor('grey')
    ax.set_xlabel('Geo Lon (°)')
    ax.set_ylabel('Geo Lat (°)')
    z = np.reshape(Es['fo'][ind_time, ind_grid], alon_2d.shape)
    mesh = ax.pcolormesh(alon_2d, alat_2d, z)
    cbar = fig.colorbar(mesh, ax=ax)
    ax.scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red',
               s=20, edgecolors="black", linewidths=0.5)
    cbar.set_label('$fo$Es (MHz)')
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    return


def PyIRI_plot_foE(E, aUT, alon, alat, alon_2d, alat_2d, sun,
                   UT, plot_dir, plot_name='PyIRI_foE.png'):
    """Plot foE.

    Parameters
    ----------
    E : dict
        Dictionary output of IRI_monthly_mean_parameters.
    aUT : array-like
        Array of universal times in hours used in PyIRI.
    alon : array-like
        Flattened array of geo longitudes in degrees.
    alat : array-like
        Flattened array of geo latitudes in degrees.
    alon_2d : array-like
        2-D array of geo longitudes in degrees.
    alat_2d : array-like
        2-D array of geo latitudes in degrees.
    sun : dict
        Dictionary output of IRI_monthly_mean_parameters.
    UT : float
        UT time frame from array aUT to plot.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Name for the output figure, without directory
        (default='PyIRI_foE_min_max.png')

    """
    ind_time = np.where(aUT == UT)
    ind_grid = np.where(np.isfinite(alon))
    figname = os.path.join(plot_dir, plot_name)
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6, 3),
                           constrained_layout=True)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180 + 45, 90))
    plt.yticks(np.arange(-90, 90 + 45, 45))
    ax.set_facecolor('grey')
    ax.set_xlabel('Geo Lon (°)')
    ax.set_ylabel('Geo Lat (°)')
    z = np.reshape(E['fo'][ind_time, ind_grid], alon_2d.shape)
    mesh = ax.pcolormesh(alon_2d, alat_2d, z)
    cbar = fig.colorbar(mesh, ax=ax)
    ax.scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red',
               s=20, edgecolors="black", linewidths=0.5)
    cbar.set_label('$fo$E (MHz)')
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    return


def PyIRI_plot_foF2(F2, aUT, alon, alat, alon_2d, alat_2d, sun,
                    UT, plot_dir, plot_name='PyIRI_foF2.png'):
    """Plot foF2.

    Parameters
    ----------
    F2 : dict
        Dictionary output of IRI_monthly_mean_parameters.
    aUT : array-like
        Array of universal times in hours used in PyIRI.
    alon : array-like
        Flattened array of geo longitudes in degrees.
    alat : array-like
        Flattened array of geo latitudes in degrees.
    alon_2d : array-like
        2-D array of geo longitudes in degrees.
    alat_2d : array-like
        2-D array of geo latitudes in degrees.
    sun : dict
        Dictionary output of IRI_monthly_mean_parameters.
    UT : float
        UT time frame from array aUT to plot.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Name for the output figure, without directory
        (default='PyIRI_foF2_min_max.png')

    """
    ind_time = np.where(aUT == UT)
    ind_grid = np.where(np.isfinite(alon))
    figname = os.path.join(plot_dir, plot_name)
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6, 3),
                           constrained_layout=True)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180 + 45, 90))
    plt.yticks(np.arange(-90, 90 + 45, 45))
    ax.set_facecolor('grey')
    ax.set_xlabel('Geo Lon (°)')
    ax.set_ylabel('Geo Lat (°)')
    z = np.reshape(F2['fo'][ind_time, ind_grid], alon_2d.shape)
    mesh = ax.pcolormesh(alon_2d, alat_2d, z)
    cbar = fig.colorbar(mesh, ax=ax)
    ax.scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red',
               s=20, edgecolors="black", linewidths=0.5)
    cbar.set_label('$fo$F2 (MHz)')
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    return


def PyIRI_plot_NmF2(F2, aUT, alon, alat, alon_2d, alat_2d, sun,
                    UT, plot_dir, plot_name='PyIRI_NmF2.png'):
    """Plot NmF2.

    Parameters
    ----------
    F2 : dict
        Dictionary output of IRI_monthly_mean_parameters.
    aUT : array-like
        Array of universal times in hours used in PyIRI.
    alon : array-like
        Flattened array of geo longitudes in degrees.
    alat : array-like
        Flattened array of geo latitudes in degrees.
    alon_2d : array-like
        2-D array of geo longitudes in degrees.
    alat_2d : array-like
        2-D array of geo latitudes in degrees.
    sun : dict
        Dictionary output of IRI_monthly_mean_parameters.
    UT : float
        UT time frame from array aUT to plot.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Name for the output figure, without directory
        (default='PyIRI_NmF2_min_max.png')

    """
    ind_time = np.where(aUT == UT)
    ind_grid = np.where(np.isfinite(alon))
    figname = os.path.join(plot_dir, plot_name)
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6, 3),
                           constrained_layout=True)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180 + 45, 90))
    plt.yticks(np.arange(-90, 90 + 45, 45))
    ax.set_facecolor('grey')
    ax.set_xlabel('Geo Lon (°)')
    ax.set_ylabel('Geo Lat (°)')
    z = np.reshape(F2['Nm'][ind_time, ind_grid], alon_2d.shape)
    mesh = ax.pcolormesh(alon_2d, alat_2d, z)
    cbar = fig.colorbar(mesh, ax=ax)
    ax.scatter(sun['lon'][ind_time], sun['lat'][ind_time],
               c='red', s=20, edgecolors="black", linewidths=0.5)
    cbar.set_label('$Nm$F2 (m$^{-3}$)')
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    return


def PyIRI_plot_NmF1(F1, aUT, alon, alat, alon_2d, alat_2d, sun,
                    UT, plot_dir, plot_name='PyIRI_NmF1.png'):
    """Plot NmF1.

    Parameters
    ----------
    F1 : dict
        Dictionary output of IRI_monthly_mean_parameters.
    aUT : array-like
        Array of universal times in hours used in PyIRI.
    alon : array-like
        Flattened array of geo longitudes in degrees.
    alat : array-like
        Flattened array of geo latitudes in degrees.
    alon_2d : array-like
        2-D array of geo longitudes in degrees.
    alat_2d : array-like
        2-D array of geo latitudes in degrees.
    sun : dict
        Dictionary output of IRI_monthly_mean_parameters.
    UT : float
        UT time frame from array aUT to plot.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Name for the output figure, without directory
        (default='PyIRI_NmF1_min_max.png')

    """
    ind_time = np.where(aUT == UT)
    ind_grid = np.where(np.isfinite(alon))
    figname = os.path.join(plot_dir, plot_name)
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6, 3),
                           constrained_layout=True)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180 + 45, 90))
    plt.yticks(np.arange(-90, 90 + 45, 45))
    ax.set_facecolor('grey')
    ax.set_xlabel('Geo Lon (°)')
    ax.set_ylabel('Geo Lat (°)')
    z = np.reshape(F1['Nm'][ind_time, ind_grid], alon_2d.shape)
    mesh = ax.pcolormesh(alon_2d, alat_2d, z)
    cbar = fig.colorbar(mesh, ax=ax)
    ax.scatter(sun['lon'][ind_time], sun['lat'][ind_time],
               c='red', s=20, edgecolors="black", linewidths=0.5)
    cbar.set_label('$Nm$F1 (m$^{-3}$)')
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    return


def PyIRI_plot_foF1(F1, aUT, alon, alat, alon_2d, alat_2d, sun,
                    UT, plot_dir, plot_name='PyIRI_foF1.png'):
    """Plot foF1.

    Parameters
    ----------
    F1 : dict
        Dictionary output of IRI_monthly_mean_parameters.
    aUT : array-like
        Array of universal times in hours used in PyIRI.
    alon : array-like
        Flattened array of geo longitudes in degrees.
    alat : array-like
        Flattened array of geo latitudes in degrees.
    alon_2d : array-like
        2-D array of geo longitudes in degrees.
    alat_2d : array-like
        2-D array of geo latitudes in degrees.
    sun : dict
        Dictionary output of IRI_monthly_mean_parameters.
    UT : float
        UT time frame from array aUT to plot.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Name for the output figure, without directory
        (default='PyIRI_foF1_min_max.png')

    """
    ind_time = np.where(aUT == UT)
    ind_grid = np.where(np.isfinite(alon))
    figname = os.path.join(plot_dir, plot_name)
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 3),
                           constrained_layout=True)
    ax[0].set_facecolor('grey')
    ax[0].set_xlabel('Geo Lon (°)')
    ax[0].set_ylabel('Geo Lat (°)')
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180 + 45, 90))
    plt.yticks(np.arange(-90, 90 + 45, 45))
    abc = ['(a)', '(b)']
    ax[0].text(130, 70, abc[0], c='white')
    P = F1['P'][ind_time, ind_grid]
    foF1 = F1['fo'][ind_time, ind_grid]
    # --------------------------------
    z = np.reshape(P, alon_2d.shape)
    mesh = ax[0].pcolormesh(alon_2d, alat_2d, z)
    cbar0 = fig.colorbar(mesh, ax=ax[0])
    ax[0].scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red', s=20,
                  edgecolors="black", linewidths=0.5, zorder=2)
    cbar0.set_label('Probability')
    ax[1].set_facecolor('grey')
    ax[1].set_xlabel('Geo Lon (°)')
    ax[1].set_ylabel(' ')
    ax[1].text(130, 70, abc[1], c='white')
    ax[1].scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red', s=20,
                  edgecolors="black", linewidths=0.5, zorder=2)
    # --------------------------------
    z2 = np.reshape(foF1, alon_2d.shape)
    mesh = ax[1].pcolormesh(alon_2d, alat_2d, z2)
    cbar = fig.colorbar(mesh, ax=ax[1])
    cbar.set_label('$fo$F1 (MHz)')
    ax[0].title.set_text('Probability')
    ax[1].title.set_text('foF1')
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    return


def PyIRI_plot_1location_diurnal_par(F2, F1, E, Es, alon, alat,
                                     lon_plot, lat_plot, aUT, plot_dir,
                                     plot_name='PyIRI_diurnal.png'):
    """Plot diurnal parameters for one location.

    Parameters
    ----------
    F2 : dict
        Dictionary output of IRI_monthly_mean_parameters.
    F1 : dict
        Dictionary output of IRI_monthly_mean_parameters.
    E : dict
        Dictionary output of IRI_monthly_mean_parameters.
    Es : dict
        Dictionary output of IRI_monthly_mean_parameters.
    alon : array-like
        Flattened array of geo longitudes in degrees.
    alat : array-like
        Flattened array of geo latitudes in degrees.
    lon_plot : float
        Longitude location for EDP.
    lat_plot : array-like
        Latitude location for EDP.
    aUT : array-like
        Array of universal times in hours used in PyIRI.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Name for the output figure, without directory
        (default='PyIRI_foF1_min_max.png')

    """
    figname = os.path.join(plot_dir, plot_name)
    fig, ax = plt.subplots(4, 1, sharex=True, sharey=False, figsize=(4, 7),
                           constrained_layout=True)
    plt.xlim([0, 24])
    plt.xticks(np.arange(0, 24 + 4, 4))

    ind_grid = np.where((alon == lon_plot) & (alat == lat_plot))[0]

    # --------------------------------
    ax[0].set_facecolor('grey')
    ax[0].set_xlabel('UT (hours)')
    ax[0].set_ylabel('Peak Density (m$^{-3}$)')
    ax[0].plot(aUT, F2['Nm'][:, ind_grid], label='NmF2', c='red', zorder=2)
    ax[0].plot(aUT, F1['Nm'][:, ind_grid], label='NmF1', c='green', zorder=1)
    ax[0].plot(aUT, E['Nm'][:, ind_grid], label='NmE', c='yellow')
    ax[0].plot(aUT, Es['Nm'][:, ind_grid], label='NmEs', c='blue')
    ax[0].legend(loc='upper left', prop={'size': 6})
    # --------------------------------
    ax[1].set_facecolor('grey')
    ax[1].set_xlabel('UT (hours)')
    ax[1].set_ylabel('Peak Height (km)')
    ax[1].plot(aUT, F2['hm'][:, ind_grid], label='hmF2', c='red', zorder=2)
    ax[1].plot(aUT, F1['hm'][:, ind_grid], label='hmF1', c='green', zorder=1)
    ax[1].plot(aUT, E['hm'][:, ind_grid], label='hmE', c='yellow')
    ax[1].plot(aUT, Es['hm'][:, ind_grid], label='hmEs', c='blue')
    ax[1].legend(loc='upper left', prop={'size': 6})
    # --------------------------------
    ax[2].set_facecolor('grey')
    ax[2].set_xlabel('UT (hours)')
    ax[2].set_ylabel('Top Thickness (km)')
    ax[2].plot(aUT, F2['B_top'][:, ind_grid], label='B_top F2', c='red',
               zorder=2)
    ax[2].plot(aUT, E['B_top'][:, ind_grid], label='B_top E', c='yellow'
               , zorder=1)
    ax[2].plot(aUT, Es['B_top'][:, ind_grid], label='B_top Es', c='blue')
    ax[2].legend(loc='upper left', prop={'size': 6})
    # --------------------------------
    ax[3].set_facecolor('grey')
    ax[3].set_xlabel('UT (hours)')
    ax[3].set_ylabel('Bottom Thickness (km)')
    ax[3].plot(aUT, F2['B_bot'][:, ind_grid], label='B_bot F2', c='red',
               zorder=2)
    ax[3].plot(aUT, F1['B_bot'][:, ind_grid], label='B_bot F1', c='green',
               zorder=1)
    ax[3].plot(aUT, E['B_bot'][:, ind_grid], label='B_bot E', c='yellow')
    ax[3].plot(aUT, Es['B_top'][:, ind_grid], label='B_bot Es', c='blue')
    ax[3].legend(loc='upper left', prop={'size': 6})
    # --------------------------------
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    return


def PyIRI_plot_1location_diurnal_density(EDP, alon, alat, lon_plot, lat_plot,
                                         aalt, aUT, plot_dir,
                                         plot_name='PyIRI_EDP_diurnal.png'):
    """Plot diurnal parameters for one location.

    Parameters
    ----------
    EDP : array-like
        3-D electron density array output of IRI_density_1day
        with shape [N_T, N_V, N_H].
    alon : array-like
        Flattened array of geo longitudes in degrees.
    alat : array-like
        Flattened array of geo latitudes in degrees.
    lon_plot : float
        Longitude location for EDP.
    lat_plot : array-like
        Latitude location for EDP.
    aalt : array-like
        Flattened array of altitudes in km.
    aUT : array-like
        Array of universal times in hours used in PyIRI.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Name for the output figure, without directory
        (default='PyIRI_foF1_min_max.png')

    """
    figname = os.path.join(plot_dir, plot_name)
    fig, ax = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True)
    plt.xlim([0, 24])
    plt.xticks(np.arange(0, 24 + 4, 4))
    ax.set_facecolor('grey')
    ax.set_xlabel('UT (hours)')
    ax.set_ylabel('Altitude (km)')
    ind_grid = np.where((alon == lon_plot) & (alat == lat_plot))[0]
    z = np.transpose(np.reshape(EDP[:, :, ind_grid], (aUT.size, aalt.size)))
    mesh = ax.pcolormesh(aUT, aalt, z)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label('Electron Density (m$^{-3}$)')
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    return
