import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
#--------------------------------------------------------------------------------------------
dir='/Users/vmakarevich/Documents/Science_VF2/PyIRI/'
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
def PyIRI_plot_mag_dip_lat(mag, alon, alat, alon_2d, alat_2d):
    Figname=dir+'Fig/PyIRI_mag_dip_lat.pdf'
    fig, ax = plt.subplots(1, 1)
    ax.set_facecolor('grey')   
    ax.set_xlabel('Geo Lon ($^{\circ}$)')   
    ax.set_ylabel('Geo Lat ($^{\circ}$)')  
    z=np.reshape(mag['mag_dip_lat'], alon_2d.shape)

    levels=np.linspace(-90, 90, 40)
    levels_cb=np.linspace(-90, 90, 5)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180+45, 45))
    plt.yticks(np.arange(-90, 90+45, 45))
    contour=ax.contourf(alon_2d, alat_2d, z, levels=levels)
    for c in contour.collections:
        c.set_edgecolor("face")
    cbar=fig.colorbar(contour, ax=ax, ticks=levels_cb)
    cbar.set_label('Mag Dip Lat ($^{\circ}$)')
    plt.title('Alt = 300 km')
    plt.savefig(Figname)
    return()
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
def PyIRI_plot_inc(mag, alon, alat, alon_2d, alat_2d):
    Figname=dir+'Fig/PyIRI_inc.pdf'
    fig, ax = plt.subplots(1, 1)
    ax.set_facecolor('grey')   
    ax.set_xlabel('Geo Lon ($^{\circ}$)')   
    ax.set_ylabel('Geo Lat ($^{\circ}$)')  
    z=np.reshape(mag['inc'], alon_2d.shape)

    levels=np.linspace(-90, 90, 40)
    levels_cb=np.linspace(-90, 90, 5)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180+45, 45))
    plt.yticks(np.arange(-90, 90+45, 45))
    contour=ax.contourf(alon_2d, alat_2d, z, levels=levels)
    for c in contour.collections:
        c.set_edgecolor("face")
    cbar=fig.colorbar(contour, ax=ax, ticks=levels_cb)
    cbar.set_label('Inclination ($^{\circ}$)')
    plt.title('Alt = 300 km')
    plt.savefig(Figname)
    return()
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
def PyIRI_plot_modip(mag, alon, alat, alon_2d, alat_2d):
    Figname=dir+'Fig/PyIRI_modip.pdf'
    fig, ax = plt.subplots(1, 1)
    ax.set_facecolor('grey')   
    ax.set_xlabel('Geo Lon ($^{\circ}$)')   
    ax.set_ylabel('Geo Lat ($^{\circ}$)')  
    z=np.reshape(mag['modip'], alon_2d.shape)

    levels=np.linspace(-90, 90, 40)
    levels_cb=np.linspace(-90, 90, 5)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180+45, 45))
    plt.yticks(np.arange(-90, 90+45, 45))
    contour=ax.contourf(alon_2d, alat_2d, z, levels=levels)
    for c in contour.collections:
        c.set_edgecolor("face")
    cbar=fig.colorbar(contour, ax=ax, ticks=levels_cb)
    cbar.set_label('Modip ($^{\circ}$)')
    plt.title('Alt = 300 km')
    plt.savefig(Figname)
    return()
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
def PyIRI_plot_B_F1_bot_min_max(F1, aUT, alon, alat, alon_2d, alat_2d, sun, UT):
    ind_time=np.where(aUT == UT)
    ind_grid=np.where(np.isfinite(alon))
    
    Figname=dir+'Fig/PyIRI_B_F1_bot_min_max.pdf'
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 3), constrained_layout=True)

    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180+45, 90))
    plt.yticks(np.arange(-90, 90+45, 45))
    abc=['(a)','(b)']
    atitle=['Solar Min', 'Solar Max']
    
    levels=np.linspace(10, 100, 40)
    levels_cb=np.linspace(10, 100, 10)
    
    for isol in range(0,2): 
        ax[isol].set_facecolor('grey')           
        ax[isol].set_xlabel('Geo Lon ($^{\circ}$)')   
        ax[isol].set_ylabel('Geo Lat ($^{\circ}$)')  
        if isol == 1:
            ax[1].set_ylabel(' ')  
        z=np.reshape(F1['B_bot'][ind_time, ind_grid, isol], alon_2d.shape)
        contour=ax[isol].contourf(alon_2d, alat_2d, z, levels=levels)
        for c in contour.collections:
                    c.set_edgecolor("face")

        ax[isol].text(140, 70, abc[isol], c='white') 
        ax[isol].title.set_text(atitle[isol]) 
        ax[isol].scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red', s=20, edgecolors= "black", linewidths=0.5)  
    
    cbar=fig.colorbar(contour, ticks=levels_cb)
    cbar.set_label('$B^{F1}_{bot}$ (km)')
    plt.savefig(Figname, format='pdf', bbox_inches='tight')
    
    return()
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
def PyIRI_plot_B_F2_bot_min_max(F2, aUT, alon, alat, alon_2d, alat_2d, sun, UT):
    ind_time=np.where(aUT == UT)
    ind_grid=np.where(np.isfinite(alon))
    
    Figname=dir+'Fig/PyIRI_B_F2_bot_min_max.pdf'
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 3), constrained_layout=True)

    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180+45, 90))
    plt.yticks(np.arange(-90, 90+45, 45))
    abc=['(a)','(b)']
    atitle=['Solar Min', 'Solar Max']
    
    levels=np.linspace(10, 60, 40)
    levels_cb=np.linspace(10, 60, 6)
    
    for isol in range(0,2):
        
        ax[isol].set_facecolor('grey')           
        ax[isol].set_xlabel('Geo Lon ($^{\circ}$)')   
        ax[isol].set_ylabel('Geo Lat ($^{\circ}$)')  
        if isol == 1:
            ax[1].set_ylabel(' ')  
        z=np.reshape(F2['B_bot'][ind_time, ind_grid, isol], alon_2d.shape)
        contour=ax[isol].contourf(alon_2d, alat_2d, z, levels=levels)
        for c in contour.collections:
                    c.set_edgecolor("face")

        ax[isol].text(140, 70, abc[isol], c='white') 
        ax[isol].title.set_text(atitle[isol]) 
        ax[isol].scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red', s=20, edgecolors= "black", linewidths=0.5)  
    
    cbar=fig.colorbar(contour, ticks=levels_cb)
    cbar.set_label('$B^{F2}_{bot}$ (km)')
    plt.savefig(Figname, format='pdf', bbox_inches='tight')
    
    return()
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
def PyIRI_plot_B_F2_top_min_max(F2, aUT, alon, alat, alon_2d, alat_2d, sun, UT):
    ind_time=np.where(aUT == UT)
    ind_grid=np.where(np.isfinite(alon))
    
    Figname=dir+'Fig/PyIRI_B_F2_top_min_max.pdf'
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 3), constrained_layout=True)

    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180+45, 90))
    plt.yticks(np.arange(-90, 90+45, 45))
    abc=['(a)','(b)']
    atitle=['Solar Min', 'Solar Max']
    
    levels=np.linspace(20, 55, 40)
    levels_cb=np.linspace(20, 55, 8)
    
    for isol in range(0,2):
        
        ax[isol].set_facecolor('grey')           
        ax[isol].set_xlabel('Geo Lon ($^{\circ}$)')   
        ax[isol].set_ylabel('Geo Lat ($^{\circ}$)')  
        if isol == 1:
            ax[1].set_ylabel(' ')  
        z=np.reshape(F2['B_top'][ind_time, ind_grid, isol], alon_2d.shape)
        contour=ax[isol].contourf(alon_2d, alat_2d, z, levels=levels)
        for c in contour.collections:
                    c.set_edgecolor("face")

        ax[isol].text(140, 70, abc[isol], c='white') 
        ax[isol].title.set_text(atitle[isol]) 
        ax[isol].scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red', s=20, edgecolors= "black", linewidths=0.5)  
    
    cbar=fig.colorbar(contour, ticks=levels_cb)
    cbar.set_label('$B^{F2}_{top}$ (km)')
    plt.savefig(Figname, format='pdf', bbox_inches='tight')
    
    return()
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
def PyIRI_plot_M3000_min_max(F2, aUT, alon, alat, alon_2d, alat_2d, sun, UT):
    ind_time=np.where(aUT == UT)
    ind_grid=np.where(np.isfinite(alon))
    
    Figname=dir+'Fig/PyIRI_M3000_min_max.pdf'
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 3), constrained_layout=True)

    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180+45, 90))
    plt.yticks(np.arange(-90, 90+45, 45))
    abc=['(a)','(b)']
    atitle=['Solar Min', 'Solar Max']
    
    levels=np.linspace(2.2, 3.8, 40)
    levels_cb=np.linspace(2.2, 3.8, 9)
    
    for isol in range(0,2):
        
        ax[isol].set_facecolor('grey')           
        ax[isol].set_xlabel('Geo Lon ($^{\circ}$)')   
        ax[isol].set_ylabel('Geo Lat ($^{\circ}$)')  
        if isol == 1:
            ax[1].set_ylabel(' ')  
        z=np.reshape(F2['M3000'][ind_time, ind_grid, isol], alon_2d.shape)
        contour=ax[isol].contourf(alon_2d, alat_2d, z, levels=levels)
        for c in contour.collections:
                    c.set_edgecolor("face")

        ax[isol].text(140, 70, abc[isol], c='white') 
        ax[isol].title.set_text(atitle[isol]) 
        ax[isol].scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red', s=20, edgecolors= "black", linewidths=0.5)  
    
    cbar=fig.colorbar(contour, ticks=levels_cb)
    cbar.set_label('MUF(3000)F2/$fo$F2')
    plt.savefig(Figname, format='pdf', bbox_inches='tight')
    
    return()
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
def PyIRI_plot_hmF2_min_max(F2, aUT, alon, alat, alon_2d, alat_2d, sun, UT):
    ind_time=np.where(aUT == UT)
    ind_grid=np.where(np.isfinite(alon))
    
    Figname=dir+'Fig/PyIRI_hmF2_min_max.pdf'
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 3), constrained_layout=True)

    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180+45, 90))
    plt.yticks(np.arange(-90, 90+45, 45))
    abc=['(a)','(b)']
    atitle=['Solar Min', 'Solar Max']
    
    levels=np.linspace(200, 500, 40)
    levels_cb=np.linspace(200, 500, 7)
    
    for isol in range(0,2):
        
        ax[isol].set_facecolor('grey')           
        ax[isol].set_xlabel('Geo Lon ($^{\circ}$)')   
        ax[isol].set_ylabel('Geo Lat ($^{\circ}$)')  
        if isol == 1:
            ax[1].set_ylabel(' ')  
        z=np.reshape(F2['hm'][ind_time, ind_grid, isol], alon_2d.shape)
        contour=ax[isol].contourf(alon_2d, alat_2d, z, levels=levels)
        for c in contour.collections:
                    c.set_edgecolor("face")

        ax[isol].text(140, 70, abc[isol], c='white') 
        ax[isol].title.set_text(atitle[isol]) 
        ax[isol].scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red', s=20, edgecolors= "black", linewidths=0.5)  
        
    
    cbar=fig.colorbar(contour, ticks=levels_cb)
    cbar.set_label('$hm$F2 (km)')
    plt.savefig(Figname, format='pdf', bbox_inches='tight')
    
    return()
#-------------------------------------------------------------------------------------------- 
#--------------------------------------------------------------------------------------------
def PyIRI_plot_hmF1_min_max(F1, aUT, alon, alat, alon_2d, alat_2d, sun, UT):
    ind_time=np.where(aUT == UT)
    ind_grid=np.where(np.isfinite(alon))
    
    Figname=dir+'Fig/PyIRI_hmF1_min_max.pdf'
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 3), constrained_layout=True)

    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180+45, 90))
    plt.yticks(np.arange(-90, 90+45, 45))
    abc=['(a)','(b)']
    atitle=['Solar Min', 'Solar Max']
    
    levels=np.linspace(100, 350, 40)
    levels_cb=np.linspace(100, 350, 6)
    
    for isol in range(0,2):
        
        ax[isol].set_facecolor('grey')           
        ax[isol].set_xlabel('Geo Lon ($^{\circ}$)')   
        ax[isol].set_ylabel('Geo Lat ($^{\circ}$)')  
        if isol == 1:
            ax[1].set_ylabel(' ')  
        z=np.reshape(F1['hm'][ind_time, ind_grid, isol], alon_2d.shape)
        contour=ax[isol].contourf(alon_2d, alat_2d, z, levels=levels)
        for c in contour.collections:
                    c.set_edgecolor("face")

        ax[isol].text(140, 70, abc[isol], c='white') 
        ax[isol].title.set_text(atitle[isol]) 
        ax[isol].scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red', s=20, edgecolors= "black", linewidths=0.5)  
        
    cbar=fig.colorbar(contour, ticks=levels_cb)
    cbar.set_label('$hm$F1 (km)')
    plt.savefig(Figname, format='pdf', bbox_inches='tight')
    
    return()
#-------------------------------------------------------------------------------------------- 
#--------------------------------------------------------------------------------------------
def PyIRI_plot_foEs_min_max(Es, aUT, alon, alat, alon_2d, alat_2d, sun, UT):
    ind_time=np.where(aUT == UT)
    ind_grid=np.where(np.isfinite(alon))
    
    Figname=dir+'Fig/PyIRI_foEs_min_max.pdf'
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 3), constrained_layout=True)

    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180+45, 90))
    plt.yticks(np.arange(-90, 90+45, 45))
    abc=['(a)','(b)']
    atitle=['Solar Min', 'Solar Max']
    
    levels=np.linspace(0, 12, 40)
    levels_cb=np.linspace(0, 12, 13)
    
    for isol in range(0,2):
        
        ax[isol].set_facecolor('grey')           
        ax[isol].set_xlabel('Geo Lon ($^{\circ}$)')   
        ax[isol].set_ylabel('Geo Lat ($^{\circ}$)')  
        if isol == 1:
            ax[1].set_ylabel(' ')  
        z=np.reshape(Es['fo'][ind_time, ind_grid, isol], alon_2d.shape)
        contour=ax[isol].contourf(alon_2d, alat_2d, z, levels=levels)
        for c in contour.collections:
                    c.set_edgecolor("face")

        ax[isol].text(140, 70, abc[isol], c='white') 
        ax[isol].title.set_text(atitle[isol])
    
        ax[isol].scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red', s=20, edgecolors= "black", linewidths=0.5)  
    
    cbar=fig.colorbar(contour, ticks=levels_cb)
    cbar.set_label('$fo$Es (MHz)')
    plt.savefig(Figname, format='pdf', bbox_inches='tight')
    
    return()
#-------------------------------------------------------------------------------------------- 
#--------------------------------------------------------------------------------------------
def PyIRI_plot_foE_min_max(E, aUT, alon, alat, alon_2d, alat_2d, sun, UT):
    ind_time=np.where(aUT == UT)
    ind_grid=np.where(np.isfinite(alon))
    
    Figname=dir+'Fig/PyIRI_foE_min_max.pdf'
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 3), constrained_layout=True)

    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180+45, 90))
    plt.yticks(np.arange(-90, 90+45, 45))
    abc=['(a)','(b)']
    atitle=['Solar Min', 'Solar Max']
    
    levels=np.linspace(0, 4, 40)
    levels_cb=np.linspace(0, 4, 5)
    
    for isol in range(0,2):
        
        ax[isol].set_facecolor('grey')           
        ax[isol].set_xlabel('Geo Lon ($^{\circ}$)')   
        ax[isol].set_ylabel('Geo Lat ($^{\circ}$)')  
        if isol == 1:
            ax[1].set_ylabel(' ')  
        z=np.reshape(E['fo'][ind_time, ind_grid, isol], alon_2d.shape)
        contour=ax[isol].contourf(alon_2d, alat_2d, z, levels=levels)
        for c in contour.collections:
                    c.set_edgecolor("face")

        ax[isol].text(140, 70, abc[isol], c='white') 
        ax[isol].title.set_text(atitle[isol])
    
        ax[isol].scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red', s=20, edgecolors= "black", linewidths=0.5)  
    
    cbar=fig.colorbar(contour, ticks=levels_cb)
    cbar.set_label('$fo$E (MHz)')
    plt.savefig(Figname, format='pdf', bbox_inches='tight')
    
    return()
#-------------------------------------------------------------------------------------------- 
#--------------------------------------------------------------------------------------------
def PyIRI_plot_foF2_min_max(F2, aUT, alon, alat, alon_2d, alat_2d, sun, UT):
    ind_time=np.where(aUT == UT)
    ind_grid=np.where(np.isfinite(alon))
    
    Figname=dir+'Fig/PyIRI_foF2_min_max.pdf'
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 3), constrained_layout=True)

    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180+45, 90))
    plt.yticks(np.arange(-90, 90+45, 45))
    abc=['(a)','(b)']
    atitle=['Solar Min', 'Solar Max']
    
    levels=np.linspace(0, 16, 40)
    levels_cb=np.linspace(0, 16, 5)
    
    for isol in range(0,2):
        
        ax[isol].set_facecolor('grey')           
        ax[isol].set_xlabel('Geo Lon ($^{\circ}$)')   
        ax[isol].set_ylabel('Geo Lat ($^{\circ}$)')  
        if isol == 1:
            ax[1].set_ylabel(' ')  
        z=np.reshape(F2['fo'][ind_time, ind_grid, isol], alon_2d.shape)
        contour=ax[isol].contourf(alon_2d, alat_2d, z, levels=levels)
        for c in contour.collections:
                    c.set_edgecolor("face")

        ax[isol].text(140, 70, abc[isol], c='white') 
        ax[isol].title.set_text(atitle[isol])
        ax[isol].scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red', s=20, edgecolors= "black", linewidths=0.5)  
    
    cbar=fig.colorbar(contour, ticks=levels_cb)
    cbar.set_label('$fo$F2 (MHz)')
    plt.savefig(Figname, format='pdf', bbox_inches='tight')
    
    return()
#-------------------------------------------------------------------------------------------- 
def PyIRI_plot_NmF2_min_max(F2, aUT, alon, alat, alon_2d, alat_2d, sun, UT):
    ind_time=np.where(aUT == UT)
    ind_grid=np.where(np.isfinite(alon))
    
    Figname=dir+'Fig/PyIRI_NmF2_min_max.pdf'
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 3), constrained_layout=True)

    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180+45, 90))
    plt.yticks(np.arange(-90, 90+45, 45))
    abc=['(a)','(b)']
    atitle=['Solar Min', 'Solar Max']
    
    levels=np.linspace(0, 4e12, 40)
    levels_cb=np.linspace(0, 4e12, 5)
    
    for isol in range(0,2):
        
        ax[isol].set_facecolor('grey')           
        ax[isol].set_xlabel('Geo Lon ($^{\circ}$)')   
        ax[isol].set_ylabel('Geo Lat ($^{\circ}$)')  
        if isol == 1:
            ax[1].set_ylabel(' ')  
        z=np.reshape(F2['Nm'][ind_time, ind_grid, isol], alon_2d.shape)
        contour=ax[isol].contourf(alon_2d, alat_2d, z, levels=levels)
        for c in contour.collections:
                    c.set_edgecolor("face")

        ax[isol].text(140, 70, abc[isol], c='white') 
        ax[isol].title.set_text(atitle[isol])
        ax[isol].scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red', s=20, edgecolors= "black", linewidths=0.5)  
    
    cbar=fig.colorbar(contour, ticks=levels_cb)
    cbar.set_label('$Nm$F2 (m$^{-3}$)')
    plt.savefig(Figname, format='pdf', bbox_inches='tight')
    
    return()
#-------------------------------------------------------------------------------------------- 
#--------------------------------------------------------------------------------------------
def PyIRI_plot_foF1_min_max(F1, aUT, alon, alat, alon_2d, alat_2d, sun, UT):
    ind_time=np.where(aUT == UT)
    ind_grid=np.where(np.isfinite(alon))
    
    Figname=dir+'Fig/PyIRI_foF1_min_max.pdf'
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 3), constrained_layout=True)
    ax[0].set_facecolor('grey')   
    ax[0].set_xlabel('Geo Lon ($^{\circ}$)')   
    ax[0].set_ylabel('Geo Lat ($^{\circ}$)')  
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180+45, 90))
    plt.yticks(np.arange(-90, 90+45, 45))
    abc=['(a)','(b)','(c)']
    ax[0].text(130, 70, abc[0], c='white')  
    levels0=np.linspace(0, 1, 40)
    levels_cb0=np.linspace(0, 1, 5)
    P=F1['P'][ind_time, ind_grid, 0]
    foF1_min=F1['fo'][ind_time, ind_grid, 0]
    foF1_max=F1['fo'][ind_time, ind_grid, 1]    
    #--------------------------------
    z=np.reshape(P, alon_2d.shape)
    contour0=ax[0].contourf(alon_2d, alat_2d, z, levels=levels0)
    for c in contour0.collections:
                c.set_edgecolor("face")
    ax[0].scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red', s=20, edgecolors= "black", linewidths=0.5, zorder=2) 
    cbar0 = plt.colorbar(contour0, ticks=levels_cb0)
    cbar0.set_label('Probability')
    levels=np.linspace(0, 6, 40)
    levels_cb=np.linspace(0, 6, 6)
    ax[1].set_facecolor('grey')   
    ax[1].set_xlabel('Geo Lon ($^{\circ}$)')   
    ax[1].set_ylabel(' ')  
    ax[1].text(130, 70, abc[1], c='white')  
    levels1=np.linspace(3, 6, 40)
    levels_cb1=np.linspace(3, 6, 4)
    ax[1].scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red', s=20, edgecolors= "black", linewidths=0.5, zorder=2) 
    #--------------------------------
    z=np.reshape(foF1_min, alon_2d.shape)
    contour1=ax[1].contourf(alon_2d, alat_2d, z, levels=levels1)
    for c in contour1.collections:
                c.set_edgecolor("face")
    cbar1 = plt.colorbar(contour1, ticks=levels_cb1)
    cbar1.set_label('$fo$F1 (MHz)')
    ax[2].set_facecolor('grey')   
    ax[2].set_xlabel('Geo Lon ($^{\circ}$)')   
    ax[2].set_ylabel(' ')  
    ax[2].text(130, 70, abc[2], c='white') 
    ax[2].scatter(sun['lon'][ind_time], sun['lat'][ind_time], c='red', s=20, edgecolors= "black", linewidths=0.5, zorder=2)  
    #--------------------------------
    z2=np.reshape(foF1_max, alon_2d.shape)
    contour2=ax[2].contourf(alon_2d, alat_2d, z2, levels=levels1)
    for c in contour2.collections:
                c.set_edgecolor("face")
    ax[0].title.set_text('Probability')
    ax[1].title.set_text('Solar Min')
    ax[2].title.set_text('Solar Max')

    plt.savefig(Figname, format='pdf', bbox_inches='tight')
    plt.show()
    return()
#-------------------------------------------------------------------------------------------- 
#--------------------------------------------------------------------------------------------
def PyIRI_3D_hm_limits(F2,F1,E, aUT, alon, alat, alon_2d, alat_2d, UT):
    ind_time=np.where(aUT == UT)
    ind_grid=np.where(np.isfinite(alon))
    
    ind_max=ind_time, ind_grid, 1
    
    hmF2_2d=np.reshape(F2['hm'][ind_max], alon_2d.shape)
    hmF1_2d=np.reshape(F1['hm'][ind_max], alon_2d.shape)
    hmE_2d=np.reshape(E['hm'][ind_max], alon_2d.shape)
    
    Figname=dir+'Fig/PyIRI_3D_hm_limits.pdf'
    
    fig = plt.figure(figsize = (3, 3))
    ax = plt.axes(projection ="3d")
    fontsize=5 
    ax.set_xlabel('Geo Lon ($^{\circ}$)', fontsize=fontsize, labelpad=-10)
    ax.set_ylabel('Geo Lat ($^{\circ}$)', fontsize=fontsize, labelpad=-10)
    ax.set_zlabel('Altitude (km)', fontsize=fontsize, labelpad=-10)
    plt.xticks(np.arange(-180, 180+45, 90))
    plt.yticks(np.arange(-90, 90+45, 45))
    ax.set_zlim(0, 500)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.tick_params(axis='z', labelsize=fontsize)

    ax.plot_surface(alon_2d, alat_2d, hmF2_2d, alpha=0.5, color='red', label='$hm$F2')
    ax.plot_surface(alon_2d, alat_2d, hmF1_2d, alpha=0.8, color='orange', label='$hm$F1')
    ax.plot_surface(alon_2d, alat_2d, hmE_2d, alpha=0.8, color='purple', label='$hm$E')
    ax.xaxis._axinfo["grid"]['linewidth'] =0.3
    ax.yaxis._axinfo["grid"]['linewidth'] =0.3
    ax.zaxis._axinfo["grid"]['linewidth'] =0.3
    ax.view_init(5, 150)
    ax.text(-90, -90, 350,  '$hm$F2', size=fontsize, zorder=1, color='red') 
    ax.text(-40, -40, 200,  '$hm$F1', size=fontsize, zorder=1, color='orange') 
    ax.text(-90, -90, 120,  '$hm$E', size=fontsize, zorder=1, color='purple') 
    ax.tick_params(axis='x', which='major', pad=-3)
    ax.tick_params(axis='y', which='major', pad=-3)
    ax.tick_params(axis='z', which='major', pad=-3)
    plt.savefig(Figname, format='pdf', bbox_inches='tight')
    
#-------------------------------------------------------------------------------------------- 
#--------------------------------------------------------------------------------------------
def PyIRI_EDP_sample(EDP, aUT, alon, alat, alon_2d, alat_2d, aalt, UT):   
    Figname=dir+'Fig/PyIRI_EDP_sample.pdf'
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(3, 3), constrained_layout=True)
    plt.xlim([0, 2.5e12])
    plt.ylim([0, 700])

    ax.set_xlabel('Electron Density (m$^{-3}$)')
    ax.set_ylabel('Altitude (km)')
    ax.set_facecolor("grey")

    
    ind_grid=np.where((alon ==0) & (alat ==0))
    ind_time=np.where(aUT == UT)
    ind_vert=np.where(aalt >= 0)
    
    ind_min=0, ind_time, ind_vert, ind_grid    
    x=np.reshape(EDP[ind_min], aalt.shape)
    ax.plot(x, aalt, c='black', label='Sol min')
    
    ind_max=1, ind_time, ind_vert, ind_grid    
    x=np.reshape(EDP[ind_max], aalt.shape)
    ax.plot(x, aalt, c='white', label='Sol max')
    
    leg = ax.legend(loc='upper right', prop={'size': 8})
    return()

#-------------------------------------------------------------------------------------------- 
#--------------------------------------------------------------------------------------------
def PyIRI_EDP_sample_1day(EDP, aUT, alon, alat, alon_2d, alat_2d, aalt, UT):   
    Figname=dir+'Fig/PyIRI_EDP_sample.pdf'
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(3, 3), constrained_layout=True)
    plt.xlim([0, 2.5e12])
    plt.ylim([0, 700])

    ax.set_xlabel('Electron Density (m$^{-3}$)')
    ax.set_ylabel('Altitude (km)')
    ax.set_facecolor("grey")

    
    ind_grid=np.where((alon ==0) & (alat ==0))
    ind_time=np.where(aUT == UT)
    ind_vert=np.where(aalt >= 0)
    
    ind=ind_time, ind_vert, ind_grid    
    
    x=np.reshape(EDP[ind], aalt.shape)
    ax.plot(x, aalt, c='black')
    print('NmF2=', np.max(x))
    
    return()
