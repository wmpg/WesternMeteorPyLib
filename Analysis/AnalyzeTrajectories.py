""" Collect trajectory pickle files and analyze them by plotting desired parameters, exporting parameters in 
a summary file, etc. """

from __future__ import print_function, absolute_import, division

import os


import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

from Utils.PlotCelestial import CelestialPlot
from Utils.Pickling import loadPickle
from Utils.OSTools import listDirRecursive
from Utils.Math import angleBetweenSphericalCoords



def collectTrajPickles(dir_path, traj_type='original'):
    """ Collect all trajectory .pickle files in the given directory and load them to memory. """

    # Get all files in the given directory structure
    dir_files = listDirRecursive(dir_path)

    # Select only pickle files
    pickle_files = [file_path for file_path in dir_files if '.pickle' in os.path.split(file_path)[1]]

    # Select only Monte Carlo pickle files if monte_carlo is True
    if traj_type == 'mc':
        pickle_files = [pickle_f for pickle_f in pickle_files if '_mc_trajectory.pickle' in pickle_f]

    # Select gural trajectory
    elif traj_type == 'gural':
        pickle_files = [pickle_f for pickle_f in pickle_files if '_gural_trajectory.pickle' in pickle_f]

    # Select non-Monte Carlo pickle files
    else:
        pickle_files = [pickle_f for pickle_f in pickle_files if ('trajectory.pickle' in pickle_f) \
            and not ('_mc' in pickle_f) and not ('_gural' in pickle_f)]


    # Load pickle files to memory
    pickle_trajs = [loadPickle(*os.path.split(pickle_f)) for pickle_f in pickle_files]

    return pickle_trajs




def plotGeocentricRadiants(pickle_trajs, ra_cent=None, dec_cent=None, radius=1, plt_handle=None, label=None, 
    plot_stddev=True, **kwargs):
    """ Plots geocentric radiants of the given pickle files. """

    ra_list = []
    dec_list = []
    vg_list = []

    for traj in pickle_trajs:

        # Don't take trajectories where the radiant is not calculated
        if traj.orbit.ra_g is None:
            continue


        # Check if the coordinates are within the given radius (if such central coordinates are given at all)
        if ra_cent is not None:

            # Calculate the angle between the centre RA/Dec and the given point. Skip the point if it is 
            # outside the given radius
            if angleBetweenSphericalCoords(np.radians(dec_cent), np.radians(ra_cent), traj.orbit.dec_g, \
                traj.orbit.ra_g) > np.radians(radius):
                continue

        ra_list.append(traj.orbit.ra_g)
        dec_list.append(traj.orbit.dec_g)
        vg_list.append(traj.orbit.v_g)


    ra_list = np.array(ra_list)
    dec_list = np.array(dec_list)
    vg_list = np.array(vg_list)

    print(np.c_[np.degrees(ra_list), np.degrees(dec_list), vg_list/1000])


    if plt_handle is None:
        plt_handle = CelestialPlot(ra_list, dec_list, projection='stere', bgcolor='w')


    if plot_stddev:

        if label is None:
            label = ''

        ra_stddev = np.degrees(scipy.stats.circstd(ra_list))
        dec_stddev = np.degrees(np.std(dec_list))

        label += " $\sigma_{{RA}}$ = {:.2f}$\degree$".format(ra_stddev)
        label += ", "
        label += "$\sigma_{{Dec}}$ = {:.2f}$\degree$".format(dec_stddev)


    plt_handle.scatter(ra_list, dec_list, c=vg_list/1000, label=label, **kwargs)


    return plt_handle




if __name__ == "__main__":


    #dir_path = "../DenisGEMcases/"
    dir_path = "../DenisGEMcases_5_sigma"
    #dir_path = "../Romulan2012Geminids"



    # Load trajectory objects from Monte Carlo pickle files
    traj_pickles_mc = collectTrajPickles(dir_path, traj_type='mc')

    # Load trajectory objects from ordinary line of sight pickle solutions
    traj_pickles_los = collectTrajPickles(dir_path, traj_type='original')

    # Load Gural trajectory objects
    traj_pickles_gural = collectTrajPickles(dir_path, traj_type='gural')


    # # Coordinates of the centre
    # ra_cent = 113.0
    # dec_cent = 32.5
    # radius = 4.5

    ra_cent = None
    dec_cent = None
    radius = 1


    # Plot geocentric radiants of Line of Sight solutions
    m = plotGeocentricRadiants(traj_pickles_los, ra_cent=ra_cent, dec_cent=dec_cent, radius=radius, label='LoS:', s=10, marker='s')

    # Plot geocentric radiants of Monte Carlo solutions
    plotGeocentricRadiants(traj_pickles_mc, ra_cent=ra_cent, dec_cent=dec_cent, radius=radius, label='MC:', plt_handle=m, marker='+')

    # Plot gural geocentric radiants
    plotGeocentricRadiants(traj_pickles_gural, ra_cent=ra_cent, dec_cent=dec_cent, radius=radius, label='Gural:', plt_handle=m, marker='o', s=10)
    
    
    # # Gural solver results
    # ra_list = np.radians(np.array([113.339, 112.946, 112.946, 113.830, 113.904, 113.046]))
    # dec_list = np.radians(np.array([32.680, 32.570, 32.464, 32.460, 33.294, 33.618]))
    # vg_list = np.array([32.644, 33.477, 34.854, 33.026, 33.803, 34.829])

    # ra_stddev = np.degrees(scipy.stats.circstd(ra_list))
    # dec_stddev = np.degrees(np.std(dec_list))

    # label = 'MPF:'

    # label += " $\sigma_{{RA}}$ = {:.2f}$\degree$".format(ra_stddev)
    # label += ", "
    # label += "$\sigma_{{Dec}}$ = {:.2f}$\degree$".format(dec_stddev)

    # m.scatter(ra_list, dec_list, c=vg_list, label=label, marker='o', s=10)


    m.colorbar(label='$V_g (km/s)$')
    plt.legend(loc='upper right')

    plt.tight_layout()
    #plt.savefig('CAMS_GEM_solver_comparison_5_sigma.png', dpi=300)

    plt.show()