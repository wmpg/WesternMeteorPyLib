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

from Utils.Dcriteria import calcDN, calcDV, calcDVuncert


def calculateDistanceProfile(inputs, metric):
    """ Sort the given points by the given metric function and return the calculated distance profile. """


    profile = []

    # Check if any points were given
    if not inputs:
        return profile


    # List for points that were already processed
    processed = []

    # Pointer to current point
    current_index = 0

    while True:

        min_dist = np.inf
        nn_index = 0

        # Find the nearest neighbour to the current point
        for i, point in enumerate(inputs):

            # Check that the point was not processed
            if i not in processed:

                # Calculate the distance from the current point to the given point
                dist = metric(*(inputs[current_index] + point))

                # Check if this distance is smaller than the smallest one before
                if dist < min_dist:
                    min_dist = dist
                    nn_index = i



        current_index = nn_index
        processed.append(current_index)

        profile.append(min_dist)


        if len(processed) == len(inputs):
            break


    return profile





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




def plotRadiants(pickle_trajs, plot_type='geocentric', ra_cent=None, dec_cent=None, radius=1, plt_handle=None, 
    label=None, plot_stddev=True, **kwargs):
    """ Plots geocentric radiants of the given pickle files. 

    Arguments:
        pickle_trajs: [list] A list of trajectory objects loaded from .pickle files.

    Keyword arguments:
        plot_type: [str] Type of radiants to plot.
            - 'geocentric' - RA_g, Dec_g, Vg plot
            - 'heliocentric ecliptic' - Lh, Bg, Vh plot

        ra_cent: [float] Right ascension used for selecing only radiants in given radius on the sky (degrees).
        dec_cent: [float] Declination used for selecing only radiants in given radius on the sky (degrees).
        radius: [float] Radius for selecting radiants centred on ra_cent, dec_cent (degrees).

        plt_handle: [plt object] Matplotlib plt handle (e.g. plt variable when doing plt.plot(...)).
        label: [str] Label for the legend, used only when plot_stddev=True
        plot_stddev: [bool] Add standard deviation in the legend label. True by default.

    """

    ra_list = []
    dec_list = []
    vg_list = []

    sol_list = []

    lh_list = []
    lh_std_list = []
    bh_list = []
    bh_std_list = []
    vh_list = []
    vh_std_list = []

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

        sol_list.append(traj.orbit.la_sun)

        lh_list.append(traj.orbit.L_h)
        bh_list.append(traj.orbit.B_h)
        vh_list.append(traj.orbit.v_h)
        

        if traj.uncertanties is not None:
            lh_std_list.append(traj.uncertanties.L_h)
            bh_std_list.append(traj.uncertanties.B_h)
            vh_std_list.append(traj.uncertanties.v_h)


    ra_list = np.array(ra_list)
    dec_list = np.array(dec_list)
    vg_list = np.array(vg_list)

    sol_list = np.array(sol_list)

    lh_list = np.array(lh_list)
    lh_std_list = np.array(lh_std_list)
    bh_list = np.array(bh_list)
    bh_std_list = np.array(bh_std_list)
    vh_list = np.array(vh_list)
    vh_std_list = np.array(vh_std_list)


    # Choose the appropriate coordinates for plotting
    if plot_type == 'geocentric':
        x_list = ra_list
        y_list = dec_list
        z_list = vg_list/1000

        # Create inputs for calculating the distance profile
        distance_input = []
        for ra, dec, sol, vg in zip(ra_list, dec_list, sol_list, vg_list):
            distance_input.append([ra, dec, sol, vg/1000])

        # Calculate the distance profile
        dist_profile = calculateDistanceProfile(distance_input, calcDN)


    elif plot_type == 'heliocentric ecliptic':
        x_list = lh_list
        y_list = bh_list
        z_list = vh_list/1000

        # Create inputs for calculating the distance profile
        distance_input = []

        if traj.uncertanties is not None:
            

            for Lh, Lh_std, Bh, Bh_std, sol, vh, vh_std in zip(lh_list, lh_std_list, bh_list, bh_std_list, sol_list, vh_list, vh_std_list):
                distance_input.append([Lh, Lh_std, Bh, Bh_std, sol, vh/1000, vh_std/1000])

            # Calculate the distance profile
            dist_profile = calculateDistanceProfile(distance_input, calcDVuncert)
    
        else:

            for Lh, Bh, sol, vh in zip(lh_list, bh_list, sol_list, vh_list):
                distance_input.append([Lh, Bh, sol, vh/1000])

            # Calculate the distance profile
            dist_profile = calculateDistanceProfile(distance_input, calcDV)



    print(np.c_[np.degrees(x_list), np.degrees(y_list), z_list])


    if plt_handle is None:
        plt_handle = CelestialPlot(x_list, y_list, projection='stere', bgcolor='k')


    if plot_stddev:

        if label is None:
            label = ''

        ra_stddev = np.degrees(scipy.stats.circstd(x_list))
        dec_stddev = np.degrees(np.std(y_list))

        label += "{:d} orbits, $\sigma_{{RA}}$ = {:.2f}$\degree$".format(len(x_list), ra_stddev)
        label += ", "
        label += "$\sigma_{{Dec}}$ = {:.2f}$\degree$".format(dec_stddev)


    plt_handle.scatter(x_list, y_list, c=z_list, label=label, **kwargs)

 
    return plt_handle, dist_profile




if __name__ == "__main__":


    #dir_path = "../DenisGEMcases/"
    #dir_path = "../DenisGEMcases_5_sigma"
    #dir_path = "../Romulan2012Geminids"
    #dir_path = "../SimulatedMeteors/CAMO/PER"
    dir_path = "../SimulatedMeteors/CAMO/2011Draconids"



    # Load trajectory objects from Monte Carlo pickle files
    traj_pickles_mc = collectTrajPickles(dir_path, traj_type='mc')

    # Load trajectory objects from ordinary line of sight pickle solutions
    traj_pickles_los = collectTrajPickles(dir_path, traj_type='original')

    # Load Gural trajectory objects
    traj_pickles_gural = collectTrajPickles(dir_path, traj_type='gural')


    # # Coordinates of the centre (GEM)
    # ra_cent = 113.0
    # dec_cent = 32.5
    # radius = 5.0


    # # Coordinates of the centre (PER)
    # ra_cent = 48.2
    # dec_cent = 58.1
    # radius = 15.0

    # Coordinates of the centre (DRA)
    ra_cent = 263.387
    dec_cent = 55.9181
    radius = 15.0

    # ra_cent = None
    # dec_cent = None
    # radius = 1


    plot_type = 'geocentric'
    #plot_type = 'heliocentric ecliptic'

    # Plot geocentric radiants of Line of Sight solutions
    m, los_profile = plotRadiants(traj_pickles_los, plot_type=plot_type, ra_cent=ra_cent, dec_cent=dec_cent, \
        radius=radius, label='LoS:', s=10, marker='s')

    # Plot geocentric radiants of Monte Carlo solutions
    _, mc_profile = plotRadiants(traj_pickles_mc, plot_type=plot_type, ra_cent=ra_cent, dec_cent=dec_cent, \
        radius=radius, label='MC:', plt_handle=m, marker='+')

    # Plot gural geocentric radiants
    _, gural_profile = plotRadiants(traj_pickles_gural, plot_type=plot_type, ra_cent=ra_cent, \
        dec_cent=dec_cent, radius=radius, label='Gural:', plt_handle=m, marker='x', s=15)
    
    
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

    if plot_type == 'geocentric':
        colorbar_label = '$V_g (km/s)$'

    elif plot_type == 'heliocentric ecliptic':
        colorbar_label = '$V_h (km/s)$'
    

    m.colorbar(label=colorbar_label)
    plt.legend(loc='upper right')

    # plt.tight_layout()
    #plt.savefig('CAMS_GEM_solver_comparison_5_sigma.png', dpi=300)

    plt.show()


    # Plot distance profiles
    plt.plot(los_profile, label='LoS')
    plt.plot(mc_profile, label='MC')
    plt.plot(gural_profile, label='Gural')

    plt.legend()
    plt.show()
