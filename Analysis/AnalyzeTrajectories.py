""" Collect trajectory pickle files and analyze them by plotting desired parameters, exporting parameters in 
a summary file, etc. """

from __future__ import print_function, absolute_import, division

import sys
import os


import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

from TrajSim.ShowerSim import SimMeteor, AblationModelVelocity
from Utils.PlotCelestial import CelestialPlot
from Utils.Pickling import loadPickle
from Utils.OSTools import listDirRecursive
from Utils.Math import angleBetweenSphericalCoords, RMSD

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





def collectTrajPickles(dir_path, traj_type='original', unique=False):
    """ Recursively collect all trajectory .pickle files in the given directory and load them to memory. 
    
    Arguments:
        dir_path: [str] Path to the directory.

    Keyword arguments:
        traj_type: [str] Type of the picke file to load. 'original' by default.
            - 'sim_met' - simulated meteors
            - 'mc' - Monte Carlo trajectory
            - 'gural' - Gural trajectory
            - <anything else> - any other .pickle format will be loaded

        unique: [bool] Return only unique file names, and if there are more file names with the same name,
            return the one that is in the directory with the minimum depth.

    Return:
        [list] A list of loaded objects.

    """

    def _checkUniquenessAndDepth(lst, index):
        """ Checks if the file name with the given index is unique, and if not, if it has the smallest depth. """

        ref_name, ref_depth = lst[index]

        min_depth = np.inf

        for entry in lst:
            file_name, depth = entry

            if (ref_name == file_name):
                min_depth = min(min_depth, depth)

        # If the given depth is the minimum depth, return True
        if min_depth == ref_depth:
            return True

        else:
            return False



    # Get all files in the given directory structure
    dir_files = listDirRecursive(dir_path)

    # Select only pickle files
    pickle_files = [file_path for file_path in dir_files if '.pickle' in os.path.split(file_path)[1]]


    # Select SimMet pickle files
    if traj_type == 'sim_met':
        pickle_files = [pickle_f for pickle_f in pickle_files if '_sim_met.pickle' in pickle_f]

    # Select only Monte Carlo pickle files if monte_carlo is True
    elif traj_type == 'mc':
        pickle_files = [pickle_f for pickle_f in pickle_files if '_mc_trajectory.pickle' in pickle_f]

    # Select MILIG trajectories
    elif traj_type == 'milig':
        pickle_files = [pickle_f for pickle_f in pickle_files if '_milig.pickle' in pickle_f]

    # Select gural trajectory
    elif 'gural' in traj_type:
        pickle_files = [pickle_f for pickle_f in pickle_files if '_{:s}_trajectory.pickle'.format(traj_type) \
            in pickle_f]

    # Select non-Monte Carlo pickle files
    else:
        pickle_files = [pickle_f for pickle_f in pickle_files if ('trajectory.pickle' in pickle_f) \
            and not ('_mc' in pickle_f) and not ('_gural' in pickle_f)]


    # Get only unique file names. If there are duplicated, get those which have the smallest directory depth,
    #   and if the depth is the same, return the first one alphabetically
    if unique:

        pickle_files = sorted(pickle_files)

        # Extract file names and their depths
        name_depth_list = []
        for file_name in pickle_files:

            # Split by the directory
            s = file_name.split(os.sep)

            # Store the name with the depth
            name_depth_list.append([s[-1], len(s)])


        pickle_files_unique = []

        # Find unique file names with the smalled directory depth. If depths are equal, the first file will be 
        #   chosen
        added_names = []
        for i, (pickle_file, entry) in enumerate(zip(pickle_files, name_depth_list)):

            file_name, depth = entry

            # Check if the file name is unique and it has the smallest depth, and add it to the final list if it is
            if _checkUniquenessAndDepth(name_depth_list, i) and (file_name not in added_names):
                pickle_files_unique.append(pickle_file)
                added_names.append(file_name)



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




def plotOrbitElements(pickle_trajs, plt_type='sol_a', plt_handle=None, **kwargs):
    """ Plot the orbital elements of the given trajectories. """


    if plt_handle is None:
        _, plt_handle = plt.subplots(1, 1)


    if plt_type == 'sol_a':


        la_sun_list = []
        a_list = []

        # Go thorugh all pickles
        for pick in pickle_trajs:

            # NOTE: In older versions of the simulator, the orbit is named 'orb', while the trajectory 
            #   objects has 'orbit'. This has been changed to 'orbit' everywhere, but some old pickle file 
            #   might be around.
            try:
                orbit = pick.orb
            except:
                orbit = pick.orbit


            if orbit.la_sun is None:
                continue


            # Extract solar longitudes
            la_sun_list.append(np.degrees(orbit.la_sun))

            # Extract the semi-major axis
            a_list.append(orbit.a)



        la_sun_list = np.array(la_sun_list)
        a_list = np.array(a_list)


        # Handle the 0-360 boundary
        if (np.max(la_sun_list) - np.min(la_sun_list)) > 180:
            la_sun_list = la_sun_list[la_sun_list < 180] + 360

        # Plot the orbits
        plt.scatter(la_sun_list, a_list, **kwargs)

        # Get the X ticks
        x_ticks = [item for item in plt.gca().get_xticks()]

        # Substract 360 from all ticks larger then 360
        x_ticks = [xt if xt < 360 else str(float(xt) - 360) for xt in x_ticks]
        

        plt.xlabel('Solar longitude (deg)')

        plt.ylabel('Semi-major axis (AU)')


    return plt_handle



def compareTrajToSim(dir_path, sim_meteors, traj_list, solver_name, radiant_extent, vg_extent, vmax=5):
    """ Compares results of a simulation to results of trajectory solving. """


    traj_sim_pairs = []

    # Find the simulation that matches the trajectory solution
    for traj in traj_list:
        for sim in sim_meteors:

            if traj.jdt_ref == sim.jdt_ref:
                traj_sim_pairs.append([traj, sim])
                break


    vg_diffs = []
    radiant_diffs = []
    failed_count = 0
    # Go through all the pairs and calculate the difference in the geocentric velocity and distance between
    #   the true and the estimated radiant
    for entry in traj_sim_pairs:
        
        traj, sim = entry

        # Skip the orbit if it was not estimated properby
        if traj.orbit.v_g is None:
            failed_count += 1
            continue

        # Difference in the geocentric velocity (km/s)
        vg_diff = (traj.orbit.v_g - sim.v_g)/1000

        # Difference in radiant (degrees)
        radiant_diff = np.degrees(angleBetweenSphericalCoords(sim.dec_g, sim.ra_g, traj.orbit.dec_g, \
            traj.orbit.ra_g))


        # Check if the results are within the given extents
        if (radiant_diff > radiant_extent) or (abs(vg_diff) > vg_extent):
            failed_count += 1
            continue


        vg_diffs.append(vg_diff)
        radiant_diffs.append(radiant_diff)


    # Calculate standard deviations
    vg_std = RMSD(np.array(vg_diffs))
    radiant_std = RMSD(np.array(radiant_diffs))

    # Define limits of the plot
    extent = [0, radiant_extent, -vg_extent, vg_extent]

    # Plot a 2D histogram
    plt.hexbin(radiant_diffs, vg_diffs, gridsize=20, extent=extent, vmin=0, vmax=vmax, cmap='viridis_r')

    # Plot a dVg = 0 line
    rad_plt_arr = np.linspace(0, radiant_extent, 10)
    plt.plot(rad_plt_arr, np.zeros_like(rad_plt_arr), linestyle='--', color='k', linewidth=1)

    plt.xlim(0, radiant_extent)
    plt.ylim(-vg_extent, vg_extent)

    plt.title('{:s}, failures: {:d}, $\sigma_R$ = {:.2f} deg, $\sigma_V$ = {:.2f} km/s'.format(solver_name, failed_count, radiant_std, vg_std))

    plt.xlabel('Radiant difference (deg)')
    plt.ylabel('Vg difference (km/s)')

    plt.savefig(os.path.join(dir_path, 'solution_comparison_{:s}.png'.format(solver_name)), dpi=300)

    plt.show()


if __name__ == "__main__":


    #dir_path = "../DenisGEMcases/"
    #dir_path = "../DenisGEMcases_5_sigma"
    #dir_path = "../Romulan2012Geminids"

    dir_path = "../SimulatedMeteors/EMCCD/2011Draconids"

    #dir_path = "../SimulatedMeteors/CAMO/2011Draconids"
    #dir_path = "../SimulatedMeteors/CAMO/2012Perseids"

    #dir_path = "../SimulatedMeteors/SOMN_sim/2011Draconids"
    #dir_path = "../SimulatedMeteors/SOMN_sim/2012Ursids"
    #dir_path = "../SimulatedMeteors/SOMN_sim/2012Perseids"
    #dir_path = "../SimulatedMeteors/SOMN_sim/2015Taurids"
    #dir_path = "../SimulatedMeteors/CAMSsim/2012Perseids"


    # Minimum convergence angle (deg)
    min_conv_angle = 5.0



    solvers = ['milig', 'mc', 'gural0', 'gural1', 'gural3']
    plot_labels = ['MILIG', 'Monte Carlo', 'Gural (constant)', 'Gural (linear)', 'Gural (exp)']
    markers = ['o', 's', '+', 'x', 'D']
    sizes = [20, 20, 40, 40, 10]



    # Load simulated meteors
    sim_meteors = collectTrajPickles(dir_path, traj_type='sim_met')



    # ### PLOT ORBIT ELEMENTS ###
    # ##########################################################################################################

    # # Plot simulated meteors
    # plt_handle = plotOrbitElements(sim_meteors, plt_type='sol_a', label='Simulated', marker='o', s=5)


    # # Make sure all lengths are the same
    # if sum([len(solvers), len(plot_labels), len(markers), len(sizes)]) != 4*len(solvers):
    #     print('The lenghts of solvers, plots, markers and sizes is not the same!')
    #     sys.exit()

    # # Make plots from data with different solvers

    # for solver, plt_lbl, marker, size in zip(solvers, plot_labels, markers, sizes):

    #     # Load trajectories
    #     traj_list = collectTrajPickles(dir_path, traj_type=solver)


    #     # Filter by convergence angle
    #     if 'gural' in solver:

    #         # Remove all trajectories with the convergence angle less then 15 deg
    #         traj_list = [traj for traj in traj_list if np.degrees(traj.max_convergence) >= min_conv_angle]

    #         pass

    #     else:

    #         # Remove all trajectories with the convergence angle less then 15 deg
    #         traj_list = [traj for traj in traj_list if np.degrees(traj.best_conv_inter.conv_angle) \
    #             >= min_conv_angle]


    #     # Plot trajectories
    #     plt_handle = plotOrbitElements(traj_list, plt_type='sol_a', plt_handle=plt_handle, label=plt_lbl, \
    #         alpha=0.5, s=size, marker=marker)


    
    # # Get the limits of the plot
    # x_min, x_max = plt_handle.get_xlim()

    
    # # Plot the 7:2 resonance with Jupiter
    # sol_arr = np.linspace(x_min, x_max, 10)

    # plt.plot(sol_arr, np.zeros_like(sol_arr) + 2.24, linestyle='--', color='k')
    # plt.plot(sol_arr, np.zeros_like(sol_arr) + 2.28, linestyle='--', color='k')

    # # Set xlim
    # plt.xlim([x_min, x_max])

    # # Limit a from 1.5 AU to 3.5 AU
    # plt.ylim([1.5, 3.5])


    # plt.legend()

    # # Save the figure
    # plt.savefig(os.path.join(dir_path, 'solver_comparison_sol_a.png'), dpi=300)

    # plt.show()

    # ##########################################################################################################



    ### PLOT TRAJECTORY SOLVER PRECISION GRAPHS ###
    ##########################################################################################################


    # Compare trajectories to simulations
    for solver, solver_name in zip(solvers, plot_labels):

        # Load trajectories
        traj_list = collectTrajPickles(dir_path, traj_type=solver)

        # Filter by convergence angle
        if 'gural' in solver:

            # Remove all trajectories with the convergence angle less then 15 deg
            traj_list = [traj for traj in traj_list if np.degrees(traj.max_convergence) >= min_conv_angle]

            pass

        else:

            # Remove all trajectories with the convergence angle less then 15 deg
            traj_list = [traj for traj in traj_list if np.degrees(traj.best_conv_inter.conv_angle) \
                >= min_conv_angle]

        # Plot the 2D histogram comparing the results, radiants within X degrees, Vg within X km/s
        compareTrajToSim(dir_path, sim_meteors, traj_list, solver_name, 0.5, 1.0, vmax=5)



    ##########################################################################################################



    sys.exit()


    ##########################################################################################################

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
