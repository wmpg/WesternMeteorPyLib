""" Collect trajectory pickle files and analyze them by plotting desired parameters, exporting parameters in 
a summary file, etc. """

from __future__ import print_function, absolute_import, division

import sys
import os


import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

from wmpl.Utils.PlotCelestial import CelestialPlot
from wmpl.Utils.Pickling import loadPickle
from wmpl.Utils.OSTools import listDirRecursive
from wmpl.Utils.Math import angleBetweenSphericalCoords, RMSD

from wmpl.Utils.Dcriteria import calcDN, calcDV, calcDVuncert

### This import is needed to be able to load SimMeteor pickle files
from wmpl.TrajSim.ShowerSim import SimMeteor, AblationModelVelocity, LinearDeceleration, ConstantVelocity
###


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


    # Select intersecting planes trajectories
    elif traj_type == 'planes':
        pickle_files = [pickle_f for pickle_f in pickle_files if '_planes.pickle' in pickle_f]

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



def pairTrajAndSim(traj_list, sim_meteors, radiant_extent, vg_extent):
    """ Given a list of trajectories, simulated meteor object and max deviations in radiant and Vg, 
        return the list of radiant and Vg deviations and convergence angles. 
    """

    traj_sim_pairs = []

    # Find the simulation that matches the trajectory solution (the JDs might be a bit off, but find the one
    #   that matches the best)
    for traj in traj_list:
        
        min_indx = 0
        min_diff = np.inf

        for i, sim in enumerate(sim_meteors):

            # Try to check simulation and trajectory unique identifiers, if they exist
            if hasattr(traj, 'traj_id') and hasattr(sim, 'unique_id'):

                if traj.traj_id == sim.unique_id:
                    min_indx = i
                    min_diff = -1

                    print('Found pair using unique ID!')

                    # Break the loop because the pair was found
                    break


            # Find the best matching JD
            jd_diff = abs(traj.jdt_ref - sim.jdt_ref)

            if jd_diff < min_diff:
                min_diff = jd_diff
                min_indx = i


        # Add the pair to the list
        traj_sim_pairs.append([traj, sim_meteors[min_indx]])



    vg_diffs = []
    radiant_diffs = []
    conv_angles = []
    failed_count = 0

    # Go through all the pairs and calculate the difference in the geocentric velocity and distance between
    #   the true and the estimated radiant
    for entry in traj_sim_pairs:
        
        traj, sim = entry

        # Skip the orbit if it was not estimated properly
        if traj.orbit.v_g is None:
            failed_count += 1
            continue


        print('vgs:', traj.orbit.v_g, sim.v_g)

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

        # Store the convergence angle
        if hasattr(traj, 'best_conv_inter'):

            # This is a wmpl trajectory object
            conv_ang = traj.best_conv_inter.conv_angle

        else:
            # Gural type trajectory object
            conv_ang = traj.max_convergence

        conv_angles.append(conv_ang)



    return failed_count, radiant_diffs, vg_diffs, conv_angles




def compareTrajToSim(dir_path, sim_meteors, traj_list, solver_name, radiant_extent, vg_extent, vmax=5, 
    show_plot=True, ret_conv_angles=False):
    """ Compares results of a simulation to results of trajectory solving. """


    # Pair trajectories and simulations
    failed_count, radiant_diffs, vg_diffs, _ = pairTrajAndSim(traj_list, sim_meteors, radiant_extent, \
        vg_extent)


    vg_diffs_std = np.array(vg_diffs)
    radiant_diffs_std = np.array(radiant_diffs)    

    # Reject all differences outside 3 standard deviations
    for i in range(10):

        # Calculate the standard deviation
        vg_std = RMSD(vg_diffs_std)
        radiant_std = RMSD(radiant_diffs_std)

        # Reject all values outside 3 standard deviations
        vg_diffs_std = vg_diffs_std[np.abs(vg_diffs_std) < 3*vg_std]
        radiant_diffs_std = radiant_diffs_std[np.abs(radiant_diffs_std) < 3*radiant_std]


    #################################################################################################

    # Define limits of the plot
    extent = [0, radiant_extent, -vg_extent, vg_extent]

    # Plot a 2D histogram
    plt.hexbin(radiant_diffs, vg_diffs, gridsize=20, extent=extent, vmin=0, vmax=vmax, cmap='viridis_r')

    # Plot a dVg = 0 line
    rad_plt_arr = np.linspace(0, radiant_extent, 10)
    plt.plot(rad_plt_arr, np.zeros_like(rad_plt_arr), linestyle='--', color='k', linewidth=1)


    # Plot 3 sigma lines
    sigma_value = 3
    y_arr = np.linspace(-sigma_value*vg_std, sigma_value*vg_std, 10)
    plt.plot(np.zeros_like(y_arr) + sigma_value*radiant_std, y_arr, linewidth=1, color='0.5')
    
    x_arr = np.linspace(0, sigma_value*radiant_std, 10)
    plt.plot(x_arr, np.zeros_like(x_arr) + sigma_value*vg_std, linewidth=1, color='0.5')
    plt.plot(x_arr, np.zeros_like(x_arr) - sigma_value*vg_std, linewidth=1, color='0.5')


    plt.xlim(0, radiant_extent)
    plt.ylim(-vg_extent, vg_extent)

    plt.title('{:s}, failures: {:d}, $\sigma_R$ = {:.2f} deg, $\sigma_V$ = {:.2f} km/s'.format(solver_name, \
        failed_count, radiant_std, vg_std))

    plt.xlabel('Radiant difference (deg)')
    plt.ylabel('Vg difference (km/s)')

    plt.savefig(os.path.join(dir_path, 'solution_comparison_{:s}.png'.format(solver_name.replace(' ', '_'))),\
        dpi=300)

    if show_plot:
        plt.show()

    plt.clf()
    plt.close()


    return failed_count, radiant_std, vg_std


if __name__ == "__main__":


    # #dir_path = "../SimulatedMeteors/EMCCD/2011Draconids"
    # #dir_path = "../SimulatedMeteors/CABERNET/2011Draconids"

    # #dir_path = "../SimulatedMeteors/SOMN_sim/LongFireball"
    # #dir_path = "../SimulatedMeteors/SOMN_sim/LongFireball_nograv"


    # # Minimum convergence angle (deg)
    # #   Allsky - 15 deg
    # #   CAMS - 10 deg
    # #   CAMO - 1 deg
    # min_conv_angle = 10.0

    # # Radiant precision plot limits
    # #   Allsky - 5
    # #   CAMS - 1
    # #   CAMO - 0.5
    # sigma_r_max = 1.0 #deg
    # sigma_v_max = 1.0 #km/s


    # Show the comparison plots on screen. If False, the plots will just be saved, but not shown on the screen
    show_plot = False


    # min duration = Minimum duration of meteor in seconds (-1 to turn this filter off)
    data_list = [ # Trajectory directory                     Min Qc  dRadiant max   dVg max   min duration  skip solver plot
        # ["../SimulatedMeteors/CAMO/2011Draconids",            1.0,        0.5,         0.5,       -1, []],
        # ["../SimulatedMeteors/CAMO/2012Geminids",             1.0,        0.5,         0.5,       -1, ['MPF const', 'MPF const-FHAV']],
        # ["../SimulatedMeteors/CAMO/2012Perseids",             1.0,        0.5,         0.5,       -1, []] ]
        #
        # ["../SimulatedMeteors/CAMSsim/2011Draconids",         10.0,       1.0,         1.0,       -1, []],
        # ["../SimulatedMeteors/CAMSsim/2012Geminids",          10.0,       1.0,         1.0,       -1, []],
        # ["../SimulatedMeteors/CAMSsim/2012Perseids",          10.0,       1.0,         1.0,       -1, []] ]
        #
        # ["../SimulatedMeteors/SOMN_sim/2011Draconids",        15.0,       5.0,         5.0,       -1, []],
        # ["../SimulatedMeteors/SOMN_sim/2012Geminids",           15.0,       5.0,         5.0,       -1, []],
        # ["../SimulatedMeteors/SOMN_sim/2012Perseids",         15.0,       5.0,         5.0,       -1, []] ]
        
        # ["../SimulatedMeteors/SOMN_sim/2015Taurids",          15.0,       5.0,         5.0,       -1, []]]
        # ["../SimulatedMeteors/SOMN_sim/LongFireball",          5.0,       0.5,         0.5,        4, []]
        # ["../SimulatedMeteors/SOMN_sim/LongFireball_nograv",   5.0,       0.5,         0.5,        4, []]]
        #
        # ["../SimulatedMeteors/CAMO/2014Ursids",               1.0,        0.5,         0.5,       -1, []],
        # ["../SimulatedMeteors/CAMSsim/2014Ursids",            10.0,       1.0,         1.0,       -1, []],
        # ["../SimulatedMeteors/SOMN_sim/2014Ursids",           15.0,       5.0,         5.0,       -1, []],

        ["../SimulatedMeteors/Hamburg_stations/Hamburg_fall",   1.0,       0.5,         0.5,        -1, ['planes', 'milig', 'mc', 'gural0', 'gural0fha', 'gural1', 'gural3']]]




    solvers = ['planes', 'los', 'milig', 'mc', 'gural0', 'gural0fha', 'gural1', 'gural3']
    solvers_plot_labels = ['IP', 'LoS', 'LoS-FHAV', 'Monte Carlo', 'MPF const', 'MPF const-FHAV', 'MPF linear', 'MPF exp']
    markers = ['v', 'o', 's', '+', 'x', '.', 'D', 'd']
    sizes = [20, 20, 20, 40, 40, 20, 10, 20]


    results_list = []
    systems_list = []
    showers_list = []
    system_shower_solver_skip_list = []

    # Go through all simulations
    for dir_path, min_conv_angle, sigma_r_max, sigma_v_max, min_duration, \
        skip_plotting_solver_results in data_list:

        print('Plotting:', dir_path)

        # Split the path into components
        path = os.path.normpath(dir_path)
        path = path.split(os.sep)

        # Extract the system and the shower name
        system_name = path[-2].replace("_", "").replace('sim', '')
        shower_name = path[-1]
        shower_name = shower_name[:4] + ' ' + shower_name[4:]


        # Save system path
        system_path = os.path.join(*path[:-1])

        if not system_name in systems_list:
            systems_list.append(system_name)


        if not shower_name in showers_list:
            showers_list.append(shower_name)


        # Skip plotting aggregatd results for the combination of system, shower and solver
        if skip_plotting_solver_results:
            system_shower_solver_skip_list.append([system_name, shower_name, skip_plotting_solver_results])


        # Load simulated meteors
        sim_meteors = collectTrajPickles(dir_path, traj_type='sim_met')


        ### PLOT TRAJECTORY SOLVER PRECISION GRAPHS ###
        ##########################################################################################################


        # Compare trajectories to simulations
        for solver, solver_name in zip(solvers, solvers_plot_labels):

            # Load trajectories
            traj_list = collectTrajPickles(dir_path, traj_type=solver)


            # Filter by convergence angle
            if 'gural' in solver:

                # Remove all trajectories with the convergence angle less then min_conv_angle deg
                traj_list = [traj for traj in traj_list if np.degrees(traj.max_convergence) >= min_conv_angle]

            else:

                # Remove all trajectories with the convergence angle less then min_conv_angle deg
                traj_list = [traj for traj in traj_list if np.degrees(traj.best_conv_inter.conv_angle) \
                    >= min_conv_angle]

            # Skip the solver if there are no trajectories to plot
            if not traj_list:
                print('Skipping {:s} solver, no data...'.format(solver))
                continue


            # Filter by minimum duration
            if min_duration > 0:
                
                print('Filtering by minimum duration of {:.2f} seconds!'.format(min_duration))
                
                filtered_traj_list = []

                # Go through all trajectories
                for traj in traj_list:

                    if 'gural' in solver:

                        # Go through all observations and find the total duration
                        first_beginning = np.min([time_data[0] for time_data in traj.times])
                        last_ending = np.max([time_data[-1] for time_data in traj.times])


                    else:

                        # Go through all observations and find the total duration
                        first_beginning = np.min([obs.time_data[0] for obs in traj.observations])
                        last_ending = np.max([obs.time_data[-1] for obs in traj.observations])

                    total_duration = last_ending - first_beginning


                    if total_duration >= min_duration:
                        filtered_traj_list.append(traj)


                print('Taking {:d}/{:d} trajectories after duration filtering'.format(len(filtered_traj_list),\
                    len(traj_list)))

                traj_list = filtered_traj_list



            # Plot the 2D histogram comparing the results, radiants within X degrees, Vg within X km/s
            failed_count, radiant_std, vg_std = compareTrajToSim(dir_path, sim_meteors, traj_list, \
                solver_name, sigma_r_max, sigma_v_max, vmax=10, show_plot=show_plot)


            results_list.append([system_name, shower_name, solver_name, failed_count, len(traj_list), \
                radiant_std, vg_std, system_path])


        ##########################################################################################################


    # Set line styles for given shower
    linestyle_list = ['dotted', 'dashed', 'solid']

    # Generate colors for different showers
    #color_list = plt.cm.inferno(np.linspace(0.8, 0.2, 2))
    color_list = ['k']

    # Plot the comparison between solvers of one system
    for system_name in systems_list:

        print(system_name)

        # Only select the values for the given system
        system_results = [entry for entry in results_list if system_name == entry[0]]


        # Find the maximum Vg deviation
        vg_std_max = max([entry[6] for entry in system_results])

        plot_handle_list = []
        xticks_values = []
        xticks_labels = []


        # Find the maximum radiant deviation for the given solver
        #for i, solver_name_iter in enumerate(solvers_plot_labels):
        radiant_std_mean = np.median([entry[5] for entry in system_results for solver_name_iter in solvers_plot_labels if solver_name_iter == entry[2]])

        # Compute text padding
        pad_x = 0.02*vg_std_max
        pad_y = 0.01*radiant_std_mean


        # Go through all solvers
        for i, solver_name_iter in enumerate(solvers_plot_labels):

            # Compute the left_limit position of the boxes for the given solver
            left_limit = 1.5*vg_std_max*i

            # Round the left_limit point
            left_limit = round(left_limit, 1)

            # Plot the name of the solver
            plt.text(left_limit + pad_x, pad_y, solver_name_iter, rotation=90, verticalalignment='bottom', 
                horizontalalignment='left', fontsize=10, color='r', zorder=5, weight='bold')

            print("{:<17s}".format(solver_name_iter), end='')
                
            left_limit_list = []
            failure_list = []

            vg_shower_std_max = 0
            radiant_std_max = 0

            # Go through all showers
            for j, shower_name_iter in enumerate(showers_list):

                # Only select results for the given shower
                shower_results = [entry for entry in system_results if shower_name_iter == entry[1]]

                # Skip plotting if there are no results for a given shower
                if len(shower_results) == 0:
                    continue


                for result in shower_results:

                    system_name, shower_name, solver_name, failed_count, total_count, radiant_std, vg_std, \
                        system_path = result

                    # Take only the results for the given solver
                    if solver_name_iter != solver_name:
                        continue


                    # Skip plotting if this combination of shower, solver and system is in the no-plot list
                    skip_plot = False
                    for no_plot_entry in system_shower_solver_skip_list:
                        system_name_tmp, shower_name_tmp, skip_plotting_solver_results = no_plot_entry

                        if (system_name == system_name_tmp) and (shower_name == shower_name_tmp) \
                            and (solver_name in skip_plotting_solver_results):

                            skip_plot = True
                            break


                    if (vg_std > vg_shower_std_max) and (not skip_plot):
                        vg_shower_std_max = vg_std

                    if (radiant_std > radiant_std_max) and (not skip_plot):
                        radiant_std_max = radiant_std
                    
                    #print("Failed {:d}/{:d}".format(failed_count, total_count))

                    failure_list.append(failed_count)


                    # Plot the standard deviation box
                    right_limit = left_limit + vg_std
                    x_arr = np.linspace(left_limit, right_limit, 10)
                    y_arr = np.linspace(0, radiant_std, 10)

                    left_limit_list.append(left_limit)


                    # Select line styles and colors
                    color_name = color_list[(j//2)%len(color_list)]
                    linestyle = linestyle_list[j%len(linestyle_list)]

                    if not skip_plot:
                        upper = plt.plot(x_arr, np.zeros_like(x_arr) + radiant_std, color=color_name, label=shower_name_iter, linestyle=linestyle)
                        plt.plot(np.zeros_like(x_arr) + left_limit, y_arr, color=color_name, linestyle=linestyle)
                        plt.plot(np.zeros_like(x_arr) + right_limit, y_arr, color=color_name, linestyle=linestyle)

                        # Add the legend only for the first solver
                        if solver_name_iter == solvers_plot_labels[0]:
                            plot_handle_list.append(upper[0])


                    #print("{:s}, {:s}, {:d}, {:.2f}, {:.2f}".format(shower_name_iter, solver_name, failed_count, radiant_std, vg_std))
                    print(" & {:2d} & \\ang{{{:.2f}}} & \\SI{{{:.2f}}}".format(failed_count, radiant_std, \
                        vg_std) + "{\\kilo \\metre \\per \\second}", end='')

            print('\\\\')

            
            # Write the number of failed solutions per solver
            failed_count_str = "/".join(map(str, failure_list))
            plt.text(left_limit + vg_shower_std_max/2, 1.01*radiant_std_max, failed_count_str, ha='center')


            # Add X ticks
            vg_tick = round(vg_std_max/2, 1)

            xticks_values.append(left_limit)
            xticks_values.append(left_limit + vg_tick)

            xticks_labels.append('0')
            xticks_labels.append('{:.1f}'.format(vg_tick))

        
        plt.legend(handles=plot_handle_list)

        # Replace X ticks
        plt.xticks(xticks_values, xticks_labels, rotation=90)

        plt.ylabel('Radiant error (deg)')
        plt.xlabel('Velocity error (km/s)')


        # Increase the top limit a bit
        _, y_max = plt.gca().get_ylim()

        plt.gca().set_ylim(0, 1.2*y_max)


        plt.tight_layout()

        # Save the figure
        plt.savefig(os.path.join(system_path, system_name + '_solver_comparison.png'), dpi=300)

        plt.show()






    sys.exit()



    ### PLOT ORBITAL ELEMENTS OF SELECT SHOWER ###
    ##########################################################################################################


    dir_path = "../SimulatedMeteors/SOMN_sim/2015Taurids"

    # Load simulated meteors
    sim_meteors = collectTrajPickles(dir_path, traj_type='sim_met')

    # Plot simulated meteors
    plt_handle = plotOrbitElements(sim_meteors, plt_type='sol_a', label='Simulated', marker='o', s=5)


    # Make sure all lengths are the same
    if sum([len(solvers), len(solvers_plot_labels), len(markers), len(sizes)]) != 4*len(solvers):
        print('The lenghts of solvers, plots, markers and sizes is not the same!')
        sys.exit()

    # Make plots from data with different solvers

    for solver, plt_lbl, marker, size in zip(solvers, solvers_plot_labels, markers, sizes):

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


        # Plot trajectories
        plt_handle = plotOrbitElements(traj_list, plt_type='sol_a', plt_handle=plt_handle, label=plt_lbl, \
            alpha=0.5, s=size, marker=marker)


        # Compute the stddev of differences in semi-major axis for every solver
        traj_list_filtered = [traj for traj in traj_list if traj.orbit.a is not None]
        a_diff = [traj.orbit.a - 2.26 for traj in traj_list if traj.orbit.a is not None]

        # Remove outliers
        removed_count = 0
        for i in range(100):
            
            a_diff_stddev = np.std(a_diff)

            for a_temp in a_diff:
                if np.abs(a_temp) > 3*a_diff_stddev:

                    # Get the index of the a_diff to remove
                    a_rm_indx = a_diff.index(a_temp)

                    # Remove a from lists
                    a_diff.remove(a_temp)
                    traj_list_filtered.pop(a_rm_indx)

                    removed_count += 1
        
        #print(plt_lbl, a_diff)
        print(plt_lbl, a_diff_stddev, 'AU', 'removed', removed_count)

        # # Plot the estimated semi-major axes
        # plt.clf()
        # plt.scatter([np.degrees(traj.orbit.la_sun) for traj in traj_list_filtered], [traj.orbit.a for traj in traj_list_filtered], marker='x', color='r', label=plt_lbl)

        # # Plot the original points
        # plotOrbitElements(sim_meteors, plt_type='sol_a', label='Simulated', marker='o', s=5, plt_handle=plt)

        # plt.legend()
        # plt.show()

    
    # Get the limits of the plot
    x_min, x_max = plt_handle.get_xlim()

    
    # Plot the 7:2 resonance with Jupiter
    sol_arr = np.linspace(x_min, x_max, 10)

    plt.plot(sol_arr, np.zeros_like(sol_arr) + 2.24, linestyle='--', color='k')
    plt.plot(sol_arr, np.zeros_like(sol_arr) + 2.28, linestyle='--', color='k')

    # Set xlim
    plt.xlim([x_min, x_max])

    # Limit a from 1.5 AU to 3.5 AU
    #plt.ylim([1.5, 3.5])
    plt.ylim([2.2, 2.35])


    plt.legend()

    # Save the figure
    plt.savefig(os.path.join(dir_path, 'solver_comparison_sol_a_zoom.png'), dpi=300)

    plt.show()

    sys.exit()

    ##########################################################################################################







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
