""" Produces graphs which show how many standard deviations the estimated value is off from the real simulated
    value. 
"""

import os

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import scipy.stats

from wmpl.TrajSim.AnalyzeTrajectories import collectTrajPickles, pairTrajAndSim
from wmpl.Utils.Math import angleBetweenSphericalCoords

### This import is needed to be able to load SimMeteor pickle files
from wmpl.TrajSim.ShowerSim import SimMeteor, AblationModelVelocity, LinearDeceleration, ConstantVelocity
###


if __name__ == "__main__":

    ### SOMN ###
    # Path to the folder with simulated and solved data + data sigma
    # dir_and_sigma = ["../SimulatedMeteors/SOMN_sim_2station/2012Geminids_1000", 2.0]
    # dir_and_sigma = ["../SimulatedMeteors/SOMN_sim/2011Draconids", 1.5]
    # dir_and_sigma = ["../SimulatedMeteors/SOMN_sim/2012Perseids", 1.7]
    # min_conv_angle = 15.0
    # radiant_diff_max = 5.0


    ### ###


    ## CAMS ###
    # Path to the folder with simulated and solved data
    #dir_and_sigma = ["../SimulatedMeteors/CAMSsim_2station/2012Geminids_1000", 2.0]
    #dir_and_sigma = ["../SimulatedMeteors/CAMSsim/2012Perseids", 1.5]
    dir_and_sigma = ["../SimulatedMeteors/CAMSsim/2011Draconids", 1.2]
    min_conv_angle = 10.0
    radiant_diff_max = 1.0

    ##


    # ### CAMO ###
    # # Path to the folder with simulated and solved data
    # # dir_and_sigma = ["../SimulatedMeteors/CAMO/2012Geminids_1000", 4.5]
    # #dir_and_sigma = ["../SimulatedMeteors/CAMO/2011Draconids", 3.5]
    # # dir_and_sigma = ["../SimulatedMeteors/CAMO/2012Perseids", 1.5]

    # min_conv_angle = 1.0
    # radiant_diff_max = 0.5

    # ### ###

    
    dir_path, normal_distribution_sigma = dir_and_sigma

    # Load simulated meteors
    sim_meteors = collectTrajPickles(dir_path, traj_type='sim_met')

    # Load MC estimated trajectories
    traj_list = collectTrajPickles(dir_path, traj_type='mc')


    # Remove all trajectories with the convergence angle less then min_conv_angle deg
    traj_list = [traj for traj in traj_list if np.degrees(traj.best_conv_inter.conv_angle) \
        >= min_conv_angle]


    # Pair simulations and trajectories
    traj_sim_pairs = pairTrajAndSim(traj_list, sim_meteors)

    vinit_stds = []
    radiant_stds = []
    for (traj, sim) in traj_sim_pairs:


        # Skip the orbit if it was not estimated properly
        if traj.orbit.v_g is None:
            continue

        # Difference in the initial velocity (m/s)
        vinit_diff = traj.v_init - sim.v_begin

        # Difference in geocentric radiant (degrees)
        radiant_diff = np.degrees(angleBetweenSphericalCoords(sim.dec_g, sim.ra_g, traj.orbit.dec_g, \
            traj.orbit.ra_g))


        # Reject everything larger than the threshold
        if radiant_diff > radiant_diff_max:
            continue


        # # Skip zero velocity uncertanties
        # if traj.uncertanties.v_init == 0:
        #     continue

        # # Compute the number of standard deviations of the real error in the velocity
        # vinit_diff_std = vinit_diff/traj.uncertanties.v_init

        # # # Skip all cases where the difference is larger than 0.5 km/s
        # # if np.abs(vinit_diff_std) > 500:
        # #     continue

        
        # print(vinit_diff, traj.uncertanties.v_init, vinit_diff_std)




        ### Compute the number of standard deviations of the real error in the radiant ###

        # Compute unified standard deviation for RA and Dec
        radiant_std = np.degrees(np.hypot(traj.uncertanties.ra_g, traj.uncertanties.dec_g))

        if radiant_std == 0:
            continue

        radiant_diff_std = radiant_diff/radiant_std

        # Reject standard deviations larger than 10 sigma
        if (radiant_diff_std > 10) or (radiant_diff_std < 0):
            continue


        ### ###


        #vinit_stds.append(vinit_diff_std)

        radiant_stds.append(radiant_diff_std)


    # Plot the cumulative histogram of radiant errors in sigmas
    bins, edges, _ = plt.hist(radiant_stds, bins=len(radiant_stds), cumulative=True, density=True, \
        color='0.7', label='Simulated vs. estimated values')


    # Plot the gaussian distribution
    gauss_samples = np.abs(scipy.stats.norm.rvs(size=10*len(radiant_stds), loc=0.0, scale=normal_distribution_sigma))
    plt.hist(gauss_samples, bins=len(gauss_samples), cumulative=True, density=True, histtype='step', 
        color='k', zorder=3)

    # x_arr = np.linspace(0, 5, 100)
    # plt.plot(x_arr, scipy.stats.norm.ppf(x_arr, loc=0, scale=1.0))

    black_line = mlines.Line2D([], [], color='k', markersize=15, label='Normal distribution $\sigma$ = {:.1f}'.format(normal_distribution_sigma))
    plt.legend(handles=list(plt.gca().get_legend_handles_labels()[0]) + [black_line])


    plt.xlim(xmax=4)
    plt.ylim([0, 1.0])
    plt.xlabel('True radiant error ($\sigma$)')
    plt.ylabel('Normalized count')

    plt.grid(color='0.8')

    plt.savefig(os.path.join(dir_path, "true_vs_estimated_error.png"), dpi=300)

    plt.show()