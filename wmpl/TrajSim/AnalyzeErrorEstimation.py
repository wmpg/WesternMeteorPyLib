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

    # ### SOMN ###
    # min_conv_angle = 15.0
    # radiant_diff_max = 5.0

    # dir_path_list = ["/mnt/bulk/SimulatedMeteors/SOMN_sim_2station/2012Geminids_1000", \
    #                  "/mnt/bulk/SimulatedMeteors/SOMN_sim/2011Draconids",\
    #                  "/mnt/bulk/SimulatedMeteors/SOMN_sim/2012Perseids"]

    # ###


    # ## CAMS ###
    # # Path to the folder with simulated and solved data
    # dir_path_list = ["/mnt/bulk/SimulatedMeteors/CAMSsim_2station/2012Geminids_1000",
    #             "/mnt/bulk/SimulatedMeteors/CAMSsim/2012Perseids",
    #             "/mnt/bulk/SimulatedMeteors/CAMSsim/2011Draconids"]
    # min_conv_angle = 10.0
    # radiant_diff_max = 1.0

    # ##


    ### CAMO ###
    # Path to the folder with simulated and solved data
    dir_path_list = ["/mnt/bulk/SimulatedMeteors/CAMO/2012Geminids_1000", \
                     "/mnt/bulk/SimulatedMeteors/CAMO/2011Draconids", \
                     "/mnt/bulk/SimulatedMeteors/CAMO/2012Perseids"]

    min_conv_angle = 1.0
    radiant_diff_max = 0.5

    ### ###


    # Go through all directories
    for dir_path in dir_path_list:

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
        radiant_errors = []
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
            radiant_errors.append(radiant_diff)
            radiant_stds.append(radiant_diff_std)



        # Fit a truncated normal distribution to the data
        dist_min = 0
        dist_max = 10
        truncnorm_dist_params = scipy.stats.truncnorm.fit(radiant_stds, dist_min, dist_max, loc=0)
        print(truncnorm_dist_params)

        # Fit a chi2 distribution to the data
        df = 2
        chi2_dist_params = scipy.stats.chi2.fit(radiant_stds, df, loc=0)
        print(chi2_dist_params)

        # Kolmogorov-Smirnov tests
        kstest_truncnorm = scipy.stats.kstest(radiant_stds, scipy.stats.truncnorm.cdf, truncnorm_dist_params)
        kstest_chi2 = scipy.stats.kstest(radiant_stds, scipy.stats.chi2.cdf, chi2_dist_params)

        print(kstest_truncnorm)
        print(kstest_chi2)


        x_arr = np.linspace(0, np.max(radiant_stds), 100)


        # # Plot a histogram of radiant errors in sigmas
        # bins, edges, _ = plt.hist(radiant_stds, bins=int(np.floor(np.sqrt(len(radiant_stds)))), cumulative=False, density=True, \
        #     color='0.7', label='Simulated vs. estimated values')

        # # Plot fitted truncated normal distribution
        # plt.plot(x_arr, scipy.stats.truncnorm.pdf(x_arr, *truncnorm_dist_params))

        # # Plot fitted chi2 distribution
        # plt.plot(x_arr, scipy.stats.chi2.pdf(x_arr, *chi2_dist_params))


        # plt.show()



        # Plot the cumulative histogram of radiant errors in sigmas
        bins, edges, _ = plt.hist(radiant_stds, bins=len(radiant_stds), cumulative=True, density=True, \
            color='0.7', label='Simulated vs. estimated values')

        # Plot the fitted truncnorm distribution
        plt.plot(x_arr, scipy.stats.truncnorm.cdf(x_arr, *truncnorm_dist_params), linestyle='dashed', \
            color='k', label='Trucnorm distribution: $\\sigma$ = {:.2f}'.format(truncnorm_dist_params[3]))

        # Plot the fitted chi2 distribution
        plt.plot(x_arr, scipy.stats.chi2.cdf(x_arr, *chi2_dist_params), color='k',
            label='$\\chi^2$ distribution: $k$ = {:.2f}, scale = {:.2f}'.format(chi2_dist_params[0], chi2_dist_params[2]))

        plt.legend(loc='lower right')


        plt.xlim(xmax=4)
        plt.ylim([0, 1.0])
        plt.xlabel('True radiant error ($\sigma$)')
        plt.ylabel('Cumulative normalized count')

        plt.grid(color='0.8')

        plt.savefig(os.path.join(dir_path, "true_vs_estimated_error.png"), dpi=300)

        plt.show()