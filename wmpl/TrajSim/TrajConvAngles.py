""" Plot radiant and velocity error vs. convergence angle for all solvers of the given shower and system. """


import os

import numpy as np
import matplotlib.pyplot as plt

from wmpl.TrajSim.AnalyzeTrajectories import collectTrajPickles, pairTrajAndSim, calcTrajSimDiffs
from wmpl.Utils.Math import histogramEdgesEqualDataNumber

### This import is needed to be able to load SimMeteor pickle files
from wmpl.TrajSim.ShowerSim import SimMeteor, AblationModelVelocity, LinearDeceleration, ConstantVelocity
###





if __name__ == "__main__":



    # ## CAMO ###
    # plot_title = "CAMO"
    # dir_path = "../SimulatedMeteors/CAMO/2012Geminids_1000"

    # solvers = ['planes', 'los', 'milig', 'mc', 'gural1', 'gural3']
    # solvers_plot_labels = ['IP', 'LoS', 'LoS-FHAV', 'Monte Carlo', 'MPF linear', 'MPF exp']

    # ## ###



    # ### CAMS ###
    # plot_title = "Moderate FOV (CAMS-like)"
    # dir_path = "../SimulatedMeteors/CAMSsim_2station/2012Geminids_1000"

    # solvers = ['planes', 'los', 'milig', 'mc', 'gural0', 'gural0fha', 'gural1', 'gural3']
    # solvers_plot_labels = ['IP', 'LoS', 'LoS-FHAV', 'Monte Carlo', 'MPF const', 'MPF const-FHAV', \
    #     'MPF linear', 'MPF exp']

    # ### ###



    ### SOMN ###
    plot_title = "All-sky"
    dir_path = "./mnt/bulk/SimulatedMeteors/SOMN_sim_2station/2012Geminids_1000"

    solvers = ['planes', 'los', 'milig', 'mc', 'gural0', 'gural0fha', 'gural1', 'gural3']
    solvers_plot_labels = ['IP', 'LoS', 'LoS-FHAV', 'Monte Carlo', 'MPF const', 'MPF const-FHAV', \
        'MPF linear', 'MPF exp']

    ### ###



    # Number of historgram bins
    conv_hist_bins = 30



    ##########################################################################################################



    # Split the path into components
    path = os.path.normpath(dir_path)
    path = path.split(os.sep)

    # Extract the system and the shower name
    system_name = path[-2].replace("_", "").replace('sim', '')
    shower_name = path[-1]
    shower_name = shower_name[:4] + ' ' + shower_name[4:]


    # Load simulated meteors
    sim_meteors = collectTrajPickles(dir_path, traj_type='sim_met')


    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

    # Compare trajectories to simulations
    for solver, solver_name in zip(solvers, solvers_plot_labels):

        # Load trajectories
        traj_list = collectTrajPickles(dir_path, traj_type=solver)

        # Pair simulations and trajectories
        traj_sim_pairs = pairTrajAndSim(traj_list, sim_meteors)

        # Compute radiant and velocity errors
        radiant_diffs, vg_diffs, conv_angles, _ = calcTrajSimDiffs(traj_sim_pairs, np.inf, np.inf)

        # Get edges of histogram and compute medians of errors in every bin
        edges = histogramEdgesEqualDataNumber(conv_angles, conv_hist_bins)

        binned_radiant_diffs = []
        binned_vg_diffs = []
        for i in range(len(edges) - 1):

            min_val = edges[i]
            max_val = edges[i + 1]

            # Compute median radiant and vg error of all trajectories in the given range of conv angles
            radiant_diff_med = np.median([radiant_diff for radiant_diff, conv_ang in zip(radiant_diffs, conv_angles) if (conv_ang >= min_val) and (conv_ang < max_val)])
            vg_diff_med = np.median([vg_diff for vg_diff, conv_ang in zip(vg_diffs, conv_angles) if (conv_ang >= min_val) and (conv_ang < max_val)])

            binned_radiant_diffs.append(radiant_diff_med)
            binned_vg_diffs.append(vg_diff_med)


        # Plot MPF results dotted
        if solver_name.startswith('MPF'):
            linestyle = (0, (1, 1))
            linewidth = 1

            if 'exp' in solver_name:
                marker = '*'

            elif 'const-FHAV' in solver_name:
                marker = '+'

            elif 'const' in solver_name:
                marker = 'x'

            else:
                marker = None

            markersize = 6


        elif solver_name == "Monte Carlo":
            linestyle = 'solid'
            linewidth = 2
            marker = '*'
            markersize = 8

        else:
            linestyle = 'solid'
            linewidth = 1

            if solver_name == "LoS":
                marker = 'x'

            elif solver_name == "LoS-FHAV":
                marker = '+'

            else:
                marker = None

            markersize = 6


        # Plot convergence angle vs radiant error
        ax1.plot(edges[:-1], binned_radiant_diffs, label=solver_name, linestyle=linestyle, \
            linewidth=linewidth, marker=marker, markersize=markersize, zorder=3)

        # Plot convergence angle vs Vg error
        ax2.plot(edges[:-1], binned_vg_diffs, label=solver_name, linestyle=linestyle, linewidth=linewidth, \
            marker=marker, markersize=markersize, zorder=3)

    # Add a zero velocity error line
    ax2.plot(edges[:-1], np.zeros_like(edges[:-1]), color='k', linestyle='--')


    ax1.legend()
    ax1.grid()
    ax1.set_ylabel('Radiant error (deg)')
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(left=0, right=np.max(edges[:-1]))

    ax2.grid()
    ax2.set_ylabel('$V_g$ error (km/s)')
    ax2.set_xlabel('Convergence angle (deg)')

    plt.title(plot_title)

    plt.tight_layout()

    plt.subplots_adjust(hspace=0.1)

    # Save the figure
    plt.savefig(os.path.join(dir_path, system_name + '_' + shower_name.replace(' ', '_') \
        + '_conv_angle.png'), dpi=300)

    plt.show()