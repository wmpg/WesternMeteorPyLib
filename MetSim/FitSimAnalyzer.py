""" Analyzing simulations done with FitSim. """

from __future__ import print_function, division, absolute_import

import os
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider

from Config import config
from Utils.Pickling import loadPickle
from Utils.TrajConversions import jd2Date
from MetSim.MetSim import loadInputs
from MetSim.FitSim import calcVelocity


# Minimum difference between slider
SLIDER_EPSILON = 0.01


class FitSimAnalyzer(object):

    def __init__(self, dir_path_mir, traj_pickle_file):



        # Name of input file for meteor parameters
        meteor_inputs_file = config.met_sim_input_file

        # Load input meteor data
        met, consts = loadInputs(meteor_inputs_file)


        # Load the pickled trajectory
        self.traj = loadPickle(dir_path_mir, traj_pickle_file)


        self.results_list = []
        self.full_cost_list = []

        # Go through all observations
        for station_ind, obs in enumerate(self.traj.observations):

            # Name of the results file
            results_file = jd2Date(self.traj.jdt_ref, dt_obj=True).strftime('%Y%m%d_%H%M%S') + "_" \
                + str(self.traj.observations[station_ind].station_id) + "_simulations.npy"

            results_file = os.path.join(dir_path_mir, results_file)

            # Add the results file to the results list
            self.results_list.append(results_file)

            # Take the parameters of the observation with the highest beginning height
            obs_time = self.traj.observations[station_ind].time_data
            obs_length = self.traj.observations[station_ind].length


            # Fit only the first 25% of the observed trajectory
            len_part = int(0.25*len(obs_time))

            # If the first 25% has less than 4 points, than take the first 4 points
            if len_part < 4:
                len_part = 4

            
            # Cut the observations to the first part of the trajectory
            obs_time = obs_time[:len_part]
            obs_length = obs_length[:len_part]
            

            # Calculate observed velocities
            velocities, time_diffs = calcVelocity(obs_time, obs_length)
            print(velocities)

            # Calculate the RMS of velocities
            vel_rms = np.sqrt(np.mean((velocities[1:] - self.traj.v_init)**2))

            print('Vel RMS:', vel_rms)

            # Calculate the along track differences
            along_track_diffs = (velocities[1:] - self.traj.v_init)*time_diffs[1:]

            # Calculate the full 3D residuals
            full_residuals = np.sqrt(along_track_diffs**2 \
                + self.traj.observations[station_ind].v_residuals[:len_part][1:]**2 \
                + self.traj.observations[station_ind].h_residuals[:len_part][1:]**2)

            # Calculate the average 3D deviation from the estimated trajectory
            full_cost = np.sum(np.abs(np.array(full_residuals)))/len(full_residuals)

            self.full_cost_list.append(full_cost)




        # Load solutions from a file
        self.loadSimulations()

        # Initialize the plot framework
        self.initGrid()

        # Initialize main plots
        self.dens_min_init, self.dens_max_init = self.updatePlots(init=True)

        self.dens_min = self.dens_min_init
        self.dens_max = self.dens_max_init


        ### SLIDERS

        # Sliders for density
        self.sl_ind_dev_1 = Slider(self.ax_sl_11, 'Min', self.dens_min, self.dens_max, valinit=self.dens_min)
        self.sl_ind_dev_2 = Slider(self.ax_sl_12, 'Max', self.dens_min, self.dens_max, valinit=self.dens_max, slidermin=self.sl_ind_dev_1)
        self.ax_sl_12.set_xlabel('Density')


        # Turn on slider updating
        self.sl_ind_dev_1.on_changed(self.updateSliders)
        self.sl_ind_dev_2.on_changed(self.updateSliders)


        ######



        plt.show()



    def loadSimulations(self):
        """ Load simulation results from a file. """

        self.solutions_list = []

        for results_file in self.results_list:

            # Load the numpy array with the results from a file
            solutions = np.load(results_file)

            self.solutions_list.append(solutions)



    def initGrid(self):
        """ Initializes the plot framework. """

        ### Create grid

        # Main gridspec
        gs = gridspec.GridSpec(6, 2)
        gs.update(hspace=0.5, bottom=0.05, top=0.95, left=0.05, right=0.98)

        # Index vs. deviations axes
        gs_ind = gridspec.GridSpecFromSubplotSpec(6, 2, subplot_spec=gs[:3, :], wspace=0.0, hspace=2.0)
        ax_ind_1 = plt.subplot(gs_ind[:5, 0])
        ax_ind_2 = plt.subplot(gs_ind[:5, 1], sharex=ax_ind_1, sharey=ax_ind_1)

        # Mass colorbar axis
        self.ax_cbar = plt.subplot(gs_ind[5, :])

        # Velocity vs. deviations axies
        gs_vel = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[3:5, :], wspace=0.0, hspace=0.2)
        ax_vel_1 = plt.subplot(gs_vel[0, 0])
        ax_vel_2 = plt.subplot(gs_vel[0, 1], sharex=ax_vel_1, sharey=ax_vel_1)


        # Disable left tick labels on plots to the right
        ax_ind_2.tick_params(labelleft='off')
        ax_vel_2.tick_params(labelleft='off')


        # Sliders
        gs_sl = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[5, :], wspace=0.15, hspace=0.2)

        # Slider upper left axis
        self.ax_sl_11 = plt.subplot(gs_sl[0, 0])

        # Slider lower left axis
        self.ax_sl_12 = plt.subplot(gs_sl[1, 0])

        # Slider upper right axis
        self.ax_sl_21 = plt.subplot(gs_sl[0, 1])

        # Slider lower right axis
        self.ax_sl_22 = plt.subplot(gs_sl[1, 1])

        ######


        self.axes_cost = [ax_ind_1, ax_ind_2]
        self.axes_velocity = [ax_vel_1, ax_vel_2]






    def updatePlots(self, init=False):
        """ Updates the plots with the given range of densities. 
    
        Keyword arguments:
            init: [bool] If True, plots will be shown with no constrain on density. False by defualt.

        """

        # List of cost plot handles
        plot_cost_handles = []

        # List of velocity plot handles
        plot_velocity_handles = []

        dens_min = 10000
        dens_max = 0

        # Go through all results file
        for n, (full_cost, solutions) in enumerate(zip(self.full_cost_list, self.solutions_list)):

            # Cost function plots
            ax_cost = self.axes_cost[n]

            # Velocity plots
            ax_vel = self.axes_velocity[n]


            # Extract initial velocities
            v_init_all = solutions[:, 1]

            # Idenfity unique initial velocities
            v_init_unique = np.unique(v_init_all)


            # List for velocities which are possible under the measurement RMS
            v_possible = []


            # List of velocities vs. best cost pairs
            vel_cost_pairs = []

            # Go through initial velocities
            for i, v_init in enumerate(v_init_unique):

                # Select only those with the given initial velocity
                select_ind = np.where(v_init_all == v_init)

                # Select the solution by the v init
                solutions_v_init = solutions[select_ind]


                # Extract densities from the solutions
                densities = solutions_v_init[:, 3]

                # If the plots are not being initializes, i.e. a constrain on densities was given,
                # select only those simulations in between selected densities
                if not init:

                    # Select the solutions only in the range of selected densities
                    solutions_v_init = solutions_v_init[(densities >= self.dens_min) & (densities <= self.dens_max), :]

                else:

                    # Store the largest and the smallest density
                    if np.max(densities) > dens_max:
                        dens_max = np.max(densities)

                    if np.min(densities) < dens_min:
                        dens_min = np.min(densities)


                # Sort by cost
                solutions_v_init[solutions_v_init[:, 0].argsort()]

                # Extract costs from the solution
                costs = solutions_v_init[:, 0]


                vel_cost_pairs.append([v_init, costs[0]])

                # Add the velocity to the list if the first cost is below the measurement RMS cost
                if costs[0] < full_cost:
                    v_possible.append(v_init)


                # Set text to mark the velocity
                ax_cost.text(0, costs[0], str(int(v_init)), ha='right')

                # Index vs. cost scatter plot where the color represents the mass
                scat_ind_dev = ax_cost.scatter(range(len(costs)), costs, c=solutions_v_init[:, 2], s=((i+1)**2)/2, 
                    norm=matplotlib.colors.LogNorm(), zorder=3)




            plot_cost_handles.append(scat_ind_dev)

            # Print the range of possible velocities from the simulations and threshold deviations
            if v_possible:
                v_possible = sorted(v_possible)
                print('Site', n+1,  'possible range of velocities:', min(v_possible), max(v_possible))

            else:
                print('No possible velocities!')


            # Plot the cost function values of the RMS of lengths along the track
            rms_cost_x = np.linspace(0, len(costs), 1000)
            rms_cost_y = np.array([full_cost]*1000)
            ax_cost.plot(rms_cost_x, rms_cost_y, label='RMS along track cost', linestyle='--', linewidth=2, 
                zorder=3)


            # Set the Y limit from 0 to 2x the threshold cost
            ax_cost.set_ylim(0, 2*full_cost)

            ax_cost.set_xlabel('Index')
            ax_cost.legend()

            ax_cost.set_title(str(n + 1))
            ax_cost.grid()


            ### Plot velocities vs. best cost

            vel_cost_pairs = np.array(vel_cost_pairs)
            vels, best_costs = vel_cost_pairs.T

            plot_vel_dev = ax_vel.plot(vels, best_costs, marker='x', label='Model V')

            plot_velocity_handles.append(plot_vel_dev)

            # Plot the threshold cost
            vel_rms_cost_x = np.linspace(np.min(vels), np.max(vels), 1000)
            vel_rms_cost_y = np.zeros_like(vel_rms_cost_x) + full_cost
            ax_vel.plot(vel_rms_cost_x, vel_rms_cost_y, linestyle='--', linewidth=2, zorder=3, label='RMS cost')

            # Plot the initial velocity from the trajectory solver
            v_init_orig_x = np.zeros(10) + self.traj.v_init
            v_init_orig_y = np.linspace(0, full_cost, 10)
            ax_vel.plot(v_init_orig_x, v_init_orig_y, color='r', zorder=3, label='$V_{init}$')

            # Plot the no-atmosphere velocity from the trajectory solver
            v_init_orig_x = np.zeros(10) + self.traj.orbit.v_inf
            v_init_orig_y = np.linspace(0, full_cost, 10)
            ax_vel.plot(v_init_orig_x, v_init_orig_y, color='g', zorder=3, label='$V_{\infty}$')

            ax_vel.set_xlabel('Velocity (m/s)')

            # Set the Y limit from 0 to 2x the threshold cost
            ax_vel.set_ylim(0, 2*full_cost)

            ax_vel.grid()

            ax_vel.legend()

            ######


        # Extract plot handles
        self.plot_ind_dev_1 = plot_cost_handles[0]
        self.plot_ind_dev_1 = plot_cost_handles[1]

        self.plot_vel_dev_1 = plot_velocity_handles[0]
        self.plot_vel_dev_2 = plot_velocity_handles[1]

        # Set mass scatter plot labels and colorbar
        self.axes_cost[0].set_ylabel('Average absolute deviations (m)')

        # Plot the masses colorbar
        plt.gcf().colorbar(self.plot_ind_dev_1, label='Mass (kg)', cax=self.ax_cbar, orientation='horizontal')


        return dens_min, dens_max



    def clearPlots(self):
        """ Clears all axes. """

        for ax_cost in self.axes_cost:
            ax_cost.cla()

        for ax_vel in self.axes_velocity:
            ax_vel.cla()


        self.ax_cbar.cla()



    def updateSliders(self, val):
        """ Update slider values. """

        # Get slider values
        self.dens_min = self.sl_ind_dev_1.val
        self.dens_max = self.sl_ind_dev_2.val

        # Make sure the sliders do not go beyond one another
        if self.dens_min > self.dens_max - SLIDER_EPSILON:
            self.sl_ind_dev_1.set_val(self.dens_max - SLIDER_EPSILON)

        if self.dens_max < self.dens_min + SLIDER_EPSILON:
            self.sl_ind_dev_2.set_val(self.dens_min + SLIDER_EPSILON)

        # Get slider values
        self.dens_min = self.sl_ind_dev_1.val
        self.dens_max = self.sl_ind_dev_2.val


        # Clear plots
        self.clearPlots()

        # Update plots with the given density range
        self.updatePlots()






if __name__ == "__main__":


    

    # dir_path_mir = "../MirfitPrepare/20160929_062945_mir/"
    dir_path_mir = "../MirfitPrepare/20161007_052346_mir/"
    # dir_path_mir = "../MirfitPrepare/20161007_052749_mir/"
    

    # Trajectory pickle file
    #traj_pickle_file = "20160929_062945_trajectory.pickle"
    traj_pickle_file = "20161007_052346_trajectory.pickle"
    # traj_pickle_file = "20161007_052749_trajectory.pickle"
    


    FitSimAnalyzer(dir_path_mir, traj_pickle_file)