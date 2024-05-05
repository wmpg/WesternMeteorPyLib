""" The script monitors for any changes in the .ecsv files in the specified directory and automatically
updates the trajectory solution which is shown in a plot. """

import sys
import time
import os

import numpy as np

from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import cm

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QDesktopWidget
from PyQt5.QtCore import Qt

from wmpl.Formats.ECSV import loadECSVs
from wmpl.Formats.GenericFunctions import solveTrajectoryGeneric, addSolverOptions

class FileMonitorApp(QMainWindow):
    def __init__(self, dir_path, solver_kwargs):
        super().__init__()

        self.dir_path = dir_path
        self.last_update_time = 0

        self.traj = None

        # Save the trajectory solver keyword arguments
        self.solver_kwargs = solver_kwargs

        self.initUI()
        self.initialRun()
        self.initObserver()


    def initUI(self):

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        self.figure, ((self.ax_res, self.ax_mag), (self.ax_lag, self.ax_vel)) = plt.subplots(nrows=2, ncols=2, dpi=150, figsize=(10, 5))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Initial window positioning (temporary values)
        screen = QApplication.primaryScreen().geometry()
        initial_width = screen.width() // 2  # Half the screen width
        initial_height = screen.height()*0.9     # Full screen height (minus the taskbar)
        initial_left = screen.width() - initial_width  # Right aligned

        self.setGeometry(int(initial_left), int(50), int(initial_width), int(initial_height))
        self.setWindowTitle('Trajectory Plotter')
        self.show()

    def initObserver(self):

        self.observer = Observer()
        event_handler = PatternMatchingEventHandler(patterns=["*.ecsv"], ignore_directories=True)
        event_handler.on_modified = self.onModified
        self.observer.schedule(event_handler, self.dir_path, recursive=True)
        self.observer.start()

    def initialRun(self):

        # Find all .ecsv files at startup and plot their trajectories, ignoring "REJECT" directories
        file_paths = self.findEcsvFiles(self.dir_path)

        if file_paths:
            self.computeTrajectory(file_paths)

    def findEcsvFiles(self, directory):
        file_paths = []
        for dirpath, dirnames, files in os.walk(directory):

            # Skip directories with "REJECT" in the name
            dirnames[:] = [d for d in dirnames if "REJECT" not in d.upper()]
            file_paths.extend([os.path.join(dirpath, f) for f in files if f.endswith('.ecsv')])

        return file_paths

    def onModified(self, event):

        current_time = time.time()

        if "REJECT" in os.path.dirname(event.src_path):

            print(f'Ignoring processing for file in "REJECT" directory: {event.src_path}')
            
            # Skip processing for files in "REJECT" directories
            return None

        # Limit the update rate
        max_update_rate = 0.5 # seconds

        if event.src_path.endswith('.ecsv') and (current_time - self.last_update_time >= max_update_rate):

            file_paths = self.findEcsvFiles(self.dir_path)
            
            print(f'Update triggered by modification in: {event.src_path}')

            self.computeTrajectory(file_paths)

            self.updatePlot()

            self.last_update_time = current_time

    def computeTrajectory(self, ecsv_paths):
        """ Compute the trajectory solution for the given .ecsv files. """

        print(f'Computing trajectory for {len(ecsv_paths)} files...')
        for file_path in ecsv_paths:
            print(f'Processing file: {os.path.basename(file_path)}')

        # Load the observations into container objects
        jdt_ref, meteor_list = loadECSVs(ecsv_paths)

        # Check that there are more than 2 ECSV files given
        if len(ecsv_paths) < 2:
            print("At least 2 files are needed for trajectory estimation!")
            return False
        
        # Unpack the kwargs into an object
        class Kwargs:
            pass

        kwargs = Kwargs()
        for key, value in self.solver_kwargs.items():
            setattr(kwargs, key, value)

        max_toffset = None
        if kwargs.maxtoffset:
            max_toffset = kwargs.maxtoffset[0]

        vinitht = None
        if kwargs.vinitht:
            vinitht = kwargs.vinitht[0]
        
        # Solve the trajectory (MC always disabled!)
        self.traj = solveTrajectoryGeneric(jdt_ref, meteor_list, self.dir_path, solver=kwargs.solver, \
            max_toffset=max_toffset, monte_carlo=False, save_results=False, \
            geometric_uncert=kwargs.uncertgeom, gravity_correction=(not kwargs.disablegravity), 
            gravity_factor=kwargs.gfact,
            plot_all_spatial_residuals=False, plot_file_type=kwargs.imgformat, \
            show_plots=False, v_init_part=kwargs.velpart, v_init_ht=vinitht, \
            show_jacchia=kwargs.jacchia,
            estimate_timing_vel=(False if kwargs.notimefit is None else kwargs.notimefit), \
            fixed_times=kwargs.fixedtimes, mc_noise_std=kwargs.mcstd)
        

        self.updatePlot()


    def updatePlot(self):

        # Clear all axes
        self.ax_res.clear()
        self.ax_mag.clear()
        self.ax_lag.clear()
        self.ax_vel.clear()

        if self.traj is not None:

            print("Updating plot...")

            # marker type, size multiplier
            markers = [
            ['x', 2 ],
            ['+', 8 ],
            ['o', 1 ],
            ['s', 1 ],
            ['d', 1 ],
            ['v', 1 ],
            ['*', 1.5 ],
            ]
            
            # Plot the trajectory fit residuals
            for i, obs in enumerate(sorted(self.traj.observations, key=lambda x:x.rbeg_ele, reverse=True)):

                marker, size_multiplier = markers[i%len(markers)]

                # Calculate root mean square of the total residuals
                total_res_rms = np.sqrt(obs.v_res_rms**2 + obs.h_res_rms**2)

                # Compute total residuals, take the signs from vertical residuals
                tot_res = np.sign(obs.v_residuals)*np.hypot(obs.v_residuals, obs.h_residuals)

                # Plot total residuals
                self.ax_res.scatter(tot_res, obs.meas_ht/1000, marker=marker, \
                    s=10*size_multiplier, label='{:s}, RMSD = {:.2f} m'.format(str(obs.station_id), \
                    total_res_rms), zorder=3)

                # Mark ignored points
                if np.any(obs.ignore_list):

                    ignored_ht = obs.model_ht[obs.ignore_list > 0]
                    ignored_tot_res = np.sign(obs.v_residuals[obs.ignore_list > 0])\
                        *np.hypot(obs.v_residuals[obs.ignore_list > 0], obs.h_residuals[obs.ignore_list > 0])


                    self.ax_res.scatter(ignored_tot_res, ignored_ht/1000, facecolors='none', edgecolors='k', \
                        marker='o', zorder=3, s=20)
                    
            self.ax_res.set_xlabel('Total Residuals (m)')
            self.ax_res.set_ylabel('Height (km)')
            self.ax_res.legend()
            self.ax_res.grid(True)

            # Set the residual limits to +/-10m if they are smaller than that
            res_lim = 10
            if np.abs(self.ax_res.get_xlim()).max() < res_lim:
                self.ax_res.set_xlim(-res_lim, res_lim)


            # Plot the absolute magnitude vs height
            first_ignored_plot = True
            if np.any([obs.absolute_magnitudes is not None for obs in self.traj.observations]):

                # Go through all observations
                for obs in sorted(self.traj.observations, key=lambda x: x.rbeg_ele, reverse=True):

                    # Check if the absolute magnitude was given
                    if obs.absolute_magnitudes is not None:

                        # Filter out None absolute magnitudes
                        filter_mask = np.array([abs_mag is not None for abs_mag in obs.absolute_magnitudes])

                        # Extract data that is not ignored
                        used_heights = obs.model_ht[filter_mask & (obs.ignore_list == 0)]
                        used_magnitudes = obs.absolute_magnitudes[filter_mask & (obs.ignore_list == 0)]

                        # Filter out magnitudes fainter than mag 8
                        mag_mask = np.array([abs_mag < 8 for abs_mag in used_magnitudes])
                        
                        # Avoid crash if no magnitudes exceed the threshold
                        if np.any(mag_mask):
                            used_heights = used_heights[mag_mask]
                            used_magnitudes = used_magnitudes[mag_mask]

                        else:
                            continue

                        plt_handle = self.ax_mag.plot(used_magnitudes, used_heights/1000, marker='x', \
                            label=str(obs.station_id), zorder=3)

                        # Mark ignored absolute magnitudes
                        if np.any(obs.ignore_list):

                            # Extract data that is ignored
                            ignored_heights = obs.model_ht[filter_mask & (obs.ignore_list > 0)]
                            ignored_magnitudes = obs.absolute_magnitudes[filter_mask & (obs.ignore_list > 0)]

                            self.ax_mag.scatter(ignored_magnitudes, ignored_heights/1000, facecolors='k', \
                                edgecolors=plt_handle[0].get_color(), marker='o', s=8, zorder=4)


                self.ax_mag.set_xlabel('Absolute magnitude')
                self.ax_mag.invert_xaxis()

                # Set the same Y limits as the residuals plot
                self.ax_mag.set_ylim(self.ax_res.get_ylim())

                self.ax_mag.legend()
                self.ax_mag.grid(True)


            


            # Generate a list of colors to use for markers
            colors = cm.viridis(np.linspace(0, 0.8, len(self.traj.observations)))

            # Only use one type of markers if there are not a lot of stations
            plot_markers = ['x']

            # Keep colors non-transparent if there are not a lot of stations
            alpha = 1.0


            # If there are more than 5 stations, interleave the colors with another colormap and change up
            #   markers
            if len(self.traj.observations) > 5:
                colors_alt = cm.inferno(np.linspace(0, 0.8, len(self.traj.observations)))
                for i in range(len(self.traj.observations)):
                    if i%2 == 1:
                        colors[i] = colors_alt[i]

                plot_markers.append("+")

                # Add transparency for more stations
                alpha = 0.75


            # Sort observations by first height to preserve color linearity
            obs_ht_sorted = sorted(self.traj.observations, key=lambda x: x.model_ht[0])

            # Plot the lag
            for i, obs in enumerate(obs_ht_sorted):

                # Extract lag points that were not ignored
                used_times = obs.time_data[obs.ignore_list == 0]
                used_lag = obs.lag[obs.ignore_list == 0]

                # Choose the marker
                marker = plot_markers[i%len(plot_markers)]

                # Plot the lag
                plt_handle = self.ax_lag.plot(used_lag, used_times, marker=marker, label=str(obs.station_id), 
                    zorder=3, markersize=3, color=colors[i], alpha=alpha)


                # Plot ignored lag points
                if np.any(obs.ignore_list):

                    ignored_times = obs.time_data[obs.ignore_list > 0]
                    ignored_lag = obs.lag[obs.ignore_list > 0]

                    self.ax_lag.scatter(ignored_lag, ignored_times, facecolors='k', edgecolors=plt_handle[0].get_color(), 
                        marker='o', s=8, zorder=4, label='{:s} ignored points'.format(str(obs.station_id)))
                    

            self.ax_lag.set_xlabel('Lag (m)')
            self.ax_lag.set_ylabel('Time (s)')
            self.ax_lag.legend()
            self.ax_lag.grid(True)
            self.ax_lag.invert_yaxis()



            # Possible markers for velocity
            vel_markers = ['x', '+', '.', '2']

            vel_max = -np.inf
            vel_min = np.inf
            
            first_ignored_plot = True


            # Plot velocities from each observed site
            for i, obs in enumerate(obs_ht_sorted):

                # Mark ignored velocities
                if np.any(obs.ignore_list):

                    # Extract data that is not ignored
                    ignored_times = obs.time_data[1:][obs.ignore_list[1:] > 0]
                    ignored_velocities = obs.velocities[1:][obs.ignore_list[1:] > 0]

                    # Set the label only for the first occurence
                    if first_ignored_plot:

                        self.ax_vel.scatter(ignored_velocities/1000, ignored_times, facecolors='none', edgecolors='k', \
                            zorder=4, s=30, label='Ignored points')

                        first_ignored_plot = False

                    else:
                        self.ax_vel.scatter(ignored_velocities/1000, ignored_times, facecolors='none', edgecolors='k', \
                            zorder=4, s=30)


                # Plot all point to point velocities
                self.ax_vel.scatter(obs.velocities[1:]/1000, obs.time_data[1:], marker=vel_markers[i%len(vel_markers)], 
                    c=colors[i].reshape(1,-1), alpha=alpha, label='{:s}'.format(str(obs.station_id)), zorder=3)


                # Determine the max/min velocity and height, as this is needed for plotting both height/time axes
                vel_max = max(np.max(obs.velocities[1:]/1000), vel_max)
                vel_min = min(np.min(obs.velocities[1:]/1000), vel_min)


            self.ax_vel.set_xlabel('Velocity (km/s)')

            self.ax_vel.legend()
            self.ax_vel.grid()

            # Set absolute limits for velocities
            vel_min = max(vel_min, -20)
            vel_max = min(vel_max, 100)

            # Set velocity limits to +/- 3 km/s
            self.ax_vel.set_xlim([vel_min - 3, vel_max + 3])

            # Set time limits to be the same as the lag plot
            self.ax_vel.set_ylim(self.ax_lag.get_ylim())


        # Set a tight layout
        self.figure.tight_layout()

        self.canvas.draw()



    def run(self):

        try:
            sys.exit(app.exec_())
        except KeyboardInterrupt:
            self.observer.stop()
            self.observer.join()


if __name__ == "__main__":

    import argparse

    ### Parse command line arguments ###

    arg_parser = argparse.ArgumentParser(description="Automatically computes the trajectory solution given .ecsv files and shows the trajectory solution in a window. The trajectory solution is kept updated as the .ecsv files are modified.")

    arg_parser.add_argument("dir_path", type=str, help="Path to the directory to watch for .ecsv files.")

    # Add other solver options
    arg_parser = addSolverOptions(arg_parser, skip_velpart=False)

    cml_args = arg_parser.parse_args()

    ### ###

    app = QApplication(sys.argv)
    file_monitor = FileMonitorApp(cml_args.dir_path, cml_args.__dict__)
    file_monitor.run()
