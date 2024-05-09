""" The script monitors for any changes in the .ecsv files in the specified directory and automatically
updates the trajectory solution which is shown in a plot. """

import sys
import time
import os
import hashlib
import base64

import numpy as np

from watchdog.observers.polling import PollingObserver
from watchdog.events import PatternMatchingEventHandler

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import cm

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QHBoxLayout
from PyQt5.QtGui import QPixmap, QIcon, QColor, QPainter
from PyQt5.QtCore import Qt

from wmpl.Formats.ECSV import loadECSVs
from wmpl.Formats.GenericFunctions import solveTrajectoryGeneric, addSolverOptions
from wmpl.Utils.Pickling import savePickle



class DirectoryMonitor(PatternMatchingEventHandler):
    def __init__(self, callback, **kwargs):
        super().__init__(**kwargs)
        self.callback = callback

    def on_created(self, event):
        print(f"Detected creation: {event.src_path}")
        self.callback(event)

    def on_modified(self, event):
        if event.is_directory:
            return
        self.callback(event)

    def on_moved(self, event):
        """ Extended event handler for file moves. """
        print(f"Detected move from {event.src_path} to {event.dest_path}")
        self.callback(event)

    def on_deleted(self, event):
        print(f"Detected deletion: {event.src_path}")
        self.callback(event)



class FileMonitorApp(QMainWindow):
    def __init__(self, dir_path, solver_kwargs):
        super().__init__()

        self.dir_path = dir_path
        self.last_update_time = 0

        self.traj = None

        # Save the trajectory solver keyword arguments
        self.solver_kwargs = solver_kwargs

        # Initialize the checksum dictionary
        self.file_checksums = {}

        # Initialize the points to highlight on the plots
        self.highlighted_points = []

        # Plotted points
        self.plotted_points = {}

        self.initUI()
        self.initialRun()
        self.initObserver()


    def initUI(self):

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)
        layout.setSpacing(10)  # Adds spacing between widgets in the layout
        layout.setContentsMargins(10, 10, 10, 10)  # Adds margins around the entire layout (left, top, right, bottom)

        # Set the icon
        icon_data = b"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"
        pixmap = QPixmap()
        pixmap.loadFromData(base64.b64decode(icon_data))
        self.setWindowIcon(QIcon(pixmap))

        # Status bar setup
        self.status_bar = QWidget()
        self.status_bar.setMaximumHeight(30)  # Limit the maximum height of the status bar
        self.status_layout = QHBoxLayout(self.status_bar)
        self.status_layout.setContentsMargins(0, 0, 0, 0)  # No additional margins needed within the status bar
        self.status_label = QLabel("Ready")
        self.status_light = QLabel()

        # Initial light color (red by default, green when ready)
        self.setStatus('gray', "Waiting for data...")

        self.status_layout.addWidget(self.status_light)
        self.status_layout.addWidget(self.status_label)
        self.status_layout.addStretch(1)
        layout.addWidget(self.status_bar)

        # Add a canvas for the plot
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

        self.canvas.draw()
        QApplication.processEvents()

    def initObserver(self):
        
        # Poll the directory for changes every 1 second
        self.observer = PollingObserver(timeout=1.0)

        # Create the event handler
        event_handler = DirectoryMonitor(self.onModified, patterns=["*.ecsv"], ignore_directories=False)
        self.observer.schedule(event_handler, self.dir_path, recursive=True)
        self.observer.start()

    def initialRun(self):

        # Find all .ecsv files at startup and plot their trajectories, ignoring "REJECT" directories
        file_paths, self.file_checksums = self.findEcsvFiles(self.dir_path)

        if file_paths:
            self.computeTrajectory(file_paths)

    
    def findEcsvFiles(self, directory):

        file_paths = []
        file_checksums = {}

        for dirpath, dirnames, files in os.walk(directory):

            # Skip directories with "REJECT" in the name
            dirnames[:] = [d for d in dirnames if "REJECT" not in d.upper()]

            for f in files:
                if f.endswith('.ecsv'):

                    full_path = os.path.join(dirpath, f)
                    file_paths.append(full_path)

                    # Compute the checksum
                    with open(full_path, "rb") as file:
                        file_contents = file.read()
                        checksum = hashlib.md5(file_contents).hexdigest()
                        file_checksums[full_path] = checksum

        return file_paths, file_checksums
    

    def setStatus(self, color, text):

        # Create a pixmap of a circle with the desired color
        pixmap = QPixmap(16, 16)
        pixmap.fill(QColor('transparent'))
        painter = QPainter(pixmap)
        painter.setBrush(QColor(color))
        painter.setPen(QColor('black'))
        painter.drawEllipse(0, 0, 16, 16)
        painter.end()
        self.status_light.setPixmap(pixmap)

        self.status_label.setText(text)


    def onModified(self, event):

        current_time = time.time()
        max_update_rate = 1.0 # seconds


        needs_update = False

        # Check if the event is move/rename
        if event.event_type == 'moved':

            if "_REJECT" in os.path.basename(event.dest_path).upper():
                print(f"Directory marked as REJECT: {event.dest_path}")
                return None
            
            elif "_REJECT" in os.path.basename(event.src_path).upper():
                print(f"Directory unmarked as REJECT: {event.src_path}")
                needs_update = True

        if event.event_type == 'deleted':
            print(f"Item deleted: {event.src_path}")
            needs_update = True

        if event.src_path.endswith('.ecsv') or needs_update:
            if current_time - self.last_update_time >= max_update_rate:
            
                file_paths, new_checksums = self.findEcsvFiles(self.dir_path)

                # Check for changes using checksums
                unchanged = all(self.file_checksums.get(path) == checksum for path, checksum in new_checksums.items() if path in self.file_checksums)

                # If the file names used in the trajectory computation have changed, recompute the trajectory
                # Check that the keys in the new and old checksums are the same
                files_changed = set(self.file_checksums.keys()) != set(new_checksums.keys())

                
                if not unchanged or not self.file_checksums or files_changed:
                    
                    # Update stored checksums
                    self.file_checksums = new_checksums

                    print(f'Update triggered by modification in: {event.src_path}')

                    self.computeTrajectory(file_paths)
                    self.updatePlot()

                    self.last_update_time = current_time

                else:
                    print("No changes detected in the .ecsv files since the last computation.")


    def computeTrajectory(self, ecsv_paths):
        """ Compute the trajectory solution for the given .ecsv files. """

        # Indicate that the trajectory is being computed
        self.setStatus('red', "Computing trajectory...")

        QApplication.processEvents()

        print(f'Computing trajectory for {len(ecsv_paths)} files...')
        for file_path in ecsv_paths:
            print(f'Processing file: {os.path.basename(file_path)}')

        # Load the observations into container objects
        jdt_ref, meteor_list = loadECSVs(ecsv_paths)

        # Check that there are more than 2 ECSV files given
        if len(ecsv_paths) < 2:
            self.setStatus('gray', "At least two stations are needed for trajectory estimation!")
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

        
        # Save the report and the pickle file
        savePickle(self.traj, self.traj.output_dir, self.traj.file_name + '_self.trajectory.pickle')

        # Save self.trajectory report with original points
        self.traj.saveReport(self.traj.output_dir, self.traj.file_name + '_report.txt', \
            uncertainties=None, verbose=False)
        

        self.updatePlot()

        self.setStatus('green', "Ready!")


    def onClick(self, event):
        """ Mark the clicked point on all plots. """

        all_axes = [
            # Ax handle, y_type, x_type
            [self.ax_res, "ht", "res"],
            [self.ax_mag, "ht", "mag"],
            [self.ax_lag, "time", "lag"],
            [self.ax_vel, "time", "vel"]
            ]

        if event.inaxes is None:
            return

        xdata = event.xdata
        ydata = event.ydata

        min_dist = float('inf')
        closest_point = None
        closest_label = None

        # Check all artists in the axes
        for ax, y_type, x_type in all_axes:

            # Normalize the x and y scales using the data limits
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            aspect = (xlim[1]-xlim[0])/(ylim[1]-ylim[0])

            # Handle lines
            for line in ax.get_lines():

                xd, yd = line.get_xdata(), line.get_ydata()
                label = line.get_label()

                for x, y in zip(xd, yd):

                    dist = np.sqrt((x - xdata)**2 + (aspect*(y - ydata))**2)

                    if dist < min_dist:
                        min_dist = dist
                        closest_point = (y_type, x, y)
                        closest_label = label

            # Handle scatter plots
            for coll in ax.collections:
                
                # Get the offsets which are the points in the scatter
                offsets = coll.get_offsets()

                for x, y in offsets:
                    dist = np.sqrt((x - xdata)**2 + (aspect*(y - ydata))**2)

                    if dist < min_dist:
                        min_dist = dist
                        closest_point = (y_type, x, y)
                        closest_label = coll.get_label()

        # Clear previously highlighted points
        for point in self.highlighted_points:
            point.remove()
        self.highlighted_points.clear()

        # Highlight the new closest point in all relevant plots
        if closest_point and closest_label:

            for ax, y_type, x_type in all_axes:

                pt_y_type, x_coord, y_coord = closest_point

                # Get the time and height arrays for the station
                time_data, ht_data = self.plotted_points["time_ht_mapping"][closest_label]

                # Get the index of the plotted point given the Y type
                idx = 0
                if pt_y_type == "ht":
                    idx = np.argmin(np.abs(ht_data - y_coord))
                elif pt_y_type == "time":
                    idx = np.argmin(np.abs(time_data - y_coord))

                # If y_type is different, look up the Y value in the trajectory data
                if y_type != pt_y_type:

                    # Look up the Y value in the required type
                    if y_type == "ht":
                        y_coord = ht_data[idx]
                    elif y_type == "time":
                        y_coord = time_data[idx]

                # Get the X value in the required type using the index
                if x_type == "res":
                    
                    # Find the index of the closest point in the residual plot
                    plot_idx = np.argmin(np.abs(self.plotted_points["res"][closest_label][1] - y_coord))
                    x_coord = self.plotted_points["res"][closest_label][0][plot_idx]

                elif x_type == "mag":
                    if closest_label in self.plotted_points["mag"]:
                        plot_idx = np.argmin(np.abs(self.plotted_points["mag"][closest_label][1] - y_coord))
                        x_coord = self.plotted_points["mag"][closest_label][0][plot_idx]
                    else:
                        continue

                elif x_type == "lag":
                    plot_idx = np.argmin(np.abs(self.plotted_points["lag"][closest_label][1] - y_coord))
                    x_coord = self.plotted_points["lag"][closest_label][0][plot_idx]

                elif x_type == "vel":
                    plot_idx = np.argmin(np.abs(self.plotted_points["vel"][closest_label][1] - y_coord))
                    x_coord = self.plotted_points["vel"][closest_label][0][plot_idx]
            
                # Plot the point
                scatter = ax.scatter(x_coord, y_coord, zorder=5, s=100, edgecolor='r', facecolor='none')
                self.highlighted_points.append(scatter)

            self.canvas.draw()


    def updatePlot(self):

        # Clear all axes
        self.ax_res.clear()
        self.ax_mag.clear()
        self.ax_lag.clear()
        self.ax_vel.clear()

        # Clear highlighted points
        for point in self.highlighted_points:
            point.remove()
        self.highlighted_points.clear()

        # Reset the plotted points
        self.plotted_points = {
            "res": {},
            "mag": {},
            "lag": {},
            "vel": {},
            "time_ht_mapping": {}
        }

        if self.traj is not None:

            # Add time-height mapping for each station
            for obs in self.traj.observations:
                self.plotted_points["time_ht_mapping"][str(obs.station_id)] = [obs.time_data, 
                                                                               obs.model_ht/1000
                                                                               ]

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
                    s=10*size_multiplier, label=str(obs.station_id), zorder=3)
                
                # Add data to plotted points
                self.plotted_points['res'][str(obs.station_id)] = [tot_res, obs.meas_ht/1000]

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
                        
                        # Add data to plotted points
                        self.plotted_points['mag'][str(obs.station_id)] = [used_magnitudes, used_heights/1000]

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
                
                # Add data to plotted points
                self.plotted_points['lag'][str(obs.station_id)] = [used_lag, used_times]


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
                    c=colors[i].reshape(1,-1), alpha=alpha, label=str(obs.station_id), zorder=3)
                
                # Add data to plotted points
                self.plotted_points['vel'][str(obs.station_id)] = [obs.velocities[1:]/1000, obs.time_data[1:]]

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

        # Connect the click event
        self.figure.canvas.mpl_connect('button_press_event', self.onClick)

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
