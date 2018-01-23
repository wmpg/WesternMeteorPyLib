""" Tool for marking times of arrival of seismic/air waves in IRIS data. """

from __future__ import print_function, division, absolute_import

import os
import sys
import datetime

import obspy
import numpy as np
import scipy.signal
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from matplotlib.widgets import Slider


INPUT_DATA_FILE = 'data.txt'
OUTPUT_CSV = 'data_picks.csv'

from Fireballs.GetIRISData import readStationAndWaveformsListFile, butterworthBandpassFilter, \
    plotAllWaveforms
from Utils.Earth import greatCircleDistance
from Utils.PlotMap import GroundMap
from Utils.TrajConversions import datetime2JD



class WaveformPicker(object):
    def __init__(self, dir_path, v_sound, t0, data_list, lat_centre, lon_centre, waveform_window=600):
        """

        Arguments:
            data_list: [list]

        Keyword arguments:
            waveform_window: [int] Number of seconds for the wavefrom window.
        """

        self.dir_path = dir_path

        self.v_sound = v_sound
        self.t0 = t0

        self.data_list = data_list

        self.lat_centre = lat_centre
        self.lon_centre = lon_centre

        self.waveform_window = waveform_window


        self.current_station = 0
        self.current_wavefrom_raw = None
        self.current_wavefrom_delta = None

        # List of picks
        self.pick_list = []

        self.pick_group = 0

        # Define a list of colors for groups
        self.pick_group_colors = ['r', 'b', 'g', 'y', 'm']


        # Current station map handle
        self.current_station_scat = None

        # Station waveform marker handle
        self.current_station_all_marker = None

        # Picks on all waveform plot handle
        self.all_waves_picks_handle = None

        # Handle for pick text
        self.pick_text_handle = None

        # handle for pick marker on the wavefrom
        self.pick_wavefrom_handle = None


        # Default bandpass values
        self.bandpass_low_default = 2.0
        self.bandpass_high_default = 8.0

        # Flag indicating whether CTRL is pressed or not
        self.ctrl_pressed = False


        ### Sort stations by distance from source ###

        # Calculate distances of station from source
        self.source_dists = []

        for stat_entry in self.data_list:

            stat_lat, stat_lon = stat_entry[2], stat_entry[3]

            # Calculate the distance in kilometers
            dist = greatCircleDistance(np.radians(lat_centre), np.radians(lon_centre), \
                np.radians(stat_lat), np.radians(stat_lon))

            self.source_dists.append(dist)


        # Get sorted arguments
        dist_sorted_args = np.argsort(self.source_dists)

        # Sort the stations by distance
        self.data_list = [self.data_list[i] for i in dist_sorted_args]
        self.source_dists = [self.source_dists[i] for i in dist_sorted_args]


        #############################################



        # Init the plot framework
        self.initPlot()

        # Extract the list of station locations
        self.lat_list = [np.radians(entry[2]) for entry in data_list]
        self.lon_list = [np.radians(entry[3]) for entry in data_list]


        # Init ground map
        self.m = GroundMap(self.lat_list, self.lon_list, ax=self.ax_map)

        # Plot stations
        self.m.scatter(self.lat_list, self.lon_list, c='w', s=1)

        # Plot source location
        self.m.scatter([np.radians(lat_centre)], [np.radians(lon_centre)], marker='*', c='yellow')



        self.updatePlot()




    def initPlot(self):
        """ Initializes the plot framework. """


        ### Init the basic grid ###

        gs = gridspec.GridSpec(4, 3, width_ratios=[1, 1, 1], height_ratios=[10, 5, 1, 1])


        # All waveform axis
        self.ax_all_waves = plt.subplot(gs[0, 0:2])

        # Map axis
        self.ax_map = plt.subplot(gs[0, 2])

        # Waveform axis
        self.ax_wave = plt.subplot(gs[1, :])

        # Register a mouse press event on the waveform axis
        plt.gca().figure.canvas.mpl_connect('button_press_event', self.onWaveMousePress)

        # Init what happes when a keyboard key is pressed or released
        plt.gca().figure.canvas.mpl_connect('key_press_event', self.onKeyPress)
        plt.gca().figure.canvas.mpl_connect('key_release_event', self.onKeyRelease)

        # Previous button axis
        self.ax_prev_btn = plt.subplot(gs[2, 0])

        # Next button axis
        self.ax_next_btn = plt.subplot(gs[3, 0])

        # Bandpass options
        bandpass_gridspec = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[2:4, 1])
        self.ax_bandpass_low = plt.subplot(bandpass_gridspec[0])
        self.ax_bandpass_high = plt.subplot(bandpass_gridspec[1])
        self.ax_bandpass_button = plt.subplot(bandpass_gridspec[2])

        # Spectrogram button
        self.ax_specgram_btn = plt.subplot(bandpass_gridspec[3])
        

        # Pick list
        picks_gridspec = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[2:, 2])
        self.ax_picks = plt.subplot(picks_gridspec[:3])

        # Disable ticks and axis lines
        self.ax_picks.set_axis_off()

        # Export CSV button
        self.ax_export_csv_btn = plt.subplot(picks_gridspec[3])


        ############################


        # Create 'prev' button
        self.prev_btn = Button(self.ax_prev_btn, 'Previous')
        self.prev_btn.on_clicked(self.decrementStation)

        # Create 'next' button
        self.next_btn = Button(self.ax_next_btn, 'Next')
        self.next_btn.on_clicked(self.incrementStation)


        # Bandpass sliders and button
        self.bandpass_low_slider = Slider(self.ax_bandpass_low, 'Low:', 0.1, 5, \
            valinit=self.bandpass_low_default)
        self.bandpass_high_slider = Slider(self.ax_bandpass_high, 'High:', 3, 10, \
            valinit=self.bandpass_high_default, slidermin=self.bandpass_low_slider)
        
        self.bandpass_button = Button(self.ax_bandpass_button, 'Bandpass filter')
        self.bandpass_button.on_clicked(self.filterBandpass)


        # Spectrogram button
        self.specgram_btn = Button(self.ax_specgram_btn, 'Spectrogram of raw data')
        self.specgram_btn.on_clicked(self.showSpectrogram)


        # Export CSV button
        self.export_csv_btn = Button(self.ax_export_csv_btn, 'Export CSV: ' + OUTPUT_CSV)
        self.export_csv_btn.on_clicked(self.exportCSV)

        # Plot all waveforms
        plotAllWaveforms(self.dir_path, self.data_list, self.v_sound, self.t0, self.lat_centre, \
            self.lon_centre, ax=self.ax_all_waves, waveform_window=self.waveform_window)



    def onKeyPress(self, event):
        
        if event.key == 'control':
            self.ctrl_pressed = True


        elif event.key == '+':
            
            # Increment the pick group
            self.pick_group += 1

            self.updatePlot()


        elif event.key == '-':
            # Decrement the pick group

            if self.pick_group > 0:
                self.pick_group -= 1

            self.updatePlot()




    def onKeyRelease(self, event):

        if event.key == 'control':
            self.ctrl_pressed = False




    def addPick(self, pick_group, station_no, pick_time):
        """ Adds the pick to the list of picks. """


        self.pick_list.append([pick_group, station_no, pick_time])

        self.updatePickList()




    def removePick(self, station_no, pick_time_remove):
        """ Removes the pick from the list of picks with the closest time. """


        if len(self.pick_list):

            closest_pick_indx = None
            min_time_diff = np.inf

            # Go though all stations and find the pick with closest to the given time
            for i, entry in enumerate(self.pick_list):

                pick_grp, stat_no, pick_time = entry

                # Check if current station
                if stat_no == self.current_station:

                    time_diff = abs(pick_time - pick_time_remove)

                    # Store the minimum time difference
                    if time_diff < min_time_diff:

                        min_time_diff = time_diff
                        closest_pick_indx = i



            if closest_pick_indx is not None:
                
                # Remove pick on the given station closest the given time
                self.pick_list.pop(closest_pick_indx)


                self.updatePickList()




    def updatePickList(self):
        """ Updates the list of picks on the screen for the given station. """

        stations_with_picks = []
        all_pick_times = []

        for entry in self.pick_list:

            pick_grp, station_no, pick_time = entry

            stations_with_picks.append(station_no)
            all_pick_times.append(pick_time)


        # Remove old picks on all wavefrom plot
        if self.all_waves_picks_handle is not None:
            self.all_waves_picks_handle.remove()


        # Get distances of of pick stations
        dists_with_picks = [self.source_dists[stat_no] for stat_no in stations_with_picks]

        # Mark picks on the all waveform plot
        self.all_waves_picks_handle = self.ax_all_waves.scatter(dists_with_picks, all_pick_times, \
            marker='*', s=50, c='r')


        self.updatePlot()




    def updatePickTextAndWaveMarker(self):
        """ Updates the list of picks on the screen. """


        current_station_groups = []
        current_station_picks = []

        for entry in self.pick_list:

            pick_grp, station_no, pick_time = entry

            # Take picks taken on the current station
            if station_no == self.current_station:
                current_station_groups.append(pick_grp)
                current_station_picks.append(pick_time)


        # Remove old pick text
        if self.pick_text_handle is not None:
            self.pick_text_handle.remove()

        # Generate the pick string
        pick_txt_str  = 'Change group: +/-\n'
        pick_txt_str += 'Add/remove pick: CTRL + left/right click \n'
        pick_txt_str += '\n'
        pick_txt_str += 'Current group: {:5d}\n\n'.format(self.pick_group)
        pick_txt_str += 'Picks: Group, Time\n'
        pick_txt_str += '------\n'
        pick_txt_str += "\n".join(["{:5d},   {:.2f}".format(gr, pt) for gr, pt in zip(current_station_groups, \
            current_station_picks)])

        # Print picks on screen
        self.pick_text_handle = self.ax_picks.text(0, 1, pick_txt_str, va='top', fontsize=7)



        if len(current_station_picks) > 0:

            # Get a list of colors per groups
            color_list = [self.pick_group_colors[grp%len(self.pick_group_colors)] \
                for grp in current_station_groups]

            # Set pick marker on the current wavefrom
            self.ax_wave.scatter(current_station_picks, 
                [0]*len(current_station_picks), marker='*', c=color_list, s=50)

            # Plot group numbers above picks
            #self.pick_wavefrom_text_handles = []
            for c, grp, pt in zip(color_list, current_station_groups, current_station_picks):
                self.ax_wave.text(pt, 0, str(grp), color=c, ha='center', va='bottom')




    def onWaveMousePress(self, event):

        # Check if the mouse was pressed within the waveform axis
        if event.inaxes == self.ax_wave:

            # Check if CTRL is pressed
            if self.ctrl_pressed:

                pick_time = event.xdata

                # Check if left button was pressed
                if event.button == 1:

                    # Extract network and station code
                    net, station_code = self.data_list[self.current_station][:2]

                    print('Adding pick on station {:s} at {:.2f}'.format(net + ": " + station_code, \
                        pick_time))

                    self.addPick(self.pick_group, self.current_station, pick_time)


                # Check if right button was pressed
                elif event.button == 3:
                    print('Removing pick...')

                    self.removePick(self.current_station, pick_time)





    def incrementStation(self, event):
        """ Increments the current station index. """

        self.current_station += 1

        if self.current_station >= len(self.data_list):
            self.current_station = 0


        self.updatePlot()




    def decrementStation(self, event):
        """ Decrements the current station index. """

        self.current_station -= 1

        if self.current_station < 0:
            self.current_station = len(self.data_list) - 1


        self.updatePlot()



    def markCurrentStation(self):
        """ Mark the position of the current station on the map. """

        # Extract current station
        stat_entry = self.data_list[self.current_station]

        # Extract station coordinates
        stat_lat, stat_lon = stat_entry[2], stat_entry[3]


        if self.current_station_scat is None:

            # Mark the current station on the map
            self.current_station_scat = self.m.scatter([np.radians(stat_lat)], [np.radians(stat_lon)], s=20, \
                edgecolors='r', facecolors='none')

        else:

            # Calculate map coordinates
            stat_x, stat_y = self.m.m(stat_lon, stat_lat)

            # Set the new position
            self.current_station_scat.set_offsets([[stat_x], [stat_y]])




    def drawWaveform(self, waveform_data=None):
        """ Draws the current waveform from the current station in the wavefrom window. Custom wavefrom 
            can be given an drawn, which is used when bandpass filtering is performed. 

        """

        # Clear waveform axis
        self.ax_wave.cla()


        # Extract current station
        stat_entry = self.data_list[self.current_station]

        # Unpack the station entry
        net, station_code, stat_lat, stat_lon, stat_elev, station_name, mseed_file = stat_entry

        # Get the miniSEED file path
        mseed_file_path = os.path.join(self.dir_path, mseed_file)


        # Read the miniSEED file
        mseed = obspy.read(mseed_file_path)


        # Unpact miniSEED data
        delta = mseed[0].stats.delta
        start_time = mseed[0].stats.starttime
        end_time = mseed[0].stats.endtime


        # Check if the waveform data is already given or not
        if waveform_data is None:
            waveform_data = mseed[0].data

            # Store raw data for bookkeeping on first open
            self.current_wavefrom_raw = waveform_data


        # Convert the beginning and the end time to datetime objects
        start_datetime = start_time.datetime
        end_datetime = end_time.datetime

        self.current_wavefrom_delta = delta
        self.current_waveform_time = np.arange(0, (end_datetime - start_datetime).total_seconds() + delta, \
            delta)


        ### BANDPASS FILTERING ###

        # Init the butterworth bandpass filter
        butter_b, butter_a = butterworthBandpassFilter(self.bandpass_low_default, \
            self.bandpass_high_default, 1.0/delta, order=6)

        # Filter the data
        waveform_data = scipy.signal.filtfilt(butter_b, butter_a, waveform_data)

        ##########################


        # Construct time array, 0 is at start_datetime
        time_data = np.copy(self.current_waveform_time)

        # Cut the waveform data length to match the time data
        waveform_data = waveform_data[:len(time_data)]
        time_data = time_data[:len(waveform_data)]


        # Calculate the time of arrival assuming constant propagation with the given speed of sound
        t_arrival = self.source_dists[self.current_station]/(self.v_sound/1000) + self.t0

        # Calculate the limits of the plot to be within the given window limit
        time_win_min = t_arrival - self.waveform_window/2
        time_win_max = t_arrival + self.waveform_window/2


        # Plot the estimated time of arrival
        self.ax_wave.plot([t_arrival]*2, [np.min(waveform_data), np.max(waveform_data)], color='red', \
            alpha=0.5, zorder=3)


        # Plot the wavefrom
        self.ax_wave.plot(time_data, waveform_data, color='k', linewidth=0.2, zorder=3)


        # Set the time limits to be within the given window
        self.ax_wave.set_xlim(time_win_min, time_win_max)

        self.ax_wave.grid(color='#ADD8E6', linestyle='dashed', linewidth=0.5, alpha=0.5)


        # Add text with station label
        self.ax_wave.text(time_win_min, np.max(waveform_data), net + ": " + station_code \
            + ", {:d} km".format(int(self.source_dists[self.current_station])) , va='top', ha='left')




    def markStationWaveform(self):
        """ Mark the currently shown waveform in the plot of all waveform. """

        if self.current_station_all_marker is not None:
            self.current_station_all_marker.remove()


        # Calculate the position
        dist = self.source_dists[self.current_station]

        # Calcualte the time of arrival
        t_arrival = self.source_dists[self.current_station]/(self.v_sound/1000) + self.t0

        # Plot the marker
        self.current_station_all_marker = self.ax_all_waves.scatter(dist, t_arrival, marker='x', s=200, \
            linewidths=3, c='g', alpha=0.5)




    def showSpectrogram(self, event):
        """ Show the spectrogram of the waveform in the current window. """


        # Get time limits of the shown waveform
        x_min, x_max = self.ax_wave.get_xlim()

        # Extract the time and waveform
        crop_window = (self.current_waveform_time >= x_min) & (self.current_waveform_time <= x_max)
        wave_arr = self.current_wavefrom_raw[crop_window]


        ### Show the spectrogram ###
        
        fig = plt.figure()
        ax_spec = fig.add_subplot(111)

        ax_spec.specgram(wave_arr, Fs=1.0/self.current_wavefrom_delta, cmap=plt.cm.inferno)

        ax_spec.set_xlabel('Time (s)')
        ax_spec.set_ylabel('Frequency (Hz)')

        fig.show()

        ###



    def filterBandpass(self, event):
        """ Run bandpass filtering using values set on sliders. """

        # Get bandpass filter values
        bandpass_low = self.bandpass_low_slider.val
        bandpass_high = self.bandpass_high_slider.val

        # Init the butterworth bandpass filter
        butter_b, butter_a = butterworthBandpassFilter(bandpass_low, bandpass_high, \
            1.0/self.current_wavefrom_delta, order=6)

        # Filter the data
        waveform_data = scipy.signal.filtfilt(butter_b, butter_a, np.copy(self.current_wavefrom_raw))


        # Plot the updated waveform
        self.drawWaveform(waveform_data)




    def updatePlot(self):
        """ Update the plot after changes. """

        # Mark the position of the current station on the map
        self.markCurrentStation()

        # Plot the wavefrom from the current station
        self.drawWaveform()

        # Set an arrow pointing to the current station on the waveform
        self.markStationWaveform()

        # Update the pick list text and plot marker on the waveform
        self.updatePickTextAndWaveMarker()


        # Reset bandpass filter values to default
        self.bandpass_low_slider.set_val(self.bandpass_low_default)
        self.bandpass_high_slider.set_val(self.bandpass_high_default)

        plt.draw()
        plt.pause(0.001)




    def exportCSV(self, event):
        """ Save picks to a CSV file. """

        # Open the output CSV
        with open(os.path.join(self.dir_path, OUTPUT_CSV), 'w') as f:

            # Write the header
            f.write('Pick group, Network, Code, Lat, Lon, Elev, Pick JD\n')

            # Go through all picks
            for entry in self.pick_list:

                # Unpack pick data
                pick_group, station_no, pick_time = entry

                # Extract current station
                stat_entry = self.data_list[station_no]

                # Unpack the station entry
                net, station_code, stat_lat, stat_lon, stat_elev, station_name, mseed_file = stat_entry

                # Get the miniSEED file path
                mseed_file_path = os.path.join(self.dir_path, mseed_file)


                # Read the miniSEED file
                mseed = obspy.read(mseed_file_path)

                # Find datetime of the beginning of the file
                start_datetime = mseed[0].stats.starttime.datetime

                # Calculate Julian date of the pick time
                pick_jd = datetime2JD(start_datetime + datetime.timedelta(seconds=pick_time))


                # Write the CSV entry
                f.write("{:d}, {:s}, {:s}, {:.6f}, {:.6f}, {:.2f}, {:.8f}\n".format(pick_group, net, \
                    station_code, stat_lat, stat_lon, stat_elev, pick_jd))



        print('CSV written to:', OUTPUT_CSV)




if __name__ == "__main__":



    ### WAVEFORM DATA PARAMETERS ###
    ##########################################################################################################


    # Name of the folder where data files will be stored
    dir_path = '../Seismic data/2018-01-16 Michigan fireball'

    # Geo coordinates of the wave release centre
    lat_centre = 42.646767
    lon_centre = -83.925573

    # Speed of sound (m/s)
    v_sound = 310

    # Time offset of wave release from the reference time
    t0 = 0


    ##########################################################################################################


    # Load the station and waveform files list
    data_file_path = os.path.join(dir_path, INPUT_DATA_FILE)
    if os.path.isfile(data_file_path):
        
        data_list = readStationAndWaveformsListFile(data_file_path)

    else:
        print('Station and waveform data file not found! Download the waveform files first!')
        sys.exit()



    # Init the wavefrom picker
    WaveformPicker(dir_path, v_sound, t0, data_list, lat_centre, lon_centre)

    plt.tight_layout()

    #plt.waitforbuttonpress(timeout=-1)

    plt.show()