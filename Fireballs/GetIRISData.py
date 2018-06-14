""" Functions for fetching USarray waveforms. """


from __future__ import print_function, division, absolute_import

import os
import sys
import datetime
import urllib

import numpy as np
import scipy.signal
import obspy

from Utils.Earth import greatCircleDistance
from Utils.OSTools import mkdirP
from Utils.PlotMap import GroundMap
from Utils.Math import subsampleAverage

DATA_FILE = 'data.txt'


def getIRISStations(lat_centre, lon_centre, deg_radius, start_date, end_date, network='*', channel='BDF', 
    orfeus=False):
    """ Retrieves seismic stations from the IRIS web service in a radius around given geographical position 
        which were active during the specified range of dates.

    Arguments:
        lat_centre: [float] Latitude of the query centre (+N, degrees).
        lon_centre: [float] Longitude of the query centre (+E, degrees).
        deg_radius: [float] Query radius (degrees).
        start_date: [str] First date when a station was recording (YYYY-MM-DD format).
        end_date: [str] Final date when a station was recording (YYYY-MM-DD format).
        
    Keyword arguments:
        network: [str] Filter retrieved stations by the given seismic network code. * by default, meaning
            stations for all networks will be retrieved.
        channel: [str] Seismograph channel. BDF by default.
        orfeus: [bool] If True, the European ORFEUS website will be queried instead of IRIS.

    Return:
        [list] A list of stations and their parameters.
    """

    # Use the European ORFEUS data access site
    if orfeus:

        # Construct ORFEUS URL
        iris_url = ("http://www.orfeus-eu.org/fdsnws/station/1/query?network={:s}&latitude={:.3f}&longitude={:.3f}" \
            "&maxradius={:.3f}&start={:s}&end={:s}&channel={:s}&format=text" \
            "&includerestricted=false&nodata=404").format(network, lat_centre, lon_centre, deg_radius, \
            start_date, end_date, channel)

    else:
        # Construct IRIS URL
        iris_url = ("http://service.iris.edu/fdsnws/station/1/query?net={:s}&latitude={:.3f}&longitude={:.3f}" \
            "&maxradius={:.3f}&start={:s}&end={:s}&cha={:s}&nodata=404&format=text" \
            "&matchtimeseries=true").format(network, lat_centre, lon_centre, deg_radius, start_date, \
            end_date, channel)

    # Retrieve station list
    stations_txt = urllib.urlopen(iris_url).read()

    station_list = []

    # Return an empty list if no stations were retrieved
    if not stations_txt:
        return station_list

    # Parse the stations
    for entry in stations_txt.split('\n')[1:]:

        entry = entry.split('|')

        # Skip empty rows
        if len(entry) != 8:
            continue
        
        # Unpack the line
        network, station_code, lat, lon, elev, station_name, start_work, end_work = entry

        station_list.append([network, station_code, float(lat), float(lon), float(elev), station_name])


    return station_list




def getIRISWaveformFiles(network, station_code, start_datetime, end_datetime, dir_path='.', channel='BDF', \
    orfeus=False):
    """ Download weaveform files from the IRIS site. 
    
    Arguments:
        network: [str] Network code.
        station_code: [str] Station code.
        start_datetime: [datetime object] Datetime of the beginning of the data chunk to retrieve.
        end_datetime: [datetime object] Datetime of the end of the data chunk to retrieve. This cannot be
            more than 2 hours of data (this is not an IRIS limitation, but a limitation set here, so nobody
            clogs the IRIS servers with unreasonable requests).
        
    Keyword arguments:
        dir_path: [str] Full path to location where the miniSEED files will be saved.
        channel: [str] Seismograph channel. BDF by default.
        orfeus: [bool] If True, the European ORFEUS website will be queried instead of IRIS.

    Return:
        mseed_file_path: [str] Path to the downloaded miniSEED data file.
    """

    # Make strings from datetime objects
    sd = start_datetime
    start_time = "{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d}.{:03d}".format(sd.year, sd.month, sd.day, \
        sd.hour, sd.minute, sd.second, sd.microsecond//1000)

    ed = end_datetime
    end_time = "{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d}.{:03d}".format(ed.year, ed.month, ed.day, \
        ed.hour, ed.minute, ed.second, ed.microsecond//1000)

    #start_time = "2010-02-27T06:30:00.000"
    #end_time = "2010-02-27T10:30:00.000"

    # Check that no more then 2 hours of data is requested
    if (end_datetime - start_datetime).total_seconds() > 2*3600:
        print('No more than 2 hours of waveform data can be requested!')
        return None


    # Use the European ORFEUS data access site if specified
    if orfeus:

        # Construct ORFEUS URL
        iris_url = ("http://www.orfeus-eu.org/fdsnws/dataselect/1/query?network={:s}&station={:s}" \
                "&channel={:s}&start={:s}&end={:s}").format(network, station_code, channel, start_time, \
                end_time)

    else:
        
        # Construct IRIS URL
        iris_url = ("http://service.iris.edu/fdsnws/dataselect/1/query?net={:s}&sta={:s}&cha={:s}" \
                "&start={:s}&end={:s}").format(network, station_code, channel, start_time, end_time)


    # Construct a file name
    mseed_file = network + '_' + station_code + '_{:s}_'.format(channel) + start_time.replace(':', '.') \
        + '_' + end_time.replace(':', '.') + '.mseed'


    mseed_file_path = os.path.join(dir_path, mseed_file)

    # Get the miniSEED file
    iris_file = urllib.urlopen(iris_url)
    if iris_file:

        # Save the file if it has any length
        with open(mseed_file_path,'wb') as f:
            try:
                f.write(iris_file.read())
            except urllib.URLError:
                print('Connection error! Could not download the waveform!')


        return mseed_file

    else:
        return None



def writeStationAndWaveformsListFile(data_list, file_path):
    """ Writes the list of stations and miniSEED file names to disk. """

    with open(file_path, 'w') as f:
        
        for entry in data_list:
        
            #network, station_code, stat_lat, stat_lon, stat_elev, station_name, mseed_file = entry

            f.write('{:s}|{:s}|{:.6f}|{:.6f}|{:.6f}|{:s}|{:s}\n'.format(*entry))



def readStationAndWaveformsListFile(file_path):
    """ Reads the list of stations and miniSEED file names to disk. """

    if os.path.isfile(file_path):

        with open(file_path) as f:

            data_list = []

            for line in f:

                if not line:
                    continue

                line = line.replace('\n', '')
                line = line.split('|')

                network, station_code, stat_lat, stat_lon, stat_elev, station_name, mseed_file = line
                stat_lat, stat_lon, stat_elev = list(map(float, [stat_lat, stat_lon, stat_elev]))

                data_list.append([network, station_code, stat_lat, stat_lon, stat_elev, station_name, \
                    mseed_file])


            return data_list

    else:
        return []

        





def getAllWaveformFiles(lat_centre, lon_centre, deg_radius, start_datetime, end_datetime, network='*', \
    channel='BDF', dir_path='.', orfeus=False):
    """ Retrieves and saves waveforms as miniSEED files of all seismic stations from the IRIS web service in 
        a radius around given geographical position and for the given range of times.

    Arguments:
        lat_centre: [float] Latitude of the query centre (+N, degrees).
        lon_centre: [float] Longitude of the query centre (+E, degrees).
        deg_radius: [float] Query radius (degrees).
        start_datetime: [datetime object] Datetime of the beginning of the data chunk to retrieve.
        end_datetime: [datetime object] Datetime of the end of the data chunk to retrieve. This cannot be
            more than 2 hours of data (this is not an IRIS limitation, but a limitation set here, so nobody
            clogs the IRIS servers with unreasonable requests).
        
    Keyword arguments:
        network: [str] Filter retrieved stations by the given seismic network code. * by default, meaning
            stations for all networks will be retrieved.
        dir_path: [str] Full path to location where the miniSEED files will be saved.
        orfeus: [bool] If True, the European ORFEUS website will be queried instead of IRIS.

    """


    # Station activity date range
    sd = start_datetime
    start_date = "{:04d}-{:02d}-{:02d}".format(sd.year, sd.month, sd.day)
    ed = sd + datetime.timedelta(days=1)
    end_date = "{:04d}-{:02d}-{:02d}".format(ed.year, ed.month, ed.day)


    # Make the data directory
    mkdirP(dir_path)


    # Get a list of stations active on specified dates around the given location
    station_list = getIRISStations(lat_centre, lon_centre, deg_radius, start_date, end_date, \
        network=network, channel=channel, orfeus=orfeus)


    print('STATIONS:')
    print(station_list)


    # A list of station data and waveform files
    data_list = []

    # Go through all stations, retrieve and save waveforms
    for station_data in station_list:

        # Unpack station info
        network, station_code, stat_lat, stat_lon, stat_elev, station_name = station_data

        print('Downloading data:', network, station_code)
    
        # Retreive the waveform of the given station
        mseed_file = getIRISWaveformFiles(network, station_code, start_datetime, end_datetime, \
            channel=channel, dir_path=dir_path, orfeus=orfeus)

        # Add the mseed file to the data list
        if mseed_file is not None:
            station_data.append(mseed_file)
            data_list.append(station_data)


    # Save the list of station parameters and data files to disk
    writeStationAndWaveformsListFile(data_list, os.path.join(dir_path, DATA_FILE))

    print('Data file: ', DATA_FILE, 'written!')



def butterworthBandpassFilter(lowcut, highcut, fs, order=5):
    """ Butterworth bandpass filter.

    Argument:
        lowcut: [float] Lower bandpass frequency (Hz).
        highcut: [float] Upper bandpass frequency (Hz).
        fs: [float] Sampling rate (Hz).

    Keyword arguments:
        order: [int] Butterworth filter order.

    Return:
        (b, a): [tuple] Butterworth filter.

    """

    # Calculate the Nyquist frequency
    nyq = 0.5*fs

    low = lowcut/nyq
    high = highcut/nyq

    # Init the filter
    b, a = scipy.signal.butter(order, [low, high], btype='band')

    return b, a



def convolutionDifferenceFilter(waveform_data):
    """ Apply the convolution filter on data as suggested in Kalenda et al. (2014). """

    # Apply the filter 
    filtered_data = np.convolve(waveform_data, [-0.5, 1.0, -0.5], mode='same')

    # Detrend data
    filtered_data = filtered_data - np.mean(filtered_data)

    return filtered_data



def plotStationMap(data_list, lat_centre, lon_centre, ax=None):
    """ Plots the map of siesmic stations from loaded data file. """


    if ax is None:
        ax = plt.gca()

    # Find unique networks
    networks = set([entry[0] for entry in data_list])


    # Extract the list of station locations
    lat_list = [np.radians(entry[2]) for entry in data_list]
    lon_list = [np.radians(entry[3]) for entry in data_list]

    # Plot stations
    m = GroundMap(lat_list, lon_list, ax=ax, color_scheme='light')

    # Plot different networks with different colours
    for net in networks:

        # Extract the list of station locations
        lat_net_list = [np.radians(entry[2]) for entry in data_list if entry[0] == net]
        lon_net_list = [np.radians(entry[3]) for entry in data_list if entry[0] == net]

        m.scatter(lat_net_list, lon_net_list, s=2, label=net)

    # Plot source location
    m.scatter([np.radians(lat_centre)], [np.radians(lon_centre)], marker='*', c='yellow', edgecolor='k', linewidth=0.1, label='Source')

    plt.legend(loc='upper left')

    ax.set_title('Source location: {:.6f}, {:.6f}'.format(lat_centre, lon_centre))



def plotAllWaveforms(dir_path, data_list, v_sound, t0, lat_centre, lon_centre, ax=None, waveform_window=None,\
    difference_filter_all=False):
    """ Bandpass filter and plot all waveforms from the given data list. 

    Keyword arguments:
        waveform_window: [int] If given, the waveforms will be cut around the modelled time of arrival line
            with +/- waveform_window/2 seconds. None by default, which means the whole waveform will be 
            plotted.
        difference_filter_all: [bool] If True, the Kalenda et al. (2014) difference filter will be applied
                on the data plotted in the overview plot of all waveforms.
    """

    if ax is None:
        ax = plt.gca()

    max_wave_value = 0
    min_wave_value = np.inf
    max_time = 0

    # Go though all stations and waveforms
    for entry in data_list:

        net, station_code, stat_lat, stat_lon, stat_elev, station_name, mseed_file = entry

        print('Plotting:', net, station_code)

        mseed_file_path = os.path.join(dir_path, mseed_file)

        try:
            
            # Read the miniSEED file
            if os.path.isfile(mseed_file_path):

                mseed = obspy.read(mseed_file_path)

            else:
                print('File {:s} does not exist!'.format(mseed_file_path))
                continue


        except TypeError as e:

            print('Opening file {:s} failed with error: {:s}'.format(mseed_file_path, e))
            continue
        

        # Unpack miniSEED data
        delta = mseed[0].stats.delta
        waveform_data = mseed[0].data

        # Extract time
        start_datetime = mseed[0].stats.starttime.datetime
        end_datetime = mseed[0].stats.endtime.datetime


        # ### TEST!!!
        # print(station_code, start_datetime, end_datetime)
        # print(len(waveform_data))
        # print()

        # ###
        

        # Skip stations with no data
        if len(waveform_data) == 0:
            continue



        # Apply the Kalenda et al. (2014) difference filter instead of Butterworth
        if difference_filter_all:

            waveform_data = convolutionDifferenceFilter(waveform_data)

        else:

            ### BANDPASS FILTERING ###

            # Init the butterworth bandpass filter
            butter_b, butter_a = butterworthBandpassFilter(0.8, 5.0, 1.0/delta, order=6)

            # Filter the data
            waveform_data = scipy.signal.filtfilt(butter_b, butter_a, waveform_data)

            # Average and subsample the array for quicker plotting (reduces 40Hz to 10Hz)
            waveform_data = subsampleAverage(waveform_data, 4)
            delta *= 4

            ##########################


        # Calculate the distance from the source point to this station (kilometers)
        station_dist = greatCircleDistance(np.radians(lat_centre), np.radians(lon_centre), \
            np.radians(stat_lat), np.radians(stat_lon))

        # Construct time array, 0 is at start_datetime
        time_data = np.arange(0, (end_datetime - start_datetime).total_seconds(), delta)

        # Cut the waveform data length to match the time data
        waveform_data = waveform_data[:len(time_data)]
        time_data = time_data[:len(waveform_data)]

        
        # Detrend the waveform and normalize to fixed width
        waveform_data = waveform_data - np.mean(waveform_data)

        #waveform_data = waveform_data/np.percentile(waveform_data, 99)*2
        waveform_data = waveform_data/np.max(waveform_data)*10

        # Add the distance to the waveform
        waveform_data += station_dist


        # Cut the waveforms around the time of arrival, if the window for cutting was given.
        if waveform_window is not None:

            # Time of arrival
            toa = station_dist/(v_sound/1000) + t0

            # Cut the waveform around the time of arrival
            crop_indices = (time_data >= toa - waveform_window/2) & (time_data <= toa + waveform_window/2)
            time_data = time_data[crop_indices]
            waveform_data = waveform_data[crop_indices]
            

            # Skip plotting if array empty
            if len(time_data) == 0:
                continue


        # Replace all NaNs with 0s
        waveform_data = np.nan_to_num(waveform_data, 0)
        
        max_time = np.max([max_time, np.max(time_data)])

        # Keep track of minimum and maximum waveform values (used for plotting)
        max_wave_value = np.max([max_wave_value, np.max(waveform_data)])
        min_wave_value = np.min([min_wave_value, np.min(waveform_data)])

        # Plot the waveform on the the time vs. distance graph
        ax.plot(waveform_data, time_data, color='k', alpha=0.4, linewidth=0.2, zorder=3)

        # Print the name of the station
        ax.text(np.mean(waveform_data), np.max(time_data), net + '-' + station_code, rotation=270, \
            va='bottom', ha='center', size=4, zorder=3)



    toa_line_time = np.linspace(0, max_time, 10)

    # Plot the constant sound speed line (assumption is that the release happened at t = 0)
    ax.plot((toa_line_time)*v_sound/1000, (toa_line_time + t0), color='r', alpha=0.25, linewidth=1, \
        zorder=3, label="$V_s = " + "{:d}".format(int(v_sound)) + r" \rm{ms^{-1}}$")

    
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Time (s)')

    ax.set_ylim(0, max_time + 500)
    ax.set_xlim(0, max_wave_value)

    ax.grid(color='#ADD8E6', linestyle='dashed', linewidth=0.5, alpha=0.5)



if __name__ == "__main__":


    import matplotlib.pyplot as plt


    ### WAVEFORM RETRIEVAL PARAMETERS ###
    ##########################################################################################################


    # ### 2018-01-16 Michigan fireball

    # # Name of the folder where data files will be stored
    # dir_path = '../Seismic data/2018-01-16 Michigan fireball'

    # # Use the European data access site
    # orfeus = False

    # # Geo coordinates of the wave release centre
    # lat_centre = 42.646767
    # lon_centre = -83.925573

    # # Station search radius (degrees)
    # deg_radius = 10

    # # Seismic network code. Use '*' for all networks
    # network = '*'

    # # Instrument channel ('BDF' default)
    # channel = 'BDF'

    # # Speed of sound (m/s)
    # v_sound = 310

    # # Time offset of wave release from the reference time
    # t0 = 0

    # # Time range of waveform retrieval (can't be more than 2 hours!)
    # start_datetime = datetime.datetime(2018, 1, 17, 1, 8, 30)
    # end_datetime = datetime.datetime(2018, 1, 17, 2, 8, 30)

    # # Apply the Kalenda et al. (2014) difference filter to the plot of all waveforms
    # difference_filter_all = False

    # ###############


    # ### 2018-03-08 Seattle fireball

    # # Name of the folder where data files will be stored
    # dir_path = '../Seismic data/2018-03-08 Seattle fireball'

    # # Use the European data access site
    # orfeus = False

    # # Geo coordinates of the wave release centre
    # lat_centre = 47.348652
    # lon_centre = -124.075456

    # # Station search radius (degrees)
    # deg_radius = 10

    # # Seismic network code. Use '*' for all networks
    # network = '*'

    # # Instrument channel ('BDF' default)
    # channel = 'BDF'

    # # Speed of sound (m/s)
    # v_sound = 310

    # # Time offset of wave release from the reference time (seconds)
    # t0 = 159.0

    # # Time range of waveform retrieval (can't be more than 2 hours!)
    # start_datetime = datetime.datetime(2018, 3, 8, 3, 5, 20)
    # end_datetime = datetime.datetime(2018, 3, 8, 4, 30, 0)

    # # Apply the Kalenda et al. (2014) difference filter to the plot of all waveforms
    # difference_filter_all = False


    # #################


    # ### 2018-04-08 Varazdin fireball

    # # Name of the folder where data files will be stored
    # dir_path = '../Seismic data/2018-04-08 Varazdin fireball'

    # # Use the European data access site
    # orfeus = True

    # # Geo coordinates of the wave release centre
    # lat_centre = 46.238112
    # lon_centre = 16.603644

    # # Station search radius (degrees)
    # deg_radius = 2

    # # Seismic network code. Use '*' for all networks
    # network = '*'

    # # Instrument channel ('BDF' default)
    # channel = 'HHZ'

    # # Speed of sound (m/s)
    # v_sound = 310

    # # Time offset of wave release from the reference time (seconds)
    # t0 = 514.0

    # # Time range of waveform retrieval in UTC (can't be more than 2 hours!)
    # start_datetime = datetime.datetime(2018, 4, 8, 18, 40, 0)
    # end_datetime = datetime.datetime(2018, 4, 8, 19, 39, 0)

    # # Apply the Kalenda et al. (2014) difference filter to the plot of all waveforms
    # difference_filter_all = True


    # #################

    ### 2018-04-08 Varazdin fireball

    # Name of the folder where data files will be stored
    dir_path = '../Seismic data/2018-06-13 Nevada fireball'

    # Use the European data access site
    orfeus = False

    # Geo coordinates of the wave release centre
    lat_centre = 38.51115630406
    lon_centre = -115.17391599243

    # Station search radius (degrees)
    deg_radius = 2

    # Seismic network code. Use '*' for all networks
    network = '*'

    # Instrument channel ('BDF' default)
    channel = 'HHZ'

    # Speed of sound (m/s)
    v_sound = 310

    # Time offset of wave release from the reference time (seconds)
    t0 = 506.0

    # Time range of waveform retrieval in UTC (can't be more than 2 hours!)
    start_datetime = datetime.datetime(2018, 6, 13, 03, 45, 0)
    end_datetime = datetime.datetime(2018, 6, 13, 4, 45, 0)

    # Apply the Kalenda et al. (2014) difference filter to the plot of all waveforms
    difference_filter_all = False


    #################




    ##########################################################################################################


    ### Download all waveform files which are within the given geographical and temporal range ###
    ##########################################################################################################
    getAllWaveformFiles(lat_centre, lon_centre, deg_radius, start_datetime, end_datetime, network=network, \
        channel=channel, dir_path=dir_path, orfeus=orfeus)
    ##########################################################################################################



    # Load the station and waveform files list
    data_file_path = os.path.join(dir_path, DATA_FILE)
    if os.path.isfile(data_file_path):
        
        data_list = readStationAndWaveformsListFile(data_file_path)

    else:
        print('Station and waveform data file not found! Download the waveform files first!')
        sys.exit()



    if network == '*':
        network_name = 'all'
    else:
        network_name = network


    ### Plot station map ###
    ##########################################################################################################

    plotStationMap(data_list, lat_centre, lon_centre)

    plt.savefig(os.path.join(dir_path, "{:s}_{:s}_{:s}_stations.png".format(network_name, channel, \
        str(start_datetime).replace(':', '.'))), dpi=300)

    plt.show()

    ##########################################################################################################



    ### Filter and plot all downloaded waveforms ###
    ##########################################################################################################
    

    plotAllWaveforms(dir_path, data_list, v_sound, t0, lat_centre, lon_centre, \
        difference_filter_all=difference_filter_all)

    plt.title('Source location: {:.6f}, {:.6f}, Reference time: {:s} UTC, channel: {:s}'.format(lat_centre, \
        lon_centre, str(start_datetime), channel), fontsize=7)
    plt.legend(loc='lower right')

    plt.savefig(os.path.join(dir_path, "{:s}_{:s}_{:s}_waveforms.png".format(network_name, channel, \
        str(start_datetime).replace(':', '.'))), dpi=300)


    plt.show()

    ##########################################################################################################

    