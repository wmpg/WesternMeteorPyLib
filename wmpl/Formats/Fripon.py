""" Solve the trajectory from FRIIPN observations in FRIPON-stype .met files."""

from __future__ import print_function, division, absolute_import

import os
import sys
import glob
import datetime

if sys.version_info.major < 3:
    import urllib as urllibrary

else:
    import urllib.request as urllibrary


import numpy as np

from wmpl.Formats.GenericFunctions import addSolverOptions, solveTrajectoryGeneric, MeteorObservation, \
    prepareObservations
from wmpl.Formats.RMSJSON import saveJSON
from wmpl.Utils.TrajConversions import datetime2JD, raDec2AltAz_vect


### CONSTANTS ###

# Use a fixed FPS (not really important in the grand scheme of things, it's just for combatibility with
#   used functions)
FPS = 15


# Read the table of FRIPON station data from github - about 235 lines
FRIPON_STATIONS_URL = 'https://raw.githubusercontent.com/SCAMP99/scamp/master/FRIPON_location_list.csv'    


# File that holds a list of FRIPON stations (will be refreshed if internet is available)
FRIPON_STATION_FILE = "FRIPON_location_list.csv"

### ###



def loadFripon(dir_path, fripon_paths, overwrite_fripon_stations=False):


    ### Download FRIPON station data from the web, or load it locally if available ###

    # Local path to the file where it will be saved
    station_file_path = os.path.join(dir_path, FRIPON_STATION_FILE)

    # If a file available locally, use it unless the overwrite flag is set
    if not os.path.exists(station_file_path) or overwrite_fripon_stations:

        try:
            # Download the FRIPON station files
            print("Downloading FRIPON station file from:", FRIPON_STATIONS_URL)
            urllibrary.urlretrieve(FRIPON_STATIONS_URL, station_file_path)
        except:

            # If the file cannot be downloaded, try loading it locally
            print("The FRIPON station file could not be retrieved!")
            

            if os.path.exists(station_file_path):
                print("Loading a local file...")

            else:
                print("A local station file could not be found:", station_file_path)
                sys.exit("Could not read station data")


    ###


    # Load the station data into arrays
    station_data = np.loadtxt(station_file_path, delimiter=',', dtype=str, skiprows=1, usecols=(0, 1, 2, 3))
    station_ids, station_lats, station_lons, station_elevs = station_data.T
    station_lats = station_lats.astype(np.float64)
    station_lons = station_lons.astype(np.float64)
    station_elevs = station_elevs.astype(np.float64)




    ### Load data from FRIPON .met files ###
    meteor_list = []
    for fripon_met in fripon_paths:

        # Read the station name from the file name
        file_name = os.path.basename(fripon_met)
        station_id = file_name.split('_')[2]


        # Find the station info
        if station_id in station_ids:
            station_index = np.argwhere(station_ids == station_id)[0][0]
            station_lat = station_lats[station_index]
            station_lon = station_lons[station_index]
            station_ele = station_elevs[station_index]

        else:
            print("Station {:s} not found in the list of FRIPON stations!".format(station_id))
            continue


        # Load the data
        data = np.loadtxt(fripon_met, comments='#', usecols=(3, 4, 5, 6, 7), dtype=str)

        # Skip empty files
        if len(data) == 0:
            continue

        # Unpack the data
        x_data, y_data, ra_data, dec_data, dt_data = data.T

        x_data = x_data.astype(np.float64)
        y_data = y_data.astype(np.float64)
        ra_data = ra_data.astype(np.float64)
        dec_data = dec_data.astype(np.float64)


        # Convert time to JD
        jd_data = []
        for date in dt_data:
            dt = datetime.datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%f")
            jd_data.append(datetime2JD(dt))


        jd_data = np.array(jd_data)

        # Take the first time as reference time
        jdt_ref = jd_data[0]


        # Compute relative time
        time_data = (jd_data - jdt_ref)*86400


        # Compute alt/az
        azim_data, alt_data = raDec2AltAz_vect(np.radians(ra_data), np.radians(dec_data), jd_data, \
            np.radians(station_lat), np.radians(station_lon))
        azim_data, alt_data = np.degrees(azim_data), np.degrees(alt_data)


        # Init the meteor object
        meteor = MeteorObservation(jdt_ref, station_id, np.radians(station_lat), \
            np.radians(station_lon), station_ele, FPS)

        # Add data to meteor object
        for t_rel, x_centroid, y_centroid, ra, dec, azim, alt in zip(time_data, x_data, y_data, \
            ra_data, dec_data, azim_data, alt_data):

            meteor.addPoint(t_rel*FPS, x_centroid, y_centroid, azim, alt, ra, dec, 0)

        meteor.finish()

        meteor_list.append(meteor)


    ### ###


    # Normalize all observations to the same JD and precess from J2000 to the epoch of date
    return prepareObservations(meteor_list)







if __name__ == "__main__":


    import argparse


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Run the trajectory solver on FRIPON .met files.")

    arg_parser.add_argument('fripon_files', nargs="+", metavar='FRIPON_PATH', type=str, \
        help="Path to 2 of more FRIPON .met files. Wildcards are supported, so e.g. /path/to/*.met also works.")

        # Add other solver options
    arg_parser = addSolverOptions(arg_parser, skip_velpart=True)

    arg_parser.add_argument('-p', '--velpart', metavar='VELOCITY_PART', \
        help="Fixed part from the beginning of the meteor on which the initial velocity estimation using the sliding fit will start. Default is 0.4 (40 percent), but for noisier data this might be bumped up to 0.5.", \
        type=float, default=0.4)

    arg_parser.add_argument('-w', '--walk', \
        help="Recursively find FRIPON .met files in the given folder and use them for trajectory estimation.", \
        action="store_true")
    
    arg_parser.add_argument('--updatestations', \
        help="Force update of the FRIPON station data file.", \
        action="store_true")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    ### Parse command line arguments ###

    fripon_paths = []
    print('Using FRIPON .met files:')


    # If the recursive walk option is given, find all FRIPON .met files recursively in the given folder
    if cml_args.walk:

        # Take the dir path as the given path
        dir_path = cml_args.fripon_files[0]

        # Find all manual reduction FRIPON .met files in the given folder
        fripon_names = []
        for entry in sorted(os.walk(dir_path), key=lambda x: x[0]):

            dir_name, _, file_names = entry

            # Add all FRIPON .met files with picks to the processing list
            for fn in file_names:
                if fn.lower().endswith(".met"):

                    # Add FRIPON .met file, but skip duplicates
                    if fn not in fripon_names:
                        fripon_paths.append(os.path.join(dir_name, fn))
                        fripon_names.append(fn)


    else:
        for fripon_p in cml_args.fripon_files:
            for fripon_full_p in glob.glob(fripon_p):
                fripon_full_path = os.path.abspath(fripon_full_p)

                # Check that the path exists
                if os.path.exists(fripon_full_path):
                    fripon_paths.append(fripon_full_path)
                    print(fripon_full_path)
                else:
                    print('File not found:', fripon_full_path)


        # Extract dir path
        dir_path = os.path.dirname(fripon_paths[0])


    # Check that there are more than 2 FRIPON files given
    if len(fripon_paths) < 2:
        print("At least 2 files are needed for trajectory estimation!")
        sys.exit()


    max_toffset = None
    if cml_args.maxtoffset:
        max_toffset = cml_args.maxtoffset[0]

    velpart = None
    if cml_args.velpart:
        velpart = cml_args.velpart

    vinitht = None
    if cml_args.vinitht:
        vinitht = cml_args.vinitht[0]

    ### ###



    # Load the observations into container objects
    jdt_ref, meteor_list = loadFripon(dir_path, fripon_paths, 
                                      overwrite_fripon_stations=cml_args.updatestations)


    # Save the data as RMS JSON files
    saveJSON(dir_path, meteor_list)


    # Solve the trajectory
    traj = solveTrajectoryGeneric(jdt_ref, meteor_list, dir_path, solver=cml_args.solver, \
        max_toffset=max_toffset, monte_carlo=(not cml_args.disablemc), mc_runs=cml_args.mcruns, \
        geometric_uncert=cml_args.uncertgeom, gravity_correction=(not cml_args.disablegravity), 
        gravity_factor=cml_args.gfact,
        plot_all_spatial_residuals=cml_args.plotallspatial, plot_file_type=cml_args.imgformat, \
        show_plots=(not cml_args.hideplots), v_init_part=velpart, v_init_ht=vinitht, \
        show_jacchia=cml_args.jacchia, \
        estimate_timing_vel=(False if cml_args.notimefit is None else cml_args.notimefit), \
        fixed_times=cml_args.fixedtimes, mc_noise_std=cml_args.mcstd)


    