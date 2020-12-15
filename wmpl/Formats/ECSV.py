""" Solve the trajectory from observations in ECSV files. """


import os
import sys
import glob
import datetime

import numpy as np


from wmpl.Formats.CAMS import MeteorObservation, prepareObservations
from wmpl.Trajectory.Trajectory import Trajectory
from wmpl.Trajectory.GuralTrajectory import GuralTrajectory
from wmpl.Formats.GenericArgumentParser import addSolverOptions
from wmpl.Utils.TrajConversions import jd2Date, datetime2JD, altAz2RADec_vect



# Use a fixed FPS (not really important in the grand scheme of things, it's just for combatibility with
#   used functions)
FPS = 15


def loadECSVs(ecsv_paths):

    # Init meteor objects
    meteor_list = []
    for ecsv_file in ecsv_paths:

        
        station_lat = None
        station_lon = None
        station_ele = None
        station_id = None

        # Load the station information from the ECSV file
        with open(ecsv_file) as f:

            for line in f:
                if line.startswith("#"):
                    line = line.replace('\n', '').replace('\r', '').replace('{', '').replace('}', '')
                    line = line.split(':')
                    if len(line) > 1:

                        if "obs_latitude" in line[0]:
                            station_lat = float(line[1])

                        if "obs_longitude" in line[0]:
                            station_lon = float(line[1])

                        if "obs_elevation" in line[0]:
                            station_ele = float(line[1])

                        if "camera_id" in line[0]:
                            station_id = line[1].strip()

            if (station_lat is None) or (station_lon is None) or (station_ele is None) \
                or (station_id is None):

                print("Station info could not be read from file:", ecsv_file, ", skipping...")
                continue


            # Load meteor measurements
            data = np.loadtxt(ecsv_file, comments='#', delimiter=',', usecols=(0, 1, 2, 3, 4), dtype=str)

            # Skip the header
            data = data[1:]

            # Unpack data
            dt_data, azim_data, alt_data, x_data, y_data = data.T

            azim_data = azim_data.astype(np.float64)
            alt_data = alt_data.astype(np.float64)
            x_data = x_data.astype(np.float64)
            y_data = y_data.astype(np.float64)


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


            # Compute RA/Dec
            ra_data, dec_data = altAz2RADec_vect(np.radians(azim_data), np.radians(alt_data), jd_data, \
                np.radians(station_lat), np.radians(station_lon))


            # Init the meteor object
            meteor = MeteorObservation(jdt_ref, station_id, np.radians(station_lat), \
                np.radians(station_lon), station_ele, FPS)

            # Add data to meteor object
            for t_rel, x_centroid, y_centroid, ra, dec, azim, alt in zip(time_data, x_data, y_data, \
                np.degrees(ra_data), np.degrees(dec_data), azim_data, alt_data):

                meteor.addPoint(t_rel*FPS, x_centroid, y_centroid, azim, alt, ra, dec, 0)

            meteor.finish()

            meteor_list.append(meteor)


    # Normalize all observations to the same JD and precess from J2000 to the epoch of date
    return prepareObservations(meteor_list)


def solveTrajectoryECSV(ecsv_paths, dir_path, solver='original', **kwargs):
    """ Feed the list of meteors in the trajectory solver. """


    # Normalize the observations to the same reference Julian date and precess them from J2000 to the 
    # epoch of date
    jdt_ref, meteor_list = loadECSVs(ecsv_paths)

    # Create name of output directory
    output_dir = os.path.join(dir_path, jd2Date(jdt_ref, dt_obj=True).strftime("%Y%m%d-%H%M%S.%f"))


    # Init the trajectory solver
    if solver == 'original':
        traj = Trajectory(jdt_ref, output_dir=output_dir, meastype=1, **kwargs)

    elif solver.lower().startswith('gural'):
        velmodel = solver.lower().strip('gural')
        if len(velmodel) == 1:
            velmodel = int(velmodel)
        else:
            velmodel = 0

        traj = GuralTrajectory(len(meteor_list), jdt_ref, velmodel=velmodel, meastype=1, verbose=1, 
            output_dir=output_dir)

    else:
        print('No such solver:', solver)
        return 


    # Add meteor observations to the solver
    for meteor in meteor_list:

        if solver == 'original':

            traj.infillTrajectory(meteor.ra_data, meteor.dec_data, meteor.time_data, meteor.latitude, 
                meteor.longitude, meteor.height, station_id=meteor.station_id, \
                magnitudes=meteor.mag_data)

        elif solver.lower().startswith('gural'):

            # Extract velocity model is given
            try:
                velmodel = int(solver[-1])

            except: 
                # Default to the exponential model
                velmodel = 3

            traj.infillTrajectory(meteor.ra_data, meteor.dec_data, meteor.time_data, meteor.latitude, 
                meteor.longitude, meteor.height)


    # Solve the trajectory
    traj = traj.run()

    return traj





if __name__ == "__main__":

    import argparse


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Run the trajectory solver on DFN ECSV files.")

    arg_parser.add_argument('ecsv_files', nargs="+", metavar='ECSV_PATH', type=str, \
        help='Path to 2 of more ECSV files.')

        # Add other solver options
    arg_parser = addSolverOptions(arg_parser, skip_velpart=True)

    arg_parser.add_argument('-p', '--velpart', metavar='VELOCITY_PART', \
        help="Fixed part from the beginning of the meteor on which the initial velocity estimation using the sliding fit will start. Default is 0.4 (40 percent), but for noisier data this might be bumped up to 0.5.", \
        type=float, default=0.4)

    arg_parser.add_argument('-w', '--walk', \
        help="Recursively find all manual reduction JSON files in the given folder and use them for trajectory estimation.", \
        action="store_true")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    ### Parse command line arguments ###

    ecsv_paths = []
    print('Using ECSV files:')


    # If the recursive walk option is given, find all ECSV files recursively in the given folder
    if cml_args.walk:

        # Take the dir path as the given path
        dir_path = cml_args.ecsv_files[0]

        # Find all manual reduction ECSV files in the given folder
        ecsv_names = []
        for entry in sorted(os.walk(dir_path), key=lambda x: x[0]):

            dir_name, _, file_names = entry

            # Add all ECSV files with picks to the processing list
            for fn in file_names:
                if fn.lower().endswith(".ecsv"):

                    # Add ECSV file, but skip duplicates
                    if fn not in ecsv_names:
                        ecsv_paths.append(os.path.join(dir_name, fn))
                        ecsv_names.append(fn)


    else:
        for ecsv_p in cml_args.ecsv_files:
            for ecsv_full_p in glob.glob(ecsv_p):
                ecsv_full_path = os.path.abspath(ecsv_full_p)

                # Check that the path exists
                if os.path.exists(ecsv_full_path):
                    ecsv_paths.append(ecsv_full_path)
                    print(ecsv_full_path)
                else:
                    print('File not found:', ecsv_full_path)


        # Extract dir path
        dir_path = os.path.dirname(ecsv_paths[0])


    # Check that there are more than 2 ECSV files given
    if len(ecsv_paths) < 2:
        print("At least 2 JSON files are needed for trajectory estimation!")
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


    # Init the trajectory structure
    traj = solveTrajectoryECSV(ecsv_paths, dir_path, solver=cml_args.solver, max_toffset=max_toffset, \
            monte_carlo=(not cml_args.disablemc), mc_runs=cml_args.mcruns, \
            geometric_uncert=cml_args.uncertgeom, gravity_correction=(not cml_args.disablegravity), 
            plot_all_spatial_residuals=cml_args.plotallspatial, plot_file_type=cml_args.imgformat, \
            show_plots=(not cml_args.hideplots), v_init_part=velpart, v_init_ht=vinitht, \
            show_jacchia=cml_args.jacchia)
    

