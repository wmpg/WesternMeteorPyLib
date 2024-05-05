""" Solve the trajectory from observations in ECSV files. """

from __future__ import print_function, division, absolute_import

import os
import sys
import glob
import datetime

import numpy as np

from wmpl.Formats.GenericFunctions import addSolverOptions, solveTrajectoryGeneric, MeteorObservation, \
    prepareObservations, writeMiligInputFileMeteorObservation
from wmpl.Utils.TrajConversions import J2000_JD, datetime2JD, altAz2RADec_vect, equatorialCoordPrecession_vect



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
                            station_id = line[1].strip().strip("'")

                        if (station_id is None) and ("dfn_camera_codename" in line[0]):
                            station_id = line[1].strip().strip("'")
                            

            if (station_lat is None) or (station_lon is None) or (station_ele is None) \
                or (station_id is None):

                print("Station info could not be read from file:", ecsv_file, ", skipping...")
                continue


            # Load meteor measurements
            delimiter = ','
            data = np.loadtxt(ecsv_file, comments='#', delimiter=delimiter, dtype=str)

            # Determine the column indices from the header
            header = data[0].tolist()
            dt_indx = header.index('datetime')
            azim_indx = header.index('azimuth')
            alt_indx = header.index('altitude')
            x_indx = header.index('x_image')
            y_indx = header.index('y_image')

            if 'mag_data' in header:
                mag_indx = header.index('mag_data')
            else:
                mag_indx = None


            # Skip the header
            data = data[1:]

            # Unpack data
            dt_data, azim_data, alt_data, x_data, y_data = data[:, dt_indx], data[:, azim_indx], \
                data[:, alt_indx], data[:, x_indx], data[:, y_indx]

            azim_data = azim_data.astype(np.float64)
            alt_data = alt_data.astype(np.float64)
            x_data = x_data.astype(np.float64)
            y_data = y_data.astype(np.float64)

            # Get magnitude data, if any
            if mag_indx is not None:
                mag_data = data[:, mag_indx].astype(np.float64)
            else:
                mag_data = np.zeros_like(azim_data) + 10.0


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

            # Precess to J2000
            ra_data, dec_data = equatorialCoordPrecession_vect(jdt_ref, J2000_JD.days, ra_data, dec_data)



            # Init the meteor object
            meteor = MeteorObservation(jdt_ref, station_id, np.radians(station_lat), \
                np.radians(station_lon), station_ele, FPS)

            # Add data to meteor object
            for t_rel, x_centroid, y_centroid, ra, dec, azim, alt, mag in zip(time_data, x_data, y_data, \
                np.degrees(ra_data), np.degrees(dec_data), azim_data, alt_data, mag_data):

                meteor.addPoint(t_rel*FPS, x_centroid, y_centroid, azim, alt, ra, dec, mag)

            meteor.finish()


            # Check that the observation has a minimum number of points
            if len(meteor.time_data) < 4:
                print("The station {:s} has too few points (<4), skipping: {:s}".format(station_id, ecsv_file))
                continue


            meteor_list.append(meteor)


    # Normalize all observations to the same JD and precess from J2000 to the epoch of date
    return prepareObservations(meteor_list)




if __name__ == "__main__":

    import argparse


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Run the trajectory solver on DFN ECSV files.")

    arg_parser.add_argument('ecsv_files', nargs="+", metavar='ECSV_PATH', type=str, \
        help="Path to 2 of more ECSV files. Wildcards are supported, so e.g. /path/to/*.ecsv also works.")

    # Add other solver options
    arg_parser = addSolverOptions(arg_parser, skip_velpart=True)

    arg_parser.add_argument('-p', '--velpart', metavar='VELOCITY_PART', \
        help="Fixed part from the beginning of the meteor on which the initial velocity estimation using the sliding fit will start. Default is 0.4 (40 percent), but for noisier data this might be bumped up to 0.5.", \
        type=float, default=0.4)

    arg_parser.add_argument('-w', '--walk', \
        help="Recursively find all ECSV files in the given folder and use them for trajectory estimation. If a directory containing the file contains the word 'REJECT', it will be skipped. ", \
        action="store_true")
    
    arg_parser.add_argument('--writemilig', metavar='MILIG_PATH', type=str, \
        help="Write the observations to a MILIG input file and exit. The MILIG_PATH argument is the path to the output file.")

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

            # Skip all directories with the word "REJECT" in them
            if "REJECT" in dir_name:
                print("Directory {:s} skipped because it contains 'REJECT'.".format(dir_name))
                continue

            # Add all ECSV files with picks to the processing list
            for fn in file_names:
                if fn.lower().endswith(".ecsv"):

                    # Add ECSV file, but skip duplicates
                    if fn not in ecsv_names:
                        ecsv_paths.append(os.path.join(dir_name, fn))
                        ecsv_names.append(fn)

                        print(fn)


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


    # Load the observations into container objects
    jdt_ref, meteor_list = loadECSVs(ecsv_paths)


    # Write the observations to a MILIG input file and exit
    if cml_args.writemilig:
        writeMiligInputFileMeteorObservation(jdt_ref, meteor_list, cml_args.writemilig)
        print("MILIG input file written to:", cml_args.writemilig)
        print("Exiting...")
        sys.exit()



    # Check that there are more than 2 ECSV files given
    if len(ecsv_paths) < 2:
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


    # Solve the trajectory
    traj = solveTrajectoryGeneric(jdt_ref, meteor_list, dir_path, solver=cml_args.solver, \
        max_toffset=max_toffset, monte_carlo=(not cml_args.disablemc), mc_runs=cml_args.mcruns, \
        geometric_uncert=cml_args.uncertgeom, gravity_correction=(not cml_args.disablegravity), 
        gravity_factor=cml_args.gfact,
        plot_all_spatial_residuals=cml_args.plotallspatial, plot_file_type=cml_args.imgformat, \
        show_plots=(not cml_args.hideplots), v_init_part=velpart, v_init_ht=vinitht, \
        show_jacchia=cml_args.jacchia,
        estimate_timing_vel=(False if cml_args.notimefit is None else cml_args.notimefit), \
        fixed_times=cml_args.fixedtimes, mc_noise_std=cml_args.mcstd)
    

