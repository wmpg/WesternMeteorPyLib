""" Runs the trajectory solver on RMS JSON files. """


import os
import sys
import glob
import json
import datetime
import argparse

import numpy as np

from wmpl.Formats.GenericFunctions import addSolverOptions, solveTrajectoryGeneric, MeteorObservation, \
    prepareObservations
from wmpl.Utils.TrajConversions import jd2Date


def saveJSON(dir_path, meteor_list):
    """ Save observations in the RMS JSON format. 
    
    Arguments:
        dir_path: [str] Path to where the JSON files will be saved.
        meteor_list: [list of MeteorObservation objects]

    """


    for meteor in meteor_list:

        # Construct the file name
        dt = jd2Date(meteor.jdt_ref, dt_obj=True)

        json_name = "{:s}_{:s}_picks.json".format(dt.strftime("%Y%m%d_%H%M%S.%f"), meteor.station_id)

        # Init JSON dict
        json_dict = {}

        json_dict["fps"] = meteor.fps
        json_dict["jdt_ref"] = meteor.jdt_ref
        json_dict["meastype"] = 1 # ra/dec

        json_dict["centroids_labels"] = ["Time (s)",
                                         "X (px)",
                                         "Y (px)",
                                         "RA (deg)",
                                         "Dec (deg)",
                                         "Summed intensity",
                                         "Magnitude"
                                         ]

        # Construct station info
        station = {}
        station["lat"] = np.degrees(meteor.latitude)
        station["lon"] = np.degrees(meteor.longitude)
        station["elev"] = meteor.height
        station["station_id"] = meteor.station_id
        json_dict["station"] = station


        # Construct the JSON data
        centroids = np.c_[meteor.time_data, meteor.x_data, meteor.y_data, np.degrees(meteor.ra_data), \
            np.degrees(meteor.dec_data), np.ones_like(meteor.time_data), meteor.mag_data]
        centroids = centroids.tolist()

        # Sort centroids by relative time
        centroids = sorted(centroids, key=lambda x: x[0])

        json_dict["centroids"] = centroids


        # Save the JSON file
        with open(os.path.join(dir_path, json_name), 'w') as f:
            json.dump(json_dict, f, indent=4, sort_keys=True)




def initMeteorObjects(json_list):

    # Init meteor objects
    meteor_list = []
    for j in json_list:

        # Init the meteor object
        meteor = MeteorObservation(j['jdt_ref'], j['station']['station_id'], \
            np.radians(j['station']['lat']), np.radians(j['station']['lon']), j['station']['elev'], j['fps'])

        # Add data to meteor object
        for entry in j['centroids']:
            t_rel, x_centroid, y_centroid, ra, dec, intensity_sum, mag = entry

            meteor.addPoint(t_rel*j['fps'], x_centroid, y_centroid, 0.0, 0.0, ra, dec, mag)

        meteor.finish()

        meteor_list.append(meteor)


    # Normalize all observations to the same JD and precess from J2000 to the epoch of date
    return prepareObservations(meteor_list)




if __name__ == "__main__":


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Run the trajectory solver on RMS JSON files.")

    arg_parser.add_argument('json_files', nargs="+", metavar='JSON_PATH', type=str, \
        help='Path to 2 of more JSON files.')

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

    json_paths = []
    print('Using JSON files:')


    # If the recursive walk option is given, find all JSON files recursively in the given folder
    if cml_args.walk:

        # Take the dir path as the given path
        dir_path = cml_args.json_files[0]

        # Find all manual reduction JSON files in the given folder
        json_names = []
        for entry in sorted(os.walk(dir_path), key=lambda x: x[0]):

            dir_name, _, file_names = entry

            # Add all JSON files with picks to the processing list
            for fn in file_names:
                if fn.lower().endswith("_picks.json"):

                    # Add JSON file, but skip duplicates
                    if fn not in json_names:
                        json_paths.append(os.path.join(dir_name, fn))
                        json_names.append(fn)


    else:
        for json_p in cml_args.json_files:
            for json_full_p in glob.glob(json_p):
                json_full_path = os.path.abspath(json_full_p)

                # Check that the path exists
                if os.path.exists(json_full_path):
                    json_paths.append(json_full_path)
                    print(json_full_path)
                else:
                    print('File not found:', json_full_path)


        # Extract dir path
        dir_path = os.path.dirname(json_paths[0])


    # Check that there are more than 2 JSON files given
    if len(json_paths) < 2:
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
    


    # Load all json files
    json_list = []
    for json_file in json_paths:
        with open(json_file) as f:
            data = json.load(f)
            json_list.append(data)


    ### If there are stations with the same names, append "_1, _2,..." at the end of their names ###
    station_names = [j['station']['station_id'] for j in json_list]
    unique, counts = np.unique(station_names, return_counts=True)

    for station_id, count in zip(unique, counts):

        id_add_counter = 1

        # If there are more than 1 stations with the same name, add suffixes
        if count > 1:

            # Find the stations with the duplicate ID
            for j in json_list:
                if j['station']['station_id'] == station_id:
                    j['station']['station_id'] += "_{:d}".format(id_add_counter)
                    id_add_counter += 1

    ### ###




    # Normalize the observations to the same reference Julian date and precess them from J2000 to the 
    # epoch of date
    jdt_ref, meteor_list = initMeteorObjects(json_list)


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
