""" Runs the trajectory solver on RMS JSON files. """


import os
import sys
import glob
import json
import argparse

import numpy as np

from wmpl.Formats.CAMS import MeteorObservation, prepareObservations
from wmpl.Trajectory.Trajectory import Trajectory
from wmpl.Trajectory.GuralTrajectory import GuralTrajectory
from wmpl.Utils.TrajConversions import jd2Date



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

            meteor.addPoint(t_rel*j['fps'], 0, 0, ra, dec, mag)

        meteor.finish()

        meteor_list.append(meteor)


    # Normalize all observations to the same JD and precess from J2000 to the epoch of date
    return prepareObservations(meteor_list)




def solveTrajectoryRMS(json_list, solver='original', **kwargs):
    """ Feed the list of meteors in the trajectory solver. """


    # Normalize the observations to the same reference Julian date and precess them from J2000 to the 
    # epoch of date
    jdt_ref, meteor_list = initMeteorObjects(json_list)

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

        elif solver == 'gural':

            traj.infillTrajectory(meteor.ra_data, meteor.dec_data, meteor.time_data, meteor.latitude, 
                meteor.longitude, meteor.height)


    # Solve the trajectory
    traj.run()

    return traj




if __name__ == "__main__":


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Run the trajectory solver on RMS JSON files.")

    arg_parser.add_argument('json_files', nargs="+", metavar='JSON_PATH', type=str, \
        help='Path to 2 of more JSON files.')

    arg_parser.add_argument('-s', '--solver', metavar='SOLVER', help="""Trajectory solver to use. \n
        - 'original' - Monte Carlo solver
        - 'gural0' - Gural constant velocity
        - 'gural1' - Gural linear deceleration
        - 'gural2' - Gural quadratic deceleration
        - 'gural3' - Gural exponential deceleration
         """, type=str, default='original')

    arg_parser.add_argument('-t', '--maxtoffset', metavar='MAX_TOFFSET', nargs=1, \
        help='Maximum time offset between the stations.', type=float)

    arg_parser.add_argument('-v', '--vinitht', metavar='V_INIT_HT', nargs=1, \
        help='The initial veloicty will be estimated as the average velocity above this height (in km). If not given, the initial velocity will be estimated using the sliding fit which can be controlled with the --velpart option.', \
        type=float)

    arg_parser.add_argument('-p', '--velpart', metavar='VELOCITY_PART', \
        help='Fixed part from the beginning of the meteor on which the initial velocity estimation using the sliding fit will start. Default is 0.4 (40 percent), but for noisier data this might be bumped up to 0.5.', \
        type=float, default=0.4)

    arg_parser.add_argument('-d', '--disablemc', \
        help='Do not use the Monte Carlo solver, but only run the geometric solution.', action="store_true")
    
    arg_parser.add_argument('-r', '--mcruns', metavar="MC_RUNS", nargs='?', \
        help='Number of Monte Carlo runs.', type=int, default=100)

    arg_parser.add_argument('-u', '--uncertgeom', \
        help='Compute purely geometric uncertainties.', action="store_true")
    
    arg_parser.add_argument('-g', '--disablegravity', \
        help='Disable gravity compensation.', action="store_true")

    arg_parser.add_argument('-l', '--plotallspatial', \
        help='Plot a collection of plots showing the residuals vs. time, lenght and height.', \
        action="store_true")

    arg_parser.add_argument('-i', '--imgformat', metavar='IMG_FORMAT', nargs=1, \
        help="Plot image format. 'png' by default, can be 'pdf', 'eps',... ", type=str, default='png')

    arg_parser.add_argument('-x', '--hideplots', \
        help="Don't show generated plots on the screen, just save them to disk.", action="store_true")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    ### Parse command line arguments ###

    json_paths = []
    print('Using JSON files:')
    for json_p in cml_args.json_files:
        for json_full_p in glob.glob(json_p):
            json_full_path = os.path.abspath(json_full_p)

            # Check that the path exists
            if os.path.exists(json_full_path):
                json_paths.append(json_full_path)
                print(json_full_path)
            else:
                print('File not found:', json_full_path)


    # Check that there are more than 2 JSON files given
    if len(json_paths) < 2:
        print("At least 2 JSON files are needed for trajectory estimation!")
        sys.exit()

    # Extract dir path
    dir_path = os.path.dirname(json_paths[0])


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



    # Init the trajectory structure
    traj = solveTrajectoryRMS(json_list, solver=cml_args.solver, max_toffset=max_toffset, \
            monte_carlo=(not cml_args.disablemc), mc_runs=cml_args.mcruns, \
            geometric_uncert=cml_args.uncertgeom, gravity_correction=(not cml_args.disablegravity), 
            plot_all_spatial_residuals=cml_args.plotallspatial, plot_file_type=cml_args.imgformat, \
            show_plots=(not cml_args.hideplots), v_init_part=velpart, v_init_ht=vinitht)
    