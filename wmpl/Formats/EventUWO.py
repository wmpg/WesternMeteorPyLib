""" Recompute the meteor trajectory from UWO-format event.txt files. """


import os
import sys

import numpy as np

from wmpl.Formats.GenericArgumentParser import addSolverOptions
from wmpl.Trajectory.GuralTrajectory import GuralTrajectory
from wmpl.Trajectory.Trajectory import Trajectory


class StationData(object):
    def __init__(self, lat, lon, height, station_id):
        """ Container for station data. """

        self.station_id = station_id
        self.lon = lon
        self.lat = lat
        self.height = height

        self.time_data = []
        self.theta_data = []
        self.phi_data = []
        self.mag_data = []




def loadEventData(event_file_path):
    """ Load observation data from the event file. """

    with open(event_file_path) as f:

        observers = {}
        data_points = {}

        for line in f:

            line = line.replace('\n', '').replace('\r', '')


            # Check if the line gives general event data
            if line.startswith("dat"):

                # Parse the observer string
                line = line.replace("dat ; ", '')
                entries = line.split()

                # Store the data into a dictionary
                data_dict = {entries[i]: entries[i + 1] for i in range(len(entries) - 1)}


            # Check if the line gives an observer
            elif line.startswith("obs"):

                # Parse the observer string
                line = line.replace("obs ; ", '')
                entries = line.split()

                # Store the observer into a dictionary
                obs_dict = {entries[i]: entries[i + 1] for i in range(len(entries) - 1)}

                # Store the observers dictionary with the tag as the key
                observers[obs_dict["tag"]] = obs_dict


            # Check if the line gives an observation
            elif line.startswith("fit"):

                # Parse the observation string
                line = line.replace("fit ; ", '')
                entries = line.split()

                # Store the observation into a dictionary
                point_dict = {entries[i]: entries[i + 1] for i in range(len(entries) - 1)}

                # Store the observation with the tag-no as the key
                data_points[point_dict["tag"] + "-" + point_dict["no"]] = point_dict


        # Get the reference Julian date
        jd_ref = float(data_dict["jd"])

        dir_path = os.path.dirname(event_file_path)


        # Init the dictionary containing the observations
        station_data_dict = {}
        station_data_dict["jd_ref"] = jd_ref
        station_data_dict["dir_path"] = dir_path
        station_data_dict["station_data"] = []

        # Pair up observatins with stations and create StationData objects
        for obs_tag in observers:

            # Fetch all time, theta, phi, mag data from observations for this station
            data = []
            for point_key in data_points:

                # Check if the point starts with the observers tag
                if point_key.split("-")[0] == obs_tag:

                    # Extract observations
                    data.append(list(map(float, [data_points[point_key]["t"], data_points[point_key]["th"], \
                        data_points[point_key]["phi"], data_points[point_key]["mag"]])))


            # Sort the observations in time
            data = np.array(data)
            data = data[np.argsort(data[:, 0])]


            # Init the station data object
            lat = np.radians(float(observers[obs_tag]["lat"]))
            lon = np.radians(float(observers[obs_tag]["lon"]))
            elev = 1000*float(observers[obs_tag]["elv"])
            stat_data = StationData(lat, lon, elev, observers[obs_tag]["num"])

            # Add the position picks
            stat_data.time_data = data[:, 0]
            stat_data.theta_data = np.radians(data[:, 1])
            stat_data.phi_data = np.radians(data[:, 2])
            stat_data.mag_data = data[:, 3]

            # Add the station to the list of observers
            station_data_dict["station_data"].append(stat_data)


        return station_data_dict




def solveTrajectoryUWOEvent(station_data_dict, solver='original', velmodel=3, **kwargs):
        """ Runs the trajectory solver on points of the given type. 

        Keyword arguments:
            solver: [str] Trajectory solver to use:
                - 'original' (default) - "in-house" trajectory solver implemented in Python
                - 'gural' - Pete Gural's PSO solver
            velmodel: [int] Velocity propagation model for the Gural solver
                0 = constant   v(t) = vinf
                1 = linear     v(t) = vinf - |acc1| * t
                2 = quadratic  v(t) = vinf - |acc1| * t + acc2 * t^2
                3 = exponent   v(t) = vinf - |acc1| * |acc2| * exp( |acc2| * t ) (default)
        """


        # Check that there are at least two stations present
        if len(station_data_dict["station_data"]) < 2:
            print('ERROR! The event.txt file does not contain multistation data!')

            return False



        if solver == 'original':

            # Init the new trajectory solver object
            traj = Trajectory(station_data_dict["jd_ref"], output_dir=station_data_dict["dir_path"], \
                meastype=4, **kwargs)

        elif solver == 'gural':

            # Select extra keyword arguments that are present only for the gural solver
            gural_keys = ['max_toffset', 'nummonte', 'meastype', 'verbose', 'show_plots']
            gural_kwargs = {key: kwargs[key] for key in gural_keys if key in kwargs}

            # Init the new Gural trajectory solver object
            traj = GuralTrajectory(len(station_data_dict["station_data"]), station_data_dict["jd_ref"], \
                velmodel, verbose=1, output_dir=station_data_dict["dir_path"], meastype=4, \
                **gural_kwargs)


        # Infill trajectories from each site
        for stat_data in station_data_dict["station_data"]:

            # MC solver
            if solver == 'original':

                traj.infillTrajectory(stat_data.phi_data, stat_data.theta_data, stat_data.time_data, \
                    stat_data.lat, stat_data.lon, stat_data.height, \
                    station_id=stat_data.station_id, magnitudes=stat_data.mag_data)
            
            # Gural solver
            else:
                traj.infillTrajectory(stat_data.phi_data, stat_data.theta_data, stat_data.time_data, \
                    stat_data.lat, stat_data.lon, stat_data.height)


        print('Filling done!')


        # Solve the trajectory
        traj.run()

        return traj



if __name__ == "__main__":

    import argparse


    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Run the trajectory solver on the given UWO-format event.txt file.")

    arg_parser.add_argument('event_path', nargs=1, metavar='MET_PATH', type=str, \
        help='Full path to the event.txt file.')

    # Add other solver options
    arg_parser = addSolverOptions(arg_parser)

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    
    event_path = os.path.abspath(cml_args.event_path[0])

    # Check if the file path exists
    if not os.path.isfile(event_path):
        print('No such file:', event_path)
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


    # Load observations from the event.txt file
    station_data_dict = loadEventData(event_path)



    # Run trajectory solver on the loaded .met file
    solveTrajectoryUWOEvent(station_data_dict, solver=cml_args.solver, max_toffset=max_toffset, \
            monte_carlo=(not cml_args.disablemc), mc_runs=cml_args.mcruns, \
            geometric_uncert=cml_args.uncertgeom, gravity_correction=(not cml_args.disablegravity), 
            plot_all_spatial_residuals=cml_args.plotallspatial, plot_file_type=cml_args.imgformat, \
            show_plots=(not cml_args.hideplots), v_init_part=velpart, v_init_ht=vinitht)