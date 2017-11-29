""" Reruns the trajectory solution from a trajectory pickle file. """

from __future__ import print_function, division, absolute_import

import os

from Utils.Pickling import loadPickle
from Trajectory.Trajectory import Trajectory
from Trajectory.GuralTrajectory import GuralTrajectory


def solveTrajectory(dir_path, file_name, solver='original', **kwargs):


    # Load the pickles trajectory
    traj_p = loadPickle(dir_path, file_name)

    # Run the PyLIG trajectory solver
    if solver == 'original':

        # Reinitialize the trajectory solver
        traj = Trajectory(traj_p.jdt_ref, output_dir=dir_path, max_toffset=traj_p.max_toffset, \
            meastype=traj_p.meastype, **kwargs)


        # Fill the observations
        for obs in traj_p.observations:

            # Check if the trajectory had any excluded points
            if hasattr(traj_p, 'excluded_time'):
                excluded_time = obs.excluded_time

            else:
                excluded_time = None


            traj.infillTrajectory(obs.meas1, obs.meas2, obs.time_data, obs.lat, obs.lon, obs.ele, 
                station_id=obs.station_id, excluded_time=excluded_time)


    elif solver == 'gural':

        # Init the Gural solver
        # traj = GuralTrajectory(len(traj_p.observations), traj_p.jdt_ref, max_toffset=traj_p.max_toffset, \
        #     meastype=traj_p.meastype, output_dir=dir_path)

        traj = GuralTrajectory(len(traj_p.observations), traj_p.jdt_ref, velmodel=3, \
            max_toffset=traj_p.max_toffset, meastype=traj_p.meastype, output_dir=dir_path, verbose=True)


        # Fill the observations
        for obs in traj_p.observations:

            traj.infillTrajectory(obs.meas1, obs.meas2, obs.time_data, obs.lat, obs.lon, obs.ele)


    else:
        print('Unrecognized solver:', solver)


    # Run the trajectory solver
    traj.run()

    return traj



if __name__ == "__main__":

    # dir_path = os.path.abspath("../SimulatedMeteors/CAMO/Perseids/2456154.85547")
    # file_name = "20120815_083152_trajectory.pickle"

    dir_path = os.path.abspath("../SimulatedMeteors/CAMO/Perseids/2456154.70174")
    file_name = "20120815_045030_trajectory.pickle"


    # Solve the trajectory from the given pickle file
    traj = solveTrajectory(dir_path, file_name, solver='gural')