""" Reruns the trajectory solution from a trajectory pickle file. """

from __future__ import print_function, division, absolute_import

import os

import numpy as np

from Formats.EvUWO import writeEvFile
from Utils.Pickling import loadPickle
from Utils.TrajConversions import jd2Date
from Trajectory.Trajectory import Trajectory
from Trajectory.GuralTrajectory import GuralTrajectory


def dumpAsEvFiles(dir_path, file_name):
    """ Dump the given pickle file as UWO-style ev_* file. """

        # Load the pickles trajectory
    traj = loadPickle(dir_path, file_name)

    
    # Dump the results as a UWO-style ev file

    year, month, day, hour, minute, second, _ = jd2Date(traj.jdt_ref)

    for i, obs in enumerate(traj.observations):

        # Construct file name
        date_str = "{:4d}{:02d}{:02d}_{:02d}{:02d}{:02d}A_{:s}".format(year, month, day, hour, minute, second, \
            obs.station_id)

        ev_file_name = 'ev_' + date_str + '.txt'

        # Convert azimuth and altitude to theta/tphi
        theta_data = np.pi/2.0 - obs.elev_data
        phi_data = (np.pi/2.0 - obs.azim_data)%(2*np.pi)

        # Write the ev_* file
        writeEvFile(dir_path, ev_file_name, traj.jdt_ref, str(i), obs.lat, obs.lon, obs.ele, 
            obs.time_data, theta_data, phi_data)




def solveTrajectory(dir_path, file_name, solver='original', **kwargs):
    """ Rerun the trajectory solver on the given trajectory pickle file. """


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

    # dir_path = os.path.abspath("../SimulatedMeteors/CAMO_OLD/Perseids/2456149.60802")
    # file_name = "20120810_023532_trajectory.pickle"

    # dir_path = os.path.abspath("../SimulatedMeteors/CAMO_OLD/Perseids/2456150.7563")
    # file_name = "20120811_060904_trajectory.pickle"

    dir_path = os.path.abspath("/home/dvida/Desktop/test/012 - 2455896.500000")
    file_name = "20111201_000000_trajectory.pickle"


    # # Dump the pickled trajectory as UWO-style ev_* file 
    # dumpAsEvFiles(dir_path, file_name)


    # Solve the trajectory from the given pickle file
    traj = solveTrajectory(dir_path, file_name, solver='original')