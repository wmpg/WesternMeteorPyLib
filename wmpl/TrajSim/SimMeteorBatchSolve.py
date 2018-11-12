""" Batch runs trajectory solvers on simulated meteor data. """

from __future__ import print_function, division, absolute_import

import os
import sys

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

from wmpl.Utils.Pickling import loadPickle, savePickle
from wmpl.Utils.Math import lineFunc
from wmpl.Trajectory.Trajectory import Trajectory
from wmpl.Trajectory.GuralTrajectory import GuralTrajectory
from wmpl.Trajectory.Orbit import calcOrbit
from wmpl.TrajSim.ShowerSim import SimMeteor, AblationModelVelocity, ConstantVelocity, LinearDeceleration



def calcVelocityOfFirstHalf(traj):
    """ Given the trajectory, computes the average velocity of the first half of the trajectory by fitting a
        line through time vs. distance since the beginning of the meteor.
    """

    times = [t for obs in traj.observations for t in obs.time_data]
    sv_dists = [sv_dist for obs in traj.observations for sv_dist in obs.state_vect_dist]

    # Sort by time
    temp_arr = np.c_[times, sv_dists]
    temp_arr = temp_arr[np.argsort(temp_arr[:, 0])]
    times, sv_dists = temp_arr.T

    half_index = int(len(times)/2)
    times_half = times[:half_index]
    sv_dists_half = sv_dists[:half_index]

    # Fit a line to the first 50% of the points
    popt, _ = scipy.optimize.curve_fit(lineFunc, times_half, sv_dists_half)

    vel = popt[0]


    # print('vel:', vel)

    # # Plot original points
    # plt.scatter(times, sv_dists)

    # # Plot fitted line
    # time_arr = np.linspace(np.min(times), np.max(times), 100)
    # plt.plot(time_arr, lineFunc(time_arr, *popt))

    # plt.show()

    return vel





if __name__ == "__main__":



    ### INPUTS ###
    ##########################################################################################################

    shower_dir_list = [
    # Directory which contains SimMet .pickle files
    #shower_dir = os.path.abspath("../SimulatedMeteors/EMCCD/2011Draconids")

    #shower_dir = os.path.abspath("../SimulatedMeteors/CABERNET/2011Draconids")

    # os.path.abspath("../SimulatedMeteors/CAMO/2011Draconids"),
    # os.path.abspath("../SimulatedMeteors/CAMO/2014Ursids"),
    # os.path.abspath("../SimulatedMeteors/CAMO/2012Perseids"),

    # os.path.abspath("../SimulatedMeteors/CAMSsim/2011Draconids"),
    # os.path.abspath("../SimulatedMeteors/CAMSsim/2014Ursids"),
    # os.path.abspath("../SimulatedMeteors/CAMSsim/2012Perseids"),

    # os.path.abspath("../SimulatedMeteors/SOMN_sim/2011Draconids"),
    #os.path.abspath("../SimulatedMeteors/SOMN_sim/2014Ursids"),
    # os.path.abspath("../SimulatedMeteors/SOMN_sim/2012Perseids"),
    # os.path.abspath("../SimulatedMeteors/SOMN_sim/2015Taurids")

    os.path.abspath("../SimulatedMeteors/SOMN_sim/LongFireball")
    #os.path.abspath("../SimulatedMeteors/SOMN_sim/LongFireball_nograv")
    #shower_dir = os.path.abspath("../SimulatedMeteors/SOMN_sim/LongFireball_nograv")
    ]

    # Maximum time offset (seconds)
    t_max_offset = 2

    # Use gravity correction when calculating trajectories
    gravity_correction = True


    # Trajectory solvers
    traj_solvers = ['planes', 'los', 'milig', 'monte_carlo', 'gural0', 'gural0fha', 'gural1', 'gural2', 'gural3']
    #traj_solvers = ['gural0fha']
    #traj_solvers = ['planes', 'los', 'monte_carlo']
    #traj_solvers = ['los']
    #traj_solvers = ['planes', 'milig']


    ##########################################################################################################


    # Go through all systems and showers
    for shower_dir in shower_dir_list:

        sim_meteor_list = []

        # Load simulated meteors from pickle files
        for file_name in sorted(os.listdir(shower_dir)):

            if 'sim_met.pickle' in file_name:
                
                print('Loading pickle file:', file_name)

                sim_met = loadPickle(shower_dir, file_name)

                sim_meteor_list.append(sim_met)


        # Solve generated trajectories
        for met_no, sim_met in enumerate(sim_meteor_list):

            # Directory where trajectory results will be saved
            output_dir = os.path.join(shower_dir, "{:03d} - {:.6f}".format(met_no, sim_met.jdt_ref))

            # Prepare everything for saving data to disk
            sim_met.initOutput(output_dir)

            # Save info about the simulated meteor (THIS CAN BE DISABLED WITH UPDATING SOLUTIONS)
            sim_met.saveInfo(output_dir)

            # Solve the simulated meteor with multiple solvers
            for traj_solver in traj_solvers:


                if (traj_solver == 'los') or (traj_solver == 'planes') or (traj_solver == 'milig'):

                    # If just the LoS is running without MC, then save the results
                    if (traj_solver == 'los') and ('monte_carlo' not in traj_solvers):
                        save_results = True
                    else:
                        save_results = False

                    # Init the trajectory (LoS or intersecing planes)
                    traj = Trajectory(sim_met.jdt_ref, output_dir=output_dir, max_toffset=t_max_offset, \
                        meastype=2, show_plots=False, save_results=save_results, monte_carlo=False, \
                        gravity_correction=gravity_correction)


                elif traj_solver == 'monte_carlo':
                    
                    # Init the trajectory
                    traj = Trajectory(sim_met.jdt_ref, output_dir=output_dir, max_toffset=t_max_offset, \
                        meastype=2, show_plots=False, mc_runs=100, gravity_correction=gravity_correction)  ## TESING, ONLY 100 RUNS!!!

                
                elif 'gural' in traj_solver:

                    # Extract the velocity model
                    try:
                        velmodel = traj_solver.replace('gural', '')

                    except:
                    
                        # Velocity model ID
                        print('Unknown velocity model:', velmodel)
                        sys.exit()

                    # Init the new Gural trajectory solver object
                    traj = GuralTrajectory(len(sim_met.observations), sim_met.jdt_ref, output_dir=output_dir, \
                        max_toffset=t_max_offset, meastype=2, velmodel=velmodel, verbose=1, \
                        show_plots=False)

                else:
                    print(traj_solver, '- unknown trajectory solver!')
                    sys.exit()


                # Fill in observations
                for obs in sim_met.observations:

                    traj.infillTrajectory(obs.meas1, obs.meas2, obs.time_data, obs.lat, obs.lon, obs.ele, \
                        station_id=obs.station_id)


                # Solve trajectory
                traj = traj.run()


                # Calculate orbit of an intersecting planes solution
                if traj_solver == 'planes':

                    # Use the average velocity of the first half of the trajectory for the initial velocity
                    v_init_fh = calcVelocityOfFirstHalf(traj)

                    # Calculate the orbit
                    traj.orbit = calcOrbit(traj.avg_radiant, v_init_fh, traj.v_avg, traj.state_vect_avg, \
                        traj.jd_avg, stations_fixed=True, reference_init=False)


                    # Save the intersecting planes solution
                    savePickle(traj, output_dir, traj.file_name + '_planes.pickle')


                # Calculate the LoS with the average velocity of the first half
                elif traj_solver == 'milig':

                    # Use the average velocity of the first half of the trajectory for the initial velocity
                    v_init_fh = calcVelocityOfFirstHalf(traj)

                    # Calculate the orbit with the average velocity
                    traj.orbit = calcOrbit(traj.radiant_eci_mini, v_init_fh, traj.v_avg, traj.state_vect_mini, \
                        traj.rbeg_jd, stations_fixed=False, reference_init=True)

                    # Save the simulated milig solution
                    savePickle(traj, output_dir, traj.file_name + '_milig.pickle')


                if 'gural' in traj_solver:

                    # Save info about the simulation comparison
                    sim_met.saveTrajectoryComparison(traj, traj_solver, 'velmodel: ' + str(velmodel))

                else:
                    # Save info about the simulation comparison
                    sim_met.saveTrajectoryComparison(traj, traj_solver)


                # Dump measurements to a MATLAB-style file
                if traj_solver == 'monte_carlo':
                    traj.dumpMeasurements(output_dir, str(sim_met.jdt_ref) + '_meas_dump.txt')