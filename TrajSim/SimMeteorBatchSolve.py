""" Batch runs trajectory solvers on simulated meteor data. """

from __future__ import print_function, division, absolute_import

import os
import sys

import numpy as np

from Utils.Pickling import loadPickle, savePickle
from Trajectory.Trajectory import Trajectory
from Trajectory.GuralTrajectory import GuralTrajectory
from Trajectory.Orbit import calcOrbit
from TrajSim.ShowerSim import SimMeteor, AblationModelVelocity, ConstantVelocity, LinearDeceleration




if __name__ == "__main__":



    ### INPUTS ###
    ##########################################################################################################


    # Directory which contains SimMet .pickle files
    #shower_dir = os.path.abspath("../SimulatedMeteors/EMCCD/2011Draconids")

    shower_dir = os.path.abspath("../SimulatedMeteors/CABERNET/2011Draconids")

    #shower_dir = os.path.abspath("../SimulatedMeteors/CAMO/2011Draconids_TEST")
    #shower_dir = os.path.abspath("../SimulatedMeteors/CAMO/2012Perseids")

    #shower_dir = os.path.abspath("../SimulatedMeteors/Perfect_CAMO/2011Draconids")
    #shower_dir = os.path.abspath("../SimulatedMeteors/Perfect_CAMO/2011Draconids_TEST")

    #shower_dir = os.path.abspath("../SimulatedMeteors/SOMN_sim/2011Draconids")
    #shower_dir = os.path.abspath("../SimulatedMeteors/SOMN_precise_sim/LongFireball_grav")

    #shower_dir = os.path.abspath("../SimulatedMeteors/CAMSsim/2011Draconids")
    #shower_dir = os.path.abspath("../SimulatedMeteors/CAMSsim/2012Perseids")

    # Maximum time offset (seconds)
    t_max_offset = 1

    # Use gravity correction when calculating trajectories
    gravity_correction = True


    # Trajectory solvers
    traj_solvers = ['planes', 'los', 'milig', 'monte_carlo', 'gural0', 'gural1', 'gural2', 'gural3']
    #traj_solvers = ['planes', 'los', 'monte_carlo']
    #traj_solvers = ['milig']


    ##########################################################################################################



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

        # Save info about the simulated meteor
        sim_met.saveInfo(output_dir)

        # Solve the simulated meteor with multiple solvers
        for traj_solver in traj_solvers:


            if (traj_solver == 'los') or (traj_solver == 'planes') or (traj_solver == 'milig'):

                # Init the trajectory (LoS or intersecing planes)
                traj = Trajectory(sim_met.jdt_ref, output_dir=output_dir, max_toffset=t_max_offset, \
                    meastype=2, show_plots=False, save_results=False, monte_carlo=False, \
                    gravity_correction=gravity_correction)


            elif traj_solver == 'monte_carlo':
                
                # Init the trajectory
                traj = Trajectory(sim_met.jdt_ref, output_dir=output_dir, max_toffset=t_max_offset, \
                    meastype=2, show_plots=False, mc_runs=100, gravity_correction=gravity_correction)  ## TESING, ONLY 100 RUNS!!!

            
            elif 'gural' in traj_solver:

                # Extract the velocity model
                try:
                    velmodel = int(traj_solver[-1])

                except:
                
                    # Velocity model ID
                    velmodel = 3

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

                # Use the average velocity of the first part of the trajectory for the initial velocity
                time_vel = []
                for obs in traj.observations:
                    for t, v in zip(obs.time_data, obs.velocities):
                        time_vel.append([t, v])

                time_vel = np.array(time_vel)

                # Sort by time
                time_vel = time_vel[time_vel[:, 0].argsort()]

                # Calculate the velocity of the first half of the trajectory
                v_init_fh = np.mean(time_vel[int(len(time_vel)/2), 1])

                # Calculate the orbit
                traj.orbit = calcOrbit(traj.avg_radiant, v_init_fh, traj.v_avg, traj.state_vect_avg, \
                    traj.jd_avg, stations_fixed=True, reference_init=False)


                # Save the intersecting planes solution
                savePickle(traj, output_dir, traj.file_name + '_planes.pickle')


            # Calculate the orbit as MILIG does it, with the average velocity
            elif traj_solver == 'milig':

                # Calculate the orbit with the average velocity
                traj.orbit = calcOrbit(traj.radiant_eci_mini, traj.v_avg, traj.v_avg, traj.state_vect_mini, \
                    traj.rbeg_jd, stations_fixed=False, reference_init=False)

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