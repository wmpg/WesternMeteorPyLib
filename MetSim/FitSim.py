from __future__ import print_function, division, absolute_import

import datetime

import os
import sys
import copy
import multiprocessing
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.optimize

from Config import config

from Utils.Pickling import loadPickle
from Utils.PyDomainParallelizer import DomainParallelizer
from Utils.TrajConversions import jd2Date

from MetSim.MetSim import M_PROTON, loadInputs, runSimulation
from MetSim.MetalMass import loadMetalMags, calcMass


def line(x, m, k):
    return m*x + k



def costFunction(x_data, y_data, model):
    """ Calculate the cost function between the given data and the given model. 

    Arguments:
        x_data: [ndarray] observed X axis data
        y_data: [ndarray] observed Y axis data
        model: [callable] function which will return the modeled values

    Return:
        [float] cost of a function
    """

    # Get the modeled values
    y_model = model(x_data)

    # print('Y MODEL:', y_model)
    # print('Y OBS:', y_data)


    # Calculate the soft L1 approximation of absolute differences
    #cost = 2*(np.sqrt(1 + (y_data - y_model)**2) - 1)

    # Calculate the absolute deviation
    cost = np.abs(y_data - y_model)

    # Return the mean cost function value
    return np.sum(cost)/len(x_data)
    



def simGoodness(sim_time, sim_height, sim_length, obs_time, obs_height, obs_length, show_plots=False):
    """ Calculate the goodness of fit between the modelled meteor and observations.
    """

    # Normalize obseved time to 0
    obs_time -= obs_time[0]

    # Get the maximum and minimum obseved heights
    ht_max = obs_height[0]
    ht_min = obs_height[-1]
    

    # Get the simulated heights below the maximum observed height
    sim_obs_max = sim_height[sim_height <= ht_max]

    # Get the simulated heights above the minimum observed height
    sim_obs_min = sim_height[sim_height >= ht_min]


    # If there are no heights below the maximum heights, reject the solution
    if (not len(sim_obs_max)) or (not len(sim_obs_min)):
        return np.inf


    # Get the indices of corresponding max/min heights in simulated data
    sim_max_i = np.where(sim_obs_max[ 0] == sim_height)[0][0]
    sim_min_i = np.where(sim_obs_min[-1] == sim_height)[0][0]


    # Take simulated heights only in the range of observed heights
    sim_height_obs = sim_height[sim_max_i:sim_min_i]

    # Take simulated length only in the range of observed lengths
    sim_length_obs = sim_length[sim_max_i:sim_min_i]

    # Take simulated time only in the range of observed lengths
    sim_time_obs = sim_time[sim_max_i:sim_min_i]

    # Project the first observed height to the simulated lenghts
    model_ht_interpol_full = scipy.interpolate.CubicSpline(-sim_height_obs, sim_length_obs, extrapolate=True)
    first_point_length = model_ht_interpol_full(-ht_max)

    # Project the first observed height to the simulated lengths
    model_time_interpol_full = scipy.interpolate.CubicSpline(-sim_height_obs, sim_time_obs, extrapolate=True)
    first_point_time = model_time_interpol_full(-ht_max)

    # Add the projected point as the first point
    sim_height_obs = np.r_[ht_max, sim_height_obs]
    sim_length_obs = np.r_[float(first_point_length), sim_length_obs]
    sim_time_obs = np.r_[float(first_point_time), sim_time_obs]


    # Normalize simulated length to 0
    sim_length_obs -= sim_length_obs[0]

    # Normalize simulated time to 0
    sim_time_obs -= sim_time_obs[0]


    # Penalize the occurence when the simulated length does not cover the whole range of observed data
    # (if it covers the whole range, the penalization should not influence the end result)
    #penal = (obs_height[0] - obs_height[-1])/(sim_height_obs[1]  - sim_height_obs[-1])
    penal = 1.0

    # print()
    # print('Obs range:', ht_max, ht_min, 'Sim range:', sim_height_obs[0], sim_height_obs[-1])
    # print('Penal:', penal)


    # NOTE: heights are negative because the CubicSpline function requires that the independant variable
    # (heights in our case) must be increasing, but that does not influence the end cost function value


    # Interpolate time vs. length of modelled points
    model_time_length_interpol = scipy.interpolate.CubicSpline(sim_time_obs, sim_length_obs, extrapolate=True)

    # Calculate the cost function value between observations and the model
    cost_value = costFunction(obs_time, obs_length, model_time_length_interpol)*penal


    if show_plots:

        print('Cost function:', cost_value)

        plt.scatter(obs_time, obs_length - model_time_length_interpol(obs_time))

        plt.title('Time vs length residuals')
        plt.show()



        # Plot modelled points
        plt.scatter(sim_time_obs, sim_length_obs, label='Modelled', s=5, c='b')

        x_times = np.linspace(np.min(sim_time_obs), np.max(sim_time_obs), 1000)

        # Plot cubic fit
        plt.plot(x_times, model_time_length_interpol(x_times), label='Modelled spline', color='b')

        # Plot observed points
        plt.scatter(obs_time, obs_length, label='Observed', s=10, c='r')

        # Plot observed points projected to the fitted line
        plt.scatter(obs_time, model_time_length_interpol(obs_time), s=10, c='g', label='Projected')


        plt.legend()
        plt.show() 



        # Plot velocity
        sim_velocity = calcVelocity(sim_time_obs, sim_length_obs)[0]
        plt.scatter(sim_velocity[1:], sim_time_obs[1:], label='Simulated')

        obs_velocity = calcVelocity(obs_time, obs_length)[0]
        plt.scatter(obs_velocity[1:], obs_time[1:], label='Observed')

        plt.title('Velocities')
        plt.legend()

        plt.gca().invert_yaxis()


        plt.show()


        ### Plot the simulated vs. observed lag

        # Take the first part of the simulated meteor
        len_part = int(0.05*len(sim_time))
        sim_time_part = sim_time[:len_part]
        sim_length_part = sim_length[:len_part]

        # Fit a line to the first part of the simulated data
        sim_lag_fit, _ = scipy.optimize.curve_fit(line, sim_time_part, sim_length_part)

        # Calculate the model lag
        sim_lag = sim_length - line(sim_time, *sim_lag_fit)

        # Plot the simulated lag
        plt.plot(sim_lag, sim_time, label='Simulated')

        plt.gca().invert_yaxis()

        # Initerpolate the simulated length with height
        sim_lag_ht_interpol_full = scipy.interpolate.CubicSpline(-sim_height, sim_lag, extrapolate=True)

        # Find the length of simulation at the first point of observations
        sim_lag_at_first_obs = sim_lag_ht_interpol_full(-obs_height[0])

        # Interpolate the simulated time vs. height
        sim_ht_time_interpol_full = scipy.interpolate.CubicSpline(-sim_height, sim_time, extrapolate=True)

        # Find the time in simulation at the first observed height
        sim_time_at_first_obs = sim_ht_time_interpol_full(-obs_height[0])

        # Plot observed lag
        obs_lag = obs_length - line(obs_time, *sim_lag_fit) + sim_lag_at_first_obs
        plt.plot(obs_lag, obs_time + sim_time_at_first_obs, label='Observed')

        plt.title('Observed vs simulated lag')
        
        plt.xlabel('Lag (m)')
        plt.ylabel('Time (s)')

        plt.legend()

        plt.show()


        ######


        # len_residuals = obs_length - model_interpol(-obs_height)
        # print('Residual range:', np.max(len_residuals) - np.min(len_residuals))

        # # Plot residuals between observed and modelled
        # plt.scatter(len_residuals, obs_height)

        # plt.show()


        

    return cost_value




def runMetSimEvaluation(v_init, init_mass, rho, q, T_boil, m_mass, c_p, met=None, consts=None, obs_time=None, obs_height=None, obs_length=None, end_ht=None):

    # Create a copy of the meteor and constant structures
    met = copy.deepcopy(met)
    consts = copy.deepcopy(consts)

    # Set meteor parameters
    met.v_init = v_init
    met.m_init = init_mass
    met.rho = rho
    met.q = q
    met.T_boil = T_boil
    met.m_mass = m_mass
    met.c_p = c_p


    # Run the simulation with the given parameters
    results_list = runSimulation(met, consts)

    # Get the results
    results_list = np.array(results_list)

    # Unpack results
    sim_time, sim_height, sim_length, sim_velocity = results_list.T

    # Calculate the goodness of fit
    cost_value = simGoodness(sim_time, sim_height, sim_length, obs_time, obs_height, obs_length)


    # Reject all simulations which do not reach the minimum observed height
    if end_ht < sim_height[-1]:
        print('Minimum simulated height lower than observed height:', [cost_value, v_init, init_mass, rho, q, T_boil, m_mass, c_p])
        return None


    print('Solution: ', [cost_value, v_init, init_mass, rho, q, T_boil, m_mass, c_p])

    sys.stdout.flush()


    return [cost_value, v_init, init_mass, rho, q, T_boil, m_mass, c_p]




def bruteForceSearchMetSim(results_file, met, consts, mass, v_init, zg, obs_time, obs_height, obs_length, end_ht):
    """ Do a brute force search through the meteor parameter space for the best fit to the observed data.

    """


    # # Set the initial velocity
    # met.v_init = v_init

    # # Set the initial mass
    # met.m_init = mass

    # Set the zenith angle
    consts.zr = zg


    #-> Define ranges of physical parameters (min, max, step)
    ### Reference MODIFIED!: Kikwaya et al., 2011, Bulk density of small meteoroids

    # Initial velocity range
    v_init_range = np.linspace(0.98*v_init, 1.04*v_init, 31)

    # Initial mass range
    init_mass_range = np.logspace(np.log10(0.1*mass), np.log10(10*mass), 11)

    # Density range, kg/m^3
    rho_range = np.linspace(500, 6000, 21)

    # Heat of ablation range, J/kg
    q_range = [(2e6 + 9e6)/2]
    #q_range = np.linspace(2e6, 9e6, 3)

    # Boiling temperature range, K
    T_boil_range = [(1400 + 2300)/2]
    #T_boil_range = np.linspace(1400, 2300, 3)

    # Molar mass range, atomic units
    m_mass_range = np.array([20.0, 36.0, 56.0])*M_PROTON

    # Specific heat range
    c_p_range = [(600+1400)/2]
    #c_p_range = np.linspace(600, 1400, 3)




    # # Initial mass range
    # init_mass_range = np.logspace(np.log10(0.01*met.m_init), np.log10(2*met.m_init), 3)

    # # Density range, kg/m^3
    # rho_range = np.linspace(100, 8000, 5)

    # # Heat of abalation range, J/kg
    # q_range = np.linspace(2e6, 9e6, 3)

    # # Boiling temperature range, K
    # T_boil_range = np.linspace(1400, 2300, 3)

    # # Molar mass range, atomic units
    # m_mass_range = np.array([20, 36, 56])*M_PROTON

    # # Specific heat range
    # c_p_range = np.linspace(600, 1400, 3)


    #<-

    
    #-> Generate combinations of physical parameters of the meteor

    input_params = []

    for v_init in v_init_range:
        for init_mass in init_mass_range:
            for rho in rho_range:
                for q in q_range:
                    for T_boil in T_boil_range:
                        for m_mass in m_mass_range:
                            for c_p in c_p_range:

                                input_params.append([v_init, init_mass, rho, q, T_boil, m_mass, c_p])

    #<-


    # Get the number of cpu cores available
    cpu_cores = multiprocessing.cpu_count()

    # Run the parallelized function
    solutions = DomainParallelizer(input_params, runMetSimEvaluation, cores=cpu_cores, 
        kwarg_dict={'met': met, 'consts': consts, 'obs_time':obs_time, 'obs_height': obs_height, 'obs_length': obs_length, 'end_ht': end_ht})


    # Reject all solutions which are None
    solutions = [x for x in solutions if x is not None]

    solutions = np.array(solutions)

    # Sort the solutions by the increasing cost function value
    solutions = solutions[solutions[:, 0].argsort()]

    print('Best solution:', solutions[0])

    # Save the array to disk
    np.save(results_file, solutions)




def refineMetSimEvaluation(params, met=None, consts=None, obs_time=None, obs_height=None, obs_length=None, show_plots=False):
    """ Function which will be minimized by the minimizer. """


    v_init, init_mass, rho = params

    # Create a copy of the meteor and constant structures
    met = copy.deepcopy(met)
    consts = copy.deepcopy(consts)

    # Set meteor parameters
    met.v_init = v_init
    met.m_init = init_mass
    met.rho = rho

    # Run the simulation with the given parameters
    results_list = runSimulation(met, consts)

    # Get the results
    results_list = np.array(results_list)

    if not len(results_list):
        return np.inf

    # Unpack results of simulation
    sim_time, sim_height, sim_length, sim_velocity = results_list.T

    # Calculate the goodness of fit
    cost_value = simGoodness(sim_time, sim_height, sim_length, obs_time, obs_height, obs_length, show_plots=show_plots)

    print(cost_value, params)

    return cost_value




def refineSearchMetSim(met, consts, zg, obs_time, obs_height, obs_length, end_ht, best_guess):
    """ Refine the initial guess of the pohysical parameters of the meteor. """

    v_init, init_mass, rho, q, T_boil, m_mass, c_p = best_guess

    # Set the initial velocity
    met.v_init = v_init

    # Set the zenith angle
    consts.zr = zg

    print('Vinit:', v_init)
    print('Initial evaluation:', runMetSimEvaluation(init_mass, rho, q, T_boil, m_mass, c_p, met, consts, obs_time, obs_height, obs_length, end_ht))

    # refine_results = scipy.optimize.basinhopping(refineMetSimEvaluation, best_guess, minimizer_kwargs={'args':(met, consts, obs_height, obs_length)})
    # refine_results = scipy.optimize.basinhopping(refineMetSimEvaluation, [v_init, init_mass, rho], minimizer_kwargs={'args':(met, consts, obs_height, obs_length)})

    # # Define the boundaries for the optimization (per each parameter)
    # bounds = [
    #     [ 0.9*v_init,     1.1*v_init  ],  # Initial velocity
    #     [ 0.5*init_mass,    2*init_mass], # Initial mass
    #     [ 0.75*rho,      1.25*rho],       # Density
    #     [ 0.5*q,          2*q],         # Heat of ablation
    #     [ 0.5*T_boil,     2*T_boil],    # Boiling temperature
    #     [ 0.5*m_mass,     2*m_mass],    # Molar mass
    #     [ 0.5*c_p,        2*c_p]        # Specific heat
    # ]

    # Set fixed parameters
    met.q = q
    met.T_boil = T_boil
    met.m_mass = m_mass
    met.c_p = c_p


    # Define the boundaries for the optimization (per each parameter)
    bounds = [
        [ 0.95*v_init,     1.05*v_init  ],  # Initial velocity
        [ 0.25*init_mass,    4*init_mass], # Initial mass
        [ 0.5*rho,          2*rho]        # Density
    ]

    print('BOUNDS', bounds)

    # Run the minimization
    refine_results = scipy.optimize.minimize(refineMetSimEvaluation, [v_init, init_mass, rho], args=(met, consts, obs_time, obs_height, obs_length), bounds=bounds, method='TNC')



    # Print original parameters
    print('ORIGINAL:', best_guess)

    # Print refined parameters
    print(refine_results)

    # Show the residuals with the optimized solution
    refineMetSimEvaluation(refine_results.x, met, consts, obs_time, obs_height, obs_length, show_plots=True)

    


def findBestSolution(solutions, top_N=20):
    """ Among all solutions, take top N of the ones with the best cost, and return the solution which
        has the most frequent parameters among the top N.

    """

    #-> Find the most frequent parameters in the top N solutions and use them as a basis for further brute
    ### force search

    top_solutions = solutions[:top_N, 1:]


    top_params = []

    # Go through every parameter in the top N solutions
    for param in top_solutions.T:

        # Find the unique parameters in the list and find the number of their occurence
        unique, counts = np.unique(param,  return_counts=True)

        # Take the parameter which the highest number of occurences
        top_params.append(unique[counts.argmax()])


    # Find the solution with the highest number of parameters matching top parameters, and use it as the
    # best solution

    top_count = 0
    top_solution_ind = 0
    for i, solution in enumerate(top_solutions):
        
        c = 0
        for j, param in enumerate(top_params):
            if solution[j] == param:
                c += 1

        if c > top_count:
            top_count = c
            top_solution_ind = i

    
    return top_solutions[top_solution_ind]






def showResult(met, consts, v_init, zg, obs_time, obs_height, obs_length, params=None):

    # Set the initial velocity
    # met.v_init = v_init

    # Set the zenith angle
    consts.zr = zg

    # # Set meteor parameters
    # met.rho = 100
    # met.q = 7e6
    # met.T_boil = 2300
    # met.m_mass = 6.02144316e-26

    # Take the parameters from the given argument and set them as meteor physical parameters
    if len(params):
        cost_value, met.v_init, met.m_init, met.rho, met.q, met.T_boil, met.m_mass, met.c_p = params


    # Run the simulation with the given parameters
    results_list = runSimulation(met, consts)

    # Get the results
    results_list = np.array(results_list)

    # Unpack results
    sim_time, sim_height, sim_length, sim_velocity = results_list.T

    # Calculate the goodness of fit
    simGoodness(sim_time, sim_height, sim_length, obs_time, obs_height, obs_length, show_plots=True)




def showAllResults(results_file, met, consts, v_init, zg, obs_height, obs_length):


    # Load the numpy array with the results from a file
    solutions = np.load(results_file)


    for i, solution in enumerate(solutions):

        cost_value, v_init, init_mass, rho, q, T_boil, m_mass, c_p = solution

        print('\n------------------------------')
        print('Solution No.', i+1)
        print('  Cost Function:', cost_value)
        print('  Vinit:', v_init, 'm/s')
        print('  Mass:', "{:e}".format(init_mass), 'kg')
        print('  Rho:', rho, 'kg/m^3')
        print('  Heat of ablation:', q, 'J/kg')
        print('  T_boil:', T_boil, 'K')
        print('  Molar mass:', m_mass)
        print('  Specific heat:', c_p)

        showResult(met, consts, v_init, zg, obs_height, obs_length, params=solution)



# TEST FUNCTON
def showBestResult(results_file, met, consts, v_init, zg, obs_time, obs_height, obs_length):

    # Load the numpy array with the results from a file
    solutions = np.load(results_file)

    # The best solution is the first one
    best_solution = solutions[0]

    print('Solution:', best_solution)

    cost_value, v_init, init_mass, rho, q, T_boil, m_mass, c_p = best_solution

    #for v_init in np.linspace(0.94*v_init, 0.96*v_init, 10):
    #for v_init in np.linspace(0.99*v_init, 1.01*v_init, 5):
    #for zg in np.linspace(0.97*zg, 0.99*zg, 10):
    #for zg in np.linspace(0.97*zg, 0.99*zg, 10):
    
    print('Vinit:', v_init/1000)
    print('zg', np.degrees(zg))
    showResult(met, consts, v_init, zg, obs_time, obs_height, obs_length, params=best_solution)



def calcVelocity(time, length):
    """ Calculates the velocity with the given time and length. """

    # Shift the lengths one element down (for difference calculation)
    dists_shifted = np.r_[0, length][:-1]

    # Calculate distance differences from point to point (first is always 0)
    dists_diffs = length - dists_shifted

    # Shift the time one element down (for difference calculation)
    time_shifted = np.r_[0, time][:-1]

    # Calculate the time differences from point to point
    time_diffs = time - time_shifted

    # Replace zeros in time by machine precision value to avoid division by zero errors
    time_diffs[time_diffs == 0] = np.finfo(np.float64).eps

    # Calculate velocity for every point
    velocities = dists_diffs/time_diffs

    return velocities, time_diffs




if __name__ == "__main__":

    ### TRAJECTORY FILE

    # Path to the Trajectory file
    #dir_path_mir = "../MirfitPrepare/20160929_062945_mir/"
    #dir_path_mir = "../MirfitPrepare/20161007_052346_mir/"
    dir_path_mir = "../MirfitPrepare/20161007_052749_mir/"
    

    # Trajectory pickle file
    #traj_pickle_file = "20160929_062945_trajectory.pickle"
    #traj_pickle_file = "20161007_052346_trajectory.pickle"
    traj_pickle_file = "20161007_052749_trajectory.pickle"
    

    ######


    ### METAL REDUCTION FILE

    dir_path_metal = "/home/dvida/Dropbox/UWO Master's/Projects/MetalPrepare/20161007_052346_met"

    metal_file = 'state.met'

    ######


    # Name of input file for meteor parameters
    meteor_inputs_file = config.met_sim_input_file

    # Load input meteor data
    met, consts = loadInputs(meteor_inputs_file)


    # Load the pickled trajectory
    traj = loadPickle(dir_path_mir, traj_pickle_file)

    # Load parameters from the trajectory object
    v_init = traj.v_init
    v_avg = traj.orbit.v_avg
    zg = traj.orbit.zg



    ### Calculate the photometric mass of the meteoroid (kg) estimated from widefield data

    # Load apparent magnitudes from the METAL reduction
    time_mags = loadMetalMags(dir_path_metal, metal_file)

    masses = []

    # Calculate the mass and show magnitude curves
    for site, time, mag_abs in time_mags:

        plt.plot(time, mag_abs, marker='x', label='Site ' + str(site), zorder=3)

        # Calculate the mass from magnitudes
        mass = calcMass(time, mag_abs, v_avg)

        masses.append(mass)

        print()
        print('Site ', site)
        print('Mass:  ', mass, 'kg')
        print(' log10:', np.log10(mass))


    plt.gca().invert_yaxis()

    plt.xlabel('Time (s)')
    plt.ylabel('Absolute magnitude (@100km)')

    plt.legend()
    plt.grid()

    plt.show()


    mass = np.median(masses)

    print('Median mass:', mass, 'kg')

    ######


    # Plot lags
    for obs in traj.observations:
        plt.plot(obs.lag, obs.time_data, marker='x', label='Site ' + str(obs.station_id), zorder=3)

    plt.gca().invert_yaxis()

    plt.xlabel('Lag (m)')
    plt.ylabel('Time (s)')

    plt.legend()
    plt.grid()

    plt.show()



    results_list = []
    full_cost_list = []

    # Go through all observations
    for station_ind, obs in enumerate(traj.observations):

        # # Find the observation with the highest starting height
        # begin_heights = [obs.rbeg_ele for obs in traj.observations]
        # station_ind = begin_heights.index(max(begin_heights))

        # TEST FOR REPRODUCING OLD RESULTS!!!
        # station_ind = 0

        obs_height = traj.observations[station_ind].model_ht
        obs_length = traj.observations[station_ind].length
        obs_time = traj.observations[station_ind].time_data

        # End height of the observed meteor
        end_ht = np.min(obs_height)


        # Name of the results file
        results_file = jd2Date(traj.jdt_ref, dt_obj=True).strftime('%Y%m%d_%H%M%S') + "_" + str(traj.observations[station_ind].station_id) + "_simulations.npy"
        results_file = os.path.join(dir_path_mir, results_file)

        # Add the results file to the results list
        results_list.append(results_file)


        # Fit only the first 25% of the observed trajectory
        len_part = int(0.25*len(obs_time))

        # If the first 25% has less than 4 points, than take the first 4 points
        if len_part < 4:
            len_part = 4

        obs_height = obs_height[:len_part]
        obs_length = obs_length[:len_part]
        obs_time = obs_time[:len_part]

        print('Number of points:', len_part)
        

        
        t1 = datetime.datetime.now()

        # # Do the brute force search
        # bruteForceSearchMetSim(results_file, met, consts, mass, v_init, zg, obs_time, obs_height, obs_length, end_ht)
        # continue


        print('Simulation time:', datetime.datetime.now() - t1)


        # showResult(met, consts, v_init, zg, obs_height, obs_length)

        # showAllResults(results_file, met, consts, v_init, zg, obs_height, obs_length)

        showBestResult(results_file, met, consts, v_init, zg, obs_time, obs_height, obs_length)


        # Calculate observed velocities
        velocities, time_diffs = calcVelocity(obs_time, obs_length)
        print(velocities)

        # def line(x, m, k):
        #     return m*x + k

        # # Find a best fit through velocities (assume constant deceleration)
        # popt, _ = scipy.optimize.curve_fit(line, obs_time[1:], velocities[1:])

        # print(popt)

        # Calculate the RMS of velocities
        # vel_rms = np.sqrt(np.mean((velocities[1:] - line(obs_time[1:], *popt))**2))
        vel_rms = np.sqrt(np.mean((velocities[1:] - v_init)**2))

        print('Vel RMS:', vel_rms)

        # Calculate the along track differences
        #along_track_diffs = (velocities[1:] - line(obs_time[1:], *popt))*time_diffs[1:]
        along_track_diffs = (velocities[1:] - v_init)*time_diffs[1:]

        # Calculate the full 3D residuals
        full_residuals = np.sqrt(along_track_diffs**2 + traj.observations[station_ind].v_residuals[:len_part][1:]**2 
            + traj.observations[station_ind].h_residuals[:len_part][1:]**2)


        # Calculate the RMS of lengths along the track
        full_rms = np.sqrt(np.mean(full_residuals**2))

        print('3D RMS:', full_rms)

        
        # full_cost = np.sum(2*(np.sqrt(1 + np.array(along_track_diffs)**2) - 1))
        full_cost = np.sum(np.abs(np.array(full_residuals)))/len(full_residuals)

        full_cost_list.append(full_cost)

        print('3D cost:', full_cost)
        #print('RMS cost:', np.sum(rms_cost))



    ### PLOT VELOCITIES
    plt.scatter(velocities[1:], obs_time[1:])
    
    x_times = np.linspace(np.min(obs_time), np.max(obs_time), 100)
    #plt.plot(line(x_times, *popt), x_times)
    plt.plot(np.zeros_like(x_times) + v_init, x_times)

    plt.gca().invert_yaxis()

    plt.show()


    
    # # Load the numpy array with the results from a file
    # solutions = np.load(results_file)

    # # Find the best solution among the top N
    # best_solution = findBestSolution(solutions)

    # print('BEST SOLUTION:', best_solution)


    # best_guess = np.append(np.array([v_init]), best_solution)

    # refineSearchMetSim(met, consts, zg, obs_time, obs_height, obs_length, end_ht, best_guess)