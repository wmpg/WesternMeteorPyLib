
import os
import sys
import copy
import re
import json
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.optimize

from wmpl.Formats.Met import loadMet
from wmpl.Trajectory.Trajectory import Trajectory
from wmpl.MetSim.GUI import loadConstants, saveConstants, SimulationResults, MetObservations
from wmpl.MetSim.MetSimErosion import runSimulation, Constants
from wmpl.Utils.AtmosphereDensity import fitAtmPoly
from wmpl.Utils.Math import meanAngle
from wmpl.Utils.Pickling import loadPickle


def costFunc(traj, met_obs, sr, mag_sigma, len_sigma, plot_residuals=False, hideplots=False):
    """ Compute the difference between the simulated and the observed meteor. 
    
    Arguments:
        traj: [Trajectory] Trajectory object.
        met_obs: [MetObservations] Meteor observations.
        sr: [SimulationResults] Simulation results.
        mag_sigma: [float] Magnitude residual sigma.
        len_sigma: [float] Length residual sigma.

    Keyword arguments:
        plot_residuals: [bool] Plot the residuals.

    Returns:
        mag_res: [float] Magnitude residual.
        len_res: [float] Length residual.
        cost: [float] Cost function value (magnitude residual + length residual)
    """

    mag_res_sum = 0
    mag_res_count = 0
    len_res_sum = 0
    len_res_count = 0


    # Interpolate the simulated magnitude
    sim_mag_interp = scipy.interpolate.interp1d(sr.leading_frag_height_arr, sr.abs_magnitude,
        bounds_error=False, fill_value='extrapolate')

    # Interpolate the simulated length by height
    sim_len_interp = scipy.interpolate.interp1d(sr.leading_frag_height_arr, sr.leading_frag_length_arr,
        bounds_error=False, fill_value='extrapolate')

    # Interpolate the simulated time
    sim_time_interp = scipy.interpolate.interp1d(sr.leading_frag_height_arr, sr.time_arr,
        bounds_error=False, fill_value='extrapolate')

    # Interpolate the velocity
    sim_vel_interp = scipy.interpolate.interp1d(sr.leading_frag_height_arr, sr.leading_frag_vel_arr,
        bounds_error=False, fill_value='extrapolate')

    
    # Find the simulated length at the trajectory begining
    sim_len_beg = sim_len_interp(traj.rbeg_ele)

    # Find the simulated time at the trajectory begining
    sim_time_beg = sim_time_interp(traj.rbeg_ele)

    # Find the simulated velocity at the trajectory begining
    sim_vel_beg = sim_vel_interp(traj.rbeg_ele)

    # Set the simulated length at the beginning of observations to zero
    norm_sim_len = sr.leading_frag_length_arr - sim_len_beg

    # Compute the normalized time
    norm_sim_time = sr.time_arr - sim_time_beg


    norm_sim_ht = sr.leading_frag_height_arr[norm_sim_len > 0]
    norm_sim_time = norm_sim_time[norm_sim_len > 0]
    norm_sim_len = norm_sim_len[norm_sim_len > 0]

    # Interpolate the normalized length by time
    sim_norm_len_interp = scipy.interpolate.interp1d(norm_sim_time, norm_sim_len,
        bounds_error=False, fill_value='extrapolate')

    
    ### TEST

    # plt.plot(norm_sim_time, norm_sim_len)
    # for obs in traj.observations:
    #     plt.scatter(obs.time_data, obs.state_vect_dist)

    # plt.show()

    ###



    # Init a new plot
    if plot_residuals:
        fig, (ax_mag, ax_magres, ax_lag, ax_lenres) = plt.subplots(ncols=4, sharey=True)

    # Compute the magnitude and length residuals from the trajectory object
    for obs in traj.observations:

        # Set a default filter which takes all observations
        mag_filter = np.ones(len(obs.model_ht), dtype=bool)

        if obs.absolute_magnitudes is not None:

            # Filter out observations with magnitude fainter than +9 or with NaN magnitudes
            mag_filter = (obs.absolute_magnitudes < 9) & (~np.isnan(obs.absolute_magnitudes))

            # Sample the simulated magnitude at the observation heights
            sim_mag_sampled = sim_mag_interp(obs.model_ht[mag_filter])

            # Compute the magnitude residual
            mag_res_sum += np.sum(np.abs(obs.absolute_magnitudes[mag_filter] - sim_mag_sampled))
            mag_res_count += len(obs.absolute_magnitudes[mag_filter])


        # Sample the simulated normalized length at observed times
        sim_norm_len_sampled = sim_norm_len_interp(obs.time_data[mag_filter])

        # Compute the length residual
        len_res_sum += np.sum(np.abs(obs.state_vect_dist[mag_filter] - sim_norm_len_sampled))
        len_res_count += len(obs.state_vect_dist)


        # Plot the observed and simulated magnitudes
        if plot_residuals:

            if obs.absolute_magnitudes is not None:
                ax_mag.scatter(obs.absolute_magnitudes[mag_filter], obs.model_ht[mag_filter]/1000, 
                               label=obs.station_id)
                ax_mag.plot(sim_mag_sampled, obs.model_ht[mag_filter]/1000)
                ax_magres.scatter(obs.absolute_magnitudes[mag_filter] - sim_mag_sampled, 
                                  obs.model_ht[mag_filter]/1000)

            # ax_len.scatter(obs.state_vect_dist[mag_filter], obs.model_ht[mag_filter]/1000, 
            #   label=obs.station_id)
            # ax_len.plot(sim_norm_len_sampled, obs.model_ht[mag_filter]/1000)
            ax_lenres.scatter(obs.state_vect_dist[mag_filter] - sim_norm_len_sampled, 
                              obs.model_ht[mag_filter]/1000)

            # Compute the simulated lag
            sim_lag = sim_norm_len_sampled - sim_vel_beg*obs.time_data[mag_filter]

            # Compute the observed lag using the simulated velocity
            obs_lag = obs.state_vect_dist[mag_filter] - sim_vel_beg*obs.time_data[mag_filter]

            # Plot the lag
            ax_lag.scatter(obs_lag, obs.model_ht[mag_filter]/1000, label=obs.station_id)
            ax_lag.plot(sim_lag, obs.model_ht[mag_filter]/1000)



    # Compute magnitude residuals from the .met file
    if met_obs is not None:

        # Plot magnitudes for all sites
        for site in met_obs.sites:

            # Extract data (filter out inf values)
            height_data = met_obs.height_data[site][~np.isinf(met_obs.abs_mag_data[site])]
            abs_mag_data = met_obs.abs_mag_data[site][~np.isinf(met_obs.abs_mag_data[site])]

            # Sample the simulated magnitude at the observation heights
            sim_mag_sampled = sim_mag_interp(height_data)

            # Compute the magnitude residual
            mag_res_sum += np.sum(np.abs(abs_mag_data - sim_mag_sampled))
            mag_res_count += len(abs_mag_data)

            # Plot the observed and simulated magnitudes
            if plot_residuals:
                ax_mag.scatter(abs_mag_data, height_data/1000, label=site)
                ax_mag.plot(sim_mag_sampled, height_data/1000)

                ax_magres.scatter(abs_mag_data - sim_mag_sampled, height_data/1000)


    # Compute the average magnitude residual
    if mag_res_count > 0:
        mag_res = mag_res_sum/mag_res_count
    
    # Compute the average length residual
    if len_res_count > 0:
        len_res = len_res_sum/len_res_count


    # Compute the weighted residuals
    mag_res = mag_res/mag_sigma
    len_res = len_res/len_sigma

    # Compute the cost function
    cost = mag_res + len_res


    if plot_residuals:

        ax_mag.set_ylabel("Height (km)")
        ax_mag.set_xlabel("Magnitude")
        ax_magres.set_xlabel("Mag residual")

        #ax_len.set_xlabel("Length (m)")
        ax_lag.set_xlabel("Lag (m)")
        ax_lenres.set_xlabel("Len residual (m)")

        # Invert magnitude axes
        ax_mag.invert_xaxis()
        ax_magres.invert_xaxis()

        # # Invert length axes
        # ax_len.invert_yaxis()
        # ax_lenres.invert_yaxis()
        # ax_lag.invert_yaxis()

        plt.tight_layout()
        ax_mag.legend()

        # Show the plot only if hideplots is False
        if not hideplots:
            plt.show()

        else:
            plt.clf()
            plt.close()


    return mag_res, len_res, cost
            


def residualFun(params, fit_options, traj, met_obs, const, change_update_params, hideplots=False):
    """ Take the fit parameters and return the value of the cost function. 
    
    Arguments:
        params: [list] A list of values for the parameters that are being fit.
        fit_options: [dict] A dictionary of options for the fit.
        traj: [Trajectory] A Trajectory object containing the observed data.
        met_obs: [MetObs] A MetObs object containing the observed data from the METAL .met file.
        const: [Constants] Constants defining the erosion model.
        change_update_params: [list] A list of change parameters that should be updated together with their
            nominal parameter.

    Returns:
        cost: [float] The value of the cost function.
    """

    # Make a copy of the constants
    const = copy.deepcopy(const)

    # Update the constants with the fit parameters
    for i, param_name in enumerate(fit_options["fit_params"]):

        # Extract the normalization factor
        norm_fact = fit_options["norm_factors"][i]

        # Compute the unnormalized parameter value
        param_val = norm_fact*params[i]

        setattr(const, param_name, param_val)


        # Update parameters which should be updated after the change, if they are not explicitly refined
        for nominal_param_i, change_param_i in change_update_params:
            if param_name == nominal_param_i:
                setattr(const, change_param_i, param_val)



    # Print values of the fit parameters
    print("Fit parameters: ", end="")
    for i, param_name in enumerate(fit_options["fit_params"]):

        # Extract the normalization factor
        norm_fact = fit_options["norm_factors"][i]

        print("{} = {:.4e}, ".format(param_name, norm_fact*params[i]), end="")
    print()

    # Run the simulation
    sr = SimulationResults(const, *runSimulation(const, compute_wake=False))

    # Extract the weights
    mag_sigma, len_sigma = fit_options["mag_sigma"], fit_options["len_sigma"]

    # Compute the cost function
    mag_res, len_res, cost = costFunc(traj, met_obs, sr, mag_sigma, len_sigma, plot_residuals=False, hideplots=hideplots)

    print("Magnitude residual: {:f}".format(mag_res))
    print("Length residual: {:f}".format(len_res))
    print("Cost: {:f}".format(cost))
    print()

    return cost



def autoFit(fit_options, traj, met_obs, const, hideplots=False):
    """ Automatically fit the parameters to the observations. """


    ### Handle the change parameters ###

    # A list of parameters with variants that defined the changed value at the given height
    change_params = [
        ["erosion_coeff", "erosion_coeff_change"],
        ["rho", "erosion_rho_change"],
        ["sigma", "erosion_sigma_change"]
    ]
    change_params_nominal = [param_i[0] for param_i in change_params]

    # Make a list of change parameters that should follow the nominal parameter
    change_update_params = []
    for param_name in fit_options["fit_params"]:

        # Check if the parameter is one of the change parameters
        if param_name in change_params_nominal:

            # Iterate over the parameters and update them if their change counterpart is not being refined,
            # and is the same as the nominal parameter (meaning that the user didn't set a separamte value 
            # for it)
            for nominal_param_i, change_param_i in change_params:

                # Get value of the change parameters and compare it to the nominal parameter before it was
                # set to the new value by the optimizer
                nominal_param_val = getattr(const, nominal_param_i)
                change_param_val = getattr(const, change_param_i)

                if (param_name == nominal_param_i) \
                    and (change_param_i not in fit_options["fit_params"]) \
                    and (nominal_param_val == change_param_val):

                    # Add the change parameter to the list of parameters to be updated
                    change_update_params.append([nominal_param_i, change_param_i])

    if len(change_update_params):
        print()
        print("Change parameters that will follow the nominal parameter:")
        for nominal_param_i, change_param_i in change_update_params:
            print("    {} -> {}".format(nominal_param_i, change_param_i))
        print()

    ### ###


    # Make a copy of the constants
    const = copy.deepcopy(const)

    # Extract the initial parameters
    x0 = [getattr(const, param_name) for param_name in fit_options["fit_params"]]

    # Store the initial parameters as normalization factors
    fit_options["norm_factors"] = x0

    # Normalize the initial parameters
    x0 = [x0[i]/fit_options["norm_factors"][i] for i in range(len(x0))]

    # Extract the bounds (compute either absolute values or relative values)
    bounds = []
    for i, (bound_type, bound_min, bound_max) in enumerate(fit_options["fit_bounds"]):
            
        # Absolute bounds
        if bound_type == "abs":

            # Normalize the absolute bounds
            if bound_min is not None:
                abs_bound_min = bound_min/fit_options["norm_factors"][i]
            else:
                abs_bound_min = bound_min
            
            if bound_max is not None:
                abs_bound_max = bound_max/fit_options["norm_factors"][i]
            else:
                abs_bound_max = bound_max

            bounds.append((abs_bound_min, abs_bound_max))

        # Relative bounds
        elif bound_type == "rel":

            if bound_min is not None:
                rel_bound_min = bound_min*x0[i]
            else:
                rel_bound_min = bound_min

            if bound_max is not None:
                rel_bound_max = bound_max*x0[i]
            else:
                rel_bound_max = bound_max

            bounds.append((rel_bound_min, rel_bound_max))

        # Invalid bound type
        else:
            raise ValueError("Invalid bound type: {}".format(bound_type))


    # Print initial parameters and bounds for each
    print()
    print("Initial parameters:")

    for i, param_name in enumerate(fit_options["fit_params"]):

        # Extract the normalization factor
        n = fit_options["norm_factors"][i]

        # Compute the parameter value
        param_val = n*x0[i]

        # Compute the bounds
        bound_min = n*bounds[i][0]
        bound_max = n*bounds[i][1]

        # Construct the format string
        decimal_places = 6
        decimal_places_scientific = 2

        # If the parameter value is smaller than possible to represent in the given number of decimal places,
        # then use scientific notation
        if param_val < 10**(-decimal_places):
            format_str = "{{:.{}e}}".format(decimal_places_scientific)
        else:
            format_str = "{{:.{}f}}".format(decimal_places)


        print("\t{}: {} [{}, {}]".format(param_name, format_str.format(param_val),
            format_str.format(bound_min), format_str.format(bound_max)))
        
        bound_error_text = """
******************************************************************************

ERROR: The initial parameter value for {:s} is outside the bounds!

Please check the initial parameter values and the bounds.

Exiting...

******************************************************************************
        """.format(param_name)
        
        # If the intial parameter value is outside the bounds, then throw an error
        if (bound_min is not None) and (param_val < bound_min):
            print(bound_error_text)
            print("Initial parameter value is smaller than the lower bound: {} < {}".format(
                param_val, bound_min))
            sys.exit(1)
        
        if (bound_max is not None) and (param_val > bound_max):
            print(bound_error_text)
            print("Initial parameter value is larger than the upper bound: {} > {}".format(
                param_val, bound_max))
            sys.exit(1)


    print()


    # # Run the optimization using basinhopping
    # res = scipy.optimize.basinhopping(residualFun, x0, 
    #     minimizer_kwargs={"args": (fit_options, traj, met_obs, const), "method": "L-BFGS-B", "bounds": bounds})

    # Run the optimization using Nelder-Mead
    res = scipy.optimize.minimize(residualFun, x0, args=(fit_options, traj, met_obs, const, 
                                                         change_update_params, hideplots), 
        method="Nelder-Mead", bounds=bounds)

    # Extract the optimized parameters into Constants
    for i, param_name in enumerate(fit_options["fit_params"]):

        # Extract the normalization factor
        norm_fact = fit_options["norm_factors"][i]

        setattr(const, param_name, norm_fact*res.x[i])

    # Set the change parameters to follow the nominal parameters
    for nominal_param_i, change_param_i in change_update_params:
        setattr(const, change_param_i, getattr(const, nominal_param_i))

    # Print the results
    print(res)

    # Return the optimized constants
    return const



def loadFitOptions(dir_path, file_name):
    """ Load the fit options from the JSON file. """

    # Open the JSON file
    with open(os.path.join(dir_path, file_name), "r") as f:
        filtered_lines = []
        for line in f:

            # Skip the comments
            if line.strip().startswith("#"):
                continue

            # Skip the empty lines
            if len(line.replace(" ", "").replace("\t", "").replace("\n", "")) == 0:
                continue

            # Replace all "True" and "False" with "true" and "false"
            line = line.replace("True", "true").replace("False", "false")

            # Replaced apostrophes with double quotes
            line = line.replace("'", '"')

            filtered_lines.append(line)

        # Join the lines into a single string
        lines = "".join(filtered_lines)

        # Remove the trailing commas after ] and when the next line starts with a }
        lines = re.sub(r",[\n\s\t]*(?=[}\]])", "", lines)

        # Print the filtered lines
        print(lines)

        # Load the JSON file as an ordered dictionary
        fit_options = json.loads(lines, object_pairs_hook=OrderedDict)


    return fit_options




if __name__ == "__main__":

    import argparse


    #########################

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Refine meteoroid ablation model parameters using automated optimization.")

    arg_parser.add_argument('dir_path', metavar='DIR_PATH', type=str, \
        help="Path to the directory containing the meteor data. The direction has to contain the trajectory pickle file, the simulated parameters .json file, and optionally a METAL .met file with the wide-field lightcurve.")

    arg_parser.add_argument("fit_options_file", metavar="FIT_OPTIONS_FILE", type=str, \
        help="Name of the file containing the fit options. It is assumed the file is located in the same directory as the meteor data.")

    arg_parser.add_argument('--updated', action='store_true', \
        help="Load the updated simulation JSON file insted of the original one.")
    
    arg_parser.add_argument('-x', '--hideplots', \
        help="Don't show generated plots on the screen, just save them to disk.", action="store_true")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    # Check that scipy version is at least 1.7.0 - older versions don't support Nelder-Mead with bounds
    if scipy.__version__ < "1.7.0":
        print("ERROR: scipy version must be at least 1.7.0! Please upgrade scipy.")
        print("If you're using conda, then run:")
        print("conda install -c conda-forge scipy=1.7.3")
        sys.exit(1)



    # Extract the directory path
    dir_path = cml_args.dir_path


    # Select which simulation JSON file to load
    normal_json_suffix = "_sim_fit.json"
    fitted_json_suffix = "_sim_fit_fitted.json"
    if cml_args.updated:
        sim_suffix = fitted_json_suffix
    else:
        sim_suffix = "_sim_fit.json"


    # # Path to meteor data
    # #dir_path = "/home/dvida/Dropbox/UWO/Projects/metsim_fitting/nicks_events/20190831_035548"
    # dir_path = "/home/dvida/Dropbox/UWO/Projects/metsim_fitting/nicks_events/20190927_041513"
    # #dir_path = "/home/dvida/Dropbox/UWO/Projects/metsim_fitting/nicks_events/20200521_075907"
    # #dir_path = "/home/dvida/Dropbox/UWO/Projects/metsim_fitting/nicks_events/20220531_033542"


    ### FIT OPTIONS ###

    # Load the fit options from the JSON file
    if os.path.exists(os.path.join(dir_path, cml_args.fit_options_file)):
        fit_options = loadFitOptions(dir_path, cml_args.fit_options_file)
    else:
        raise FileNotFoundError("Fit options file not found: {}".format(os.path.join(dir_path, 
            cml_args.fit_options_file)))


    # # Dictionary which defines the parameters to fit and the weights
    # # of each parameter in the cost function
    # fit_options = OrderedDict()

    # # Define the magnitude variance (used to weight the cost function)
    # fit_options["mag_sigma"] = 0.2 # mag

    # # Define the length variance (used to weight the cost function)
    # fit_options["len_sigma"] = 2.0 # m


    # # The "fit_params" are names of variables in the Constants from MetSimErosion
    # fit_sets = []
    
    # # The bounds can either be absolute values or a fraction of the initial value. This is defined by either
    # # 'abs' or 'rel' in the tuple. For example, ('abs', 0.0, None) means the parameter cannot be less than 
    # # 0.0 and there is no upper bound. ('rel', 0.5, 2.0) means the parameter cannot be less than 0.5 and
    # # cannot be greater than 2 times the initial value.

    # # Initial adjustment
    # fp = OrderedDict()
    # fp["m_init"] = ('rel', 0.50, 2.00)
    # fp["v_init"] = ('rel', 0.98, 1.02)
    # fit_sets.append(fp)

    # # Multi-parameter refinement
    # fp = OrderedDict()
    # fp["m_init"]               = ('rel', 0.50, 2.00)
    # fp["v_init"]               = ('rel', 0.98, 1.02)
    # fp["rho"]                  = ('rel', 0.80, 1.20)
    # fp["sigma"]                = ('rel', 0.75, 1.25)
    # fp["erosion_coeff"]        = ('rel', 0.75, 1.25)
    # fp["erosion_height_start"] = ('rel', 0.90, 1.10)
    # fit_sets.append(fp)

    # # Erosion refinement
    # fp = OrderedDict()
    # fp["erosion_coeff"]        = ('rel', 0.75, 1.25)
    # fp["erosion_height_start"] = ('rel', 0.90, 1.10)
    # fp["erosion_mass_min"]     = ('rel', 0.20, 4.00)
    # fp["erosion_mass_max"]     = ('rel', 0.20, 4.00)
    # fit_sets.append(fp)

    # ### ###

    # # Add the fit sets to the fit options
    # fit_options["fit_sets"] = fit_sets


    ### FIND INPUT FILES ###

    # Find the _trajectory.pickle file in the dir_path
    traj_path = None
    for file_name in os.listdir(dir_path):
        if file_name.endswith("_trajectory.pickle"):
            traj_path = os.path.join(dir_path, file_name)
            break

    if traj_path is None:
        raise FileNotFoundError("Trajectory pickle file not found in {}".format(dir_path))

    # Load the trajectory
    print("Using trajectory file: {:s}".format(traj_path))
    traj = loadPickle(*os.path.split(os.path.abspath(traj_path)))


    # Find the state.met file
    state_met_path = None
    met_obs = None
    for file_name in sorted(os.listdir(dir_path), reverse=True):
        if file_name.startswith("state") and file_name.endswith(".met"):

            state_met_path = os.path.join(dir_path, file_name)

            # Load the mirfit met file
            met = loadMet(*os.path.split(os.path.abspath(state_met_path)))

            # Check that the met file is a METAL and not mirfit state file
            if not met.mirfit:

                print("Loaded METAL state.met file: {}".format(state_met_path))
                
                # Load the meteor observations
                met_obs = MetObservations(met, traj)

                break

            else:
                state_met_path = None

    
    # Load the simulation params from _sim_fit.json
    sim_params_path = None
    sim_params_path_orig = None
    for file_name in os.listdir(dir_path):

        # Load the desired file
        if file_name.endswith(sim_suffix):
            sim_params_path = os.path.join(dir_path, file_name)

        # Load the original file
        if file_name.endswith(normal_json_suffix):
            sim_params_path_orig = os.path.join(dir_path, file_name)



    if cml_args.updated:

        # If the updated file is not found, use the original file
        if sim_params_path is None:

            sim_params_path = sim_params_path_orig
            
            print()
            print("Using the original simulation params file as the updated file was not found: {}".format(sim_params_path))

    if sim_params_path is None:
        raise Exception("Could not find the simulation params file in {}".format(dir_path))

    # Load the simulation constants
    const, _ = loadConstants(sim_params_path)
    print()
    print("Loaded simulation constants from: {}".format(sim_params_path))
    print()
        
    ### ###


    ### Fit the atmospheric density polynomial ###


    # Determine the height range for fitting the density
    dens_fit_ht_beg = const.h_init
    dens_fit_ht_end = traj.rend_ele - 5000
    if dens_fit_ht_end < 15000:
        dens_fit_ht_end = 15000

    print("Atmospheric mass density fit for the range of heights: {:.2f} - {:.2f} km".format(\
        dens_fit_ht_end/1000, dens_fit_ht_beg/1000))

    # Take mean meteor lat/lon as reference for the atmosphere model
    lat_mean = np.mean([traj.rbeg_lat, traj.rend_lat])
    lon_mean = meanAngle([traj.rbeg_lon, traj.rend_lon])

    # Fit the polynomail describing the density
    dens_co = fitAtmPoly(lat_mean, lon_mean, dens_fit_ht_end, dens_fit_ht_beg, traj.jdt_ref)


    # Assign the density coefficients
    const.dens_co = dens_co

    ### ###


    # Run the simulation and extract simulation results
    print("Running the simulation...")
    sr = SimulationResults(const, *runSimulation(const, compute_wake=False))
    print("Done!")

    # Cost function test
    costFunc(traj, met_obs, sr, fit_options["mag_sigma"], fit_options["len_sigma"], plot_residuals=True, hideplots=cml_args.hideplots)


    # Go though all the fit sets
    for fp in fit_options["fit_sets"]:

        # Run the automated fitting only if the fit set is enabled
        if fp["enabled"]:

            # Split the parameter names and bounds into separate lists
            fit_options["fit_params"] = [key for key in fp.keys() if key != "enabled"]
            fit_options["fit_bounds"] = [fp[key] for key in fp.keys() if key != "enabled"]

            # Run the automated fitting
            print()
            print("#"*80)
            print("Auto fitting...")
            const = autoFit(fit_options, traj, met_obs, const, hideplots=cml_args.hideplots)
            print("Fitting done!")


    # Save the fitted constants
    saveConstants(const, dir_path, 
        os.path.basename(sim_params_path).replace(normal_json_suffix, fitted_json_suffix))


    # Run the simulation and extract simulation results
    print("Running the simulation...")
    sr = SimulationResults(const, *runSimulation(const, compute_wake=False))
    print("Done!")


    # Cost function test
    costFunc(traj, met_obs, sr, fit_options["mag_sigma"], fit_options["len_sigma"], plot_residuals=True, hideplots=cml_args.hideplots)





    
    # Interpolate the simulated length by height
    sim_len_interp = scipy.interpolate.interp1d(sr.leading_frag_height_arr, sr.leading_frag_length_arr,
        bounds_error=False, fill_value='extrapolate')

    # Interpolate the simulated time
    sim_time_interp = scipy.interpolate.interp1d(sr.leading_frag_height_arr, sr.time_arr,
        bounds_error=False, fill_value='extrapolate')

    # Interpolate the velocity
    sim_vel_interp = scipy.interpolate.interp1d(sr.leading_frag_height_arr, sr.leading_frag_vel_arr,
        bounds_error=False, fill_value='extrapolate')


    # Find the simulated length at the trajectory begining
    sim_len_beg = sim_len_interp(traj.rbeg_ele)

    # Find the simulated time at the trajectory begining
    sim_time_beg = sim_time_interp(traj.rbeg_ele)

    # Find the simulated velocity at the trajectory begining
    sim_vel_beg = sim_vel_interp(traj.rbeg_ele)

    # Set the simulated length at the beginning of observations to zero
    norm_sim_len = sr.leading_frag_length_arr - sim_len_beg

    # Compute the normalized time
    norm_sim_time = sr.time_arr - sim_time_beg


    norm_sim_ht = sr.leading_frag_height_arr[norm_sim_len > 0]
    norm_sim_time = norm_sim_time[norm_sim_len > 0]
    norm_sim_len = norm_sim_len[norm_sim_len > 0]

    # Interpolate the normalized length by time
    sim_norm_len_interp = scipy.interpolate.interp1d(norm_sim_time, norm_sim_len,
        bounds_error=False, fill_value='extrapolate')

    # Interpolate the height by normalized length
    sim_norm_ht_interp = scipy.interpolate.interp1d(norm_sim_len, norm_sim_ht,
        bounds_error=False, fill_value='extrapolate')


    # Compute the simulated lag
    sim_lag = sim_norm_len_interp(norm_sim_time) - sim_vel_beg*norm_sim_time

    # Compute the height for the simulated lag
    sim_lag_ht = norm_sim_ht


    ### ###

        

    
    fig, (ax_mag, ax_vel, ax_lag, ax_lagres) = plt.subplots(ncols=4, figsize=(14, 8), sharey=True)


    ### Plot the observations ###

    ht_max = -np.inf
    ht_min = np.inf

    # Plot the observations from the traj file
    for obs in traj.observations:

        # Plot the magnitude
        if obs.absolute_magnitudes is not None:

            mag_filter = obs.absolute_magnitudes < 9

            ax_mag.plot(obs.absolute_magnitudes[mag_filter], obs.model_ht[mag_filter]/1000, marker='x', ms=8, alpha=0.5, label=obs.station_id)


        # Compute the observed lag
        obs_lag = obs.state_vect_dist - sim_vel_beg*obs.time_data

        # Compute the corrected heights, so the simulations and the observations match
        obs_ht = sim_norm_ht_interp(obs.state_vect_dist)

        # # Use observed heights
        # obs_ht = obs.model_ht

        # Plot the observed lag
        lag_handle = ax_lag.plot(obs_lag, obs_ht/1000, 'x', ms=8, alpha=0.5, linestyle='dashed', 
            label=obs.station_id, markersize=10, linewidth=2)


        # Plot the velocity
        ax_vel.plot(obs.velocities[1:]/1000, obs_ht[1:]/1000, 'x', ms=8, alpha=0.5, linestyle='dashed', 
            label=obs.station_id, markersize=10, linewidth=2)

        # Update the min/max height
        ht_max = max(ht_max, np.max(obs.model_ht/1000))
        ht_min = min(ht_min, np.min(obs.model_ht/1000))


        # Sample the simulated normalized length at observed times
        sim_norm_len_sampled = sim_norm_len_interp(obs.time_data)

        # Compute the length residuals
        len_res = obs.state_vect_dist - sim_norm_len_sampled

        # Plot the length residuals
        ax_lagres.scatter(len_res, obs_ht/1000, marker='+', \
            c=lag_handle[0].get_color(), label="Leading, {:s}".format(obs.station_id), s=6)


    # Plot the magnitudes from the met file
    if met_obs is not None:

        # Plot magnitudes for all sites
        for site in met_obs.sites:

            # Extract data
            abs_mag_data = met_obs.abs_mag_data[site]
            height_data = met_obs.height_data[site]/1000

            # Plot the data
            ax_mag.plot(abs_mag_data, height_data, marker='x', linestyle='dashed', label=str(site), 
                markersize=5, linewidth=1)

            # Update the min/max height
            ht_max = max(ht_max, np.max(height_data))
            ht_min = min(ht_min, np.min(height_data))


    # Store mag range before simulations are plotted
    mag_min, mag_max = ax_mag.get_xlim()

    # Store the velocity range before simulations are plotted
    vel_min, vel_max = ax_vel.get_xlim()

    # Store the lag range before simulations are plotted
    lag_min, lag_max = ax_lag.get_xlim()

    ### ###


    ### Plot the simulation results ###

    # Plot the magnitude
    ax_mag.plot(sr.abs_magnitude, sr.leading_frag_height_arr/1000, label='Simulated', color='k', alpha=0.5)

    # Plot the velocity
    ax_vel.plot(sr.leading_frag_vel_arr/1000, sr.leading_frag_height_arr/1000, label='Simulated', color='k', alpha=0.5)

    # Plot the lag
    ax_lag.plot(sim_lag, sim_lag_ht/1000, label='Simulated - leading', color='k', alpha=0.5)


   
    ### ###



    # Set the height limits
    ax_mag.set_ylim(ht_min - 5, ht_max + 5)

    # Set the magnitude limits
    ax_mag.set_xlim(mag_max, mag_min)

    # Set the velocity limits
    ax_vel.set_xlim(vel_min, vel_max)

    # Set the lag limits
    ax_lag.set_xlim(lag_min, lag_max)

    ax_mag.set_xlabel("Absolute magnitude")
    ax_mag.set_ylabel("Height (km)")
    ax_mag.legend()

    ax_vel.set_xlabel("Velocity (km/s)")
    ax_lag.set_xlabel("Lag (m)")
    ax_lagres.set_xlabel("Lag residuals (m)")


    plt.tight_layout()
    plt.subplots_adjust(wspace=0.00)
    
    # Show the plot only if hideplots is False
    if not cml_args.hideplots:
        plt.show()

    else:
        plt.clf()
        plt.close()

    ### ###