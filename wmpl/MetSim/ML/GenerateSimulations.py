""" Batch generate meteor simulations using an ablation model and random physical parameters, and store the 
simulations to disk. """


from __future__ import print_function, division, absolute_import, unicode_literals

import os
import json
import copy

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

from wmpl.MetSim.GUI import SimulationResults
from wmpl.MetSim.MetSimErosion import Constants
from wmpl.MetSim.MetSimErosion import runSimulation as runSimulationErosion
from wmpl.Utils.AtmosphereDensity import fitAtmPoly
from wmpl.Utils.Math import padOrTruncate
from wmpl.Utils.OSTools import mkdirP
from wmpl.Utils.TrajConversions import J2000_JD
from wmpl.Utils.Pickling import savePickle, loadPickle
from wmpl.Utils.PyDomainParallelizer import domainParallelizer


### CONSTANTS ###

# Length of data that will be used as an input during training
DATA_LENGTH = 256

# Default number of minimum frames for simulation
MIN_FRAMES_VISIBLE = 10

### ###


class MetParam(object):
    def __init__(self, param_min, param_max):
        """ Container for physical meteor parameters. """

        # Range of values
        self.min, self.max = param_min, param_max

        # Value used in simulation
        self.val = None





##############################################################################################################
### SIMULATION PARAMETER CLASSES ###
### MAKE SURE TO ADD ANY NEW CLASSES TO THE "SIM_CLASSES" VARIABLE!

class ErosionSimParametersCAMO(object):
    def __init__(self):
        """ Range of physical parameters for the erosion model, CAMO system. """


        # Define the reference time for the atmosphere density model as J2000
        self.jdt_ref = J2000_JD.days


        ## Atmosphere density ##
        #   Use the atmosphere density for the time at J2000 and coordinates of Elginfield
        self.dens_co = fitAtmPoly(np.radians(43.19301), np.radians(-81.315555), 60000, 180000, self.jdt_ref)

        ##


        # List of simulation parameters
        self.param_list = []



        ## System parameters ##

        # System limiting magnitude (given as a range)
        self.lim_mag_faintest = +5.8
        self.lim_mag_brightest = +5.0

        # Limiting magnitude for length measurements end (given by a range)
        #   This should be the same as the two value above for all other systems except for CAMO
        self.lim_mag_len_end_faintest = +7.5
        self.lim_mag_len_end_brightest = +6.5


        # Power of a zero-magnitude meteor (Watts)
        self.P_0M = 840

        # System FPS
        self.fps = 80

        # Time lag of length measurements (range in seconds) - accomodate CAMO tracking delay of 8 frames
        #   This should be 0 for all other systems except for the CAMO mirror tracking system
        self.len_delay_min = 8.0/self.fps
        self.len_delay_max = 15.0/self.fps

        # Simulation height range (m) that will be used to map the output to a grid
        self.sim_height = MetParam(70000, 130000)


        ##


        ## Physical parameters

        # Mass range (kg)
        self.m_init = MetParam(5e-7, 1e-3)
        self.param_list.append("m_init")

        # Initial velocity range (m/s)
        self.v_init = MetParam(11000, 72000)
        self.param_list.append("v_init")

        # Zenith angle range
        self.zenith_angle = MetParam(np.radians(0.0), np.radians(80.0))
        self.param_list.append("zenith_angle")

        # Density range (kg/m^3)
        self.rho = MetParam(100, 6000)
        self.param_list.append("rho")

        # Intrinsic ablation coeff range (s^2/m^2)
        self.sigma = MetParam(0.005/1e6, 0.5/1e6)
        self.param_list.append("sigma")

        ##


        ## Erosion parameters ##
        ## Assumes no change in erosion once it starts!

        # Erosion height range
        self.erosion_height_start = MetParam(70000, 130000)
        self.param_list.append("erosion_height_start")

        # Erosion coefficient (s^2/m^2)
        self.erosion_coeff = MetParam(0.0, 1.0/1e6)
        self.param_list.append("erosion_coeff")

        # Mass index
        self.erosion_mass_index = MetParam(1.5, 3.0)
        self.param_list.append("erosion_mass_index")

        # Minimum mass for erosion
        self.erosion_mass_min = MetParam(1e-12, 1e-9)
        self.param_list.append("erosion_mass_min")

        # Maximum mass for erosion
        self.erosion_mass_max = MetParam(1e-11, 1e-7)
        self.param_list.append("erosion_mass_max")

        ## 


        ### Simulation quality checks ###

        # Minimum time above the limiting magnitude (10 frames)
        #   This is a minimum for both magnitude and length!
        self.visibility_time_min = 10.0/self.fps

        ### ###


        ### Added noise ###

        # Standard deviation of the magnitude Gaussian noise
        self.mag_noise = 0.1

        # SD of noise in length (m)
        self.len_noise = 1.0

        ### ###


        ### Fit parameters ###

        # Length of input data arrays that will be given to the neural network
        self.data_length = DATA_LENGTH

        ### ###


        ### Output normalization range ###

        # Height range (m)
        self.ht_min = 70000
        self.ht_max = 130000

        # Magnitude range
        self.mag_faintest = self.lim_mag_faintest
        self.mag_brightest = -2


        # Compute length range
        self.len_min = 0
        self.len_max = self.v_init.max*self.data_length/self.fps


        ### ###


class ErosionSimParametersCAMOWide(object):
    def __init__(self):
        """ Range of physical parameters for the erosion model, CAMO system. """


        # Define the reference time for the atmosphere density model as J2000
        self.jdt_ref = J2000_JD.days


        ## Atmosphere density ##
        #   Use the atmosphere density for the time at J2000 and coordinates of Elginfield
        self.dens_co = fitAtmPoly(np.radians(43.19301), np.radians(-81.315555), 60000, 180000, self.jdt_ref)

        ##


        # List of simulation parameters
        self.param_list = []



        ## System parameters ##

        # System limiting magnitude (given as a range)
        self.lim_mag_faintest = +5.8
        self.lim_mag_brightest = +5.0

        # Limiting magnitude for length measurements end (given by a range)
        #   This should be the same as the two value above for all other systems except for CAMO
        self.lim_mag_len_end_faintest = self.lim_mag_faintest
        self.lim_mag_len_end_brightest = self.lim_mag_brightest


        # Power of a zero-magnitude meteor (Watts)
        self.P_0M = 840

        # System FPS
        self.fps = 80

        # Time lag of length measurements (range in seconds) - accomodate CAMO tracking delay of 8 frames
        #   This should be 0 for all other systems except for the CAMO mirror tracking system
        self.len_delay_min = 0
        self.len_delay_max = 0

        # Simulation height range (m) that will be used to map the output to a grid
        self.sim_height = MetParam(70000, 130000)


        ##


        ## Physical parameters

        # Mass range (kg)
        self.m_init = MetParam(5e-7, 1e-3)
        self.param_list.append("m_init")

        # Initial velocity range (m/s)
        self.v_init = MetParam(11000, 72000)
        self.param_list.append("v_init")

        # Zenith angle range
        self.zenith_angle = MetParam(np.radians(0.0), np.radians(80.0))
        self.param_list.append("zenith_angle")

        # Density range (kg/m^3)
        self.rho = MetParam(100, 6000)
        self.param_list.append("rho")

        # Intrinsic ablation coeff range (s^2/m^2)
        self.sigma = MetParam(0.005/1e6, 0.5/1e6)
        self.param_list.append("sigma")

        ##


        ## Erosion parameters ##
        ## Assumes no change in erosion once it starts!

        # Erosion height range
        self.erosion_height_start = MetParam(70000, 130000)
        self.param_list.append("erosion_height_start")

        # Erosion coefficient (s^2/m^2)
        self.erosion_coeff = MetParam(0.0, 1.0/1e6)
        self.param_list.append("erosion_coeff")

        # Mass index
        self.erosion_mass_index = MetParam(1.5, 3.0)
        self.param_list.append("erosion_mass_index")

        # Minimum mass for erosion
        self.erosion_mass_min = MetParam(5e-12, 1e-8)
        self.param_list.append("erosion_mass_min")

        # Maximum mass for erosion
        self.erosion_mass_max = MetParam(1e-11, 5e-7)
        self.param_list.append("erosion_mass_max")

        ## 


        ### Simulation quality checks ###

        # Minimum time above the limiting magnitude (10 frames)
        #   This is a minimum for both magnitude and length!
        self.visibility_time_min = 10.0/self.fps

        ### ###


        ### Added noise ###

        # Standard deviation of the magnitude Gaussian noise
        self.mag_noise = 0.1

        # SD of noise in length (m)
        self.len_noise = 20.0

        ### ###


        ### Fit parameters ###

        # Length of input data arrays that will be given to the neural network
        self.data_length = DATA_LENGTH

        ### ###


        ### Output normalization range ###

        # Height range (m)
        self.ht_min = 70000
        self.ht_max = 130000

        # Magnitude range
        self.mag_faintest = self.lim_mag_faintest
        self.mag_brightest = -2


        # Compute length range
        self.len_min = 0
        self.len_max = self.v_init.max*self.data_length/self.fps


        ### ###


# List of classed that can be used for data generation and postprocessing
SIM_CLASSES = [ErosionSimParametersCAMO, ErosionSimParametersCAMOWide]
SIM_CLASSES_NAMES = [c.__name__ for c in SIM_CLASSES]

##############################################################################################################


class ErosionSimContainer(object):
    def __init__(self, output_dir, erosion_sim_params, random_seed=None):
        """ Simulation container for the erosion model simulation. """

        self.output_dir = output_dir

        # Structure defining the range of physical parameters
        self.params = erosion_sim_params


        # Init simulation constants
        self.const = Constants()
        self.const.dens_co = self.params.dens_co
        self.const.P_0M = self.params.P_0M

        # Set tau to CAMO faint meteor model
        self.const.lum_eff_type = 5


        # Turn on erosion, but disable erosion change
        self.const.erosion_on = True
        self.const.erosion_height_change = 0

        # Disable disruption
        self.const.disruption_on = False

        # If the random seed was given, set the random state
        if random_seed is not None:
            local_state = np.random.RandomState(random_seed)
        else:
            local_state = np.random.RandomState()

        # Randomly sample physical parameters
        for param_name in self.params.param_list:

            # Get the parameter container
            p = getattr(self.params, param_name)

            # If the maximum grain mass is being computed, make sure it's larger than the minimum grain mass
            if param_name == "erosion_mass_max":
                p.min = self.params.erosion_mass_min.val


            # Draw parameters from a distribution:
            # a) Generate all masses distributed logarithmically
            if (param_name == "m_init") or (param_name == "erosion_mass_min") \
                or (param_name == "erosion_mass_max"):

                p.val = 10**(local_state.uniform(np.log10(p.min), np.log10(p.max)))


            # b) Distribute all other values uniformely
            else:

                # Randomly generate the parameter value using an uniform distribution (and the given seed)
                p.val = local_state.uniform(p.min, p.max)


            # Assign value to simulation contants
            setattr(self.const, param_name, p.val)


        # Make sure the min grain mass is not > max grain mass and vice versa
        # TBD



        # Generate a file name from simulation_parameters
        self.file_name = "erosion_sim_v{:.2f}_m{:.2e}g_rho{:04d}_z{:04.1f}_abl{:.3f}_eh{:05.1f}_er{:.3f}_s{:.2f}".format(self.params.v_init.val/1000, 
            self.params.m_init.val*1000, int(self.params.rho.val), np.degrees(self.params.zenith_angle.val), \
            self.params.sigma.val*1e6, self.params.erosion_height_start.val/1000, \
            self.params.erosion_coeff.val*1e6, self.params.erosion_mass_index.val)



    def getNormalizedInputs(self):
        """ Normalize the inputs to the model to the 0-1 range. """

        # Normalize values in the parameter range
        normalized_values = []
        for param_name in self.params.param_list:

            # Get the parameter container
            p = getattr(self.params, param_name)

            # Compute the normalized value
            val_normed = (p.val - p.min)/(p.max - p.min)

            normalized_values.append(val_normed)


        return normalized_values
        


    def denormalizeInputs(self):
        """ Rescale input parametrs to physical values. """

        pass



    def normalizeSimulations(self, params, ht_data, len_data, mag_data):
        """ Normalize simulated data to 0-1 range. """

        ht_normed  = (ht_data - params.ht_min)/(params.ht_max - params.ht_min)
        len_normed = (len_data - params.len_min)/(params.len_max - params.len_min)
        mag_normed = (mag_data - params.mag_brightest)/(params.mag_faintest - params.mag_brightest)

        return ht_normed, len_normed, mag_normed



    def denormalizeSimulations(self):
        """ Rescale outputs to physical values. """

        pass


    def saveJSON(self):
        """ Save object as a JSON file. """


        # Create a copy of the simulation parameters
        self2 = copy.deepcopy(self)

        # Convert the density parameters to a list
        if isinstance(self2.const.dens_co, np.ndarray):
            self2.const.dens_co = self2.const.dens_co.tolist()
        if isinstance(self2.params.dens_co, np.ndarray):
            self2.params.dens_co = self2.params.dens_co.tolist()


        # Convert all simulation parameters to lists
        for sim_res_attr in self2.simulation_results.__dict__:
            attr = getattr(self2.simulation_results, sim_res_attr)
            if isinstance(attr, np.ndarray):
                setattr(self2.simulation_results, sim_res_attr, attr.tolist())


        file_path = os.path.join(self.output_dir, self.file_name + ".json")
        with open(file_path, 'w') as f:
            json.dump(self2, f, default=lambda o: o.__dict__, indent=4)

        print("Saved fit parameters to:", file_path)



    def loadJSON(self):
        """ Load results from a JSON file. """
        pass



    def runSimulation(self):
        """ Run the ablation model and srote results. 
        
        Returns:
            [str]: Path to the save picke file.
        """


        # Run the erosion simulation
        frag_main, results_list, wake_results = runSimulationErosion(self.const, compute_wake=False)

        # Store simulation results
        self.simulation_results = SimulationResults(self.const, frag_main, results_list, wake_results)

        # Delete constants in the simulation results
        if hasattr(self.simulation_results, "const"):
            del self.simulation_results.const

        # Compute synthetic observations
        res = extractSimData(self, check_only=True)
        self.time_sampled = None
        self.ht_sampled = None
        self.len_sampled = None
        self.mag_sampled = None
        if res is not None:
            _, self.time_sampled, self.ht_sampled, self.len_sampled, self.mag_sampled, _, \
                _ = extractSimData(self, check_only=False)

            # Convert synthetic data to lists
            self.time_sampled = self.time_sampled.tolist()
            self.ht_sampled = self.ht_sampled.tolist()
            self.len_sampled = self.len_sampled.tolist()
            self.mag_sampled = self.mag_sampled.tolist()


        ### Sort saved files into a directory structure split by velocity and density ###

        # Extract the velocity part
        split_file = self.file_name.split("_")
        vel = float(split_file[2].strip("v"))

        # Make velocity folder name
        vel_folder = "v{:02d}".format(int(vel))
        vel_folder_path = os.path.join(self.output_dir, vel_folder)

        # Create the velocity folder if it doesn't already exist
        if not os.path.isdir(vel_folder_path):
            os.makedirs(vel_folder_path)


        # Extract the density part
        dens = 100*int(float(split_file[4].strip("rho"))/100)

        # Make density folder name
        dens_folder = "rho{:04d}".format(dens)
        dens_folder_path = os.path.join(vel_folder_path, dens_folder)

        # Make the density folder
        if not os.path.isdir(dens_folder_path):
            os.makedirs(dens_folder_path)


        ###


        # Save results as a JSON file
        self.saveJSON()

        # Save results as a pickle file
        savePickle(self, dens_folder_path, self.file_name + ".pickle")

        return os.path.join(dens_folder_path, self.file_name + ".pickle")




def extractSimData(sim, min_frames_visible=MIN_FRAMES_VISIBLE, check_only=False, param_class_name=None, \
    postprocess_params=None):
    """ Extract input parameters and model outputs from the simulation container and normalize them. 

    Arguments:
        sim: [ErosionSimContainer object] Container with the simulation.

    Keyword arguments:
        min_frames_visible: [int] Minimum number of frames above the limiting magnitude
        check_only: [bool] Only check if the simulation satisfies filters, don' compute eveything.
            Speed up the evaluation. False by default.
        param_class_name: [str] Override the simulation parameters object with an instance of the given
            class. An exact name of the class needs to be given.
        postprocess_params: [list] A list of limiting magnitude for wide and narrow fields, and the delay in
            length measurements. None by default, in which case they will be generated herein.

    Return: 
        - None if the simulation does not satisfy filter conditions.
        - postprocess_params if check_only=True and the simulation satisfies the conditions.
        - params, ht_sampled, len_sampled, mag_sampled, input_data_normed, simulated_data_normed 
            if check_only=False and the simulation satisfies the conditions.

    """

    # Create a frash instance of the system parameters if the same parameters are used as in the simulation
    if param_class_name is None:
        #params_obj = getattr(GenerateSimulations, sim.params.__class__.__name__)
        params = globals()[sim.params.__class__.__name__]()

    # Override the system parameters using the given class
    else:
        params = globals()[param_class_name]()



    ### DRAW LIMITING MAGNITUDE AND LENGTH DELAY ###

    # If the drawn values have already been given, use them
    if postprocess_params is not None:
        lim_mag, lim_mag_len, len_delay = postprocess_params

    else:

        # Draw limiting magnitude and length end magnitude
        lim_mag     = np.random.uniform(params.lim_mag_brightest, params.lim_mag_faintest)
        lim_mag_len = np.random.uniform(params.lim_mag_len_end_brightest, params.lim_mag_len_end_faintest)

        # Draw the length delay
        len_delay = np.random.uniform(params.len_delay_min, params.len_delay_max)

        postprocess_params = [lim_mag, lim_mag_len, len_delay]


    lim_mag_faintest  = np.max([lim_mag, lim_mag_len])
    lim_mag_brightest = np.min([lim_mag, lim_mag_len])

    ### ###

    # Fix NaN values in the simulated magnitude
    sim.simulation_results.abs_magnitude[np.isnan(sim.simulation_results.abs_magnitude)] \
        = np.nanmax(sim.simulation_results.abs_magnitude)


    # Get indices that are above the faintest limiting magnitude
    indices_visible = sim.simulation_results.abs_magnitude <= lim_mag_faintest

    # If no points were visible, skip this solution
    if not np.any(indices_visible):
        return None

    ### CHECK METEOR VISIBILITY WITH THE BRIGTHER (DETECTION) LIMITING MAGNITUDE ###
    ###     (in the CAMO widefield camera)                                       ###

    # Get indices of magnitudes above the brighter limiting magnitude
    indices_visible_brighter = sim.simulation_results.abs_magnitude <= lim_mag_brightest

    # If no points were visible, skip this solution
    if not np.any(indices_visible_brighter):
        return None

    # Compute the minimum time the meteor needs to be visible
    min_time_visible = min_frames_visible/params.fps + len_delay

    time_lim_mag_bright  = sim.simulation_results.time_arr[indices_visible_brighter]
    time_lim_mag_bright -= time_lim_mag_bright[0]

    # Check if the minimum time is satisfied
    if np.max(time_lim_mag_bright) < min_time_visible:
        return None

    ### ###

    # Get the first index after the magnitude reaches visibility in the wide field
    index_first_visibility = np.argwhere(indices_visible_brighter)[0][0]

    # Set all visibility indices before the first one visible in the wide field to False
    indices_visible[:index_first_visibility] = False


    # Select time, magnitude, height, and length above the visibility limit
    time_visible = sim.simulation_results.time_arr[indices_visible]
    mag_visible  = sim.simulation_results.abs_magnitude[indices_visible]
    ht_visible   = sim.simulation_results.brightest_height_arr[indices_visible]
    len_visible  = sim.simulation_results.brightest_length_arr[indices_visible]


    # Resample the time to the system FPS
    mag_interpol = scipy.interpolate.CubicSpline(time_visible, mag_visible)
    ht_interpol  = scipy.interpolate.CubicSpline(time_visible, ht_visible)
    len_interpol = scipy.interpolate.CubicSpline(time_visible, len_visible)

    # Create a new time array according to the FPS
    time_sampled = np.arange(np.min(time_visible), np.max(time_visible), 1.0/params.fps)

    # Create new mag, height and length arrays at FPS frequency
    mag_sampled = mag_interpol(time_sampled)
    ht_sampled = ht_interpol(time_sampled)
    len_sampled = len_interpol(time_sampled)


    # Normalize time to zero
    time_sampled -= time_sampled[0]


    ### SIMULATE CAMO tracking delay for length measurements ###

    # Zero out all length measurements before the length delay (to simulate the delay of CAMO
    #   tracking)
    len_sampled[time_sampled < len_delay] = 0

    ###

    # Set all magnitudes below the brightest limiting magnitude to the faintest magnitude
    mag_sampled[mag_sampled > lim_mag] = params.lim_mag_len_end_faintest


    # Normalize the first length to zero
    first_length_index = np.argwhere(time_sampled >= len_delay)[0][0]
    len_sampled[first_length_index:] -= len_sampled[first_length_index]


    # ### Plot simulated data
    # fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)
    
    # ax1.plot(time_sampled, mag_sampled)
    # ax1.invert_yaxis()
    # ax1.set_ylabel("Magnitude")

    # ax2.plot(time_sampled, len_sampled/1000)
    # ax2.set_ylabel("Length (km)")

    # ax3.plot(time_sampled, ht_sampled/1000)
    # ax3.set_xlabel("Time (s)")
    # ax3.set_ylabel("Height (km)")

    # plt.subplots_adjust(hspace=0)

    # plt.show()

    # ### ###

    

    # Check that there are any length measurements
    if not np.any(len_sampled > 0):
        return None


    # If the simulation should only be checked that it's good, return the postprocess parameters used to 
    #   generate the data
    if check_only:
        return postprocess_params


    ### ADD NOISE ###

    # Add noise to magnitude data
    mag_sampled[mag_sampled <= lim_mag] += np.random.normal(loc=0.0, scale=params.mag_noise, \
        size=len(mag_sampled[mag_sampled <= lim_mag]))

    # Add noise to length data
    len_sampled[first_length_index:] += np.random.normal(loc=0.0, scale=params.len_noise, \
        size=len(len_sampled[first_length_index:]))

    ### ###

    # Construct input data vector with normalized values
    input_data_normed = sim.getNormalizedInputs()


    # Normalize simulated data
    ht_normed, len_normed, mag_normed = sim.normalizeSimulations(params, ht_sampled, len_sampled, mag_sampled)


    # Generate vector with simulated data
    simulated_data_normed = np.vstack([padOrTruncate(ht_normed, params.data_length), \
        padOrTruncate(len_normed, params.data_length), \
        padOrTruncate(mag_normed, params.data_length)])


    # Return input data and results
    return params, time_sampled, ht_sampled, len_sampled, mag_sampled, input_data_normed, simulated_data_normed




def generateErosionSim(output_dir, erosion_sim_params, random_seed, min_frames_visible=MIN_FRAMES_VISIBLE):
    """ Randomly generate parameters for the erosion simulation, run it, and store results. """

    # Init simulation container
    erosion_cont = ErosionSimContainer(output_dir, copy.deepcopy(erosion_sim_params), random_seed=random_seed)
    file_name = erosion_cont.file_name
    print("Running:", erosion_cont.file_name)

    # Run the simulation and save results
    file_path = erosion_cont.runSimulation()

    # Check if the simulation satisfies the visibility criteria
    res = extractSimData(erosion_cont, min_frames_visible=min_frames_visible, check_only=True)
        
    # Free up memory
    del erosion_cont

    if res is not None:
        return [file_path, res]

    else:
        return None



def saveProcessedList(data_path, results_list, param_class_name, min_frames_visible):
    """ Save a list of pickle files which passes postprocessing criteria to disk.

    Arguments:
        data_path: [str] Path to directory with simulation pickle files.
        results_list: [list] A list of pickle files which passes the filers, plus randomly drawn parameters 
            such as the limiting magnitude. If the pickle file didn't pass the filter, None is the entry.
        param_class_name: [str] Name of the parameter class used for postprocessing.
        min_frame_visible: [int] Minimum number of frames above the limting magnitude.

    """

    # Reject all None's from the results
    good_list = [entry for entry in results_list if entry is not None]

    # Load one simulation to get simulation parameters
    sim = loadPickle(*os.path.split(good_list[0][0]))

    # Compute the average minimum time the meteor needs to be visible
    min_time_visible = min_frames_visible/sim.params.fps \
        + (sim.params.len_delay_min + sim.params.len_delay_max)/2

    # Save the list of good files to disk
    simulation_resuts_file = "{:s}_lm{:+04.1f}_mintime{:.3f}s_good_files.txt".format(param_class_name, \
        (sim.params.lim_mag_faintest + sim.params.lim_mag_brightest)/2, min_time_visible)

    # If the file exists, append to it
    append = False
    if os.path.isfile(simulation_resuts_file):
        file_mode = 'a'
        append = True
    else:
        file_mode = 'w'

    with open(os.path.join(data_path, simulation_resuts_file), file_mode) as f:

        # Write the header when the file is created
        if not append:

            ### Write header ###

            # Write name of class used for postprocessing
            if param_class_name is None:
                param_class_name = sim.params.__class__.__name__
            f.write("# param_class_name = {:s}\n".format(param_class_name))

            # Write column labels
            f.write("# File name, lim mag, lim mag length, length delay (s)\n")

            ### ###

        # Write entries
        for file_name, random_params in good_list:
            f.write("{:s}, {:.8f}, {:.8f}, {:.8f}\n".format(file_name, *random_params))

    print("{:d} entries saved to {:s}".format(len(good_list), simulation_resuts_file))



if __name__ == "__main__":

    import argparse


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Randomly generate parameters for the erosion model, run it, and store results to disk.")

    arg_parser.add_argument('output_dir', metavar='OUTPUT_PATH', type=str, \
        help="Path to the output directory.")

    arg_parser.add_argument('simclass', metavar='SIM_CLASS', type=str, \
        help="Use simulation parameters from the given class. Options: {:s}".format(", ".join(SIM_CLASSES_NAMES)))

    arg_parser.add_argument('nsims', metavar='SIM_NUM', type=int, \
        help="Number of simulations to do.")

    arg_parser.add_argument('--cores', metavar='CORES', type=int, default=None, \
        help="Number of cores to use. All by default.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    # Make the output directory
    mkdirP(cml_args.output_dir)

    # Init simulation parameters with the given class name
    erosion_sim_params = SIM_CLASSES[SIM_CLASSES_NAMES.index(cml_args.simclass)]()

    # Generate simulations using multiprocessing
    input_list = [[cml_args.output_dir, copy.deepcopy(erosion_sim_params), \
        np.random.randint(0, 2**31 - 1)] for _ in range(cml_args.nsims)]
    results_list = domainParallelizer(input_list, generateErosionSim, cores=cml_args.cores)


    # Save the list of simulations that passed the criteria to disk
    saveProcessedList(cml_args.output_dir, results_list, erosion_sim_params.__class__.__name__, \
        MIN_FRAMES_VISIBLE)