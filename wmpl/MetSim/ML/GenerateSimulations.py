""" Batch generate meteor simulations using an ablation model and random physical parameters, and store the 
simulations to disk. """


from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import os
import time
import traceback
from typing import Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
from numpy.typing import ArrayLike
from wmpl.MetSim.GUI import SimulationResults
from wmpl.MetSim.MetSimErosion import Constants
from wmpl.MetSim.MetSimErosion import runSimulation as runSimulationErosion
from wmpl.Utils.AtmosphereDensity import fitAtmPoly
from wmpl.Utils.Math import padOrTruncate
from wmpl.Utils.OSTools import mkdirP
from wmpl.Utils.Pickling import loadPickle, savePickle
from wmpl.Utils.PyDomainParallelizer import domainParallelizer
from wmpl.Utils.TrajConversions import J2000_JD

### CONSTANTS ###

# Length of data that will be used as an input during training
DATA_LENGTH = 512

# Default number of minimum frames for simulation
MIN_FRAMES_VISIBLE = 10

### ###


class MetParam(object):
    def __init__(self, param_min: float, param_max: float, gen_method: str = 'uniform', default=None):
        """ Container for physical meteor parameters. """

        # Range of values
        self.min, self.max = param_min, param_max

        # Value used in simulation
        self.val = default

        # functions to give "smartness" to parameter
        self.method = gen_method
        self.link_method = (None, None)

    def generateVal(self, local_state=None):
        if local_state is None:
            local_state = np.random.RandomState()

        # setting the min and max based on the param it was linked to
        if self.link_method[0] == 'greater':
            _min, _max = (max(self.link_method[1].val, self.min), self.max)
        elif self.link_method[0] == 'less':
            _min, _max = (self.min, min(self.max, self.link_method[1].val))
        else:
            _min, _max = (self.min, self.max)

        # generating val from the min and max value, based on a probability distribution
        if self.method == 'uniform':
            self.val = local_state.uniform(_min, _max)
        elif self.method == 'log10':
            self.val = 10 ** (local_state.uniform(np.log10(_min), np.log10(_max)))

        return self.val

    def linkParam(self, param, method: Optional[str] = None):
        """
        Required relationship between current parameter and given parameter
        
        Arguments:
            param: [MetParam] 
            method: [str] 'greater' or 'less'. If 'greater', requires that this parameter is greater than
                the inputted parameter.
        """
        self.link_method = (method, param)

    def __str__(self):
        return str(self.val)


##############################################################################################################
### SIMULATION PARAMETER CLASSES ###
### MAKE SURE TO ADD ANY NEW CLASSES TO THE "SIM_CLASSES" VARIABLE!


class PhysicalParameters:
    def __init__(self):
        # Define the reference time for the atmosphere density model as J2000
        self.jdt_ref = J2000_JD.days

        ## Atmosphere density ##
        #   Use the atmosphere density for the time at J2000 and coordinates of Elginfield
        self.dens_co = fitAtmPoly(np.radians(43.19301), np.radians(-81.315555), 60000, 180000, self.jdt_ref)

        # Power of a zero-magnitude meteor (Watts)
        self.P_0M = 840

        ## Physical parameters

        # List of simulation parameters
        self.param_list = []

        # Mass range (kg)
        self.m_init = MetParam(5e-7, 1e-3, 'log10')
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
        self.sigma = MetParam(0.005 / 1e6, 0.5 / 1e6)
        self.param_list.append("sigma")

        ##

        ## Erosion parameters ##
        ## Assumes no change in erosion once it starts!

        # Erosion height range
        self.erosion_height_start = MetParam(70000, 130000)
        self.param_list.append("erosion_height_start")

        # Erosion coefficient (s^2/m^2)
        self.erosion_coeff = MetParam(0.0, 1.0 / 1e6)
        self.param_list.append("erosion_coeff")

        # Mass index
        self.erosion_mass_index = MetParam(1.5, 3.0)
        self.param_list.append("erosion_mass_index")

        # Minimum mass for erosion
        self.erosion_mass_min = MetParam(1e-12, 1e-9, 'log10')
        self.param_list.append("erosion_mass_min")

        # Maximum mass for erosion
        self.erosion_mass_max = MetParam(1e-11, 1e-7, 'log10')
        self.erosion_mass_max.linkParam(self.erosion_mass_min, 'greater')
        self.param_list.append("erosion_mass_max")

    def setParamValues(self, vals):
        for val, param in zip(vals, self.param_list):
            getattr(self, param).val = val

    def getInputs(self) -> List[float]:
        return [getattr(self, param_name).val for param_name in self.param_list]

    def getNormalizedInputs(self) -> List[float]:
        """ Normalize the physical parameters of the model to the 0-1 range. """

        # Normalize values in the parameter range
        normalized_values = []
        for param_name in self.param_list:

            # Get the parameter container
            p = getattr(self, param_name)

            # Compute the normalized value
            val_normed = (p.val - p.min) / (p.max - p.min)

            normalized_values.append(val_normed)

        return normalized_values

    def getDenormalizedInputs(self, norm_val_list: List[float]) -> List[float]:
        denormalized_values = []
        for norm_val, param_name in zip(norm_val_list, self.param_list):

            # Get the parameter container
            p = getattr(self, param_name)

            # Compute the normalized value
            val_denormed = norm_val * (p.max - p.min) + p.min

            denormalized_values.append(val_denormed)

        return denormalized_values

    def getConst(self, random_seed: int = None, override: bool = False):
        # Init simulation constants
        const = Constants()
        const.dens_co = self.dens_co
        const.P_0M = self.P_0M

        # Turn on erosion, but disable erosion change
        const.erosion_on = True
        const.erosion_height_change = 0

        # Disable disruption
        const.disruption_on = False

        if random_seed is not None:
            local_state = np.random.RandomState(random_seed)
        else:
            local_state = np.random.RandomState()

        # Randomly sample physical parameters
        for param_name in self.param_list:

            # Get the parameter container
            p = getattr(self, param_name)

            if p.val is None or override or p.val > p.max or p.val < p.min:
                p.generateVal(local_state)

            # Assign value to simulation contants
            setattr(const, param_name, p.val)
        return const


class ErosionSimParameters:
    def __init__(self):
        ## System parameters ##

        # System limiting magnitude. Starting LM is the magnitude when the camera will start observing
        # (the first instance) the meteor, and ending LM is the magnitude where it will stop observing
        # (the last instance)
        self.starting_lim_mag = MetParam(5, 10, default=8)
        self.ending_lim_mag = MetParam(5, 10, default=8)

        # if lightcurve doesn't reach this magnitude, it's discarded
        self.peak_mag_faintest = 6

        # System FPS
        self.fps = 100

        # Time lag of length measurements (range in seconds) - accomodate CAMO tracking delay of 8 frames
        #   This should be 0 for all other systems except for the CAMO mirror tracking system
        self.len_delay = MetParam(8.0 / self.fps, 15.0 / self.fps, default=15 / self.fps)

        # Simulation height range (m) that will be used to map the output to a grid
        self.sim_height = MetParam(70000, 130000)

        ##

        ### Simulation quality checks ###

        # Minimum time above the limiting magnitude (10 frames)
        #   This is a minimum for both magnitude and length!
        self.visibility_time_min = 0.2

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
        self.mag_faintest = 10
        self.mag_brightest = -2


class ErosionSimParametersCAMO(ErosionSimParameters):
    def __init__(self):
        """ Range of physical parameters for the erosion model, CAMO system. """
        super().__init__()
        ## System parameters ##

        # System limiting magnitude. Starting LM is the magnitude when the camera will start observing
        # (the first instance) the meteor, and ending LM is the magnitude where it will stop observing
        # (the last instance)
        self.starting_lim_mag = MetParam(5, 10, default=8)  # to be adjusted
        self.ending_lim_mag = MetParam(5, 10, default=8)  # to be adjusted

        # if lightcurve doesn't reach this magnitude, it's discarded
        self.peak_mag_faintest = 6  # to be adjusted

        # System FPS
        self.fps = 80

        # Time lag of length measurements (range in seconds) - accomodate CAMO tracking delay of 8 frames
        #   This should be 0 for all other systems except for the CAMO mirror tracking system
        self.len_delay = MetParam(8.0 / self.fps, 15.0 / self.fps, default=15 / self.fps)

        # Simulation height range (m) that will be used to map the output to a grid
        self.sim_height = MetParam(70000, 130000)

        ##

        ### Simulation quality checks ###

        # Minimum time above the limiting magnitude (10 frames)
        #   This is a minimum for both magnitude and length!
        self.visibility_time_min = 0.2

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
        self.mag_faintest = 10
        self.mag_brightest = -2

        ### ###


class ErosionSimParametersCAMOWide(ErosionSimParameters):
    def __init__(self):
        """ Range of physical parameters for the erosion model, CAMO system. """
        super().__init__()
        ## System parameters ##

        # System limiting magnitude. Starting LM is the magnitude when the camera will start observing
        # (the first instance) the meteor, and ending LM is the magnitude where it will stop observing
        # (the last instance)
        self.starting_lim_mag = MetParam(5, 10, default=8)  # to be adjusted
        self.ending_lim_mag = MetParam(5, 10, default=8)  # to be adjusted

        # if lightcurve doesn't reach this magnitude, it's discarded
        self.peak_mag_faintest = 6  # to be adjusted

        # System FPS
        self.fps = 80

        # Time lag of length measurements (range in seconds) - accomodate CAMO tracking delay of 8 frames
        #   This should be 0 for all other systems except for the CAMO mirror tracking system
        self.len_delay = MetParam(0, 0, default=0)

        # Simulation height range (m) that will be used to map the output to a grid
        self.sim_height = MetParam(70000, 130000)

        ##

        ### Simulation quality checks ###

        # Minimum time above the limiting magnitude (10 frames)
        #   This is a minimum for both magnitude and length!
        self.visibility_time_min = 0.2

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
        self.mag_faintest = 10
        self.mag_brightest = -2
        ### ###


# List of classed that can be used for data generation and postprocessing
SIM_CLASSES = [ErosionSimParametersCAMO, ErosionSimParametersCAMOWide]
SIM_CLASSES_NAMES = [c.__name__ for c in SIM_CLASSES]
SIM_CLASSES_DICT = dict(zip(SIM_CLASSES_NAMES, SIM_CLASSES))

##############################################################################################################


class ErosionSimContainer:
    def __init__(self, output_dir: str, random_seed: Optional[int] = None):
        """ Simulation container for the erosion model simulation. """
        self.output_dir = output_dir

        # Structure defining the range of physical parameters
        self.params = PhysicalParameters()

        self.const = self.params.getConst(random_seed)

        # Generate a file name from simulation_parameters
        self.file_name = "erosion_sim_v{:.2f}_m{:.2e}g_rho{:04d}_z{:04.1f}_abl{:.3f}_eh{:05.1f}_er{:.3f}_s{:.2f}".format(
            self.params.v_init.val / 1000,
            self.params.m_init.val * 1000,
            int(self.params.rho.val),
            np.degrees(self.params.zenith_angle.val),
            self.params.sigma.val * 1e6,
            self.params.erosion_height_start.val / 1000,
            self.params.erosion_coeff.val * 1e6,
            self.params.erosion_mass_index.val,
        )

    def getNormalizedInputs(self) -> List[float]:
        """ Normalize the physical parameters of the model to the 0-1 range. """
        return self.params.getNormalizedInputs()

    def denormalizeInputs(self, inputs):
        """ Rescale input parameters to physical values. """
        return self.params.getDenormalizedInputs(inputs)

    def saveJSON(self):
        """ Save object as a JSON file. """

        # Create a copy of the simulation parameters
        self2 = copy.deepcopy(self)

        # Convert the density parameters to a list
        if isinstance(self2.const.dens_co, np.ndarray):
            self2.const.dens_co = self2.const.dens_co.tolist()
        if isinstance(self2.params.phys_params.dens_co, np.ndarray):
            self2.params.phys_params.dens_co = self2.params.phys_params.dens_co.tolist()

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
        """ Run the ablation model and srote results. Stored in self.simulation_results """
        # Run the erosion simulation
        frag_main, results_list, wake_results = runSimulationErosion(self.const, compute_wake=False)

        # Store simulation results
        self.simulation_results = SimulationResults(self.const, frag_main, results_list, wake_results)

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
        dens = 100 * int(float(split_file[4].strip("rho")) / 100)

        # Make density folder name
        dens_folder = "rho{:04d}".format(dens)
        dens_folder_path = os.path.join(vel_folder_path, dens_folder)

        # Make the density folder
        if not os.path.isdir(dens_folder_path):
            os.makedirs(dens_folder_path)

        ###

        # # Save results as a JSON file
        # self.saveJSON()

        # Save results as a pickle file
        savePickle(self, dens_folder_path, self.file_name + ".pickle")
        # print(f'saved to {os.path.join(dens_folder_path, self.file_name + ".pickle")}')


def normalizeSimulations(
    phys_params: PhysicalParameters,
    camera_params: ErosionSimParameters,
    ht_data: ArrayLike,
    len_data: ArrayLike,
    mag_data: ArrayLike,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """ Normalize simulated data to 0-1 range. """
    # Compute length range
    len_min = 0
    len_max = phys_params.v_init.max * camera_params.data_length / camera_params.fps

    ht_normed = (ht_data - camera_params.ht_min) / (camera_params.ht_max - camera_params.ht_min)
    len_normed = (len_data - len_min) / (len_max - len_min)
    mag_normed = (camera_params.mag_faintest - mag_data) / (
        camera_params.mag_faintest - camera_params.mag_brightest
    )

    return ht_normed, len_normed, mag_normed


def extractSimData(
    sim: ErosionSimContainer,
    check_only: bool = False,
    param_class_name: Optional[str] = None,
    camera_params: Optional[ErosionSimParameters] = None,
    add_noise: bool = False,
) -> Optional[tuple]:
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
        - params, input_data_normed, simulated_data_normed if check_only=False and the simulation satisfies 
            the conditions.

    """
    if camera_params is None:
        if param_class_name is not None:
            # Override the system parameters using the given class
            camera_params = SIM_CLASSES_DICT[param_class_name]()
        elif sim.params.__class__.__name__ in SIM_CLASSES_DICT:
            # Create a frash instance of the system parameters if the same parameters are used as in the simulation
            camera_params = SIM_CLASSES_DICT[sim.params.__class__.__name__]()
        else:
            # in case no class name was given and class is storing ErosionSimParameters object. The user should always
            # have a param_class_name supplied, but this is a default.
            camera_params = ErosionSimParameters()

    param_dict = {'camera': camera_params, 'physical': sim.params}
    params = param_dict['camera']

    ### DRAW LIMITING MAGNITUDE AND LENGTH DELAY ###

    # If the drawn values have already been given, use them
    if add_noise:
        # Draw limiting magnitude and length end magnitude and length delay
        starting_lim_mag = params.starting_lim_mag.generateVal()
        ending_lim_mag = params.ending_lim_mag.generateVal()
        len_delay = params.len_delay.generateVal()
    else:
        starting_lim_mag = params.starting_lim_mag.val
        ending_lim_mag = params.ending_lim_mag.val
        len_delay = params.len_delay.val

    ### ###

    # Fix NaN values in the simulated magnitude
    sim.simulation_results.abs_magnitude[np.isnan(sim.simulation_results.abs_magnitude)] = np.nanmax(
        sim.simulation_results.abs_magnitude
    )

    # if the peak magnitude is dimmer than the faintest expected peak magnitude, discard it
    if np.min(sim.simulation_results.abs_magnitude) >= params.peak_mag_faintest:
        return None

    # Get indices that are above the faintest limiting magnitude
    min_lim_mag = min(starting_lim_mag, ending_lim_mag)
    indices_visible = np.ones(sim.simulation_results.abs_magnitude.shape, dtype=bool)
    # filtering out anything before what's visible by the wide camera
    indices_visible[: np.argmax(sim.simulation_results.abs_magnitude <= starting_lim_mag)] = False
    # if last element is too bright filtering ending doesn't do anything
    if sim.simulation_results.abs_magnitude[-1] > ending_lim_mag:
        indices_visible[-np.argmax(sim.simulation_results.abs_magnitude[::-1] <= ending_lim_mag) :] = False
    # filtering out anything dimmer than what's visible by wide and narrow cameras
    indices_visible[sim.simulation_results.abs_magnitude >= min_lim_mag] = False

    # If no points were visible, skip this solution
    if not np.any(indices_visible):
        print('not visible')
        return None

    # Compute the minimum time the meteor needs to be visible
    min_time_visible = params.visibility_time_min

    time_lim_mag_bright = sim.simulation_results.time_arr[indices_visible]
    time_lim_mag_bright -= time_lim_mag_bright[0]

    # Check if the minimum time is satisfied
    if np.max(time_lim_mag_bright) < min_time_visible:
        print('minimum time not satisfied')
        return None

    ### ###

    # Select time, magnitude, height, and length above the visibility limit
    time_visible = sim.simulation_results.time_arr[indices_visible]
    mag_visible = sim.simulation_results.abs_magnitude[indices_visible]
    ht_visible = sim.simulation_results.brightest_height_arr[indices_visible]
    len_visible = sim.simulation_results.brightest_length_arr[indices_visible]

    # Resample the time to the system FPS
    mag_interpol = scipy.interpolate.CubicSpline(time_visible, mag_visible)
    ht_interpol = scipy.interpolate.CubicSpline(time_visible, ht_visible)
    len_interpol = scipy.interpolate.CubicSpline(time_visible, len_visible)

    # Create a new time array according to the FPS
    time_sampled = np.arange(np.min(time_visible), np.max(time_visible), 1.0 / params.fps)

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

    # there should not be any truncation in the data
    if not np.any(len_sampled > 0):
        # print('no length measurements')
        return None

    # If the simulation should only be checked that it's good, return the postprocess parameters used to
    #   generate the data
    if check_only:
        return param_dict

    # TODO: add this
    ### ADD NOISE ###

    # if add_noise:
    #     # Add noise to magnitude data
    #     mag_sampled[mag_sampled <= lim_mag] += np.random.normal(
    #         loc=0.0, scale=params.mag_noise, size=len(mag_sampled[mag_sampled <= lim_mag])
    #     )

    #     # Add noise to length data
    #     len_sampled[first_length_index:] += np.random.normal(
    #         loc=0.0, scale=params.len_noise, size=len(len_sampled[first_length_index:])
    #     )

    ### ###

    # Construct input data vector with normalized values
    input_data_normed = sim.getNormalizedInputs()

    # Normalize simulated data
    ht_normed, len_normed, mag_normed = normalizeSimulations(
        param_dict['physical'], param_dict['camera'], ht_sampled, len_sampled, mag_sampled
    )

    # Generate vector with simulated data
    simulated_data_normed = np.vstack(
        [
            padOrTruncate(ht_normed, params.data_length, side='end'),
            padOrTruncate(len_normed, params.data_length, side='end'),
            padOrTruncate(mag_normed, params.data_length, side='end'),
        ]
    )
    # fig, ax = plt.subplots(4)
    # ax[0].plot(simulated_data_normed[2], simulated_data_normed[0])
    # ax[1].plot(simulated_data_normed[1], simulated_data_normed[0])
    # ax[0].set_xlabel('Mag')
    # ax[0].set_ylabel('Ht')
    # ax[1].set_xlabel('Length')
    # ax[1].set_ylabel('Ht')

    # ax[2].plot(sim.simulation_results.abs_magnitude, sim.simulation_results.brightest_height_arr)
    # ax[3].plot(sim.simulation_results.brightest_length_arr, sim.simulation_results.brightest_height_arr)
    # ax[2].set_xlabel('Mag')
    # ax[2].set_ylabel('Ht')
    # ax[3].set_xlabel('Length')
    # ax[3].set_ylabel('Ht')

    # plt.show()
    # Return input data and results
    return param_dict, input_data_normed, simulated_data_normed


def saveCleanData(output_dir: str, random_seed: int):
    # Init simulation container
    erosion_cont = ErosionSimContainer(output_dir, random_seed=random_seed)
    # print("Running:", erosion_cont.file_name)

    # Run the simulation and save results
    erosion_cont.runSimulation()


# def generateErosionSim(
#     output_dir: str,
#     erosion_sim_params: ErosionSimParameters,
#     random_seed: int,
#     min_frames_visible: int = MIN_FRAMES_VISIBLE,
# ) -> Optional[list]:
#     """ Randomly generate parameters for the erosion simulation, run it, and store results. """

#     # Check if the simulation satisfies the visibility criteria
#     res = extractSimData(erosion_cont, min_frames_visible=min_frames_visible, check_only=True)

#     # Free up memory
#     del erosion_cont

#     if res is not None:
#         return [file_name, res]

#     else:
#         return None


def saveProcessedList(data_path: str, results_list: list, param_class_name: str, min_frames_visible: int):
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
    sim = loadPickle(data_path, good_list[0][0])

    # Compute the average minimum time the meteor needs to be visible
    min_time_visible = (
        min_frames_visible / sim.params.fps + (sim.params.len_delay_min + sim.params.len_delay_max) / 2
    )

    # Save the list of good files to disk
    simulation_resuts_file = "{:s}_lm{:+04.1f}_mintime{:.3f}s_good_files.txt".format(
        param_class_name, (sim.params.lim_mag_faintest + sim.params.lim_mag_brightest) / 2, min_time_visible
    )

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
    arg_parser = argparse.ArgumentParser(
        description="Randomly generate parameters for the rosion model, run it, and store results to disk."
    )

    arg_parser.add_argument(
        'output_dir', metavar='OUTPUT_PATH', type=str, help="Path to the output directory."
    )

    # arg_parser.add_argument(
    #     'simclass',
    #     metavar='SIM_CLASS',
    #     type=str,
    #     help="Use simulation parameters from the given class. Options: {:s}".format(
    #         ", ".join(SIM_CLASSES_NAMES)
    #     ),
    # )

    arg_parser.add_argument('nsims', metavar='SIM_NUM', type=int, help="Number of simulations to do.")

    # group = arg_parser.add_mutually_exclusive_group()
    # group.add_argument(
    #     "-c", "--clean", action='store_true', help='If given, only simulate then save the clean data'
    # )
    # group.add_argument(
    #     "-pp",
    #     "--postprocess",
    #     action='store_true',
    #     help='If given, will only save post-processed data upon reading from saved clean data',
    # )

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    # Make the output directory
    mkdirP(cml_args.output_dir)

    # # Init simulation parameters for CAMO
    # if cml_args.simclass not in SIM_CLASSES_NAMES:
    #     raise KeyError(f'Sim class is not valid: {cml_args.simclass}')
    # erosion_sim_params = SIM_CLASSES_DICT[cml_args.simclass]

    # Generate simulations using multiprocessing
    input_list = [[cml_args.output_dir, np.random.randint(0, 2 ** 31 - 1)] for _ in range(cml_args.nsims)]
    results_list = domainParallelizer(input_list, saveCleanData, display=True)

    # # Save the list of simulations that passed the criteria to disk
    # saveProcessedList(
    #     cml_args.output_dir, results_list, erosion_sim_params.__class__.__name__, MIN_FRAMES_VISIBLE
    # )

