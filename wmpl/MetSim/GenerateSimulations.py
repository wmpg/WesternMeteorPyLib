""" Batch generate meteor simulations using an ablation model and random physical parameters, and store the 
simulations to disk. """


from __future__ import print_function, division, absolute_import, unicode_literals

import os
import json
import copy

import numpy as np

from wmpl.MetSim.GUI import SimulationResults
from wmpl.MetSim.MetSimErosion import Constants
from wmpl.MetSim.MetSimErosion import runSimulation as runSimulationErosion
from wmpl.Utils.AtmosphereDensity import fitAtmPoly
from wmpl.Utils.TrajConversions import J2000_JD
from wmpl.Utils.Pickling import savePickle
from wmpl.Utils.PyDomainParallelizer import domainParallelizer


class MetParam(object):
    def __init__(self, param_min, param_max):
        """ Container for physical meteor parameters. """

        # Range of values
        self.min, self.max = param_min, param_max

        # Value used in simulation
        self.val = None




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

        # System limiting magnitude
        self.lim_mag = +5.5

        # Power of a zero-magnitude meteor (Watts)
        self.P_0M = 840

        # System FPS (possibly separate for magnitude and length)
        self.mag_fps = 80
        self.len_fps = 100

        # Time lag of length measurements (seconds) - accomodate CAMO tracking delay of 8 frames
        self.len_delay = 8.0/80.0

        # Simulation height range (m) that will be used to map the output to a grid
        self.sim_height = MetParam(70000, 130000)


        ##


        ## Physical parameters

        # Mass range (kg)
        self.m_init = MetParam(5e-7, 1e-2)
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
        self.sigma = MetParam(0.001/1e6, 0.5/1e6)
        self.param_list.append("sigma")

        ##


        ## Erosion parameters ##
        ## Assumes no change in erosion once it starts!

        # Erosion height range
        self.erosion_height_start = MetParam(70000, 130000)
        self.param_list.append("erosion_height_start")

        # Erosion coefficient (s^2/m^2)
        self.erosion_coeff = MetParam(0.0, 0.5/1e6)
        self.param_list.append("erosion_coeff")

        # Mass index
        self.erosion_mass_index = MetParam(1.0, 3.0)
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
        self.visibility_time_min = 10.0/self.mag_fps


        ### ###



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

            # Randomly generate the parameter value using an uniform distribution (and the given seed)
            p.val = local_state.uniform(p.min, p.max)

            # Assign value to simulation contants
            setattr(self.const, "param_name", p.val)



        # Generate a file name from simulation_parameters
        self.file_name = "erosion_sim_v{:.2f}_m{:.2e}g_rho{:04d}_z{:04.1f}_abl{:.3f}_eh{:.1f}_er{:.3f}_s{:.2f}".format(self.params.v_init.val/1000, 
            self.params.m_init.val*1000, int(self.params.rho.val), np.degrees(self.params.zenith_angle.val), \
            self.params.sigma.val*1e6, self.params.erosion_height_start.val/1000, \
            self.params.erosion_coeff.val*1e6, self.params.erosion_mass_index.val)



    def normalizeInputs(self):
        """ Normalize the inputs to the model to the 0-1 range. """

        pass


    def denormalizeInputs(self):
        """ Rescale input parametrs to physical values. """

        pass



    def normalizeOutputs(self):
        """ Normalize model output to 0-1 range. """

        pass



    def denormalizeOutputs(self):
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
        """ Run the ablation model and srote results. """


        # Run the erosion simulation
        results_list, wake_results = runSimulationErosion(self.const, compute_wake=False)

        # Store simulation results
        self.simulation_results = SimulationResults(self.const, results_list, wake_results)


        # # Save results as a JSON file
        # self.saveJSON()

        # Save results as a pickle file
        savePickle(self, self.output_dir, self.file_name + ".pickle")





def generateErosionSim(output_dir, random_seed):
    """ Randomly generate parameters for the erosion simulation, run it, and store results. """

    # Init simulation parameters
    erosion_sim_params = ErosionSimParametersCAMO()

    # Init simulation container
    erosion_cont = ErosionSimContainer(output_dir, copy.deepcopy(erosion_sim_params), random_seed=random_seed)

    print("Running:", erosion_cont.file_name)

    # Run the simulation and save results
    erosion_cont.runSimulation()




if __name__ == "__main__":

    import argparse


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Randomly generate parameters for the rosion model, run it, and store results to disk.")

    arg_parser.add_argument('output_dir', metavar='OUTPUT_PATH', type=str, \
        help="Path to the output directory.")

    arg_parser.add_argument('nsims', metavar='SIM_NUM', type=int, \
        help="Number of simulations to do.")

    # arg_parser.add_argument('-l', '--load', metavar='LOAD_JSON', \
    #     help='Load JSON file with fit parameters.', type=str)

    # arg_parser.add_argument('-m', '--met', metavar='MET_FILE', \
    #     help='Load additional observations from a METAL or mirfit .met file.', type=str)

    # arg_parser.add_argument('-w', '--wid', metavar='WID_FILES', \
    #     help='Load mirfit wid files which will be used for wake plotting. Wildchars can be used, e.g. /path/wid*.txt.', 
    #     type=str, nargs='+')

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    # Generate simulations using multiprocessing
    input_list = [[cml_args.output_dir, np.random.randint(0, 2**32 - 1)] for _ in range(cml_args.nsims)]
    domainParallelizer(input_list, generateErosionSim)