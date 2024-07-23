""" Batch generate meteor simulations using an ablation model and random physical parameters, and store the 
simulations to disk. """


from __future__ import print_function, division, absolute_import, unicode_literals

import os
import json
import copy

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import pandas as pd
import wmpl

from wmpl.MetSim.GUI import SimulationResults
from wmpl.MetSim.MetSimErosion import Constants
from wmpl.MetSim.MetSimErosion import runSimulation as runSimulationErosion
from wmpl.MetSim.ML.GenerateSimulations import generateErosionSim,saveProcessedList,MetParam
from wmpl.MetSim.GUI import loadConstants
from wmpl.Utils.AtmosphereDensity import fitAtmPoly
from wmpl.Utils.Math import padOrTruncate
from wmpl.Utils.OSTools import mkdirP
from wmpl.Utils.TrajConversions import J2000_JD, date2JD
from wmpl.Utils.Pickling import savePickle, loadPickle
from wmpl.Utils.PyDomainParallelizer import domainParallelizer



### CONSTANTS ###

# Length of data that will be used as an input during training
DATA_LENGTH = 256

# Default number of minimum frames for simulation
MIN_FRAMES_VISIBLE = 4

### ###


##############################################################################################################
### SIMULATION PARAMETER CLASSES ###
### MAKE SURE TO ADD ANY NEW CLASSES TO THE "SIM_CLASSES" VARIABLE!


class ErosionSimParametersEMCCD(object):
    def __init__(self):
        """ Range of physical parameters for the erosion model, EMCCD system. """


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
        self.lim_mag_faintest = +9.0
        self.lim_mag_brightest = +0.0

        # Limiting magnitude for length measurements end (given by a range)
        #   This should be the same as the two value above for all other systems except for CAMO
        self.lim_mag_len_end_faintest = self.lim_mag_faintest
        self.lim_mag_len_end_brightest = self.lim_mag_brightest


        # Power of a zero-magnitude meteor (Watts)
        self.P_0M = 1210

        # System FPS
        self.fps = 30

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
        # self.visibility_time_min = 4/self.fps # DUMMY VARIABLE # 0.155

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


class ErosionSimParametersEMCCD_PER(object):
    def __init__(self):
        """ Range of physical parameters for the erosion model, EMCCD system for Perseids. """


        # Define the reference time for the atmosphere density model as J2000
        self.jdt_ref = date2JD(2020, 8, 10, 10, 0, 0)


        ## Atmosphere density ##
        #   Use the atmosphere density for the time at J2000 and coordinates of Elginfield
        self.dens_co = fitAtmPoly(np.radians(43.19301), np.radians(-81.315555), 60000, 180000, self.jdt_ref)

        ##


        # List of simulation parameters
        self.param_list = []



        ## System parameters ##

        # System limiting magnitude (given as a range)
        # self.lim_mag_faintest = +7.2    # change
        # self.lim_mag_brightest = +5.2   # change
        # self.lim_mag_faintest = +8    # change the startng height
        # self.lim_mag_brightest = +5   # change the startng height
        self.lim_mag_faintest = +6    # change the startng height
        self.lim_mag_brightest = +4   # change the startng height

        # Limiting magnitude for length measurements end (given by a range)
        #   This should be the same as the two value above for all other systems except for CAMO
        self.lim_mag_len_end_faintest = self.lim_mag_faintest
        self.lim_mag_len_end_brightest = self.lim_mag_brightest


        # Power of a zero-magnitude meteor (Watts)
        self.P_0M = 935

        # System FPS
        self.fps = 32

        # Time lag of length measurements (range in seconds) - accomodate CAMO tracking delay of 8 frames
        #   This should be 0 for all other systems except for the CAMO mirror tracking system
        self.len_delay_min = 0
        self.len_delay_max = 0

        # Simulation height range (m) that will be used to map the output to a grid
        self.sim_height = MetParam(70000, 130000)


        ##


        ## Physical parameters

        # Mass range (kg)
        self.m_init = MetParam(1e-7, 1e-4)
        self.param_list.append("m_init") # change

        # Initial velocity range (m/s)
        self.v_init = MetParam(57500, 65000)
        self.param_list.append("v_init") # change

        # Zenith angle range
        self.zenith_angle = MetParam(np.radians(15.0), np.radians(73.0))
        self.param_list.append("zenith_angle") # change

        # Density range (kg/m^3)
        self.rho = MetParam(100, 1000)
        self.param_list.append("rho")

        # Intrinsic ablation coeff range (s^2/m^2)
        self.sigma = MetParam(0.005/1e6, 0.05/1e6) 
        self.param_list.append("sigma")
        # self.sigma = MetParam(0.005/1e6, 0.5/1e6)


        ##


        ## Erosion parameters ##
        ## Assumes no change in erosion once it starts!

        # Erosion height range
        self.erosion_height_start = MetParam(100000, 120000)
        self.param_list.append("erosion_height_start")

        # Erosion coefficient (s^2/m^2)
        self.erosion_coeff = MetParam(0.0, 1/1e6)
        self.param_list.append("erosion_coeff")

        # Mass index
        self.erosion_mass_index = MetParam(1.5, 2.5)
        self.param_list.append("erosion_mass_index")

        # Minimum mass for erosion
        self.erosion_mass_min = MetParam(5e-12, 1e-10)
        self.param_list.append("erosion_mass_min")

        # Maximum mass for erosion
        self.erosion_mass_max = MetParam(1e-10, 5e-8)
        self.param_list.append("erosion_mass_max")

        ## 


        ### Simulation quality checks ###

        # Minimum time above the limiting magnitude (10 frames)
        #   This is a minimum for both magnitude and length! 
        # self.visibility_time_min = 4/self.fps # DUMMY VARIABLE # THIS IS A USELLES VARIABLE

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
        self.mag_faintest = +9
        self.mag_brightest = -2


        # Compute length range
        self.len_min = 0
        self.len_max = self.v_init.max*self.data_length/self.fps


        ### ###


class ErosionSimParametersEMCCD_PER_real_CAMO_v60(object):
    def __init__(self):
        """ Range of physical parameters for the erosion model, EMCCD system for Perseids. """


        # Define the reference time for the atmosphere density model as J2000
        self.jdt_ref = date2JD(2020, 8, 10, 10, 0, 0)


        ## Atmosphere density ##
        #   Use the atmosphere density for the time at J2000 and coordinates of Elginfield
        self.dens_co = fitAtmPoly(np.radians(43.19301), np.radians(-81.315555), 60000, 180000, self.jdt_ref)

        ##


        # List of simulation parameters
        self.param_list = []



        ## System parameters ##

        # System limiting magnitude (given as a range)
        # self.lim_mag_faintest = +7.2    # change
        # self.lim_mag_brightest = +5.2   # change
        # self.lim_mag_faintest = +8    # change the startng height
        # self.lim_mag_brightest = +5   # change the startng height
        # self.lim_mag_faintest = +5.30 # change the startng height
        # self.lim_mag_brightest = +5.20   # change the startng height

        # # Limiting magnitude for length measurements end (given by a range)
        # #   This should be the same as the two value above for all other systems except for CAMO
        # # self.lim_mag_len_end_faintest = self.lim_mag_faintest
        # # self.lim_mag_len_end_brightest = self.lim_mag_brightest
        # self.lim_mag_len_end_faintest = +5.55
        # self.lim_mag_len_end_brightest = +5.45

        # self.lim_mag_faintest = +6    # change the startng height
        # self.lim_mag_brightest = +4   # change the startng height
        # self.lim_mag_len_end_faintest = self.lim_mag_faintest
        # self.lim_mag_len_end_brightest = self.lim_mag_brightest

        self.lim_mag_faintest = +5.49    # change the startng height
        self.lim_mag_brightest = +5.48   # change the startng height
        self.lim_mag_len_end_faintest = +5.61
        self.lim_mag_len_end_brightest = +5.60

        # Power of a zero-magnitude meteor (Watts)
        self.P_0M = 935

        # System FPS
        self.fps = 32

        # Time lag of length measurements (range in seconds) - accomodate CAMO tracking delay of 8 frames
        #   This should be 0 for all other systems except for the CAMO mirror tracking system
        self.len_delay_min = 0
        self.len_delay_max = 0

        # Simulation height range (m) that will be used to map the output to a grid
        self.sim_height = MetParam(70000, 130000)


        ##


        ## Physical parameters

        # Mass range (kg)
        self.m_init = MetParam(1e-6, 2e-6)
        self.param_list.append("m_init") # change

        # Initial velocity range (m/s)
        self.v_init = MetParam(60000, 60200) # 60091.41691
        self.param_list.append("v_init") # change

        # Zenith angle range
        self.zenith_angle = MetParam(np.radians(43.35), np.radians(43.55)) # 43.466538
        self.param_list.append("zenith_angle") # change

        # Density range (kg/m^3)
        self.rho = MetParam(100, 1000)
        self.param_list.append("rho")

        # Intrinsic ablation coeff range (s^2/m^2)
        self.sigma = MetParam(0.008/1e6, 0.03/1e6) 
        self.param_list.append("sigma")
        # self.sigma = MetParam(0.005/1e6, 0.5/1e6)


        ##


        ## Erosion parameters ##
        ## Assumes no change in erosion once it starts!

        # Erosion height range
        self.erosion_height_start = MetParam(115000, 119000)
        self.param_list.append("erosion_height_start")

        # Erosion coefficient (s^2/m^2)
        self.erosion_coeff = MetParam(0.0, 1/1e6)
        self.param_list.append("erosion_coeff")

        # Mass index
        self.erosion_mass_index = MetParam(1.5, 2.5)
        self.param_list.append("erosion_mass_index")

        # Minimum mass for erosion
        self.erosion_mass_min = MetParam(5e-12, 1e-10)
        self.param_list.append("erosion_mass_min")

        # Maximum mass for erosion
        self.erosion_mass_max = MetParam(1e-10, 5e-8)
        self.param_list.append("erosion_mass_max")

        ## 


        ### Simulation quality checks ###

        # Minimum time above the limiting magnitude (10 frames)
        #   This is a minimum for both magnitude and length! 
        # self.visibility_time_min = 4/self.fps # DUMMY VARIABLE # THIS IS A USELLES VARIABLE

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
        self.mag_faintest = +9
        self.mag_brightest = -2


        # Compute length range
        self.len_min = 0
        self.len_max = self.v_init.max*self.data_length/self.fps


        ### ###


class ErosionSimParametersEMCCD_PER_v57_slow(object):
    def __init__(self):
        """ Range of physical parameters for the erosion model, EMCCD system for Perseids. """


        # Define the reference time for the atmosphere density model as J2000
        self.jdt_ref = date2JD(2020, 8, 10, 10, 0, 0)


        ## Atmosphere density ##
        #   Use the atmosphere density for the time at J2000 and coordinates of Elginfield
        self.dens_co = fitAtmPoly(np.radians(43.19301), np.radians(-81.315555), 60000, 180000, self.jdt_ref)

        ##


        # List of simulation parameters
        self.param_list = []



        ## System parameters ##

        # System limiting magnitude (given as a range)
        # self.lim_mag_faintest = +7.2    # change
        # self.lim_mag_brightest = +5.2   # change
        # self.lim_mag_faintest = +8    # change the startng height
        # self.lim_mag_brightest = +5   # change the startng height
        # self.lim_mag_faintest = +5.30 # change the startng height
        # self.lim_mag_brightest = +5.20   # change the startng height

        # # Limiting magnitude for length measurements end (given by a range)
        # #   This should be the same as the two value above for all other systems except for CAMO
        # # self.lim_mag_len_end_faintest = self.lim_mag_faintest
        # # self.lim_mag_len_end_brightest = self.lim_mag_brightest
        # self.lim_mag_len_end_faintest = +5.55
        # self.lim_mag_len_end_brightest = +5.45
        self.lim_mag_faintest = +6    # change the startng height
        self.lim_mag_brightest = +4   # change the startng height
        self.lim_mag_len_end_faintest = self.lim_mag_faintest
        self.lim_mag_len_end_brightest = self.lim_mag_brightest

        # Power of a zero-magnitude meteor (Watts)
        self.P_0M = 935

        # System FPS
        self.fps = 32

        # Time lag of length measurements (range in seconds) - accomodate CAMO tracking delay of 8 frames
        #   This should be 0 for all other systems except for the CAMO mirror tracking system
        self.len_delay_min = 0
        self.len_delay_max = 0

        # Simulation height range (m) that will be used to map the output to a grid
        self.sim_height = MetParam(70000, 130000)


        ##


        ## Physical parameters

        # Mass range (kg)
        self.m_init = MetParam(5e-7, 7e-7)
        self.param_list.append("m_init") # change

        # Initial velocity range (m/s)
        self.v_init = MetParam(57400, 57600) # 57500.18
        self.param_list.append("v_init") # change

        # Zenith angle range
        self.zenith_angle = MetParam(np.radians(45.45), np.radians(45.65))
        self.param_list.append("zenith_angle") # change

        # Density range (kg/m^3)
        self.rho = MetParam(100, 1000)
        self.param_list.append("rho")

        # Intrinsic ablation coeff range (s^2/m^2)
        self.sigma = MetParam(0.008/1e6, 0.03/1e6) 
        self.param_list.append("sigma")
        # self.sigma = MetParam(0.005/1e6, 0.5/1e6)


        ##


        ## Erosion parameters ##
        ## Assumes no change in erosion once it starts!

        # Erosion height range
        self.erosion_height_start = MetParam(114000, 119000)
        self.param_list.append("erosion_height_start")

        # Erosion coefficient (s^2/m^2)
        self.erosion_coeff = MetParam(0.0, 1/1e6)
        self.param_list.append("erosion_coeff")

        # Mass index
        self.erosion_mass_index = MetParam(1.5, 2.5)
        self.param_list.append("erosion_mass_index")

        # Minimum mass for erosion
        self.erosion_mass_min = MetParam(5e-12, 1e-10)
        self.param_list.append("erosion_mass_min")

        # Maximum mass for erosion
        self.erosion_mass_max = MetParam(1e-10, 5e-8)
        self.param_list.append("erosion_mass_max")

        ## 


        ### Simulation quality checks ###

        # Minimum time above the limiting magnitude (10 frames)
        #   This is a minimum for both magnitude and length! 
        # self.visibility_time_min = 4/self.fps # DUMMY VARIABLE # THIS IS A USELLES VARIABLE

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
        self.mag_faintest = +9
        self.mag_brightest = -2


        # Compute length range
        self.len_min = 0
        self.len_max = self.v_init.max*self.data_length/self.fps


        ### ###

class ErosionSimParametersEMCCD_PER_v59_heavy(object):
    def __init__(self):
        """ Range of physical parameters for the erosion model, EMCCD system for Perseids. """


        # Define the reference time for the atmosphere density model as J2000
        self.jdt_ref = date2JD(2020, 8, 10, 10, 0, 0)


        ## Atmosphere density ##
        #   Use the atmosphere density for the time at J2000 and coordinates of Elginfield
        self.dens_co = fitAtmPoly(np.radians(43.19301), np.radians(-81.315555), 60000, 180000, self.jdt_ref)

        ##


        # List of simulation parameters
        self.param_list = []



        ## System parameters ##

        # System limiting magnitude (given as a range)
        # self.lim_mag_faintest = +7.2    # change
        # self.lim_mag_brightest = +5.2   # change
        # self.lim_mag_faintest = +8    # change the startng height
        # self.lim_mag_brightest = +5   # change the startng height
        self.lim_mag_faintest = +4.20 # change the startng height
        self.lim_mag_brightest = +4.10   # change the startng height

        # Limiting magnitude for length measurements end (given by a range)
        #   This should be the same as the two value above for all other systems except for CAMO
        # self.lim_mag_len_end_faintest = self.lim_mag_faintest
        # self.lim_mag_len_end_brightest = self.lim_mag_brightest
        self.lim_mag_len_end_faintest = +2.95
        self.lim_mag_len_end_brightest = +2.85


        # Power of a zero-magnitude meteor (Watts)
        self.P_0M = 935

        # System FPS
        self.fps = 32

        # Time lag of length measurements (range in seconds) - accomodate CAMO tracking delay of 8 frames
        #   This should be 0 for all other systems except for the CAMO mirror tracking system
        self.len_delay_min = 0
        self.len_delay_max = 0

        # Simulation height range (m) that will be used to map the output to a grid
        self.sim_height = MetParam(70000, 130000)


        ##


        ## Physical parameters

        # Mass range (kg)
        self.m_init = MetParam(1e-5, 2e-5)
        self.param_list.append("m_init") # change

        # Initial velocity range (m/s)
        self.v_init = MetParam(59740, 59940) # 59836.84
        self.param_list.append("v_init") # change

        # Zenith angle range
        self.zenith_angle = MetParam(np.radians(39.7), np.radians(39.9))
        self.param_list.append("zenith_angle") # change

        # Density range (kg/m^3)
        self.rho = MetParam(100, 300)
        self.param_list.append("rho")

        # Intrinsic ablation coeff range (s^2/m^2)
        self.sigma = MetParam(0.008/1e6, 0.03/1e6) 
        self.param_list.append("sigma")
        # self.sigma = MetParam(0.005/1e6, 0.5/1e6)


        ##


        ## Erosion parameters ##
        ## Assumes no change in erosion once it starts!

        # Erosion height range
        self.erosion_height_start = MetParam(115000, 119000)
        self.param_list.append("erosion_height_start")

        # Erosion coefficient (s^2/m^2)
        self.erosion_coeff = MetParam(0.0, 1/1e6)
        self.param_list.append("erosion_coeff")

        # Mass index
        self.erosion_mass_index = MetParam(1.5, 2.5)
        self.param_list.append("erosion_mass_index")

        # Minimum mass for erosion
        self.erosion_mass_min = MetParam(5e-12, 1e-10)
        self.param_list.append("erosion_mass_min")

        # Maximum mass for erosion
        self.erosion_mass_max = MetParam(1e-10, 5e-8)
        self.param_list.append("erosion_mass_max")

        ## 


        ### Simulation quality checks ###

        # Minimum time above the limiting magnitude (10 frames)
        #   This is a minimum for both magnitude and length! 
        # self.visibility_time_min = 4/self.fps # DUMMY VARIABLE # THIS IS A USELLES VARIABLE

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
        self.mag_faintest = +9
        self.mag_brightest = -2


        # Compute length range
        self.len_min = 0
        self.len_max = self.v_init.max*self.data_length/self.fps


        ### ###

class ErosionSimParametersEMCCD_PER_v60_light(object):
    def __init__(self):
        """ Range of physical parameters for the erosion model, EMCCD system for Perseids. """


        # Define the reference time for the atmosphere density model as J2000
        self.jdt_ref = date2JD(2020, 8, 10, 10, 0, 0)


        ## Atmosphere density ##
        #   Use the atmosphere density for the time at J2000 and coordinates of Elginfield
        self.dens_co = fitAtmPoly(np.radians(43.19301), np.radians(-81.315555), 60000, 180000, self.jdt_ref)

        ##


        # List of simulation parameters
        self.param_list = []



        ## System parameters ##

        # System limiting magnitude (given as a range)
        # self.lim_mag_faintest = +7.2    # change
        # self.lim_mag_brightest = +5.2   # change
        # self.lim_mag_faintest = +8    # change the startng height
        # self.lim_mag_brightest = +5   # change the startng height
        # self.lim_mag_faintest = +5.70 # change the startng height
        # self.lim_mag_brightest = +5.60   # change the startng height

        # Limiting magnitude for length measurements end (given by a range)
        #   This should be the same as the two value above for all other systems except for CAMO
        # self.lim_mag_len_end_faintest = self.lim_mag_faintest
        # self.lim_mag_len_end_brightest = self.lim_mag_brightest
        # self.lim_mag_len_end_faintest = +5.75
        # self.lim_mag_len_end_brightest = +5.65

        self.lim_mag_faintest = +6    # change the startng height
        self.lim_mag_brightest = +5.5   # change the startng height
        # self.lim_mag_faintest = +5    # change the startng height
        # self.lim_mag_brightest = +4.5   # change the startng height
        self.lim_mag_len_end_faintest = self.lim_mag_faintest
        self.lim_mag_len_end_brightest = self.lim_mag_brightest


        # Power of a zero-magnitude meteor (Watts)
        self.P_0M = 935

        # System FPS
        self.fps = 32

        # Time lag of length measurements (range in seconds) - accomodate CAMO tracking delay of 8 frames
        #   This should be 0 for all other systems except for the CAMO mirror tracking system
        self.len_delay_min = 0
        self.len_delay_max = 0

        # Simulation height range (m) that will be used to map the output to a grid
        self.sim_height = MetParam(70000, 130000)


        ##


        ## Physical parameters

        # Mass range (kg)
        self.m_init = MetParam(1.0e-7, 1.1e-7)
        self.param_list.append("m_init") # change

        # Initial velocity range (m/s)
        # self.v_init = MetParam(59700, 59900) # 60051.34
        # self.v_init = MetParam(59950, 60150) # 60051.34
        self.v_init = MetParam(60000, 60150) # 60051.34
        self.param_list.append("v_init") # change

        # Zenith angle range
        self.zenith_angle = MetParam(np.radians(39.2), np.radians(39.4))
        self.param_list.append("zenith_angle") # change

        # Density range (kg/m^3)
        self.rho = MetParam(100, 1000)
        self.param_list.append("rho")

        # Intrinsic ablation coeff range (s^2/m^2)
        self.sigma = MetParam(0.008/1e6, 0.03/1e6) 
        self.param_list.append("sigma")
        # self.sigma = MetParam(0.005/1e6, 0.5/1e6)


        ##


        ## Erosion parameters ##
        ## Assumes no change in erosion once it starts!

        # Erosion height range
        self.erosion_height_start = MetParam(105000, 109000)
        self.param_list.append("erosion_height_start")

        # Erosion coefficient (s^2/m^2)
        self.erosion_coeff = MetParam(0.0, 1/1e6)
        self.param_list.append("erosion_coeff")

        # Mass index
        self.erosion_mass_index = MetParam(1.5, 2.5)
        self.param_list.append("erosion_mass_index")

        # Minimum mass for erosion
        self.erosion_mass_min = MetParam(5e-12, 1e-10)
        self.param_list.append("erosion_mass_min")

        # Maximum mass for erosion
        self.erosion_mass_max = MetParam(1e-10, 5e-8)
        self.param_list.append("erosion_mass_max")

        ## 


        ### Simulation quality checks ###

        # Minimum time above the limiting magnitude (10 frames)
        #   This is a minimum for both magnitude and length! 
        # self.visibility_time_min = 4/self.fps # DUMMY VARIABLE # THIS IS A USELLES VARIABLE

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
        self.mag_faintest = +9
        self.mag_brightest = -2


        # Compute length range
        self.len_min = 0
        self.len_max = self.v_init.max*self.data_length/self.fps


        ### ###

class ErosionSimParametersEMCCD_PER_v61_shallow(object):
    def __init__(self):
        """ Range of physical parameters for the erosion model, EMCCD system for Perseids. """


        # Define the reference time for the atmosphere density model as J2000
        self.jdt_ref = date2JD(2020, 8, 10, 10, 0, 0)


        ## Atmosphere density ##
        #   Use the atmosphere density for the time at J2000 and coordinates of Elginfield
        self.dens_co = fitAtmPoly(np.radians(43.19301), np.radians(-81.315555), 60000, 180000, self.jdt_ref)

        ##


        # List of simulation parameters
        self.param_list = []



        ## System parameters ##

        # System limiting magnitude (given as a range)
        # self.lim_mag_faintest = +7.2    # change
        # self.lim_mag_brightest = +5.2   # change
        self.lim_mag_faintest = +6    # change the startng height
        self.lim_mag_brightest = +4   # change the startng height
        # self.lim_mag_faintest = +5.50 # change the startng height
        # self.lim_mag_brightest = +4.95   # change the startng height
        # self.lim_mag_faintest = +6 # change the startng height
        # self.lim_mag_brightest = +5   # change the startng height
        

        # Limiting magnitude for length measurements end (given by a range)
        #   This should be the same as the two value above for all other systems except for CAMO
        self.lim_mag_len_end_faintest = self.lim_mag_faintest
        self.lim_mag_len_end_brightest = self.lim_mag_brightest
        # self.lim_mag_len_end_faintest = +6
        # self.lim_mag_len_end_brightest = +5
        # self.lim_mag_len_end_faintest = +5.5
        # self.lim_mag_len_end_brightest = +4.95


        # Power of a zero-magnitude meteor (Watts)
        self.P_0M = 935

        # System FPS
        self.fps = 32

        # Time lag of length measurements (range in seconds) - accomodate CAMO tracking delay of 8 frames
        #   This should be 0 for all other systems except for the CAMO mirror tracking system
        self.len_delay_min = 0
        self.len_delay_max = 0

        # Simulation height range (m) that will be used to map the output to a grid
        self.sim_height = MetParam(70000, 130000)


        ##


        ## Physical parameters

        # Mass range (kg)
        self.m_init = MetParam(6.5e-7, 7.5e-7)
        self.param_list.append("m_init") # change

        # Initial velocity range (m/s)
        # self.v_init = MetParam(61100, 61300) # 61460.80
        self.v_init = MetParam(61350, 61480) # 61460.80
        # self.v_init = MetParam(61350, 61550) # 61460.80
        self.param_list.append("v_init") # change

        # Zenith angle range
        self.zenith_angle = MetParam(np.radians(63.17), np.radians(63.27))
        self.param_list.append("zenith_angle") # change

        # Density range (kg/m^3)
        self.rho = MetParam(100, 1000)
        self.param_list.append("rho")

        # Intrinsic ablation coeff range (s^2/m^2)
        self.sigma = MetParam(0.008/1e6, 0.03/1e6) 
        self.param_list.append("sigma")
        # self.sigma = MetParam(0.005/1e6, 0.5/1e6)


        ##


        ## Erosion parameters ##
        ## Assumes no change in erosion once it starts!

        # Erosion height range
        self.erosion_height_start = MetParam(113000, 117000)
        self.param_list.append("erosion_height_start")

        # Erosion coefficient (s^2/m^2)
        self.erosion_coeff = MetParam(0.0, 1/1e6)
        self.param_list.append("erosion_coeff")

        # Mass index
        self.erosion_mass_index = MetParam(1.5, 2.5)
        self.param_list.append("erosion_mass_index")

        # Minimum mass for erosion
        self.erosion_mass_min = MetParam(5e-12, 1e-10)
        self.param_list.append("erosion_mass_min")

        # Maximum mass for erosion
        self.erosion_mass_max = MetParam(1e-10, 5e-8)
        self.param_list.append("erosion_mass_max")

        ## 


        ### Simulation quality checks ###

        # Minimum time above the limiting magnitude (10 frames)
        #   This is a minimum for both magnitude and length! 
        # self.visibility_time_min = 4/self.fps # DUMMY VARIABLE # THIS IS A USELLES VARIABLE

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
        self.mag_faintest = +9
        self.mag_brightest = -2


        # Compute length range
        self.len_min = 0
        self.len_max = self.v_init.max*self.data_length/self.fps


        ### ###

class ErosionSimParametersEMCCD_PER_v62_steep(object):
    def __init__(self):
        """ Range of physical parameters for the erosion model, EMCCD system for Perseids. """


        # Define the reference time for the atmosphere density model as J2000
        self.jdt_ref = date2JD(2020, 8, 10, 10, 0, 0)


        ## Atmosphere density ##
        #   Use the atmosphere density for the time at J2000 and coordinates of Elginfield
        self.dens_co = fitAtmPoly(np.radians(43.19301), np.radians(-81.315555), 60000, 180000, self.jdt_ref)

        ##


        # List of simulation parameters
        self.param_list = []



        ## System parameters ##

        # System limiting magnitude (given as a range)
        # self.lim_mag_faintest = +7.2    # change
        # self.lim_mag_brightest = +5.2   # change
        # self.lim_mag_faintest = +8    # change the startng height
        # self.lim_mag_brightest = +5   # change the startng height
        self.lim_mag_faintest = +5.40 # change the startng height
        self.lim_mag_brightest = +5.30   # change the startng height

        # Limiting magnitude for length measurements end (given by a range)
        #   This should be the same as the two value above for all other systems except for CAMO
        # self.lim_mag_len_end_faintest = self.lim_mag_faintest
        # self.lim_mag_len_end_brightest = self.lim_mag_brightest
        self.lim_mag_len_end_faintest = +4.75
        self.lim_mag_len_end_brightest = +4.65


        # Power of a zero-magnitude meteor (Watts)
        self.P_0M = 935

        # System FPS
        self.fps = 32

        # Time lag of length measurements (range in seconds) - accomodate CAMO tracking delay of 8 frames
        #   This should be 0 for all other systems except for the CAMO mirror tracking system
        self.len_delay_min = 0
        self.len_delay_max = 0

        # Simulation height range (m) that will be used to map the output to a grid
        self.sim_height = MetParam(70000, 130000)


        ##


        ## Physical parameters

        # Mass range (kg)
        self.m_init = MetParam(5e-7, 7e-7)
        self.param_list.append("m_init") # change

        # Initial velocity range (m/s)
        self.v_init = MetParam(62500, 62700) #62579.06
        self.param_list.append("v_init") # change

        # Zenith angle range
        self.zenith_angle = MetParam(np.radians(24.35), np.radians(24.45))
        self.param_list.append("zenith_angle") # change

        # Density range (kg/m^3)
        self.rho = MetParam(100, 1000)
        self.param_list.append("rho")

        # Intrinsic ablation coeff range (s^2/m^2)
        self.sigma = MetParam(0.008/1e6, 0.03/1e6) 
        self.param_list.append("sigma")
        # self.sigma = MetParam(0.005/1e6, 0.5/1e6)


        ##


        ## Erosion parameters ##
        ## Assumes no change in erosion once it starts!

        # Erosion height range
        self.erosion_height_start = MetParam(112000, 119000)
        self.param_list.append("erosion_height_start")

        # Erosion coefficient (s^2/m^2)
        self.erosion_coeff = MetParam(0.0, 1/1e6)
        self.param_list.append("erosion_coeff")

        # Mass index
        self.erosion_mass_index = MetParam(1.5, 2.5)
        self.param_list.append("erosion_mass_index")

        # Minimum mass for erosion
        self.erosion_mass_min = MetParam(5e-12, 1e-10)
        self.param_list.append("erosion_mass_min")

        # Maximum mass for erosion
        self.erosion_mass_max = MetParam(1e-10, 5e-8)
        self.param_list.append("erosion_mass_max")

        ## 


        ### Simulation quality checks ###

        # Minimum time above the limiting magnitude (10 frames)
        #   This is a minimum for both magnitude and length! 
        # self.visibility_time_min = 4/self.fps # DUMMY VARIABLE # THIS IS A USELLES VARIABLE

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
        self.mag_faintest = +9
        self.mag_brightest = -2


        # Compute length range
        self.len_min = 0
        self.len_max = self.v_init.max*self.data_length/self.fps


        ### ###

class ErosionSimParametersEMCCD_PER_v65_fast(object):
    def __init__(self):
        """ Range of physical parameters for the erosion model, EMCCD system for Perseids. """


        # Define the reference time for the atmosphere density model as J2000
        self.jdt_ref = date2JD(2020, 8, 10, 10, 0, 0)


        ## Atmosphere density ##
        #   Use the atmosphere density for the time at J2000 and coordinates of Elginfield
        self.dens_co = fitAtmPoly(np.radians(43.19301), np.radians(-81.315555), 60000, 180000, self.jdt_ref)

        ##


        # List of simulation parameters
        self.param_list = []



        ## System parameters ##

        # System limiting magnitude (given as a range)
        # self.lim_mag_faintest = +7.2    # change
        # self.lim_mag_brightest = +5.2   # change
        # self.lim_mag_faintest = +8    # change the startng height
        # self.lim_mag_brightest = +5   # change the startng height
        # self.lim_mag_faintest = +4.5 # change the startng height
        # self.lim_mag_brightest = +4   # change the startng height

        # # # Limiting magnitude for length measurements end (given by a range)
        # # #   This should be the same as the two value above for all other systems except for CAMO
        # # # self.lim_mag_len_end_faintest = self.lim_mag_faintest
        # # # self.lim_mag_len_end_brightest = self.lim_mag_brightest
        # self.lim_mag_len_end_faintest = +6.5
        # self.lim_mag_len_end_brightest = +5.5
        self.lim_mag_faintest = +6    # change the startng height
        self.lim_mag_brightest = +4   # change the startng height
        self.lim_mag_len_end_faintest = self.lim_mag_faintest
        self.lim_mag_len_end_brightest = self.lim_mag_brightest



        # Power of a zero-magnitude meteor (Watts)
        self.P_0M = 935

        # System FPS
        self.fps = 32

        # Time lag of length measurements (range in seconds) - accomodate CAMO tracking delay of 8 frames
        #   This should be 0 for all other systems except for the CAMO mirror tracking system
        self.len_delay_min = 0
        self.len_delay_max = 0

        # Simulation height range (m) that will be used to map the output to a grid
        self.sim_height = MetParam(70000, 130000)


        ##


        ## Physical parameters

        # Mass range (kg)
        self.m_init = MetParam(6e-7, 8e-7)
        self.param_list.append("m_init") # change

        # Initial velocity range (m/s)
        self.v_init = MetParam(64900, 65100) #64997.01929105718
        self.param_list.append("v_init") # change

        # Zenith angle range
        self.zenith_angle = MetParam(np.radians(51.58), np.radians(51.78))#51.68476673
        self.param_list.append("zenith_angle") # change

        # Density range (kg/m^3)
        self.rho = MetParam(100, 1000)
        self.param_list.append("rho")

        # Intrinsic ablation coeff range (s^2/m^2)
        self.sigma = MetParam(0.008/1e6, 0.03/1e6) 
        self.param_list.append("sigma")
        # self.sigma = MetParam(0.005/1e6, 0.5/1e6)


        ##


        ## Erosion parameters ##
        ## Assumes no change in erosion once it starts!

        # Erosion height range
        self.erosion_height_start = MetParam(111000, 116000)
        self.param_list.append("erosion_height_start")

        # Erosion coefficient (s^2/m^2)
        self.erosion_coeff = MetParam(0.0, 1/1e6)
        self.param_list.append("erosion_coeff")

        # Mass index
        self.erosion_mass_index = MetParam(1.5, 2.5)
        self.param_list.append("erosion_mass_index")

        # Minimum mass for erosion
        self.erosion_mass_min = MetParam(5e-12, 1e-10)
        self.param_list.append("erosion_mass_min")

        # Maximum mass for erosion
        self.erosion_mass_max = MetParam(1e-10, 5e-8)
        self.param_list.append("erosion_mass_max")

        ## 


        ### Simulation quality checks ###

        # Minimum time above the limiting magnitude (10 frames)
        #   This is a minimum for both magnitude and length! 
        # self.visibility_time_min = 4/self.fps # DUMMY VARIABLE # THIS IS A USELLES VARIABLE

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
        self.mag_faintest = +9
        self.mag_brightest = -2


        # Compute length range
        self.len_min = 0
        self.len_max = self.v_init.max*self.data_length/self.fps


        ### ###



class ErosionSimParametersEMCCD_GEM(object):
    def __init__(self):
        """ Range of physical parameters for the erosion model, EMCCD system for Geminids. """


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
        self.lim_mag_faintest = +8.0    # change
        self.lim_mag_brightest = +5.5   # change

        # Limiting magnitude for length measurements end (given by a range)
        #   This should be the same as the two value above for all other systems except for CAMO
        self.lim_mag_len_end_faintest = self.lim_mag_faintest
        self.lim_mag_len_end_brightest = self.lim_mag_brightest


        # Power of a zero-magnitude meteor (Watts)
        self.P_0M = 935

        # System FPS
        self.fps = 30

        # Time lag of length measurements (range in seconds) - accomodate CAMO tracking delay of 8 frames
        #   This should be 0 for all other systems except for the CAMO mirror tracking system
        self.len_delay_min = 0
        self.len_delay_max = 0

        # Simulation height range (m) that will be used to map the output to a grid
        self.sim_height = MetParam(70000, 130000)


        ##


        ## Physical parameters

        # Mass range (kg)
        self.m_init = MetParam(2e-7, 2.7e-6)   # change
        self.param_list.append("m_init")

        # Initial velocity range (m/s)
        self.v_init = MetParam(34000, 37000)
        self.param_list.append("v_init")    # change

        # Zenith angle range
        self.zenith_angle = MetParam(np.radians(11.0), np.radians(42.5))
        self.param_list.append("zenith_angle")  # change

        # Density range (kg/m^3)
        self.rho = MetParam(100, 2000)
        self.param_list.append("rho")

        # Intrinsic ablation coeff range (s^2/m^2)
        self.sigma = MetParam(0.01/1e6, 0.1/1e6) 
        self.param_list.append("sigma")

        ##


        ## Erosion parameters ##
        ## Assumes no change in erosion once it starts!

        # Erosion height range
        self.erosion_height_start = MetParam(96000, 110000)
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
        # self.visibility_time_min = 4/self.fps # DUMMY VARIABLE # 0.13 self.visibility_time_min = 

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


class ErosionSimParametersEMCCD_ORI(object):
    def __init__(self):
        """ Range of physical parameters for the erosion model, EMCCD system for Geminids. """


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
        self.lim_mag_faintest = +6.2    # change
        self.lim_mag_brightest = +4.7   # change

        # Limiting magnitude for length measurements end (given by a range)
        #   This should be the same as the two value above for all other systems except for CAMO
        self.lim_mag_len_end_faintest = self.lim_mag_faintest
        self.lim_mag_len_end_brightest = self.lim_mag_brightest


        # Power of a zero-magnitude meteor (Watts)
        self.P_0M = 935

        # System FPS
        self.fps = 30

        # Time lag of length measurements (range in seconds) - accomodate CAMO tracking delay of 8 frames
        #   This should be 0 for all other systems except for the CAMO mirror tracking system
        self.len_delay_min = 0
        self.len_delay_max = 0

        # Simulation height range (m) that will be used to map the output to a grid
        self.sim_height = MetParam(70000, 130000)


        ##


        ## Physical parameters

        # Mass range (kg)
        self.m_init = MetParam(9e-8, 1e-6)   # change
        self.param_list.append("m_init")

        # Initial velocity range (m/s)
        self.v_init = MetParam(63800, 68700)
        self.param_list.append("v_init")    # change

        # Zenith angle range
        self.zenith_angle = MetParam(np.radians(27.9), np.radians(53.5))
        self.param_list.append("zenith_angle")  # change

        # Density range (kg/m^3)
        self.rho = MetParam(100, 2000)
        self.param_list.append("rho")

        # Intrinsic ablation coeff range (s^2/m^2)
        self.sigma = MetParam(0.01/1e6, 0.1/1e6) 
        self.param_list.append("sigma")

        ##


        ## Erosion parameters ##
        ## Assumes no change in erosion once it starts!

        # Erosion height range
        self.erosion_height_start = MetParam(105000, 120000)
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
        # self.visibility_time_min = 4/self.fps # DUMMY VARIABLE # 0.13 self.visibility_time_min = 4/30

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


class ErosionSimParametersEMCCD_ETA(object):
    def __init__(self):
        """ Range of physical parameters for the erosion model, EMCCD system for Geminids. """


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
        self.lim_mag_faintest = +5.8    # change
        self.lim_mag_brightest = +3.7   # change

        # Limiting magnitude for length measurements end (given by a range)
        #   This should be the same as the two value above for all other systems except for CAMO
        self.lim_mag_len_end_faintest = self.lim_mag_faintest
        self.lim_mag_len_end_brightest = self.lim_mag_brightest


        # Power of a zero-magnitude meteor (Watts)
        self.P_0M = 935

        # System FPS
        self.fps = 30

        # Time lag of length measurements (range in seconds) - accomodate CAMO tracking delay of 8 frames
        #   This should be 0 for all other systems except for the CAMO mirror tracking system
        self.len_delay_min = 0
        self.len_delay_max = 0

        # Simulation height range (m) that will be used to map the output to a grid
        self.sim_height = MetParam(70000, 130000)


        ##


        ## Physical parameters

        # Mass range (kg)
        self.m_init = MetParam(9e-8, 1e-6)   # change
        self.param_list.append("m_init")

        # Initial velocity range (m/s)
        self.v_init = MetParam(65000, 68400)
        self.param_list.append("v_init")    # change

        # Zenith angle range
        self.zenith_angle = MetParam(np.radians(64.8), np.radians(80.2))
        self.param_list.append("zenith_angle")  # change

        # Density range (kg/m^3)
        self.rho = MetParam(100, 2000)
        self.param_list.append("rho")

        # Intrinsic ablation coeff range (s^2/m^2)
        self.sigma = MetParam(0.01/1e6, 0.1/1e6) 
        self.param_list.append("sigma")

        ##


        ## Erosion parameters ##
        ## Assumes no change in erosion once it starts!

        # Erosion height range
        self.erosion_height_start = MetParam(105000, 120000)
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
        self.erosion_mass_max = MetParam(1e-11, 1e-7)
        self.param_list.append("erosion_mass_max")

        ## 


        ### Simulation quality checks ###

        # Minimum time above the limiting magnitude (10 frames)
        #   This is a minimum for both magnitude and length!
        # self.visibility_time_min = 4/self.fps # DUMMY VARIABLE # 0.13 self.visibility_time_min = 

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


class ErosionSimParametersEMCCD_SDA(object):
    def __init__(self):
        """ Range of physical parameters for the erosion model, EMCCD system for Geminids. """


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
        self.lim_mag_faintest = +6.9    # change
        self.lim_mag_brightest = +4.8   # change

        # Limiting magnitude for length measurements end (given by a range)
        #   This should be the same as the two value above for all other systems except for CAMO
        self.lim_mag_len_end_faintest = self.lim_mag_faintest
        self.lim_mag_len_end_brightest = self.lim_mag_brightest


        # Power of a zero-magnitude meteor (Watts)
        self.P_0M = 935

        # System FPS
        self.fps = 30

        # Time lag of length measurements (range in seconds) - accomodate CAMO tracking delay of 8 frames
        #   This should be 0 for all other systems except for the CAMO mirror tracking system
        self.len_delay_min = 0
        self.len_delay_max = 0

        # Simulation height range (m) that will be used to map the output to a grid
        self.sim_height = MetParam(70000, 130000)


        ##


        ## Physical parameters

        # Mass range (kg)
        self.m_init = MetParam(2.2e-7, 2.5e-6)   # change
        self.param_list.append("m_init")

        # Initial velocity range (m/s)
        self.v_init = MetParam(37000, 43000)
        self.param_list.append("v_init")    # change

        # Zenith angle range
        self.zenith_angle = MetParam(np.radians(58.3), np.radians(72.1))
        self.param_list.append("zenith_angle")  # change

        # Density range (kg/m^3)
        self.rho = MetParam(100, 2000)
        self.param_list.append("rho")

        # Intrinsic ablation coeff range (s^2/m^2)
        self.sigma = MetParam(0.01/1e6, 0.1/1e6) 
        self.param_list.append("sigma")

        ##


        ## Erosion parameters ##
        ## Assumes no change in erosion once it starts!

        # Erosion height range
        self.erosion_height_start = MetParam(96000, 110000)
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
        # self.visibility_time_min = 4/self.fps # DUMMY VARIABLE # 0.13 self.visibility_time_min = 

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


class ErosionSimParametersEMCCD_CAP(object):
    def __init__(self):
        """ Range of physical parameters for the erosion model, EMCCD system for Geminids. """


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
        self.lim_mag_faintest = +7.6    # change
        self.lim_mag_brightest = +5.7   # change

        # Limiting magnitude for length measurements end (given by a range)
        #   This should be the same as the two value above for all other systems except for CAMO
        self.lim_mag_len_end_faintest = self.lim_mag_faintest
        self.lim_mag_len_end_brightest = self.lim_mag_brightest


        # Power of a zero-magnitude meteor (Watts)
        self.P_0M = 935

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
        # self.m_init = MetParam(3.8e-7, 5.3e-6)   # change
        self.m_init = MetParam(1e-8, 1e-6)   # change
        self.param_list.append("m_init")

        # Initial velocity range (m/s)
        # self.v_init = MetParam(22000, 27000)
        self.v_init = MetParam(20000, 30000)
        self.param_list.append("v_init")    # change

        # Zenith angle range
        self.zenith_angle = MetParam(np.radians(48), np.radians(65))
        self.param_list.append("zenith_angle")  # change

        # Density range (kg/m^3)
        self.rho = MetParam(100, 2000)
        self.param_list.append("rho")

        # Intrinsic ablation coeff range (s^2/m^2)
        self.sigma = MetParam(0.01/1e6, 0.1/1e6) 
        self.param_list.append("sigma")

        ##


        ## Erosion parameters ##
        ## Assumes no change in erosion once it starts!

        # Erosion height range
        self.erosion_height_start = MetParam(90000, 110000)
        self.param_list.append("erosion_height_start")

        # Erosion coefficient (s^2/m^2)
        self.erosion_coeff = MetParam(0.0, 1.0/1e4)
        self.param_list.append("erosion_coeff")

        # Mass index
        self.erosion_mass_index = MetParam(1, 3.5)
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
        # self.visibility_time_min = 10/self.fps # 0.13 self.visibility_time_min = 

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
SIM_CLASSES = [ErosionSimParametersEMCCD, 
               ErosionSimParametersEMCCD_PER, 
               ErosionSimParametersEMCCD_PER_v57_slow, ErosionSimParametersEMCCD_PER_v59_heavy, ErosionSimParametersEMCCD_PER_v60_light, ErosionSimParametersEMCCD_PER_v61_shallow, ErosionSimParametersEMCCD_PER_v62_steep, ErosionSimParametersEMCCD_PER_v65_fast,
               ErosionSimParametersEMCCD_PER_real_CAMO_v60,
               ErosionSimParametersEMCCD_GEM, ErosionSimParametersEMCCD_ORI,
               ErosionSimParametersEMCCD_ETA, ErosionSimParametersEMCCD_SDA, ErosionSimParametersEMCCD_CAP]
SIM_CLASSES_NAMES = [c.__name__ for c in SIM_CLASSES]

##############################################################################################################

def find_closest_index(time_arr, time_sampled):
    closest_indices = []
    for sample in time_sampled:
        closest_index = min(range(len(time_arr)), key=lambda i: abs(time_arr[i] - sample))
        closest_indices.append(closest_index)
    return closest_indices

# extrct from the pickle file the data of the real data
def real_pickle_data_and_plot(data_path, plot_color=''):

    # split the data_path to file and root
    root, name_file = os.path.split(data_path)
    print('root', root)
    
    # Shower=Shower
    # keep the first 14 characters of the ID
    name=name_file[:15]

    print('Loading pickle file: ', data_path)
    # check if there are any pickle files in the 
    if os.path.isfile(os.path.join(root, name_file)):

        traj = wmpl.Utils.Pickling.loadPickle(root, name_file)

        jd_dat=traj.jdt_ref

        vel_pickl=[]
        vel_total=[]
        time_pickl=[]
        time_total=[]
        abs_mag_pickl=[]
        abs_total=[]
        height_pickl=[]
        height_total=[]
        lag=[]
        lag_total=[]
        elev_angle_pickl=[]
        elg_pickl=[]
        tav_pickl=[]
        
        lat_dat=[]
        lon_dat=[]


        # print('len(traj.observations)', len(traj.observations))

        jj=0
        for obs in traj.observations:
            # find all the differrnt names of the variables in the pickle files
            # print(obs.__dict__.keys())
            jj+=1
            if jj==1:
                tav_pickl=obs.velocities[1:int(len(obs.velocities)/4)]
                # if tav_pickl is empty append the first value of obs.velocities
                if len(tav_pickl)==0:
                    tav_pickl=obs.velocities[1:2]
                
                vel_01=obs.velocities
                time_01=obs.time_data
                abs_mag_01=obs.absolute_magnitudes
                height_01=obs.model_ht
                lag_01=obs.lag
                elev_angle_01=obs.elev_data

            elif jj==2:
                elg_pickl=obs.velocities[1:int(len(obs.velocities)/4)]
                if len(elg_pickl)==0:
                    elg_pickl=obs.velocities[1:2]
                
                vel_02=obs.velocities
                time_02=obs.time_data
                abs_mag_02=obs.absolute_magnitudes
                height_02=obs.model_ht
                lag_02=obs.lag
                elev_angle_02=obs.elev_data

            # put it at the end obs.velocities[1:] at the end of vel_pickl list
            vel_pickl.extend(obs.velocities[1:])
            time_pickl.extend(obs.time_data[1:])
            time_total.extend(obs.time_data)
            abs_mag_pickl.extend(obs.absolute_magnitudes[1:])
            abs_total.extend(obs.absolute_magnitudes)
            height_pickl.extend(obs.model_ht[1:])
            height_total.extend(obs.model_ht)
            lag.extend(obs.lag[1:])
            lag_total.extend(obs.lag)
            elev_angle_pickl.extend(obs.elev_data)
            # length_pickl=len(obs.state_vect_dist[1:])
            
            lat_dat=obs.lat
            lon_dat=obs.lon

        # compute the linear regression
        vel_pickl = [i/1000 for i in vel_pickl] # convert m/s to km/s
        time_pickl = [i for i in time_pickl]
        time_total = [i for i in time_total]
        height_pickl = [i/1000 for i in height_pickl]
        height_total = [i/1000 for i in height_total]
        abs_mag_pickl = [i for i in abs_mag_pickl]
        abs_total = [i for i in abs_total]
        lag=[i/1000 for i in lag]
        lag_total=[i/1000 for i in lag_total]

        # print('length_pickl', length_pickl)
        # length_pickl = [i/1000 for i in length_pickl]


        # find the height when the velocity start dropping from the initial value 
        vel_init_mean = (np.mean(elg_pickl)+np.mean(tav_pickl))/2/1000
        v0=vel_init_mean
        

        # divide height_01/1000 to convert it to km
        height_01 = [i/1000 for i in height_01]
        height_02 = [i/1000 for i in height_02]
        # divide vel_01/1000 to convert it to km
        vel_01 = [i/1000 for i in vel_01]
        vel_02 = [i/1000 for i in vel_02]
        # vel_01[0], v0 = v0, vel_01[0]
        vel_01[0] = v0
        vel_02[0] = v0

        vel_total.extend(vel_01)
        vel_total.extend(vel_02)
        

        #####order the list by time
        vel_pickl = [x for _,x in sorted(zip(time_pickl,vel_pickl))]
        vel_total = [x for _,x in sorted(zip(time_total,vel_total))]
        abs_mag_pickl = [x for _,x in sorted(zip(time_pickl,abs_mag_pickl))]
        abs_total = [x for _,x in sorted(zip(time_total,abs_total))]
        height_pickl = [x for _,x in sorted(zip(time_pickl,height_pickl))]
        height_total = [x for _,x in sorted(zip(time_total,height_total))]
        lag = [x for _,x in sorted(zip(time_pickl,lag))]
        lag_total = [x for _,x in sorted(zip(time_total,lag_total))]
        # length_pickl = [x for _,x in sorted(zip(time_pickl,length_pickl))]
        time_pickl = sorted(time_pickl)
        time_total = sorted(time_total)

        # v0=(vel_sim_line[0])
        # v0=(vel_sim_line[0])
        vel_init_norot=(vel_init_mean)
        # print('mean_vel_init', vel_sim_line[0])
        vel_avg_norot=(np.mean(vel_pickl)) #trail_len / duration
        peak_mag_vel=(vel_pickl[np.argmin(abs_mag_pickl)])   

        begin_height=(height_total[0])
        end_height=(height_total[-1])
        peak_mag_height=(height_total[np.argmin(abs_total)])

        peak_abs_mag=(np.min(abs_total))
        beg_abs_mag=(abs_total[0])
        end_abs_mag=(abs_total[-1])

        duration=(time_total[-1]-time_total[0])

        zenith_angle=(90 - elev_angle_pickl[0]*180/np.pi)
        trail_len=((height_total[0] - height_total[-1])/(np.sin(np.radians(elev_angle_pickl[0]*180/np.pi))))
        
        # vel_avg_norot=( ((height_pickl[0] - height_pickl[-1])/(np.sin(np.radians(elev_angle_pickl[0]*180/np.pi))))/(time_pickl[-1]-time_pickl[0]) ) #trail_len / duration

        # name=(name_file.split('_trajectory')[0]+'A')
        

        Dynamic_pressure_peak_abs_mag=(wmpl.Utils.Physics.dynamicPressure(lat_dat, lon_dat, height_total[np.argmin(abs_total)]*1000, jd_dat, vel_pickl[np.argmin(abs_mag_pickl)]*1000))

        # check if in os.path.join(root, name_file) present and then open the .json file with the same name as the pickle file with in stead of _trajectory.pickle it has _sim_fit_latest.json
        if os.path.isfile(os.path.join(root, name_file.split('_trajectory')[0]+'_sim_fit.json')):
            with open(os.path.join(root, name_file.split('_trajectory')[0]+'_sim_fit.json'),'r') as json_file: # 20210813_061453_sim_fit.json
                data = json.load(json_file)
                mass=(data['m_init'])
                # add also rho	sigma	erosion_height_start	erosion_coeff	erosion_mass_index	erosion_mass_min	erosion_mass_max	erosion_range	erosion_energy_per_unit_cross_section	erosion_energy_per_unit_mass
                # mass=(data['m_init'])
                rho=(data['rho'])
                sigma=(data['sigma'])
                erosion_height_start=(data['erosion_height_start']/1000)
                erosion_coeff=(data['erosion_coeff'])
                erosion_mass_index=(data['erosion_mass_index'])
                erosion_mass_min=(data['erosion_mass_min'])
                erosion_mass_max=(data['erosion_mass_max'])

                # Compute the erosion range
                erosion_range=(np.log10(data['erosion_mass_max']) - np.log10(data['erosion_mass_min']))

                cost_path = os.path.join(root, name_file.split('_trajectory')[0]+'_sim_fit.json')

                # Load the constants
                const, _ = loadConstants(cost_path)
                const.dens_co = np.array(const.dens_co)

                # Compute the erosion energies
                erosion_energy_per_unit_cross_section, erosion_energy_per_unit_mass = wmpl.MetSim.MetSimErosion.energyReceivedBeforeErosion(const)
                erosion_energy_per_unit_cross_section_arr=(erosion_energy_per_unit_cross_section)
                erosion_energy_per_unit_mass_arr=(erosion_energy_per_unit_mass)

        else:
            # if no data on weight is 0
            mass=(0)
            rho=(0)
            sigma=(0)
            erosion_height_start=(0)
            erosion_coeff=(0)
            erosion_mass_index=(0)
            erosion_mass_min=(0)
            erosion_mass_max=(0)
            erosion_range=(0)
            erosion_energy_per_unit_cross_section_arr=(0)
            erosion_energy_per_unit_mass_arr=(0)

        #############################################################
        fig, axs = plt.subplots(1, 2)

        if plot_color!='':
            axs[0].plot(abs_total,height_total,linestyle='--',marker='x',label='1',color=plot_color)
        else:
            axs[0].plot(abs_mag_01,height_01,linestyle='--',marker='x',label='1')
            axs[0].plot(abs_mag_02,height_02,linestyle='--',marker='x',label='2')
        axs[0].set_xlabel('abs.mag [-]')
        axs[0].set_ylabel('height [km]')
        # invert the x axis
        axs[0].invert_xaxis()
        axs[0].legend()
        axs[0].grid(True)

        if plot_color!='':
            axs[1].plot(time_total,vel_total,linestyle='--',marker='x',label='1',color=plot_color)
        else:
            axs[1].plot(time_01,vel_01,linestyle='None', marker='.',label='1')
            axs[1].plot(time_02,vel_02,linestyle='None', marker='.',label='2')

        axs[1].set_xlabel('time [s]')
        axs[1].set_ylabel('velocity [km/s]')
        axs[1].legend()
        axs[1].grid(True)

        # Data to populate the dataframe
        data_picklefile_pd = {
            'solution_id': [name],
            # 'shower_code': [shower_code_sim],
            'vel_init_norot': [vel_init_norot],
            'vel_avg_norot': [vel_avg_norot],
            'duration': [duration],
            'mass': [mass],
            'peak_mag_height': [peak_mag_height],
            'begin_height': [begin_height],
            'end_height': [end_height],
            # 't0': [t0],
            'peak_abs_mag': [peak_abs_mag],
            'beg_abs_mag': [beg_abs_mag],
            'end_abs_mag': [end_abs_mag],
            # 'F': [F],
            'trail_len': [trail_len],
            # 'deceleration_lin': [acceleration_lin],
            # 'deceleration_parab': [acceleration_parab],
            # 'decel_parab_t0': [acceleration_parab_t0],
            # 'decel_t0': [decel_t0],
            # 'decel_jacchia': [acc_jacchia],
            'zenith_angle': [zenith_angle],
            # 'kurtosis': [kurtosyness],
            # 'skew': [skewness],
            # 'kc': [kc_par],
            'Dynamic_pressure_peak_abs_mag': [Dynamic_pressure_peak_abs_mag],
            # 'a_acc': [a3],
            # 'b_acc': [b3],
            # 'c_acc': [c3],
            # 'a_t0': [a_t0],
            # 'b_t0': [b_t0],
            # 'c_t0': [c_t0],
            # 'a1_acc_jac': [jac_a1],
            # 'a2_acc_jac': [jac_a2],
            # 'a_mag_init': [a3_Inabs],
            # 'b_mag_init': [b3_Inabs],
            # 'c_mag_init': [c3_Inabs],
            # 'a_mag_end': [a3_Outabs],
            # 'b_mag_end': [b3_Outabs],
            # 'c_mag_end': [c3_Outabs],
            'rho': [rho],
            'sigma': [sigma],
            'erosion_height_start': [erosion_height_start],
            'erosion_coeff': [erosion_coeff],
            'erosion_mass_index': [erosion_mass_index],
            'erosion_mass_min': [erosion_mass_min],
            'erosion_mass_max': [erosion_mass_max],
            'erosion_range': [erosion_range],
            'erosion_energy_per_unit_cross_section': [erosion_energy_per_unit_cross_section_arr],
            'erosion_energy_per_unit_mass': [erosion_energy_per_unit_mass_arr]
        }

        # Create the dataframe
        infov_sim = pd.DataFrame(data_picklefile_pd)

        return infov_sim, fig, axs
    else:
        print('The file does not exist')
        return None


def SIM_pickle_data_and_plot(data_path, plot_color=''):
    # split the data_path to file and root
    root, name_file = os.path.split(data_path)
    print('root', root)
    
    # Shower=Shower
    # keep the first 14 characters of the ID
    name=name_file[:15]

    print('Loading pickle file: ', data_path)
    # check if there are any pickle files in the 
    if os.path.isfile(os.path.join(root, name_file)):

        traj = wmpl.Utils.Pickling.loadPickle(root, name_file)

        # zenith_angle= data['params']['zenith_angle']['val']*180/np.pi

        vel_sim=traj.simulation_results.leading_frag_vel_arr#['brightest_vel_arr']#['leading_frag_vel_arr']#['main_vel_arr']
        ht_sim=traj.simulation_results.leading_frag_height_arr#['brightest_height_arr']['leading_frag_height_arr']['main_height_arr']
        time_sim=traj.simulation_results.time_arr#['main_time_arr']
        abs_mag_sim=traj.simulation_results.abs_magnitude
        len_sim=traj.simulation_results.brightest_length_arr#['brightest_length_arr']
        
        ht_obs=traj.ht_sampled

        # # find the index of the first element of the simulation that is equal to the first element of the observation
        index_ht_sim=next(x for x, val in enumerate(ht_sim) if val <= ht_obs[0])
        # find the index of the last element of the simulation that is equal to the last element of the observation
        index_ht_sim_end=next(x for x, val in enumerate(ht_sim) if val <= ht_obs[-1])

        abs_mag_sim=abs_mag_sim[index_ht_sim:index_ht_sim_end]
        vel_sim=vel_sim[index_ht_sim:index_ht_sim_end]
        time_sim=time_sim[index_ht_sim:index_ht_sim_end]
        ht_sim=ht_sim[index_ht_sim:index_ht_sim_end]
        len_sim=len_sim[index_ht_sim:index_ht_sim_end]

        # divide the vel_sim by 1000 considering is a list
        time_sim = [i-time_sim[0] for i in time_sim]
        vel_sim = [i/1000 for i in vel_sim]
        len_sim = [(i-len_sim[0])/1000 for i in len_sim]
        ht_sim = [i/1000 for i in ht_sim]
        
        ht_obs=[x/1000 for x in ht_obs]

        closest_indices = find_closest_index(ht_sim, ht_obs)

        v0 = vel_sim[0]

        abs_mag_sim=[abs_mag_sim[jj_index_cut] for jj_index_cut in closest_indices]
        vel_sim=[vel_sim[jj_index_cut] for jj_index_cut in closest_indices]
        time_sim=[time_sim[jj_index_cut] for jj_index_cut in closest_indices]
        ht_sim=[ht_sim[jj_index_cut] for jj_index_cut in closest_indices]
        len_sim=[len_sim[jj_index_cut] for jj_index_cut in closest_indices]

        return abs_mag_sim, ht_sim, vel_sim, time_sim, len_sim, ht_obs

        
        








if __name__ == "__main__":

    import argparse


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Randomly generate parameters for the erosion model, run it, and store results to disk.")

    arg_parser.add_argument('--output_dir', metavar='OUTPUT_PATH', type=str, default=r'C:\Users\maxiv\Desktop\20230811-082648.931419\20230811_082648_GenSim',\
        help="Path to the output directory.")

    arg_parser.add_argument('--simclass', metavar='SIM_CLASS', type=str, default='ErosionSimParametersEMCCD_PER_real_CAMO_v60', \
        help="Use simulation parameters from the given class. Options: {:s}".format(", ".join(SIM_CLASSES_NAMES)))

    arg_parser.add_argument('--nsims', metavar='SIM_NUM', type=int, default=10, \
        help="Number of simulations to do.")

    arg_parser.add_argument('--real_event', metavar='REAL_EVENT', type=str, default='20230811_082648_trajectory.pickle', \
        help="The real event name.") 

    arg_parser.add_argument('--real_data', metavar='REAL_DATA', type=str, default=r'C:\Users\maxiv\Desktop\20230811-082648.931419\20230811_082648_trajectory.pickle', \
        help="The real data base the values for the generated simulations.") 

    arg_parser.add_argument('--cores', metavar='CORES', type=int, default=None, \
        help="Number of cores to use. All by default.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    # Make the output directory
    mkdirP(cml_args.output_dir)

    # Init simulation parameters with the given class name
    erosion_sim_params = SIM_CLASSES[SIM_CLASSES_NAMES.index(cml_args.simclass)]()

    # check if cml_args.real_data is a path to a file
    if os.path.isfile(cml_args.real_data):
        print("Real data file exists")
        # if is a csv file then read it
        if cml_args.real_data.endswith('.csv'):
            real_data = pd.read_csv(cml_args.real_data)
            # delete any row that do not have 20230811_082648 in the solution_id column 
            real_data = real_data[real_data['solution_id'].str.contains(cml_args.real_event)]
            fig, axs = plt.subplots(1, 2)

            axs[0].set_xlabel('abs.mag [-]')
            axs[0].set_ylabel('height [km]')
            # invert the x axis
            axs[0].invert_xaxis()
            axs[0].grid(True)
            
            axs[1].set_xlabel('time [s]')
            axs[1].set_ylabel('velocity [km/s]')
            axs[1].grid(True)
            # plot the data
            name_file = real_data['solution_id'][0]
            fig.suptitle(real_data['solution_id'][0])
        elif cml_args.real_data.endswith('.pickle'):
            # use the pickle file to extract the data
            real_data, fig, axs = real_pickle_data_and_plot(cml_args.real_data)
            # fig.suptitle(name_file.split('_trajectory')[0]+'A')
            name_file = real_data['solution_id'][0]
            fig.suptitle(real_data['solution_id'][0])
            
        # get from real_data the beg_abs_mag value of the first row and set it as the lim_mag_faintest value
        erosion_sim_params.lim_mag_faintest = real_data['beg_abs_mag'][0]+0.01
        erosion_sim_params.lim_mag_brightest = real_data['beg_abs_mag'][0]-0.01
        erosion_sim_params.lim_mag_len_end_faintest = real_data['end_abs_mag'][0]+0.01
        erosion_sim_params.lim_mag_len_end_brightest = real_data['end_abs_mag'][0]-0.01

        # # Simulation height range (m) that will be used to map the output to a grid
        # erosion_sim_params.sim_height = MetParam(70000, 130000)

        ## Physical parameters

        # find the at what is the order of magnitude of the real_data['mass'][0]
        order = int(np.floor(np.log10(real_data['mass'][0])))
        # create a MetParam object with the mass range that is above and below the real_data['mass'][0] by 2 orders of magnitude
        erosion_sim_params.m_init = MetParam(real_data['mass'][0]-10**order, real_data['mass'][0]+10**order)

        # Initial velocity range (m/s) 
        erosion_sim_params.v_init = MetParam(real_data['vel_init_norot'][0]*1000-50, real_data['vel_init_norot'][0]*1000+150) # 60091.41691

        # Zenith angle range
        erosion_sim_params.zenith_angle = MetParam(np.radians(real_data['zenith_angle'][0]-0.01), np.radians(real_data['zenith_angle'][0]+0.01)) # 43.466538

        # print all the modfiend values
        print('min initial mag:',erosion_sim_params.lim_mag_faintest)
        print('max initial mag:',erosion_sim_params.lim_mag_brightest)
        print('min final mag:',erosion_sim_params.lim_mag_len_end_faintest)
        print('max final mag:',erosion_sim_params.lim_mag_len_end_brightest)
        print('min mass:',erosion_sim_params.m_init.min)
        print('max mass:',erosion_sim_params.m_init.max)
        print('min velocity:',erosion_sim_params.v_init.min)
        print('max velocity:',erosion_sim_params.v_init.max)
        print('min zenith angle:',np.degrees(erosion_sim_params.zenith_angle.min))
        print('max zenith angle:',np.degrees(erosion_sim_params.zenith_angle.max))


        # add the real data of duration	peak_mag_height	begin_height	end_height
        erosion_sim_params.real_duration = real_data['duration'][0]
        # erosion_sim_params.param_list.append("real_duration")

        erosion_sim_params.real_peak_abs_mag = real_data['peak_abs_mag'][0]
        # erosion_sim_params.param_list.append("real_peak_abs_mag")

        erosion_sim_params.real_peak_mag_height = real_data['peak_mag_height'][0]*1000
        # erosion_sim_params.param_list.append("real_peak_mag_height")

        erosion_sim_params.real_begin_height = real_data['begin_height'][0]*1000
        # erosion_sim_params.param_list.append("real_begin_height")

        erosion_sim_params.real_end_height = real_data['end_height'][0]*1000
        # erosion_sim_params.param_list.append("real_end_height")


        # # Density range (kg/m^3)
        # erosion_sim_params.rho = MetParam(100, 1000)

        # # Intrinsic ablation coeff range (s^2/m^2)
        # erosion_sim_params.sigma = MetParam(0.008/1e6, 0.03/1e6) 

        # # Erosion height range
        # erosion_sim_params.erosion_height_start = MetParam(115000, 119000)

        # # Erosion coefficient (s^2/m^2)
        # erosion_sim_params.erosion_coeff = MetParam(0.0, 1/1e6)

        # # Mass index
        # erosion_sim_params.erosion_mass_index = MetParam(1.5, 2.5)

        # # Minimum mass for erosion
        # erosion_sim_params.erosion_mass_min = MetParam(5e-12, 1e-10)

        # # Maximum mass for erosion
        # erosion_sim_params.erosion_mass_max = MetParam(1e-10, 5e-8)

    else:
        print("Real data file does not exist")
        real_data = None

    nsim_results=0
    print('Number of simulations:', cml_args.nsims)

    # Generate simulations using multiprocessing
    input_list = [[cml_args.output_dir, copy.deepcopy(erosion_sim_params), \
        np.random.randint(0, 2**31 - 1),MIN_FRAMES_VISIBLE] for _ in range(cml_args.nsims)]
    results_list = domainParallelizer(input_list, generateErosionSim, cores=cml_args.cores)

    # print(results_list)

    # count how many None are in the results_list
    count_none=0
    for res in results_list:
        if res is None:
            count_none+=1


    nsim_results=nsim_results+cml_args.nsims-count_none
    print('Resulted simulations:', nsim_results)
    print('Failed siulations', len(results_list)/100*count_none,'%')

    # plot the pickle files data that are not none in the results_list
    # do not plot more than 10 curves
    jj_plots_curve=0
    for res in results_list:
        if res is not None:
            
            if jj_plots_curve>100:
                break
            # get the first value of res
            abs_mag_sim, ht_sim, vel_sim, time_sim, len_sim, ht_obs=SIM_pickle_data_and_plot(res[0], plot_color='b')
            # plot the data
            # reduce the size of the lines and make it transparent
            axs[0].plot(abs_mag_sim, ht_sim, color='b', alpha=0.5, linewidth=0.5)
            axs[1].plot(time_sim, vel_sim, color='b', alpha=0.5, linewidth=0.5) #, marker='x',linestyle='--',
            # put the plotted curve behind the one that was plotted before
            axs[0].set_zorder(0)
            axs[1].set_zorder(0)
            jj_plots_curve+=1


    saveProcessedList(cml_args.output_dir, results_list, erosion_sim_params.__class__.__name__, \
        MIN_FRAMES_VISIBLE)

    # if time.time() - start_time > timeout:
    #     print("Timeout reached, breaking the loop.")
    #     break

    # give space between the subplots
    fig.tight_layout(pad=2.0)
    fig.savefig(cml_args.output_dir+os.sep+'GeratedSIM_'+name_file+'.png', dpi=300)
    plt.close(fig)
    # plt.show()