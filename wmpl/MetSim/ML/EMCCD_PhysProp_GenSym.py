"""
The code is used to extract the physical properties of the simulated showers from EMCCD observations
by selecting the most similar simulated events in the PC space using:
- Mode of the siumulated events
- The min of the KDE esults
- Principal Component Regression (PCR)
"""

import json
import copy
import re
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv
import numpy as np
import subprocess
import glob
import os
import pickle
import seaborn as sns
import scipy.spatial.distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import wmpl
import shutil
from scipy.stats import kurtosis, skew
from wmpl.Utils.OSTools import mkdirP
from matplotlib.ticker import ScalarFormatter
import math
from scipy.stats import gaussian_kde
from wmpl.Utils.PyDomainParallelizer import domainParallelizer
from scipy.linalg import svd
from wmpl.MetSim.GUI import loadConstants, saveConstants,SimulationResults
from wmpl.MetSim.MetSimErosion import runSimulation, Constants, zenithAngleAtSimulationBegin
from scipy.interpolate import interp1d
from matplotlib.colors import Normalize
from scipy.optimize import minimize
import scipy.optimize as opt
import sys
from scipy.stats import zscore
import scipy.spatial
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from wmpl.Utils.AtmosphereDensity import fitAtmPoly
from sklearn.cluster import KMeans
import scipy.stats as stats
from sklearn.preprocessing import PowerTransformer
from wmpl.MetSim.ML.GenerateSimulations import generateErosionSim,saveProcessedList,MetParam
from wmpl.Utils.TrajConversions import J2000_JD, date2JD
import warnings
import itertools
import time
from multiprocessing import Pool


# CONSTANTS ###########################################################################################

FPS = 32
NAME_SUFX_GENSIM = "_GenSim"
NAME_SUFX_CSV_OBS = "_obs.csv"
NAME_SUFX_CSV_SIM = "_sim.csv"
NAME_SUFX_CSV_SIM_NEW = "_sim_new.csv"
NAME_SUFX_CSV_CURRENT_FIT = "_fit_sim.csv"
NAME_SUFX_CSV_PHYSICAL_FIT_RESULTS = "_physical_prop.csv"

SAVE_SELECTION_FOLDER='Selection'
VAR_SEL_DIR_SUFX = '_sel_var_vs_physProp'
PCA_SEL_DIR_SUFX = '_sel_PCA_vs_physProp'
SAVE_RESULTS_FOLDER='Results'
SAVE_RESULTS_FOLDER_EVENTS_PLOTS='Results'+os.sep+'events_plots'

# sigma value of the RMSD that is considered to select a good fit
SIGMA_ERR = 1 # 1.96 # 95CI
MAG_RMSD = 0.25
# MAG_RMSD = 0.25 # for heavy
# MAG_RMSD = 0.20 # for steep fast
# MAG_RMSD = 0.15 # for shallow slow 
# MAG_RMSD = 0.05 # for small

LEN_RMSD = 0.04 # 0.02
# LEN_RMSD = 0.04
# MAG_RMSD = 0.08
# LEN_RMSD = 0.04 # 0.025

# Use the IF function, one of the logical functions, to return one value if a condition is true and another value if it's false. For example: =IF(A2>B2,"Over Budget","OK") =IF(A2=B2,B4-A4,"")

# # Calculate the cumulative probability for the z-value, the confidence level is the percentage of the area within ±z_value
# CONFIDENCE_LEVEL = (2 * stats.norm.cdf(SIGMA_ERR) - 1)*100

# Length of data that will be used as an input during training
DATA_LENGTH = 256
# Default number of minimum frames for simulation
MIN_FRAMES_VISIBLE = 4

# python -m EMCCD_PCA_Shower_PhysProp "C:\Users\maxiv\Documents\UWO\Papers\1)PCA\PCA_Error_propagation\TEST" "PER" "C:\Users\maxiv\Documents\UWO\Papers\1)PCA\PCA_Error_propagation" 1000
# python -m EMCCD_PCA_Shower_PhysProp "C:\Users\maxiv\Documents\UWO\Papers\1)PCA\PCA_Error_propagation\TEST" "PER" "C:\Users\maxiv\Documents\UWO\Papers\1)PCA\PCA_Error_propagation" 1000 > output.txt    

# FUNCTIONS ###########################################################################################

# create a txt file where you save averithing that has been printed
class Logger(object):
    def __init__(self, directory=".", filename="log.txt"):
        self.terminal = sys.stdout
        # Ensure the directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Combine the directory and filename to create the full path
        filepath = os.path.join(directory, filename)
        self.log = open(filepath, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # This might be necessary as stdout could call flush
        self.terminal.flush()

    def close(self):
        # Close the log file when done
        self.log.close()


def find_closest_index(time_arr, time_sampled):
    closest_indices = []
    for sample in time_sampled:
        closest_index = min(range(len(time_arr)), key=lambda i: abs(time_arr[i] - sample))
        closest_indices.append(closest_index)
    return closest_indices


def cubic_lag(t, a, b, c, t0):
    """
    Quadratic lag function.
    """

    # Only take times <= t0
    t_before = t[t <= t0]

    # Only take times > t0
    t_after = t[t > t0]

    # Compute the lag linearly before t0
    l_before = np.zeros_like(t_before)+c

    # Compute the lag quadratically after t0
    l_after = -abs(a)*(t_after - t0)**3 - abs(b)*(t_after - t0)**2 + c

    return np.concatenate((l_before, l_after))


def cubic_velocity(t, a, b, v0, t0):
    """
    Quadratic velocity function.
    """

    # Only take times <= t0
    t_before = t[t <= t0]

    # Only take times > t0
    t_after = t[t > t0]

    # Compute the velocity linearly before t0
    v_before = np.ones_like(t_before)*v0

    # Compute the velocity quadratically after t0
    v_after = -3*abs(a)*(t_after - t0)**2 - 2*abs(b)*(t_after - t0) + v0

    return np.concatenate((v_before, v_after))


def cubic_acceleration(t, a, b, t0):
    """
    Quadratic acceleration function.
    """

    # Only take times <= t0
    t_before = t[t <= t0]

    # Only take times > t0
    t_after = t[t > t0]

    # No deceleration before t0
    a_before = np.zeros_like(t_before)

    # Compute the acceleration quadratically after t0
    a_after = -6*abs(a)*(t_after - t0) - 2*abs(b)

    return np.concatenate((a_before, a_after))


def lag_residual(params, t_time, l_data):
    """
    Residual function for the optimization.
    """

    return np.sum((l_data - cubic_lag(t_time, *params))**2)


def vel_residual(params, t_time, l_data):
    """
    Residual function for the optimization.
    """

    return np.sum((l_data - cubic_velocity(t_time, *params))**2)


def fit_mag_polin2_RMSD(data_mag, time_data):

    # Select the data up to the minimum value
    x1 = time_data[:np.argmin(data_mag)]
    y1 = data_mag[:np.argmin(data_mag)]

    # Fit the first parabolic curve
    coeffs1 = np.polyfit(x1, y1, 2)
    fit1 = np.polyval(coeffs1, x1)

    # Select the data from the minimum value onwards
    x2 = time_data[np.argmin(data_mag):]
    y2 = data_mag[np.argmin(data_mag):]

    # Fit the second parabolic curve
    coeffs2 = np.polyfit(x2, y2, 2)
    fit2 = np.polyval(coeffs2, x2)

    # concatenate fit1 and fit2
    fit1=np.concatenate((fit1, fit2))

    residuals_pol = data_mag - fit1
    # avg_residual_pol = np.mean(abs(residuals_pol))
    rmsd_pol = np.sqrt(np.mean(residuals_pol**2))

    return fit1, residuals_pol, rmsd_pol,'Polinomial Fit'


def fit_lag_t0_RMSD_old(lag_data,time_data,velocity_data):
    v_init=velocity_data[0]
    # initial guess of deceleration decel equal to linear fit of velocity
    p0 = [np.mean(lag_data), 0, 0, np.mean(time_data)]
    opt_res = opt.minimize(lag_residual, p0, args=(np.array(time_data), np.array(lag_data)), method='Nelder-Mead')
    a_t0, b_t0, c_t0, t0 = opt_res.x
    fitted_lag_t0 = cubic_lag(np.array(time_data), a_t0, b_t0, c_t0, t0)
    
    opt_res_vel = opt.minimize(vel_residual, [a_t0, b_t0, v_init, t0], args=(np.array(time_data), np.array(velocity_data)), method='Nelder-Mead')
    a_t0, b_t0, v_init_new, t0 = opt_res_vel.x # problem with the small time
    fitted_vel_t0 = cubic_velocity(np.array(time_data), a_t0, b_t0, v_init, t0)

    fitted_acc_t0 = cubic_acceleration(np.array(time_data), a_t0, b_t0, t0)
    residuals_t0 = lag_data - fitted_lag_t0
    rmsd_t0 = np.sqrt(np.mean(residuals_t0 ** 2))

    return fitted_lag_t0, residuals_t0, rmsd_t0, 'Cubic Fit', fitted_vel_t0, fitted_acc_t0

def fit_lag_t0_RMSD(lag_data, time_data, velocity_data):
    v_init = velocity_data[0]
    # initial guess of deceleration decel equal to linear fit of velocity
    p0 = [np.mean(lag_data), 0, 0, np.mean(time_data)]
    opt_res = opt.minimize(lag_residual, p0, args=(np.array(time_data), np.array(lag_data)), method='Nelder-Mead')
    a_t0, b_t0, c_t0, t0 = opt_res.x
    fitted_lag_t0 = cubic_lag(np.array(time_data), a_t0, b_t0, c_t0, t0)
    
    # Optimize velocity residual based on initial guess from lag residual
    opt_res_vel = opt.minimize(vel_residual, [a_t0, b_t0, v_init, t0], args=(np.array(time_data), np.array(velocity_data)), method='Nelder-Mead')
    a_t0_vel, b_t0_vel, v_init_vel, t0_vel = opt_res_vel.x
    fitted_vel_t0_vel = cubic_velocity(np.array(time_data), a_t0_vel, b_t0_vel, v_init_vel, t0_vel)
    
    # # Compute fitted velocity from original lag optimization
    # fitted_vel_t0_lag = cubic_velocity(np.array(time_data), a_t0, b_t0, v_init, t0)

    # Compute fitted velocity from original lag optimization
    fitted_vel_t0_lag = cubic_velocity(np.array(time_data), a_t0, b_t0, v_init_vel, t0)

    # # Compute fitted velocity from original lag optimization
    # fitted_vel_t0_lag_vel = cubic_velocity(np.array(time_data), a_t0, b_t0, v_init_vel, t0)
    
    # Calculate residuals
    residuals_vel_vel = velocity_data - fitted_vel_t0_vel
    residuals_vel_lag = velocity_data - fitted_vel_t0_lag
    
    rmsd_vel_vel = np.sqrt(np.mean(residuals_vel_vel ** 2))
    rmsd_vel_lag = np.sqrt(np.mean(residuals_vel_lag ** 2))
    
    # Choose the best fitted velocity based on RMSD
    if rmsd_vel_vel < rmsd_vel_lag:
        best_fitted_vel_t0 = fitted_vel_t0_vel
        best_a_t0, best_b_t0, best_t0 = a_t0_vel, b_t0_vel, t0_vel
    else:
        best_fitted_vel_t0 = fitted_vel_t0_lag
        best_a_t0, best_b_t0, best_t0 = a_t0, b_t0, t0
    
    fitted_acc_t0 = cubic_acceleration(np.array(time_data), best_a_t0, best_b_t0, best_t0)
    residuals_t0 = lag_data - fitted_lag_t0
    rmsd_t0 = np.sqrt(np.mean(residuals_t0 ** 2))

    return fitted_lag_t0, residuals_t0, rmsd_t0, 'Cubic Fit', best_fitted_vel_t0, fitted_acc_t0


def find_noise_of_data(data, plot_case=False):
    # make a copy of data_obs
    data_obs = copy.deepcopy(data)

    fitted_lag_t0_lag, residuals_t0_lag, rmsd_t0_lag, fit_type_lag, fitted_vel_t0, fitted_acc_t0 = fit_lag_t0_RMSD(data_obs['lag'],data_obs['time'], data_obs['velocities'])
    # now do it for fit_mag_polin2_RMSD
    fit_pol_mag, residuals_pol_mag, rmsd_pol_mag, fit_type_mag = fit_mag_polin2_RMSD(data_obs['absolute_magnitudes'],data_obs['time'])

    # create a pd dataframe with fit_pol_mag and fitted_vel_t0 and time and height
    fit_funct = {
        'velocities': fitted_vel_t0,
        'height': data_obs['height'],
        'absolute_magnitudes': fit_pol_mag,
        'time': data_obs['time'],
        'lag': fitted_lag_t0_lag
    }

    if plot_case:
        fig, ax = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
        # flat the ax
        ax = ax.flatten()
        plot_side_by_side(data,fig, ax,'go','Obsevation')

        plot_side_by_side(fit_funct,fig, ax,'k--','fit')

        return rmsd_t0_lag, rmsd_pol_mag, fit_pol_mag, fitted_lag_t0_lag, fit_funct, fig, ax
    else:
        return rmsd_t0_lag, rmsd_pol_mag, fit_pol_mag, fitted_lag_t0_lag, fit_funct


#### Generate Observation #########################################################################

def generate_observation_realization(data, rmsd_lag, rmsd_mag, fit_funct, name='', fig='', ax='', plot_case=False):

    # print a . so that the next will be on the same line
    print('.', end='')
    # make a copy of data_obs
    data_obs = copy.deepcopy(data)
    fit_pol_mag = copy.deepcopy(fit_funct['absolute_magnitudes'])
    fitted_lag_t0_lag = copy.deepcopy(fit_funct['lag'])
    fitted_lag_t0_vel = copy.deepcopy(fit_funct['velocities'])

    if name!='':
        # print(name)
        data_obs['name']=name

    data_obs['type']='Realization'

    ### ADD NOISE ###

    # Add noise to magnitude data (Gaussian noise) for each realization
    fit_pol_mag += np.random.normal(loc=0.0, scale=rmsd_mag, size=len(data_obs['absolute_magnitudes']))
    data_obs['absolute_magnitudes']=fit_pol_mag
    # Add noise to length data (Gaussian noise) for each realization
    fitted_lag_t0_lag += np.random.normal(loc=0.0, scale=rmsd_lag, size=len(data_obs['length']))
    data_obs['lag']=fitted_lag_t0_lag
    # add noise to velocity data considering the noise as rmsd_lag/(1.0/FPS)
    fitted_lag_t0_vel += np.random.normal(loc=0.0, scale=rmsd_lag/(1.0/FPS), size=len(data_obs['velocities']))
    # fitted_lag_t0_vel += np.random.normal(loc=0.0, scale=rmsd_lag*np.sqrt(2)/(1.0/FPS), size=len(data_obs['velocities']))
    data_obs['velocities']=fitted_lag_t0_vel

    ### ###

    # data_obs['lag']=np.array(data_obs['length'])-(data_obs['v_init']*np.array(data_obs['time'])+data_obs['length'][0])
    data_obs['length']= np.array(data_obs['lag'])+(data_obs['v_init']*np.array(data_obs['time'])+data_obs['length'][0])

    # # get the new velocity with noise
    # for vel_ii in range(1,len(data_obs['time'])-1):
    #     diff_1=abs((data_obs['time'][vel_ii]-data_obs['time'][vel_ii-1])-1.0/FPS)
    #     diff_2=abs((data_obs['time'][vel_ii+1]-data_obs['time'][vel_ii-1])-1.0/FPS)

    #     if diff_1<diff_2:
    #         data_obs['velocities'][vel_ii]=(data_obs['length'][vel_ii]-data_obs['length'][vel_ii-1])/(data_obs['time'][vel_ii]-data_obs['time'][vel_ii-1])
    #     else:
    #         data_obs['velocities'][vel_ii+1]=(data_obs['length'][vel_ii+1]-data_obs['length'][vel_ii-1])/(data_obs['time'][vel_ii+1]-data_obs['time'][vel_ii-1])

    if plot_case:
        plot_side_by_side(data_obs,fig, ax)

    # compute the initial velocity
    data_obs['v_init']=data_obs['velocities'][0] # m/s
    # compute the average velocity
    data_obs['v_avg']=np.mean(data_obs['velocities']) # m/s

    # data_obs['v_avg']=data_obs['v_avg']*1000 # km/s

    pd_datfram_PCA = array_to_pd_dataframe_PCA(data_obs)

    return pd_datfram_PCA


#### Generate Simulations #########################################################################

class ErosionSimParametersEMCCD_Comet(object):
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
        self.lim_mag_faintest = +5.49    # change the startng height
        self.lim_mag_brightest = +5.48   # change the startng height
        self.lim_mag_len_end_faintest = +5.61
        self.lim_mag_len_end_brightest = +5.60

        # Power of a zero-magnitude meteor (Watts)
        self.P_0m = 935

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
        self.rho = MetParam(100, 2000)
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

# List of classed that can be used for data generation and postprocessing
SIM_CLASSES = [ErosionSimParametersEMCCD_Comet]
SIM_CLASSES_NAMES = [c.__name__ for c in SIM_CLASSES]

def run_simulation(path_and_file_MetSim, real_event):
    '''
        path_and_file = must be a json file generated file from the generate_simulationsm function or from Metsim file
    '''

    # Load the nominal simulation parameters
    const_nominal, _ = loadConstants(path_and_file_MetSim)
    const_nominal.dens_co = np.array(const_nominal.dens_co)

    dens_co=np.array(const_nominal.dens_co)

    ### Calculate atmosphere density coeffs (down to the bottom observed height, limit to 15 km) ###

    # Assign the density coefficients
    const_nominal.dens_co = dens_co

    # Turn on plotting of LCs of individual fragments 
    const_nominal.fragmentation_show_individual_lcs = True

    # # Minimum height (m)
    # const_nominal.h_kill = 60000

    # # Initial meteoroid height (m)
    # const_nominal.h_init = 180000

    try:
        # Run the simulation
        frag_main, results_list, wake_results = runSimulation(const_nominal, compute_wake=False)
        simulation_MetSim_object = SimulationResults(const_nominal, frag_main, results_list, wake_results)
    except ZeroDivisionError as e:
        print(f"Error during simulation: {e}")
        const_nominal = Constants()
        # Run the simulation
        frag_main, results_list, wake_results = runSimulation(const_nominal, compute_wake=False)
        simulation_MetSim_object = SimulationResults(const_nominal, frag_main, results_list, wake_results)

    # # print the column of simulation_MetSim_object to see what is inside
    # print(simulation_MetSim_object.__dict__.keys())
    # print(simulation_MetSim_object.const.__dict__.keys())
  
    # ax[0].plot(sr_nominal_1D_KDE.abs_magnitude, sr_nominal_1D_KDE.leading_frag_height_arr/1000, label="Mode", color='r')
    # ax[2].plot(sr_nominal.leading_frag_vel_arr/1000, sr_nominal.leading_frag_height_arr/1000, color='k', abel="Simulated")

    gensim_data_metsim = read_RunSim_output(simulation_MetSim_object, real_event, path_and_file_MetSim)

    pd_Metsim  = array_to_pd_dataframe_PCA(gensim_data_metsim)

    return simulation_MetSim_object, gensim_data_metsim, pd_Metsim

def safe_generate_erosion_sim(params):
    try:
        return generateErosionSim(*params)
    except TypeError as e:
        print(f"Error in generateErosionSim: {e}")
        return None

def generate_simulations(real_data,simulation_MetSim_object,gensim_data,numb_sim,output_folder,file_name,plot_case=False, CI_physical_param=''):

    # if real_data['solution_id'].iloc[0].endswith('.json'):
    #     mass_sim=gensim_data['mass']
    #     v_init_180km =gensim_data['vel_180km']

    # else:
    mass_sim= simulation_MetSim_object.const.m_init
    # print('mass_sim',mass_sim)

    v_init_180km = simulation_MetSim_object.const.v_init # in m/s
    # print('v_init_130km',v_init_130km)

    # Init simulation parameters with the given class name
    erosion_sim_params = SIM_CLASSES[SIM_CLASSES_NAMES.index('ErosionSimParametersEMCCD_Comet')]()
        
    # get from real_data the beg_abs_mag value of the first row and set it as the lim_mag_faintest value
    erosion_sim_params.lim_mag_faintest = real_data['beg_abs_mag'].iloc[0]+0.01
    erosion_sim_params.lim_mag_brightest = real_data['beg_abs_mag'].iloc[0]-0.01
    erosion_sim_params.lim_mag_len_end_faintest = real_data['end_abs_mag'].iloc[0]+0.01
    erosion_sim_params.lim_mag_len_end_brightest = real_data['end_abs_mag'].iloc[0]-0.01
    print('lim_mag_faintest',erosion_sim_params.lim_mag_faintest,'lim_mag_brightest',erosion_sim_params.lim_mag_brightest)
    print('lim_mag_len_end_faintest',erosion_sim_params.lim_mag_len_end_faintest,'lim_mag_len_end_brightest',erosion_sim_params.lim_mag_len_end_brightest)

    # find the at what is the order of magnitude of the real_data['mass'][0]
    order = int(np.floor(np.log10(mass_sim)))
    # create a MetParam object with the mass range that is above and below the real_data['mass'][0] by 2 orders of magnitude
    erosion_sim_params.m_init = MetParam(mass_sim-(10**order)/2, mass_sim+(10**order)/2)
    # erosion_sim_params.m_init = MetParam(mass_sim/2, mass_sim*2)

    # Initial velocity range (m/s) 
    erosion_sim_params.v_init = MetParam(v_init_180km-1000, v_init_180km+1000) # 60091.41691

    # Zenith angle range
    erosion_sim_params.zenith_angle = MetParam(np.radians(real_data['zenith_angle'].iloc[0]-0.01), np.radians(real_data['zenith_angle'].iloc[0]+0.01)) # 43.466538

    # erosion_sim_params.erosion_height_start = MetParam(real_data['peak_mag_height'].iloc[0]*1000+(real_data['begin_height'].iloc[0]-real_data['peak_mag_height'].iloc[0])*1000/2, real_data['begin_height'].iloc[0]*1000+(real_data['begin_height'].iloc[0]-real_data['peak_mag_height'].iloc[0])*1000/2) # 43.466538
    erosion_sim_params.erosion_height_start = MetParam(real_data['begin_height'].iloc[0]*1000-1000, real_data['begin_height'].iloc[0]*1000+4000) # 43.466538

    if CI_physical_param!='':
        erosion_sim_params.v_init = MetParam(CI_physical_param['v_init_180km'][0], CI_physical_param['v_init_180km'][1]) # 60091.41691
        erosion_sim_params.zenith_angle = MetParam(np.radians(CI_physical_param['zenith_angle'][0]), np.radians(CI_physical_param['zenith_angle'][1])) # 43.466538
        erosion_sim_params.m_init = MetParam(CI_physical_param['mass'][0], CI_physical_param['mass'][1])
        erosion_sim_params.rho = MetParam(CI_physical_param['rho'][0], CI_physical_param['rho'][1])
        erosion_sim_params.sigma = MetParam(CI_physical_param['sigma'][0], CI_physical_param['sigma'][1])
        erosion_sim_params.erosion_height_start = MetParam(CI_physical_param['erosion_height_start'][0], CI_physical_param['erosion_height_start'][1])
        erosion_sim_params.erosion_coeff = MetParam(CI_physical_param['erosion_coeff'][0], CI_physical_param['erosion_coeff'][1])
        erosion_sim_params.erosion_mass_index = MetParam(CI_physical_param['erosion_mass_index'][0], CI_physical_param['erosion_mass_index'][1])
        erosion_sim_params.erosion_mass_min = MetParam(CI_physical_param['erosion_mass_min'][0], CI_physical_param['erosion_mass_min'][1])
        erosion_sim_params.erosion_mass_max = MetParam(CI_physical_param['erosion_mass_max'][0], CI_physical_param['erosion_mass_max'][1])

        # check if a file with the name "log"+n_PC_in_PCA+"_"+str(len(df_sel))+"ev.txt" already exist
        if os.path.exists(output_folder+os.sep+"log_"+file_name[:15]+"_GenereateSimulations_range_NEW.txt"):
            # remove the file
            os.remove(output_folder+os.sep+"log_"+file_name[:15]+"_GenereateSimulations_range_NEW.txt")
        sys.stdout = Logger(output_folder,"log_"+file_name[:15]+"_GenereateSimulations_range_NEW.txt") # _30var_99%_13PC
    else:
        # check if a file with the name "log"+n_PC_in_PCA+"_"+str(len(df_sel))+"ev.txt" already exist
        if os.path.exists(output_folder+os.sep+"log_"+file_name[:15]+"_GenereateSimulations_range.txt"):
            # remove the file
            os.remove(output_folder+os.sep+"log_"+file_name[:15]+"GenereateSimulations_range.txt")
        sys.stdout = Logger(output_folder,"log_"+file_name[:15]+"GenereateSimulations_range.txt") # _30var_99%_13PC


    print('Run',numb_sim,'simulations with :')
    # to_plot_unit=['mass [kg]','rho [kg/m^3]','sigma [s^2/km^2]','erosion height start [km]','erosion coeff [s^2/km^2]','erosion mass index [-]','eros. mass min [kg]','eros. mass max [kg]']
    print('\\hline') #df_sel_save[df_sel_save['solution_id']==only_select_meteors_from][plotvar].values[0]
    print('Variables & min.val. & MAX.val. \\\\')

    print('\\hline')
    # - velocity: min 58992.19459103218 - MAX 60992.19459103218
    # print('- velocity: min',erosion_sim_params.v_init.min,'- MAX',erosion_sim_params.v_init.max)
    print(f"Velocity [km/s] & {'{:.4g}'.format(erosion_sim_params.v_init.min/1000)} & {'{:.4g}'.format(erosion_sim_params.v_init.max/1000)} \\\\")
    
    print('\\hline')
    # - zenith angle: min 28.736969960110045 - MAX 28.75696996011005
    # print('- zenith angle: min',np.degrees(erosion_sim_params.zenith_angle.min),'- MAX',np.degrees(erosion_sim_params.zenith_angle.max))
    print(f"Zenith ang. [deg] & {'{:.4g}'.format(np.degrees(erosion_sim_params.zenith_angle.min))} & {'{:.4g}'.format(np.degrees(erosion_sim_params.zenith_angle.max))} \\\\")

    print('\\hline') 
    # - Initial mag: min 5.45949291900601 - MAX 5.43949291900601
    # print('- Initial mag: min',erosion_sim_params.lim_mag_faintest,'- MAX',erosion_sim_params.lim_mag_brightest)
    print(f"Init. mag [-] & {'{:.4g}'.format(erosion_sim_params.lim_mag_faintest)} & {'{:.4g}'.format(erosion_sim_params.lim_mag_brightest)} \\\\")

    print('\\hline')
    # - Final mag: min 6.0268141526507435 - MAX 6.006814152650744
    # print('- Final mag: min',erosion_sim_params.lim_mag_len_end_faintest,'- MAX',erosion_sim_params.lim_mag_len_end_brightest)
    print(f"Fin. mag [-] & {'{:.4g}'.format(erosion_sim_params.lim_mag_len_end_faintest)} & {'{:.4g}'.format(erosion_sim_params.lim_mag_len_end_brightest)} \\\\")

    print('\\hline')
    # - Mass: min 5.509633400654068e-07 - MAX 1.5509633400654067e-06
    # print('- Mass: min',erosion_sim_params.m_init.min,'- MAX',erosion_sim_params.m_init.max)
    print(f"Mass [kg] & {'{:.4g}'.format(erosion_sim_params.m_init.min)} & {'{:.4g}'.format(erosion_sim_params.m_init.max)} \\\\")

    print('\\hline')
    # - rho : min 100 - MAX 1000
    # print('- rho : min',erosion_sim_params.rho.min,'- MAX',erosion_sim_params.rho.max)
    print(f"Rho [kg/m^3] & {'{:.4g}'.format(erosion_sim_params.rho.min)} & {'{:.4g}'.format(erosion_sim_params.rho.max)} \\\\")

    print('\\hline')
    # - sigma : min 8e-09 - MAX 3e-08
    # print('- sigma : min',erosion_sim_params.sigma.min,'- MAX',erosion_sim_params.sigma.max)
    print(f"sigma [s^2/km^2] & {'{:.4g}'.format(erosion_sim_params.sigma.min*1000000)} & {'{:.4g}'.format(erosion_sim_params.sigma.max*1000000)} \\\\")

    print('\\hline')
    # - erosion_height_start : min 107622.04437691614 - MAX 117622.04437691614
    # print('- erosion_height_start : min',erosion_sim_params.erosion_height_start.min,'- MAX',erosion_sim_params.erosion_height_start.max)
    print(f"Eros.height [km] & {'{:.4g}'.format(erosion_sim_params.erosion_height_start.min/1000)} & {'{:.4g}'.format(erosion_sim_params.erosion_height_start.max/1000)} \\\\")

    print('\\hline')
    # - erosion_coeff : min 0.0 - MAX 1e-06
    # print('- erosion_coeff : min',erosion_sim_params.erosion_coeff.min,'- MAX',erosion_sim_params.erosion_coeff.max)
    print(f"Eros.coeff. [s^2/km^2] & {'{:.4g}'.format(erosion_sim_params.erosion_coeff.min*1000000)} & {'{:.4g}'.format(erosion_sim_params.erosion_coeff.max*1000000)} \\\\")

    print('\\hline')
    # - erosion_mass_index : min 1.5 - MAX 2.5
    # print('- erosion_mass_index : min',erosion_sim_params.erosion_mass_index.min,'- MAX',erosion_sim_params.erosion_mass_index.max)
    print(f"Eros.mass index [-] & {'{:.4g}'.format(erosion_sim_params.erosion_mass_index.min)} & {'{:.4g}'.format(erosion_sim_params.erosion_mass_index.max)} \\\\")

    print('\\hline')
    # - erosion_mass_min : min 5e-12 - MAX 1e-10
    # print('- erosion_mass_min : min',erosion_sim_params.erosion_mass_min.min,'- MAX',erosion_sim_params.erosion_mass_min.max)
    print(f"Eros.mass min [kg] & {'{:.4g}'.format(erosion_sim_params.erosion_mass_min.min)} & {'{:.4g}'.format(erosion_sim_params.erosion_mass_min.max)} \\\\")

    print('\\hline')
    # - erosion_mass_max : min 1e-10 - MAX 5e-08
    # print('- erosion_mass_max : min',erosion_sim_params.erosion_mass_max.min,'- MAX',erosion_sim_params.erosion_mass_max.max)
    print(f"Eros.mass max [kg] & {'{:.4g}'.format(erosion_sim_params.erosion_mass_max.min)} & {'{:.4g}'.format(erosion_sim_params.erosion_mass_max.max)} \\\\")

    print('\\hline')


    # Close the Logger to ensure everything is written to the file STOP COPY in TXT file
    sys.stdout.close()

    # Reset sys.stdout to its original value if needed
    sys.stdout = sys.__stdout__

    input_list = [(output_folder, copy.deepcopy(erosion_sim_params), np.random.randint(0, 2**31 - 1), MIN_FRAMES_VISIBLE) for _ in range(numb_sim)]
    with Pool(cml_args.cores) as pool:
        results_list = pool.map(safe_generate_erosion_sim, input_list)

    count_none = sum(res is None for res in results_list)
    saveProcessedList(output_folder, results_list, erosion_sim_params.__class__.__name__, MIN_FRAMES_VISIBLE)

    print('Resulted simulations:', numb_sim - count_none)
    print('Failed simulations:', count_none)
    print('Saved', numb_sim - count_none, 'simulations in', output_folder)

    #########################
    # # Generate simulations using multiprocessing
    # input_list = [[output_folder, copy.deepcopy(erosion_sim_params), \
    #     np.random.randint(0, 2**31 - 1),MIN_FRAMES_VISIBLE] for _ in range(numb_sim)]
    # results_list = domainParallelizer(input_list, generateErosionSim, cores=cml_args.cores)

    # # print(results_list)

    # # count how many None are in the results_list
    # count_none=0
    # for res in results_list:
    #     if res is None:
    #         count_none+=1
    #         continue
        
    # saveProcessedList(output_folder, results_list, erosion_sim_params.__class__.__name__, \
    # MIN_FRAMES_VISIBLE)
    
    # print('Resulted simulations:', numb_sim-count_none)
    # print('Failed siulations', len(results_list)/100*count_none,'%')
    # print('Saved',numb_sim-count_none,'simulations in',output_folder)
    #########################

    # plot the pickle files data that are not none in the results_list
    # do not plot more than 10 curves
    if plot_case:

        fig, ax = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
        # flat the ax
        ax = ax.flatten()

        jj_plots_curve=0
        for res in results_list:
            if res is not None:
                if jj_plots_curve>100:
                    # stop if too many curves are plotted
                    break
                
                if res[0] is not None:
                    # change res[0] extension to .json
                    res[0] = res[0].replace('.pickle', '.json')
                    print(res[0]) 
                    # get the first value of res
                    gensim_data_sim = read_GenerateSimulations_output(res[0])

                    plot_side_by_side(gensim_data_sim, fig, ax, 'b-')
                    jj_plots_curve += 1
                
        plot_side_by_side(gensim_data,fig, ax,'go','Obsevation')

        return fig, ax

    

#### Plot #############################################################################


def check_axis_inversion(ax):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    is_x_inverted = x_max < x_min
    is_y_inverted = y_max < y_min
    return is_x_inverted, is_y_inverted


def plot_side_by_side(data1, fig='', ax='', colorline1='.', label1='', residuals_mag='', residuals_vel='', residual_time_pos='', residual_height_pos='', fit_funct='', mag_noise='', vel_noise='', label_fit=''):

    # check if data1 is None
    if data1 is None:
        print("Warning: data1 is None. Skipping plot.")
        return
    
    # check if it is in km/s or in m/s
    obs1 = copy.deepcopy(data1)
    if 'velocities' not in obs1 or 'height' not in obs1:
        print("Warning: Required keys missing in obs1. Skipping plot.")
        return

    # check if it is in km/s or in m/s
    obs1= copy.deepcopy(data1)
    if np.mean(obs1['velocities'])>1000:
        # convert to km/s
        obs1['velocities'] = np.array(obs1['velocities'])/1000
        obs1['height'] = np.array(obs1['height'])/1000


    # Plot the simulation results
    if residuals_mag != '' and residuals_vel != '' and residual_time_pos!='' and residual_height_pos!='':

        if fig=='' and ax=='':
            fig, ax = plt.subplots(2, 3, figsize=(14, 6),gridspec_kw={'height_ratios': [ 3, 1],'width_ratios': [ 3, 0.5, 3]}) #  figsize=(10, 5), dpi=300 0.5, 3, 3, 0.5
            # flat the ax
            ax = ax.flatten()
            return fig, ax
        
        if fit_funct!='' and mag_noise!='' and vel_noise!='':
            obs_time_err=np.array(fit_funct['time'])
            abs_mag_sim_err=np.array(fit_funct['absolute_magnitudes'])
            height_km_err=np.array(fit_funct['height'])
            vel_kms_err=np.array(fit_funct['velocities'])
            # from list to array
            if np.mean(fit_funct['height'])>1000:
                # convert to km/s
                height_km_err=np.array(fit_funct['height'])/1000
                vel_kms_err=np.array(fit_funct['velocities'])/1000

            # plot noisy area around vel_kms for vel_noise for the fix height_km
            ax[0].fill_betweenx(height_km_err, abs_mag_sim_err-mag_noise, abs_mag_sim_err+mag_noise, color='lightgray', alpha=0.5)

            # plot noisy area around vel_kms for vel_noise for the fix height_km
            ax[2].fill_between(obs_time_err, vel_kms_err-vel_noise, vel_kms_err+vel_noise, color='lightgray', alpha=0.5, label=label_fit)

            # plot noisy area around vel_kms for vel_noise for the fix height_km
            ax[1].fill_betweenx(height_km_err, -mag_noise, mag_noise, color='lightgray', alpha=0.5)

            # plot noisy area around vel_kms for vel_noise for the fix height_km
            ax[5].fill_between(obs_time_err, -vel_noise, vel_noise, color='lightgray', alpha=0.5)

        ax[0].plot(obs1['absolute_magnitudes'],obs1['height'], colorline1)
        ax[0].set_xlabel('Absolute Magnitude [-]')
        ax[0].set_ylabel('Height [km]')
        # grid on on both subplot with -- as linestyle and light gray color
        ax[0].grid(True)
        ax[0].grid(linestyle='--',color='lightgray')

        # flip the y-axis
        is_x_inverted, _ =check_axis_inversion(ax[0])
        if is_x_inverted==False:
            ax[0].invert_xaxis()

        # Get the color of the last plotted line in graph 0
        line_color = ax[0].get_lines()[-1].get_color()

        # if line_color == '#2ca02c':
        #     line_color='m'
        #     ax[0].plot(obs1['absolute_magnitudes'],obs1['height'], colorline1, color='m')

        # plot the residuals against time
        ax[1].plot(residuals_mag, residual_height_pos, '.', color=line_color)
        # ax[1].set_ylabel('Height [km]')
        ax[1].set_xlabel('Res.mag [-]')
        ax[1].tick_params(axis='x', rotation=45)

        # flip the y-axis
        is_x_inverted, _ =check_axis_inversion(ax[1])
        if is_x_inverted==False:
            ax[1].invert_xaxis()

        # ax[1].title(f'Lag Residuals')
        # ax[1].legend()
        is_x_inverted, _ =check_axis_inversion(ax[1])
        if is_x_inverted==False:
            ax[1].invert_xaxis()
        ax[1].grid(True)
        ax[1].grid(linestyle='--',color='lightgray')
        ax[1].set_ylim(ax[0].get_ylim())

        if label1!='':
            ax[2].plot(obs1['time'], obs1['velocities'], colorline1, color=line_color, label=label1)
        else:
            ax[2].plot(obs1['time'], obs1['velocities'], colorline1, color=line_color)
        # show the legend
        if label1 != '':
            ax[2].legend()

        ax[2].set_xlabel('Time [s]')
        ax[2].set_ylabel('Velocity [km/s]')
        ax[2].grid(True)
        ax[2].grid(linestyle='--',color='lightgray')

        # delete the plot in the middle
        ax[3].axis('off')
        
        # # put as the super title the name
        # plt.suptitle(name)
        ax[4].axis('off')

        # plot the residuals against time
        ax[5].plot(residual_time_pos, residuals_vel, '.', color=line_color)
        ax[5].set_ylabel('Res.vel [km/s]')
        ax[5].grid(True)
        ax[5].grid(linestyle='--',color='lightgray')
        # use the same limits of ax[3]
        ax[5].set_xlim(ax[2].get_xlim())


        # # plot the distribution of the residuals along the y axis
        # ax[5].hist(residuals_mag, bins=20, color=line_color, alpha=0.5)
        # ax[5].set_ylabel('N.data')
        # ax[5].set_xlabel('Res.mag [-]')
        # is_x_inverted, _ =check_axis_inversion(ax[5])
        # if is_x_inverted==False:
        #     ax[5].invert_xaxis()
        # ax[5].grid(True)
        # ax[5].grid(linestyle='--',color='lightgray')

        # # plot the residuals against time
        # ax[6].plot(residual_time_pos, residuals_vel, '.', color=line_color)
        # # ax[6].set_xlabel('Time [s]')
        # ax[6].set_xticks([])
        # ax[6].set_ylabel('Res.vel [km/s]')
        # ax[6].invert_yaxis()
        # # ax[3].title(f'Absolute Magnitude Residuals')
        # # ax[3].legend()
        # ax[6].grid(True)
        # ax[6].grid(linestyle='--',color='lightgray')

        # # plot the distribution of the residuals along the y axis
        # ax[7].hist(residuals_vel, bins=20, color=line_color, alpha=0.5, orientation='horizontal')
        # ax[7].set_xlabel('N.data')
        # # invert the y axis
        # ax[7].invert_yaxis()
        # ax[7].set_ylabel('Res.vel [km/s]')
        # # delete the the the line at the top ad the right
        # ax[7].spines['top'].set_visible(False)
        # ax[7].spines['right'].set_visible(False)
        # # do not show the y ticks
        # # ax[7].set_yticks([])
        # # # show the zero line
        # # ax[7].axhline(0, color='k', linewidth=0.5)
        # # grid on
        # ax[7].grid(True) 
        # # grid on
        # ax[7].grid(linestyle='--',color='lightgray')



    else :
        if fig=='' and ax=='':
            fig, ax = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
            # flat the ax
            ax = ax.flatten()
            return fig, ax
        
        # plot the magnitude curve with height
        ax[0].plot(obs1['absolute_magnitudes'],obs1['height'], colorline1)

        ax[0].set_xlabel('Absolute Magnitude [-]')
        ax[0].set_ylabel('Height [km]')
        # check if the axis is inverted
        is_x_inverted, _ =check_axis_inversion(ax[0])
        if is_x_inverted==False:
            ax[0].invert_xaxis()
        # grid on
        ax[0].grid(True)

        # plot 
        if label1 == '':
            ax[1].plot(obs1['time'], obs1['velocities'], colorline1)
        else:
            ax[1].plot(obs1['time'], obs1['velocities'], colorline1, label=label1)

        # show the legend
        if label1 != '':
            ax[1].legend()

        ax[1].set_xlabel('Time [s]')
        ax[1].set_ylabel('Velocity [km/s]')
        ax[1].grid(True)

        # grid on on both subplot with -- as linestyle and light gray color
        ax[1].grid(linestyle='--',color='lightgray')
        # grid on
        ax[0].grid(linestyle='--',color='lightgray')

    plt.tight_layout()

    

#### Reader #############################################################################


def read_GenerateSimulations_output_to_PCA(file_path, name=''):
    if name!='':   
        print(name) 
    gensim_data = read_GenerateSimulations_output(file_path)
    if gensim_data is None:
        return None
    else:
        pd_datfram_PCA = array_to_pd_dataframe_PCA(gensim_data)
        return pd_datfram_PCA


def read_GenerateSimulations_output(file_path, real_event=''):

    f = open(file_path,"r")
    data = json.loads(f.read())

    # show processed event
    print(file_path)

    if data['ht_sampled']!= None: 

        vel_sim=data['simulation_results']['leading_frag_vel_arr'][:-1]#['brightest_vel_arr']#['leading_frag_vel_arr']#['main_vel_arr']
        ht_sim=data['simulation_results']['leading_frag_height_arr'][:-1]#['brightest_height_arr']['leading_frag_height_arr']['main_height_arr']
        time_sim=data['simulation_results']['time_arr'][:-1]#['main_time_arr']
        abs_mag_sim=data['simulation_results']['abs_magnitude'][:-1]
        len_sim=data['simulation_results']['brightest_length_arr'][:-1]#['brightest_length_arr']
        Dynamic_pressure= data['simulation_results']['leading_frag_dyn_press_arr'][:-1]
        
        # ht_obs=data['ht_sampled']
        # try:
        #     index_ht_sim=next(x for x, val in enumerate(ht_sim) if val <= ht_obs[0])
        # except StopIteration:
        #     # index_ht_sim = None
        #     print('The first element of the observation is not in the simulation')
        #     return None

        # try:
        #     index_ht_sim_end=next(x for x, val in enumerate(ht_sim) if val <= ht_obs[-1])
        # except StopIteration:
        #     # index_ht_sim_end = None
        #     print('The last element of the observation is not in the simulation')
        #     return None
        
        if real_event!= '':
            mag_obs=real_event['absolute_magnitudes']
        else:
            mag_obs=data['mag_sampled']

        print('read_GenerateSimulations_output mag',mag_obs[0],'-',mag_obs[-1])

        try:
            # find the index of the first element of abs_mag_sim that is smaller than the first element of mag_obs
            index_abs_mag_sim_start = next(i for i, val in enumerate(abs_mag_sim) if val <= mag_obs[0])
            index_abs_mag_sim_start = index_abs_mag_sim_start + np.random.randint(2)
        except StopIteration:
            print("The first observation height is not within the simulation data range.")
            return None
        try:   
            index_abs_mag_sim_end = next(i for i, val in enumerate(abs_mag_sim[::-1]) if val <= mag_obs[-1])
            index_abs_mag_sim_end = len(abs_mag_sim) - index_abs_mag_sim_end - 1        
        except StopIteration:
            print("The first observation height is not within the simulation data range.")
            return None
        
        # print('mag',index_abs_mag_sim_start,'-',index_abs_mag_sim_end,'\nheight',index_ht_sim,'-',index_ht_sim_end)
            
        abs_mag_sim = abs_mag_sim[index_abs_mag_sim_start:index_abs_mag_sim_end]
        vel_sim = vel_sim[index_abs_mag_sim_start:index_abs_mag_sim_end]
        time_sim = time_sim[index_abs_mag_sim_start:index_abs_mag_sim_end]
        ht_sim = ht_sim[index_abs_mag_sim_start:index_abs_mag_sim_end]
        len_sim = len_sim[index_abs_mag_sim_start:index_abs_mag_sim_end]
        Dynamic_pressure = Dynamic_pressure[index_abs_mag_sim_start:index_abs_mag_sim_end]



        # abs_mag_sim=abs_mag_sim[index_ht_sim:index_ht_sim_end]
        # vel_sim=vel_sim[index_ht_sim:index_ht_sim_end]
        # time_sim=time_sim[index_ht_sim:index_ht_sim_end]
        # ht_sim=ht_sim[index_ht_sim:index_ht_sim_end]
        # len_sim=len_sim[index_ht_sim:index_ht_sim_end]

        # closest_indices = find_closest_index(ht_sim, ht_obs)

        # Dynamic_pressure= data['simulation_results']['leading_frag_dyn_press_arr']
        # Dynamic_pressure= Dynamic_pressure[index_ht_sim:index_ht_sim_end]
        # Dynamic_pressure=[Dynamic_pressure[jj_index_cut] for jj_index_cut in closest_indices]

        # abs_mag_sim=[abs_mag_sim[jj_index_cut] for jj_index_cut in closest_indices]
        # vel_sim=[vel_sim[jj_index_cut] for jj_index_cut in closest_indices]
        # time_sim=[time_sim[jj_index_cut] for jj_index_cut in closest_indices]
        # ht_sim=[ht_sim[jj_index_cut] for jj_index_cut in closest_indices]
        # len_sim=[len_sim[jj_index_cut] for jj_index_cut in closest_indices]

        # divide the vel_sim by 1000 considering is a list
        time_sim = [i-time_sim[0] for i in time_sim]
        # vel_sim = [i/1000 for i in vel_sim]
        len_sim = [i-len_sim[0] for i in len_sim]
        # ht_sim = [i/1000 for i in ht_sim]

        # Load the constants
        const, _ = loadConstants(file_path)
        const.dens_co = np.array(const.dens_co)

        # Compute the erosion energies
        erosion_energy_per_unit_cross_section, erosion_energy_per_unit_mass = wmpl.MetSim.MetSimErosion.energyReceivedBeforeErosion(const)

        gensim_data = {
        'name': file_path,
        'type': 'Simulation',
        'v_init': vel_sim[0], # m/s
        'velocities': vel_sim, # m/s
        'height': ht_sim, # m
        'absolute_magnitudes': abs_mag_sim,
        'lag': len_sim-(vel_sim[0]*np.array(time_sim)+len_sim[0]), # m
        'length': len_sim, # m
        'time': time_sim, # s
        'v_avg': np.mean(vel_sim), # m/s
        'v_init_180km': data['params']['v_init']['val'], # m/s
        'Dynamic_pressure_peak_abs_mag': Dynamic_pressure[np.argmin(abs_mag_sim)],
        'zenith_angle': data['params']['zenith_angle']['val']*180/np.pi,
        'mass': data['params']['m_init']['val'],
        'rho': data['params']['rho']['val'],
        'sigma': data['params']['sigma']['val'],
        'erosion_height_start': data['params']['erosion_height_start']['val']/1000,
        'erosion_coeff': data['params']['erosion_coeff']['val'],
        'erosion_mass_index': data['params']['erosion_mass_index']['val'],
        'erosion_mass_min': data['params']['erosion_mass_min']['val'],
        'erosion_mass_max': data['params']['erosion_mass_max']['val'],
        'erosion_range': np.log10(data['params']['erosion_mass_max']['val']) - np.log10(data['params']['erosion_mass_min']['val']),
        'erosion_energy_per_unit_cross_section': erosion_energy_per_unit_cross_section,
        'erosion_energy_per_unit_mass': erosion_energy_per_unit_mass
        }

        return gensim_data
    
    else:
        return None


def Old_GenSym_json_get_vel_lag(data):

    ht_sim=data['simulation_results']['leading_frag_height_arr'][:-1]#['brightest_height_arr']['leading_frag_height_arr']['main_height_arr']
    ht_obs=data['ht_sampled']
    time_sampled = np.array(data['time_sampled'])
    len_sampled = np.array(data['len_sampled'])

    closest_indices = find_closest_index(ht_sim, ht_obs)

    vel_sim=data['simulation_results']['leading_frag_vel_arr'][:-1]#['brightest_vel_arr']#['leading_frag_vel_arr']#['main_vel_arr']
    vel_sim=[vel_sim[jj_index_cut] for jj_index_cut in closest_indices]

    # get the new velocity with noise
    for vel_ii in range(1,len(time_sampled)):
        if time_sampled[vel_ii]-time_sampled[vel_ii-1]<1.0/FPS:
        # if time_sampled[vel_ii] % 0.03125 < 0.000000001:
            if vel_ii+1<len(len_sampled):
                vel_sim[vel_ii+1]=(len_sampled[vel_ii+1]-len_sampled[vel_ii-1])/(time_sampled[vel_ii+1]-time_sampled[vel_ii-1])
        else:
            vel_sim[vel_ii]=(len_sampled[vel_ii]-len_sampled[vel_ii-1])/(time_sampled[vel_ii]-time_sampled[vel_ii-1])

    data['vel_sampled']=vel_sim
    
    lag_sim=len_sampled-(vel_sim[0]*time_sampled+len_sampled[0])

    data['lag_sampled']=lag_sim

    return data


def read_with_noise_GenerateSimulations_output(file_path):

    f = open(file_path,"r")
    data = json.loads(f.read())

    if data['ht_sampled']!= None: 

        ht_sim=data['simulation_results']['leading_frag_height_arr']#['brightest_height_arr']['leading_frag_height_arr']['main_height_arr']

        ht_obs=data['ht_sampled']

        closest_indices = find_closest_index(ht_sim, ht_obs)

        Dynamic_pressure= data['simulation_results']['leading_frag_dyn_press_arr']
        Dynamic_pressure=[Dynamic_pressure[jj_index_cut] for jj_index_cut in closest_indices]

        # Load the constants
        const, _ = loadConstants(file_path)
        const.dens_co = np.array(const.dens_co)

        # Compute the erosion energies
        erosion_energy_per_unit_cross_section, erosion_energy_per_unit_mass = wmpl.MetSim.MetSimErosion.energyReceivedBeforeErosion(const)

        # if is not present the vel_sampled in the data
        if 'vel_sampled' not in data:
            data = Old_GenSym_json_get_vel_lag(data)

        gensim_data = {
        'name': file_path,
        'type': 'Observation_sim',
        'v_init_180km': data['params']['v_init']['val'], # m/s
        'v_init': data['vel_sampled'][0], # m/s
        'velocities': data['vel_sampled'], # m/s
        'height': data['ht_sampled'], # m
        'absolute_magnitudes': data['mag_sampled'],
        'lag': data['lag_sampled'], # m
        'length': data['len_sampled'], # m
        'time': data['time_sampled'], # s
        'v_avg': np.mean(data['vel_sampled']), # m/s
        'Dynamic_pressure_peak_abs_mag': Dynamic_pressure[np.argmin(data['mag_sampled'])],
        'zenith_angle': data['params']['zenith_angle']['val']*180/np.pi,
        'mass': data['params']['m_init']['val'],
        'rho': data['params']['rho']['val'],
        'sigma': data['params']['sigma']['val'],
        'erosion_height_start': data['params']['erosion_height_start']['val']/1000,
        'erosion_coeff': data['params']['erosion_coeff']['val'],
        'erosion_mass_index': data['params']['erosion_mass_index']['val'],
        'erosion_mass_min': data['params']['erosion_mass_min']['val'],
        'erosion_mass_max': data['params']['erosion_mass_max']['val'],
        'erosion_range': np.log10(data['params']['erosion_mass_max']['val']) - np.log10(data['params']['erosion_mass_min']['val']),
        'erosion_energy_per_unit_cross_section': erosion_energy_per_unit_cross_section,
        'erosion_energy_per_unit_mass': erosion_energy_per_unit_mass
        }

        return gensim_data
    
    else:
        return None


def read_RunSim_output(simulation_MetSim_object, real_event, MetSim_phys_file_path):
    '''
    dict_keys(['const', 'frag_main', 'time_arr', 'luminosity_arr', 'luminosity_main_arr', 'luminosity_eroded_arr', 
    'electron_density_total_arr', 'tau_total_arr', 'tau_main_arr', 'tau_eroded_arr', 'brightest_height_arr', 
    'brightest_length_arr', 'brightest_vel_arr', 'leading_frag_height_arr', 'leading_frag_length_arr', 
    'leading_frag_vel_arr', 'leading_frag_dyn_press_arr', 'mass_total_active_arr', 'main_mass_arr', 
    'main_height_arr', 'main_length_arr', 'main_vel_arr', 'main_dyn_press_arr', 'abs_magnitude', 
    'abs_magnitude_main', 'abs_magnitude_eroded', 'wake_results', 'wake_max_lum'])

    in const

    dict_keys(['dt', 'total_time', 'n_active', 'm_kill', 'v_kill', 'h_kill', 'len_kill', 'h_init', 'P_0m', 
    'dens_co', 'r_earth', 'total_fragments', 'wake_psf', 'wake_extension', 'rho', 'm_init', 'v_init', 
    'shape_factor', 'sigma', 'zenith_angle', 'gamma', 'rho_grain', 'lum_eff_type', 'lum_eff', 'mu', 
    'erosion_on', 'erosion_bins_per_10mass', 'erosion_height_start', 'erosion_coeff', 'erosion_height_change', 
    'erosion_coeff_change', 'erosion_rho_change', 'erosion_sigma_change', 'erosion_mass_index', 'erosion_mass_min', 
    'erosion_mass_max', 'disruption_on', 'compressive_strength', 'disruption_height', 'disruption_erosion_coeff', 
    'disruption_mass_index', 'disruption_mass_min_ratio', 'disruption_mass_max_ratio', 'disruption_mass_grain_ratio', 
    'fragmentation_on', 'fragmentation_show_individual_lcs', 'fragmentation_entries', 'fragmentation_file_name', 
    'electron_density_meas_ht', 'electron_density_meas_q', 'erosion_beg_vel', 'erosion_beg_mass', 'erosion_beg_dyn_press', 
    'mass_at_erosion_change', 'energy_per_cs_before_erosion', 'energy_per_mass_before_erosion', 'main_mass_exhaustion_ht', 'main_bottom_ht'])
    '''

    vel_sim=simulation_MetSim_object.leading_frag_vel_arr #main_vel_arr
    ht_sim=simulation_MetSim_object.leading_frag_height_arr #main_height_arr
    time_sim=simulation_MetSim_object.time_arr
    abs_mag_sim=simulation_MetSim_object.abs_magnitude
    len_sim=simulation_MetSim_object.leading_frag_length_arr #main_length_arr
    Dynamic_pressure=simulation_MetSim_object.leading_frag_dyn_press_arr # main_dyn_press_arr
    
    mag_obs=real_event['absolute_magnitudes']

    print('read_RunSim_output mag',mag_obs[0],'-',mag_obs[-1])

    try:
        # find the index of the first element of abs_mag_sim that is smaller than the first element of mag_obs
        index_abs_mag_sim_start = next(i for i, val in enumerate(abs_mag_sim) if val < mag_obs[0])
        index_abs_mag_sim_start = index_abs_mag_sim_start # + np.random.randint(2)
    except StopIteration:
        print("The first observation height is not within the simulation data range.")
        return None
    try:   
        index_abs_mag_sim_end = next(i for i, val in enumerate(abs_mag_sim[::-1]) if val < mag_obs[-1])
        index_abs_mag_sim_end = len(abs_mag_sim) - index_abs_mag_sim_end # - 1           
    except StopIteration:
        print("The first observation height is not within the simulation data range.")
        return None
        
    abs_mag_sim = abs_mag_sim[index_abs_mag_sim_start:index_abs_mag_sim_end]
    vel_sim = vel_sim[index_abs_mag_sim_start:index_abs_mag_sim_end]
    time_sim = time_sim[index_abs_mag_sim_start:index_abs_mag_sim_end]
    ht_sim = ht_sim[index_abs_mag_sim_start:index_abs_mag_sim_end]
    len_sim = len_sim[index_abs_mag_sim_start:index_abs_mag_sim_end]
    Dynamic_pressure = Dynamic_pressure[index_abs_mag_sim_start:index_abs_mag_sim_end]

    # ht_obs=real_event['height']
    # try:
    #     # find the index of the first element of the simulation that is equal to the first element of the observation
    #     index_ht_sim = next(x for x, val in enumerate(ht_sim) if val <= ht_obs[0])
    # except StopIteration:
    #     print("The first observation height is not within the simulation data range.")
    #     index_ht_sim = 0

    # try:
    #     # find the index of the last element of the simulation that is equal to the last element of the observation
    #     index_ht_sim_end = next(x for x, val in enumerate(ht_sim) if val <= ht_obs[-1])
    # except StopIteration:
    #     print("The last observation height is not within the simulation data range.")
    #     index_ht_sim_end = len(ht_sim) - 2 # at -1 there is Nan in some sim value


    # abs_mag_sim=abs_mag_sim[index_ht_sim:index_ht_sim_end]
    # vel_sim=vel_sim[index_ht_sim:index_ht_sim_end]
    # time_sim=time_sim[index_ht_sim:index_ht_sim_end]
    # ht_sim=ht_sim[index_ht_sim:index_ht_sim_end]
    # len_sim=len_sim[index_ht_sim:index_ht_sim_end]
    # Dynamic_pressure= Dynamic_pressure[index_ht_sim:index_ht_sim_end]

    # closest_indices = find_closest_index(ht_sim, ht_obs)

    # abs_mag_sim=[abs_mag_sim[jj_index_cut] for jj_index_cut in closest_indices]
    # vel_sim=[vel_sim[jj_index_cut] for jj_index_cut in closest_indices]
    # time_sim=[time_sim[jj_index_cut] for jj_index_cut in closest_indices]
    # ht_sim=[ht_sim[jj_index_cut] for jj_index_cut in closest_indices]
    # len_sim=[len_sim[jj_index_cut] for jj_index_cut in closest_indices]
    # Dynamic_pressure=[Dynamic_pressure[jj_index_cut] for jj_index_cut in closest_indices]

    # divide the vel_sim by 1000 considering is a list
    time_sim = [i-time_sim[0] for i in time_sim]
    # vel_sim = [i/1000 for i in vel_sim]
    len_sim = [i-len_sim[0] for i in len_sim]
    # ht_sim = [i/1000 for i in ht_sim]

    output_phys = read_MetSim_phyProp_output(MetSim_phys_file_path)

    gensim_data = {
        'name': MetSim_phys_file_path,
        'type': 'MetSim',
        'v_init': vel_sim[0], # m/s
        'velocities': vel_sim, # m/s
        'v_init_180km': simulation_MetSim_object.const.v_init, # m/s
        'height': ht_sim, # m
        'absolute_magnitudes': abs_mag_sim,
        'lag': len_sim-(vel_sim[0]*np.array(time_sim)+len_sim[0]), # m
        'length': len_sim, # m
        'time': time_sim, # s
        'v_avg': np.mean(vel_sim), # m/s
        'Dynamic_pressure_peak_abs_mag': Dynamic_pressure[np.argmin(abs_mag_sim)],
        'zenith_angle': simulation_MetSim_object.const.zenith_angle*180/np.pi,
        'mass': output_phys[0],
        'rho': output_phys[1],
        'sigma': output_phys[2],
        'erosion_height_start': output_phys[3],
        'erosion_coeff': output_phys[4],
        'erosion_mass_index': output_phys[5],
        'erosion_mass_min': output_phys[6],
        'erosion_mass_max': output_phys[7],
        'erosion_range': output_phys[8],
        'erosion_energy_per_unit_cross_section': output_phys[9],
        'erosion_energy_per_unit_mass': output_phys[10]
        }

    return gensim_data


def read_pickle_reduction_file(file_path, MetSim_phys_file_path='', obs_sep=False):


    with open(file_path, 'rb') as f:
        traj = pickle.load(f, encoding='latin1')

    v_avg = traj.v_avg
    jd_dat=traj.jdt_ref
    obs_data = []
    for obs in traj.observations:
        if obs.station_id == "01G" or obs.station_id == "02G" or obs.station_id == "01F" or obs.station_id == "02F" or obs.station_id == "1G" or obs.station_id == "2G" or obs.station_id == "1F" or obs.station_id == "2F":
            obs_dict = {
                'v_init': obs.v_init, # m/s
                'velocities': np.array(obs.velocities), # m/s
                'height': np.array(obs.model_ht), # m
                'absolute_magnitudes': np.array(obs.absolute_magnitudes),
                'lag': np.array(obs.lag), # m
                'length': np.array(obs.length), # m
                'time': np.array(obs.time_data), # s
                # 'station_id': obs.station_id
                'elev_data':  np.array(obs.elev_data)
            }
            obs_dict['velocities'][0] = obs_dict['v_init']
            obs_data.append(obs_dict)
                
            lat_dat=obs.lat
            lon_dat=obs.lon

        else:
            print(obs.station_id,'Station not in the list of stations')
            continue
    
    # Save distinct values for the two observations
    obs1, obs2 = obs_data[0], obs_data[1]
    
    # Combine obs1 and obs2
    combined_obs = {}
    for key in ['velocities', 'height', 'absolute_magnitudes', 'lag', 'length', 'time', 'elev_data']:
        combined_obs[key] = np.concatenate((obs1[key], obs2[key]))

    # Order the combined observations based on time
    sorted_indices = np.argsort(combined_obs['time'])
    for key in ['time', 'velocities', 'height', 'absolute_magnitudes', 'lag', 'length', 'elev_data']:
        combined_obs[key] = combined_obs[key][sorted_indices]

    # check if any value is below 10 absolute_magnitudes and print find values below 10 absolute_magnitudes
    if np.any(combined_obs['absolute_magnitudes'] > 14):
        print('Found values below 14 absolute magnitudes:', combined_obs['absolute_magnitudes'][combined_obs['absolute_magnitudes'] > 14])
    
    # delete any values above 10 absolute_magnitudes and delete the corresponding values in the other arrays
    combined_obs = {key: combined_obs[key][combined_obs['absolute_magnitudes'] < 14] for key in combined_obs.keys()}

    Dynamic_pressure_peak_abs_mag=(wmpl.Utils.Physics.dynamicPressure(lat_dat, lon_dat, combined_obs['height'][np.argmin(combined_obs['absolute_magnitudes'])], jd_dat, combined_obs['velocities'][np.argmin(combined_obs['absolute_magnitudes'])]))
    const=Constants()
    zenith_angle=zenithAngleAtSimulationBegin(const.h_init, traj.rbeg_ele, traj.orbit.zc, const.r_earth)

    if MetSim_phys_file_path != '':
        output_phys = read_MetSim_phyProp_output(MetSim_phys_file_path)
        type_sim='MetSim'
        
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

        type_sim='Observation'

        # put all the varible in a array mass, rho, sigma, erosion_height_start, erosion_coeff, erosion_mass_index, erosion_mass_min, erosion_mass_max, erosion_range, erosion_energy_per_unit_cross_section_arr, erosion_energy_per_unit_mass_arr
        output_phys = [mass, rho, sigma, erosion_height_start, erosion_coeff, erosion_mass_index, erosion_mass_min, erosion_mass_max, erosion_range, erosion_energy_per_unit_cross_section_arr, erosion_energy_per_unit_mass_arr]

    # delete the elev_data from the combined_obs
    del combined_obs['elev_data']

    # add to combined_obs the avg velocity and the peak dynamic pressure and all the physical parameters
    combined_obs['name'] = file_path    
    combined_obs['v_init'] = combined_obs['velocities'][0]
    combined_obs['v_init_180km'] = combined_obs['velocities'][0]+100
    combined_obs['type'] = type_sim
    combined_obs['v_avg'] = v_avg
    combined_obs['Dynamic_pressure_peak_abs_mag'] = Dynamic_pressure_peak_abs_mag
    combined_obs['zenith_angle'] = zenith_angle*180/np.pi
    combined_obs['mass'] = output_phys[0]
    combined_obs['rho'] = output_phys[1]
    combined_obs['sigma'] = output_phys[2]
    combined_obs['erosion_height_start'] = output_phys[3]
    combined_obs['erosion_coeff'] = output_phys[4]
    combined_obs['erosion_mass_index'] = output_phys[5]
    combined_obs['erosion_mass_min'] = output_phys[6]
    combined_obs['erosion_mass_max'] = output_phys[7]
    combined_obs['erosion_range'] = output_phys[8]
    combined_obs['erosion_energy_per_unit_cross_section'] = output_phys[9]
    combined_obs['erosion_energy_per_unit_mass'] = output_phys[10]

    if obs_sep:
        return combined_obs, obs1, obs2
    else:
        return combined_obs


def read_MetSim_phyProp_output(MetSim_phys_file_path):

    # check if in os.path.join(root, name_file) present and then open the .json file with the same name as the pickle file with in stead of _trajectory.pickle it has _sim_fit_latest.json
    if os.path.isfile(MetSim_phys_file_path):
        with open(MetSim_phys_file_path,'r') as json_file: # 20210813_061453_sim_fit.json
            print('Loading Physical Characteristics MetSim file:', MetSim_phys_file_path)
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

            cost_path = os.path.join(MetSim_phys_file_path)

            # Load the constants
            const, _ = loadConstants(cost_path)
            const.dens_co = np.array(const.dens_co)

            # Compute the erosion energies
            erosion_energy_per_unit_cross_section, erosion_energy_per_unit_mass = wmpl.MetSim.MetSimErosion.energyReceivedBeforeErosion(const)
            erosion_energy_per_unit_cross_section_arr=(erosion_energy_per_unit_cross_section)
            erosion_energy_per_unit_mass_arr=(erosion_energy_per_unit_mass)

    else:
        print('No json file:',MetSim_phys_file_path)

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

    # put all the varible in a array mass, rho, sigma, erosion_height_start, erosion_coeff, erosion_mass_index, erosion_mass_min, erosion_mass_max, erosion_range, erosion_energy_per_unit_cross_section_arr, erosion_energy_per_unit_mass_arr
    output_phys = [mass, rho, sigma, erosion_height_start, erosion_coeff, erosion_mass_index, erosion_mass_min, erosion_mass_max, erosion_range, erosion_energy_per_unit_cross_section_arr, erosion_energy_per_unit_mass_arr]
    
    return output_phys



def array_to_pd_dataframe_PCA(data):

    if data is None:
        # Handle the None case, maybe log an error or return an empty DataFrame
        print(f"Warning: 'data' is None for source returning an empty DataFrame.")
        return pd.DataFrame()  # or any other appropriate action
    # do a copy of data_array
    data_array = data.copy()

    # compute the linear regression
    data_array['v_init'] = data_array['v_init']/1000
    data_array['v_avg'] = data_array['v_avg']/1000
    data_array['velocities'] = [i/1000 for i in data_array['velocities']] # convert m/s to km/s
    data_array['height'] = [i/1000 for i in data_array['height']]
    data_array['lag']=[i/1000 for i in data_array['lag']]
    v0=data_array['v_init']

    # from 'time_sampled' extract the last element and save it in a list
    duration = data_array['time'][-1]
    begin_height = data_array['height'][0]
    end_height = data_array['height'][-1]
    peak_abs_mag = data_array['absolute_magnitudes'][np.argmin(data_array['absolute_magnitudes'])]
    F_param = (begin_height - (data_array['height'][np.argmin(data_array['absolute_magnitudes'])])) / (begin_height - end_height)
    peak_mag_height = data_array['height'][np.argmin(data_array['absolute_magnitudes'])]
    beg_abs_mag	= data_array['absolute_magnitudes'][0]
    end_abs_mag	= data_array['absolute_magnitudes'][-1]
    trail_len = data_array['length'][-1]
    avg_lag = np.mean(data_array['lag'])


    kc_par = begin_height + (2.86 - 2*np.log(data_array['v_init']))/0.0612

    # fit a line to the throught the vel_sim and ht_sim
    a, b = np.polyfit(data_array['time'],data_array['velocities'], 1)
    acceleration_lin = a

    t0 = np.mean(data_array['time'])

    # initial guess of deceleration decel equal to linear fit of velocity
    p0 = [a, 0, 0, t0]

    opt_res = opt.minimize(lag_residual, p0, args=(np.array(data_array['time']), np.array(data_array['lag'])), method='Nelder-Mead')

    # sample the fit for the velocity and acceleration
    a_t0, b_t0, c_t0, t0 = opt_res.x

    # compute reference decelearation
    t_decel_ref = (t0 + np.max(data_array['time']))/2
    decel_t0 = cubic_acceleration(t_decel_ref, a_t0, b_t0, t0)[0]

    a_t0=-abs(a_t0)
    b_t0=-abs(b_t0)

    acceleration_parab_t0=a_t0*6 + b_t0*2

    a3, b3, c3 = np.polyfit(data_array['time'],data_array['velocities'], 2)
    acceleration_parab=a3*2 + b3

    # Assuming the jacchiaVel function is defined as:
    def jacchiaVel(t, a1, a2, v_init):
        return v_init - np.abs(a1) * np.abs(a2) * np.exp(np.abs(a2) * t)

    # Generating synthetic observed data for demonstration
    t_observed = np.array(data_array['time'])  # Observed times

    # Residuals function for optimization
    def residuals(params):
        a1, a2 = params
        predicted_velocity = jacchiaVel(t_observed, a1, a2, v0)
        return np.sum((data_array['velocities'] - predicted_velocity)**2)

    # Initial guess for a1 and a2
    initial_guess = [0.005,	10]

    # Apply minimize to the residuals
    result = minimize(residuals, initial_guess)

    # Results
    jac_a1, jac_a2 = abs(result.x)

    acc_jacchia = abs(jac_a1)*abs(jac_a2)**2

    try:
        # fit a line to the throught the obs_vel and ht_sim
        index_ht_peak = next(x for x, val in enumerate(data_array['height']) if val <= peak_mag_height)
    except StopIteration:
        # Handle the case where no height is less than or equal to peak_mag_height
        index_ht_peak = len(data_array['height']) // 2

    # Check if the arrays are non-empty before fitting the polynomial
    if len(data_array['height'][:index_ht_peak]) > 0 and len(data_array['absolute_magnitudes'][:index_ht_peak]) > 0:
        a3_Inabs, b3_Inabs, c3_Inabs = np.polyfit(data_array['height'][:index_ht_peak], data_array['absolute_magnitudes'][:index_ht_peak], 2)
    else:
        # Handle the case of empty input arrays
        a3_Inabs, b3_Inabs, c3_Inabs = 0, 0, 0

    # Check if the arrays are non-empty before fitting the polynomial
    if len(data_array['height'][index_ht_peak:]) > 0 and len(data_array['absolute_magnitudes'][index_ht_peak:]) > 0:
        a3_Outabs, b3_Outabs, c3_Outabs = np.polyfit(data_array['height'][index_ht_peak:], data_array['absolute_magnitudes'][index_ht_peak:], 2)
    else:
        # Handle the case of empty input arrays
        a3_Outabs, b3_Outabs, c3_Outabs = 0, 0, 0

    # # check if the ht_obs[:index_ht_peak] and abs_mag_obs[:index_ht_peak] are empty
    # a3_Inabs, b3_Inabs, c3_Inabs = np.polyfit(data_array['height'][:index_ht_peak], data_array['absolute_magnitudes'][:index_ht_peak], 2)

    # # check if the ht_obs[index_ht_peak:] and abs_mag_obs[index_ht_peak:] are empty
    # a3_Outabs, b3_Outabs, c3_Outabs = np.polyfit(data_array['height'][index_ht_peak:], data_array['absolute_magnitudes'][index_ht_peak:], 2)


    ######## SKEW KURT ################ 
    # create a new array with the same values as time_pickl
    index=[]
    # if the distance between two index is smalle than 0.05 delete the second one
    for i in range(len(data_array['time'])-1):
        if data_array['time'][i+1]-data_array['time'][i] < 0.01:
            # save the index as an array
            index.append(i+1)
    # delete the index from the list
    time_pickl = np.delete(data_array['time'], index)
    abs_mag_pickl = np.delete(data_array['time'], index)

    abs_mag_pickl = [0 if math.isnan(x) else x for x in abs_mag_pickl]

    # subrtract the max value of the mag to center it at the origin
    mag_sampled_norm = (-1)*(abs_mag_pickl - np.max(abs_mag_pickl))
    # check if there is any negative value and add the absolute value of the min value to all the values
    mag_sampled_norm = mag_sampled_norm + np.abs(np.min(mag_sampled_norm))
    # normalize the mag so that the sum is 1
    time_sampled_norm= time_pickl - np.mean(time_pickl)
    # mag_sampled_norm = mag_sampled_norm/np.sum(mag_sampled_norm)
    mag_sampled_norm = mag_sampled_norm/np.max(mag_sampled_norm)
    # substitute the nan values with zeros
    mag_sampled_norm = np.nan_to_num(mag_sampled_norm)

    # create an array with the number the ammount of same number equal to the value of the mag
    mag_sampled_distr = []
    mag_sampled_array=np.asarray(mag_sampled_norm*1000, dtype = 'int')
    for i in range(len(abs_mag_pickl)):
        # create an integer form the array mag_sampled_array[i] and round of the given value
        numbs=mag_sampled_array[i]
        # invcrease the array number by the mag_sampled_distr numbs 
        # array_nu=(np.ones(numbs+1)*i_pos).astype(int)
        array_nu=(np.ones(numbs+1)*time_sampled_norm[i])
        mag_sampled_distr=np.concatenate((mag_sampled_distr, array_nu))
    
    # # # plot the mag_sampled_distr as an histogram
    # plt.hist(mag_sampled_distr)
    # plt.show()

    # kurtosyness.append(kurtosis(mag_sampled_distr))
    # skewness.append(skew(mag_sampled_distr))
    kurtosyness=kurtosis(mag_sampled_distr)
    skewness=skew(mag_sampled_distr)

    ################################# 

    

    # Data to populate the dataframe
    data_picklefile_pd = {
        'solution_id': [data_array['name']],
        'type': [data_array['type']],
        'vel_init_norot': [data_array['v_init']],
        'vel_avg_norot': [data_array['v_avg']],
        'v_init_180km': [data_array['v_init_180km']],
        'duration': [duration],
        'peak_mag_height': [peak_mag_height],
        'begin_height': [begin_height],
        'end_height': [end_height],
        'peak_abs_mag': [peak_abs_mag],
        'beg_abs_mag': [beg_abs_mag],
        'end_abs_mag': [end_abs_mag],
        'F': [F_param],
        'trail_len': [trail_len],
        't0': [t0],
        'deceleration_lin': [acceleration_lin],
        'deceleration_parab': [acceleration_parab],
        'decel_parab_t0': [acceleration_parab_t0],
        'decel_t0': [decel_t0],
        'decel_jacchia': [acc_jacchia],
        'zenith_angle': [data_array['zenith_angle']],
        'kurtosis': [kurtosyness],
        'skew': [skewness],
        'avg_lag': [avg_lag],
        'kc': [kc_par], 
        'Dynamic_pressure_peak_abs_mag': [data_array['Dynamic_pressure_peak_abs_mag']],
        'a_acc': [a3],
        'b_acc': [b3],
        'c_acc': [c3],
        'a_t0': [a_t0],
        'b_t0': [b_t0],
        'c_t0': [c_t0],
        'a1_acc_jac': [jac_a1],
        'a2_acc_jac': [jac_a2],
        'a_mag_init': [a3_Inabs],
        'b_mag_init': [b3_Inabs],
        'c_mag_init': [c3_Inabs],
        'a_mag_end': [a3_Outabs],
        'b_mag_end': [b3_Outabs],
        'c_mag_end': [c3_Outabs],
        'mass': [data_array['mass']],
        'rho': [data_array['rho']],
        'sigma': [data_array['sigma']],
        'erosion_height_start': [data_array['erosion_height_start']],
        'erosion_coeff': [data_array['erosion_coeff']],
        'erosion_mass_index': [data_array['erosion_mass_index']],
        'erosion_mass_min': [data_array['erosion_mass_min']],
        'erosion_mass_max': [data_array['erosion_mass_max']],
        'erosion_range': [data_array['erosion_range']],
        'erosion_energy_per_unit_cross_section': [data_array['erosion_energy_per_unit_cross_section']],
        'erosion_energy_per_unit_mass': [data_array['erosion_energy_per_unit_mass']]
    }

    # Create the dataframe
    panda_dataframe_PCA = pd.DataFrame(data_picklefile_pd)

    if data_array['mass']==0:
        # delete the mass 
        panda_dataframe_PCA = panda_dataframe_PCA.drop(columns=['mass', 'rho', 'sigma', 'erosion_height_start', 'erosion_coeff', 'erosion_mass_index', 'erosion_mass_min', 'erosion_mass_max', 'erosion_range', 'erosion_energy_per_unit_cross_section', 'erosion_energy_per_unit_mass'])

    return panda_dataframe_PCA



########## Utils ##########################

# Function to get trajectory data folder
def find_and_extract_trajectory_files(directory, MetSim_extention):
    trajectory_files = []
    file_names = []
    output_folders = []
    input_folders = []
    trajectory_Metsim_file = []

    for root, dirs, files in os.walk(directory):

        # go in each folder and find the file with the end _trajectory.pickle but skip the folder with the name GenSim
        if 'GenSim' in root:
            continue
        
        csv_file_found=False

        for file in files:
            if file.endswith(NAME_SUFX_CSV_OBS):
                # open
                csv_file_found=True
                real_data = pd.read_csv(os.path.join(root, file))
                if root not in real_data['solution_id'][0]:
                    print('The solution_id in the csv file is not the same as the folder name or does not exist in the folder name:', root)
                    continue
                # split real_data['solution_id'][0] in the directory and the name of the file
                _ , file_from_csv = os.path.split(real_data['solution_id'][0])
                
                base_name = os.path.splitext(file_from_csv)[0]  # Remove the file extension
                #check if the file_from_csv endswith "_trajectory" if yes then extract the number 20230405_010203
                if base_name.endswith("_trajectory"):
                    variable_name = base_name.replace("_trajectory", "")  # Extract the number 20230405_010203
                    output_folder_name = base_name.replace("_trajectory", NAME_SUFX_GENSIM) # _GenSim folder whre all generated simulations are stored
                else:
                    variable_name = base_name
                    output_folder_name = base_name + NAME_SUFX_GENSIM
                

                if file_from_csv.endswith("json"):
                    # MetSim_phys_file_path = os.path.join(root, file_from_csv)

                    # from namefile_sel json file open the json file and save the namefile_sel.const part as file_name_obs+'_sim_fit.json'
                    with open(os.path.join(root, file_from_csv)) as json_file:
                        data = json.load(json_file)
                        const_part = data['const']
                        MetSim_phys_file_path = os.path.join(root, output_folder_name)+os.sep+variable_name+'_sim_fit.json'
                        with open(os.path.join(root, output_folder_name)+os.sep+variable_name+'_sim_fit.json', 'w') as outfile:
                            json.dump(const_part, outfile, indent=4)

                else:
                    # check if MetSim_phys_file_path exist
                    if os.path.isfile(os.path.join(root, variable_name + MetSim_extention)):
                        # print did not find with th given extention revert to default
                        MetSim_phys_file_path = os.path.join(root, variable_name + MetSim_extention)
                    elif os.path.isfile(os.path.join(root, variable_name + '_sim_fit_latest.json')):
                        print(base_name,': No MetSim file with the given extention', MetSim_extention,'reverting to default extention _sim_fit_latest.json')
                        MetSim_phys_file_path = os.path.join(root, variable_name + '_sim_fit_latest.json')
                    else:
                        # do not save the rest of the files
                        print(base_name,': No MetSim file with the given extention', MetSim_extention,'do not consider the folder')
                        continue


                input_folders.append(root)
                trajectory_files.append(os.path.join(root, file))
                file_names.append(variable_name)
                output_folders.append(os.path.join(root, output_folder_name))
                trajectory_Metsim_file.append(MetSim_phys_file_path)

                

        if csv_file_found==False:   
            for file in files:
                if file.endswith("_trajectory.pickle"):
                    base_name = os.path.splitext(file)[0]  # Remove the file extension
                    variable_name = base_name.replace("_trajectory", "")  # Extract the number 20230405_010203
                    output_folder_name = base_name.replace("_trajectory", NAME_SUFX_GENSIM) # _GenSim folder whre all generated simulations are stored

                    # check if MetSim_phys_file_path exist
                    if os.path.isfile(os.path.join(root, variable_name + MetSim_extention)):
                        # print did not find with th given extention revert to default
                        MetSim_phys_file_path = os.path.join(root, variable_name + MetSim_extention)
                    elif os.path.isfile(os.path.join(root, variable_name + '_sim_fit_latest.json')):
                        print(base_name,': No MetSim file with the given extention', MetSim_extention,'reverting to default extention _sim_fit_latest.json')
                        MetSim_phys_file_path = os.path.join(root, variable_name + '_sim_fit_latest.json')
                    else:
                        # do not save the rest of the files
                        print(base_name,': No MetSim file with the given extention', MetSim_extention,'do not consider the folder')
                        continue

                    input_folders.append(root)
                    trajectory_files.append(os.path.join(root, file))
                    file_names.append(variable_name)
                    output_folders.append(os.path.join(root, output_folder_name))
                    trajectory_Metsim_file.append(MetSim_phys_file_path)

    input_list = [[trajectory_files[ii], file_names[ii], input_folders[ii], output_folders[ii], trajectory_Metsim_file[ii]] for ii in range(len(trajectory_files))]

    return input_list



def update_sigma_values(file_path, mag_sigma, len_sigma, More_complex_fit=False, Custom_refinement=False):
    with open(file_path, 'r') as file:
        content = file.read()
        
    # Modify mag_sigma and len_sigma
    content = re.sub(r'"mag_sigma":\s*[\d.]+', f'"mag_sigma": {mag_sigma}', content)
    content = re.sub(r'"len_sigma":\s*[\d.]+', f'"len_sigma": {len_sigma}', content)
    
    if More_complex_fit:
        # Enable "More complex fit - overall fit"
        content = re.sub(
            r'(# More complex fit - overall fit\s*\{[^{}]*"enabled":\s*)false', 
            r'\1true',
            content
        )
    else:
        # Enable "More complex fit - overall fit"
        content = re.sub(
            r'(# More complex fit - overall fit\s*\{[^{}]*"enabled":\s*)true', 
            r'\1false',
            content
        )
    
    if Custom_refinement:
        # Enable "Custom refinement of erosion parameters - improves wake"
        content = re.sub(
            r'(# Custom refinement of erosion parameters - improves wake\s*\{[^{}]*"enabled":\s*)false', 
            r'\1true',
            content
        )
    else:
        # Enable "Custom refinement of erosion parameters - improves wake"
        content = re.sub(
            r'(# Custom refinement of erosion parameters - improves wake\s*\{[^{}]*"enabled":\s*)true', 
            r'\1false',
            content
        )

    # Save the modified content back to the file
    with open(file_path, 'w') as file:
        file.write(content)
    
    print('modified options file:', file_path)



########## Distance ##########################


# Function to find the knee of the distance plot
def find_knee_dist_index(data_meteor_pd, window_of_smothing_avg=3, std_multip_threshold=1, output_path='', around_meteor='', N_sim_sel_force=0):
    dist_for_meteor=np.array(data_meteor_pd['distance_meteor'])
    #make subtraction of the next element and the previous element of data_for_meteor["distance_meteor"]
    # diff_distance_meteor = np.diff(dist_for_meteor[:int(len(dist_for_meteor)/10)])
    diff_distance_meteor = np.diff(dist_for_meteor)
    # histogram plot of the difference with the count on the x axis and diff_distance_meteor on the y axis 
    indices = np.arange(len(diff_distance_meteor))
    # create the cumulative sum of the diff_distance_meteor
    cumsum_diff_distance_meteor = np.cumsum(diff_distance_meteor)
    # normalize the diff_distance_meteor xnormalized = (x - xminimum) / range of x
    diff_distance_meteor_normalized = (diff_distance_meteor - np.min(diff_distance_meteor)) / (np.max(diff_distance_meteor) - np.min(diff_distance_meteor))

    def moving_average_smoothing(data, window_size):
        smoothed_data = np.convolve(data, np.ones(window_size)/window_size, mode='same')
        return smoothed_data

    # apply the smoothing finction
    smoothed_diff_distance_meteor = moving_average_smoothing(diff_distance_meteor_normalized, window_of_smothing_avg)
    
    # fid the first value of the smoothed_diff_distance_meteor that is smaller than the std of the smoothed_diff_distance_meteor
    index10percent = np.where(smoothed_diff_distance_meteor < np.std(smoothed_diff_distance_meteor)*std_multip_threshold)[0][0]-2
    
    if N_sim_sel_force!=0:
        index10percent = N_sim_sel_force

    if index10percent<0: # below does not work problem with finding the mode on KDE later on
        index10percent=0

    if output_path!='':

        # Define a custom palette
        custom_palette_orange = {
            'Real': "darkorange",
            'Simulation': "darkorange",
            'Simulation_sel': "darkorange",
            'MetSim': "darkorange",
            'Realization': "darkorange",
            'Observation': "darkorange"
        }

        # dimension of the plot 15,5
        plt.figure(figsize=(15,5))

        plt.subplot(1,2,2)
        sns.histplot(data_meteor_pd, x="distance_meteor", hue="type", kde=True, cumulative=True, bins=len(dist_for_meteor), palette=custom_palette_orange) # , stat='density' to have probability
        plt.xlabel('Distance in PCA space')
        plt.ylabel('Number of events')
        plt.title('Cumulative distance in PCA space')
        plt.axvline(x=(dist_for_meteor[index10percent]), color="darkorange", linestyle='--', label='Knee distance')

        if len(dist_for_meteor)>100:
            plt.ylim(0,100) 
        elif len(dist_for_meteor)>50:
            plt.ylim(0,50)
        
        plt.legend()
        # delete the legend
        plt.legend().remove()


        plt.subplot(1,2,1)
        # sns.histplot(diff_distance_meteor_normalized, kde=True, bins=len(distance_meteor_sel_save))
        #make the bar plot 0.5 transparency
        
        plt.bar(indices, diff_distance_meteor_normalized,color="darkorange", alpha=0.5, edgecolor='black')
        plt.xlabel('Number of events')
        plt.ylabel('Normalized difference')
        plt.title('Distance difference Normalized')
        # put a horizontal line at len(curr_sel['distance_meteor'])
        plt.axvline(x=index10percent, color="darkorange", linestyle='--') 
        if len(dist_for_meteor)>100:
            plt.xlim(-1,100) 
        elif len(dist_for_meteor)>50:
            plt.xlim(-1,50)

        # find the mean of the first 100 elements of diff_distance_meteor_normalized and put a horizontal line
        # plt.axhline(y=np.std(smoothed_diff_distance_meteor), color="darkorange", linestyle='--')

        # set a sup title
        plt.suptitle(around_meteor)

        # give more space
        plt.tight_layout()  
        # plt.show()

        # save the figure maximized and with the right name
        plt.savefig(output_path+os.sep+around_meteor+os.sep+around_meteor+'_knee'+str(index10percent+1)+'ev_MAXdist'+str(np.round(dist_for_meteor[index10percent],2))+'.png', dpi=300)

        # close the figure
        plt.close()

    return index10percent

# function to use the mahaloby distance and from the mean of the selected shower
def dist_PCA_space_select_sim(df_sim_PCA, shower_current_PCA_single, cov_inv, meanPCA_current, df_sim_shower, shower_current_single, N_sim_sel_force=0, output_dir=''):
    N_sim_sel_all=100
    print('calculate distance for',shower_current_single['solution_id'])

    df_sim_PCA_for_now = df_sim_PCA.drop(['type'], axis=1).values

    distance_current = []
    for i_sim in range(len(df_sim_PCA_for_now)):
        distance_current.append(mahalanobis_distance(df_sim_PCA_for_now[i_sim], shower_current_PCA_single, cov_inv))

    # create an array with lenght equal to the number of simulations and set it to shower_current_PCA['solution_id'][i_shower]
    solution_id_dist = [shower_current_single['solution_id']] * len(df_sim_PCA_for_now)
    df_sim_shower['solution_id_dist'] = solution_id_dist
    df_sim_shower['distance_meteor'] = distance_current
    # sort the distance and select the n_selected closest to the meteor
    df_sim_shower_dis = df_sim_shower.sort_values(by=['distance_meteor']).reset_index(drop=True)
    df_sim_selected = df_sim_shower_dis[:N_sim_sel_all].drop(['type'], axis=1)
    df_sim_selected['type'] = 'Simulation_sel'

    # create a dataframe with the selected simulated shower characteristics
    df_sim_PCA_dist = df_sim_PCA
    df_sim_PCA_dist['distance_meteor'] = distance_current
    df_sim_PCA_dist = df_sim_PCA_dist.sort_values(by=['distance_meteor']).reset_index(drop=True)
    # delete the shower code
    df_sim_selected_PCA = df_sim_PCA_dist[:N_sim_sel_all].drop(['type','distance_meteor'], axis=1)

    # make df_sim_selected_PCA an array
    df_sim_selected_PCA = df_sim_selected_PCA.values
    distance_current_mean = []
    for i_shower in range(len(df_sim_selected)):
        distance_current_mean.append(scipy.spatial.distance.euclidean(meanPCA_current, df_sim_selected_PCA[i_shower]))
    df_sim_selected['distance_mean']=distance_current_mean # from the mean of the selected shower

    df_curr_sel_curv = df_sim_selected.copy()

    around_meteor=shower_current_single['solution_id']
    # check if around_meteor is a file in a folder
    if os.path.exists(around_meteor):
        # split in file and directory
        _, around_meteor = os.path.split(around_meteor)
        around_meteor = around_meteor[:15]

    mkdirP(output_dir+os.sep+around_meteor)
    window_of_smothing_avg=3
    std_multip_threshold=1
    if N_sim_sel_force!=0:
        print(around_meteor,'select the best',N_sim_sel_force,'simulations')
        dist_to_cut=find_knee_dist_index(df_curr_sel_curv,window_of_smothing_avg,std_multip_threshold, output_dir, around_meteor, N_sim_sel_force)
        # change of curvature print
        df_curr_sel_curv=df_curr_sel_curv.iloc[:dist_to_cut]
    else:
        dist_to_cut=find_knee_dist_index(df_curr_sel_curv,window_of_smothing_avg,std_multip_threshold, output_dir, around_meteor)
        print(around_meteor,'index of the knee distance',dist_to_cut+1)
        # change of curvature print
        df_curr_sel_curv=df_curr_sel_curv.iloc[:dist_to_cut+1]

    return df_sim_selected, df_curr_sel_curv


#### Matrix function ############################################################################



# Function to perform Varimax rotation
def varimax(Phi, gamma=1.0, q=20, tol=1e-6):
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for i in range(q):
        d_old = d
        Lambda = np.dot(Phi, R)
        u, s, vh = svd(np.dot(Phi.T, np.asarray(Lambda) ** 3 - (gamma / p) * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T, Lambda))))))
        R = np.dot(u, vh)
        d = np.sum(s)
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return np.dot(Phi, R)

# Function to perform mahalanobis distance
def mahalanobis_distance(x, mean, cov_inv):
    diff = x - mean
    return np.sqrt(np.dot(np.dot(diff, cov_inv), diff.T))



# PCA ####################################################################################

def PCASim(df_sim_shower, df_obs_shower, OUT_PUT_PATH, PCA_percent=99, N_sim_sel=0, variable_PCA=[], No_var_PCA=['kurtosis','skew','a1_acc_jac','a2_acc_jac','a_acc','b_acc','c_acc','c_mag_init','c_mag_end','a_t0', 'b_t0', 'c_t0'], file_name_obs='', cores_parallel=None, PCA_pairplot=False, esclude_real_solution_from_selection=False):
    '''
    This function generate the simulated shower from the erosion model and apply PCA.
    The function read the json file in the folder and create a csv file with the simulated shower and take the data from GenerateSimulation.py folder.
    The function return the dataframe of the selected simulated shower.

    'solution_id','type','vel_init_norot','vel_avg_norot','duration',
    'mass','peak_mag_height','begin_height','end_height','t0','peak_abs_mag','beg_abs_mag','end_abs_mag',
    'F','trail_len','deceleration_lin','deceleration_parab','decel_jacchia','decel_t0','zenith_angle', 'kurtosis','skew',
    'kc','Dynamic_pressure_peak_abs_mag',
    'a_acc','b_acc','c_acc','a1_acc_jac','a2_acc_jac','a_mag_init','b_mag_init','c_mag_init','a_mag_end','b_mag_end','c_mag_end',
    'rho','sigma','erosion_height_start','erosion_coeff', 'erosion_mass_index',
    'erosion_mass_min','erosion_mass_max','erosion_range',
    'erosion_energy_per_unit_cross_section', 'erosion_energy_per_unit_mass'

    '''

    # if variable_PCA is not empty
    if variable_PCA != []:
        # add to variable_PCA array 'type','solution_id'
        variable_PCA = ['solution_id','type'] + variable_PCA
        if No_var_PCA != []:
            # remove from variable_PCA the variables in No_var_PCA
            for var in No_var_PCA:
                variable_PCA.remove(var)

    else:
        # put in variable_PCA all the variables except mass
        variable_PCA = list(df_obs_shower.columns)
        # check if mass is in the variable_PCA
        if 'mass' in variable_PCA:
            # remove mass from variable_PCA
            variable_PCA.remove('mass')
        # if No_var_PCA is not empty
        if No_var_PCA != []:
            # remove from variable_PCA the variables in No_var_PCA
            for var in No_var_PCA:
                # check if the variable is in the variable_PCA
                if var in variable_PCA:
                    variable_PCA.remove(var)

    scaled_sim=df_sim_shower[variable_PCA].copy()
    scaled_sim=scaled_sim.drop(['type','solution_id'], axis=1)

    print(len(scaled_sim.columns),'Variables for PCA:\n',scaled_sim.columns)

    # Standardize each column separately
    scaler = StandardScaler()
    df_sim_var_sel_standardized = scaler.fit_transform(scaled_sim)
    df_sim_var_sel_standardized = pd.DataFrame(df_sim_var_sel_standardized, columns=scaled_sim.columns)

    # Identify outliers using Z-score method on standardized data
    z_scores = np.abs(zscore(df_sim_var_sel_standardized))
    threshold = 3
    outliers = (z_scores > threshold).any(axis=1)

    # outlier number 0 has alway to be the False
    if outliers[0]==True:
        print('The MetSim reduction is an outlier, still keep it for the PCA analysis')
        outliers[0]=False

    # Assign df_sim_shower to the version without outliers
    df_sim_shower = df_sim_shower[~outliers].copy()


    # if PCA_pairplot:

    # scale the data so to be easily plot against each other with the same scale
    df_sim_var_sel = df_sim_shower[variable_PCA].copy()
    df_sim_var_sel = df_sim_var_sel.drop(['type','solution_id'], axis=1)

    if len(df_sim_var_sel)>10000:
        # pick randomly 10000 events
        print('Number of events in the simulated :',len(df_sim_var_sel))
        df_sim_var_sel=df_sim_var_sel.sample(n=10000)

    # make a subplot of the distribution of the variables
    fig, axs = plt.subplots(int(np.ceil(len(variable_PCA[2:])/5)), 5, figsize=(20, 15))
    # flat it
    axs = axs.flatten()
    for i, var in enumerate(variable_PCA[2:]):
        # plot the distribution of the variable
        sns.histplot(df_sim_var_sel[var], kde=True, ax=axs[i], color='b', alpha=0.5, bins=20)
        # axs[i//4, i%4].set_title('Distribution of '+var)
        # put a vertical line for the df_obs_shower[var] value
        axs[i].axvline(df_obs_shower[var].values[0], color='limegreen', linestyle='--', linewidth=5)
        # x axis
        axs[i].set_xlabel(var)
        # # grid
        # axs[i//5, i%5].grid()
        if i != 0 and i != 5 and i != 10 and i != 15 and i != 20:
            # delete the y axis
            axs[i].set_ylabel('')

    # delete the plot that are not used
    for i in range(len(variable_PCA[2:]), len(axs)):
        fig.delaxes(axs[i])

    # space between the subplots
    plt.tight_layout()

    # save the figure
    plt.savefig(OUT_PUT_PATH+os.sep+file_name_obs+'_var_hist_real.png')
    # close the figure
    plt.close()
        


    ##################################### delete var that are not in the 5 and 95 percentile of the simulated shower #####################################

    # check if a file with the name "log"+n_PC_in_PCA+"_"+str(len(df_sel))+"ev.txt" already exist
    if os.path.exists(OUT_PUT_PATH+os.sep+"log_"+file_name_obs[:15]+"_"+str(len(variable_PCA)-2)+"var_"+str(PCA_percent)+"%.txt"):
        # remove the file
        os.remove(OUT_PUT_PATH+os.sep+"log_"+file_name_obs[:15]+"_"+str(len(variable_PCA)-2)+"var_"+str(PCA_percent)+"%.txt")
    sys.stdout = Logger(OUT_PUT_PATH,"log_"+file_name_obs[:15]+"_"+str(len(variable_PCA)-2)+"var_"+str(PCA_percent)+"%.txt") # _30var_99%_13PC

    df_all = pd.concat([df_sim_shower[variable_PCA],df_obs_shower[variable_PCA]], axis=0, ignore_index=True)
    # delete nan
    df_all = df_all.dropna()

    # create a copy of df_sim_shower for the resampling
    df_sim_shower_resample=df_sim_shower.copy()
    # df_obs_shower_resample=df_obs_shower.copy()
    No_var_PCA_perc=[]
    # check that all the df_obs_shower for variable_PCA is within th 5 and 95 percentie of df_sim_shower of variable_PCA
    for var in variable_PCA:
        if var != 'type' and var != 'solution_id':
            # check if the variable is in the df_obs_shower
            if var in df_obs_shower.columns:
                # check if the variable is in the df_sim_shower
                if var in df_sim_shower.columns:

                    ii_all=0
                    for i_var in range(len(df_obs_shower[var])):
                        # check if all the values are outside the 5 and 95 percentile of the df_sim_shower if so delete the variable from the variable_PCA
                        if df_obs_shower[var][i_var] < np.percentile(df_sim_shower[var], 1) or df_obs_shower[var][i_var] > np.percentile(df_sim_shower[var], 99):
                            ii_all=+ii_all

                    print(var)

                    if ii_all==len(df_obs_shower[var]):
                        print('The observed and all realization',var,'are not within the 1 and 99 percentile of the simulated meteors')
                        # delete the variable from the variable_PCA
                        variable_PCA.remove(var)
                        # save the var deleted in a variable
                        No_var_PCA_perc.append(var)

                        df_all = df_all.drop(var, axis=1)
                    else:
                        shapiro_test = stats.shapiro(df_all[var])
                        print("Initial Shapiro-Wilk Test:", shapiro_test.statistic,"p-val", shapiro_test.pvalue)

                        if var=='zenith_angle':
                            # # do the cosine of the zenith angle
                            # df_all[var]=np.cos(np.radians(df_all[var]))
                            # # df_all[var]=transform_to_gaussian(df_all[var])
                            # df_sim_shower_resample[var]=np.cos(np.radians(df_sim_shower_resample[var]))
                            print('Variable ',var,' is not transformed')

                        elif var=='vel_init_norot':
                            # do the cosine of the zenith angle
                            # df_all[var]=transform_to_gaussian(df_all[var])
                            print('Variable ',var,' is not transformed')

                        else:

                            pt = PowerTransformer(method='yeo-johnson')
                            df_all[var]=pt.fit_transform(df_all[[var]])
                            df_sim_shower_resample[var]=pt.fit_transform(df_sim_shower_resample[[var]])

                        shapiro_test = stats.shapiro(df_all[var])
                        print("NEW Shapiro-Wilk Test:", shapiro_test.statistic,"p-val", shapiro_test.pvalue)
                        
                else:
                    print('Variable ',var,' is not in the simulated shower')
            else:
                print('Variable ',var,' is not in the observed shower')



    # if PCA_pairplot:
    df_all_nameless_plot=df_all.copy()

    if len(df_all_nameless_plot)>10000:
        # pick randomly 10000 events
        print('Number of events in the simulated:',len(df_all_nameless_plot))
        df_all_nameless_plot=df_all_nameless_plot.sample(n=10000)

    # make a subplot of the rho againist each variable_PCA as a scatter plot
    fig, axs = plt.subplots(int(np.ceil(len(variable_PCA[2:])/5)), 5, figsize=(20, 15))
    # flat it
    axs = axs.flatten()
    for i, var in enumerate(variable_PCA[2:]):
        # plot the distribution of the variable
        sns.histplot(df_all_nameless_plot[var].values[:len(df_sim_shower[variable_PCA])], kde=True, ax=axs[i], color='b', alpha=0.5, bins=20)
        # axs[i//4, i%4].set_title('Distribution of '+var)
        # put a vertical line for the df_obs_shower[var] value
        # print(df_all_nameless_plot['solution_id'].values[len(df_sim_shower[variable_PCA])])
        axs[i].axvline(df_all_nameless_plot[var].values[len(df_sim_shower[variable_PCA])], color='limegreen', linestyle='--', linewidth=5)       
        # x axis
        axs[i].set_xlabel(var)
        # # grid
        # axs[i//5, i%5].grid()
        if i != 0 and i != 5 and i != 10 and i != 15 and i != 20:
            # delete the y axis
            axs[i].set_ylabel('')
    
    # space between the subplots
    plt.tight_layout()

    # save the figure
    plt.savefig(OUT_PUT_PATH+os.sep+file_name_obs+'_var_hist_yeo-johnson.png')
    # close the figure
    plt.close()

    ####################################################################################################################

    # Now we have all the data and we apply PCA to the dataframe
    df_all_nameless=df_all.drop(['type','solution_id'], axis=1)

    # print the data columns names
    df_all_columns_names=(df_all_nameless.columns)

    # Separating out the features
    scaled_df_all = df_all_nameless[df_all_columns_names].values

    # performing preprocessing part so to make it readeble for PCA
    scaled_df_all = StandardScaler().fit_transform(scaled_df_all)


    #################################
    # Applying PCA function on the data for the number of components
    pca = PCA(PCA_percent/100) #PCA_percent
    # pca = PCA() #PCA_percent
    all_PCA = pca.fit_transform(scaled_df_all) # fit the data and transform it

    #count the number of PC
    print('Number of PC:',pca.n_components_)

    ################################# Apply Varimax rotation ####################################
    loadings = pca.components_.T

    rotated_loadings = varimax(loadings)

    # # chage the loadings to the rotated loadings in the pca components
    pca.components_ = rotated_loadings.T

    # Transform the original PCA scores with the rotated loadings ugly PC space but same results
    # all_PCA = np.dot(all_PCA, rotated_loadings.T[:pca.n_components_, :pca.n_components_])

    ############### PCR ########################################################################################

    # Example limits for the physical variables (adjust these based on your domain knowledge)
    limits = {
        'mass': (np.min(df_sim_shower['mass']), np.max(df_sim_shower['mass'])),  # Example limits
        'rho': (np.min(df_sim_shower['rho']), np.max(df_sim_shower['rho'])),
        'sigma': (np.min(df_sim_shower['sigma']), np.max(df_sim_shower['sigma'])),
        'erosion_height_start': (np.min(df_sim_shower['erosion_height_start']), np.max(df_sim_shower['erosion_height_start'])),
        'erosion_coeff': (np.min(df_sim_shower['erosion_coeff']), np.max(df_sim_shower['erosion_coeff'])),
        'erosion_mass_index': (np.min(df_sim_shower['erosion_mass_index']), np.max(df_sim_shower['erosion_mass_index'])),
        'erosion_mass_min': (np.min(df_sim_shower['erosion_mass_min']), np.max(df_sim_shower['erosion_mass_min'])),
        'erosion_mass_max': (np.min(df_sim_shower['erosion_mass_max']), np.max(df_sim_shower['erosion_mass_max']))
    }

    exclude_columns = ['type', 'solution_id']
    physical_vars = ['mass','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max'] #, 'erosion_range', 'erosion_energy_per_unit_cross_section', 'erosion_energy_per_unit_mass'

    # Delete specific columns from variable_PCA
    variable_PCA_no_info = [col for col in variable_PCA if col not in exclude_columns]

    # # Scale the data
    # scaled_sim = pd.DataFrame(scaler.fit_transform(df_sim_shower[variable_PCA_no_info + physical_vars]), columns=variable_PCA_no_info + physical_vars)

    # Define X and y (now y contains only the PCA observable parameters)
    X = df_sim_shower_resample[variable_PCA_no_info]
    y = df_sim_shower_resample[physical_vars]

    # Split the data into training and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)

    # Loop over the number of principal components
    print("PCR Predictions with "+str(pca.n_components_)+"PC :")

    pca_copy=copy.deepcopy(pca)
    # PCR: Principal Component Regression inform that the predicted variable is always positive
    pcr = make_pipeline(StandardScaler(), pca_copy, LinearRegression())

    pcr.fit(X_train, y_train)
    # Predict using the models
    y_pred_pcr = pcr.predict(df_sim_shower_resample[variable_PCA_no_info])
    # to_plot_unit=['mass [kg]','rho [kg/m^3]','sigma [s^2/km^2]','erosion height start [km]','erosion coeff [s^2/km^2]','erosion mass index [-]','eros. mass min [kg]','eros. mass max [kg]']
    to_plot_unit = [r'$m_0$ [kg]', r'$\rho$ [kg/m$^3$]', r'$\sigma$ [s$^2$/km$^2$]', r'$h_{e}$ [km]', r'$\eta$ [s$^2$/km$^2$]', r'$s$ [-]', r'$m_{l}$ [kg]', r'$m_{u}$ [kg]'] #,r'log($m_{u}$)-log($m_{l}$) [-]']
    # multiply y_pred_pcr that has the 'erosion_coeff'*1000000 and 'sigma'*1000000
    y_pred_pcr[:,4]=y_pred_pcr[:,4]*1000000
    y_pred_pcr[:,2]=y_pred_pcr[:,2]*1000000
    # Get the real values
    real_values = df_sim_shower_resample[physical_vars].iloc[0].values
    # multiply the real_values
    real_values[4]=real_values[4]*1000000
    real_values[2]=real_values[2]*1000000

    # # Apply limits to the predictions
    # for i, var in enumerate(physical_vars):
    #     y_pred_pcr[:, i] = np.clip(y_pred_pcr[:, i], limits[var][0], limits[var][1])

    # Print the predictions alongside the real values
    print("Predicted vs Real Values:")
            # print(output_dir+os.sep+'PhysicProp'+n_PC_in_PCA+'_'+str(len(curr_sel))+'ev_dist'+str(np.round(np.min(curr_sel['distance_meteor']),2))+'-'+str(np.round(np.max(curr_sel['distance_meteor']),2))+'.png')
    for i, unit in enumerate(to_plot_unit):
        y_pred_pcr[0, i]= abs(y_pred_pcr[0, i])
        print(f'{unit}: Predicted: {y_pred_pcr[0, i]:.4g}, Real: {real_values[i]:.4g}')

    pcr_results_physical_param = y_pred_pcr.copy()
    print('--------------------------')

    ############### PCR ########################################################################################


    # # select only the column with in columns_PC with the same number of n_components
    columns_PC = ['PC' + str(x) for x in range(1, pca.n_components_+1)]

    # create a dataframe with the PCA space
    df_all_PCA = pd.DataFrame(data = all_PCA, columns = columns_PC)

    ### plot var explained by each PC bar

    percent_variance = np.round(pca.explained_variance_ratio_* 100, decimals =2)

    # plot the explained variance ratio of each principal componenets base on the number of column of the original dimension
    plt.bar(x= range(1,len(percent_variance)+1), height=percent_variance, tick_label=columns_PC, color='black')
    # ad text at the top of the bar with the percentage of variance explained
    for i in range(1,len(percent_variance)+1):
        # reduce text size
        plt.text(i, percent_variance[i-1], str(percent_variance[i-1])+'%', ha='center', va='bottom', fontsize=5)

    plt.ylabel('Percentance of Variance Explained')
    plt.xlabel('Principal Component')
    # save the figure
    plt.savefig(OUT_PUT_PATH+os.sep+file_name_obs+'PCAexplained_variance_ratio_'+str(len(variable_PCA)-2)+'var_'+str(PCA_percent)+'%_'+str(pca.n_components_)+'PC.png')
    # close the figure
    plt.close()
    # plt.show()

    ### plot covariance matrix

    # make the image big as the screen
    # plt.figure(figsize=(20, 20))

    # Compute the correlation coefficients
    # cov_data = pca.components_.T
    # varimax rotation
    cov_data = rotated_loadings

    # Plot the correlation matrix
    img = plt.matshow(cov_data.T, cmap=plt.cm.coolwarm, vmin=-1, vmax=1)
    plt.colorbar(img)

    # Mapping of original variable names to LaTeX-style labels
    variable_map = {
        'vel_init_norot': r"$v_i$",
        'vel_avg_norot': r"$v_{avg}$",
        'duration': r"$t$",
        'peak_mag_height': r"$h_{p}$",
        'begin_height': r"$h_{beg}$",
        'end_height': r"$h_{end}$",
        'peak_abs_mag': r"$M_{p}$",
        'beg_abs_mag': r"$M_{beg}$",
        'end_abs_mag': r"$M_{end}$",
        'F': r"$F$",
        'trail_len': r"$L$",
        't0': r"$t_0$",
        'deceleration_lin': r"$dAcc_{lin}$",
        'deceleration_parab': r"$dAcc_{par}$",
        'decel_parab_t0': r"$dAcc_{p_{t_0}}$",
        'decel_t0': r"$dAcc_{p1_{t_0}}$",
        'decel_jacchia': r"$dAcc_{jac}$",
        'zenith_angle': r"$\zeta$",
        'avg_lag': r"$lag_{avg}$",
        'kc': r"$k_c$",
        'Dynamic_pressure_peak_abs_mag': r"$P_p$",
        'a_mag_init': r"$Mfit_{a_{int}}$",
        'b_mag_init': r"$Mfit_{b_{int}}$",
        'a_mag_end': r"$Mfit_{a_{fin}}$",
        'b_mag_end': r"$Mfit_{b_{fin}}$"
    }
    
    # Convert the given array to LaTeX-style labels
    latex_labels = [variable_map.get(var, var) for var in variable_PCA]

    rows_8 = [x for x in latex_labels]

    # add to the columns the PC number the percent_variance
    columns_PC_with_var = ['PC' + str(x) + ' (' + str(percent_variance[x-1]) + '%)' for x in range(1, pca.n_components_+1)]

    # Add the variable names as labels on the x-axis and y-axis
    plt.xticks(range(len(rows_8)-2), rows_8[2:], rotation=90)
    # yticks with variance explained
    plt.yticks(range(len(columns_PC_with_var)), columns_PC_with_var)

    # plot the influence of each component on the original dimension
    for i in range(cov_data.shape[0]):
        for j in range(cov_data.shape[1]):
            plt.text(i, j, "{:.1f}".format(cov_data[i, j]), size=5, color='black', ha="center", va="center")   
    # save the figure
    plt.savefig(OUT_PUT_PATH+os.sep+file_name_obs+'PCAcovariance_matrix_'+str(len(variable_PCA)-2)+'var_'+str(PCA_percent)+'%_'+str(pca.n_components_)+'PC.png')
    # close the figure
    plt.close()
    # plt.show()
    ###

    # print the number of simulation selected
    print('PCA run for', len(df_sim_shower),'simulations, delete ',len(outliers)-len(df_sim_shower),' outliers')

    # if len(No_var_PCA_perc) > 0:
    #     for No_var_PCA_perc in No_var_PCA_perc:
    #         print('Observable data variable [',No_var_PCA_perc,'] is not within the 5 and 95 percentile of the simulated shower')

    # print the name of the variables used in PCA
    print('Variables used in PCA: ',df_all_nameless.columns)

    print("explained variance ratio: \n",percent_variance)

    print(str(len(variable_PCA)-2)+' var = '+str(PCA_percent)+'% of the variance explained by ',pca.n_components_,' PC')


    # add the shower code to the dataframe
    df_all_PCA['type'] = df_all['type'].values

    # delete the lines after len(df_sim_shower) to have only the simulated shower
    df_sim_PCA = df_all_PCA.drop(df_all_PCA.index[len(df_sim_shower):])
    df_obs_PCA = df_all_PCA.drop(df_all_PCA.index[:len(df_sim_shower)])

    
    ########### Distance metric takes in to account varinace explained ####################################################################

    if esclude_real_solution_from_selection:
        df_all_PCA_cov = df_all_PCA[df_all_PCA['type'] != 'Real'].copy()
    else:
        # delete the type Real from
        df_all_PCA_cov = df_all_PCA.copy()

    # Get explained variances of principal components
    explained_variance = pca.explained_variance_ratio_

    # Calculate mean and inverse covariance matrix for Mahalanobis distance
    cov_matrix = df_all_PCA_cov.drop(['type'], axis=1).cov()

    # Modify covariance matrix based on explained variances
    for i in range(len(explained_variance)):
        cov_matrix.iloc[i, :] /= explained_variance[i]

    # # Modify covariance matrix to positively reflect variance explained
    # for i in range(len(explained_variance)):
    #     cov_matrix.iloc[i, :] *= explained_variance[i]

    cov_inv = inv(cov_matrix)

    ############## SELECTION ###############################################

    # group them by Observation, Realization type and the other group by MetSim, Simulation
    # meanPCA = df_all_PCA.groupby('type').mean() # does not work

    df_all_PCA['solution_id'] = df_all['solution_id']
    # Create a new column to group by broader categories
    group_mapping = {
        'Observation': 'obs',
        'Realization': 'obs',
        'Real': 'sim',
        'MetSim': 'sim',
        'Simulation': 'sim'
    }
    df_all_PCA['group'] = df_all_PCA['type'].map(group_mapping)
    df_obs_shower['group'] = df_obs_shower['type'].map(group_mapping)
    df_obs_PCA['group'] = df_obs_PCA['type'].map(group_mapping)

    # # Group by the new column and calculate the mean
    # meanPCA = df_all_PCA.groupby('group').mean()

    # # drop the sim column
    # meanPCA = meanPCA.drop(['sim'], axis=0)

    # Ensure that only numeric columns are used in the mean calculation
    df_numeric = df_all_PCA.select_dtypes(include=[np.number])

    # Group by the new column and calculate the mean only for numeric columns
    meanPCA = df_numeric.groupby(df_all_PCA['group']).mean()

    # Drop the 'sim' row if it exists
    meanPCA = meanPCA.drop(['sim'], axis=0, errors='ignore')

    # print(meanPCA)

    meanPCA_current = meanPCA.loc[(meanPCA.index == 'obs')].values.flatten()
    # take only the value of the mean of the first row
    shower_current = df_obs_shower[df_obs_shower['group'] == 'obs']
    shower_current_PCA = df_obs_PCA[df_obs_PCA['group'] == 'obs']

    # trasform the dataframe in an array
    shower_current_PCA = shower_current_PCA.drop(['type','group'], axis=1).values

    # define the distance
    mkdirP(OUT_PUT_PATH+os.sep+SAVE_SELECTION_FOLDER)      
    if esclude_real_solution_from_selection:
        # delete the type Real from
        input_list_obs_dist = [[df_sim_PCA[df_sim_PCA['type'] != 'Real'], shower_current_PCA[ii], cov_inv, meanPCA_current, df_sim_shower[df_sim_shower['type'] != 'Real'], shower_current.iloc[ii], N_sim_sel, OUT_PUT_PATH+os.sep+SAVE_SELECTION_FOLDER] for ii in range(len(shower_current))]
        df_sim_selected_both_df = domainParallelizer(input_list_obs_dist, dist_PCA_space_select_sim, cores=cores_parallel)

    else:  
        input_list_obs_dist = [[df_sim_PCA, shower_current_PCA[ii], cov_inv, meanPCA_current, df_sim_shower, shower_current.iloc[ii], N_sim_sel, OUT_PUT_PATH+os.sep+SAVE_SELECTION_FOLDER] for ii in range(len(shower_current))]
        df_sim_selected_both_df = domainParallelizer(input_list_obs_dist, dist_PCA_space_select_sim, cores=cores_parallel)


    # separet df_sim_selected the '<class 'tuple'>' to a list of dataframe called df_sim_selected_all and df_sim_selected_knee
    df_sim_selected_all = []
    df_sim_selected_knee = []
    for item in df_sim_selected_both_df:
        if isinstance(item, tuple):
            df_sim_selected_all.append(item[0])
            df_sim_selected_knee.append(item[1])

    df_sim_selected_all = pd.concat(df_sim_selected_all)
    df_sel_shower = pd.concat(df_sim_selected_knee)

    # DELETE ALL INDEX

    # Insert the column at the first position
    df_sim_selected_all.insert(1, 'distance_mean', df_sim_selected_all.pop('distance_mean'))
    df_sim_selected_all.insert(1, 'distance_meteor', df_sim_selected_all.pop('distance_meteor'))
    df_sim_selected_all.insert(1, 'solution_id_dist', df_sim_selected_all.pop('solution_id_dist'))
    df_sim_selected_all.insert(1, 'type', df_sim_selected_all.pop('type'))

    df_sim_selected_all.reset_index(drop=True, inplace=True)

    df_sim_selected_all.to_csv(OUT_PUT_PATH+os.sep+file_name_obs+'_sim_sel.csv', index=False)

    # Insert the column at the first position
    df_sel_shower.insert(1, 'distance_mean', df_sel_shower.pop('distance_mean'))
    df_sel_shower.insert(1, 'distance_meteor', df_sel_shower.pop('distance_meteor'))
    df_sel_shower.insert(1, 'solution_id_dist', df_sel_shower.pop('solution_id_dist'))
    df_sel_shower.insert(1, 'type', df_sel_shower.pop('type'))

    df_sel_shower.reset_index(drop=True, inplace=True)

    df_sel_shower.to_csv(OUT_PUT_PATH+os.sep+file_name_obs+'_sim_sel_bf_knee.csv', index=False)

    if isinstance(df_sel_shower, tuple):
        df_sel_shower = df_sel_shower[0]
    if isinstance(df_sim_selected_all, tuple):
        df_sim_selected_all = df_sim_selected_all[0]

    # DELETE ALL old INDEX

    # Create the new DataFrame by filtering df_sim_PCA
    df_sel_PCA = df_all_PCA[df_all_PCA['solution_id'].isin(df_sel_shower['solution_id'])]
    # change all df_sel_PCA 'type' to Simulation_sel
    df_sel_PCA['type'] = 'Simulation_sel'
    # reset the index
    df_sel_PCA.reset_index(drop=True, inplace=True)

    # df_sel_shower_no_repetitions = df_sim_shower[df_sim_shower['solution_id'].isin(df_sel_shower['solution_id'])]
    # # change all df_sel_PCA 'type' to Simulation_sel
    # df_sel_shower_no_repetitions['type'] = 'Simulation_sel'
    # # reset the index
    # df_sel_shower_no_repetitions.reset_index(drop=True, inplace=True)
    
    df_sel_shower_no_repetitions = df_sel_shower.copy()

    # group by solution_id_dist and keep only n_confront_sel from each group
    df_sel_shower_no_repetitions = df_sel_shower_no_repetitions.groupby('solution_id_dist').head(len(df_sel_shower_no_repetitions))

    # order by distance_meteor
    df_sel_shower_no_repetitions = df_sel_shower_no_repetitions.sort_values('distance_meteor')

    # count duplicates and add a column for the number of duplicates
    df_sel_shower_no_repetitions['num_duplicates'] = df_sel_shower_no_repetitions.groupby('solution_id')['solution_id'].transform('size') 
    
    df_sel_shower_no_repetitions['solution_id_dist'] = df_obs_shower['solution_id'].values[0]

    df_sel_shower_no_repetitions.drop_duplicates(subset='solution_id', keep='first', inplace=True)            

    # save df_sel_shower_real to disk
    df_sel_shower_no_repetitions.to_csv(OUT_PUT_PATH+os.sep+SAVE_SELECTION_FOLDER+os.sep+file_name_obs+'_sim_sel_to_optimize.csv', index=False)



    print('\nSUCCESS: the simulated meteor have been selected\n')

    # Close the Logger to ensure everything is written to the file STOP COPY in TXT file
    sys.stdout.close()

    # Reset sys.stdout to its original value if needed
    sys.stdout = sys.__stdout__

    ########### save dist to observed shower ########################################

    # # save dist also on selected shower
    # distance_current = []
    # for i_shower in range(len(shower_current)):
    #     distance_current.append(scipy.spatial.distance.euclidean(meanPCA_current, shower_current_PCA[i_shower]))
    # shower_current['distance_mean']=distance_current # from the mean of the selected shower
    # shower_current.to_csv(OUT_PUT_PATH+os.sep+file_name_obs+'_obs_and_dist.csv', index=False)

    # PLOT the selected simulated shower ########################################

    # dataframe with the simulated and the selected meteors in the PCA space
    # df_sim_sel_PCA = pd.concat([df_sim_PCA,df_sel_PCA], axis=0)

    if PCA_pairplot:

        df_sim_shower_small=df_sim_shower.copy()

        if len(df_sim_shower_small)>10000: # w/o takes forever to plot
            # pick randomly 10000 events
            df_sim_shower_small=df_sim_shower_small.sample(n=10000)

        print('generating sel sim histogram plot...')

        # Define a custom palette
        custom_palette = {
            'Real': "r",
            'Simulation': "b",
            'Simulation_sel': "darkorange",
            'MetSim': "k",
            'Realization': "mediumaquamarine",
            'Observation': "limegreen"
        }


        curr_df = pd.concat([df_sim_shower_small,df_sel_shower,df_obs_shower], axis=0)

        curr_df['num_type'] = curr_df.groupby('type')['type'].transform('size')
        curr_df['weight'] = 1 / curr_df['num_type']
        

        fig, axs = plt.subplots(int(np.ceil(len(variable_PCA[2:])/5)), 5, figsize=(20, 15))
        # flatten the axs
        axs = axs.flatten()

        # to_plot_unit=['init vel [km/s]','avg vel [km/s]','duration [s]','begin height [km]','peak height [km]','end height [km]','begin abs mag [-]','peak abs mag [-]','end abs mag [-]','F parameter [-]','zenith angle [deg]','deceleration [km/s^2]','trail lenght [km]','kurtosis','skew']

        # to_plot=['vel_init_norot','vel_avg_norot','duration','begin_height','peak_mag_height','end_height','beg_abs_mag','peak_abs_mag','end_abs_mag','F','zenith_angle','decel_parab_t0','trail_len','kurtosis','skew']

        # deleter form curr_df the mass
        #curr_df=curr_df.drop(['mass'], axis=1)
        for ii, var in enumerate(variable_PCA[2:]):

            # if var in ['decel_parab_t0','decel_t0']:
            #     sns.histplot(curr_df, x=x_plot[x_plot>-500], weights=curr_df['weight'][x_plot>-500],hue='type', ax=axs[ii], kde=True, palette=custom_palette, bins=20)
            #     axs[ii].set_xticks([np.round(np.min(x_plot[x_plot>-500]),2),np.round(np.max(x_plot[x_plot>-500]),2)])
            
            # else:

            sns.histplot(curr_df, x=var, weights=curr_df['weight'], hue='type', ax=axs[ii], kde=True, palette=custom_palette, bins=20)
            axs[ii].set_xticks([np.round(np.min(curr_df[var]),2),np.round(np.max(curr_df[var]),2)])

            # if beg_abs_mag','peak_abs_mag','end_abs_mag inver the x axis
            if var in ['beg_abs_mag','peak_abs_mag','end_abs_mag']:
                axs[ii].invert_xaxis()

            # Set the x-axis formatter to ScalarFormatter
            axs[ii].xaxis.set_major_formatter(ScalarFormatter())
            axs[ii].ticklabel_format(useOffset=False, style='plain', axis='x')
            # Set the number of x-axis ticks to 3
            # axs[ii].xaxis.set_major_locator(MaxNLocator(nbins=3))

            axs[ii].set_ylabel('probability')
            axs[ii].set_xlabel(var)
            axs[ii].get_legend().remove()
            # check if there are more than 3 ticks and if yes only use the first and the last

            # put y axis in log scale
            axs[ii].set_yscale('log')
            axs[ii].set_ylim(0.01,1)

            
        # more space between the subplots
        plt.tight_layout()
        # # full screen
        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()

        # save the figure
        fig.savefig(OUT_PUT_PATH+os.sep+file_name_obs+'_Histograms_'+str(len(variable_PCA)-2)+'var_'+str(PCA_percent)+'%_'+str(pca.n_components_)+'PC.png', dpi=300)
        plt.close()

        if len(df_sim_PCA)>10000: # w/o takes forever to plot
            # df_sim_PCA=df_sim_PCA.sample(n=10000)
            # pick only the one with the same index in df_sim_shower_small
            df_sim_PCA = df_sim_PCA[df_sim_PCA.index.isin(df_sim_shower_small.index)] 
        
        print('generating PCA space plot...')

        df_sim_sel_PCA = pd.concat([df_sim_PCA,df_sel_PCA,df_obs_PCA], axis=0)

        # Select only the numeric columns for percentile calculations
        numeric_columns = df_sim_sel_PCA.select_dtypes(include=[np.number]).columns

        # Create a new column for point sizes
        df_sim_sel_PCA['point_size'] = df_sim_sel_PCA['type'].map({
            'Simulation_sel': 5,
            'Simulation': 5,
            'MetSim': 20,
            'Realization': 20,    
            'Observation': 40
        })
        

        # open a new figure to plot the pairplot
        fig = plt.figure(figsize=(10, 10), dpi=300)

        # # fig = sns.pairplot(df_sim_sel_PCA, hue='type', plot_kws={'alpha': 0.6, 's': 5, 'edgecolor': 'k'},corner=True)
        # fig = sns.pairplot(df_sim_sel_PCA, hue='type',corner=True, palette='bright', diag_kind='kde', plot_kws={'s': 5, 'edgecolor': 'k'})
        # # plt.show()

        # Create the pair plot without points initially
        fig = sns.pairplot(df_sim_sel_PCA[numeric_columns.append(pd.Index(['type']))], hue='type', corner=True, palette=custom_palette, diag_kind='kde', plot_kws={'s': 5, 'edgecolor': 'k'})

        # Overlay scatter plots with custom point sizes
        for i in range(len(fig.axes)):
            for j in range(len(fig.axes)):
                if i > j:
                    # check if the variable is in the list of the numeric_columns and set the axis limit
                    if df_sim_sel_PCA.columns[j] in numeric_columns and df_sim_sel_PCA.columns[i] in numeric_columns:

                        ax = fig.axes[i, j]
                        sns.scatterplot(data=df_sim_sel_PCA, x=df_sim_sel_PCA.columns[j], y=df_sim_sel_PCA.columns[i], hue='type', size='point_size', sizes=(5, 40), ax=ax, legend=False, edgecolor='k', palette=custom_palette)

                        # ax.set_xlim(percentiles_1[df_sim_sel_PCA.columns[j]], percentiles_99[df_sim_sel_PCA.columns[j]])
                        # ax.set_ylim(percentiles_1[df_sim_sel_PCA.columns[i]], percentiles_99[df_sim_sel_PCA.columns[i]])

        # delete the last row of the plot
        # fig.axes[-1, -1].remove()
        # Hide the last row of plots
        # for ax in fig.axes[-1]:
        #     ax.remove()

        # Adjust the subplots layout parameters to give some padding
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        # plt.show()
        
        # save the figure
        fig.savefig(OUT_PUT_PATH+os.sep+file_name_obs+'PCAspace_sim_sel_real_'+str(len(variable_PCA)-2)+'var_'+str(PCA_percent)+'%_'+str(pca.n_components_)+'PC.png')
        # close the figure
        plt.close()

        print('generating result variable plot...')

        output_folder=OUT_PUT_PATH+os.sep+file_name_obs+VAR_SEL_DIR_SUFX
        # check if the output_folder exists
        if not os.path.isdir(output_folder):
            mkdirP(output_folder)

        # df_sim_PCA,df_sel_PCA,df_obs_PCA
        # print(df_sim_shower)
        # loop all physical variables
        physical_vars = ['mass','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max']
        for var_phys in physical_vars:
            # make a subplot of the rho againist each variable_PCA as a scatter plot
            fig, axs = plt.subplots(int(np.ceil(len(variable_PCA[2:])/5)), 5, figsize=(20, 15))
            # flat it
            axs = axs.flatten()

            for i, var in enumerate(variable_PCA[2:]):
                # plot the rho againist the variable with black borders
                axs[i].scatter(df_sim_shower_small[var], df_sim_shower_small[var_phys], c='b') #, edgecolors='k', alpha=0.5

                axs[i].scatter(df_sel_shower[var], df_sel_shower[var_phys], c='orange') #, edgecolors='k', alpha=0.5
                # put a green vertical line for the df_obs_shower[var] value
                axs[i].axvline(shower_current[var].values[0], color='limegreen', linestyle='--', linewidth=5)
                # put a horizontal line for the rho of the first df_sim_shower_small
                axs[i].axhline(df_sim_shower[var_phys].values[0], color='k', linestyle='-', linewidth=2)
                # axs[i].set_title(var)
                # as a suptitle put the variable_PCA
                # fig.suptitle(var_phys)
                if i == 0 or i == 5 or i == 10 or i == 15 or i == 20:
                    # as a suptitle put the variable_PCA
                    axs[i].set_ylabel(var_phys)

                # x axis
                axs[i].set_xlabel(var)

                # grid
                axs[i].grid()
                # make y axis log if the variable is 'erosion_mass_min' 'erosion_mass_max'
                if var_phys == 'erosion_mass_min' or var_phys == 'erosion_mass_max':
                    axs[i].set_yscale('log')

            plt.tight_layout()
            # save the figure
            plt.savefig(output_folder+os.sep+file_name_obs+var_phys+'_vs_var_select_PCA.png')
            # close the figure
            plt.close()

        print('generating PCA position plot...')

        output_folder=OUT_PUT_PATH+os.sep+file_name_obs+PCA_SEL_DIR_SUFX
        # check if the output_folder exists
        if not os.path.isdir(output_folder):
            mkdirP(output_folder)

        # loop all pphysical variables
        physical_vars = ['mass','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max']
        for var_phys in physical_vars:

            # make a subplot of the rho againist each variable_PCA as a scatter plot
            fig, axs = plt.subplots(int(np.ceil(len(columns_PC)/5)), 5, figsize=(20, 15))

            # flatten the axs array
            axs = axs.flatten()
            for i, var in enumerate(columns_PC):
                # plot the rho againist the variable with black borders
                axs[i].scatter(df_sim_PCA[var], df_sim_shower_small[var_phys], c='b') #, edgecolors='k', alpha=0.5

                axs[i].scatter(df_sel_PCA[var], df_sel_shower_no_repetitions[var_phys], c='orange') #, edgecolors='k', alpha=0.5
                # put a green vertical line for the df_obs_shower[var] value
                axs[i].axvline(df_obs_PCA[var].values[0], color='limegreen', linestyle='--', linewidth=5)
                # put a horizontal line for the rho of the first df_sim_shower_small
                axs[i].axhline(df_sim_shower[var_phys].values[0], color='k', linestyle='-', linewidth=2)
                # axs[i].set_title(var)
                # # as a suptitle put the variable_PCA
                # fig.suptitle(var_phys)
                if i == 0 or i == 5 or i == 10 or i == 15 or i == 20:
                    # as a suptitle put the variable_PCA
                    axs[i].set_ylabel(var_phys)
                # axis x
                axs[i].set_xlabel(var)
                # grid
                axs[i].grid()
                # make y axis log if the variable is 'erosion_mass_min' 'erosion_mass_max'
                if var_phys == 'erosion_mass_min' or var_phys == 'erosion_mass_max':
                    axs[i].set_yscale('log')

            # delete the subplot that are not used
            for i in range(len(columns_PC), len(axs)):
                fig.delaxes(axs[i])

            plt.tight_layout()
            # save the figure
            plt.savefig(output_folder+os.sep+file_name_obs+var_phys+'_vs_var_select_PC_space.png')
            # close the figure
            plt.close()


    return df_sel_shower, df_sel_shower_no_repetitions, df_sim_selected_all, pcr_results_physical_param, pca.n_components_






def PCAcorrelation_selPLOT(curr_sim_init, curr_sel, n_PC_in_PCA='',output_dir=''):

    curr_sim=curr_sim_init.copy()
    if len(curr_sim)>10000:
        # pick randomly 10000 events
        print('Number of events in the simulated :',len(curr_sim))
        curr_sim=curr_sim.sample(n=10000).copy()

    curr_sel=curr_sel.copy()
    curr_sel = curr_sel.drop_duplicates(subset='solution_id')
    curr_df_sim_sel=pd.concat([curr_sim,curr_sel], axis=0, ignore_index=True)

    # Define your label mappings
    label_mappings = {
        'mass': 'mass [kg]',
        'rho': 'rho [kg/m^3]',
        'sigma': 'sigma [s^2/km^2]',
        'erosion_height_start': 'erosion height start [km]',
        'erosion_coeff': 'erosion coeff [s^2/km^2]',
        'erosion_mass_index': 'erosion mass index [-]',
        'erosion_mass_min': 'log eros. mass min [kg]',
        'erosion_mass_max': 'log eros. mass max [kg]'
    }

    # Define a custom palette
    custom_palette = {
        'Real': "r",
        'Simulation': "b",
        'Simulation_sel': "darkorange",
        'MetSim': "k",
        'Realization': "mediumaquamarine",
        'Observation': "limegreen"
    }

    to_plot8 = ['type', 'mass', 'rho', 'sigma', 'erosion_height_start', 'erosion_coeff', 'erosion_mass_index', 'erosion_mass_min', 'erosion_mass_max']
    hue_column = 'type'


    # Create a PairGrid
    pairgrid = sns.PairGrid(curr_df_sim_sel[to_plot8], hue=hue_column, palette=custom_palette)

    # Map the plots
    pairgrid.map_lower(sns.scatterplot, edgecolor='k', palette=custom_palette)
    # for the upper triangle delete x and y axis
    # pairgrid.map_diag(sns.kdeplot)
    # pairgrid.map_diag(sns.histplot, kde=True, color='k', edgecolor='k')
    # pairgrid.add_legend()

    # Update the labels
    for ax in pairgrid.axes.flatten():
        if ax is not None:  # Check if the axis exists
            xlabel = ax.get_xlabel()
            ylabel = ax.get_ylabel()
            if ylabel in label_mappings:
                ax.set_ylabel(label_mappings[ylabel])
            if xlabel in label_mappings:
                ax.set_xlabel(label_mappings[xlabel])
            if ylabel in ['erosion_mass_min', 'erosion_mass_max']:#'sigma', 
                ax.set_yscale('log')
            if xlabel in ['erosion_mass_min', 'erosion_mass_max']: #'sigma', 
                ax.set_xscale('log')

    # # Calculate the correlation matrix
    # corr = curr_df_sim_sel[to_plot8[1:]].corr()

    corr = curr_sel[to_plot8[1:]].corr()

    # Find the min and max correlation values
    vmin = corr.values.min()
    vmax = corr.values.max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = sns.color_palette('coolwarm', as_cmap=True)

    # Fill the upper triangle plots with the correlation matrix values and color it with the coolwarm cmap
    for i, row in enumerate(to_plot8[1:]):
        for j, col in enumerate(to_plot8[1:]):
            if i < j:
                ax = pairgrid.axes[i, j]  # Adjust index to fit the upper triangle
                corr_value = corr.loc[row, col]
                ax.text(0.5, 0.5, f'{corr_value:.2f}', horizontalalignment='center', verticalalignment='center', fontsize=12, color='black', transform=ax.transAxes)
                ax.set_facecolor(cmap(norm(corr_value)))
                # cmap = sns.color_palette('coolwarm', as_cmap=True)
                # ax.set_facecolor(cmap(corr_value))

                # Remove the axis labels
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
            if i == j:
                ax = pairgrid.axes[i, j]
                ax.set_axis_off()

    # Adjust layout
    plt.tight_layout()

    fig_name = (output_dir+os.sep+'MixPhysicPropPairPlot_'+str(n_PC_in_PCA)+'PC_'+str(len(curr_sel))+'ev.png')
    plt.savefig(fig_name, dpi=300)

    # Close the figure
    plt.close()

    ##########################################################################
    ##########################################################################





def PCA_physicalProp_KDE_MODE_PLOT(df_sim, df_obs, df_sel, n_PC_in_PCA, fit_funct, mag_noise_real, len_noise_real, Metsim_folderfile_json='', file_name_obs='', folder_file_name_real='', output_dir='', total_distribution=False, save_log=False):
    print('PCA_physicalProp_KDE_MODE_PLOT')
    output_dir_OG=output_dir

    pd_datafram_PCA_selected_mode_min_KDE=pd.DataFrame()

    # sigma5=5

    # 5 sigma confidence interval
    # five_sigma=False
    # mag_noise = MAG_RMSD*SIGMA_ERR
    # len_noise = LEN_RMSD*SIGMA_ERR
    mag_noise = mag_noise_real.copy()
    len_noise = len_noise_real.copy()

    # # Standard deviation of the magnitude Gaussian noise 1 sigma
    # # SD of noise in length (m) 1 sigma in km
    len_noise= len_noise/1000
    # velocity noise 1 sigma km/s
    # vel_noise = (len_noise*np.sqrt(2)/(1/FPS))
    vel_noise = (len_noise/(1/FPS))

    # check if end with pickle
    if folder_file_name_real.endswith('.pickle'):
        data_file_real = read_pickle_reduction_file(folder_file_name_real)
    elif folder_file_name_real.endswith('.json'):
        data_file_real = read_with_noise_GenerateSimulations_output(folder_file_name_real)

    _, _, _, residuals_mag_real, residuals_vel_real, _, residual_time_pos_real, residual_height_pos_real = RMSD_calc_diff(data_file_real, fit_funct)

    if total_distribution:
        df_sel['solution_id_dist'] = df_obs['solution_id'].iloc[0]
        df_obs=df_obs.iloc[[0]]

    # Get the default color cycle
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Create an infinite cycle of colors
    infinite_color_cycle = itertools.cycle(color_cycle)

    for jj in range(len(df_obs)):

        fig, ax = plt.subplots(2, 3, figsize=(14, 6),gridspec_kw={'height_ratios': [ 3, 0.5],'width_ratios': [ 3, 0.5, 3]})
        # fig, ax = plt.subplots(2, 4)
        # flat the ax
        ax = ax.flatten()
        
        around_meteor=df_obs.iloc[jj]['solution_id']
        curr_sel = df_sel[df_sel['solution_id_dist'] == around_meteor]
        curr_sel['erosion_coeff']=curr_sel['erosion_coeff']*1000000
        curr_sel['sigma']=curr_sel['sigma']*1000000

        # check if around_meteor is a file in a folder
        is_real=False
        if os.path.exists(around_meteor):
            is_real=True
            # split in file and directory
            _, around_meteor = os.path.split(around_meteor)
            around_meteor = around_meteor[:15]

        if total_distribution==False:
            output_dir=output_dir_OG+os.sep+SAVE_SELECTION_FOLDER+os.sep+around_meteor

        plot_side_by_side(data_file_real, fig, ax, 'go', file_name_obs[:15]+'\nRMSDmag '+str(round(mag_noise_real,3))+' RMSDlen '+str(round(len_noise_real/1000,3)), residuals_mag_real, residuals_vel_real, residual_time_pos_real, residual_height_pos_real, fit_funct, mag_noise, vel_noise,'Std.dev. realizations')

        densest_point = ''

        print('Number of selected events:',len(curr_sel))

        if len(curr_sel)<2:
            print('Check if the event is below RMSD')
            ii=0
            Metsim_flag=False
            try:
                namefile_sel = curr_sel['solution_id'].iloc[ii]
            except IndexError:
                # Handle the error
                print(f"Index {ii} is out of bounds for 'solution_id' in curr_sel.")
                namefile_sel = None
                continue
            # namefile_sel = curr_sel['solution_id'].iloc[ii]
            
            # chec if the file exist
            if not os.path.isfile(namefile_sel):
                print('file '+namefile_sel+' not found')
                continue

            else:
                if namefile_sel.endswith('.pickle'):
                    data_file = read_pickle_reduction_file(namefile_sel)
                    pd_datafram_PCA_sim = array_to_pd_dataframe_PCA(data_file)

                elif namefile_sel.endswith('.json'):
                    # open the json file with the name namefile_sel 
                    f = open(namefile_sel,"r")
                    data = json.loads(f.read())
                    if 'ht_sampled' in data:
                        data_file = read_GenerateSimulations_output(namefile_sel, data_file_real)
                        pd_datafram_PCA_sim = array_to_pd_dataframe_PCA(data_file)

                    else:
                        Metsim_flag=True
                        _, data_file, pd_datafram_PCA_sim = run_simulation(namefile_sel, data_file_real)
            
            rmsd_mag, rmsd_vel, rmsd_lag, residuals_mag, residuals_vel, residuals_len, residual_time_pos, residual_height_pos = RMSD_calc_diff(data_file, fit_funct)
            
            color_line=next(infinite_color_cycle)

            if Metsim_flag:
                
                # plot_side_by_side(data_file, fig, ax, '-k', ii, residuals_mag, residuals_vel, residual_time_pos, residual_height_pos)
                
                plot_side_by_side(data_file, fig, ax, '-k', 'Metsim data event\n\
RMSDmag '+str(round(rmsd_mag,3))+' RMSDlen '+str(round(rmsd_lag,3))+'\n\
        m:'+str('{:.2e}'.format(curr_sel.iloc[ii]['mass'],1))+' F:'+str(round(curr_sel.iloc[ii]['F'],2))+'\n\
        rho:'+str(round(curr_sel.iloc[ii]['rho']))+' sigma:'+str(round(curr_sel.iloc[ii]['sigma'],4))+'\n\
        er.height:'+str(round(curr_sel.iloc[ii]['erosion_height_start'],2))+' er.log:'+str(round(curr_sel.iloc[ii]['erosion_range'],1))+'\n\
        er.coeff:'+str(round(curr_sel.iloc[ii]['erosion_coeff'],3))+' er.index:'+str(round(curr_sel.iloc[ii]['erosion_mass_index'],2)), residuals_mag, residuals_vel, residual_time_pos, residual_height_pos)
            
            
                                                                            
            else:

                plot_side_by_side(data_file, fig, ax, '-','RMSDmag '+str(round(rmsd_mag,3))+' RMSDlen '+str(round(rmsd_lag,3))+' \n\
        m:'+str('{:.2e}'.format(curr_sel.iloc[ii]['mass'],1))+' F:'+str(round(curr_sel.iloc[ii]['F'],2))+'\n\
        rho:'+str(round(curr_sel.iloc[ii]['rho']))+' sigma:'+str(round(curr_sel.iloc[ii]['sigma'],4))+'\n\
        er.height:'+str(round(curr_sel.iloc[ii]['erosion_height_start'],2))+' er.log:'+str(round(curr_sel.iloc[ii]['erosion_range'],1))+'\n\
        er.coeff:'+str(round(curr_sel.iloc[ii]['erosion_coeff'],3))+' er.index:'+str(round(curr_sel.iloc[ii]['erosion_mass_index'],2)), residuals_mag, residuals_vel, residual_time_pos, residual_height_pos)

                # change first line color
                ax[0].lines[1].set_color(color_line)
                ax[1].lines[1].set_color(color_line)
                ax[2].lines[1].set_color(color_line)
                ax[5].lines[1].set_color(color_line)

            # pu the leggend putside the plot and adjust the plot base on the screen size
            ax[2].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', borderaxespad=0.)
            # the legend do not fit in the plot, so adjust the plot
            plt.subplots_adjust(right=.7)
            plt.subplots_adjust(wspace=0.2)

            # make more space
            plt.tight_layout()

            # split in file and directory
            _, name_file = os.path.split(curr_sel['solution_id'].iloc[ii])
            if rmsd_mag<MAG_RMSD*SIGMA_ERR and rmsd_lag<LEN_RMSD*SIGMA_ERR:
                shutil.copy(curr_sel['solution_id'].iloc[ii], output_dir_OG+os.sep+SAVE_RESULTS_FOLDER_EVENTS_PLOTS+os.sep+name_file)
                fig.suptitle(name_file+' SELECTED')
                # delete the extension of the file
                name_file = name_file.split('.')[0]
                plt.tight_layout()
                # put a sup title with the name of the file and write selected
                plt.savefig(output_dir_OG+os.sep+SAVE_RESULTS_FOLDER_EVENTS_PLOTS+os.sep+name_file+'_Heigh_MagVelCoef.png')

                pd_datafram_PCA_selected_mode_min_KDE = pd.concat([pd_datafram_PCA_selected_mode_min_KDE, pd_datafram_PCA_sim], axis=0)
            else:   
                fig.suptitle(name_file+' NOT SELECTED')

            plt.savefig(output_dir+os.sep+file_name_obs+'_'+around_meteor+'_Heigh_MagVelCoef.png')

            PCA_PhysicalPropPLOT(curr_sel, df_sim, n_PC_in_PCA, output_dir, file_name_obs, densest_point, save_log)

            # return pd_datafram_PCA_selected_mode_min_KDE    
                            
        else:

            print('compute the MODE and KDE for the selected meteors')
            var_kde = ['mass', 'rho', 'sigma', 'erosion_height_start', 'erosion_coeff', 'erosion_mass_index', 'erosion_mass_min', 'erosion_mass_max']

            # create the dataframe with the selected variable
            curr_sel_data = curr_sel[var_kde].values

            if len(curr_sel) > 8:
                try:

                    # def density_function(x):
                    #     # Insert the logic of your objective function here
                    #     # This example uses a simple sum of squares of x
                    #     # Replace it with the actual function you want to minimize
                    #     return np.sum(np.square(x))
                    
                    # # Objective function for maximization (negative density for minimization)
                    # def objective_function(x):
                    #     return -density_function(x)
                    
                    # # Bounds for optimization within all the sim space
                    # bounds = [(np.min(curr_sel_data[:, i]), np.max(curr_sel_data[:, i])) for i in range(curr_sel_data.shape[1])]

                    # # Perform global optimization using differential evolution
                    # print('Starting global optimization using differential evolution.')
                    # result = differential_evolution(objective_function, bounds)

                    # if result.success:
                    #     densest_point = result.x
                    #     print(f"Densest point found using differential evolution:\n {densest_point}")
                    # else:
                    #     print('Optimization was unsuccessful.')
                    #     densest_point = ''

                    kde = gaussian_kde(dataset=curr_sel_data.T)  # Note the transpose to match the expected input shape

                    # Negative of the KDE function for optimization
                    def neg_density(x):
                        return -kde(x)

                    # Bounds for optimization within all the sim space
                    # data_sim = df_sim[var_kde].values
                    bounds = [(np.min(curr_sel_data[:, i]), np.max(curr_sel_data[:, i])) for i in range(curr_sel_data.shape[1])]

                    # Initial guesses: curr_sel_data mean, curr_sel_data median, and KMeans centroids
                    mean_guess = np.mean(curr_sel_data, axis=0)
                    median_guess = np.median(curr_sel_data, axis=0)

                    # KMeans centroids as additional guesses
                    kmeans = KMeans(n_clusters=5, n_init='auto').fit(curr_sel_data)  # Adjust n_clusters based on your understanding of the curr_sel_data
                    centroids = kmeans.cluster_centers_

                    # Combine all initial guesses
                    initial_guesses = [mean_guess, median_guess] + centroids.tolist()

                    # Perform optimization from each initial guess
                    results = [minimize(neg_density, x0, method='L-BFGS-B', bounds=bounds) for x0 in initial_guesses]

                    # Filter out unsuccessful optimizations and find the best result
                    successful_results = [res for res in results if res.success]

                    if successful_results:
                        best_result = min(successful_results, key=lambda x: x.fun)
                        densest_point = best_result.x
                        print("Densest point using KMeans centroid:\n", densest_point)
                    else:
                        # raise ValueError('Optimization was unsuccessful. Consider revising the strategy.')
                        print('Optimization was unsuccessful. Consider revising the strategy.')
                        # revise the optimization strategy
                        print('Primary optimization strategies were unsuccessful. Trying fallback strategy (Grid Search).')
                        # Fallback strategy: Grid Search
                        grid_size = 5  # Define the grid size for the search
                        grid_points = [np.linspace(bound[0], bound[1], grid_size) for bound in bounds]
                        grid_combinations = list(itertools.product(*grid_points))

                        best_grid_point = None
                        best_grid_density = -np.inf

                        for point in grid_combinations:
                            density = kde(point)
                            if density > best_grid_density:
                                best_grid_density = density
                                best_grid_point = point

                        if best_grid_point is not None:
                            densest_point = np.array(best_grid_point)
                            print("Densest point found using Grid Search:\n", densest_point)
                        else:
                            print("None of the strategy worked no KDE result, change the selected simulations")
                except np.linalg.LinAlgError as e:
                    print(f"LinAlgError: {str(e)}")
            else:
                print('Not enough data to perform the KDE need more than 8 meteors')
       
            # if pickle change the extension and the code ##################################################################################################
            if Metsim_folderfile_json != '':
                # Load the nominal simulation parameters
                const_nominal, _ = loadConstants(Metsim_folderfile_json)
            else:
                const_nominal, _ = loadConstants()

            const_nominal.dens_co = np.array(const_nominal.dens_co)

            dens_co=np.array(const_nominal.dens_co)

            # print(const_nominal.__dict__)

            ### Calculate atmosphere density coeffs (down to the bottom observed height, limit to 15 km) ###

            # Determine the height range for fitting the density
            dens_fit_ht_beg = const_nominal.h_init
            # dens_fit_ht_end = const_nominal.h_final

            # Assign the density coefficients
            const_nominal.dens_co = dens_co

            # Turn on plotting of LCs of individual fragments 
            const_nominal.fragmentation_show_individual_lcs = True

            # # change the sigma of the fragmentation
            # const_nominal.sigma = 1.0

            # 'rho': 209.27575861617834, 'm_init': 1.3339843905562902e-05, 'v_init': 59836.848805126894, 'shape_factor': 1.21, 'sigma': 1.387556841276162e-08, 'zenith_angle': 0.6944268835985749, 'gamma': 1.0, 'rho_grain': 3000, 'lum_eff_type': 5, 'lum_eff': 0.7, 'mu': 3.8180000000000003e-26, 'erosion_on': True, 'erosion_bins_per_10mass': 10, 'erosion_height_start': 117311.48011974395, 'erosion_coeff': 6.356639734390828e-07, 'erosion_height_change': 0, 'erosion_coeff_change': 3.3e-07, 'erosion_rho_change': 3700, 'erosion_sigma_change': 2.3e-08, 'erosion_mass_index': 1.614450928834309, 'erosion_mass_min': 4.773894502090459e-11, 'erosion_mass_max': 7.485333377052805e-10, 'disruption_on': False, 'compressive_strength': 2000, 

            # create a copy of the const_nominal
            const_nominal_1D_KDE = copy.deepcopy(const_nominal)
            const_nominal_allD_KDE = copy.deepcopy(const_nominal)

            var_cost=['m_init','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max']
            # print for each variable the kde
            percent_diff_1D=[]
            percent_diff_allD=[]
            for i in range(len(var_kde)):

                x=curr_sel[var_kde[i]]

                # Check if dataset has multiple elements
                if len(x) < 2:
                    # If dataset has fewer than 2 elements, duplicate the single element or skip
                    print(f"Dataset for {var_kde[i]} has less than 2 elements. Duplicating elements to compute KDE.")
                    x = np.concatenate([x, x])  # Duplicate elements to have at least two

                # Compute KDE
                kde = gaussian_kde(x)
                
                # Define the range for which you want to compute KDE values, with more points for higher accuracy
                kde_x = np.linspace(x.min(), x.max(), 1000)
                kde_values = kde(kde_x)
                
                # Find the mode (x-value where the KDE curve is at its maximum)
                mode_index = np.argmax(kde_values)
                mode = kde_x[mode_index]
                
                real_val=df_sim[var_kde[i]].iloc[0]

                print()
                if df_sim['type'].iloc[0]=='MetSim' or df_sim['type'].iloc[0]=='Real':
                    print(f"MetSim value {var_kde[i]}: {'{:.4g}'.format(real_val)}")
                    print(f"1D Mode of KDE for {var_kde[i]}: {'{:.4g}'.format(mode)} percent diff: {'{:.4g}'.format(abs((real_val-mode)/(real_val+mode))/2*100)}%")
                    percent_diff_1D.append(abs((real_val-mode)/(real_val+mode))/2*100)
                    if densest_point!='':
                        print(f"Mult.dim. KDE densest {var_kde[i]}:  {'{:.4g}'.format(densest_point[i])} percent diff: {'{:.4g}'.format(abs((real_val-densest_point[i])/(real_val+densest_point[i]))/2*100)}%")
                        percent_diff_allD.append(abs((real_val-densest_point[i])/(real_val+densest_point[i]))/2*100)
                # print the value of const_nominal
                # print(f"const_nominal {var_cost[i]}:  {'{:.4g}'.format(const_nominal.__dict__[var_cost[i]])}")

                if var_cost[i] == 'sigma' or var_cost[i] == 'erosion_coeff':
                    # put it back as it was
                    const_nominal_1D_KDE.__dict__[var_cost[i]]=mode/1000000
                    if densest_point!='':
                        const_nominal_allD_KDE.__dict__[var_cost[i]]=densest_point[i]/1000000
                elif var_cost[i] == 'erosion_height_start':
                    # put it back as it was
                    const_nominal_1D_KDE.__dict__[var_cost[i]]=mode*1000
                    if densest_point!='':
                        const_nominal_allD_KDE.__dict__[var_cost[i]]=densest_point[i]*1000
                else:
                    # add each to const_nominal_1D_KDE and const_nominal_allD_KDE
                    const_nominal_1D_KDE.__dict__[var_cost[i]]=mode
                    if densest_point!='':
                        const_nominal_allD_KDE.__dict__[var_cost[i]]=densest_point[i]

            # check if the file output_folder+os.sep+file_name+'_sim_sel_optimized.csv' exists then read
            if os.path.exists(output_dir+os.sep+file_name_obs+'_sim_sel_optimized.csv'):
                df_sel_optimized_check = pd.read_csv(output_dir+os.sep+file_name_obs+'_sim_sel_optimized.csv')
            else:
                df_sel_optimized_check = pd.DataFrame()
                df_sel_optimized_check['solution_id']=''
            
            # save the const_nominal as a json file saveConstants(const, dir_path, file_name):
            if total_distribution:
                if output_dir+os.sep+around_meteor+'_mode_TOT.json' not in df_sel_optimized_check['solution_id'].values:
                    saveConstants(const_nominal_1D_KDE,output_dir,around_meteor+'_mode_TOT.json')
                    _, gensim_data_sim, pd_datafram_PCA_sim = run_simulation(output_dir+os.sep+around_meteor+'_mode_TOT.json', data_file_real)
                else:
                    print('already optimized')
                    _, gensim_data_sim, pd_datafram_PCA_sim = run_simulation(output_dir+os.sep+around_meteor+'_mode_TOT.json', data_file_real)

            else:
                if output_dir+os.sep+around_meteor+'_mode.json' not in df_sel_optimized_check['solution_id'].values:
                    saveConstants(const_nominal_1D_KDE,output_dir,around_meteor+'_mode.json')
                    _, gensim_data_sim, pd_datafram_PCA_sim = run_simulation(output_dir+os.sep+around_meteor+'_mode.json', data_file_real)
                else:
                    print('already optimized')
                    _, gensim_data_sim, pd_datafram_PCA_sim = run_simulation(output_dir+os.sep+around_meteor+'_mode.json', data_file_real)

            if pd_datafram_PCA_sim is None:
                return pd_datafram_PCA_selected_mode_min_KDE
            if gensim_data_sim is None:
                return pd_datafram_PCA_selected_mode_min_KDE

            rmsd_mag, rmsd_vel, rmsd_lag, residuals_mag, residuals_vel, residuals_len, residual_time_pos, residual_height_pos = RMSD_calc_diff(gensim_data_sim, fit_funct)

            plot_side_by_side(gensim_data_sim, fig, ax, 'r-', 'MODE : RMSDmag '+str(round(rmsd_mag,3))+' RMSDlen '+str(round(rmsd_lag,3))+'\n\
        m:'+str('{:.2e}'.format(pd_datafram_PCA_sim.iloc[0]['mass'],1))+' F:'+str(round(pd_datafram_PCA_sim.iloc[0]['F'],2))+'\n\
        rho:'+str(round(pd_datafram_PCA_sim.iloc[0]['rho']))+' sigma:'+str(round(pd_datafram_PCA_sim.iloc[0]['sigma']*1000000,4))+'\n\
        er.height:'+str(round(pd_datafram_PCA_sim.iloc[0]['erosion_height_start'],2))+' er.log:'+str(round(pd_datafram_PCA_sim.iloc[0]['erosion_range'],1))+'\n\
        er.coeff:'+str(round(pd_datafram_PCA_sim.iloc[0]['erosion_coeff']*1000000,3))+' er.index:'+str(round(pd_datafram_PCA_sim.iloc[0]['erosion_mass_index'],2)), residuals_mag, residuals_vel, residual_time_pos, residual_height_pos)

            # pd_datafram_PCA_sim['erosion_coeff']=pd_datafram_PCA_sim['erosion_coeff']/1000000
            # pd_datafram_PCA_sim['sigma']=pd_datafram_PCA_sim['sigma']/1000000

            print('real noise mag', round(mag_noise_real,3),''+str(SIGMA_ERR)+'sig',round(MAG_RMSD*SIGMA_ERR,3),''+str(SIGMA_ERR*2)+'sig',round(MAG_RMSD*SIGMA_ERR*2,3),'|| MODE noise mag', round(rmsd_mag,3), '\nreal noise len', round(len_noise_real/1000,3),''+str(SIGMA_ERR)+'sig',round(LEN_RMSD*SIGMA_ERR,3),''+str(SIGMA_ERR*2)+'sig',round(LEN_RMSD*SIGMA_ERR*2,3),'|| MODE noise len', round(rmsd_lag,3))
            select_mode_print='No'
            if rmsd_mag<MAG_RMSD*SIGMA_ERR and rmsd_lag<LEN_RMSD*SIGMA_ERR:
                select_mode_print='Yes'
                print('below 5 sigma noise, SAVED')
                pd_datafram_PCA_selected_mode_min_KDE = pd.concat([pd_datafram_PCA_selected_mode_min_KDE, pd_datafram_PCA_sim], axis=0)

                if total_distribution:
                    shutil.copy(output_dir+os.sep+around_meteor+'_mode_TOT.json', output_dir_OG+os.sep+SAVE_RESULTS_FOLDER_EVENTS_PLOTS+os.sep+around_meteor+'_mode_TOT.json')
                else:
                    shutil.copy(output_dir+os.sep+around_meteor+'_mode.json', output_dir_OG+os.sep+SAVE_RESULTS_FOLDER_EVENTS_PLOTS+os.sep+around_meteor+'_mode.json')
                fig.suptitle(around_meteor+' '+select_mode_print+' mode selected') # , fontsize=16
                # pu the leggend putside the plot and adjust the plot base on the screen size
                ax[2].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', borderaxespad=0.)
                # the legend do not fit in the plot, so adjust the plot
                plt.subplots_adjust(right=.7)
                plt.subplots_adjust(wspace=0.2)
                # make more space
                plt.tight_layout()
                if is_real:
                    plt.savefig(output_dir_OG+os.sep+SAVE_RESULTS_FOLDER_EVENTS_PLOTS+os.sep+around_meteor+'_MODE_DensPoint_Heigh_MagVelCoef.png')
                else:
                    plt.savefig(output_dir_OG+os.sep+SAVE_RESULTS_FOLDER_EVENTS_PLOTS+os.sep+file_name_obs+'_'+around_meteor+'_MODE_DensPoint_Heigh_MagVelCoef.png')

            fig.suptitle(around_meteor+' '+select_mode_print+' mode selected') # , fontsize=16

            if densest_point!='':

                if total_distribution:
                    if output_dir+os.sep+around_meteor+'_DensPoint_TOT.json' not in df_sel_optimized_check['solution_id'].values:
                        saveConstants(const_nominal_allD_KDE,output_dir,around_meteor+'_DensPoint_TOT.json')
                        _, gensim_data_sim, pd_datafram_PCA_sim = run_simulation(output_dir+os.sep+around_meteor+'_DensPoint_TOT.json', data_file_real)
                    else:
                        print('already optimized')
                        _, gensim_data_sim, pd_datafram_PCA_sim = run_simulation(output_dir+os.sep+around_meteor+'_DensPoint_TOT.json', data_file_real)
                else:
                    if output_dir+os.sep+around_meteor+'_mode.json' not in df_sel_optimized_check['solution_id'].values:
                        saveConstants(const_nominal_allD_KDE,output_dir,around_meteor+'_DensPoint.json')
                        _, gensim_data_sim, pd_datafram_PCA_sim = run_simulation(output_dir+os.sep+around_meteor+'_DensPoint.json', data_file_real)
                    else:
                        print('already optimized')
                        _, gensim_data_sim, pd_datafram_PCA_sim = run_simulation(output_dir+os.sep+around_meteor+'_DensPoint.json', data_file_real)

                # save the const_nominal as a json file saveConstants(const, dir_path, file_name):
                # saveConstants(const_nominal_allD_KDE,output_dir_OG,file_name_obs+'_sim_fit.json')
                # check if pd_datafram_PCA_sim is empty
                if pd_datafram_PCA_sim is None:
                    return pd_datafram_PCA_selected_mode_min_KDE
                if gensim_data_sim is None:
                    return pd_datafram_PCA_selected_mode_min_KDE

                # _, gensim_data_sim, pd_datafram_PCA_sim = run_simulation(output_dir_OG+os.sep+file_name_obs+'_sim_fit.json', data_file_real)

                rmsd_mag, rmsd_vel, rmsd_lag, residuals_mag, residuals_vel, residuals_len, residual_time_pos, residual_height_pos = RMSD_calc_diff(gensim_data_sim, fit_funct)
                print('real noise mag', round(mag_noise_real,3),''+str(SIGMA_ERR)+'sig',round(MAG_RMSD*SIGMA_ERR,3),''+str(SIGMA_ERR*2)+'sig',round(MAG_RMSD*SIGMA_ERR*2,3),'|| Dens.point noise mag', round(rmsd_mag,3), '\nreal noise len', round(len_noise_real/1000,3),''+str(SIGMA_ERR)+'sig',round(LEN_RMSD*SIGMA_ERR,3),''+str(SIGMA_ERR*2)+'sig',round(LEN_RMSD*MAG_RMSD*2,3),'|| Dens.point noise len', round(rmsd_lag,3))
            
                plot_side_by_side(gensim_data_sim, fig, ax, 'b-', 'Dens.point : RMSDmag '+str(round(rmsd_mag,3))+' RMSDlen '+str(round(rmsd_lag,3))+'\n\
        m:'+str('{:.2e}'.format(pd_datafram_PCA_sim.iloc[0]['mass'],1))+' F:'+str(round(pd_datafram_PCA_sim.iloc[0]['F'],2))+'\n\
        rho:'+str(round(pd_datafram_PCA_sim.iloc[0]['rho']))+' sigma:'+str(round(pd_datafram_PCA_sim.iloc[0]['sigma']*1000000,4))+'\n\
        er.height:'+str(round(pd_datafram_PCA_sim.iloc[0]['erosion_height_start'],2))+' er.log:'+str(round(pd_datafram_PCA_sim.iloc[0]['erosion_range'],1))+'\n\
        er.coeff:'+str(round(pd_datafram_PCA_sim.iloc[0]['erosion_coeff']*1000000,3))+' er.index:'+str(round(pd_datafram_PCA_sim.iloc[0]['erosion_mass_index'],2)), residuals_mag, residuals_vel, residual_time_pos, residual_height_pos)

                # pd_datafram_PCA_sim['erosion_coeff']=pd_datafram_PCA_sim['erosion_coeff']/1000000
                # pd_datafram_PCA_sim['sigma']=pd_datafram_PCA_sim['sigma']/1000000
                select_kde_print='No'
                if rmsd_mag<MAG_RMSD*SIGMA_ERR and rmsd_lag<LEN_RMSD*SIGMA_ERR:
                    select_kde_print='Yes'
                    print('below',SIGMA_ERR,'sigma noise, SAVED')
                    pd_datafram_PCA_selected_mode_min_KDE = pd.concat([pd_datafram_PCA_selected_mode_min_KDE, pd_datafram_PCA_sim], axis=0)

                    if total_distribution:
                        # shuty copy the file
                        shutil.copy(output_dir+os.sep+around_meteor+'_DensPoint_TOT.json', output_dir_OG+os.sep+SAVE_RESULTS_FOLDER_EVENTS_PLOTS+os.sep+around_meteor+'_DensPoint_TOT.json')
                    else:
                        shutil.copy(output_dir+os.sep+around_meteor+'_DensPoint.json', output_dir_OG+os.sep+SAVE_RESULTS_FOLDER_EVENTS_PLOTS+os.sep+around_meteor+'_DensPoint.json')
                    fig.suptitle(around_meteor+' '+select_mode_print+' mode selected and '+select_kde_print+' densest point selected') # , fontsize=16
                    # pu the leggend putside the plot and adjust the plot base on the screen size
                    ax[2].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', borderaxespad=0.)
                    # the legend do not fit in the plot, so adjust the plot
                    plt.subplots_adjust(right=.7)
                    plt.subplots_adjust(wspace=0.2)
                    # make more space
                    plt.tight_layout()
                    if is_real:
                        plt.savefig(output_dir_OG+os.sep+SAVE_RESULTS_FOLDER_EVENTS_PLOTS+os.sep+around_meteor+'_MODE_DensPoint_Heigh_MagVelCoef.png')
                    else:
                        plt.savefig(output_dir_OG+os.sep+SAVE_RESULTS_FOLDER_EVENTS_PLOTS+os.sep+file_name_obs+'_'+around_meteor+'_MODE_DensPoint_Heigh_MagVelCoef.png')

                # put a sup title
                fig.suptitle(around_meteor+' '+select_mode_print+' mode selected and '+select_kde_print+' densest point selected') # , fontsize=16
            # pu the leggend putside the plot and adjust the plot base on the screen size
            ax[2].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', borderaxespad=0.)
            # the legend do not fit in the plot, so adjust the plot
            plt.subplots_adjust(right=.7)
            plt.subplots_adjust(wspace=0.2)

            # make more space
            plt.tight_layout()

            if is_real:
                plt.savefig(output_dir+os.sep+around_meteor+'_MODE_DensPoint_Heigh_MagVelCoef.png')
            else:
                plt.savefig(output_dir+os.sep+file_name_obs+'_'+around_meteor+'_MODE_DensPoint_Heigh_MagVelCoef.png')

            # close the plot
            plt.close()

            curr_sel['erosion_coeff']=curr_sel['erosion_coeff']/1000000
            curr_sel['sigma']=curr_sel['sigma']/1000000

            PCA_PhysicalPropPLOT(curr_sel, df_sim, n_PC_in_PCA, output_dir, file_name_obs, densest_point, save_log)

    return pd_datafram_PCA_selected_mode_min_KDE
            



def RMSD_calc_diff_old(data_file, fit_funct):

    # from list to array
    height_km_err=np.array(fit_funct['height'])/1000
    abs_mag_sim_err=np.array(fit_funct['absolute_magnitudes'])
    obs_time_err=np.array(fit_funct['time'])
    vel_kms_err=np.array(fit_funct['velocities'])/1000  
    lag_kms_err=np.array(fit_funct['lag'])/1000
    
    # from list to array
    height_km=np.array(data_file['height'])/1000
    abs_mag_sim=np.array(data_file['absolute_magnitudes'])
    obs_time=np.array(data_file['time'])
    vel_kms=np.array(data_file['velocities'])/1000
    lag_residual = np.array(data_file['lag'])/1000
    residual_time_pos = np.array(data_file['time'])
    residual_height_pos = height_km.copy()

    # find the closest index with find_closest_index of height_km and height_km_err
    # find the closest index with find_closest_index of height_km and height_km_err
    index_err_RMSD = find_closest_index(height_km, height_km_err) # height_km, height_km_err)
    # find the difference between the two arrays
    residuals_mag = (abs_mag_sim_err-abs_mag_sim[index_err_RMSD])

    index_err_RMSD = find_closest_index(obs_time, obs_time_err) # height_km, height_km_err)
    residuals_vel = (vel_kms_err-vel_kms[index_err_RMSD])
    residuals_len = (lag_kms_err-lag_residual[index_err_RMSD])

    residual_time_pos = obs_time_err
    residual_height_pos = height_km_err

    # calculate the RMSD
    rmsd_mag = np.sqrt(np.mean(residuals_mag**2))
    rmsd_vel = np.sqrt(np.mean(residuals_vel**2))
    rmsd_lag = np.sqrt(np.mean(residuals_len**2))

    return rmsd_mag, rmsd_vel, rmsd_lag, residuals_mag, residuals_vel, residuals_len, residual_time_pos, residual_height_pos



def RMSD_calc_diff(data_file, fit_funct):

    # Check if data_file and fit_funct are not None
    if data_file is None or fit_funct is None:
        print('Error: data_file or fit_funct is None')
        return 9999,9999,9999,9999,9999,9999,0, 100

    # Check if required keys are present in data_file and fit_funct
    required_keys = ['height', 'absolute_magnitudes', 'time', 'velocities', 'lag']
    for key in required_keys:
        if key not in data_file or key not in fit_funct:
            print(f'Error: Missing key {key} in data_file or fit_funct')
            return 9999,9999,9999,9999,9999,9999,0, 100

    # from list to array
    height_km_err = np.array(fit_funct['height']) / 1000
    abs_mag_sim_err = np.array(fit_funct['absolute_magnitudes'])
    obs_time_err = np.array(fit_funct['time'])
    vel_kms_err = np.array(fit_funct['velocities']) / 1000  
    lag_kms_err = np.array(fit_funct['lag']) / 1000
    
    # from list to array
    height_km = np.array(data_file['height']) / 1000
    abs_mag_sim = np.array(data_file['absolute_magnitudes'])
    obs_time = np.array(data_file['time'])
    vel_kms = np.array(data_file['velocities']) / 1000
    lag_residual = np.array(data_file['lag']) / 1000
    residual_time_pos = np.array(data_file['time'])
    residual_height_pos = height_km.copy()

    # Define the range of heights for interpolation
    common_height_min = max(min(height_km), min(height_km_err))
    common_height_max = min(max(height_km), max(height_km_err))

    if common_height_min > common_height_max: # handle the case where there is no overlap in height
        print('No overlap in height')
        return 9999,9999,9999,9999,9999,9999,obs_time_err[0], height_km_err[0]
    
    common_heights = np.linspace(common_height_min, common_height_max, num=len(height_km_err))  # Adjust the number of points as needed

    # Interpolate the magnitudes
    interp_magnitudes1 = interp1d(height_km, abs_mag_sim, kind='linear', fill_value="extrapolate")
    interp_magnitudes2 = interp1d(height_km_err, abs_mag_sim_err, kind='linear', fill_value="extrapolate")

    # Get magnitudes at the common heights
    magnitudes1_common = interp_magnitudes1(common_heights) 
    magnitudes2_common = interp_magnitudes2(common_heights)

    # Calculate the magnitude differences
    magnitude_differences = magnitudes1_common - magnitudes2_common

    # Calculate the RMSD for magnitudes
    rmsd_mag = np.sqrt(np.mean(magnitude_differences**2))

    # # Determine the fraction of matching points for magnitudes
    # total_possible_points_mag = len(common_heights)
    # matching_points_mag = np.sum((common_heights >= common_height_min) & (common_heights <= common_height_max))
    # fraction_matching_mag = matching_points_mag / total_possible_points_mag

    # # Apply a penalty to the RMSD for magnitudes based on the fraction of matching points
    # penalty_factor_mag = 1 / fraction_matching_mag if fraction_matching_mag > 0 else 9999
    # adjusted_rmsd_mag = rmsd_mag * penalty_factor_mag

    # Interpolate the velocities
    interp_velocities1 = interp1d(obs_time, vel_kms, kind='linear', fill_value="extrapolate")
    interp_velocities2 = interp1d(obs_time_err, vel_kms_err, kind='linear', fill_value="extrapolate")

    # Get velocities at the common times
    common_times_min = max(min(obs_time), min(obs_time_err))
    common_times_max = min(max(obs_time), max(obs_time_err))
    common_times = np.linspace(common_times_min, common_times_max, num=len(obs_time_err))
    velocities1_common = interp_velocities1(common_times)
    velocities2_common = interp_velocities2(common_times)

    # Calculate the velocity differences
    velocity_differences = velocities1_common - velocities2_common

    # Calculate the RMSD for velocities
    rmsd_vel = np.sqrt(np.mean(velocity_differences**2))

    # # Determine the fraction of matching points for velocities
    # total_possible_points_vel = len(common_times)
    # matching_points_vel = np.sum((common_times >= common_times_min) & (common_times <= common_times_max))
    # fraction_matching_vel = matching_points_vel / total_possible_points_vel

    # # Apply a penalty to the RMSD for velocities based on the fraction of matching points
    # penalty_factor_vel = 1 / fraction_matching_vel if fraction_matching_vel > 0 else 9999
    # adjusted_rmsd_vel = rmsd_vel * penalty_factor_vel

    # Interpolate the lag residuals
    interp_lag1 = interp1d(obs_time, lag_residual, kind='linear', fill_value="extrapolate")
    interp_lag2 = interp1d(obs_time_err, lag_kms_err, kind='linear', fill_value="extrapolate")

    # Get lags at the common times
    lags1_common = interp_lag1(common_times)
    lags2_common = interp_lag2(common_times)

    # Calculate the lag differences
    lag_differences = lags1_common - lags2_common

    # Calculate the RMSD for lags
    rmsd_lag = np.sqrt(np.mean(lag_differences**2))

    # # Determine the fraction of matching points for lags
    # total_possible_points_lag = len(common_times)
    # matching_points_lag = np.sum((common_times >= min(obs_time)) & (common_times <= max(obs_time)))
    # fraction_matching_lag = matching_points_lag / total_possible_points_lag

    # # Apply a penalty to the RMSD for lags based on the fraction of matching points
    # penalty_factor_lag = 1 / fraction_matching_lag if fraction_matching_lag > 0 else 9999
    # adjusted_rmsd_lag = rmsd_lag * penalty_factor_lag

    residual_time_pos = common_times
    residual_height_pos = common_heights

    # if rmsd_mag is nan give 9999
    if np.isnan(rmsd_mag):
        rmsd_mag = 9999
    if np.isnan(rmsd_vel):
        rmsd_vel = 9999
    if np.isnan(rmsd_lag):
        rmsd_lag = 9999

    return rmsd_mag, rmsd_vel, rmsd_lag, magnitude_differences, velocity_differences, lag_differences, residual_time_pos, residual_height_pos



def PCA_LightCurveRMSDPLOT_optimize(df_sel_shower, df_obs_shower, output_dir, fit_funct='', gen_Metsim='', mag_noise_real = 0.1, len_noise_real = 20.0, file_name_obs='', number_event_to_optimize=0, run_optimization=True):

    # merge curr_sel and curr_obs
    curr_sel = df_sel_shower.copy()

    pd_datafram_PCA_selected_optimized=pd.DataFrame()

    # sigma5=5

    # 5 sigma confidence interval
    # five_sigma=False
    # mag_noise = MAG_RMSD*SIGMA_ERR
    # len_noise = LEN_RMSD*SIGMA_ERR
    mag_noise = mag_noise_real.copy()
    len_noise = len_noise_real.copy()

    # # Standard deviation of the magnitude Gaussian noise 1 sigma
    # # SD of noise in length (m) 1 sigma in km
    len_noise= len_noise/1000
    # velocity noise 1 sigma km/s
    # vel_noise = (len_noise*np.sqrt(2)/(1/FPS))
    vel_noise = (len_noise/(1/FPS))

    # # put the first plot in 2 sublots
    # fig, ax = plt.subplots(1, 2, figsize=(17, 5))

    # # group by solution_id_dist and keep only n_confront_sel from each group
    # curr_sel = curr_sel.groupby('solution_id_dist').head(len(number_event_to_optimize))
    # check if distance_meteor is in the columns
    no_distance_flag = False
    if 'distance_meteor' in curr_sel.columns:
        # order by distance_meteor
        curr_sel = curr_sel.sort_values('distance_meteor')
    else:
        no_distance_flag = True

    if number_event_to_optimize == 0:
        number_event_to_optimize = len(df_sel_shower)

    # pick from the first n_confront_sel
    curr_sel = curr_sel.head(number_event_to_optimize)

    # # count duplicates and add a column for the number of duplicates
    # curr_sel['num_duplicates'] = curr_sel.groupby('solution_id')['solution_id'].transform('size')

    # curr_sel.drop_duplicates(subset='solution_id', keep='first', inplace=True)

    curr_sel['erosion_coeff']=curr_sel['erosion_coeff']*1000000
    curr_sel['sigma']=curr_sel['sigma']*1000000
    
    # check if end with pickle
    if df_obs_shower.iloc[0]['solution_id'].endswith('.pickle'):
        data_file_real = read_pickle_reduction_file(df_obs_shower.iloc[0]['solution_id'])
    elif df_obs_shower.iloc[0]['solution_id'].endswith('.json'):
        data_file_real = read_with_noise_GenerateSimulations_output(df_obs_shower.iloc[0]['solution_id'])

    _, _, _, residuals_mag_real, residuals_vel_real, _, residual_time_pos_real, residual_height_pos_real = RMSD_calc_diff(data_file_real, fit_funct)

    # Get the default color cycle
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Create an infinite cycle of colors
    infinite_color_cycle = itertools.cycle(color_cycle)

    for ii in range(len(curr_sel)):

        fig, ax = plt.subplots(2, 3, figsize=(14, 6),gridspec_kw={'height_ratios': [ 3, 0.5],'width_ratios': [ 3, 0.5, 3]})
        # fig, ax = plt.subplots(2, 4)
        # flat the ax
        ax = ax.flatten()
        
        # pick the ii element of the solution_id column 
        namefile_sel=curr_sel.iloc[ii]['solution_id']
        Metsim_flag=False

        # chec if the file exist
        if not os.path.isfile(namefile_sel):
            print('file '+namefile_sel+' not found')
            continue
        else:
            if namefile_sel.endswith('.pickle'):
                data_file = read_pickle_reduction_file(namefile_sel)

            elif namefile_sel.endswith('.json'):
                # open the json file with the name namefile_sel 
                f = open(namefile_sel,"r")
                data = json.loads(f.read())
                if 'ht_sampled' in data:
                    data_file = read_GenerateSimulations_output(namefile_sel, data_file_real)

                else:
                    if gen_Metsim == '':
                        print('no data for the Metsim file')
                        continue

                    else:
                        # make a copy of gen_Metsim
                        data_file = gen_Metsim.copy()
                        # file metsim
                        Metsim_flag=True

            rmsd_mag, rmsd_vel, rmsd_lag, residuals_mag, residuals_vel, residuals_len, residual_time_pos, residual_height_pos = RMSD_calc_diff(data_file, fit_funct)

        print('real noise mag', round(mag_noise_real,3),''+str(SIGMA_ERR)+'sig',round(MAG_RMSD*SIGMA_ERR,3),''+str(SIGMA_ERR*2)+'sig',round(MAG_RMSD*SIGMA_ERR*2,3),'|| Event noise mag', round(rmsd_mag,3), '\nreal noise len', round(len_noise_real/1000,3),''+str(SIGMA_ERR)+'sig',round(LEN_RMSD*SIGMA_ERR,3),''+str(SIGMA_ERR*2)+'sig',round(LEN_RMSD*MAG_RMSD*2,3),'|| Event noise len', round(rmsd_lag,3))
        plot_side_by_side(data_file_real, fig, ax, 'go', file_name_obs[:15]+'\nRMSDmag '+str(round(mag_noise_real,3))+' RMSDlen '+str(round(len_noise_real/1000,3)), residuals_mag_real, residuals_vel_real, residual_time_pos_real, residual_height_pos_real, fit_funct, mag_noise, vel_noise, 'Std.dev. realizations')
        
        color_line=next(infinite_color_cycle)

        if Metsim_flag:
            
            # plot_side_by_side(data_file, fig, ax, '-k', ii, residuals_mag, residuals_vel, residual_time_pos, residual_height_pos)
            if no_distance_flag:
                plot_side_by_side(data_file, fig, ax, '-k', 'Metsim data event\n\
RMSDmag '+str(round(rmsd_mag,3))+' RMSDlen '+str(round(rmsd_lag,3))+'\n\
    m:'+str('{:.2e}'.format(curr_sel.iloc[ii]['mass'],1))+' F:'+str(round(curr_sel.iloc[ii]['F'],2))+'\n\
    rho:'+str(round(curr_sel.iloc[ii]['rho']))+' sigma:'+str(round(curr_sel.iloc[ii]['sigma'],4))+'\n\
    er.height:'+str(round(curr_sel.iloc[ii]['erosion_height_start'],2))+' er.log:'+str(round(curr_sel.iloc[ii]['erosion_range'],1))+'\n\
    er.coeff:'+str(round(curr_sel.iloc[ii]['erosion_coeff'],3))+' er.index:'+str(round(curr_sel.iloc[ii]['erosion_mass_index'],2)), residuals_mag, residuals_vel, residual_time_pos, residual_height_pos)
            
            else:
                plot_side_by_side(data_file, fig, ax, '-k', 'Metsim data event\n\
RMSDmag '+str(round(rmsd_mag,3))+' RMSDlen '+str(round(rmsd_lag,3))+'\n\
    N°duplic. '+str(round(curr_sel.iloc[ii]['num_duplicates']))+' min dist:'+str(round(curr_sel.iloc[ii]['distance_meteor'],2))+'\n\
    m:'+str('{:.2e}'.format(curr_sel.iloc[ii]['mass'],1))+' F:'+str(round(curr_sel.iloc[ii]['F'],2))+'\n\
    rho:'+str(round(curr_sel.iloc[ii]['rho']))+' sigma:'+str(round(curr_sel.iloc[ii]['sigma'],4))+'\n\
    er.height:'+str(round(curr_sel.iloc[ii]['erosion_height_start'],2))+' er.log:'+str(round(curr_sel.iloc[ii]['erosion_range'],1))+'\n\
    er.coeff:'+str(round(curr_sel.iloc[ii]['erosion_coeff'],3))+' er.index:'+str(round(curr_sel.iloc[ii]['erosion_mass_index'],2)), residuals_mag, residuals_vel, residual_time_pos, residual_height_pos)
            
           
                                                                        
        else:

            # if color_line == '#2ca02c':
            #     color_line='m'

            # plot_side_by_side(data_file, fig, ax, '-', ii, residuals_mag, residuals_vel, residual_time_pos, residual_height_pos)

            if no_distance_flag:
                plot_side_by_side(data_file, fig, ax, '-','RMSDmag '+str(round(rmsd_mag,3))+' RMSDlen '+str(round(rmsd_lag,3))+'\n\
    m:'+str('{:.2e}'.format(curr_sel.iloc[ii]['mass'],1))+' F:'+str(round(curr_sel.iloc[ii]['F'],2))+'\n\
    rho:'+str(round(curr_sel.iloc[ii]['rho']))+' sigma:'+str(round(curr_sel.iloc[ii]['sigma'],4))+'\n\
    er.height:'+str(round(curr_sel.iloc[ii]['erosion_height_start'],2))+' er.log:'+str(round(curr_sel.iloc[ii]['erosion_range'],1))+'\n\
    er.coeff:'+str(round(curr_sel.iloc[ii]['erosion_coeff'],3))+' er.index:'+str(round(curr_sel.iloc[ii]['erosion_mass_index'],2)), residuals_mag, residuals_vel, residual_time_pos, residual_height_pos)

            else:
                plot_side_by_side(data_file, fig, ax, '-','RMSDmag '+str(round(rmsd_mag,3))+' RMSDlen '+str(round(rmsd_lag,3))+'\n\
    N°duplic. '+str(round(curr_sel.iloc[ii]['num_duplicates']))+' min dist:'+str(round(curr_sel.iloc[ii]['distance_meteor'],2))+'\n\
    m:'+str('{:.2e}'.format(curr_sel.iloc[ii]['mass'],1))+' F:'+str(round(curr_sel.iloc[ii]['F'],2))+'\n\
    rho:'+str(round(curr_sel.iloc[ii]['rho']))+' sigma:'+str(round(curr_sel.iloc[ii]['sigma'],4))+'\n\
    er.height:'+str(round(curr_sel.iloc[ii]['erosion_height_start'],2))+' er.log:'+str(round(curr_sel.iloc[ii]['erosion_range'],1))+'\n\
    er.coeff:'+str(round(curr_sel.iloc[ii]['erosion_coeff'],3))+' er.index:'+str(round(curr_sel.iloc[ii]['erosion_mass_index'],2)), residuals_mag, residuals_vel, residual_time_pos, residual_height_pos)

            # change first line color
            ax[0].lines[1].set_color(color_line)
            ax[1].lines[1].set_color(color_line)
            ax[2].lines[1].set_color(color_line)
            ax[5].lines[1].set_color(color_line)

        # split the name from the path
        _, file_name_title = os.path.split(curr_sel.iloc[ii]['solution_id'])
        # suptitle of the plot
        fig.suptitle(file_name_title)
        
        # pu the leggend putside the plot and adjust the plot base on the screen size
        ax[2].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', borderaxespad=0.)
        # the legend do not fit in the plot, so adjust the plot
        plt.subplots_adjust(right=.7)
        plt.subplots_adjust(wspace=0.2)

        # make more space
        plt.tight_layout()

        file_json_save_phys_NOoptimized=output_dir+os.sep+SAVE_SELECTION_FOLDER+os.sep+file_name_title
        if Metsim_flag:
            file_json_save_phys=output_dir+os.sep+SAVE_SELECTION_FOLDER+os.sep+file_name_title[:23]+'_fitted.json'
            file_json_save_results=output_dir+os.sep+SAVE_RESULTS_FOLDER_EVENTS_PLOTS+os.sep+file_name_title[:23]+'_fitted.json'
            const_nominal, _ = loadConstants(namefile_sel)
            saveConstants(const_nominal,output_dir,file_name_obs+'_sim_fit.json')
        else:
            file_json_save_phys=output_dir+os.sep+SAVE_SELECTION_FOLDER+os.sep+file_name_obs[:15]+'_'+file_name_title[:23]+'_fitted.json'
            file_json_save_results=output_dir+os.sep+SAVE_RESULTS_FOLDER_EVENTS_PLOTS+os.sep+file_name_obs[:15]+'_'+file_name_title[:23]+'_fitted.json'
            # from namefile_sel json file open the json file and save the namefile_sel.const part as file_name_obs+'_sim_fit.json'
            with open(namefile_sel) as json_file:
                data = json.load(json_file)
                const_part = data['const']
                with open(output_dir+os.sep+file_name_obs+'_sim_fit.json', 'w') as outfile:
                    json.dump(const_part, outfile, indent=4)

        shutil.copy(namefile_sel, file_json_save_phys_NOoptimized)

        if run_optimization:

            # check if file_json_save_phys is present
            if not os.path.isfile(file_json_save_phys):

                if rmsd_mag<=mag_noise_real and rmsd_lag<=len_noise_real/1000:
                    print('below sigma noise, SAVED')

                    shutil.copy(output_dir+os.sep+file_name_obs+'_sim_fit.json', file_json_save_phys)

                    pd_datafram_PCA_selected_optimized = pd.concat([pd_datafram_PCA_selected_optimized, curr_sel.iloc[ii]], axis=0)

                    # suptitle of the plot
                    fig.suptitle(file_name_title+' PERFECT below sigma noise')

                    # pu the leggend putside the plot and adjust the plot base on the screen size
                    ax[2].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', borderaxespad=0.)
                    # the legend do not fit in the plot, so adjust the plot
                    plt.subplots_adjust(right=.7)
                    plt.subplots_adjust(wspace=0.2)
                    # make more space
                    plt.tight_layout()
                    plt.savefig(output_dir+os.sep+SAVE_RESULTS_FOLDER_EVENTS_PLOTS+os.sep+file_name_title[:23]+'_RMSDmag'+str(round(rmsd_mag,2))+'_RMSDlen'+str(round(rmsd_lag,2))+'_Heigh_MagVelCoef.png')
                    shutil.copy(output_dir+os.sep+file_name_obs+'_sim_fit_fitted.json', file_json_save_results)

                    plt.savefig(output_dir+os.sep+SAVE_SELECTION_FOLDER+os.sep+file_name_title[:23]+'_RMSDmag'+str(round(rmsd_mag,2))+'_RMSDlen'+str(round(rmsd_lag,2))+'_Heigh_MagVelCoef.png')

                    # close the plot
                    plt.close()
                    continue

                elif rmsd_mag<MAG_RMSD*SIGMA_ERR and rmsd_lag<LEN_RMSD*SIGMA_ERR:
                    print('below',SIGMA_ERR,'sigma noise, OPTIMIZED')

                    update_sigma_values(output_dir+os.sep+'AutoRefineFit_options.txt', mag_noise_real, len_noise_real, False, False) # More_complex_fit=False, Custom_refinement=False

                elif rmsd_mag<MAG_RMSD*SIGMA_ERR*2 and rmsd_lag<LEN_RMSD*SIGMA_ERR*2:
                    print('between 5-10 sigma noise, try major OPTIMIZATION and SAVE')

                    update_sigma_values(output_dir+os.sep+'AutoRefineFit_options.txt', mag_noise_real, len_noise_real, True, True) # More_complex_fit=False, Custom_refinement=False

                else:
                    print('above',SIGMA_ERR*2,'sigma noise, NO OPTIMIZATION and NOT SAVED')
                    
                    shutil.copy(output_dir+os.sep+file_name_obs+'_sim_fit.json', file_json_save_phys)

                    fig.suptitle(file_name_title+' BAD no optimization and no save')

                    plt.savefig(output_dir+os.sep+SAVE_SELECTION_FOLDER+os.sep+file_name_title[:23]+'_RMSDmag'+str(round(rmsd_mag,2))+'_RMSDlen'+str(round(rmsd_lag,2))+'_Heigh_MagVelCoef.png')

                    # close the plot
                    plt.close()
                    continue



                print('runing the optimization...')
                # this creates a ew file called output_dir+os.sep+file_name_obs+'_sim_fit_fitted.json'
                subprocess.run(
                    ['python', '-m', 'wmpl.MetSim.AutoRefineFit', 
                    output_dir, 'AutoRefineFit_options.txt', '-x'], 
                    # stdout=subprocess.PIPE, 
                    # stderr=subprocess.PIPE, 
                    text=True
                )

                # save the 20230811_082648_sim_fit_fitted.json as a json file in the output_dir+os.sep+SAVE_SELECTION_FOLDER+os.sep+file_name_title[:23]+'_sim_fit_fitted.json'
                shutil.copy(output_dir+os.sep+file_name_obs+'_sim_fit_fitted.json', file_json_save_phys)
            else:
                print('file '+file_json_save_phys+' already exist, read it...')

            _, gensim_data_optimized, pd_datafram_PCA_sim_optimized = run_simulation(file_json_save_phys, data_file_real)


            rmsd_mag, rmsd_vel, rmsd_lag, residuals_mag, residuals_vel, residuals_len, residual_time_pos, residual_height_pos = RMSD_calc_diff(gensim_data_optimized, fit_funct)

            print('real noise mag', round(mag_noise_real,3),''+str(SIGMA_ERR)+'sig',round(mag_noise_real*SIGMA_ERR,3),''+str(SIGMA_ERR*2)+'sig',round(MAG_RMSD*SIGMA_ERR*2,3),'|| Event noise mag', round(rmsd_mag,3), '\nreal noise len', round(len_noise_real/1000,3),''+str(SIGMA_ERR)+'sig',round(LEN_RMSD*SIGMA_ERR,3),''+str(SIGMA_ERR*2)+'sig',round(LEN_RMSD*SIGMA_ERR*2,3),'|| Event noise len', round(rmsd_lag,3))

            if Metsim_flag:
                
                plot_side_by_side(gensim_data_optimized, fig, ax, 'k--', 'Optimized MetSim RMSDmag '+str(round(rmsd_mag,3))+' RMSDlen '+str(round(rmsd_lag,3))+'\n\
    m:'+str('{:.2e}'.format(pd_datafram_PCA_sim_optimized.iloc[0]['mass'],1))+' F:'+str(round(pd_datafram_PCA_sim_optimized.iloc[0]['F'],2))+'\n\
    rho:'+str(round(pd_datafram_PCA_sim_optimized.iloc[0]['rho']))+' sigma:'+str(round(pd_datafram_PCA_sim_optimized.iloc[0]['sigma']*1000000,4))+'\n\
    er.height:'+str(round(pd_datafram_PCA_sim_optimized.iloc[0]['erosion_height_start'],2))+' er.log:'+str(round(pd_datafram_PCA_sim_optimized.iloc[0]['erosion_range'],1))+'\n\
    er.coeff:'+str(round(pd_datafram_PCA_sim_optimized.iloc[0]['erosion_coeff']*1000000,3))+' er.index:'+str(round(pd_datafram_PCA_sim_optimized.iloc[0]['erosion_mass_index'],2)), residuals_mag, residuals_vel, residual_time_pos, residual_height_pos)

            else:
                plot_side_by_side(gensim_data_optimized, fig, ax, '--', 'Optimized RMSDmag '+str(round(rmsd_mag,3))+' RMSDlen '+str(round(rmsd_lag,3))+'\n\
    m:'+str('{:.2e}'.format(pd_datafram_PCA_sim_optimized.iloc[0]['mass'],1))+' F:'+str(round(pd_datafram_PCA_sim_optimized.iloc[0]['F'],2))+'\n\
    rho:'+str(round(pd_datafram_PCA_sim_optimized.iloc[0]['rho']))+' sigma:'+str(round(pd_datafram_PCA_sim_optimized.iloc[0]['sigma']*1000000,4))+'\n\
    er.height:'+str(round(pd_datafram_PCA_sim_optimized.iloc[0]['erosion_height_start'],2))+' er.log:'+str(round(pd_datafram_PCA_sim_optimized.iloc[0]['erosion_range'],1))+'\n\
    er.coeff:'+str(round(pd_datafram_PCA_sim_optimized.iloc[0]['erosion_coeff']*1000000,3))+' er.index:'+str(round(pd_datafram_PCA_sim_optimized.iloc[0]['erosion_mass_index'],2)), residuals_mag, residuals_vel, residual_time_pos, residual_height_pos)

                # change first line color
                ax[0].lines[-1].set_color(color_line)
                ax[1].lines[-1].set_color(color_line)
                ax[2].lines[-1].set_color(color_line)
                ax[5].lines[-1].set_color(color_line)
            ax[0].lines[-1].set_marker("x")
            ax[1].lines[-1].set_marker("x")
            ax[2].lines[-1].set_marker("x")
            ax[5].lines[-1].set_marker("x")


        if rmsd_mag<MAG_RMSD*SIGMA_ERR and rmsd_lag<LEN_RMSD*SIGMA_ERR:
            print('below',SIGMA_ERR,'sigma noise, OPTIMIZED and SAVED')

            # output_folder+os.sep+SAVE_RESULTS_FOLDER_EVENTS_PLOTS
            if run_optimization:
                fig.suptitle(file_name_title+' optimized SELECTED')
            else:
                fig.suptitle(file_name_title+' simulation SELECTED')
            # pu the leggend putside the plot and adjust the plot base on the screen size
            ax[2].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', borderaxespad=0.)
            # the legend do not fit in the plot, so adjust the plot
            plt.subplots_adjust(right=.7)
            plt.subplots_adjust(wspace=0.2)
            # make more space
            plt.tight_layout()
            plt.savefig(output_dir+os.sep+SAVE_RESULTS_FOLDER_EVENTS_PLOTS+os.sep+file_name_title[:23]+'_RMSDmag'+str(round(rmsd_mag,2))+'_RMSDlen'+str(round(rmsd_lag,2))+'_Heigh_MagVelCoef.png')
            
            if run_optimization:
                # output_dir+os.sep+file_name_obs+'_sim_fit.json'
                shutil.copy(output_dir+os.sep+file_name_obs+'_sim_fit_fitted.json', file_json_save_results)
                pd_datafram_PCA_selected_optimized = pd.concat([pd_datafram_PCA_selected_optimized, pd_datafram_PCA_sim_optimized], axis=0)
            else:
                pd_datafram_PCA_sim = array_to_pd_dataframe_PCA(data_file)
                shutil.copy(file_json_save_phys_NOoptimized, output_dir+os.sep+SAVE_RESULTS_FOLDER_EVENTS_PLOTS+os.sep+file_name_title)
                # remove curr_sel.iloc[[ii]].drop(columns=['rmsd_mag', 'rmsd_len', 'solution_id_dist', 'distance_meteor', 'distance_mean']) rmsd_mag	rmsd_len solution_id_dist	distance_meteor	distance_mean
                pd_datafram_PCA_selected_optimized = pd.concat([pd_datafram_PCA_selected_optimized, pd_datafram_PCA_sim], axis=0)

        else:
            print('above',SIGMA_ERR,'sigma noise, OPTIMIZATION NOT SAVED')
            if run_optimization:
                fig.suptitle(file_name_title+' BAD optimization was not good enough')
            else:
                fig.suptitle(file_name_title+' NOT SELECTED simulation')

        # pu the leggend putside the plot and adjust the plot base on the screen size
        ax[2].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', borderaxespad=0.)
        # the legend do not fit in the plot, so adjust the plot
        plt.subplots_adjust(right=.7)
        plt.subplots_adjust(wspace=0.2)

        # make more space
        plt.tight_layout()

        plt.savefig(output_dir+os.sep+SAVE_SELECTION_FOLDER+os.sep+file_name_title[:23]+'_RMSDmag'+str(round(rmsd_mag,2))+'_RMSDlen'+str(round(rmsd_lag,2))+'_Heigh_MagVelCoef.png')

        # close the plot
        plt.close()

    # # save df_sel_shower_real to disk add the RMSD
    # pd_datafram_PCA_selected_optimized.to_csv(output_dir+os.sep+file_name_obs+'_sim_sel_optimized.csv', index=False)

    return pd_datafram_PCA_selected_optimized






def PCA_PhysicalPropPLOT(df_sel_shower_real, df_sim_shower, n_PC_in_PCA, output_dir, file_name, Min_KDE_point='', save_log=True):
    
    df_sim_shower_small=df_sim_shower.copy()

    df_sel_shower=df_sel_shower_real.copy()

    if len(df_sim_shower_small)>10000: # w/o takes forever to plot
        # pick randomly 10000 events
        df_sim_shower_small=df_sim_shower_small.sample(n=10000)
        if 'MetSim' not in df_sim_shower_small['type'].values and 'Real' not in df_sim_shower_small['type'].values:
            df_sim_shower_small = pd.concat([df_sim_shower_small.iloc[[0]], df_sim_shower_small])
        
    if save_log:
        # check if a file with the name "log"+n_PC_in_PCA+"_"+str(len(df_sel))+"ev.txt" already exist
        if os.path.exists(output_dir+os.sep+"log_"+file_name[:15]+"_CI"+str(n_PC_in_PCA)+"PC.txt"):
            # remove the file
            os.remove(output_dir+os.sep+"log_"+file_name[:15]+"_CI"+str(n_PC_in_PCA)+"PC.txt")
        sys.stdout = Logger(output_dir,"log_"+file_name[:15]+"_CI"+str(n_PC_in_PCA)+"PC.txt") # _30var_99%_13PC



    curr_df_sim_sel = pd.concat([df_sim_shower_small,df_sel_shower], axis=0)

    # multiply the erosion coeff by 1000000 to have it in km/s
    curr_df_sim_sel['erosion_coeff']=curr_df_sim_sel['erosion_coeff']*1000000
    curr_df_sim_sel['sigma']=curr_df_sim_sel['sigma']*1000000
    curr_df_sim_sel['erosion_energy_per_unit_cross_section']=curr_df_sim_sel['erosion_energy_per_unit_cross_section']/1000000
    curr_df_sim_sel['erosion_energy_per_unit_mass']=curr_df_sim_sel['erosion_energy_per_unit_mass']/1000000

    group_mapping = {
        'Simulation_sel': 'selected',
        'MetSim': 'simulated',
        'Real': 'simulated',
        'Simulation': 'simulated'
    }
    curr_df_sim_sel['group'] = curr_df_sim_sel['type'].map(group_mapping)

    curr_df_sim_sel['num_group'] = curr_df_sim_sel.groupby('group')['group'].transform('size')
    curr_df_sim_sel['weight'] = 1 / curr_df_sim_sel['num_group']

    curr_df_sim_sel['num_type'] = curr_df_sim_sel.groupby('type')['type'].transform('size')
    curr_df_sim_sel['weight_type'] = 1 / curr_df_sim_sel['num_type']

    curr_sel = curr_df_sim_sel[curr_df_sim_sel['group'] == 'selected'].copy()
    # curr_sim = curr_df_sim_sel[curr_df_sim_sel['group'] == 'simulated'].copy()

    # with color based on the shower but skip the first 2 columns (shower_code, shower_id)
    to_plot=['mass','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max','erosion_range','erosion_energy_per_unit_mass','erosion_energy_per_unit_cross_section','erosion_energy_per_unit_cross_section']
    # to_plot_unit=['mass [kg]','rho [kg/m^3]','sigma [s^2/km^2]','erosion height start [km]','erosion coeff [s^2/km^2]','erosion mass index [-]','log eros. mass min [kg]','log eros. mass max [kg]','log eros. mass range [-]','erosion energy per unit mass [MJ/kg]','erosion energy per unit cross section [MJ/m^2]','erosion energy per unit cross section [MJ/m^2]']
    to_plot_unit = [r'$m_0$ [kg]', r'$\rho$ [kg/m$^3$]', r'$\sigma$ [s$^2$/km$^2$]', r'$h_{e}$ [km]', r'$\eta$ [s$^2$/km$^2$]', r'$s$ [-]', r'log($m_{l}$) [-]', r'log($m_{u}$) [-]',r'log($m_{u}$)-log($m_{l}$) [-]']


    fig, axs = plt.subplots(3, 3)
    # from 2 numbers to one numbr for the subplot axs
    axs = axs.flatten()


    
    print('\\hline')
    if len(Min_KDE_point) > 0:    
        # print('var & $real$ & $1D_{KDE}$ & $1D_{KDE}\\%_{dif}$ & $allD_{KDE}$ & $allD_{KDE}\\%_{dif}$\\\\')
        # print('var & real & mode & min$_{KDE}$ & -1\\sigma/+1\\sigma & -2\\sigma/+2\\sigma \\\\')
        print('Variables & '+str(df_sim_shower['type'].iloc[0])+' & Mode & Dens.Point $ & 95\\%CIlow & 95\\%CIup \\\\')
    else:
        print('Variables & '+str(df_sim_shower['type'].iloc[0])+' & Mode & 95\\%CIlow & 95\\%CIup \\\\')

    ii_densest=0        
    for i in range(9):
        # put legendoutside north
        plotvar=to_plot[i]

        if plotvar == 'erosion_mass_min' or plotvar == 'erosion_mass_max':
            # take the log of the erosion_mass_min and erosion_mass_max
            curr_df_sim_sel[plotvar]=np.log10(curr_df_sim_sel[plotvar])
            curr_sel[plotvar]=np.log10(curr_sel[plotvar])
            if len(Min_KDE_point) > 0:
                Min_KDE_point[ii_densest]=np.log10(Min_KDE_point[ii_densest])
                # Min_KDE_point[ii_densest-1]=np.log10(Min_KDE_point[ii_densest-1])
        # sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar], weights=curr_df_sim_sel['weight'],hue='shower_code', ax=axs[i], kde=True, palette='bright', bins=20)
        sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar], weights=curr_df_sim_sel['weight'], hue='group', ax=axs[i], palette='bright', bins=20)
        unique_values_count = curr_sel[plotvar].nunique()
        if unique_values_count > 1:
            # # add the kde to the plot probability density function
            sns.histplot(curr_sel, x=curr_sel[plotvar], weights=curr_sel['weight'], bins=20, ax=axs[i], fill=False, edgecolor=False, color='r', kde=True, binrange=[np.min(curr_df_sim_sel[plotvar]),np.max(curr_df_sim_sel[plotvar])])
            kde_line = axs[i].lines[-1]
            axs[i].lines[-1].remove()
        else:
            kde_line = None

        # if the only_select_meteors_from is equal to any curr_df_sim_sel plot the observed event value as a vertical red line
        # check if curr_df_sim_sel['type']=='MetSim' is in the curr_df_sim_sel['type'].values
        if 'MetSim' in curr_df_sim_sel['type'].values:
            # get the value of the observed event
            axs[i].axvline(x=curr_df_sim_sel[curr_df_sim_sel['type']=='MetSim'][plotvar].values[0], color='k', linewidth=2)
        elif 'Real' in curr_df_sim_sel['type'].values:
            axs[i].axvline(x=curr_df_sim_sel[curr_df_sim_sel['type']=='Real'][plotvar].values[0], color='g', linewidth=2, linestyle='--')

        if plotvar == 'erosion_mass_min' or plotvar == 'erosion_mass_max':
            # put it back as it was
            curr_df_sim_sel[plotvar]=10**curr_df_sim_sel[plotvar]
            curr_sel[plotvar]=10**curr_sel[plotvar]    

        # get te 97.72nd percentile and the 2.28th percentile of curr_sel[plotvar] and call them sigma_97 and sigma_2
        sigma_95=np.percentile(curr_sel[plotvar], 95)
        sigma_84=np.percentile(curr_sel[plotvar], 84.13)
        sigma_15=np.percentile(curr_sel[plotvar], 15.87)
        sigma_5=np.percentile(curr_sel[plotvar], 5)

        if kde_line is not None:
            # Get the x and y data from the KDE line
            kde_line_Xval = kde_line.get_xdata()
            kde_line_Yval = kde_line.get_ydata()

            # Find the index of the maximum y value
            max_index = np.argmax(kde_line_Yval)
            if i!=8:
                # Plot a dot at the maximum point
                # axs[i].plot(kde_line_Xval[max_index], kde_line_Yval[max_index], 'ro')  # 'ro' for red dot
                axs[i].axvline(x=kde_line_Xval[max_index], color='red', linestyle='-.')

            x_10mode=kde_line_Xval[max_index]
            if plotvar == 'erosion_mass_min' or plotvar == 'erosion_mass_max':
                x_10mode=10**kde_line_Xval[max_index]

            if len(Min_KDE_point) > 0:     
                if len(Min_KDE_point)>ii_densest:                    

                    # Find the index with the closest value to densest_point[ii_dense] to all y values
                    densest_index = find_closest_index(kde_line_Xval, [Min_KDE_point[ii_densest]])

                    # add also the densest_point[i] as a blue dot
                    # axs[i].plot(Min_KDE_point[ii_densest], kde_line_Yval[densest_index[0]], 'bo')
                    axs[i].axvline(x=Min_KDE_point[ii_densest], color='blue', linestyle='-.')

                    if plotvar == 'erosion_mass_min' or plotvar == 'erosion_mass_max':
                        Min_KDE_point[ii_densest]=10**(Min_KDE_point[ii_densest])
                    
                    if i<9:
                        print('\\hline') #df_sel_save[df_sel_save['solution_id']==only_select_meteors_from][plotvar].values[0]
                        # print(f"{to_plot_unit[i]} & ${'{:.4g}'.format(df_sel_save[df_sel_save['solution_id']==only_select_meteors_from][plotvar].values[0])}$ & ${'{:.4g}'.format(x_10mode)}$ & $ {'{:.2g}'.format(percent_diff_1D[i])}$\\% & $ {'{:.4g}'.format(densest_point[i])}$ & $ {'{:.2g}'.format(percent_diff_allD[i])}$\\% \\\\")
                        # print(to_plot_unit[i]+'& $'+str(x[max_index])+'$ & $'+str(percent_diff_1D[i])+'$\\% & $'+str(densest_point[ii_densest])+'$ & $'+str(percent_diff_allD[i])+'\\% \\\\')
                        # print(f"{to_plot_unit[i]} & ${'{:.4g}'.format(df_sel_save[df_sel_save['solution_id']==only_select_meteors_from][plotvar].values[0])}$ & ${'{:.4g}'.format(x_10mode)}$ & $ {'{:.2g}'.format(percent_diff_1D[i])}$\\% & $ {'{:.4g}'.format(densest_point[i])}$ & $ {'{:.2g}'.format(percent_diff_allD[i])}$\\% \\\\")
                        # print(f"{to_plot_unit[i]} & {'{:.4g}'.format(df_sel_save[df_sel_save['solution_id']==only_select_meteors_from][plotvar].values[0])} & {'{:.4g}'.format(x_10mode)} & {'{:.4g}'.format(densest_point[i])} & {'{:.4g}'.format(sigma_15)} / {'{:.4g}'.format(sigma_84)} & {'{:.4g}'.format(sigma_2)} / {'{:.4g}'.format(sigma_97)} \\\\")
                        print(f"{to_plot_unit[i]} & {'{:.4g}'.format(curr_df_sim_sel[plotvar].iloc[0])} & {'{:.4g}'.format(x_10mode)} & {'{:.4g}'.format(Min_KDE_point[i])} & {'{:.4g}'.format(sigma_5)} & {'{:.4g}'.format(sigma_95)} \\\\")
                    ii_densest=ii_densest+1 
            else:
                if i<9:
                    print('\\hline') 
                    print(f"{to_plot_unit[i]} & {'{:.4g}'.format(curr_df_sim_sel[plotvar].iloc[0])} & {'{:.4g}'.format(x_10mode)} & {'{:.4g}'.format(sigma_5)} & {'{:.4g}'.format(sigma_95)} \\\\")
        else:   
            if i<9:
                print('\\hline') 
                print(f"{to_plot_unit[i]} & {'{:.4g}'.format(curr_df_sim_sel[plotvar].iloc[0])} & {'{:.4g}'.format(sigma_5)} & {'{:.4g}'.format(sigma_95)} \\\\")

        axs[i].set_ylabel('probability')
        axs[i].set_xlabel(to_plot_unit[i])

        # check if y axis is above 1 if so set_ylim(0,1)
        if axs[i].get_ylim()[1]>1:
            axs[i].set_ylim(0,1)
        
        # # plot the legend outside the plot
        # axs[i].legend()
        axs[i].get_legend().remove()
            

        if i==0:
            # place the xaxis exponent in the bottom right corner
            axs[i].xaxis.get_offset_text().set_x(1.10)

    # # more space between the subplots erosion_coeff sigma
    plt.tight_layout()

    print('\\hline')
    

    # save the figure maximized and with the right name
    fig.savefig(output_dir+os.sep+file_name+'_PhysicProp'+str(n_PC_in_PCA)+'PC_'+str(len(curr_sel))+'ev.png', dpi=300) # _dist'+str(np.round(np.min(curr_sel['distance_meteor']),2))+'-'+str(np.round(np.max(curr_sel['distance_meteor']),2))+'

    # close the figure
    plt.close()

    if save_log:
        # Close the Logger to ensure everything is written to the file STOP COPY in TXT file
        sys.stdout.close()

        # Reset sys.stdout to its original value if needed
        sys.stdout = sys.__stdout__

    ii_densest=0
    if 'solution_id_dist' in df_sel_shower_real.columns:
        # the plot can get suck if too many reliazations
        if len(df_sel_shower_real['solution_id_dist'].unique())<60:
            if len(df_sel_shower_real['solution_id_dist'].unique())>1:
                print('plot the distribution of the Realization',len(df_sel_shower_real['solution_id_dist'].unique()))
                fig, axs = plt.subplots(3, 3)
                # from 2 numbers to one numbr for the subplot axs
                axs = axs.flatten()

                # ii_densest=0        
                for i in range(9):
                    # put legendoutside north
                    plotvar=to_plot[i]

                    if plotvar == 'erosion_mass_min' or plotvar == 'erosion_mass_max':

                        sns.histplot(curr_df_sim_sel, x=np.log10(curr_df_sim_sel[plotvar]), weights=curr_df_sim_sel['weight'], hue='group', ax=axs[i], palette='bright', bins=20, binrange=[np.log10(np.min(curr_df_sim_sel[plotvar])),np.log10(np.max(curr_df_sim_sel[plotvar]))])
                        # sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar], weights=curr_df_sim_sel['weight'],hue='solution_id_dist', ax=axs[i], multiple="stack", kde=True, bins=20, binrange=[np.min(df_sel_save[plotvar]),np.max(df_sel_save[plotvar])])
                        sns.histplot(curr_df_sim_sel, x=np.log10(curr_df_sim_sel[plotvar]), weights=curr_df_sim_sel['weight'],hue='solution_id_dist', ax=axs[i], multiple="stack", bins=20, binrange=[np.log10(np.min(curr_df_sim_sel[plotvar])),np.log10(np.max(curr_df_sim_sel[plotvar]))])
                        # # add the kde to the plot as a probability density function
                        sns.histplot(curr_sel, x=np.log10(curr_sel[plotvar]), weights=curr_sel['weight'], bins=20, ax=axs[i],  multiple="stack", fill=False, edgecolor=False, color='r', kde=True, binrange=[np.log10(np.min(curr_df_sim_sel[plotvar])),np.log10(np.max(curr_df_sim_sel[plotvar]))])
                        
                        kde_line = axs[i].lines[-1]
                        # delete from the plot the axs[i].lines[-1]
                        axs[i].lines[-1].remove()
                        
                        # if the only_select_meteors_from is equal to any curr_df_sim_sel plot the observed event value as a vertical red line
                        if 'MetSim' in curr_df_sim_sel['type'].values:
                            # get the value of the observed event
                            axs[i].axvline(x=np.log10(curr_df_sim_sel[curr_df_sim_sel['type']=='MetSim'][plotvar].values[0]), color='k', linewidth=2)
                        elif 'Real' in curr_df_sim_sel['type'].values:
                            axs[i].axvline(x=np.log10(curr_df_sim_sel[curr_df_sim_sel['type']=='Real'][plotvar].values[0]), color='g', linewidth=2, linestyle='--')

                        # if len(Min_KDE_point) > 0:
                        #     Min_KDE_point[ii_densest]=np.log10(Min_KDE_point[ii_densest])
                        #     # Min_KDE_point[ii_densest-1]=np.log10(Min_KDE_point[ii_densest-1])
                    
                    else:

                        sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar], weights=curr_df_sim_sel['weight'], hue='group', ax=axs[i], palette='bright', bins=20, binrange=[np.min(curr_df_sim_sel[plotvar]),np.max(curr_df_sim_sel[plotvar])])
                        # sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar], weights=curr_df_sim_sel['weight'],hue='solution_id_dist', ax=axs[i], multiple="stack", kde=True, bins=20, binrange=[np.min(df_sel_save[plotvar]),np.max(df_sel_save[plotvar])])
                        sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar], weights=curr_df_sim_sel['weight'], hue='solution_id_dist', ax=axs[i], multiple="stack", bins=20, binrange=[np.min(curr_df_sim_sel[plotvar]),np.max(curr_df_sim_sel[plotvar])])
                        # # add the kde to the plot as a probability density function
                        sns.histplot(curr_sel, x=curr_sel[plotvar], weights=curr_sel['weight'], bins=20, ax=axs[i],  multiple="stack", fill=False, edgecolor=False, color='r', kde=True, binrange=[np.min(curr_df_sim_sel[plotvar]),np.max(curr_df_sim_sel[plotvar])])
                        
                        kde_line = axs[i].lines[-1]

                        # delete from the plot the axs[i].lines[-1]
                        axs[i].lines[-1].remove()

                        # if the only_select_meteors_from is equal to any curr_df_sim_sel plot the observed event value as a vertical red line
                        if 'MetSim' in curr_df_sim_sel['type'].values:
                            # get the value of the observed event
                            axs[i].axvline(x=curr_df_sim_sel[curr_df_sim_sel['type']=='MetSim'][plotvar].values[0], color='k', linewidth=2)
                        elif 'Real' in curr_df_sim_sel['type'].values:
                            axs[i].axvline(x=curr_df_sim_sel[curr_df_sim_sel['type']=='Real'][plotvar].values[0], color='g', linewidth=2, linestyle='--')
                        # put the value of diff_percent_1d at th upper left of the line

                    axs[i].set_ylabel('probability')
                    axs[i].set_xlabel(to_plot_unit[i])
                    # check if y axis is above 1 if so set_ylim(0,1)
                    if axs[i].get_ylim()[1]>1:
                        axs[i].set_ylim(0,1)
                    
                    # # plot the legend outside the plot
                    # axs[i].legend()
                    axs[i].get_legend().remove()

                    # # Get the x and y data from the KDE line
                    # kde_line_Xval = kde_line.get_xdata()
                    # kde_line_Yval = kde_line.get_ydata()

                    # if i != 8:
                    #     axs[i].plot(kde_line_Xval[max_index], kde_line_Yval[max_index], 'ro')

                    # if i==0:
                    #     # place the xaxis exponent in the bottom right corner
                    #     axs[i].xaxis.get_offset_text().set_x(1.10)
                    # if len(Min_KDE_point) > 0:     
                    #     if len(Min_KDE_point)>ii_densest:                    

                    #         # Find the index with the closest value to densest_point[ii_dense] to all y values
                    #         densest_index = find_closest_index(kde_line_Xval, [Min_KDE_point[ii_densest]])

                    #         # add also the densest_point[i] as a blue dot
                    #         axs[i].plot(Min_KDE_point[ii_densest], kde_line_Yval[densest_index[0]], 'bo')
                    #         ii_densest=ii_densest+1
                # # more space between the subplots erosion_coeff sigma
                plt.tight_layout()

                # save the figure maximized and with the right name
                fig.savefig(output_dir+os.sep+file_name+'_PhysicProp_Reliazations_'+str(n_PC_in_PCA)+'PC_'+str(len(curr_sel))+'ev.png', dpi=300)



def PCA_LightCurveCoefPLOT(df_sel_shower_real, df_obs_shower, output_dir, fit_funct='', gensim_data_obs='', mag_noise_real= 0.1, len_noise_real = 20.0, file_name_obs='', trajectory_Metsim_file='', output_folder_of_csv=''):

    # number to confront
    n_confront_obs=1
    if output_folder_of_csv=='':
        n_confront_sel=7
    else:
        n_confront_sel=9

    # number of PC in PCA
    with_noise=True

    # is the input data noisy
    noise_data_input=False

    # activate jachia
    jacchia_fit=False

    # activate parabolic fit
    parabolic_fit=False

    t0_fit=False

    mag_fit=False

    # 5 sigma confidence interval
    # five_sigma=False
    # mag_noise = MAG_RMSD*SIGMA_ERR
    # len_noise = LEN_RMSD*SIGMA_ERR
    mag_noise = mag_noise_real.copy()
    len_noise = len_noise_real.copy()

    # # Standard deviation of the magnitude Gaussian noise 1 sigma
    # # SD of noise in length (m) 1 sigma in km
    len_noise= len_noise/1000
    # velocity noise 1 sigma km/s
    # vel_noise = (len_noise*np.sqrt(2)/(1/FPS))
    vel_noise = (len_noise/(1/FPS))

    # put the first plot in 2 sublots
    fig, ax = plt.subplots(1, 2, figsize=(17, 5))

    df_sel_shower = df_sel_shower_real.copy()

    # # group by solution_id_dist and keep only n_confront_sel from each group
    # df_sel_shower = df_sel_shower.groupby('solution_id_dist').head(len(df_sel_shower))

    # check if distance_meteor is in the columns
    if 'distance_meteor' in df_sel_shower.columns:
        # order by distance_meteor
        df_sel_shower = df_sel_shower.sort_values('distance_meteor')

    # # count duplicates and add a column for the number of duplicates
    # df_sel_shower['num_duplicates'] = df_sel_shower.groupby('solution_id')['solution_id'].transform('size')

    # df_sel_shower.drop_duplicates(subset='solution_id', keep='first', inplace=True)

    df_sel_shower['erosion_coeff']=df_sel_shower['erosion_coeff']*1000000
    df_sel_shower['sigma']=df_sel_shower['sigma']*1000000

    if n_confront_obs<len(df_obs_shower):
        df_obs_shower=df_obs_shower.head(n_confront_obs)
    
    # if n_confront_sel<len(df_sel_shower):
    #     df_sel_shower=df_sel_shower.head(n_confront_sel)  

    # merge curr_sel and curr_obs
    curr_sel = pd.concat([df_obs_shower,df_sel_shower], axis=0)

    metsim_numbs=0
    for ii in range(len(curr_sel)):
        # pick the ii element of the solution_id column 
        namefile_sel=curr_sel.iloc[ii]['solution_id']
        Metsim_flag=False
        print('real',trajectory_Metsim_file,'- sel',namefile_sel)

        # chec if the file exist
        if not os.path.isfile(namefile_sel):
            print('file '+namefile_sel+' not found')
            continue
        else:
            if namefile_sel.endswith('.pickle'):
                data_file = read_pickle_reduction_file(namefile_sel)
                data_file_real=data_file.copy()

            elif namefile_sel.endswith('.json'):
                # open the json file with the name namefile_sel 
                f = open(namefile_sel,"r")
                data = json.loads(f.read())
                if 'ht_sampled' in data:
                    if ii==0:
                        data_file = read_with_noise_GenerateSimulations_output(namefile_sel)
                    else:
                        data_file = read_GenerateSimulations_output(namefile_sel, gensim_data_obs)

                else:
                    if trajectory_Metsim_file == '':
                        print('no data for the Metsim file')
                        continue
                    elif trajectory_Metsim_file == namefile_sel:
                        _, data_file, _ = run_simulation(trajectory_Metsim_file, gensim_data_obs)
                        # file metsim
                        Metsim_flag=True

                    else:
                        # make a copy of gen_Metsim
                        _, data_file, _ = run_simulation(namefile_sel, gensim_data_obs)

            height_km=np.array(data_file['height'])/1000
            abs_mag_sim=np.array(data_file['absolute_magnitudes'])
            obs_time=np.array(data_file['time'])
            vel_kms=np.array(data_file['velocities'])/1000
            lag_km=np.array(data_file['lag'])/1000

        if ii==0:
            
            if with_noise==True and fit_funct!='':
                # from list to array
                height_km_err=np.array(fit_funct['height'])/1000
                lag_km_err=np.array(fit_funct['lag'])/1000
                abs_mag_sim_err=np.array(fit_funct['absolute_magnitudes'])
                

                # plot noisy area around vel_kms for vel_noise for the fix height_km
                ax[0].fill_betweenx(height_km_err, abs_mag_sim_err-mag_noise, abs_mag_sim_err+mag_noise, color='lightgray', alpha=0.5)

                obs_time_err=np.array(fit_funct['time'])
                vel_kms_err=np.array(fit_funct['velocities'])/1000

                # plot noisy area around vel_kms for vel_noise for the fix height_km
                ax[1].fill_between(obs_time_err, vel_kms_err-vel_noise, vel_kms_err+vel_noise, color='lightgray', alpha=0.5, label='Std.dev. realizations')

            ax[0].plot(abs_mag_sim,height_km)

            ax[1].plot(obs_time, vel_kms,label=file_name_obs[:15]+'\nRMSDmag '+str(round(mag_noise_real,3))+' RMSDlen '+str(round(len_noise_real/1000,3)))

        elif ii<=n_confront_sel:

            rmsd_mag, rmsd_vel, rmsd_lag, _, _, _, _, _ = RMSD_calc_diff(data_file, fit_funct)
            
            if Metsim_flag:
                metsim_numbs=ii
                ax[0].plot(abs_mag_sim,height_km, 'k')
                if 'distance_meteor' in df_sel_shower.columns:
                    ax[1].plot(obs_time, vel_kms, 'k', label='Metsim data reduction\n\
RMSDmag '+str(round(rmsd_mag,3))+' RMSDlen '+str(round(rmsd_lag,3))+'\n\
    N°duplic. '+str(round(curr_sel.iloc[ii]['num_duplicates']))+' min dist:'+str(round(curr_sel.iloc[ii]['distance_meteor'],2))+'\n\
    m:'+str('{:.2e}'.format(curr_sel.iloc[ii]['mass'],1))+' F:'+str(round(curr_sel.iloc[ii]['F'],2))+'\n\
    rho:'+str(round(curr_sel.iloc[ii]['rho']))+' sigma:'+str(round(curr_sel.iloc[ii]['sigma'],4))+'\n\
    er.height:'+str(round(curr_sel.iloc[ii]['erosion_height_start'],2))+' er.log:'+str(round(curr_sel.iloc[ii]['erosion_range'],1))+'\n\
    er.coeff:'+str(round(curr_sel.iloc[ii]['erosion_coeff'],3))+' er.index:'+str(round(curr_sel.iloc[ii]['erosion_mass_index'],2)))        
                else:
                    ax[1].plot(obs_time, vel_kms, 'k', label='Metsim data reduction\n\
RMSDmag '+str(round(rmsd_mag,3))+' RMSDlen '+str(round(rmsd_lag,3))+'\n\
    m:'+str('{:.2e}'.format(curr_sel.iloc[ii]['mass'],1))+' F:'+str(round(curr_sel.iloc[ii]['F'],2))+'\n\
    rho:'+str(round(curr_sel.iloc[ii]['rho']))+' sigma:'+str(round(curr_sel.iloc[ii]['sigma'],4))+'\n\
    er.height:'+str(round(curr_sel.iloc[ii]['erosion_height_start'],2))+' er.log:'+str(round(curr_sel.iloc[ii]['erosion_range'],1))+'\n\
    er.coeff:'+str(round(curr_sel.iloc[ii]['erosion_coeff'],3))+' er.index:'+str(round(curr_sel.iloc[ii]['erosion_mass_index'],2)))                                                                         
            else:

                ax[0].plot(abs_mag_sim,height_km)

                # Get the color of the last plotted line in graph 0
                line_color = ax[0].get_lines()[-1].get_color()
                # if line_color == '#2ca02c':
                #     line_color='m'
                #     ax[0].plot(abs_mag_sim,height_km, color='m')
                
                if 'distance_meteor' in df_sel_shower.columns:
                    ax[1].plot(obs_time, vel_kms, color=line_color ,label='RMSDmag '+str(round(rmsd_mag,3))+' RMSDlen '+str(round(rmsd_lag,3))+'\n\
    N°duplic. '+str(round(curr_sel.iloc[ii]['num_duplicates']))+' min dist:'+str(round(curr_sel.iloc[ii]['distance_meteor'],2))+'\n\
    m:'+str('{:.2e}'.format(curr_sel.iloc[ii]['mass'],1))+' F:'+str(round(curr_sel.iloc[ii]['F'],2))+'\n\
    rho:'+str(round(curr_sel.iloc[ii]['rho']))+' sigma:'+str(round(curr_sel.iloc[ii]['sigma'],4))+'\n\
    er.height:'+str(round(curr_sel.iloc[ii]['erosion_height_start'],2))+' er.log:'+str(round(curr_sel.iloc[ii]['erosion_range'],1))+'\n\
    er.coeff:'+str(round(curr_sel.iloc[ii]['erosion_coeff'],3))+' er.index:'+str(round(curr_sel.iloc[ii]['erosion_mass_index'],2)))
                else:
                    ax[1].plot(obs_time, vel_kms, color=line_color ,label='RMSDmag '+str(round(rmsd_mag,3))+' RMSDlen '+str(round(rmsd_lag,3))+'\n\
    m:'+str('{:.2e}'.format(curr_sel.iloc[ii]['mass'],1))+' F:'+str(round(curr_sel.iloc[ii]['F'],2))+'\n\
    rho:'+str(round(curr_sel.iloc[ii]['rho']))+' sigma:'+str(round(curr_sel.iloc[ii]['sigma'],4))+'\n\
    er.height:'+str(round(curr_sel.iloc[ii]['erosion_height_start'],2))+' er.log:'+str(round(curr_sel.iloc[ii]['erosion_range'],1))+'\n\
    er.coeff:'+str(round(curr_sel.iloc[ii]['erosion_coeff'],3))+' er.index:'+str(round(curr_sel.iloc[ii]['erosion_mass_index'],2)))

        rmsd_mag, rmsd_vel, rmsd_lag, _, _, _, _, _ = RMSD_calc_diff(data_file, fit_funct)

        # add a column rmsd_mag and rmsd_vel to curr_sel to the one with the same curr_sel.iloc[ii]['solution_id']
        df_sel_shower_real.loc[df_sel_shower_real['solution_id']==curr_sel.iloc[ii]['solution_id'],'rmsd_mag'] = rmsd_mag
        # df_sel_shower_real.loc[df_sel_shower_real['solution_id']==curr_sel.iloc[ii]['solution_id'],'rmsd_vel'] = rmsd_vel
        df_sel_shower_real.loc[df_sel_shower_real['solution_id']==curr_sel.iloc[ii]['solution_id'],'rmsd_len'] = rmsd_lag

#############ADD COEF#############################################
        if mag_fit==True:

            index_ht_peak = np.argmin(abs_mag_sim)

            ax[0].plot(curr_sel.iloc[ii]['a_mag_init']*np.array(height_km[:index_ht_peak])**2+curr_sel.iloc[ii]['b_mag_init']*np.array(height_km[:index_ht_peak])+curr_sel.iloc[ii]['c_mag_init'],height_km[:index_ht_peak], color=ax[0].lines[-1].get_color(), linestyle='None', marker='<')# , markersize=5

            ax[0].plot(curr_sel.iloc[ii]['a_mag_end']*np.array(height_km[index_ht_peak:])**2+curr_sel.iloc[ii]['b_mag_end']*np.array(height_km[index_ht_peak:])+curr_sel.iloc[ii]['c_mag_end'],height_km[index_ht_peak:], color=ax[0].lines[-1].get_color(), linestyle='None', marker='>')# , markersize=5
            
        if parabolic_fit==True:
            ax[1].plot(obs_time,curr_sel.iloc[ii]['a_acc']*np.array(obs_time)**2+curr_sel.iloc[ii]['b_acc']*np.array(obs_time)+curr_sel.iloc[ii]['c_acc'], color=ax[1].lines[-1].get_color(), linestyle='None', marker='o')# , markersize=5
        
        # Assuming the jacchiaVel function is defined as:
        def jacchiaVel(t, a1, a2, v_init):
            return v_init - np.abs(a1) * np.abs(a2) * np.exp(np.abs(a2) * t)
        if jacchia_fit==True:
            ax[1].plot(obs_time, jacchiaVel(np.array(obs_time), curr_sel.iloc[ii]['a1_acc_jac'], curr_sel.iloc[ii]['a2_acc_jac'],vel_kms[0]), color=ax[1].lines[-1].get_color(), linestyle='None', marker='d') 

        if t0_fit==True: # quadratic_velocity(t, a, v0, t0) 
            ax[1].plot(obs_time, cubic_velocity(np.array(obs_time), curr_sel.iloc[ii]['a_t0'], curr_sel.iloc[ii]['b_t0'], curr_sel.iloc[ii]['vel_init_norot'], curr_sel.iloc[ii]['t0']), color=ax[1].lines[-1].get_color(), linestyle='None', marker='s') 


    # change the first plotted line style to be a dashed line
    ax[0].lines[0].set_linestyle("None")
    ax[1].lines[0].set_linestyle("None")
    # change the first plotted marker to be a x
    ax[0].lines[0].set_marker("o")
    ax[1].lines[0].set_marker("o")
    # change first line color
    ax[0].lines[0].set_color('g')
    ax[1].lines[0].set_color('g')
    # change the zorder=-1 of the first line
    ax[0].lines[0].set_zorder(n_confront_sel)
    ax[1].lines[0].set_zorder(n_confront_sel)

    if metsim_numbs != 0:
        ax[0].lines[metsim_numbs].set_color('black')
        ax[1].lines[metsim_numbs].set_color('black')


    # change dot line color
    if mag_fit==True:
        ax[0].lines[1].set_color('g')
        ax[0].lines[2].set_color('g')


# check how many of the jacchia_fit and parabolic_fit and t0_fit are set to true
    numcheck=0
    if jacchia_fit==True:
        numcheck+=1
    if parabolic_fit==True:
        numcheck+=1
    if t0_fit==True:
        numcheck+=1

    if numcheck==1:
        ax[1].lines[1].set_color('g')
        ax[1].lines[1].set_zorder(n_confront_sel)
    if numcheck==2:
        ax[1].lines[1].set_color('g')
        ax[1].lines[2].set_color('g')
        ax[1].lines[1].set_zorder(n_confront_sel)
        ax[1].lines[2].set_zorder(n_confront_sel)
    if numcheck==3:
        ax[1].lines[1].set_color('g')
        ax[1].lines[2].set_color('g')
        ax[1].lines[3].set_color('g')
        ax[1].lines[1].set_zorder(n_confront_sel)
        ax[1].lines[2].set_zorder(n_confront_sel)
        ax[1].lines[3].set_zorder(n_confront_sel)

    # # change the zorder=-1 of the first line
    # ax[0].lines[1].set_zorder(n_confront_sel)
    # ax[0].lines[2].set_zorder(n_confront_sel)


    # grid on on both subplot with -- as linestyle and light gray color
    ax[1].grid(linestyle='--',color='lightgray')
    # grid on
    ax[0].grid(linestyle='--',color='lightgray')
    # ax[0].set_title(current_shower+' height vs mag')


    if n_confront_sel <= 5:
        # pu the leggend putside the plot and adjust the plot base on the screen size
        ax[-1].legend(bbox_to_anchor=(1.05, 1.1), loc='upper left', borderaxespad=0.)
        # the legend do not fit in the plot, so adjust the plot
        plt.subplots_adjust(right=0.8)
    else:
        # pu the leggend putside the plot and adjust the plot base on the screen size
        ax[-1].legend(bbox_to_anchor=(1.05, 1.1), loc='upper left', borderaxespad=0.,fontsize="10",ncol=2)
        # the legend do not fit in the plot, so adjust the plot
        plt.subplots_adjust(right=.6)
        # push the two subplots left
        # plt.subplots_adjust(left=-.0001)
        plt.subplots_adjust(wspace=0.2)


    # invert the x axis
    ax[0].invert_xaxis()

    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('Velocity [km/s]')
    ax[0].set_xlabel('Absolute Magnitude [-]')
    ax[0].set_ylabel('Height [km]')

    plt.savefig(output_dir+os.sep+file_name_obs+'_Heigh_MagVelCoef.png')

    # close the plot
    plt.close()

    if output_folder_of_csv=='':
        # save df_sel_shower_real to disk add the RMSD
        df_sel_shower_real.to_csv(output_dir+os.sep+SAVE_SELECTION_FOLDER+os.sep+file_name_obs+'_sim_sel_to_optimize.csv', index=False)
    else:
        # save df_sel_shower_real to disk add the RMSD
        df_sel_shower_real.to_csv(output_folder_of_csv, index=False)





if __name__ == "__main__":

    import argparse

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Fom Observation and simulated data weselect the most likely through PCA, run it, and store results to disk.")
    # C:\Users\maxiv\Desktop\RunTest\TRUEerosion_sim_v59.84_m1.33e-02g_rho0209_z39.8_abl0.014_eh117.3_er0.636_s1.61.json
    # C:\Users\maxiv\Desktop\20230811-082648.931419
    # 'C:\Users\maxiv\Desktop\jsontest\Simulations_PER_v65_fast\TRUEerosion_sim_v65.00_m7.01e-04g_rho0709_z51.7_abl0.015_eh115.2_er0.483_s2.46.json'
    arg_parser.add_argument('input_dir', metavar='INPUT_PATH', type=str, \
        help="Path were are store both simulated and observed shower .csv file.")
    
    arg_parser.add_argument('--MetSim_json', metavar='METSIM_JSON', type=str, default='_sim_fit_latest.json', \
        help="json file extension where are stored the MetSim constats, by default _sim_fit_latest.json.")   

    arg_parser.add_argument('--nobs', metavar='OBS_NUM', type=int, default=50, \
        help="Number of Observation that will be resampled.")
    
    arg_parser.add_argument('--nsim', metavar='SIM_NUM', type=int, default=1000, \
        help="Number of simulations to generate.")
    
    arg_parser.add_argument('--min_nres', metavar='SIM_RES', type=int, default=30, \
        help="Minimum number of results that are in the CI that have to be found.")

    arg_parser.add_argument('--nsel_forced', metavar='SEL_NUM_FORCED', type=int, default=0, \
        help="Number of selected simulations forced to consider instead of choosing the knee of the distance function.")
    
    arg_parser.add_argument('--PCA_percent', metavar='PCA_PERCENT', type=int, default=99, \
        help="Percentage of the variance explained by the PCA.")

    arg_parser.add_argument('--YesPCA', metavar='YESPCA', type=str, default=[], \
        help="Use specific variable to considered in PCA.")

    arg_parser.add_argument('--NoPCA', metavar='NOPCA', type=str, default=['v_init_180km','kurtosis','skew','a1_acc_jac','a2_acc_jac','a_acc','b_acc','c_acc','c_mag_init','c_mag_end','a_t0', 'b_t0', 'c_t0'], \
        help="Use specific variable NOT considered in PCA.")

    arg_parser.add_argument('--save_test_plot', metavar='SAVE_TEST_PLOT', type=bool, default=False, \
        help="save test plots of the realization and the simulations and more plots in PCA control plots.")
    
    arg_parser.add_argument('--optimize', metavar='OPTIMIZE', type=bool, default=False, \
        help="Run optimization step to have more precise results but increase the computation time.")
    
    arg_parser.add_argument('--esclude_real_solution_from_selection', metavar='ESCLUDE_REAL_SOLUTION_FROM_SELECTION', type=bool, default=False, \
        help="When use a generate simulation you can select to exclude the real result with True or also consider it in the distance calculations with False.")
    
    arg_parser.add_argument('--number_optimized', metavar='NUMBER_OPTIMZED', type=int, default=0, \
        help="Number of optimized simulations that have to be optimized starting from the best, 0 means all.")
    
    arg_parser.add_argument('--ref_opt_path', metavar='REF_OPT_PATH', type=str, default=r'/home/mvovk/WesternMeteorPyLib/wmpl/MetSim/AutoRefineFit_options.txt', \
        help="path and name of like C: path + AutoRefineFit_options.txt")

    arg_parser.add_argument('--cores', metavar='CORES', type=int, default=None, \
        help="Number of cores to use. All by default.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################
    warnings.filterwarnings('ignore')

    if cml_args.optimize:
        # check if the file exist
        if not os.path.isfile(cml_args.ref_opt_path):
            # If the file is not found, check in the parent directory
            parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cml_args.ref_opt_path = os.path.join(parent_directory, 'AutoRefineFit_options.txt')
            if not os.path.isfile(cml_args.ref_opt_path):
                print('file '+cml_args.ref_opt_path+' not found')
                print("You need to specify the correct path and name of the AutoRefineFit_options.txt file in --ref_opt_path, like: C:\\path\\AutoRefineFit_options.txt")
                sys.exit()

    # check if is a file or a directory
    if os.path.isdir(cml_args.input_dir):
        # pack the 3 lists in a tuple
        input_folder_file = find_and_extract_trajectory_files(cml_args.input_dir, cml_args.MetSim_json)
        
    else:
        # check if the file exists
        if not os.path.isfile(cml_args.input_dir):
            print('The file does not exist')
            sys.exit()
        else:
            # split the dir and file
            trajectory_files = [cml_args.input_dir]
            # for the output folder delete the extension of the file and add _GenSim
            output_folders = [os.path.splitext(cml_args.input_dir)[0]+NAME_SUFX_GENSIM]
            file_names = [os.path.splitext(os.path.split(cml_args.input_dir)[1])[0]]
            input_folders = [os.path.split(cml_args.input_dir)[0]]

            # check if the file ends with .json
            if cml_args.input_dir.endswith('.json'):
                # open the json file and save the const part as file_name_obs+'_sim_fit.json'
                with open(cml_args.input_dir) as json_file:
                    data = json.load(json_file)
                    const_part = data['const']
                    MetSim_phys_file_path = [input_folders[0]+os.sep+file_names[0]+'_const.json']
                    with open(input_folders[0]+os.sep+file_names[0]+'_const.json', 'w') as outfile:
                        json.dump(const_part, outfile, indent=4)
            else:
                # check if MetSim_phys_file_path exist
                if os.path.isfile(os.path.join(input_folders[0], file_names[0] + cml_args.MetSim_json)):
                    # print did not find with th given extention revert to default
                    MetSim_phys_file_path = os.path.join(input_folders[0], file_names[0] + cml_args.MetSim_json)
                elif os.path.isfile(os.path.join(input_folders[0], file_names[0] + '_sim_fit_latest.json')):
                    print(file_names[0],': No MetSim file with the given extention', cml_args.MetSim_json,'reverting to default extention _sim_fit_latest.json')
                    MetSim_phys_file_path = os.path.join(input_folders[0], file_names[0] + '_sim_fit_latest.json')
                else:
                    # do not save the rest of the files
                    print(file_names[0],': No MetSim file with the given extention', cml_args.MetSim_json,'do not consider the folder')
                    # raise an error if the file is not a csv, pickle or json file
                    raise ValueError('File format not supported. Please provide a csv, pickle file with a Metsim manual reduction or json from generate simulations.')

            input_folder_file = [[trajectory_files[0], file_names[0], input_folders[0], output_folders[0], MetSim_phys_file_path[0]]]
    

    # print only the file name in the directory split the path and take the last element
    print('Number of trajectory.pickle files find',len(input_folder_file))
    for trajectory_file, file_name, input_folder, output_folder, trajectory_Metsim_file in input_folder_file:
        print('processing file:',file_name)
        # print(trajectory_file)
        # print(input_folder)
        # print(output_folder)
        # print(trajectory_Metsim_file)

        start_time = time.time()

        # chek if input_folder+os.sep+file_name+NAME_SUFX_CSV_OBS exist
        if os.path.isfile(output_folder+os.sep+file_name+NAME_SUFX_CSV_OBS):
            # read the csv file
            trajectory_file = output_folder+os.sep+file_name+NAME_SUFX_CSV_OBS

        # check if the output_folder exists
        if not os.path.isdir(output_folder):
            mkdirP(output_folder)

        print()

        ######################### OBSERVATION ###############################
        print('--- OBSERVATION ---')

        # check the extension of the file
        if trajectory_file.endswith('.csv'):
            # read the csv file
            pd_dataframe_PCA_obs_real = pd.read_csv(trajectory_file)
            # check the column name solution_id	and see if it matches a file i the folder
            if not input_folder in pd_dataframe_PCA_obs_real['solution_id'][0]:
                # if the solution_id is in the name of the file then the file is the real data
                print('The folder of the csv file is different')

            if pd_dataframe_PCA_obs_real['type'][0] != 'Observation' and pd_dataframe_PCA_obs_real['type'][0] != 'Observation_sim':
                # raise an error saing that the type is wrong and canot be processed by PCA
                raise ValueError('Type of the csv file is wrong and canot be processed by script.')

            if pd_dataframe_PCA_obs_real['solution_id'][0].endswith('.pickle'):
                # read the pickle file
                gensim_data_obs = read_pickle_reduction_file(pd_dataframe_PCA_obs_real['solution_id'][0])

            # json file
            elif pd_dataframe_PCA_obs_real['solution_id'][0].endswith('.json'): 
                # read the json file with noise
                gensim_data_obs = read_with_noise_GenerateSimulations_output(pd_dataframe_PCA_obs_real['solution_id'][0])

            else:
                # raise an error if the file is not a csv, pickle or json file
                raise ValueError('File format not supported. Please provide a csv, pickle or json file.')

            rmsd_t0_lag, rmsd_pol_mag, fit_pol_mag, fitted_lag_t0_lag, fit_funct = find_noise_of_data(gensim_data_obs)

            print('read the csv file:',trajectory_file)

        else:

            if trajectory_file.endswith('.pickle'):
                # read the pickle file
                gensim_data_obs = read_pickle_reduction_file(trajectory_file) #,trajectory_Metsim_file

            # json file
            elif trajectory_file.endswith('.json'): 
                # read the json file with noise
                gensim_data_obs = read_with_noise_GenerateSimulations_output(trajectory_file)
                
            else:
                # raise an error if the file is not a csv, pickle or json file
                raise ValueError('File format not supported. Please provide a csv, pickle or json file.')
            
            pd_dataframe_PCA_obs_real = array_to_pd_dataframe_PCA(gensim_data_obs)
            pd_dataframe_PCA_obs_real['type'] = 'Observation'

            if cml_args.save_test_plot:
                # run generate observation realization with the gensim_data_obs
                rmsd_t0_lag, rmsd_pol_mag, fit_pol_mag, fitted_lag_t0_lag, fit_funct, fig, ax = find_noise_of_data(gensim_data_obs,cml_args.save_test_plot)
                # make the results_list to incorporate all rows of pd_dataframe_PCA_obs_real
                results_list = []
                for ii in range(cml_args.nobs):
                    results_pd = generate_observation_realization(gensim_data_obs, rmsd_t0_lag, rmsd_pol_mag, fit_funct,'realization_'+str(ii+1), fig, ax, cml_args.save_test_plot) 
                    results_list.append(results_pd)

                # plot noisy area around vel_kms for vel_noise for the fix height_km
                ax[0].fill_betweenx(np.array(fit_funct['height'])/1000, np.array(fit_funct['absolute_magnitudes'])-rmsd_pol_mag, np.array(fit_funct['absolute_magnitudes'])+rmsd_pol_mag, color='lightgray', alpha=0.5)

                # plot noisy area around vel_kms for vel_noise for the fix height_km
                ax[1].fill_between(np.array(fit_funct['time']), np.array(fit_funct['velocities'])/1000-(rmsd_t0_lag/1000/(1/FPS)), np.array(fit_funct['velocities'])/1000+(rmsd_t0_lag/1000/(1/FPS)), color='lightgray', alpha=0.5, label='Std.dev. realizations')
                # ax[1].fill_between(np.array(fit_funct['time']), np.array(fit_funct['velocities'])/1000-(rmsd_t0_lag/1000*np.sqrt(2)/(1/FPS)), np.array(fit_funct['velocities'])/1000+(rmsd_t0_lag/1000*np.sqrt(2)/(1/FPS)), color='lightgray', alpha=0.5, label='Std.dev. realizations')
                # Save the figure as file with instead of _trajectory.pickle it has file+std_dev.png on the desktop
                plt.savefig(output_folder+os.sep+file_name+'obs_realizations.png', dpi=300)

                plt.close()

            else:      
                rmsd_t0_lag, rmsd_pol_mag, fit_pol_mag, fitted_lag_t0_lag, fit_funct = find_noise_of_data(gensim_data_obs)       
                input_list_obs = [[gensim_data_obs, rmsd_t0_lag, rmsd_pol_mag, fit_funct,'realization_'+str(ii+1)] for ii in range(cml_args.nobs)]
                results_list = domainParallelizer(input_list_obs, generate_observation_realization, cores=cml_args.cores)
            
            df_obs_realiz = pd.concat(results_list)
            pd_dataframe_PCA_obs_real = pd.concat([pd_dataframe_PCA_obs_real, df_obs_realiz])
            # re index the dataframe
            pd_dataframe_PCA_obs_real.reset_index(drop=True, inplace=True)

            # check if there is a column with the name 'mass'
            if 'mass' in pd_dataframe_PCA_obs_real.columns:
                #delete from the real_data panda dataframe mass rho sigma
                pd_dataframe_PCA_obs_real = pd_dataframe_PCA_obs_real.drop(columns=['mass', 'rho', 'sigma', 'erosion_height_start', 'erosion_coeff', 'erosion_mass_index', 'erosion_mass_min', 'erosion_mass_max', 'erosion_range', 'erosion_energy_per_unit_cross_section', 'erosion_energy_per_unit_mass'])

            pd_dataframe_PCA_obs_real.to_csv(output_folder+os.sep+file_name+NAME_SUFX_CSV_OBS, index=False)
            # print saved csv file
            print()
            print('saved obs csv file:',output_folder+os.sep+file_name+NAME_SUFX_CSV_OBS)
        
        print()




        ######################## SIMULATIONTS ###############################
        print('--- SIMULATIONS ---')

        print('Run MetSim file:',trajectory_Metsim_file)

        # use Metsim_file in 
        simulation_MetSim_object, gensim_data_Metsim, pd_datafram_PCA_sim_Metsim = run_simulation(trajectory_Metsim_file, gensim_data_obs)

        # open the folder and extract all the json files
        os.chdir(input_folder)

        # chek in directory if it exist a csv file with input_folder+os.sep+file_name+NAME_SUFX_CSV_SIM
        if os.path.isfile(output_folder+os.sep+file_name+NAME_SUFX_CSV_SIM):
            # read the csv file
            pd_datafram_PCA_sim = pd.read_csv(output_folder+os.sep+file_name+NAME_SUFX_CSV_SIM)
            print('read the csv file:',output_folder+os.sep+file_name+NAME_SUFX_CSV_SIM)

        else:

            # open the folder and extract all the json files
            os.chdir(output_folder)

            extension = 'json'
            # walk thorought the directories and find all the json files inside each folder inside the directory
            all_jsonfiles_check = [i for i in glob.glob('**/*.{}'.format(extension), recursive=True)]

            if len(all_jsonfiles_check) == 0 or len(all_jsonfiles_check) < cml_args.nsim:
                if len(all_jsonfiles_check) != 0:
                    print('In the sim folder there are already',len(all_jsonfiles_check),'json files')
                    print('Add',cml_args.nsim - len(all_jsonfiles_check),' json files')
                number_sim_to_run_and_simulation_in_folder = cml_args.nsim - len(all_jsonfiles_check)
                
                # run the new simulations
                if cml_args.save_test_plot:
                    fig, ax = generate_simulations(pd_dataframe_PCA_obs_real,simulation_MetSim_object,gensim_data_obs,number_sim_to_run_and_simulation_in_folder,output_folder,file_name,cml_args.save_test_plot)
                    # plot gensim_data_Metsim
                    plot_side_by_side(gensim_data_Metsim,fig, ax,'k-',str(pd_datafram_PCA_sim_Metsim['type'].iloc[0]))

                    # plot noisy area around vel_kms for vel_noise for the fix height_km
                    ax[0].fill_betweenx(np.array(fit_funct['height'])/1000, np.array(fit_funct['absolute_magnitudes'])-rmsd_pol_mag, np.array(fit_funct['absolute_magnitudes'])+rmsd_pol_mag, color='lightgray', alpha=0.5)

                    # plot noisy area around vel_kms for vel_noise for the fix height_km
                    ax[1].fill_between(np.array(fit_funct['time']), np.array(fit_funct['velocities'])/1000-(rmsd_t0_lag/1000/(1/FPS)), np.array(fit_funct['velocities'])/1000+(rmsd_t0_lag/1000/(1/FPS)), color='lightgray', alpha=0.5, label='Std.dev. realizations')
                    # ax[1].fill_between(np.array(fit_funct['time']), np.array(fit_funct['velocities'])/1000-(rmsd_t0_lag/1000*np.sqrt(2)/(1/FPS)), np.array(fit_funct['velocities'])/1000+(rmsd_t0_lag/1000*np.sqrt(2)/(1/FPS)), color='lightgray', alpha=0.5, label='Std.dev. realizations')

                    # save the plot
                    plt.savefig(output_folder+os.sep+file_name+'_obs_sim.png', dpi=300)
                    # close the plot
                    plt.close()
                    # print saved csv file
                    print('saved image '+output_folder+os.sep+file_name+'_obs_sim.png')
                else:
                    generate_simulations(pd_dataframe_PCA_obs_real,simulation_MetSim_object,gensim_data_obs,number_sim_to_run_and_simulation_in_folder,output_folder,file_name,cml_args.save_test_plot)
                    
            print('start reading the json files')

            all_jsonfiles = [i for i in glob.glob('**/*.{}'.format(extension), recursive=True)]

            # add the output_folder to all_jsonfiles
            all_jsonfiles = [output_folder+os.sep+file for file in all_jsonfiles]

            # open the folder and extract all the json files
            os.chdir(input_folder)

            print('Number of simulated files: ',len(all_jsonfiles))

            input_list = [[all_jsonfiles[ii], 'simulation_'+str(ii+1)] for ii in range(len(all_jsonfiles))]
            results_list = domainParallelizer(input_list, read_GenerateSimulations_output_to_PCA, cores=cml_args.cores)
            
            # if no read the json files in the folder and create a new csv file
            pd_datafram_PCA_sim = pd.concat(results_list)
            
            # concatenate the two dataframes
            pd_datafram_PCA_sim = pd.concat([pd_datafram_PCA_sim_Metsim, pd_datafram_PCA_sim])
            # print(df_sim_shower)
            pd_datafram_PCA_sim.reset_index(drop=True, inplace=True)
            
            if pd_dataframe_PCA_obs_real['solution_id'].iloc[0].endswith('.json'): 
                print('REAL json file:',trajectory_Metsim_file)
                # change the type column to Real
                pd_datafram_PCA_sim['type'].iloc[0] = 'Real'

            pd_datafram_PCA_sim.to_csv(output_folder+os.sep+file_name+NAME_SUFX_CSV_SIM, index=False)
            # print saved csv file
            print('saved sim csv file:',output_folder+os.sep+file_name+NAME_SUFX_CSV_SIM)

            
        if pd_dataframe_PCA_obs_real['solution_id'].iloc[0].endswith('.json'): 
            print('REAL json file:',trajectory_Metsim_file)
            # change the type column to Real
            pd_datafram_PCA_sim['type'].iloc[0] = 'Real'


        # save the trajectory_file in the output_folder
        shutil.copy(pd_dataframe_PCA_obs_real['solution_id'][0], output_folder)

        # delete any file that end with _good_files.txt in the output_folder
        files = [f for f in os.listdir(output_folder) if f.endswith('_good_files.txt')]
        for file in files:
            os.remove(os.path.join(output_folder, file))

        print()
            


        ######################## SELECTION ###############################

        print('--- SELECTION ---')
        
        pd_datafram_PCA_selected_before_knee, pd_datafram_PCA_selected_before_knee_NO_repetition, pd_datafram_PCA_selected_all, pcr_results_physical_param, pca_N_comp = PCASim(pd_datafram_PCA_sim, pd_dataframe_PCA_obs_real, output_folder, cml_args.PCA_percent, cml_args.nsel_forced, cml_args.YesPCA, cml_args.NoPCA, file_name, cml_args.cores, cml_args.save_test_plot, cml_args.esclude_real_solution_from_selection)

        print('PLOT: best 7 simulations selected and add the RMSD value to csv selected')
        # plot of the best 7 selected simulations and add the RMSD value to csv selected
        PCA_LightCurveCoefPLOT(pd_datafram_PCA_selected_before_knee_NO_repetition, pd_dataframe_PCA_obs_real, output_folder, fit_funct, gensim_data_obs, rmsd_pol_mag, rmsd_t0_lag, file_name, trajectory_Metsim_file)

        print('PLOT: the physical characteristics of the selected simulations Mode and KDE')
        PCA_PhysicalPropPLOT(pd_datafram_PCA_selected_before_knee, pd_datafram_PCA_sim, pca_N_comp, output_folder, file_name)

        print('PLOT: correlation of the selected simulations')
        # plot correlation function of the selected simulations
        PCAcorrelation_selPLOT(pd_datafram_PCA_sim, pd_datafram_PCA_selected_before_knee_NO_repetition, pca_N_comp, output_folder)


        mkdirP(output_folder+os.sep+SAVE_RESULTS_FOLDER)
        mkdirP(output_folder+os.sep+SAVE_RESULTS_FOLDER_EVENTS_PLOTS)
        print('Selected simulations and generate KDE and MODE plot')
        # pd_datafram_PCA_selected_mode_min_KDE = PCA_physicalProp_KDE_MODE_PLOT(pd_datafram_PCA_sim, pd_dataframe_PCA_obs_real, pd_datafram_PCA_selected_before_knee, pca_N_comp, fit_funct, rmsd_pol_mag, rmsd_t0_lag, trajectory_Metsim_file, file_name, pd_dataframe_PCA_obs_real['solution_id'].iloc[0], output_folder)
        input_list_obs = [[pd_datafram_PCA_sim, pd_dataframe_PCA_obs_real.iloc[[ii]].reset_index(drop=True), pd_datafram_PCA_selected_before_knee[pd_datafram_PCA_selected_before_knee['solution_id_dist'] == pd_dataframe_PCA_obs_real['solution_id'].iloc[ii]], pca_N_comp, fit_funct, rmsd_pol_mag, rmsd_t0_lag, trajectory_Metsim_file, file_name, pd_dataframe_PCA_obs_real['solution_id'].iloc[0], output_folder] for ii in range(len(pd_dataframe_PCA_obs_real))]
        results_list = domainParallelizer(input_list_obs, PCA_physicalProp_KDE_MODE_PLOT, cores=cml_args.cores)
    
        # if no read the json files in the folder and create a new csv file
        pd_datafram_PCA_selected_mode_min_KDE = pd.concat(results_list)

        pd_datafram_PCA_selected_mode_min_KDE_TOT = PCA_physicalProp_KDE_MODE_PLOT(pd_datafram_PCA_sim, pd_dataframe_PCA_obs_real, pd_datafram_PCA_selected_before_knee, pca_N_comp, fit_funct, rmsd_pol_mag, rmsd_t0_lag, trajectory_Metsim_file, file_name, pd_dataframe_PCA_obs_real['solution_id'].iloc[0], output_folder,True, True)

        # concatenate the two dataframes
        pd_datafram_PCA_selected_lowRMSD = pd.concat([pd_datafram_PCA_selected_mode_min_KDE_TOT, pd_datafram_PCA_selected_mode_min_KDE])

        # reset index
        pd_datafram_PCA_selected_lowRMSD.reset_index(drop=True, inplace=True)

        pd_datafram_PCA_selected_lowRMSD['type'] = 'Simulation_sel'
        
        # # save df_sel_shower_real to disk add the RMSD
        # pd_datafram_PCA_selected_lowRMSD.to_csv(output_folder+os.sep+SAVE_RESULTS_FOLDER+os.sep+file_name+'_sim_sel_results.csv', index=False)

        # print('PLOT: the physical characteristics of the selected simulations with no repetitions')
        # PCA_PhysicalPropPLOT(pd_datafram_PCA_selected_before_knee_NO_repetition, pd_datafram_PCA_sim, pca_N_comp, output_folder, file_name)

        # print(pd_datafram_PCA_selected_lowRMSD)
        # split in directory and filename
        filename_list = []
        # print(pd_datafram_PCA_selected_lowRMSD['solution_id'].values)
        if 'solution_id' in pd_datafram_PCA_selected_lowRMSD.columns:
            # check if in pd_datafram_PCA_selected_lowRMSD there is any json file that is not in the selected simulations
            for solution_id in pd_datafram_PCA_selected_lowRMSD['solution_id'].values:
                directory, filename = os.path.split(solution_id)
                filename_list.append(filename)
            # print(filename_list)
            json_files = [f for f in os.listdir(output_folder+os.sep+SAVE_RESULTS_FOLDER_EVENTS_PLOTS) if f.endswith('.json')]
            for json_file in json_files:
                folder_and_jsonfile_result = output_folder+os.sep+SAVE_RESULTS_FOLDER_EVENTS_PLOTS+os.sep+json_file
                if json_file not in filename_list:
                    # print that is found a json file that is not in the selected simulations
                    print(folder_and_jsonfile_result,'\njson file found in the Results directory that is not in '+file_name+'_sim_sel_results.csv')
                    f = open(folder_and_jsonfile_result,"r")
                    data = json.loads(f.read())
                    if 'ht_sampled' in data:
                        data_file = read_GenerateSimulations_output(folder_and_jsonfile_result, gensim_data_obs)
                        pd_datafram_PCA_sim_resulsts=array_to_pd_dataframe_PCA(data_file)
                    else:
                        _, _, pd_datafram_PCA_sim_resulsts = run_simulation(folder_and_jsonfile_result, gensim_data_obs)
                    # Add the simulation results to pd_datafram_PCA_selected_lowRMSD
                    pd_datafram_PCA_selected_lowRMSD = pd.concat([pd_datafram_PCA_selected_lowRMSD, pd_datafram_PCA_sim_resulsts])
            pd_datafram_PCA_selected_lowRMSD['type'] = 'Simulation_sel'
            pd_datafram_PCA_selected_lowRMSD.reset_index(drop=True, inplace=True)

        print()

        ######################## OPTIMIZATION ###############################

        if cml_args.optimize:
            print('--- OPTIMIZATION ---')

            # put the autoref option in the folder
            try:
                # Attempt to copy the file
                shutil.copy(cml_args.ref_opt_path, output_folder)
                print("File copied successfully.")
            except (FileNotFoundError, PermissionError) as e:
                # If there is an error, print an appropriate message
                print(f"Error: {e}")
                print("You need to specify the correct path and name of the AutoRefineFit_options.txt file, like: C:\\path\\AutoRefineFit_options.txt")
                raise 

        # plot the values and find the RMSD of each of them
        pd_datafram_PCA_selected_optimized = PCA_LightCurveRMSDPLOT_optimize(pd_datafram_PCA_selected_before_knee_NO_repetition, pd_dataframe_PCA_obs_real, output_folder, fit_funct, gensim_data_Metsim, rmsd_pol_mag, rmsd_t0_lag, file_name, cml_args.number_optimized, cml_args.optimize) # file_name, trajectory_Metsim_file, 
        
        # concatenate the two dataframes
        pd_datafram_PCA_selected_lowRMSD = pd.concat([pd_datafram_PCA_selected_optimized, pd_datafram_PCA_selected_lowRMSD])

        print('Check the manual reduction')
        # check also the manual reduction
        pd_datafram_PCA_selected_optimized_Metsim = PCA_LightCurveRMSDPLOT_optimize(pd_datafram_PCA_sim_Metsim, pd_dataframe_PCA_obs_real, output_folder, fit_funct, gensim_data_Metsim, rmsd_pol_mag, rmsd_t0_lag, file_name, cml_args.number_optimized, cml_args.optimize) # file_name, trajectory_Metsim_file, 
        
        # concatenate the two dataframes
        pd_datafram_PCA_selected_lowRMSD = pd.concat([pd_datafram_PCA_selected_optimized_Metsim, pd_datafram_PCA_selected_lowRMSD])

        # get all the json file in output_folder+os.sep+SAVE_RESULTS_FOLDER_EVENTS_PLOTS
        json_files_results = [f for f in os.listdir(output_folder+os.sep+SAVE_RESULTS_FOLDER_EVENTS_PLOTS) if f.endswith('.json')]
        # check if any json_files_results is in pd_datafram_PCA_selected_lowRMSD['solution_id'].values
        if 'solution_id' in pd_datafram_PCA_selected_lowRMSD.columns:
            for json_file in json_files_results:
                if json_file not in pd_datafram_PCA_selected_lowRMSD['solution_id'].values:
                    # print that is found a json file that is not in the selected simulations
                    print(output_folder+os.sep+SAVE_RESULTS_FOLDER_EVENTS_PLOTS+os.sep+json_file,'\njson file found in the Results directory that is not in '+file_name+'_sim_sel_results.csv')
                    f = open(output_folder+os.sep+SAVE_RESULTS_FOLDER_EVENTS_PLOTS+os.sep+json_file,"r")
                    data = json.loads(f.read())
                    if 'ht_sampled' in data:
                        data_file = read_GenerateSimulations_output(output_folder+os.sep+SAVE_RESULTS_FOLDER_EVENTS_PLOTS+os.sep+json_file)
                        pd_datafram_PCA_sim_resulsts=array_to_pd_dataframe_PCA(data_file)
                    else:
                        _, data_file, pd_datafram_PCA_sim_resulsts = run_simulation(output_folder+os.sep+SAVE_RESULTS_FOLDER_EVENTS_PLOTS+os.sep+json_file, gensim_data_obs)
                    
                    rmsd_mag, rmsd_vel, rmsd_lag, _, _, _, _, _ = RMSD_calc_diff(data_file, fit_funct)
                    # Add the simulation results that have a rmsd_mag and rmsd_len that is below RMSD to pd_datafram_PCA_selected_lowRMSD
                    if rmsd_mag < MAG_RMSD*SIGMA_ERR and rmsd_lag < LEN_RMSD*SIGMA_ERR:
                        # print to added to the selected simulations pd_datafram_PCA_sim_resulsts['solution_id'].values[0]
                        print('Added to the selected simulations:',output_folder+os.sep+SAVE_RESULTS_FOLDER_EVENTS_PLOTS+os.sep+json_file)
                        pd_datafram_PCA_selected_lowRMSD = pd.concat([pd_datafram_PCA_selected_lowRMSD, pd_datafram_PCA_sim_resulsts])
                        # pd_datafram_PCA_selected_lowRMSD['type'] = 'Simulation_sel'
                        # pd_datafram_PCA_selected_lowRMSD.reset_index(drop=True, inplace=True)

                
        print()

        ######################## RESULTS ###############################

        print('--- RESULTS ---')

        old_results_number = 0
        result_number = 0
        ii_repeat = 0
        pd_results = pd.DataFrame()
        # while cml_args.min_nres > result_number:
        print(cml_args.min_nres,'simulatd to found')
        while cml_args.min_nres > result_number:

            # reset index
            pd_datafram_PCA_selected_lowRMSD.reset_index(drop=True, inplace=True)

            pd_datafram_PCA_selected_lowRMSD['type'] = 'Simulation_sel'

            # delete any row from the csv file that has the same value of mass, rho, sigma, erosion_height_start, erosion_coeff, erosion_mass_index, erosion_mass_min, erosion_mass_max, erosion_range, erosion_energy_per_unit_cross_section, erosion_energy_per_unit_mass
            if 'mass' in pd_datafram_PCA_selected_lowRMSD.columns:                  
                # Drop duplicate rows based on the specified columns
                pd_datafram_PCA_selected_lowRMSD = pd_datafram_PCA_selected_lowRMSD.drop_duplicates(subset=[
                    'mass', 'rho', 'sigma', 'erosion_height_start', 'erosion_coeff', 
                    'erosion_mass_index', 'erosion_mass_min', 'erosion_mass_max', 
                    'erosion_range', 'erosion_energy_per_unit_cross_section', 
                    'erosion_energy_per_unit_mass'
                ])
                pd_datafram_PCA_selected_lowRMSD.reset_index(drop=True, inplace=True)

            pd_results = pd.concat([pd_results, pd_datafram_PCA_selected_lowRMSD])

            # save and update the disk 
            pd_results.to_csv(output_folder+os.sep+SAVE_RESULTS_FOLDER+os.sep+file_name+'_sim_sel_results.csv', index=False)

            
            if 'solution_id' in pd_results.columns:
                print('PLOT: the physical characteristics results')
                PCA_PhysicalPropPLOT(pd_results, pd_datafram_PCA_sim, pca_N_comp, output_folder+os.sep+SAVE_RESULTS_FOLDER, file_name)
                print('PLOT: correlation matrix of the results')
                PCAcorrelation_selPLOT(pd_datafram_PCA_sim, pd_results, pca_N_comp, output_folder+os.sep+SAVE_RESULTS_FOLDER)
                print('PLOT: best 9 results and add the RMSD value to csv selected')
                PCA_LightCurveCoefPLOT(pd_results, pd_dataframe_PCA_obs_real, output_folder+os.sep+SAVE_RESULTS_FOLDER, fit_funct, gensim_data_obs, rmsd_pol_mag, rmsd_t0_lag, file_name, trajectory_Metsim_file,output_folder+os.sep+SAVE_RESULTS_FOLDER+os.sep+file_name+'_sim_sel_results.csv')
                print()
                print('SUCCES: the physical characteristics range is in the results folder')
            else:
                # print('FAIL: Not found any result below magRMSD',rmsd_pol_mag*SIGMA_ERR,'and lenRMSD',rmsd_t0_lag*SIGMA_ERR/1000)
                print('FAIL: Not found any result below magRMSD',MAG_RMSD*SIGMA_ERR,'and lenRMSD',LEN_RMSD*SIGMA_ERR)
                break


            # check if only 1 in len break
            if len(pd_results) == 1:
                print('Only one result found')
                # create a dictionary with the physical parameters
                CI_physical_param = {
                    'v_init_180km': [pd_results['v_init_180km'].values[0], pd_results['v_init_180km'].values[0]],
                    'zenith_angle': [pd_results['zenith_angle'].values[0], pd_results['zenith_angle'].values[0]],
                    'mass': [pd_results['mass'].values[0], pd_results['mass'].values[0]],
                    'rho': [pd_results['rho'].values[0], pd_results['rho'].values[0]],
                    'sigma': [pd_results['sigma'].values[0], pd_results['sigma'].values[0]],
                    'erosion_height_start': [pd_results['erosion_height_start'].values[0], pd_results['erosion_height_start'].values[0]],
                    'erosion_coeff': [pd_results['erosion_coeff'].values[0], pd_results['erosion_coeff'].values[0]],
                    'erosion_mass_index': [pd_results['erosion_mass_index'].values[0], pd_results['erosion_mass_index'].values[0]],
                    'erosion_mass_min': [pd_results['erosion_mass_min'].values[0], pd_results['erosion_mass_min'].values[0]],
                    'erosion_mass_max': [pd_results['erosion_mass_max'].values[0], pd_results['erosion_mass_max'].values[0]]
                }

            else:
                print('Number of results found:',len(pd_results))
                columns_physpar = ['v_init_180km','zenith_angle','mass', 'rho', 'sigma', 'erosion_height_start', 'erosion_coeff', 
                    'erosion_mass_index', 'erosion_mass_min', 'erosion_mass_max']
                
                ###############################################################################
                
                # # Calculate the quantiles
                # quantiles = pd_results[columns_physpar].quantile([0.05, 0.95])

                # # Convert the quantiles to a dictionary
                # CI_physical_param = {col: quantiles[col].tolist() for col in columns_physpar}

                ###############################################################################

                # Calculate the quantiles
                quantiles = pd_results[columns_physpar].quantile([0.1, 0.9])

                # Get the minimum and maximum values
                min_val = pd_results[columns_physpar].min()
                max_val = pd_results[columns_physpar].max()

                # Calculate the extended range using the logic provided
                extended_min = min_val - (quantiles.loc[0.1] - min_val)
                # consider the value extended_min<0 Check each column in extended_min and set to min_val if negative
                for col in columns_physpar:
                    if extended_min[col] < 0:
                        extended_min[col] = min_val[col]
                extended_max = max_val + (max_val - quantiles.loc[0.9])

                # Convert the extended range to a dictionary
                CI_physical_param = {col: [extended_min[col], extended_max[col]] for col in columns_physpar}
                
                ###############################################################################


            # check if v_init_180km are the same value
            if CI_physical_param['v_init_180km'][0] == CI_physical_param['v_init_180km'][1]:
                CI_physical_param['v_init_180km'] = [CI_physical_param['v_init_180km'][0] - CI_physical_param['v_init_180km'][0]/1000, CI_physical_param['v_init_180km'][1] + CI_physical_param['v_init_180km'][1]/1000]
            if CI_physical_param['zenith_angle'][0] == CI_physical_param['zenith_angle'][1]:
                CI_physical_param['zenith_angle'] = [CI_physical_param['zenith_angle'][0] - CI_physical_param['zenith_angle'][0]/10000, CI_physical_param['zenith_angle'][1] + CI_physical_param['zenith_angle'][1]/10000]
            if CI_physical_param['mass'][0] == CI_physical_param['mass'][1]:
                CI_physical_param['mass'] = [CI_physical_param['mass'][0] - CI_physical_param['mass'][0]/10, CI_physical_param['mass'][1] + CI_physical_param['mass'][1]/10]
            if np.round(CI_physical_param['rho'][0]/100) == np.round(CI_physical_param['rho'][1]/100):
                CI_physical_param['rho'] = [CI_physical_param['rho'][0] - CI_physical_param['rho'][0]/5, CI_physical_param['rho'][1] + CI_physical_param['rho'][1]/5]
            if CI_physical_param['sigma'][0] == CI_physical_param['sigma'][1]:
                CI_physical_param['sigma'] = [CI_physical_param['sigma'][0] - CI_physical_param['sigma'][0]/10, CI_physical_param['sigma'][1] + CI_physical_param['sigma'][1]/10]
            if CI_physical_param['erosion_height_start'][0] == CI_physical_param['erosion_height_start'][1]:
                CI_physical_param['erosion_height_start'] = [CI_physical_param['erosion_height_start'][0] - CI_physical_param['erosion_height_start'][0]/100, CI_physical_param['erosion_height_start'][1] + CI_physical_param['erosion_height_start'][1]/100]
            if CI_physical_param['erosion_coeff'][0] == CI_physical_param['erosion_coeff'][1]:
                CI_physical_param['erosion_coeff'] = [CI_physical_param['erosion_coeff'][0] - CI_physical_param['erosion_coeff'][0]/10, CI_physical_param['erosion_coeff'][1] + CI_physical_param['erosion_coeff'][1]/10]
            if CI_physical_param['erosion_mass_index'][0] == CI_physical_param['erosion_mass_index'][1]:
                CI_physical_param['erosion_mass_index'] = [CI_physical_param['erosion_mass_index'][0] - CI_physical_param['erosion_mass_index'][0]/10, CI_physical_param['erosion_mass_index'][1] + CI_physical_param['erosion_mass_index'][1]/10]
            if CI_physical_param['erosion_mass_min'][0] == CI_physical_param['erosion_mass_min'][1]:
                CI_physical_param['erosion_mass_min'] = [CI_physical_param['erosion_mass_min'][0] - CI_physical_param['erosion_mass_min'][0]/10, CI_physical_param['erosion_mass_min'][1] + CI_physical_param['erosion_mass_min'][1]/10]
            if CI_physical_param['erosion_mass_max'][0] == CI_physical_param['erosion_mass_max'][1]:
                CI_physical_param['erosion_mass_max'] = [CI_physical_param['erosion_mass_max'][0] - CI_physical_param['erosion_mass_max'][0]/10, CI_physical_param['erosion_mass_max'][1] + CI_physical_param['erosion_mass_max'][1]/10]
                

            # Multiply the 'erosion_height_start' values by 1000
            CI_physical_param['erosion_height_start'] = [x * 1000 for x in CI_physical_param['erosion_height_start']]

            print('CI_physical_param:',CI_physical_param)

            result_number = len(pd_results)

            if cml_args.min_nres <= result_number:
                # print the number of results found
                print('SUCCES: Number of results found:',result_number)
                break
            else:
                if old_results_number == result_number:
                    print('Same number of results found:',result_number)
                    ii_repeat+=1
                if ii_repeat==3:
                    print('STOP: After 3 times the same number of results found')
                    print('STOP: After new simulation within 95%CI no new simulation below magRMSD',MAG_RMSD*SIGMA_ERR,'and lenRMSD',LEN_RMSD*SIGMA_ERR)
                    print('STOP: Number of results found:',result_number)
                    break
                old_results_number = result_number
                print('regenerate new simulation in the CI range')
                generate_simulations(pd_dataframe_PCA_obs_real, simulation_MetSim_object, gensim_data_obs, cml_args.min_nres, output_folder, file_name, False, CI_physical_param)
                
                # look for the good_files = glob.glob(os.path.join(output_folder, '*_good_files.txt'))
                good_files = [f for f in os.listdir(output_folder) if f.endswith('_good_files.txt')]                

                # Construct the full path to the good file
                good_file_path = os.path.join(output_folder, good_files[0])

                # Read the file, skipping the first line
                df_good_files = pd.read_csv(good_file_path, skiprows=1)

                # Rename the columns
                df_good_files.columns = ["File name", "lim mag", "lim mag length", "length delay (s)"]

                # Extract the first column into an array
                file_names = df_good_files["File name"].to_numpy()

                # Change the file extension to .json
                all_jsonfiles = [file_name.replace('.pickle', '.json') for file_name in file_names]

                # open the folder and extract all the json files
                os.chdir(input_folder)

                print('Number of simulated files in 95CI : ',len(all_jsonfiles))

                input_list = [[all_jsonfiles[ii], 'simulation_'+str(ii+1)] for ii in range(len(all_jsonfiles))]
                results_list = domainParallelizer(input_list, read_GenerateSimulations_output_to_PCA, cores=cml_args.cores)
                
                # if no read the json files in the folder and create a new csv file
                pd_datafram_NEWsim_good = pd.concat(results_list)

                pd_datafram_NEWsim_good.to_csv(output_folder+os.sep+file_name+NAME_SUFX_CSV_SIM_NEW, index=False)
                # print saved csv file
                print('saved sim csv file:',output_folder+os.sep+file_name+NAME_SUFX_CSV_SIM_NEW)

                input_list_obs = [[pd_datafram_NEWsim_good.iloc[[ii]].reset_index(drop=True), pd_dataframe_PCA_obs_real, output_folder, fit_funct, gensim_data_Metsim, rmsd_pol_mag, rmsd_t0_lag, file_name, 0, False] for ii in range(len(pd_datafram_NEWsim_good))]
                results_list = domainParallelizer(input_list_obs, PCA_LightCurveRMSDPLOT_optimize, cores=cml_args.cores)

                # base on the one selected
                pd_datafram_PCA_selected_lowRMSD = pd.concat(results_list)
            
        # Timing end
        end_time = time.time()
        
        # Compute elapsed time
        elapsed_time = end_time - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        # print('Elapsed time in seconds:',elapsed_time)
        print(f"Elapsed time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")

        print()


