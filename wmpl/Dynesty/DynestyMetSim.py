"""
Nested sampling with Dynesty for MetSim meteor data, generate plots and tables 
for given trajectory.pickle file or generate new observation from metsim.json solution file
for EMCCD and CAMO cameras.

Author: Maximilian Vovk
Date: 2025-02-25
"""


import pickle
import gzip
import sys
import json, base64
import os
import io
import copy
import shutil
import time
import warnings
import datetime
import re
import multiprocessing
import math
import signal
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed


import numpy as np
import pandas as pd
import scipy
from scipy.ndimage import gaussian_filter as norm_kde
from scipy.optimize import minimize,curve_fit
from scipy.stats import norm, invgamma
# Import the correct scipy.integrate.simpson function
try:
    from scipy.integrate import simps as simpson
except ImportError:
    from scipy.integrate import simpson as simpson

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator, ScalarFormatter, FormatStrFormatter
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec

try:
    import dynesty
    from dynesty import plotting as dyplot
    from dynesty.utils import quantile as _quantile
    DYNESTY_FOUND = True
except ImportError:
    print("Dynesty package not found. Install dynesty to use the Dynesty functions.")
    DYNESTY_FOUND = False

from wmpl.MetSim.GUI import loadConstants, SimulationResults, FragmentationEntry, saveConstants
from wmpl.MetSim.MetSimErosion import runSimulation, Constants, zenithAngleAtSimulationBegin
from wmpl.Utils.Math import lineFunc, mergeClosePoints, meanAngle
from wmpl.MetSim.ML.GenerateSimulations import generateErosionSim,saveProcessedList,MetParam
from wmpl.Utils.Physics import calcMass, dynamicPressure, calcRadiatedEnergy
from wmpl.Utils.TrajConversions import J2000_JD, date2JD
from wmpl.Utils.AtmosphereDensity import fitAtmPoly
from wmpl.Utils.Pickling import loadPickle
from wmpl.MetSim.MetSimErosionCyTools import luminousEfficiency


class TimeoutException(Exception):
    """Custom exception for timeouts."""
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Function execution timed out")

# create a txt file where you save everything that has been printed
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


def jsonDefault(o):
    # numpy arrays -> lists
    if isinstance(o, np.ndarray):
        return o.tolist()

    # numpy scalars -> python scalars
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()

    # sets -> lists
    if isinstance(o, set):
        return list(o)

    # custom objects -> dict (best effort)
    if hasattr(o, "__dict__"):
        return vars(o)

    # last resort (keeps the dump from crashing)
    return str(o)



def meteorAbsMagnitudeToApparent(abs_mag, distance):
    """ Calculate apparent magnitude from absolute magnitude and distance.

    Either a single set of values can be given (i.e. every argument is a float number), or all arguments 
    must be numpy arrays.
        
    Arguments:
        abs_mag: [float or ndarray] Absolute magnitude of the meteor.
        distance: [float or ndarray] Distance to the meteor (meters).

    Return:
        apparent_mag: [float or ndarray] Apparent magnitude.

    """
    
    return 5*np.log10(distance/100000) + abs_mag


def _prep_unique_sorted(x, y):
    o = np.argsort(x)
    x = np.asarray(x)[o]
    y = np.asarray(y)[o]
    _, ui = np.unique(x, return_index=True)
    return x[ui], y[ui]

def buildGlobalTimeAxis(
    data,
    height_key="height",
    time_key="time",
    station_key="flag_station",
    length_key="length",     # optional in data
    lag_key="lag",           # optional in data
    v_init=None,             # if given, recompute lag = length - v_init*time
    min_overlap_pts=8,
    offset_tol=2e-3,):
    """ Align station clocks to the station that starts at the highest altitude.

    If length exists, also align length and re-zero it at the same global start.
    If v_init is provided and length exists, recompute lag from (length,time).

    Arguments:
        data: [dict] Dictionary containing the observation data (height, time, station, etc.).

    Keyword arguments:
        height_key: [str] Key for height data. "height" by default.
        time_key: [str] Key for time data. "time" by default.
        station_key: [str] Key for station identifiers. "flag_station" by default.
        length_key: [str] Key for length data. "length" by default.
        lag_key: [str] Key for lag data. "lag" by default.
        v_init: [float] Initial velocity. If provided, lag is recomputed. None by default.
        min_overlap_pts: [int] Minimum number of overlapping points required for alignment. 8 by default.
        offset_tol: [float] Offset tolerance for time alignment. 2e-3 by default.

    Return:
        out: [dict] Updated data dictionary with aligned time (and optionally length/lag).
        did_align: [bool] True if alignment was performed.

    """
    out = {k: np.asarray(v) for k, v in data.items()}
    h = out[height_key].astype(float)
    t = out[time_key].astype(float)
    s = out[station_key]

    stations = np.unique(s)
    if stations.size <= 1:
        # sort by time + set t0 at first finite
        idx = np.argsort(t)
        out = {k: np.asarray(v)[idx] for k, v in out.items()}
        return out, False

    # Reference station = highest max height
    maxh = {st: np.nanmax(h[s == st]) for st in stations}
    ref = max(maxh, key=maxh.get)

    # ref t(h)
    href, tref = _prep_unique_sorted(h[s == ref], t[s == ref])
    href_min, href_max = np.nanmin(href), np.nanmax(href)
    def ref_of_h(hq): return np.interp(hq, href, tref)

    # need alignment?
    need_align = False
    for st in stations:
        if st == ref:
            continue
        hs, ts = _prep_unique_sorted(h[s == st], t[s == st])
        h_lo = max(href_min, np.nanmin(hs))
        h_hi = min(href_max, np.nanmax(hs))
        if h_hi <= h_lo:
            continue
        h_match = hs[(hs >= h_lo) & (hs <= h_hi)]
        if h_match.size < min_overlap_pts:
            continue
        dt = ref_of_h(h_match) - np.interp(h_match, hs, ts)
        if np.abs(np.nanmedian(dt)) > offset_tol:
            need_align = True
            break

    # Align time (and optionally length/lag)
    if need_align:
        t_new = t.copy()

        # If we have length, we will align it too
        has_length = (length_key in out)
        L_new = out[length_key].astype(float).copy() if has_length else None

        for st in stations:
            if st == ref:
                continue
            ms = (s == st)
            hs, ts = _prep_unique_sorted(h[ms], t[ms])

            h_lo = max(href_min, np.nanmin(hs))
            h_hi = min(href_max, np.nanmax(hs))
            if h_hi <= h_lo:
                continue

            h_match = hs[(hs >= h_lo) & (hs <= h_hi)]
            if h_match.size < min_overlap_pts:
                h_match = np.array([h_lo, h_hi], dtype=float)

            # time offset
            offset_t = np.nanmedian(ref_of_h(h_match) - np.interp(h_match, hs, ts))
            t_new[ms] = t[ms] + offset_t

            # length offset (match length(h) on overlap)
            if has_length:
                href2, Lref = _prep_unique_sorted(h[s == ref], out[length_key][s == ref])
                def Lref_of_h(hq): return np.interp(hq, href2, Lref)

                hs2, Ls = _prep_unique_sorted(h[ms], out[length_key][ms])
                offset_L = np.nanmedian(Lref_of_h(h_match) - np.interp(h_match, hs2, Ls))
                L_new[ms] = out[length_key][ms] + offset_L

        out[time_key] = t_new
        if has_length:
            out[length_key] = L_new

        # Global start = reference station top point (highest height)
        ref_mask = (s == ref)
        idx_ref_top = np.nanargmax(h[ref_mask])

        t0 = out[time_key][ref_mask][idx_ref_top]
        out[time_key] = out[time_key] - t0

        if has_length:
            L0 = out[length_key][ref_mask][idx_ref_top]
            out[length_key] = out[length_key] - L0

        # Recompute lag consistently if possible
        if has_length and (v_init is not None):
            out[lag_key] = out[length_key] - (float(v_init)*out[time_key])
        elif lag_key in out:
            # at least re-zero lag at the same global start
            lag0 = out[lag_key][ref_mask][idx_ref_top]
            out[lag_key] = out[lag_key] - lag0

    # Finally sort by time (required for integration)
    idx = np.argsort(out[time_key])
    out = {k: np.asarray(v)[idx] for k, v in out.items()}

    # Drop duplicate times
    tt = out[time_key]
    _, keep = np.unique(tt, return_index=True)
    keep = np.sort(keep)
    out = {k: np.asarray(v)[keep] for k, v in out.items()}

    return out, need_align



###############################################################################
# Function: plotting function
###############################################################################

def plotJSONDataVsObs(obs_data, out_folder, best_noise_lum=0, best_noise_lag=0):
    """ Plot data from json files in the output folder against observations.
    
    The real LogL is computed with the best noise levels if provided. Initially is defined as guess.

    Arguments:
        obs_data: [object] Object containing the observational data.
        out_folder: [str] Path to the output folder.

    Keyword arguments:
        best_noise_lum: [float] Best noise level for luminosity. 0 by default.
        best_noise_lag: [float] Best noise level for lag. 0 by default.

    Return:
        None

    """

    # check if there are json files in the output folder that could be plotted
    json_files = [f for f in os.listdir(out_folder) if f.endswith('.json')]

    # delete any json file that start with obs_data_
    json_files = [f for f in json_files if not f.startswith('obs_data_')]

    # delete any json file that ends with _with_noise.json
    json_files = [f for f in json_files if not f.endswith('_with_noise.json')]
    
    for const_json_name in json_files:
        print(f"Plotting simulation from json file: {const_json_name}")
        # create the full path to the json file
        const_json_file = os.path.join(out_folder, const_json_name)
        try:
            # Load the constants from the JSON files
            const_manual, _ = loadConstants(const_json_file)
            const_manual.dens_co = np.array(const_manual.dens_co)
            # # Run the simulation
            # frag_main, results_list, wake_results = runSimulation(const_manual, compute_wake=False)
            # simulation_manual_MetSim_object = SimulationResults(const_manual, frag_main, results_list, wake_results)
            # delete the .json from the name:
            json_name = const_json_name[:-5]

            # create a fake guess_real dictionary to run logLikelihoodDynesty
            fixed_values_manual = {}   
            for key in const_manual.__dict__.keys():
                # put all the keys in fixed_values_manual
                fixed_values_manual[key] = const_manual.__dict__[key]
            flags_dict_manual = {"v_init": ["norm"],
                                    "zenith_angle": ["norm"]}
            guess_manual = [const_manual.v_init, const_manual.zenith_angle]
            # reate a folder to save the plots from the json files
            json_plots_folder = os.path.join(out_folder, "json_plots")
            if not os.path.exists(json_plots_folder):
                os.makedirs(json_plots_folder)
            if best_noise_lum!=0 and best_noise_lag!=0:
                guess_manual.extend([best_noise_lum, best_noise_lag])
                flags_dict_manual["noise_lum"] = ["invgamma"]
                flags_dict_manual["noise_lag"] = ["invgamma"]
            
            # compute log likelihood between obs_data and simulation_manual_MetSim_object
            manual_logL = logLikelihoodDynesty(guess_manual, obs_data, flags_dict_manual, fixed_values_manual, timeout=20)
            # run the simulation with the same parameters and same proecess as in run_simulation
            var_names_manual = list(flags_dict_manual.keys())
            simulation_manual_MetSim_object = runSimulationDynesty(guess_manual, obs_data, var_names_manual, fixed_values_manual)
            if best_noise_lum!=0 and best_noise_lag!=0:
                print(f"{json_name} real LogL = {manual_logL:.1f}")
                # Plot the data with residuals and the best fit
                plotSimVsObsResiduals(obs_data, simulation_manual_MetSim_object, json_plots_folder, json_name, color_sim='slategray', label_sim=f'LogL={manual_logL:.1f}')
                plotObsVsHeight(obs_data, simulation_manual_MetSim_object, json_plots_folder, json_name, color_sim='slategray', label_sim=f'LogL={manual_logL:.1f}')

            else:
                print(f"{json_name} intial guess LogL ~ {manual_logL:.1f}")
                # Plot the data with residuals and the best fit
                plotSimVsObsResiduals(obs_data, simulation_manual_MetSim_object, json_plots_folder, json_name, color_sim='slategray', label_sim=f'LogL$\\approx${manual_logL:.1f}')
                plotObsVsHeight(obs_data, simulation_manual_MetSim_object, json_plots_folder, json_name, color_sim='slategray', label_sim=f'LogL$\\approx${manual_logL:.1f}')

        except Exception as e:
            print(f"Error encountered loading json file {const_json_file}: {e}")


# Plotting function
def plotSimVsObsResiduals(obs_data, sim_data=None, output_folder='', file_name='', color_sim='black', label_sim='Best Fit'):
    """ Plot the observations vs simulation data with residuals.

    Arguments:
        obs_data: [object] Object containing the observational data.

    Keyword arguments:
        sim_data: [object] Object containing the simulation data. None by default.
        output_folder: [str] Path to the output folder. Empty string by default.
        file_name: [str] Name of the file. Empty string by default.
        color_sim: [str] Color for the simulation plot. 'black' by default.
        label_sim: [str] Label for the simulation plot. 'Best Fit' by default.

    Return:
        None

    """

    # Create the figure and main GridSpec with specified height ratios
    fig = plt.figure(figsize=(14, 6))
    gs_main = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1])

    # Define colormap
    cmap = plt.get_cmap("tab10")
    station_colors = {}  # Dictionary to store colors assigned to stations

    ### ABSOLUTE MAGNITUDES PLOT ###

    # Create a sub GridSpec for Plot 0 and Plot 1 with width ratios
    gs01 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[0, 0:2], wspace=0, width_ratios=[3, 1])

    # Plot 0 and 1: Side by side, sharing the y-axis
    ax0 = fig.add_subplot(gs01[0])
    ax1 = fig.add_subplot(gs01[1], sharey=ax0)

    # for each station in obs_data_plot
    for station in np.unique(obs_data.stations_lum):
        # Assign a unique color if not already used
        if station not in station_colors:
            station_colors[station] = cmap(len(station_colors) % 10)  # Use modulo to cycle colors
        # plot the height vs. absolute_magnitudes
        ax0.plot(obs_data.absolute_magnitudes[np.where(obs_data.stations_lum == station)], \
                 obs_data.height_lum[np.where(obs_data.stations_lum == station)]/1000, 'x--', \
                 color=station_colors[station], label=station)
    # chek if np.unique(obs_data.stations_lag) and np.unique(obs_data.stations_lum) are the same
    # print('testing unique stations plot',np.unique(obs_data.stations_lag), np.unique(obs_data.stations_lum))
    if not np.array_equal(np.unique(obs_data.stations_lag), np.unique(obs_data.stations_lum)):
        # take the one that are not in the other in lag
        stations_lag = np.setdiff1d(np.unique(obs_data.stations_lag), np.unique(obs_data.stations_lum))
        if len(stations_lag) != 0:
            # take the one that are shared between lag and lum
            # stations_lag = np.intersect1d(np.unique(obs_data.stations_lag), np.unique(obs_data.stations_lum))
            # print('stations_lag',stations_lag)
            # Suppose stations_lag is your array of station IDs you care about
            mask = np.isin(obs_data.stations_lag, stations_lag)
            # Filter heights for only those stations
            filtered_heights = obs_data.height_lag[mask]
            # Get the maximum of that subset
            max_height_lag = filtered_heights.max()
            # print a horizonal along the x axis at the height_lag[0] darkgray
            ax0.axhline(y=max_height_lag/1000, color='gray', linestyle='-.', linewidth=1, label=f"{', '.join(stations_lag)}", zorder=2)

    ax0.set_xlabel('Absolute Magnitudes')
    # flip the x-axis
    ax0.invert_xaxis()
    ax0.legend()
    # ax0.tick_params(axis='x', rotation=45)
    ax0.set_ylabel('Height [km]')
    ax0.grid(True, linestyle='--', color='lightgray')
    # save the x-axis limits
    xlim_abs_mag = ax0.get_xlim()
    # fix the x-axis limits to xlim_abs_mag
    ax0.set_xlim(xlim_abs_mag)
    # save the y-axis limits
    ylim_abs_mag = ax0.get_ylim()
    # fix the y-axis limits to ylim_abs_mag
    ax0.set_ylim(ylim_abs_mag)
    

    ax1.fill_betweenx([np.min(obs_data.height_lum)/1000,np.max(obs_data.height_lum)/1000], -obs_data.noise_mag, obs_data.noise_mag, color='darkgray', alpha=0.2)
    ax1.fill_betweenx([np.min(obs_data.height_lum)/1000,np.max(obs_data.height_lum)/1000], -obs_data.noise_mag*2, obs_data.noise_mag*2, color='lightgray', alpha=0.2)
    ax1.plot([0, 0], [np.min(obs_data.height_lum)/1000,np.max(obs_data.height_lum)/1000],color='lightgray')
    ax1.set_xlabel('Res.Mag')
    # flip the x-axis
    # ax1.invert_xaxis()
    # ax1.tick_params(axis='x', rotation=45)
    ax1.tick_params(labelleft=False)  # Hide y-axis tick labels
    ax1.grid(True, linestyle='--', color='lightgray')

    ### LUMINOSITY PLOT ###

    # Create a sub GridSpec for Plot 0 and Plot 1 with width ratios
    gs02 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[1, 0:2], wspace=0, width_ratios=[3, 1])

    # Plot 0 and 1: Side by side, sharing the y-axis
    ax4 = fig.add_subplot(gs02[0])
    ax5 = fig.add_subplot(gs02[1], sharey=ax4)

    # for each station in obs_data_plot
    for station in np.unique(obs_data.stations_lum):
        # Assign a unique color if not already used
        if station not in station_colors:
            station_colors[station] = cmap(len(station_colors) % 10)  # Use modulo to cycle colors
        # plot the height vs. absolute_magnitudes
        ax4.plot(obs_data.luminosity[np.where(obs_data.stations_lum == station)], \
                 obs_data.height_lum[np.where(obs_data.stations_lum == station)]/1000, 'x--', \
                 color=station_colors[station], label=station)
    # chek if np.unique(obs_data.stations_lag) and np.unique(obs_data.stations_lum) are the same
    if not np.array_equal(np.unique(obs_data.stations_lag), np.unique(obs_data.stations_lum)):
        if len(stations_lag) != 0:
            # print a horizonal along the x axis at the height_lag[0] darkgray
            ax4.axhline(y=max_height_lag/1000, color='gray', linestyle='-.', linewidth=1, label=f"{', '.join(stations_lag)}", zorder=2)
    ax4.set_xlabel('Luminosity [W]')
    # ax4.tick_params(axis='x', rotation=45)
    ax4.set_ylabel('Height [km]')
    ax4.grid(True, linestyle='--', color='lightgray')
    # save the x-axis limits
    xlim_lum = ax4.get_xlim()
    # fix the x-axis limits to xlim_lum
    ax4.set_xlim(xlim_lum)
    # save the y-axis limits
    ylim_lum = ax4.get_ylim()
    # fix the y-axis limits to ylim_lum
    ax4.set_ylim(ylim_lum)

    ax5.fill_betweenx([np.min(obs_data.height_lum)/1000,np.max(obs_data.height_lum)/1000], -obs_data.noise_lum, obs_data.noise_lum, color='darkgray', alpha=0.2)
    ax5.fill_betweenx([np.min(obs_data.height_lum)/1000,np.max(obs_data.height_lum)/1000], -obs_data.noise_lum*2, obs_data.noise_lum*2, color='lightgray', alpha=0.2)
    ax5.plot([0, 0], [np.min(obs_data.height_lum)/1000,np.max(obs_data.height_lum)/1000],color='lightgray')
    ax5.set_xlabel('Res.Lum [J/s]')
    # ax5.tick_params(axis='x', rotation=45)
    ax5.tick_params(labelleft=False)  # Hide y-axis tick labels
    ax5.grid(True, linestyle='--', color='lightgray')

    ### VELOCITY PLOT ###

    # Plot 2 and 6: Vertically stacked, sharing the x-axis (Time) with height ratios
    gs_col2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[:, 2], hspace=0, height_ratios=[3, 1])
    ax2 = fig.add_subplot(gs_col2[0, 0])
    ax6 = fig.add_subplot(gs_col2[1, 0], sharex=ax2)

    # for each station in obs_data_plot
    for station in np.unique(obs_data.stations_lag):
        # Assign a unique color if not already used
        if station not in station_colors:
            station_colors[station] = cmap(len(station_colors) % 10)  # Use modulo to cycle colors
        vel_plot= obs_data.velocities[np.where(obs_data.stations_lag == station)]/1000
        time_plot = obs_data.time_lag[np.where(obs_data.stations_lag == station)]
        # plot the height vs. absolute_magnitudes
        ax2.plot(time_plot[1:], \
                 vel_plot[1:], '.', \
                 color=station_colors[station], label=station)
    ax2.set_ylabel('Velocity [km/s]')
    ax2.legend()
    ax2.tick_params(labelbottom=False)  # Hide x-axis tick labels
    ax2.grid(True, linestyle='--', color='lightgray')
    # save the x-axis limits
    xlim_vel = ax2.get_xlim()
    # fix the x-axis limits to xlim_vel
    ax2.set_xlim(xlim_vel)
    # save the y-axis limits
    ylim_vel = ax2.get_ylim()
    # fix the y-axis limits to ylim_vel
    ax2.set_ylim(ylim_vel)

    
    # pick the second to first tat is the samllest between the all the stations
    for station in np.unique(obs_data.stations_lag):
        station_time = obs_data.time_lag[np.where(obs_data.stations_lag == station)]
        if 'second_last_index' not in locals():
            second_last_index = np.argsort(station_time)[1]
        else:
            second_last_index = min(second_last_index, np.argsort(station_time)[1])
    # Plot 6: Res.Vel vs. Time
    ax6.fill_between([obs_data.time_lag[second_last_index], np.max(obs_data.time_lag)], -obs_data.noise_vel/1000, obs_data.noise_vel/1000, color='darkgray', alpha=0.2)
    ax6.fill_between([obs_data.time_lag[second_last_index], np.max(obs_data.time_lag)], -obs_data.noise_vel*2/1000, obs_data.noise_vel*2/1000, color='lightgray', alpha=0.2)
    ax6.plot([obs_data.time_lag[second_last_index], np.max(obs_data.time_lag)], [0, 0], color='lightgray')
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Res.Vel [km/s]')
    ax6.grid(True, linestyle='--', color='lightgray')

    ### LAG PLOT ###

    # Plot 3 and 7: Vertically stacked, sharing the x-axis (Time) with height ratios
    gs_col3 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[:, 3], hspace=0, height_ratios=[3, 1])
    ax3 = fig.add_subplot(gs_col3[0, 0])
    ax7 = fig.add_subplot(gs_col3[1, 0], sharex=ax3)

    # for each station in obs_data_plot
    for station in np.unique(obs_data.stations_lag):
        # Assign a unique color if not already used
        if station not in station_colors:
            station_colors[station] = cmap(len(station_colors) % 10)  # Use modulo to cycle colors
        # plot the height vs. absolute_magnitudes
        ax3.plot(obs_data.time_lag[np.where(obs_data.stations_lag == station)], \
                 obs_data.lag[np.where(obs_data.stations_lag == station)], 'x:', \
                 color=station_colors[station], label=station)
    ax3.set_ylabel('Lag [m]')
    ax3.tick_params(labelbottom=False)  # Hide x-axis tick labels
    ax3.grid(True, linestyle='--', color='lightgray')
    # save the x-axis limits
    xlim_lag = ax3.get_xlim()
    # fix the x-axis limits to xlim_lag
    ax3.set_xlim(xlim_lag)
    # save the y-axis limits
    ylim_lag = ax3.get_ylim()
    # fix the y-axis limits to ylim_lag
    ax3.set_ylim(ylim_lag)

    # Plot 7: Res.Vel vs. Time
    ax7.fill_between([np.min(obs_data.time_lag), np.max(obs_data.time_lag)], -obs_data.noise_lag, obs_data.noise_lag, color='darkgray', alpha=0.2)
    ax7.fill_between([np.min(obs_data.time_lag), np.max(obs_data.time_lag)], -obs_data.noise_lag*2, obs_data.noise_lag*2, color='lightgray', alpha=0.2)
    ax7.plot([np.min(obs_data.time_lag), np.max(obs_data.time_lag)], [0, 0], color='lightgray')
    ax7.set_xlabel('Time [s]')
    ax7.set_ylabel('Res.Lag [m]')
    ax7.grid(True, linestyle='--', color='lightgray')

    # Adjust the overall layout to prevent overlap
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    # make the suptitle
    # fig.suptitle(file_name)

    # check if 'const' in the object obs_data.keys()
    if hasattr(obs_data, 'const'):

        ax0.plot(obs_data.abs_magnitude, obs_data.leading_frag_height_arr/1000, '--', color='black', linewidth=0.5, label='No Noise', zorder=2)
        # inerpoate the abs_magnitude_arr to the leading_frag_height_arr
        no_noise_mag = np.interp(obs_data.height_lum, 
                                        np.flip(obs_data.leading_frag_height_arr), 
                                        np.flip(obs_data.abs_magnitude))
        ax0.legend()
        # make the difference between the no_noise_mag and the obs_data.abs_magnitude
        diff_mag = no_noise_mag - obs_data.absolute_magnitudes
        ax1.plot(diff_mag, obs_data.height_lum/1000, '.', markersize=3, color='black', label='No Noise')
        
        # # for ax5 add a noise that changes for the left and right side of the curve base on the -2.5*np.log10((self.luminosity_arr+self.noise_lum)/self.P_0m) and 2.5*np.log10((self.luminosity_arr+self.noise_lum)/self.P_0m)
        # ax1.fill_betweenx(obs_data.leading_frag_height_arr/1000, \
        #                   -2.5*(np.log10((obs_data.luminosity_arr-obs_data.noise_lum)/obs_data.P_0m)-np.log10((obs_data.luminosity_arr)/obs_data.P_0m)), \
        #                   -2.5*(np.log10((obs_data.luminosity_arr+obs_data.noise_lum)/obs_data.P_0m)-np.log10((obs_data.luminosity_arr)/obs_data.P_0m)), \
        #                     color='darkgray', alpha=0.2)
        # ax1.fill_betweenx(obs_data.leading_frag_height_arr/1000, \
        #                   -2.5*(np.log10((obs_data.luminosity_arr-2*obs_data.noise_lum)/obs_data.P_0m)-np.log10((obs_data.luminosity_arr)/obs_data.P_0m)), \
        #                   -2.5*(np.log10((obs_data.luminosity_arr+2*obs_data.noise_lum)/obs_data.P_0m)-np.log10((obs_data.luminosity_arr)/obs_data.P_0m)), \
        #                     color='lightgray', alpha=0.2)

        ax4.plot(obs_data.luminosity_arr, obs_data.leading_frag_height_arr/1000, '--', color='black', linewidth=0.5, label='No Noise', zorder=2)
        
        # interpolate to make sure they are the same length and discard points after height starts increasing if it does at any point obs_metsim_obj.traj.observations[0].model_ht
        no_noise_lum = np.interp(obs_data.height_lum, 
                                        np.flip(obs_data.leading_frag_height_arr), 
                                        np.flip(obs_data.luminosity_arr))

        # make the difference between the no_noise_intensity and the obs_data.luminosity_arr
        diff_lum = obs_data.luminosity - no_noise_lum
        ax5.plot(diff_lum, obs_data.height_lum/1000, '.', markersize=3, color='black', label='No Noise')


        # find the obs_data.leading_frag_height_arr index is close to np.max(obs_data.height_lum) wihouth nan
        index = np.argmin(np.abs(obs_data.leading_frag_height_arr[~np.isnan(obs_data.leading_frag_height_arr)]-np.max(obs_data.height_lag)))
        # plot velocity_arr vs leading_frag_time_arr
        ax2.plot(obs_data.time_arr-obs_data.time_arr[index], \
                 obs_data.leading_frag_vel_arr/1000, '--', color='black', linewidth=0.5, label='No Noise', zorder=2)
        ax2.legend()

        # inerpoate the velocity_arr to the leading_frag_time_arr
        no_noise_vel = np.interp(obs_data.height_lag,
                                    np.flip(obs_data.leading_frag_height_arr),
                                    np.flip(obs_data.leading_frag_vel_arr))
        
        # make the difference between the no_noise_vel and the obs_data.velocities
        diff_vel = obs_data.velocities - no_noise_vel
        ax6.plot(obs_data.time_lag, diff_vel/1000, '.', markersize=3, color='black', label='No Noise')

        # plot lag_arr vs leading_frag_time_arr withouth nan values
        lag_no_noise = (obs_data.leading_frag_length_arr-obs_data.leading_frag_length_arr[index])\
              - ((obs_data.v_init)*(obs_data.time_arr-obs_data.time_arr[index]))
        lag_no_noise -= lag_no_noise[index]
        # plot lag_arr vs leading_frag_time_arr
        ax3.plot(obs_data.time_arr-obs_data.time_arr[index], \
                 lag_no_noise, '--', color='black', linewidth=0.5, label='No Noise', zorder=2)

        # inerpoate the lag_arr to the leading_frag_time_arr
        no_noise_lag = np.interp(obs_data.height_lag,
                                    np.flip(obs_data.leading_frag_height_arr),
                                    np.flip(lag_no_noise))
        
        # make the difference between the no_noise_lag and the obs_data.lag
        diff_lag = obs_data.lag - no_noise_lag 
        ax7.plot(obs_data.time_lag, diff_lag, '.', markersize=3, color='black', label='No Noise')
        
    #     fig.suptitle(f"Simulated Test case {file_name}", fontsize=16, fontweight='bold')  # Adjust y for better spacing
    # else:
    #     fig.suptitle(f"{file_name}", fontsize=16, fontweight='bold')

    # Check if sim_data was provided
    if sim_data is not None:
        
        # # Plot simulated data
        # ax0.plot(sim_data.abs_magnitude, sim_data.leading_frag_height_arr/1000,'--', color=color_sim, label='wmpl')
        # ax4.plot(sim_data.luminosity_arr, sim_data.leading_frag_height_arr/1000,'--', color=color_sim, label='wmpl')

        # integration time step in self.const.dt for luminosity integration and abs_magnitude_integration check if any sim_data.stations_lum does not have '1T' or '2T' in the name as CAMO narrowfield do not have smearing because it follows the meteor
        if ((1/obs_data.fps_lum) > sim_data.const.dt): # and (not any('1T' in station for station in obs_data.stations_lum) or not any('2T' in station for station in obs_data.stations_lum)):
            sim_data.luminosity_arr, sim_data.abs_magnitude = integrateLuminosity(sim_data.time_arr,sim_data.time_arr,sim_data.luminosity_arr,sim_data.const.dt,obs_data.fps_lum,obs_data.P_0m)

        # Plot simulated data
        ax0.plot(sim_data.abs_magnitude, sim_data.leading_frag_height_arr/1000, color=color_sim, label=label_sim)
        ax0.legend()
        
        # inerpoate the abs_magnitude_arr to the leading_frag_height_arr
        sim_mag = np.interp(obs_data.height_lum, 
                                        np.flip(sim_data.leading_frag_height_arr), 
                                        np.flip(sim_data.abs_magnitude))
        
        # make the difference between the no_noise_mag and the obs_data.abs_magnitude
        sim_diff_mag = sim_mag - obs_data.absolute_magnitudes
        # for each station in obs_data_plot
        for station in np.unique(obs_data.stations_lum):
            # plot the height vs. absolute_magnitudes
            ax1.plot(sim_diff_mag[np.where(obs_data.stations_lum == station)], \
                    obs_data.height_lum[np.where(obs_data.stations_lum == station)]/1000, '.', \
                    color=station_colors[station], label=station)

        ax4.plot(sim_data.luminosity_arr, sim_data.leading_frag_height_arr/1000, color=color_sim, label=label_sim) 

        # interpolate to make sure they are the same length and discard points after height starts increasing if it does at any point obs_metsim_obj.traj.observations[0].model_ht
        sim_lum = np.interp(obs_data.height_lum, 
                                        np.flip(sim_data.leading_frag_height_arr), 
                                        np.flip(sim_data.luminosity_arr))
        
        # make the difference between the no_noise_intensity and the obs_data.luminosity_arr
        sim_diff_lum = obs_data.luminosity - sim_lum
        # for each station in obs_data_plot
        for station in np.unique(obs_data.stations_lum):
            # plot the height vs. absolute_magnitudes
            ax5.plot(sim_diff_lum[np.where(obs_data.stations_lum == station)], \
                    obs_data.height_lum[np.where(obs_data.stations_lum == station)]/1000, '.', \
                    color=station_colors[station], label=station)

        # find the obs_data.leading_frag_height_arr index is close to np.max(obs_data.height_lum) wihouth nan
        index = np.argmin(np.abs(sim_data.leading_frag_height_arr[~np.isnan(sim_data.leading_frag_height_arr)]-np.max(obs_data.height_lag)))
        # plot velocity_arr vs leading_frag_time_arr
        ax2.plot(sim_data.time_arr-sim_data.time_arr[index], sim_data.leading_frag_vel_arr/1000, color=color_sim, label=label_sim)
        ax2.legend()

        # inerpoate the velocity_arr to the leading_frag_time_arr
        sim_vel = np.interp(obs_data.height_lag,
                                    np.flip(sim_data.leading_frag_height_arr),
                                    np.flip(sim_data.leading_frag_vel_arr))
        
        # make the difference between the no_noise_vel and the obs_data.velocities
        sim_diff_vel = obs_data.velocities - sim_vel

        # for each station in obs_data_plot
        for station in np.unique(obs_data.stations_lag):
            time_plot= obs_data.time_lag[np.where(obs_data.stations_lag == station)]
            vel_plot = sim_diff_vel[np.where(obs_data.stations_lag == station)]/1000
            # plot the height vs. absolute_magnitudes
            ax6.plot(time_plot[1:], \
                    vel_plot[1:], '.', \
                        color=station_colors[station], label=station)

        # plot lag_arr vs leading_frag_time_arr withouth nan values
        sim_lag = (sim_data.leading_frag_length_arr-sim_data.leading_frag_length_arr[index])\
              - ((obs_data.v_init)*(sim_data.time_arr-sim_data.time_arr[index]))
        
        sim_lag -= sim_lag[index]
        # plot lag_arr vs leading_frag_time_arr
        ax3.plot(sim_data.time_arr-sim_data.time_arr[index], sim_lag, color=color_sim, label=label_sim)

        # inerpoate the lag_arr to the leading_frag_time_arr
        sim_lag = np.interp(obs_data.height_lag,
                                    np.flip(sim_data.leading_frag_height_arr),
                                    np.flip(sim_lag))
        
        # make the difference between the no_noise_lag and the obs_data.lag
        sim_diff_lag = obs_data.lag - sim_lag
        
        # for each station in obs_data_plot
        for station in np.unique(obs_data.stations_lag):
            # plot the height vs. absolute_magnitudes
            ax7.plot(obs_data.time_lag[np.where(obs_data.stations_lag == station)], \
                    sim_diff_lag[np.where(obs_data.stations_lag == station)], '.', \
                        color=station_colors[station], label=station)
        

    # Save the plot
    print('file saved: '+output_folder +os.sep+ file_name+'_LumLag_plot.png')
    # fig.savefig(output_folder +os.sep+ file_name +'_LumLag_plot.png', dpi=300)

    # save the figure
    fig.savefig(output_folder +os.sep+ file_name +'_LumLag_plot.png', 
            bbox_inches='tight',
            pad_inches=0.1,       # a little padding around the edge
            dpi=300)

    # Display the plot
    plt.close(fig)


# ---- globals for worker processes ----
_GLOBALS = {}

def _init_worker(obs_data, variables, flags_dict, fixed_values, align_height):
    # set once per process to avoid re-pickling per task
    _GLOBALS['obs_data'] = obs_data
    _GLOBALS['variables'] = variables
    _GLOBALS['flags_dict'] = flags_dict
    _GLOBALS['fixed_values'] = fixed_values
    _GLOBALS['align_height'] = align_height

def _apply_log_flags_to_sample(sample, variables, flags_dict):
    s = np.array(sample, dtype=float).copy()
    for i, var in enumerate(variables):
        flags = flags_dict.get(var, [])
        if isinstance(flags, (list, tuple)) and ('log' in flags):
            s[i] = 10.0**s[i]
    return s

def _compute_sim_lag(sim, obs_data):
    """
    Compute lag exactly as in your snippet:
        index = argmin |h - np.max(obs_data.height_lag)|
        lag = (L - L[index]) - obs_data.v_init*(t - t[index])
        lag -= lag[index]
    Uses the closest *valid* height index (maps back to full array index).
    """
    h = np.asarray(sim.leading_frag_height_arr)
    t = np.asarray(sim.time_arr)
    L = np.asarray(sim.leading_frag_length_arr)

    # Fallback if no height_lag in obs_data
    ref_h = getattr(obs_data, 'height_lag', None)
    if ref_h is None or len(ref_h) == 0:
        ref_h = np.asarray(getattr(obs_data, 'height_lum', [np.nan]))[0]
    else:
        ref_h = np.asarray(obs_data.height_lag)[0]

    valid = ~np.isnan(h)
    if not np.any(valid):
        return np.full_like(h, np.nan, dtype=float)

    h_valid = h[valid]
    idx_in_valid = np.argmin(np.abs(h_valid - ref_h))
    # map back to full-array index to avoid off-by-one when NaNs exist
    idx = np.flatnonzero(valid)[idx_in_valid]

    v0 = getattr(obs_data, 'v_init', np.nan)  # use obs_data.v_init as in your code
    lag = (L - L[idx]) - (v0*(t - t[idx]))
    lag = lag - lag[idx]  # zero at reference
    return lag


def _worker_simulate_and_interp(sample_equal_row):
    """
    Runs one posterior sample through runSimulation() and returns
    (lum_at_hl, mag_at_hl, vel_at_hl, lag_at_hl) as 1D arrays.
    Returns None on failure (caller will skip).
    """
    try:
        obs_data     = _GLOBALS['obs_data']
        variables    = _GLOBALS['variables']
        flags_dict   = _GLOBALS['flags_dict']
        fixed_values = _GLOBALS['fixed_values']
        align_height = _GLOBALS['align_height']

        pars = _apply_log_flags_to_sample(sample_equal_row, variables, flags_dict)
        sim  = runSimulationDynesty(pars, obs_data, variables, fixed_values)

        const_saved = sim.const  # save original const

        # NEW: integrate per your condition
        _maybe_integrate_luminosity(sim, obs_data)

        # build lag exactly per your definition
        sim_lag = _compute_sim_lag(sim, obs_data)


        # prepare monotone height for interp
        h  = np.asarray(sim.leading_frag_height_arr)
        ok = ~np.isnan(h)
        if np.count_nonzero(ok) < 4:
            return None
        h  = h[ok]
        order = np.argsort(h)  # increasing
        h   = h[order]
        lum = np.asarray(sim.luminosity_arr)[ok][order]
        mag = np.asarray(sim.abs_magnitude)[ok][order]
        vel = np.asarray(sim.leading_frag_vel_arr)[ok][order]
        lag = np.asarray(sim_lag)[ok][order]

        hl = np.asarray(obs_data.height_lum)
        hv = np.asarray(obs_data.height_lag)

        lum_hl = np.interp(hl, h, lum, left=np.nan, right=np.nan)
        mag_hl = np.interp(hl, h, mag, left=np.nan, right=np.nan)
        vel_hv = np.interp(hv, h, vel, left=np.nan, right=np.nan)
        lag_hv = np.interp(hv, h, lag, left=np.nan, right=np.nan)

        # mass_before = mass_best[np.argmin(np.abs(heights - erosion_height_change))]

        # check if erosion_height_change is in variables
        if 'erosion_height_change' in variables:
            mass_at_erosion_change = const_saved.mass_at_erosion_change
            erosion_height_change = const_saved.erosion_height_change
            mass = np.asarray(sim.mass_total_active_arr)[:-1]
            # if mass_before is None use the old method
            if mass_at_erosion_change is None:
                mass_at_erosion_change = mass[np.argmin(np.abs(h[:-1] - erosion_height_change))]
            # compute rho_mass_weighted
            erosion_rho_change = const_saved.erosion_rho_change
            rho_mass_weighted = const_saved.rho*(abs(const_saved.m_init-mass_at_erosion_change)/const_saved.m_init) + erosion_rho_change*(mass_at_erosion_change/const_saved.m_init)
            rho_volume_weighted = const_saved.m_init/((abs(const_saved.m_init-mass_at_erosion_change)/const_saved.rho) + (mass_at_erosion_change/erosion_rho_change))
            # print(f"rho_mass_weighted: {rho_mass_weighted} rho_volume_weighted: {rho_volume_weighted} and rho: {const_saved.rho}")
            erosion_dyn_press_change = sim.leading_frag_dyn_press_arr[np.argmin(np.abs(sim.leading_frag_height_arr[:-1] - erosion_height_change))]
        else:
            rho_mass_weighted = const_saved.rho
            rho_volume_weighted = const_saved.rho
            erosion_dyn_press_change = None

        const_backup = {
        "rho_mass_weighted": rho_mass_weighted,
        "rho_volume_weighted": rho_volume_weighted,
        "erosion_beg_vel": const_saved.erosion_beg_vel if hasattr(const_saved, 'erosion_beg_vel') else None,
        "erosion_beg_mas": const_saved.erosion_beg_mas if hasattr(const_saved, 'erosion_beg_mas') else None,
        "erosion_beg_dyn_press": const_saved.erosion_beg_dyn_press if hasattr(const_saved, 'erosion_beg_dyn_press') else None,
        "mass_at_erosion_change": const_saved.mass_at_erosion_change if hasattr(const_saved, 'mass_at_erosion_change') else None,
        "dyn_press_at_erosion_change": erosion_dyn_press_change,
        "energy_per_cs_before_erosion": const_saved.energy_per_cs_before_erosion if hasattr(const_saved, 'energy_per_cs_before_erosion') else None,
        "energy_per_mass_before_erosion": const_saved.energy_per_mass_before_erosion if hasattr(const_saved, 'energy_per_mass_before_erosion') else None,
        "main_mass_exhaustion_ht": const_saved.main_mass_exhaustion_ht if hasattr(const_saved, 'main_mass_exhaustion_ht') else None,
        "main_bottom_ht": const_saved.main_bottom_ht if hasattr(const_saved, 'main_bottom_ht') else None
        }

        return lum_hl, mag_hl, vel_hv, lag_hv, const_backup
    except Exception:
        return None

def _quantiles_from_samples(arr_2d, qs):
    """arr_2d shape (S,H). Returns dict of quantiles along axis=0 ignoring NaNs."""
    out = {}
    for name, q in qs.items():
        out[name] = np.nanquantile(arr_2d, q, axis=0, method='linear')
    return out

def _maybe_integrate_luminosity(sim, obs_data):
    """
    If (1/fps_lum) > sim.const.dt, integrate luminosity over fps window and
    recompute abs magnitude using obs_data.P_0m.
    """
    # Pull dt safely
    dt = None
    if hasattr(sim, 'const') and hasattr(sim.const, 'dt'):
        dt = sim.const.dt
    elif hasattr(sim, 'dt'):
        dt = sim.dt

    fps = getattr(obs_data, 'fps_lum', None)
    P0m = getattr(obs_data, 'P_0m', None)

    if dt is None or fps is None or P0m is None:
        return  # nothing we can do

    try:
        if fps > 0 and (1.0/float(fps)) > float(dt):
            L_new, mag_new = integrateLuminosity(
                np.asarray(sim.time_arr),
                np.asarray(sim.time_arr),            # same as your call
                np.asarray(sim.luminosity_arr),
                float(dt), float(fps), float(P0m)
            )
            sim.luminosity_arr = np.asarray(L_new)
            sim.abs_magnitude  = np.asarray(mag_new)
    except Exception:
        # be silent and keep raw arrays if integration fails for any reason
        pass

def _plot_distrib_weighted(rho_mass_weighted_list, weights, output_folder="", file_name="name",var_name="var", label="var", colors='black'):
    if not DYNESTY_FOUND:
        return np.nan, np.nan, np.nan
        
    print("Creating distribution plot...")
    var_corrected_lo, var_corrected_median, var_corrected_hi = _quantile(rho_mass_weighted_list, [0.025, 0.5, 0.975], weights=weights)
    # Create figure tau
    fig = plt.figure(figsize=(8, 6))
    ax_dist = fig.add_subplot(111)

    smooth = 0.02
    lo, hi = np.min(rho_mass_weighted_list), np.max(rho_mass_weighted_list)
    nbins = int(round(10./smooth))
    hist, edges = np.histogram(rho_mass_weighted_list, bins=nbins, weights=weights, range=(lo, hi))
    hist = norm_kde(hist, 10.0)
    bin_centers = 0.5*(edges[:-1] + edges[1:])

    ax_dist.fill_between(bin_centers, hist, color=colors, alpha=0.6)

    # Percentile lines
    ax_dist.axvline(var_corrected_median, color=colors, linestyle='--', linewidth=1.5)
    ax_dist.axvline(var_corrected_lo, color=colors, linestyle='--', linewidth=1.5)
    ax_dist.axvline(var_corrected_hi, color=colors, linestyle='--', linewidth=1.5)

    # Title and formatting
    plus = var_corrected_hi - var_corrected_median
    minus = var_corrected_median - var_corrected_lo
    fmt = lambda v: f"{v:.4g}" if np.isfinite(v) else "---"
    title = rf"Tot N. Runs {len(rho_mass_weighted_list)} — {label} = {fmt(var_corrected_median)}$^{{+{fmt(plus)}}}_{{-{fmt(minus)}}}$"
    ax_dist.set_title(title, fontsize=20)
    # ax_dist.tick_params(axis='x', labelbottom=False)
    ax_dist.tick_params(axis='y', left=False, labelleft=False)
    ax_dist.set_ylabel("")
    ax_dist.set_xlabel(f'{label}', fontsize=20)
    ax_dist.spines['left'].set_visible(False)
    ax_dist.spines['right'].set_visible(False)
    ax_dist.spines['top'].set_visible(False)
    # x axis from 0 to tau_corrected*2
    # ax_dist.set_xlim(0, var_corrected_hi+var_corrected_median)
    plt.savefig(os.path.join(output_folder, f"{file_name}_{var_name}_distribution.png"), bbox_inches='tight')
    return var_corrected_median, var_corrected_lo, var_corrected_hi

def posteriorBandsVsHeightParallel(
    dynesty_results,
    obs_data,
    flags_dict,
    fixed_values,
    output_folder='',
    file_name='',
    nsamples=500,
    seed=0,
    n_workers=None,          # default: os.cpu_count()-1
    chunksize=8,             # tune for your machine
    color_best='black',
    label_best='Best Fit'):
    """ Parallel computation of posterior bands at observation heights/times.

    This function resamples the Dynesty posterior (equal weights), runs simulations in parallel
    for each sample, and computes quantile bands for luminosity, magnitude, velocity, and lag.
    It also plots the results and returns a dictionary of data.

    Arguments:
        dynesty_results: [object] Result object from dynesty sampling.
        obs_data: [object] Object containing the observational data.
        flags_dict: [dict] Dictionary of flags for parameters.
        fixed_values: [dict] Dictionary of fixed parameter values.

    Keyword arguments:
        output_folder: [str] Path to save plots. '' by default.
        file_name: [str] Base name for output files. '' by default.
        nsamples: [int] Number of posterior samples to draw. 500 by default.
        seed: [int] Random seed for resampling. 0 by default.
        n_workers: [int] Number of parallel workers. Defaults to os.cpu_count()-1.
        chunksize: [int] Chunksize for process pool. 8 by default.
        color_best: [str] Color for the best fit line in plots. 'black' by default.
        label_best: [str] Label for the best fit line. 'Best Fit' by default.

    Return:
        results: [dict] Dictionary containing:
            - 'samples_eq': Resampled parameters.
            - 'weights': weights used.
            - 'lum_samples', 'mag_samples', 'vel_samples', 'lag_samples': Arrays of simulated data.
            - 'bands': Dictionary of quantile bands for each observable.
            - 'best_guess': Dictionary of best fit data.
            - 'sim_best': Simulation object for the best fit.
            - 'rho_mass_weighted_estimate': Statistics on mass-weighted density.

    """

    if not DYNESTY_FOUND:
        print("Dynesty package not found. Install dynesty to use the Dynesty functions.")
        return None

    rng = np.random.default_rng(seed)
    variables = list(flags_dict.keys())

    # weights -> equal-weight resampling
    logwt = np.asarray(dynesty_results.logwt)
    w = np.exp(logwt - np.max(logwt))
    w /= np.sum(w)
    samples_eq = dynesty.utils.resample_equal(dynesty_results.samples, w)
    if nsamples is not None and nsamples < samples_eq.shape[0]:
        idx_keep = rng.choice(samples_eq.shape[0], size=nsamples, replace=False)
        samples_eq = samples_eq[idx_keep]

    H_l = len(obs_data.height_lum)
    H_v = len(obs_data.height_lag) if hasattr(obs_data, 'height_lag') and len(obs_data.height_lag) > 0 else len(obs_data.height_lum)
    S = samples_eq.shape[0]

    lum_samples = np.full((S, H_l), np.nan)
    mag_samples = np.full((S, H_l), np.nan)
    vel_samples = np.full((S, H_v), np.nan)
    lag_samples = np.full((S, H_v), np.nan)
    const_backups = [None]*S  # to store const backups if needed

    align_height = np.max(obs_data.height_lag) if hasattr(obs_data, 'height_lag') and len(obs_data.height_lag) > 0 \
                   else np.max(obs_data.height_lum)

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)
        # n_workers = multiprocessing.cpu_count()

    # --- try parallel; if anything goes wrong, do sequential as fallback ---
    ran_parallel = False
    try:
        ctx = multiprocessing.get_context("spawn")  # Windows-safe
        print(f"[{file_name}] Running {S} simulations with {n_workers} workers...", flush=True)
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(obs_data, variables, flags_dict, fixed_values, align_height),
        ) as ex:
            # submit and keep a map to their sample index
            future_to_idx = {
                ex.submit(_worker_simulate_and_interp, samples_eq[sidx]): sidx
                for sidx in range(S)
            }

            done = 0
            for fut in as_completed(future_to_idx):
                sidx = future_to_idx[fut]
                res = fut.result()
                if res is not None:
                    lum_hl, mag_hl, vel_hl, lag_hl, const_backup = res
                    lum_samples[sidx] = lum_hl
                    mag_samples[sidx] = mag_hl
                    vel_samples[sidx] = vel_hl
                    lag_samples[sidx] = lag_hl
                    const_backups[sidx] = const_backup
                done += 1
                # progress print
                print(f"[{file_name}] {done}/{S} simulations done", flush=True)

        ran_parallel = True
    except Exception as e:
        print(f"[meteor uncertainty bands plot] Parallel run failed ({e}). Falling back to sequential.", flush=True)
    
    if not ran_parallel:
        print(f"[{file_name}] Running {S} simulations sequentially...", flush=True)
        _init_worker(obs_data, variables, flags_dict, fixed_values, align_height)
        done = 0
        for sidx in range(S):
            res = _worker_simulate_and_interp(samples_eq[sidx])
            if res is not None:
                lum_hl, mag_hl, vel_hl, lag_hl, const_backup = res
                lum_samples[sidx] = lum_hl
                mag_samples[sidx] = mag_hl
                vel_samples[sidx] = vel_hl
                lag_samples[sidx] = lag_hl
                const_backups[sidx] = const_backup
            done += 1
            print(f"[{file_name}] {done}/{S} simulations done", flush=True)

    # Quantile bands
    qs = {
        "p16": 0.158655, "p50": 0.5, "p84": 0.841345,
        "p025": 0.025, "p975": 0.975,
        "p00015": 0.00135, "p99985": 0.99865
    }
    bands_lum = _quantiles_from_samples(lum_samples, qs)
    bands_mag = _quantiles_from_samples(mag_samples, qs)
    bands_vel = _quantiles_from_samples(vel_samples, qs)
    bands_lag = _quantiles_from_samples(lag_samples, qs)

    # Best-guess curve (re-use your existing code)
    best_idx  = int(np.argmax(dynesty_results.logl))
    best_pars = _apply_log_flags_to_sample(dynesty_results.samples[best_idx], variables, flags_dict)
    sim_best  = runSimulationDynesty(best_pars, obs_data, variables, fixed_values)
    _maybe_integrate_luminosity(sim_best, obs_data)
    best_lag_full = _compute_sim_lag(sim_best, obs_data)


    ok = ~np.isnan(sim_best.leading_frag_height_arr)
    order = np.argsort(sim_best.leading_frag_height_arr[ok])
    hb = sim_best.leading_frag_height_arr[ok][order]
    hl = np.asarray(obs_data.height_lum)
    hv = np.asarray(obs_data.height_lag)

    best_lum = np.interp(hl, hb, np.asarray(sim_best.luminosity_arr)[ok][order], left=np.nan, right=np.nan)
    best_mag = np.interp(hl, hb, np.asarray(sim_best.abs_magnitude)[ok][order], left=np.nan, right=np.nan)
    best_vel = np.interp(hv, hb, np.asarray(sim_best.leading_frag_vel_arr)[ok][order], left=np.nan, right=np.nan)
    best_lag = np.interp(hv, hb, np.asarray(best_lag_full)[ok][order], left=np.nan, right=np.nan)

    # (Optional) call your previous plotting routine here to draw bands + obs + best.
    # You can keep the exact plotting code you already have, just swap in the arrays
    # from lum_samples/mag_samples/... and bands_* dicts.

    # ---- PLOT & SAVE ----
    _plot_bands_obs_best(
        obs_data,
        hl, hv,
        bands_lum, bands_mag, bands_vel, bands_lag,
        best_lum, best_mag, best_vel, best_lag,
        output_folder=output_folder,
        file_name=file_name,
        color_best=color_best,
        label_best=label_best
    )

    # ---- PLOT & SAVE ----
    
    # extract the rho_mass_weighted from const_backups
    rho_mass_weighted_list = []
    rho_volume_weighted_list = []
    for const in const_backups:
        if const is not None:
            rho_mass_weighted_list.append(const["rho_mass_weighted"])
            rho_volume_weighted_list.append(const["rho_volume_weighted"])
        else:
            rho_mass_weighted_list.append(np.nan)
            rho_volume_weighted_list.append(np.nan)
    
    # from list to numpy array
    rho_mass_weighted_list = np.array(rho_mass_weighted_list)
    rho_low95_real, rho_median_real, rho_high95_real = _quantile(rho_mass_weighted_list, [0.025, 0.5, 0.975], weights=w)

    return {
        'samples_eq': samples_eq,
        'weights': w,
        'lum_samples': lum_samples,
        'mag_samples': mag_samples,
        'vel_samples': vel_samples,
        'lag_samples': lag_samples,
        'bands': {
            'lum': bands_lum,
            'mag': bands_mag,
            'vel': bands_vel,
            'lag': bands_lag,
        },
        'best_guess': {
            'luminosity': best_lum,
            'abs_magnitude': best_mag,
            'velocity': best_vel,
            'lag': best_lag,
        },
        'sim_best': sim_best,
        'const_backups': const_backups,
        'rho_array': rho_mass_weighted_list,
        'rho_mass_weighted_estimate': {
            'median': rho_median_real,
            'low95': rho_low95_real,
            'high95': rho_high95_real,
        }
    }


def _plot_bands_obs_best(
    obs_data,
    hl, hv,                                  # heights (m) used for interpolation
    bands_lum, bands_mag, bands_vel, bands_lag,
    best_lum, best_mag, best_vel, best_lag,
    output_folder='', file_name='', color_best='black', label_best='Best Fit'):
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    heights_km_lum = np.asarray(hl)/1000.0
    heights_km_lag = np.asarray(hv)/1000.0

    fig = plt.figure(figsize=(15, 4))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.3)

    ax_lum = fig.add_subplot(gs[0, 0])
    ax_mag = fig.add_subplot(gs[0, 1])
    ax_vel = fig.add_subplot(gs[0, 2])
    ax_lag = fig.add_subplot(gs[0, 3])

    def shade(ax, lo3, hi3, lo2, hi2, lo1, hi1, y):
        # 3σ (light), 2σ (mid), 1σ (dark)
        ax.fill_betweenx(y, lo3, hi3, color='gray', alpha=0.10, linewidth=0)
        ax.fill_betweenx(y, lo2, hi2, color='gray', alpha=0.18, linewidth=0)
        ax.fill_betweenx(y, lo1, hi1, color='gray', alpha=0.28, linewidth=0)

    # --- stable station colors across panels ---
    cmap = plt.get_cmap("tab10")
    station_colors = {}
    plotted_stations = set()

    station_handles = []
    station_labels  = []

    # --- LUMINOSITY ---
    shade(ax_lum, bands_lum["p00015"], bands_lum["p99985"],
                 bands_lum["p025"],   bands_lum["p975"],
                 bands_lum["p16"],    bands_lum["p84"], heights_km_lum)
    for st in np.unique(getattr(obs_data, 'stations_lum', [])):
        # Assign a unique color if not already used
        if st not in station_colors:
            station_colors[st] = cmap(len(station_colors) % 10)
        m = obs_data.stations_lum == st
        ax_lum.plot(
            obs_data.luminosity[m], obs_data.height_lum[m]/1000.0,
            linestyle='--', marker='x', markersize=4, linewidth=1,
            color=station_colors.get(st, 'C0'),
            label=str(st) if st not in plotted_stations else None
        )
        plotted_stations.add(st)
    ax_lum.plot(best_lum, heights_km_lum, color=color_best, lw=1.8, label=label_best)
    ax_lum.set_xlabel("Luminosity [W]")
    ax_lum.set_ylabel("Height [km]")
    ax_lum.grid(True, linestyle='--', color='lightgray')

    # --- ABS MAG ---
    shade(ax_mag, bands_mag["p00015"], bands_mag["p99985"],
                 bands_mag["p025"],   bands_mag["p975"],
                 bands_mag["p16"],    bands_mag["p84"], heights_km_lum)
    for st in np.unique(getattr(obs_data, 'stations_lum', [])):
        if st not in station_colors:
            station_colors[st] = cmap(len(station_colors) % 10)
        m = obs_data.stations_lum == st
        h, = ax_mag.plot(
            obs_data.absolute_magnitudes[m], obs_data.height_lum[m]/1000.0,
            linestyle='--', marker='x', markersize=4, linewidth=1,
            color=station_colors.get(st, 'C0'),
        )
        station_handles.append(h); station_labels.append(str(st))
    ax_mag.plot(best_mag, heights_km_lum, color=color_best, lw=1.8)
    ax_mag.set_xlabel("Abs. Magnitude")
    ax_mag.invert_xaxis()
    ax_mag.grid(True, linestyle='--', color='lightgray')

    # --- VELOCITY (km/s) ---
    shade(ax_vel, bands_vel["p00015"]/1000.0, bands_vel["p99985"]/1000.0,
                 bands_vel["p025"]/1000.0,   bands_vel["p975"]/1000.0,
                 bands_vel["p16"]/1000.0,    bands_vel["p84"]/1000.0, heights_km_lag)
    for st in np.unique(getattr(obs_data, 'stations_lag', [])):
        if st not in station_colors:
            station_colors[st] = cmap(len(station_colors) % 10)
        vel_plot_st= obs_data.velocities[np.where(obs_data.stations_lag == st)]/1000
        height_plot_st = obs_data.height_lag[np.where(obs_data.stations_lag == st)]/1000.0
        ax_vel.plot(
           vel_plot_st[1:], height_plot_st[1:],
            linestyle='--', marker='x', markersize=4, linewidth=1,
            color=station_colors.get(st, 'C0'),
        )
    ax_vel.plot(best_vel/1000.0, heights_km_lag, color=color_best, lw=1.8)
    ax_vel.set_xlabel("Velocity [km/s]")
    ax_vel.grid(True, linestyle='--', color='lightgray')

    # --- LAG (m) ---
    shade(ax_lag, bands_lag["p00015"], bands_lag["p99985"],
                 bands_lag["p025"],   bands_lag["p975"],
                 bands_lag["p16"],    bands_lag["p84"], heights_km_lag)

    for st in np.unique(getattr(obs_data, 'stations_lag', [])):
        if st not in station_colors:
            station_colors[st] = cmap(len(station_colors) % 10)
        m = obs_data.stations_lag == st
        h, = ax_lag.plot(
            obs_data.lag[m], obs_data.height_lag[m]/1000.0,
            linestyle='--', marker='x', markersize=4, linewidth=1,
            color=station_colors.get(st, 'C0'),
            label=str(st)
        )
        station_handles.append(h); station_labels.append(str(st))
    best_line, = ax_lag.plot(best_lag, heights_km_lag, color=color_best, lw=1.8, label=label_best)
    ax_lag.set_xlabel("Lag [m]")
    ax_lag.grid(True, linestyle='--', color='lightgray')

    # --- Legend with σ labels ---
    proxy_1s = Patch(facecolor='gray', alpha=0.28, label='1σ')
    proxy_2s = Patch(facecolor='gray', alpha=0.18, label='2σ')
    proxy_3s = Patch(facecolor='gray', alpha=0.10, label='3σ')

    # take the index of the unique station_handles and station_labels
    unique_stations = {}
    for h, l in zip(station_handles, station_labels):
        if l not in unique_stations:
            unique_stations[l] = h
    station_handles = list(unique_stations.values())
    station_labels  = list(unique_stations.keys())

    handles = [best_line] + station_handles + [proxy_1s, proxy_2s, proxy_3s]
    labels  = [label_best] + station_labels + ['1σ', '2σ', '3σ']
    ax_lag.legend(handles, labels, loc='best', fontsize=8, frameon=True)
    # add a suptitle with file_name if it contains a date-time stamp like YYYYMMDD_HHMMSS
    if file_name:
        m = re.search(r'\d{8}_\d{6}', file_name)
        if m:
            fig.suptitle(m.group(0))

    plt.tight_layout()
    if output_folder and file_name:
        outpath = os.path.join(output_folder, file_name + "_posterior_bands_vs_height.png")
        fig.savefig(outpath, dpi=300, bbox_inches='tight')
        print(f"[saved] {outpath}")
    plt.close(fig)


def posteriorBandsTopkCheck(
    dynesty_results,
    obs_data,
    flags_dict,
    fixed_values,
    top_k=10,
    output_folder='',
    file_name='topk_check',
    n_workers=None,
    color_best='black',
    label_best='Best Fit',):
    """ Quick-check version: run only the best `top_k` samples (highest logl).

    This function selects the top k samples from the posterior based on log-likelihood,
    creates a subset of results, and runs the parallel band computation on this subset.
    Useful for quick visualization of the best fits.

    Arguments:
        dynesty_results: [object] Result object from dynesty sampling.
        obs_data: [object] Object containing the observational data.
        flags_dict: [dict] Dictionary of flags for parameters.
        fixed_values: [dict] Dictionary of fixed parameter values.

    Keyword arguments:
        top_k: [int] Number of top samples to use. 10 by default.
        output_folder: [str] Path to save plots. '' by default.
        file_name: [str] Base name for output files. 'topk_check' by default.
        n_workers: [int] Number of parallel workers. None by default (uses posteriorBandsVsHeightParallel default).
        color_best: [str] Color for the best fit line in plots. 'black' by default.
        label_best: [str] Label for the best fit line. 'Best Fit' by default.

    Return:
        results: [dict] Dictionary containing the same structure as posteriorBandsVsHeightParallel.
            - 'samples_eq': Subset of parameters (top k).
            - 'weights': Equal weights/zeros (as passed to parallel func).
            - 'lum_samples', 'mag_samples', ...: Arrays of simulated data for the top k samples.
            - ... (see posteriorBandsVsHeightParallel for full details).

    """
    # pick top-K by log-likelihood
    logl = np.asarray(dynesty_results.logl)
    if top_k <= 0:
        raise ValueError("top_k must be >= 1")
    k = min(top_k, logl.size)
    top_idx = np.argsort(logl)[-k:]  # ascending -> take last k
    # keep in descending order (optional)
    top_idx = top_idx[np.argsort(logl[top_idx])[::-1]]

    # build a tiny dynesty-like view for reuse of the parallel function
    class _MiniResult:
        pass
    mini = _MiniResult()
    mini.samples = np.asarray(dynesty_results.samples)[top_idx]
    # fabricate equal weights for top-k (we won’t resample inside)
    mini.logwt   = np.zeros(k, dtype=float)
    mini.logl    = logl[top_idx]

    # Reuse the parallel function but tell it “nsamples=k” so it won’t subsample
    return posteriorBandsVsHeightParallel(
        dynesty_results=mini,
        obs_data=obs_data,
        flags_dict=flags_dict,
        fixed_values=fixed_values,
        output_folder=output_folder,
        file_name=file_name,
        nsamples=k,
        n_workers=n_workers,
        color_best=color_best,
        label_best=label_best,
    )


def plotObsVsHeight(obs_data, sim_data=None, output_folder='', file_name='', color_sim='black', label_sim='Best Fit'):
    """ Plot various observables vs height (Luminosity, Absolute Magnitude, Velocity, Lag).

    Arguments:
        obs_data: [object] Object containing the observational data.

    Keyword arguments:
        sim_data: [object] Object containing the simulation data. None by default.
        output_folder: [str] Path to the output folder. Empty string by default.
        file_name: [str] Name of the file. Empty string by default.
        color_sim: [str] Color for the simulation plot. 'black' by default.
        label_sim: [str] Label for the simulation plot. 'Best Fit' by default.

    Return:
        None

    """
    fig = plt.figure(figsize=(15, 4))
    # # take only the reg ex with YYYYMMDD_HHMMSS
    file_name_print = re.search(r'\d{8}_\d{6}', file_name)
    # check if file_name_print is a None type object you can use .group(0)
    if file_name_print is not None:
        fig.suptitle(file_name_print.group(0))

    gs_main = gridspec.GridSpec(1, 4, figure=fig, wspace=0.3)

    cmap = plt.get_cmap("tab10")
    station_colors = {}
    station_handles = []
    station_labels  = []

    def getColor(station):
        if station not in station_colors:
            station_colors[station] = cmap(len(station_colors) % 10)
        return station_colors[station]

    # Define plotting grid pairs (main, residual) with shared y-axis
    axes = []
    for i in range(4):
        gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[0, i], wspace=0, width_ratios=[3, 1])
        ax_main = fig.add_subplot(gs[0])
        ax_resid = fig.add_subplot(gs[1], sharey=ax_main)
        axes.append((ax_main, ax_resid))

    ax_lum, ax_lum_res = axes[0]
    ax_mag, ax_mag_res = axes[1]
    ax_vel, ax_vel_res = axes[2]
    ax_lag, ax_lag_res = axes[3]

    #### OBSERVATIONAL DATA ####

    # LUMINOSITY (with camera legend)
    for station in np.unique(obs_data.stations_lum):
        mask = obs_data.stations_lum == station
        color = getColor(station)
        ax_lum.plot(obs_data.luminosity[mask], obs_data.height_lum[mask]/1000, 'x--', color=color)

    # # Draw a line for extra lag-only cameras (not in lum)
    # if not np.array_equal(np.unique(obs_data.stations_lag), np.unique(obs_data.stations_lum)):
    #     stations_lag_only = np.setdiff1d(np.unique(obs_data.stations_lag), np.unique(obs_data.stations_lum))
    #     if len(stations_lag_only) > 0:
    #         mask = np.isin(obs_data.stations_lag, stations_lag_only)
    #         ax_lum.plot(obs_data.luminosity[mask], obs_data.height_lag[mask]/1000, 'x--', color='gray', label='Lag-only')

    ax_lum.set_xlabel("Luminosity [W]")
    ax_lum.set_ylabel("Height [km]")
    ax_lum.grid(True, linestyle='--', color='lightgray')

    ax_lum_res.fill_betweenx([np.min(obs_data.height_lum)/1000,np.max(obs_data.height_lum)/1000], -obs_data.noise_lum, obs_data.noise_lum, color='darkgray', alpha=0.2)
    ax_lum_res.fill_betweenx([np.min(obs_data.height_lum)/1000,np.max(obs_data.height_lum)/1000], -2*obs_data.noise_lum, 2*obs_data.noise_lum, color='lightgray', alpha=0.2)
    ax_lum_res.plot([0, 0], ax_lum.get_ylim(), color='lightgray')
    ax_lum_res.set_xlabel("Res. Lum")
    ax_lum_res.grid(True, linestyle='--', color='lightgray')
    ax_lum_res.tick_params(labelleft=False)

    # ABS MAGNITUDE
    for station in np.unique(obs_data.stations_lum):
        mask = obs_data.stations_lum == station
        color = getColor(station)
        h, = ax_mag.plot(obs_data.absolute_magnitudes[mask], obs_data.height_lum[mask]/1000, 'x--', color=color)
        station_handles.append(h); station_labels.append(str(station))
    ax_mag.set_xlabel("Abs. Magnitude")
    ax_mag.invert_xaxis()
    ax_mag.grid(True, linestyle='--', color='lightgray')

    ax_mag_res.fill_betweenx([np.min(obs_data.height_lum)/1000,np.max(obs_data.height_lum)/1000], -obs_data.noise_mag, obs_data.noise_mag, color='darkgray', alpha=0.2)
    ax_mag_res.fill_betweenx([np.min(obs_data.height_lum)/1000,np.max(obs_data.height_lum)/1000], -2*obs_data.noise_mag, 2*obs_data.noise_mag, color='lightgray', alpha=0.2)
    ax_mag_res.plot([0, 0], ax_mag.get_ylim(), color='lightgray')
    ax_mag_res.set_xlabel("Res. Mag")
    ax_mag_res.grid(True, linestyle='--', color='lightgray')
    ax_mag_res.tick_params(labelleft=False)

    # VELOCITY
    for station in np.unique(obs_data.stations_lag):
        mask = obs_data.stations_lag == station
        color = getColor(station)
        height_plot = obs_data.height_lag[mask]/1000
        vel_plot = obs_data.velocities[mask]/1000
        ax_vel.plot(vel_plot[1:], height_plot[1:], 'x--', color=color)
    ax_vel.set_xlabel("Velocity [km/s]")
    ax_vel.grid(True, linestyle='--', color='lightgray')

    ax_vel_res.fill_betweenx([np.min(obs_data.height_lag)/1000,np.max(obs_data.height_lag)/1000], -obs_data.noise_vel/1000, obs_data.noise_vel/1000, color='darkgray', alpha=0.2)
    ax_vel_res.fill_betweenx([np.min(obs_data.height_lag)/1000,np.max(obs_data.height_lag)/1000], -2*obs_data.noise_vel/1000, 2*obs_data.noise_vel/1000, color='lightgray', alpha=0.2)
    ax_vel_res.plot([0, 0], ax_vel.get_ylim(), color='lightgray')
    ax_vel_res.set_xlabel("Res. Vel")
    ax_vel_res.grid(True, linestyle='--', color='lightgray')
    ax_vel_res.tick_params(labelleft=False)

    # LAG
    for station in np.unique(obs_data.stations_lag):
        mask = obs_data.stations_lag == station
        color = getColor(station)
        # obs_data.lag[mask] = (obs_data.length[mask]) - ((24610)*(obs_data.time_lag[mask]))
        h, = ax_lag.plot(obs_data.lag[mask], obs_data.height_lag[mask]/1000, 'x--', color=color, label=station)
        station_handles.append(h); station_labels.append(str(station))
    ax_lag.set_xlabel("Lag [m]")
    ax_lag.grid(True, linestyle='--', color='lightgray')

    ax_lag_res.fill_betweenx([np.min(obs_data.height_lag)/1000,np.max(obs_data.height_lag)/1000], -obs_data.noise_lag, obs_data.noise_lag, color='darkgray', alpha=0.2)
    ax_lag_res.fill_betweenx([np.min(obs_data.height_lag)/1000,np.max(obs_data.height_lag)/1000], -2*obs_data.noise_lag, 2*obs_data.noise_lag, color='lightgray', alpha=0.2)
    ax_lag_res.plot([0, 0], ax_lag.get_ylim(), color='lightgray')
    ax_lag_res.set_xlabel("Res. Lag")
    ax_lag_res.grid(True, linestyle='--', color='lightgray')
    ax_lag_res.tick_params(labelleft=False)

    ### FIX AXIS LIMITS BEFORE PLOTTING SIM DATA ###
    for ax_main, ax_res in axes:
        xlim = ax_main.get_xlim()
        ylim = ax_main.get_ylim()
        ax_main.set_xlim(xlim)
        ax_main.set_ylim(ylim)
        ax_res.set_ylim(ylim)
    
    # # only for lag plot use the same y limits as mag plot
    # ax_lag.set_ylim(ax_mag.get_ylim())
    # ax_lag_res.set_ylim(ax_mag.get_ylim())
    # # for the max value of the x axis of te lag plot is 100 and keep the min value the same
    # ax_lag.set_xlim(ax_lag.get_xlim()[0], 100)


    #### SIMULATED DATA (Interpolated) ####
    if sim_data is not None:
        # # Integrate luminosity/magnitude if needed
        # if (1/obs_data.fps_lum) > sim_data.const.dt:
        #     sim_data.luminosity_arr, sim_data.abs_magnitude = integrateLuminosity(
        #         sim_data.time_arr, sim_data.time_arr, sim_data.luminosity_arr,
        #         sim_data.const.dt, obs_data.fps_lum, obs_data.P_0m
        #     )

        # SMOOTH PLOTTING using time-based sampling
        ax_lum.plot(sim_data.luminosity_arr, sim_data.leading_frag_height_arr/1000, color=color_sim, label=label_sim)
        ax_mag.plot(sim_data.abs_magnitude, sim_data.leading_frag_height_arr/1000, color=color_sim)
        ax_vel.plot(sim_data.leading_frag_vel_arr/1000, sim_data.leading_frag_height_arr/1000, color=color_sim)

        index = np.argmin(np.abs(sim_data.leading_frag_height_arr[~np.isnan(sim_data.leading_frag_height_arr)]-np.max(obs_data.height_lag)))
        # plot lag_arr vs leading_frag_time_arr withouth nan values
        sim_lag = (sim_data.leading_frag_length_arr-sim_data.leading_frag_length_arr[index])\
              - ((obs_data.v_init)*(sim_data.time_arr-sim_data.time_arr[index]))
            #   - ((24710)*(sim_data.time_arr-sim_data.time_arr[index]))        
        sim_lag -= sim_lag[index]

        best_line, = ax_lag.plot(sim_lag, sim_data.leading_frag_height_arr/1000, color=color_sim, label=label_sim)
        
        # RESIDUALS (interpolated at obs heights)
        lum_res = obs_data.luminosity - np.interp(obs_data.height_lum, np.flip(sim_data.leading_frag_height_arr), np.flip(sim_data.luminosity_arr))
        mag_res = np.interp(obs_data.height_lum, np.flip(sim_data.leading_frag_height_arr), np.flip(sim_data.abs_magnitude)) - obs_data.absolute_magnitudes
        vel_res = (obs_data.velocities - np.interp(obs_data.height_lag, np.flip(sim_data.leading_frag_height_arr), np.flip(sim_data.leading_frag_vel_arr)))/1000
        lag_res = obs_data.lag - np.interp(obs_data.height_lag, np.flip(sim_data.leading_frag_height_arr), np.flip(sim_lag))

        # ax_lum_res.plot(lum_res, obs_data.height_lum/1000, '.', color=color_sim)
        # ax_mag_res.plot(mag_res, obs_data.height_lum/1000, '.', color=color_sim)
        # ax_vel_res.plot(vel_res, obs_data.height_lag/1000, '.', color=color_sim)
        # ax_lag_res.plot(lag_res, obs_data.height_lag/1000, '.', color=color_sim)

        # LUMINOSITY residuals (station-based)
        for station in np.unique(obs_data.stations_lum):
            mask = np.where(obs_data.stations_lum == station)
            ax_lum_res.plot(lum_res[mask], obs_data.height_lum[mask]/1000, '.', color=station_colors[station])

        # MAGNITUDE residuals (station-based)
        for station in np.unique(obs_data.stations_lum):
            mask = np.where(obs_data.stations_lum == station)
            ax_mag_res.plot(mag_res[mask], obs_data.height_lum[mask]/1000, '.', color=station_colors[station])

        # VELOCITY residuals (station-based)
        for station in np.unique(obs_data.stations_lag):
            mask = np.where(obs_data.stations_lag == station)
            height_plot = obs_data.height_lag[mask]/1000
            vel_plot = vel_res[mask]
            ax_vel_res.plot(vel_plot[1:], height_plot[1:], '.', color=station_colors[station])

        # LAG residuals (station-based)
        for station in np.unique(obs_data.stations_lag):
            mask = np.where(obs_data.stations_lag == station)
            ax_lag_res.plot(lag_res[mask], obs_data.height_lag[mask]/1000, '.', color=station_colors[station])

    # take the index of the unique station_handles and station_labels
    unique_stations = {}
    for h, l in zip(station_handles, station_labels):
        if l not in unique_stations:  # exclude stations 01G and 02G from the legend  and l != '01G' and l != '02G'
            unique_stations[l] = h
    station_handles = list(unique_stations.values())
    station_labels  = list(unique_stations.keys())

    handles = [best_line] + station_handles
    labels  = [label_sim] + station_labels
    ax_lag.legend(handles, labels, loc='best', fontsize=8, frameon=True)

    ### Finalize and save ###
    plt.tight_layout()
    fig.savefig(os.path.join(output_folder, file_name + '_vs_height.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plotDynestyResults(dynesty_run_results, obs_data, flags_dict, fixed_values, output_folder='', file_name='', log_file='', cores=None, save_backup=False):
    """ Plot the dynesty results (trace plots, corner plots) and save them.
    
    Arguments:
        dynesty_run_results: [object] Result object from dynesty sampling.
        obs_data: [object] Object containing the observational data.
        flags_dict: [dict] Dictionary of flags for parameters.
        fixed_values: [dict] Dictionary of fixed parameter values.

    Keyword arguments:
        output_folder: [str] Path to save plots. '' by default.
        file_name: [str] Base name for output files. '' by default.
        log_file: [str] Path to log file. '' by default.
        cores: [int] Number of cores (unused in this function but kept for signature compatibility). None by default.
        save_backup: [bool] Flag to save backup json files. False by default.

    Return:
        None

    """

    if not DYNESTY_FOUND:
        print("Dynesty package not found. Install dynesty to use the Dynesty functions.")
        return

    if log_file == '':
        log_file = os.path.join(output_folder, f"log_{file_name}.txt")

    # check if _combined in the file_name
    if '_combined' in file_name:
        # remove _combined from the file_name
        file_name = file_name.replace('_combined', '')

    print(dynesty_run_results.summary())
    print('information gain:', dynesty_run_results.information[-1])
    print('niter i.e number of metsim simulated events\n')

    fig, axes = dyplot.runplot(dynesty_run_results,
                                label_kwargs={"fontsize": 15},  # Reduce axis label size
                                )
    # save the figure
    plt.savefig(output_folder +os.sep+ file_name +'_dynesty_runplot.png', 
                bbox_inches='tight',
                pad_inches=0.1,       # a little padding around the edge
                dpi=300)
    plt.close(fig)

    variables = list(flags_dict.keys())

    logwt = dynesty_run_results.logwt

    # Subtract the maximum logwt for numerical stability
    logwt_shifted = logwt - np.max(logwt)
    weights = np.exp(logwt_shifted)

    # Normalize so that sum(weights) = 1
    weights /= np.sum(weights)

    samples_equal = dynesty.utils.resample_equal(dynesty_run_results.samples, weights)
    # all_samples = dynesty.utils.resample_equal(dynesty_run_results.samples, weights)

    # Mapping of original variable names to LaTeX-style labels
    variable_map = {
        'v_init': r"$v_0$ [km/s]",
        'zenith_angle': r"$z_c$ [rad]",
        'm_init': r"$m_0$ [kg]",
        'rho': r"$\rho$ [kg/m$^3$]",
        'sigma': r"$\sigma$ [kg/MJ]",
        'erosion_height_start': r"$h_e$ [km]",
        'erosion_coeff': r"$\eta$ [kg/MJ]",
        'erosion_mass_index': r"$s$",
        'erosion_mass_min': r"$m_{l}$ [kg]",
        'erosion_mass_max': r"$m_{u}$ [kg]",
        'erosion_height_change': r"$h_{e2}$ [km]",
        'erosion_coeff_change': r"$\eta_{2}$ [kg/MJ]",
        'erosion_rho_change': r"$\rho_{2}$ [kg/m$^3$]",
        'erosion_sigma_change': r"$\sigma_{2}$ [kg/MJ]",
        'compressive_strength': r"$P_{compress}$ [Pa]",
        'disruption_mass_index': r"$s$",
        'disruption_mass_min_ratio': r"$m_{min}/m_{disr}$",
        'disruption_mass_max_ratio': r"$m_{max}/m_{disr}$",
        'disruption_mass_grain_ratio': r"$m_{gr}/m_{disr}$",
        'height': r"$h$ [km]",
        'mass_percent': r"$m_{percent}$ [\%]",
        'number': r"$N$",
        'sigma': r"$\sigma$ [kg/MJ]",
        'erosion_coeff': r"$\eta$ [kg/MJ]",
        'grain_mass_min': r"$m_{l}$ [kg]",
        'grain_mass_max': r"$m_{u}$ [kg]",
        'mass_index': r"$s$",
        'noise_lag': r"$\sigma_{lag}$ [m]",
        'noise_lum': r"$\sigma_{lum}$ [W]"
    }

    # Mapping of original variable names to LaTeX-style labels
    variable_map_plot = {
        'v_init': r"$v_0$ [m/s]",
        'zenith_angle': r"$z_c$ [rad]",
        'm_init': r"$m_0$ [kg]",
        'rho': r"$\rho$ [kg/m$^3$]",
        'sigma': r"$\sigma$ [kg/J]",
        'erosion_height_start': r"$h_e$ [m]",
        'erosion_coeff': r"$\eta$ [kg/J]",
        'erosion_mass_index': r"$s$",
        'erosion_mass_min': r"$m_{l}$ [kg]",
        'erosion_mass_max': r"$m_{u}$ [kg]",
        'erosion_height_change': r"$h_{e2}$ [m]",
        'erosion_coeff_change': r"$\eta_{2}$ [kg/J]",
        'erosion_rho_change': r"$\rho_{2}$ [kg/m$^3$]",
        'erosion_sigma_change': r"$\sigma_{2}$ [kg/J]",
        'compressive_strength': r"$P_{compress}$ [Pa]",
        'disruption_mass_index': r"$s$",
        'disruption_mass_min_ratio': r"$m_{min}/m_{disr}$",
        'disruption_mass_max_ratio': r"$m_{max}/m_{disr}$",
        'disruption_mass_grain_ratio': r"$m_{gr}/m_{disr}$",
        'height': r"$h$ [m]",
        'mass_percent': r"$m_{percent}$ [\%]",
        'number': r"$N$",
        'sigma': r"$\sigma$ [kg/J]",
        'erosion_coeff': r"$\eta$ [kg/J]",
        'grain_mass_min': r"$m_{l}$ [kg]",
        'grain_mass_max': r"$m_{u}$ [kg]",
        'mass_index': r"$s$",
        'noise_lag': r"$\sigma_{lag}$ [m]",
        'noise_lum': r"$\sigma_{lum}$ [W]"
    }

    # check if there are variables in the flags_dict that are not in the variable_map
    for variable in variables:
        
        fragmentation_regex = r'_([A-Z]{1,2})\d+'
        # check if variable is a fragmentation variable
        if re.search(fragmentation_regex, variable):
            # save in regex_var and take the variable without the fragmentation part
            fragm_type = re.search(fragmentation_regex, variable).group(0)
            # print(f"Found fragmentation variable: {fragm_type}")
            # now take the one that matches the regex
            variable_with_no_fragm_type = variable.replace(fragm_type, "")
            # print(f"Found fragmentation type: {variable_with_no_fragm_type}")
            # delete the _ from fragm_type
            fragm_type = fragm_type.replace("_", "")
            # add to the fragmentation map for the variable the name that is identical to regex_var
            variable_map[variable] = fragm_type+' '+variable_map[variable_with_no_fragm_type]
            variable_map_plot[variable] = fragm_type+' '+variable_map_plot[variable_with_no_fragm_type]

        if variable not in variable_map:
            print(f"Warning: {variable} not found in variable_map")
            # Add the variable to the map with a default label
            variable_map[variable] = variable
        if variable not in variable_map_plot:
            print(f"Warning: {variable} not found in variable_map_plot")
            # Add the variable to the map with a default label
            variable_map_plot[variable] = variable
    labels = [variable_map[variable] for variable in variables]
    labels_plot = [variable_map_plot[variable] for variable in variables]

    ndim = len(variables)
    sim_num = np.argmax(dynesty_run_results.logl)
    # print('Best Fit index:', sim_num)
    # copy the Best Fit values
    best_guess = copy.deepcopy(dynesty_run_results.samples[sim_num])
    best_guess_table = copy.deepcopy(dynesty_run_results.samples[sim_num])
    # for variable in variables: for 
    for i, variable in enumerate(variables):
        if 'log' in flags_dict[variable]:  
            samples_equal[:, i] = 10**(samples_equal[:, i])
            # all_samples[:, i] = 10**(all_samples[:, i])
            best_guess[i] = 10**(best_guess[i])
            best_guess_table[i] = 10**(best_guess_table[i])
            labels_plot[i] =r"$\log_{10}$(" +labels_plot[i]+")"
        # check variable is 'v_init' or 'erosion_height_start' divide by 1000
        if 'v_init' in variable or 'height' in variable:
            samples_equal[:, i] = samples_equal[:, i]/1000
            best_guess_table[i] = best_guess_table[i]/1000
        # check variable is 'erosion_coeff' or 'sigma' divide by 1e6
        if 'sigma' in variable or 'erosion_coeff' in variable:
            samples_equal[:, i] = samples_equal[:, i]*1e6
            best_guess_table[i] = best_guess_table[i]*1e6

    constjson_bestfit = Constants()

    constjson_bestfit.__dict__['P_0m'] = obs_data.P_0m
    constjson_bestfit.__dict__['lum_eff_type'] = obs_data.lum_eff_type
    constjson_bestfit.__dict__['disruption_on'] = obs_data.disruption_on
    constjson_bestfit.__dict__['dens_co'] = obs_data.dens_co
    constjson_bestfit.__dict__['dt'] = obs_data.dt
    constjson_bestfit.__dict__['h_kill'] = obs_data.h_kill
    constjson_bestfit.__dict__['v_kill'] = obs_data.v_kill

    # change Constants that have the same variable names and the one fixed
    for variable in variables:
        if variable in constjson_bestfit.__dict__.keys():
            constjson_bestfit.__dict__[variable] = best_guess[variables.index(variable)]

    # do te same for the fixed values
    for variable in fixed_values.keys():
        if variable in constjson_bestfit.__dict__.keys():
            constjson_bestfit.__dict__[variable] = fixed_values[variable]
            print(f"Fixed value for {variable}: {fixed_values[variable]}")

    def convertToSerializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif obj is None:
            return None
        elif isinstance(obj, dict):
            return {key: convertToSerializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convertToSerializable(item) for item in obj]
        else:
            return obj

    print('Best Fit:')
    # write the best fit variable names and then the Best Fit values
    for i in range(len(best_guess)):
        print(variables[i],':\t', best_guess[i])
    # print('logL:', dynesty_run_results.logl[sim_num])
    # copy the dynesty_run_results results to a variable avoid overwriting the original one
    dynesty_run_results_new = copy.deepcopy(dynesty_run_results)
    best_guess_logL = logLikelihoodDynesty(dynesty_run_results_new.samples[sim_num], obs_data, flags_dict, fixed_values, timeout=20)
    print('logL:', best_guess_logL, ' dynesty logL:', dynesty_run_results.logl[sim_num])

    real_logL = None
    diff_logL = None
    if hasattr(obs_data, 'const'):
        
        # copy the variables list
        variables_real = copy.deepcopy(variables)
        flags_dict_real = copy.deepcopy(flags_dict)
        fixed_values_real = copy.deepcopy(fixed_values)
        # chek in variables there is noise_lag or noise_lum and if so remove it from the list
        if 'noise_lag' in variables:
            variables_real.remove('noise_lag')
            del flags_dict_real['noise_lag']
            # add in the fixed_values_real the noise_lag value that was introduced
            fixed_values_real['noise_lag'] = obs_data.noise_lag
        if 'noise_lum' in variables:
            variables_real.remove('noise_lum')
            del flags_dict_real['noise_lum']
            # add in the fixed_values_real the noise_lum value that was introduced
            fixed_values_real['noise_lum'] = obs_data.noise_lum

        # feed variables in the obs_data.const of obs_data as guess_var
        guess_real = [obs_data.const[variable] for variable in variables_real if variable in obs_data.const]

        # remove the 'log' from the flags_dict_copy
        for variable in variables_real:
            if 'log' in flags_dict_real[variable]:
                flags_dict_real[variable] = [flag for flag in flags_dict_real[variable] if flag != 'log']

        real_logL = logLikelihoodDynesty(guess_real, obs_data, flags_dict_real, fixed_values_real, timeout=20)
        diff_logL = best_guess_logL - real_logL
        # use logLikelihoodDynesty to compute the logL
        print('REAL logL:', real_logL)
        print('DIFF logL:', diff_logL)

    ### PLOT best fit ###

    # create a folder to save the fit plots
    if not os.path.exists(output_folder +os.sep+ 'fit_plots'):
        os.makedirs(output_folder +os.sep+ 'fit_plots')

    best_guess_obj_plot = runSimulationDynesty(best_guess, obs_data, variables, fixed_values)

    # find the index of m_init in variables
    i_m_init = variables.index('m_init')
    tau = (calcRadiatedEnergy(np.array(obs_data.time_lum), np.array(obs_data.absolute_magnitudes), P_0m=obs_data.P_0m))*2/(samples_equal[:, i_m_init]*obs_data.velocities[0]**2)*100
    # calculate the weights calculate the weighted median and the 95 CI for tau
    # tau_low95, tau_median, tau_high95 = _quantile(tau, [0.025, 0.5, 0.975],  weights=weights)
    tau_median, tau_low95, tau_high95 = _plot_distrib_weighted(
        tau,
        weights=weights,
        output_folder=output_folder,
        file_name=file_name,
        var_name='tau',
        label='$\\tau$ [%]',
        colors='olive')
    print(f"Tau: {tau_median:.4f} 95CI ({tau_low95:.4f} - {tau_high95:.4f}) %")

    # find erosion change height
    if 'erosion_height_change' in variables and 'm_init' in variables:
        erosion_height_change = best_guess[variables.index('erosion_height_change')]
        m_init = best_guess[variables.index('m_init')]

        heights = np.array(best_guess_obj_plot.leading_frag_height_arr, dtype=np.float64)[:-1]
        mass_best = np.array(best_guess_obj_plot.mass_total_active_arr, dtype=np.float64)[:-1]

        # mass_before = mass_best[np.argmin(np.abs(heights - erosion_height_change))]
        mass_before = best_guess_obj_plot.const.mass_at_erosion_change
        # if mass_before is None use the old method
        if mass_before is None:
            mass_before = mass_best[np.argmin(np.abs(heights - erosion_height_change))]
        
        # # precise erosion tal energy calculation ########################
        rho_total_arr = samples_equal[:, variables.index('rho')].astype(float)*(abs(m_init-mass_before)/m_init) + samples_equal[:, variables.index('erosion_rho_change')].astype(float)*(mass_before/m_init)
    else:
        rho_total_arr = samples_equal[:, variables.index('rho')].astype(float)

    rho_median_approx, rho_low95_approx, rho_high95_approx = _plot_distrib_weighted(
        rho_total_arr,
        weights=weights,
        output_folder=output_folder,
        file_name=file_name,
        var_name='rho_mass_weighted',
        label='$\\rho$ [kg/m$^3$]')
    # rho_low95_approx, rho_median_approx, rho_high95_approx = _quantile(rho_total_arr, [0.025, 0.5, 0.975], weights=weights)
    print(f"Approx. mass weighted $\\rho$ : {rho_median_approx:.2f} 95CI ({rho_low95_approx:.2f} - {rho_high95_approx:.2f}) kg/m^3")

    # inerpoate the abs_magnitude_arr to the leading_frag_height_arr
    sim_mag = np.interp(obs_data.height_lum, 
                                    np.flip(best_guess_obj_plot.leading_frag_height_arr), 
                                    np.flip(best_guess_obj_plot.abs_magnitude))
    
    # make the difference between the no_noise_mag and the obs_data.abs_magnitude
    sim_diff_mag = sim_mag - obs_data.absolute_magnitudes

    obs_data.noise_mag = np.std(sim_diff_mag)

    lum_eff_val = tau_median
    # fid the fixed_values that have the lum_eff
    for key in fixed_values.keys():
        # exact name is lum_eff
        if 'lum_eff' == key:    
            print(f"Fixed value for {key}: {fixed_values[key]}")
            lum_eff_val = fixed_values[key]
            break


    # try:
    #     # find the index of m_init in variables
    #     tau_real = (calcRadiatedEnergy(np.array(obs_data.time_lum), np.array(obs_data.absolute_magnitudes), P_0m=obs_data.P_0m))/(simpson(np.array(best_guess_obj_plot.luminosity_arr[index_up:index_down]),x=np.array(best_guess_obj_plot.time_arr[index_up:index_down]))/lum_eff_val)*100
    #     print(f"first heigth obs: {np.max(obs_data.height_lum):.2f} m, last height obs: {obs_data.height_lum[-1]:.2f} m, first height sim: {best_guess_obj_plot.leading_frag_height_arr[index_up]:.2f} m, last height sim: {best_guess_obj_plot.leading_frag_height_arr[index_down]:.2f} m")
    #     print(f"total radiated energy:", calcRadiatedEnergy(np.array(obs_data.time_lum), np.array(obs_data.absolute_magnitudes), P_0m=obs_data.P_0m), "J and total simulated radiated energy:",simpson(np.array(best_guess_obj_plot.luminosity_arr[index_up:index_down]),x=np.array(best_guess_obj_plot.time_arr[index_up:index_down]))/lum_eff_val,"J")
    #     print(f"Tau real best fit: {tau_real:.4f} %")
    # except Exception as e:
    #     print("Error calculating tau real:", e)
    #     tau_real = None

    # Plot the data with residuals and the best fit
    plotSimVsObsResiduals(obs_data, best_guess_obj_plot, output_folder +os.sep+ 'fit_plots', file_name + "_best_fit")
    plotObsVsHeight(obs_data, best_guess_obj_plot, output_folder +os.sep+ 'fit_plots', file_name + "_best_fit")
    # plotObsVsHeight(obs_data, None, output_folder , file_name + "_data_only")

    _ = constructConstants(best_guess, obs_data, variables, fixed_values, dir_path=output_folder +os.sep+ 'fit_plots', file_name= file_name + '_sim_fit_dynesty_BestGuess.json')

    ### TABLE OF POSTERIOR SUMMARY STATISTICS ###

    def summaryResultsTable(results,
                            variables,
                            labels_plot,
                            flags_dict_total,
                            smooth=0.02):
        """ Summarize dynesty results, using the sample of max weight as the mode.
        
        Arguments:
            results: [object] Dynesty results object.
            variables: [list] List of variable names.
            labels_plot: [list] List of variable labels for plotting.
            flags_dict_total: [dict] Dictionary of flags for parameter transformation.

        Keyword arguments:
            smooth: [float or int] Smoothing parameter for histogram, or number of bins if int. 0.02 by default.

        Return:
            summary_df: [pandas.DataFrame] DataFrame containing detailed statistics (Low95, Mode, Mean, Median, High95) with units.
            summary_df_sml: [pandas.DataFrame] DataFrame containing statistics with only log transformations applied.

        """
        samples = results.samples               # shape (nsamps, ndim)
        weights = results.importance_weights()  # shape (nsamps,)

        # normalize weights
        w = weights.copy()
        w /= np.sum(w)

        # find the single sample index with highest weight
        mode_idx = np.nanargmax(w)   # index of peak-weight sample
        mode_raw = samples[mode_idx] # array shape (ndim,)

        rows = []
        rows_sml = []
        for i, (var, lab) in enumerate(zip(variables, labels_plot)):
            x = samples[:, i].astype(float)
            # mask out NaNs
            mask = ~np.isnan(x)
            x_valid = x[mask]
            w_valid = w[mask]
            if x_valid.size == 0:
                rows.append((var, lab, *([np.nan]*5)))
                rows_sml.append((var, lab, *([np.nan]*5)))
                continue
            # renormalize
            w_valid /= np.sum(w_valid)

            # weighted quantiles
            low95, med, high95 = _quantile(x_valid,
                                        [0.025, 0.5, 0.975],
                                        weights=w_valid)
            # weighted mean
            mean_raw = np.sum(x_valid*w_valid)
            # simple mode from max-weight sample
            mode_value = mode_raw[i]

            # mode via corner logic
            lo, hi = np.min(x), np.max(x)
            if isinstance(smooth, int):
                hist, edges = np.histogram(x, bins=smooth, weights=w, range=(lo,hi))
            else:
                nbins = int(round(10./smooth))
                hist, edges = np.histogram(x, bins=nbins, weights=w, range=(lo,hi))
                hist = norm_kde(hist, 10.0)
            centers = 0.5*(edges[1:] + edges[:-1])
            mode_Ndim = centers[np.argmax(hist)]

            # now apply your log & unit transforms *after* computing stats
            def transform(v):
                if 'log' in flags_dict_total.get(var, ''):
                    v = 10**v
                if 'v_init' in var or 'height' in var:
                    v = v/1e3
                if 'sigma' in var or 'erosion_coeff' in var:
                    v = v*1e6
                return v
            
            def log_transf(v):
                if 'log' in flags_dict_total.get(var, ''):
                    v = 10**v
                return v

            rows.append((
                var,
                lab,
                transform(low95),
                transform(mode_value),
                transform(mode_Ndim),
                transform(mean_raw),
                transform(med),
                transform(high95),
            ))

            rows_sml.append((
                var,
                lab,
                log_transf(low95),
                log_transf(mode_value),
                log_transf(mode_Ndim),
                log_transf(mean_raw),
                log_transf(med),
                log_transf(high95),
            ))
        # sort by variable name


        return pd.DataFrame( rows,columns=["Variable","Label","Low95","Mode","Mode_{Ndim}","Mean","Median","High95"]), pd.DataFrame( rows_sml,columns=["Variable","Label","Low95","Mode","Mode_{Ndim}","Mean","Median","High95"])

    
    summary_df, summary_df_sml = summaryResultsTable(
    dynesty_run_results,
    variables,
    labels,
    flags_dict
    )

    posterior_mean = summary_df['Mean'].values
    posterior_median = summary_df['Median'].values  
    approx_modes = summary_df['Mode'].values
    approx_modes_Ndim = summary_df['Mode_{Ndim}'].values
    lower_95 = summary_df['Low95'].values
    upper_95 = summary_df['High95'].values

    all_samples_mean = summary_df_sml['Mean'].values
    all_samples_median = summary_df_sml['Median'].values
    approx_modes_all = summary_df_sml['Mode'].values
    approx_modes_Ndim_all = summary_df_sml['Mode_{Ndim}'].values

    # # Posterior mean (per dimension)
    # posterior_mean = np.mean(samples_equal, axis=0)      # shape (ndim,)
    # all_samples_mean = np.mean(all_samples, axis=0)      # shape (ndim,)

    # # Posterior median (per dimension)
    # posterior_median = np.median(samples_equal, axis=0)  # shape (ndim,)
    # all_samples_median = np.median(all_samples, axis=0)  # shape (ndim,)

    # # 95% credible intervals (2.5th and 97.5th percentiles)
    # lower_95 = np.percentile(samples_equal, 2.5, axis=0)   # shape (ndim,)
    # upper_95 = np.percentile(samples_equal, 97.5, axis=0)  # shape (ndim,)

    # # Function to approximate mode using histogram binning
    # def approximate_mode_1d(samples):
    #     hist, bin_edges = np.histogram(samples, bins='auto', density=True)
    #     idx_max = np.argmax(hist)
    #     return 0.5*(bin_edges[idx_max] + bin_edges[idx_max + 1])

    # approx_modes = [approximate_mode_1d(samples_equal[:, d]) for d in range(ndim)]
    # approx_modes_all = [approximate_mode_1d(all_samples[:, d]) for d in range(ndim)]

    ### MODE PLOT and json ###

    approx_mode_obj_plot = runSimulationDynesty(approx_modes_all, obs_data, variables, fixed_values)

    # Plot the data with residuals and the best fit
    plotSimVsObsResiduals(obs_data, approx_mode_obj_plot, output_folder +os.sep+ 'fit_plots', file_name+'_mode','red', 'Mode')
    plotObsVsHeight(obs_data, approx_mode_obj_plot, output_folder +os.sep+ 'fit_plots', file_name + "_mode_height", color_sim='red', label_sim='Mode')

    _ = constructConstants(approx_modes_all, obs_data, variables, fixed_values, dir_path=output_folder +os.sep+ 'fit_plots', file_name= file_name + '_sim_fit_dynesty_mode.json')

    ### MEAN PLOT and json ###

    mean_obj_plot = runSimulationDynesty(all_samples_mean, obs_data, variables, fixed_values)

    # Plot the data with residuals and the best fit
    plotSimVsObsResiduals(obs_data, mean_obj_plot, output_folder +os.sep+ 'fit_plots', file_name+'_mean','blue', 'Mean')
    plotObsVsHeight(obs_data, mean_obj_plot, output_folder +os.sep+ 'fit_plots', file_name + "_mean_height", color_sim='blue', label_sim='Mean')

    _ = constructConstants(all_samples_mean, obs_data, variables, fixed_values, dir_path=output_folder +os.sep+ 'fit_plots', file_name= file_name + '_sim_fit_dynesty_mean.json')

    ### MEDIAN PLOT and json ###

    median_obj_plot = runSimulationDynesty(all_samples_median, obs_data, variables, fixed_values)

    # Plot the data with residuals and the best fit
    plotSimVsObsResiduals(obs_data, median_obj_plot, output_folder +os.sep+ 'fit_plots', file_name+'_median','cornflowerblue', 'Median')
    plotObsVsHeight(obs_data, median_obj_plot, output_folder +os.sep+ 'fit_plots', file_name + "_median_height", color_sim='cornflowerblue', label_sim='Median')

    _ = constructConstants(all_samples_median, obs_data, variables, fixed_values, dir_path=output_folder +os.sep+ 'fit_plots', file_name= file_name + '_sim_fit_dynesty_median.json')

    ### PLOT JSON DATA VS OBS ###

    # check if dynesty_run_results_new.samples[sim_num] or fixed_values have noise_lag or noise_lum
    if 'noise_lag' in variables:
        best_noise_lag = dynesty_run_results_new.samples[sim_num][variables.index('noise_lag')]
    elif 'noise_lag' in fixed_values.keys():
        best_noise_lag = fixed_values['noise_lag']
    else:
        print('No noise_lag found in variables or fixed_values')
    if 'noise_lum' in variables:
        best_noise_lum = dynesty_run_results_new.samples[sim_num][variables.index('noise_lum')]
    elif 'noise_lum' in fixed_values.keys():
        best_noise_lum = fixed_values['noise_lum']
    else:
        print('No noise_lum found in variables or fixed_values')

    print('Correct the LogL with the best noise values for the json files (if any):')
    plotJSONDataVsObs(obs_data,output_folder,best_noise_lum=best_noise_lum,best_noise_lag=best_noise_lag)

    ### PLOT posterior bands vs height and save to backup ###

    if save_backup:
        
        # check if there is a _posterior_backup.pkl.gz file in the output_folder
        backup_file_check = os.path.join(output_folder, f"{file_name}_posterior_backup.pkl.gz")
        if os.path.exists(backup_file_check):
            print(f"Loading existing backup file: {backup_file_check}")
             # to load the backup file later
            with gzip.open(backup_file_check, "rb") as f:
                backup_small = pickle.load(f)

            bands_lum = backup_small['bands']['lum']
            bands_mag = backup_small['bands']['mag']
            bands_vel = backup_small['bands']['vel']
            bands_lag = backup_small['bands']['lag']

            best_lum = backup_small['best_guess']['luminosity']
            best_mag = backup_small['best_guess']['abs_magnitude']
            best_vel = backup_small['best_guess']['velocity']
            best_lag = backup_small['best_guess']['lag']

            hl = np.asarray(obs_data.height_lum)
            hv = np.asarray(obs_data.height_lag)

            _plot_bands_obs_best(
                obs_data,
                hl, hv,
                bands_lum, bands_mag, bands_vel, bands_lag,
                best_lum, best_mag, best_vel, best_lag,
                output_folder=output_folder,
                file_name=f'{file_name}',
                color_best='black',
                label_best='Best Fit'
            )

        else:
            ## TAKES A LOT OF TIME IF NSAMPLES IS NONE ###
            print(f"Running all the simulations to have the uncertaty regions with {cores} cores")        

            # # ONLY FOR TESTING PURPOSES
            # backup_data = posteriorBandsTopkCheck(
            #     dynesty_run_results, obs_data, flags_dict, fixed_values,
            #     top_k=10, output_folder=output_folder, file_name=f'{file_name}_top1000_check',
            # )

            backup_data = posteriorBandsVsHeightParallel(
                dynesty_results=dynesty_run_results,
                obs_data=obs_data,
                flags_dict=flags_dict,
                fixed_values=fixed_values,
                output_folder=output_folder,
                file_name=f'{file_name}',
                nsamples=None,  # use all samples
                n_workers=cores,
            )

            # save the backup_data to a json file
            backup_small = {
                "dynesty": {
                    "file_name": file_name,
                    "samples_eq": backup_data["samples_eq"].tolist(),
                    "weights": backup_data["weights"].tolist(),
                    "variables": variables,
                    "flags_dict": flags_dict,
                    "fixed_values": fixed_values,
                    "const_backups": backup_data["const_backups"],
                    "rho_array": backup_data["rho_array"],
                    "rho_mass_weighted_estimate": {k: float(v) for k, v in backup_data["rho_mass_weighted_estimate"].items()}
                },

                "best_guess": {
                    k: np.asarray(v, dtype=np.float32)
                    for k, v in backup_data["best_guess"].items()
                },

                "bands": {
                    k: {q: v.astype(np.float32) for q, v in backup_data["bands"][k].items()}
                    for k in ("lum", "mag", "vel", "lag")
                },
            }

            backup_file = os.path.join(output_folder, f"{file_name}_posterior_backup.pkl.gz")
            with gzip.open(backup_file, "wb", compresslevel=4) as f:
                pickle.dump(backup_small, f, protocol=pickle.HIGHEST_PROTOCOL)

            # # to load the backup file later
            # backup_file = os.path.join(output_folder, f"{file_name}_posterior_backup.pkl.gz")
            # with gzip.open(backup_file, "rb") as f:
            #     backup_small = pickle.load(f)

        rho_median_real, rho_lo_real, rho_hi_real = backup_small['dynesty']['rho_mass_weighted_estimate']['median'], backup_small['dynesty']['rho_mass_weighted_estimate']['low95'], backup_small['dynesty']['rho_mass_weighted_estimate']['high95']
        rho_total_arr = backup_small['dynesty']['rho_array']
        _plot_distrib_weighted(
            rho_total_arr,
            weights=weights,
            output_folder=output_folder,
            file_name=file_name,
            var_name='rho_mass_weighted',
            label='$\\rho$ [kg/m$^3$]'
        )
        print(
            f"Real mass weighted $\\rho$ : {rho_median_real:.2f} 95CI ({rho_lo_real:.2f} - {rho_hi_real:.2f}) kg/m^3"
        )


    ### GENERATE LATEX TABLE ###

    truth_values_plot = {}
    # if 'dynesty_run_results has const
    if hasattr(obs_data, 'const'):

        truth_values_plot = {}

        # Extract values from dictionary
        for variable in variables:
            if variable in obs_data.const:  # Use dictionary lookup instead of hasattr()
                truth_values_plot[variable] = obs_data.const[variable]
            else:
                print(f"Warning: {variable} not found in obs_data.const")

        # if 'noise_lag' take it from obs_data.noise_lag
        if 'noise_lag' in flags_dict.keys():
            truth_values_plot['noise_lag'] = obs_data.noise_lag
        # if 'noise_mag' take it from obs_data.noise_mag
        if 'noise_lum' in flags_dict.keys():
            truth_values_plot['noise_lum'] = obs_data.noise_lum

        # Convert to array safely
        truths = np.array([truth_values_plot.get(variable, np.nan) for variable in variables])

        # Apply log10 safely if needed
        for variable in variables:
            if 'log' in flags_dict.get(variable, []):
                if variable in truth_values_plot:
                    truth_values_plot[variable] = np.log10(truth_values_plot[variable]) #np.log10(np.maximum(truth_values_plot[variable], 1e-10))
                else:
                    print(f"Skipping {variable}: Missing from truth_values_plot")

        for i, variable in enumerate(variables): 
            # check variable is 'v_init' or 'erosion_height_start' divide by 1000
            if 'v_init' in variable or 'height' in variable:
                truths[i] = truths[i]/1000
            # check variable is 'erosion_coeff' or 'sigma' divide by 1e6
            if 'sigma' in variable or 'erosion_coeff' in variable:
                truths[i] = truths[i]*1e6

        # Compare to true theta
        # bias = posterior_mean - truths
        bias = best_guess_table - truths
        abs_error = np.abs(bias)
        rel_error = abs_error/np.abs(truths)*100

        # Coverage check
        coverage_mask = (truths >= lower_95) & (truths <= upper_95)
        print("Coverage mask per dimension:", coverage_mask)
        print("Fraction of dimensions covered:", coverage_mask.mean())

        # Generate LaTeX table
        latex_str = r"""\begin{table}[htbp]
    \centering
    \renewcommand{\arraystretch}{1.2} % Increase row height for readability
    \setlength{\tabcolsep}{4pt} % Adjust column spacing
    \resizebox{\textwidth}{!}{ % Resizing table to fit page width
    \begin{tabular}{lrrrrrrr|rr|c}
    \hline
    Parameter & 2.5CI & True Value & Best Fit & Mode & Mean & Median & 97.5CI & Abs.Error & Rel.Error\% & Cover \\
    \hline

    """
        for i, label in enumerate(labels):
            coverage_val = "\ding{51}" if coverage_mask[i] else "\ding{55}"  # Use checkmark/x for coverage
            latex_str += (f"    {label} & {lower_95[i]:.4g} & {truths[i]:.4g} & {best_guess_table[i]:.4g} & {approx_modes[i]:.4g} "
                        f"& {posterior_mean[i]:.4g} & {posterior_median[i]:.4g} & {upper_95[i]:.4g} "
                        f"& {abs_error[i]:.4g} & {rel_error[i]:.4g}\% & {coverage_val} \\\\\n") #\hline\n

    else:
        # Generate LaTeX table
        latex_str = r"""\begin{table}[htbp]
    \centering
    \renewcommand{\arraystretch}{1.2} % Increase row height for readability
    \setlength{\tabcolsep}{4pt} % Adjust column spacing
    \resizebox{\textwidth}{!}{ % Resizing table to fit page width
    \begin{tabular}{lrrrrrr}
    \hline
    Parameter & 2.5CI & Best Fit & Mode & Mean & Median & 97.5CI\\
    \hline

    """
        # & Mode
        # {approx_modes[i]:.4g} &
        for i, label in enumerate(labels):
            latex_str += (f"    {label} & {lower_95[i]:.4g} & {best_guess_table[i]:.4g} & {approx_modes[i]:.4g} & {posterior_mean[i]:.4g} "
                        f"& {posterior_median[i]:.4g} & {upper_95[i]:.4g} \\\\\n") #  \hline\n

    latex_str += r"""\hline
    \end{tabular}} 
    \caption{"""

    # check if the file_name has a _ in it if so put \ before it
    if '_' in file_name:
        file_name_caption = file_name.replace('_', '\_')
    else:
        file_name_caption = file_name

    if hasattr(obs_data, 'const'):
        latex_str += f"Posterior summary statistics for {file_name_caption} test case. The Best Fit is the simulation with the highest likelihood. Absolute and relative errors are calculated based on the Best Fit. The Cover column indicates whether the true value lies within the 95\% CI."
    else:
        latex_str += f"Posterior summary statistics for {file_name_caption} meteor. The Best Fit is the simulation with the highest likelihood."
    latex_str += r"""}
    \label{tab:posterior_summary}
    \end{table}"""

    # Capture the printed output of summary()
    summary_buffer = io.StringIO()
    sys.stdout = summary_buffer  # Redirect stdout
    dynesty_run_results.summary()  # This prints to the buffer instead of stdout
    sys.stdout = sys.__stdout__  # Reset stdout
    summary_str = summary_buffer.getvalue()  # Get the captured text

    # Save to a .tex file
    with open(output_folder+os.sep+file_name+"_results_table.tex", "w") as f:
        f.write(latex_str)
        f.close()

    # add to the log_file
    with open(log_file, "a") as f:
        f.write('\n'+summary_str)
        f.write("H info.gain: "+str(np.round(dynesty_run_results.information[-1],3))+'\n')
        f.write("niter i.e number of metsim simulated events\n")
        f.write("ncall i.e. number of likelihood evaluations\n")
        f.write("eff(%) i.e. (niter/ncall)*100 eff. of the logL call \n")
        f.write("logz i.e. final estimated evidence\n")
        f.write("H info.gain i.e. big H very small peak posterior, low H broad posterior distribution no need for a lot of live points\n")
        f.write(f"\nTau: {tau_median:.2f} 95CI ({tau_low95:.2f} - {tau_high95:.2f}) %\n")
        f.write(f"Approx. mass weighted $\\rho$ : {rho_median_approx:.2f} 95CI ({rho_low95_approx:.2f} - {rho_high95_approx:.2f}) kg/m^3\n")
        if save_backup:
            f.write(f"Real mass weighted $\\rho$ : {rho_median_real:.2f} 95CI ({rho_lo_real:.2f} - {rho_hi_real:.2f}) kg/m^3\n")
        f.write("\nBest fit Noise:\n")
        f.write("{:<8}|{:>12}|{:>12}|{:>12}|{:>12}|\n".format(
            "Noise:", "Abs.Mag [-]", "Lum [W]", "Vel [km/s]", "Lag [m]"))
        f.write("{:<8}|{:>12.2f}|{:>12.2f}|{:>12.2f}|{:>12.2f}|\n".format(
            "Value:", obs_data.noise_mag, obs_data.noise_lum, obs_data.noise_vel/1000, obs_data.noise_lag))
        f.write("\nBest fit:\n")
        for i in range(len(best_guess)):
            f.write(variables[i]+':\t'+str(best_guess[i])+'\n')
        f.write('\nBest fit logL: '+str(best_guess_logL)+'\n') # dynesty_run_results.logl[sim_num])+'\n')
        if diff_logL is not None:
            f.write('REAL logL: '+str(real_logL)+'\n')
            f.write('Diff logL (Best fit - REAL): '+str(diff_logL)+'\n')
            f.write('Rel.Error % diff logL: '+str(abs(diff_logL/real_logL)*100)+'%\n')
            f.write('\nCoverage mask per dimension: '+str(coverage_mask)+'\n')
            f.write('Fraction of dimensions covered: '+str(coverage_mask.mean())+'\n')


    # Print LaTeX code for quick copy-pasting
    print(latex_str)

    ### Plot distribution of samples ###

    combined_samples_copy_plot = copy.deepcopy(dynesty_run_results.samples)
    labels_plot_copy_plot = labels.copy()
    if hasattr(obs_data, 'const'):
        truth_values_plot_distr = truths.copy()
    for j, var in enumerate(variables):
        if 'log' in flags_dict.get(var, '') and not ('mass_min' in var or 'mass_max' in var or 'm_init' in var or 'compressive_strength' in var):
            combined_samples_copy_plot[:, j] = 10 ** combined_samples_copy_plot[:, j]
        if 'log' in flags_dict.get(var, '') and ('mass_min' in var or 'mass_max' in var or 'm_init' in var or 'compressive_strength' in var):
            labels_plot_copy_plot[j] =r"$\log_{10}$(" +labels_plot_copy_plot[j]+")"
        if 'v_init' in var or 'height' in var:
            combined_samples_copy_plot[:, j] = combined_samples_copy_plot[:, j]/1000.0
        if 'sigma' in var or 'erosion_coeff' in var:
            combined_samples_copy_plot[:, j] = combined_samples_copy_plot[:, j]*1e6

    print('saving distribution plot...')

    # Extract from combined_results
    samples = combined_samples_copy_plot
    # samples = combined_results.samples
    weights = dynesty_run_results.importance_weights()
    w = weights/np.sum(weights)
    ndim = samples.shape[1]

    # Plot grid settings
    if ndim > 12:
        ncols = 5
    else:
        ncols = 4
    nrows = math.ceil(ndim/ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5*ncols, 2.5*nrows))
    axes = axes.flatten()

    # Define smoothing value
    smooth = 0.02  # or pass it as argument

    for i in range(ndim):
        ax = axes[i]
        x = samples[:, i].astype(float)
        mask = ~np.isnan(x)
        x_valid = x[mask]
        w_valid = w[mask]

        if x_valid.size == 0:
            ax.axis('off')
            continue

        # Compute histogram
        lo, hi = np.min(x_valid), np.max(x_valid)
        if isinstance(smooth, int):
            hist, edges = np.histogram(x_valid, bins=smooth, weights=w_valid, range=(lo, hi))
        else:
            nbins = int(round(10./smooth))
            hist, edges = np.histogram(x_valid, bins=nbins, weights=w_valid, range=(lo, hi))
            hist = norm_kde(hist, 10.0)  # dynesty-style smoothing

        centers = 0.5*(edges[1:] + edges[:-1])

        # Fill under the curve
        ax.fill_between(centers, hist, color='blue', alpha=0.6)

        # ax.plot(centers, hist, color='blue')
        ax.set_yticks([])
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))

        # Set label + quantile title
        row = summary_df.iloc[i]
        label = row["Label"]
        median = row["Median"]
        low = row["Low95"]
        high = row["High95"]
        minus = median - low
        plus = high - median

        if 'log' in flags_dict.get(variables[i], '') and ('mass_min' in variables[i] or 'mass_max' in variables[i] or 'm_init' in variables[i] or 'compressive_strength' in variables[i]):
            # put a dashed blue line at the median
            ax.axvline(np.log10(median), color='blue', linestyle='--', linewidth=1.5)
            # put a dashed Blue line at the 2.5 and 97.5 percentiles
            ax.axvline(np.log10(low), color='blue', linestyle='--', linewidth=1.5)
            ax.axvline(np.log10(high), color='blue', linestyle='--', linewidth=1.5)
            if hasattr(obs_data, 'const'):
                # put a dashed black line at the truth value
                ax.axvline(np.log10(truth_values_plot_distr[i]), color='black', linewidth=1.5)
            
        else:
            # put a dashed blue line at the median
            ax.axvline(median, color='blue', linestyle='--', linewidth=1.5)
            # put a dashed Blue line at the 2.5 and 97.5 percentiles
            ax.axvline(low, color='blue', linestyle='--', linewidth=1.5)
            ax.axvline(high, color='blue', linestyle='--', linewidth=1.5)
            if hasattr(obs_data, 'const'):
                # put a dashed black line at the truth value
                ax.axvline(truth_values_plot_distr[i], color='black', linewidth=1.5)

        fmt = lambda v: f"{v:.4g}" if np.isfinite(v) else "---"
        title = rf"{label} = {fmt(median)}$^{{+{fmt(plus)}}}_{{-{fmt(minus)}}}$"
        ax.set_title(title, fontsize=15)
        ax.set_xlabel(labels_plot_copy_plot[i], fontsize=12)

    # Remove unused axes
    for j in range(ndim, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    # plt.show()
    plt.savefig(output_folder+os.sep+file_name+'_distrib_plot.png', 
            bbox_inches='tight',
            pad_inches=0.1,       # a little padding around the edge
            dpi=300)
    plt.close(fig)

    ### Plot the trace plot ###

    print('saving trace plot...')

    if hasattr(obs_data, 'const'):
        # 25310it [5:59:39,  1.32s/it, batch: 0 | bound: 10 | nc: 30 | ncall: 395112 | eff(%):  6.326 | loglstar:   -inf < -16256.467 <    inf | logz: -16269.475 +/-  0.049 | dlogz: 15670.753 >  0.010]
        truth_plot = np.array([truth_values_plot[variable] for variable in variables])

        fig, axes = dyplot.traceplot(dynesty_run_results, truths=truth_plot, labels=labels_plot,
                                    label_kwargs={"fontsize": 15},  # Reduce axis label size
                                    title_kwargs={"fontsize": 15},  # Reduce title font size
                                    title_fmt='.2e',  # Scientific notation for titles
                                    truth_color='black', show_titles=True,
                                    trace_cmap='viridis', connect=True,
                                    connect_highlight=range(5))
        # # make a super title
        # fig.suptitle(f"Simulated Test case {file_name}", fontsize=16, fontweight='bold')  # Adjust y for better spacing

    else:

        fig, axes = dyplot.traceplot(dynesty_run_results, labels=labels_plot,
                                    label_kwargs={"fontsize": 15},  # Reduce axis label size
                                    title_kwargs={"fontsize": 15},  # Reduce title font size
                                    title_fmt='.2e',  # Scientific notation for titles
                                    show_titles=True,
                                    trace_cmap='viridis', connect=True,
                                    connect_highlight=range(5))
        # # make a super title
        # fig.suptitle(f"{file_name}", fontsize=16, fontweight='bold')  # Adjust y for better spacing

    # Adjust spacing and tick label size
    fig.subplots_adjust(hspace=0.5)  # Increase spacing between plots

    # make so that the the upper part of plot that is not used cropped out

    # save the figure
    plt.savefig(output_folder+os.sep+file_name+'_trace_plot.png', 
            bbox_inches='tight',
            pad_inches=0.1,       # a little padding around the edge
            dpi=300)

    # show the trace plot
    # plt.show()
    plt.close(fig)

    ### Plot the corner plot ###

    print('saving correlation plot...')

    # Define weighted correlation
    def weighted_corr(x, y, w):
        """Weighted Pearson correlation of x and y with weights w."""
        w = np.asarray(w)
        x = np.asarray(x)
        y = np.asarray(y)
        w_sum = w.sum()
        x_mean = (w*x).sum()/w_sum
        y_mean = (w*y).sum()/w_sum
        cov_xy = (w*(x - x_mean)*(y - y_mean)).sum()/w_sum
        var_x  = (w*(x - x_mean)**2).sum()/w_sum
        var_y  = (w*(y - y_mean)**2).sum()/w_sum
        return cov_xy/np.sqrt(var_x*var_y)


    # Trace Plots
    fig, axes = plt.subplots(ndim, ndim, figsize=(35, 15))
    axes = axes.reshape((ndim, ndim))  # reshape axes

    # Plot grid settings
    if ndim > 12:
        label_fontsize = 10
        title_fontsize = 12
    else:
        label_fontsize = 13
        title_fontsize = 18

    if hasattr(obs_data, 'const'):
        # Increase spacing between subplots
        fg, ax = dyplot.cornerplot(
            dynesty_run_results, 
            color='blue', 
            truths=truth_plot,  # Use the defined truth values
            truth_color='black', 
            show_titles=True, 
            max_n_ticks=3, 
            quantiles=None, 
            labels=labels_plot,  # Update axis labels
            label_kwargs={"fontsize": label_fontsize},  # Reduce axis label size
            title_kwargs={"fontsize": title_fontsize},  # Reduce title font size
            title_fmt='.2e',  # Scientific notation for titles
            fig=(fig, axes[:, :ndim])
        )
        # add a super title
        # fg.suptitle(f"Simulated Test case {file_name}", fontsize=16, fontweight='bold')  # Adjust y for better spacing

    else:

        # Increase spacing between subplots
        fg, ax = dyplot.cornerplot(
            dynesty_run_results, 
            color='blue', 
            show_titles=True, 
            max_n_ticks=3, 
            quantiles=None, 
            labels=labels_plot,  # Update axis labels
            label_kwargs={"fontsize": label_fontsize},  # Reduce axis label size
            title_kwargs={"fontsize": title_fontsize},  # Reduce title font size
            title_fmt='.2e',  # Scientific notation for titles
            fig=(fig, axes[:, :ndim])
        )
        # add a super title
        # fg.suptitle(f"{file_name}", fontsize=16, fontweight='bold')  # Adjust y for better spacing

    # # Reduce tick size
    # for ax_row in ax:
    #     for ax_ in ax_row:
    #         ax_.tick_params(axis='both', labelsize=6)  # Reduce tick number size

    # Apply scientific notation and horizontal tick labels
    for ax_row in ax:
        for ax_ in ax_row:
            ax_.tick_params(axis='both', labelsize=10, direction='in')

            # Set tick labels to be horizontal
            for label in ax_.get_xticklabels():
                label.set_rotation(0)
            for label in ax_.get_yticklabels():
                label.set_rotation(45)

            if ax_ is None:
                continue  # if cornerplot left some entries as None
            
            # Get the actual major tick locations.
            x_locs = ax_.xaxis.get_majorticklocs()
            y_locs = ax_.yaxis.get_majorticklocs()

            # Only update the formatter if we actually have tick locations:
            if len(x_locs) > 0:
                ax_.xaxis.set_major_formatter(FormatStrFormatter('%.4g'))
            if len(y_locs) > 0:
                ax_.yaxis.set_major_formatter(FormatStrFormatter('%.4g'))

    for i in range(ndim):
        for j in range(ndim):
            # In some corner-plot setups, the upper-right triangle can be None
            if ax[i, j] is None:
                continue
            
            # Remove y-axis labels (numbers) on the first column (j==0)
            if j != 0:
                ax[i, j].set_yticklabels([])  
                # or ax[i, j].tick_params(labelleft=False) if you prefer

            # Remove x-axis labels (numbers) on the bottom row (i==ndim-1)
            if i != ndim - 1:
                ax[i, j].set_xticklabels([])  
                # or ax[i, j].tick_params(labelbottom=False)


    # Overlay weighted correlations in the upper triangle
    norm = Normalize(vmin=-1, vmax=1)
    cmap = cm.get_cmap('coolwarm')
    samples = dynesty_run_results['samples'].T  # shape (ndim, nsamps)
    weights = dynesty_run_results.importance_weights()

    cmap = plt.colormaps['coolwarm']
    norm = Normalize(vmin=-1, vmax=1)

    for i in range(ndim):
        for j in range(ndim):
            if j <= i or ax[i, j] is None:
                continue

            panel = ax[i, j]
            x = samples[j]
            y = samples[i]
            weigh_corr = weighted_corr(x, y, weights)

            color = cmap(norm(weigh_corr))
            # paint the background patch
            panel.patch.set_facecolor(color)
            panel.patch.set_alpha(1.0)

            # fallback rectangle if needed
            panel.add_patch(
                plt.Rectangle(
                    (0,0), 1, 1,
                    transform=panel.transAxes,
                    facecolor=color,
                    zorder=0
                )
            )

            panel.text(
                0.5, 0.5,
                f"{weigh_corr:.2f}",
                transform=panel.transAxes,
                ha='center', va='center',
                fontsize=25, color='black'
            )
            panel.set_xticks([]); panel.set_yticks([])
            for spine in panel.spines.values():
                spine.set_visible(False)

    # Build the NxN matrix of weigh_corr_ij
    corr_mat = np.zeros((ndim, ndim))
    for i in range(ndim):
        for j in range(ndim):
            corr_mat[i, j] = weighted_corr(samples[i], samples[j], weights)

    # Wrap it in a DataFrame (so you get row/column labels)
    df_corr = pd.DataFrame(
        corr_mat,
        index=labels_plot,
        columns=labels_plot
    )

    # Save to CSV (or TSV, whichever you prefer)
    outpath = os.path.join(
        output_folder,
        f"{file_name}_weighted_correlation_matrix.csv"
    )
    df_corr.to_csv(outpath, float_format="%.4f")
    print(f"Saved weighted correlation matrix to:\n  {outpath}")


    # fg.subplots_adjust(
    #     left=0.05, right=0.95,    # whatever margins you like
    #     bottom=0.05, top=0.8,    # top < 1.0 to open space for the suptitle
    #     wspace=0.1, hspace=0.3)  # Increase spacing between plots

    # Adjust spacing and tick label size
    fg.subplots_adjust(wspace=0.1, hspace=0.3, top=0.978) # Increase spacing between plots

    # save the figure
    plt.savefig(output_folder+os.sep+file_name+'_correlation_plot.png', dpi=300)

    # close the figure
    plt.close(fig)

###############################################################################
# Function: read prior to generate bounds
###############################################################################


def loadPriorsAndGenerateBounds(object_meteor, file_path="", user_inputs=None):
    """ Read the prior file and generate the bounds, flags, and fixed values for the dynesty sampler.

    Arguments:
        object_meteor: [object] Meteor object containing observational data.

    Keyword arguments:
        file_path: [str] Path to the prior file. Empty string by default.
        user_inputs: [dict] Dictionary of user override values. None by default.

    Return:
        bounds: [list] List of tuples defining the bounds for each parameter.
        flags_dict: [dict] Dictionary of flags for parameter transformation (e.g., 'log', 'norm').
        fixed_values: [dict] Dictionary of fixed parameter values.

    """

    user_inputs = {} if user_inputs is None else dict(user_inputs)

    # Heights (compute defaults from object if not provided)
    # for typical meteors: begin ~ max(height), end ~ min(height)
    h_beg_default  = float(np.max(object_meteor.height_lum))
    h_end_default  = float(np.min(object_meteor.height_lum))
    h_peak_default = float(object_meteor.height_lum[np.argmax(object_meteor.luminosity)])

    h_beg  = float(user_inputs.get("h_beg",  h_beg_default))
    h_end  = float(user_inputs.get("h_end",  h_end_default))
    h_peak = float(user_inputs.get("h_peak", h_peak_default))
    v_0 = user_inputs.get("v_0", user_inputs.get("v_init", getattr(object_meteor, "v_init", np.nan)))
    m_0 = user_inputs.get("m_0", user_inputs.get("m_init", getattr(object_meteor, "m_init", np.nan)))
    zc_0 = user_inputs.get("zc_0", user_inputs.get("zenith_angle", getattr(object_meteor, "zenith_angle", np.nan)))
    n_lag0 = user_inputs.get("n_lag0", getattr(object_meteor, "noise_lag", np.nan))
    n_lum0 = user_inputs.get("n_lum0", getattr(object_meteor, "noise_lum", np.nan))

    # Helper: prefer user override, else object_meteor attr/dict
    def get_estimate(name):
        if name in user_inputs and user_inputs[name] is not None and not (isinstance(user_inputs[name], float) and np.isnan(user_inputs[name])):
            return user_inputs[name]
        if hasattr(object_meteor, name):
            return getattr(object_meteor, name)
        if name in getattr(object_meteor, "__dict__", {}):
            return object_meteor.__dict__[name]
        return None

    # Default bounds
    default_bounds = {
        "v_init": (500, np.nan),
        "zenith_angle": (0.01, np.nan),
        "m_init": (np.nan, np.nan),
        "rho": (100, 4000),
        "sigma": (0.001/1e6, 0.05/1e6),
        "erosion_height_start": (
            h_beg - 100 - (h_beg - h_peak)/2,
            h_beg + 100 + (h_beg - h_peak)/2),
        # h_beg -100- (h_beg - h_peak),\
        # h_beg +100+ abs(h_beg - h_end)), 
        "erosion_coeff": (1/1e12, 2/1e6),  # log transformation applied later
        "erosion_mass_index": (1, 3),
        "erosion_mass_min": (5e-12, 1e-9),  # log transformation applied later
        "erosion_mass_max": (1e-10, 1e-7),  # log transformation applied later
        "rho_grain": (3000, 3500),
        "erosion_height_change": (h_end - 100, h_beg + 100),
        "erosion_rho_change": (100, 4000),
        "erosion_sigma_change": (0.001/1e6, 0.05/1e6),
        "erosion_coeff_change": (1/1e12, 2/1e6),  # log transformation applied later
        "noise_lag": (10, object_meteor.noise_lag), # more of a peak around the real value
        "noise_lum": (5, object_meteor.noise_lum) # look for more values at higher uncertainty can be because of the noise
    }

    default_flags = {
        "v_init": ["norm"],
        "zenith_angle": ["norm"],
        "m_init": [],
        "rho": [],
        "sigma": [],
        "erosion_height_start": [],
        "erosion_coeff": ["log"],
        "erosion_mass_index": [],
        "erosion_mass_min": ["log"],
        "erosion_mass_max": ["log"],
        "rho_grain": [],
        "erosion_height_change": [],
        "erosion_coeff_change": ["log"],
        "erosion_rho_change": [],
        "erosion_sigma_change": [],
        "noise_lag": ["invgamma"],
        "noise_lum": ["invgamma"]
    }

    rho_grain_real = 3000
    if hasattr(object_meteor, 'const'):
        if hasattr(object_meteor.const, 'rho_grain'):
            rho_grain_real = object_meteor.const.rho_grain
        else:
            if isinstance(object_meteor.const, dict):
                if "rho_grain" in object_meteor.const:
                    rho_grain_real = object_meteor.const["rho_grain"]
                # check if object_meteor.const.rho_grain exist as key
                if "rho_grain" in object_meteor.const.keys():
                    rho_grain_real = object_meteor.const["rho_grain"]

    if file_path == "":
        print("No prior file provided. Using default bounds.")
        default_bounds.pop("zenith_angle")
        default_bounds.pop("rho_grain")
        default_bounds.pop("erosion_height_change")
        default_bounds.pop("erosion_coeff_change")
        default_bounds.pop("erosion_rho_change")
        default_bounds.pop("erosion_sigma_change")

        bounds = [default_bounds[key] for key in default_bounds]
        flags_dict = {key: default_flags.get(key, []) for key in default_bounds}
        # for the one that have log transformation, apply it
        for i, key in enumerate(default_bounds):
            if "log" in flags_dict[key]:
                bounds[i] = np.log10(bounds[i][0]), np.log10(bounds[i][1])
        # check if any of the values are np.nan and replace them with the object_meteor values
        for i, key in enumerate(default_bounds):
            bounds[i] = list(bounds[i])
            est = get_estimate(key)
            # now check if the values are np.nan and if the flag key is 'norm' then divide by 10
            if np.isnan(bounds[i][0]) and est is not None:
                bounds[i][0] = est - 10**int(np.floor(np.log10(abs(est))))
            if np.isnan(bounds[i][1]) and est is not None and "norm" in flags_dict[key]:
                bounds[i][1] = est
            elif np.isnan(bounds[i][1]) and est is not None:
                bounds[i][1] = est + 10**int(np.floor(np.log10(abs(est))))
            bounds[i] = tuple(bounds[i])
            
        # checck if stil any bounds are np.nan and raise an error
        for i, key in enumerate(default_bounds):
            if np.isnan(bounds[i][0]) or np.isnan(bounds[i][1]):
                raise ValueError(f"The value for {key} is np.nan and it is not in the dictionary")

        fixed_values = {
            "zenith_angle": float(get_estimate("zenith_angle") if get_estimate("zenith_angle") is not None else object_meteor.zenith_angle),
            "rho_grain": rho_grain_real,
            "erosion_height_change": 1,
            "erosion_coeff_change": 1/1e7,
            "erosion_rho_change": rho_grain_real,
            "erosion_sigma_change": 0.001/1e6
        }

    else:
        bounds = []
        flags_dict = {}
        fixed_values = {}

        eval_ctx = {
            "h_beg": h_beg, "h_end": h_end, "h_peak": h_peak,
            "v_0": v_0, "m_0": m_0, "zc_0": zc_0, "n_lag0": n_lag0, "n_lum0": n_lum0,
            "np": np
        }

        def safe_eval(value):
            try:
                return eval(value, {"__builtins__": {}, "np": np, **eval_ctx}, {})
            except Exception:
                return value

        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                parts = line.split('#')[0].strip().split(',')
                name = parts[0].strip()

                if "fix" in parts:
                    val = parts[1].strip() if len(parts) > 1 else "nan"
                    evaluated_val = safe_eval(val)
                    # print(f"Fixing parameter '{name}' to value: {evaluated_val}")
                    fixed_values[name] = np.nan if str(val).lower() == "nan" else evaluated_val
                    # Only proceed with fallback logic if it's numeric and NaN
                    if isinstance(fixed_values[name], (int, float)) and np.isnan(fixed_values[name]):
                        est = get_estimate(name)
                        if est is not None:
                            fixed_values[name] = est
                        elif name == "erosion_height_start":
                            fixed_values[name] = h_beg
                        else:
                            fixed_values[name] = np.mean(default_bounds[name])
                    continue

                min_val = parts[1].strip() if len(parts) > 1 else "nan"
                max_val = parts[2].strip() if len(parts) > 2 else "nan"
                flags = [flag.strip() for flag in parts[3:]] if len(parts) > 3 else []
                # Handle NaN values and default replacement
                min_val = safe_eval(min_val) if isinstance(min_val, str) and min_val.lower() != "nan" else default_bounds.get(name, (np.nan, np.nan))[0]
                max_val = safe_eval(max_val) if isinstance(max_val, str) and max_val.lower() != "nan" else default_bounds.get(name, (np.nan, np.nan))[1]
                # print(f"Setting bounds for '{name}': min={min_val}, max={max_val}, flags={flags}")

                min_val, max_val, flags = boundsMinMaxFlags(
                    name, object_meteor, min_val, max_val, flags, default_bounds, default_flags
                )
                # Store flags
                flags_dict[name] = flags
                bounds.append((min_val, max_val))

    variable_loaded = list(fixed_values.keys()) + list(flags_dict.keys())

    if 'v_init' not in variable_loaded:
        fixed_values['v_init'] = float(get_estimate("v_init") if get_estimate("v_init") is not None else object_meteor.v_init)
    if 'zenith_angle' not in variable_loaded:
        fixed_values['zenith_angle'] = float(get_estimate("zenith_angle") if get_estimate("zenith_angle") is not None else object_meteor.zenith_angle)
    if 'm_init' not in variable_loaded:
        fixed_values['m_init'] = float(get_estimate("m_init") if get_estimate("m_init") is not None else object_meteor.m_init)
    if 'rho_grain' not in variable_loaded:
        fixed_values['rho_grain'] = rho_grain_real
    if 'erosion_height_change' not in variable_loaded:
        fixed_values['erosion_height_change'] = 1

    if len(bounds) + len(fixed_values) < 10:
        raise ValueError("The number of bounds and fixed values should 10 or above")

    return bounds, flags_dict, fixed_values


def appendExtraPriorsToBounds(object_meteor, bounds, flags_dict, fixed_values, file_path, user_inputs=None):
    """ Append extra priors from a .extraprior file to existing bounds, flags, and fixed values.

    Adds numbering to identical fragment types (e.g., M0, M1).
    Ensures all required fragmentation parameters are present per block context.
    
    Arguments:
        object_meteor: [object] Meteor object containing observational data.
        bounds: [list] Existing bounds list.
        flags_dict: [dict] Existing flags dictionary.
        fixed_values: [dict] Existing fixed values dictionary.
        file_path: [str] Path to the extra prior file.

    Keyword arguments:
        user_inputs: [dict] Dictionary of user override values. None by default.

    Return:
        bounds: [list] Updated bounds list.
        flags_dict: [dict] Updated flags dictionary.
        fixed_values: [dict] Updated fixed values dictionary.

    """
    
    # optional user overrides
    user_inputs = {} if user_inputs is None else dict(user_inputs)

    # Heights (begin ~ max(height), end ~ min(height), peak ~ max luminosity)
    h_beg_default  = float(np.max(object_meteor.height_lum))
    h_end_default  = float(np.min(object_meteor.height_lum))
    h_peak_default = float(object_meteor.height_lum[np.argmax(object_meteor.luminosity)])

    h_beg  = float(user_inputs.get("h_beg",  h_beg_default))
    h_end  = float(user_inputs.get("h_end",  h_end_default))
    h_peak = float(user_inputs.get("h_peak", h_peak_default))

    # Also allow v_0/m_0/zc_0 naming if you want to reference them in expressions
    v_0 = user_inputs.get("v_0", user_inputs.get("v_init", getattr(object_meteor, "v_init", np.nan)))
    m_0 = user_inputs.get("m_0", user_inputs.get("m_init", getattr(object_meteor, "m_init", np.nan)))
    zc_0 = user_inputs.get("zc_0", user_inputs.get("zenith_angle", getattr(object_meteor, "zenith_angle", np.nan)))
    n_lag0 = user_inputs.get("n_lag0", getattr(object_meteor, "noise_lag", np.nan))
    n_lum0 = user_inputs.get("n_lum0", getattr(object_meteor, "noise_lum", np.nan))

    if not file_path.endswith(".extraprior"):
        print("WARNING: Provided file is not an .extraprior file, return bounds, flags and fixed parameters")
        return bounds, flags_dict, fixed_values

    flare_start_end, flare_start_init = findStrongestFlare(object_meteor.height_lum, object_meteor.absolute_magnitudes)

    default_bounds = {
        "height": (flare_start_end, flare_start_init),
        "sigma": (0.001/1e6, 0.05/1e6),
        "erosion_coeff": (1/1e12, 2/1e6),
        "grain_mass_min": (5e-12, 1e-9),
        "grain_mass_max": (1e-10, 1e-7),
        "mass_index": (1, 3),
        "gamma": (0.5, 1),
        "mass_percent": (10, 100),
        "number": (1, 3)
    }

    default_flags = {
        "height": [],
        "sigma": [],
        "erosion_coeff": [],
        "grain_mass_min": [],
        "grain_mass_max": [],
        "mass_index": [],
        "gamma": [],
        "mass_percent": [],
        "number": []
    }

    # copy to the default_bounds the one from bounds that are already defined
    for i, bound in enumerate(bounds):
        orig_key = list(flags_dict.keys())[i]
        key = orig_key

        if key == "erosion_mass_index":
            key = "mass_index"
        elif key == "erosion_mass_min":
            key = "grain_mass_min"
        elif key == "erosion_mass_max":
            key = "grain_mass_max"

        if key in default_bounds:
            default_bounds[key] = bound
            # print(f"Updated default_bounds[{key}] = {bound}")

            # small fix: if we remapped the name, fetch flags from the original key
            default_flags[key] = flags_dict.get(orig_key, [])
            # print(f"Updated default_flags[{key}] = {flags_dict.get(orig_key, [])}")

    extraprior_bounds = []
    extraprior_flags = {}
    extraprior_fixed = {}

    current_type = None
    current_index = -1
    current_defined_vars = []

    # safe_eval can use user-friendly symbols
    eval_ctx = {
        "h_beg": h_beg, "h_end": h_end, "h_peak": h_peak,
        "flare_start_init": flare_start_init, "flare_start_end": flare_start_end,
        "v_0": v_0, "m_0": m_0, "zc_0": zc_0, "n_lag0": n_lag0, "n_lum0": n_lum0,
        "np": np
    }

    def safe_eval(value):
        try:
            return eval(value, {"__builtins__": {}, "np": np, **eval_ctx}, {})
        except Exception:
            return value

    # make indexing per-fragment-type (M0, M1, F0, ...)
    type_counts = defaultdict(int)

    def finalize_fragment_block():
        nonlocal current_type, current_index, current_defined_vars, extraprior_fixed
        if current_type is None:
            return

        suffix = f"{current_type}{current_index}"

        if current_type == "M":
            required_vars = ["height", "erosion_coeff"]
        elif current_type == "A":
            required_vars = ["height", "sigma", "gamma"]
        elif current_type == "F":
            required_vars = ["height", "number", "mass_percent", "sigma"]
        elif current_type == "EF":
            required_vars = ["height", "number", "mass_percent", "erosion_coeff",
                             "grain_mass_min", "grain_mass_max", "mass_index"]
        elif current_type == "D":
            required_vars = ["height", "mass_percent", "grain_mass_min", "grain_mass_max", "mass_index"]
        else:
            required_vars = []

        for var in required_vars:
            var_name = f"{var}_{suffix}"
            if var_name in extraprior_fixed or var_name in extraprior_flags:
                continue

            # use sensible defaults for FIXED missing required vars (because fixed values won't be repaired later)
            if var == "height":
                default_val = float(h_beg)  # "first frame height" equivalent
            elif var == "number":
                default_val = 1
            elif var == "mass_percent":
                default_val = 10
            elif var == "gamma":
                default_val = 1.2
            elif var == "mass_index":
                default_val = 2.0
            elif var == "sigma":
                # Use typical default (mid of bounds, or pick a known canonical if you prefer)
                default_val = float(np.mean(default_bounds["sigma"]))
            elif var == "erosion_coeff":
                default_val = float(np.mean(default_bounds["erosion_coeff"]))
            elif var == "grain_mass_min":
                default_val = float(np.mean(default_bounds["grain_mass_min"]))
            elif var == "grain_mass_max":
                default_val = float(np.mean(default_bounds["grain_mass_max"]))
            else:
                default_val = 1

            extraprior_fixed[var_name] = default_val
            print(f"Added missing required param as FIX: {var_name} = {default_val}")

        current_defined_vars.clear()

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('#') or not line:
                continue

            if line.startswith('- '):
                finalize_fragment_block()
                current_type = line[2:].strip()
                current_index = type_counts[current_type]
                type_counts[current_type] += 1
                continue

            parts = line.split('#')[0].strip().split(',')
            name = parts[0].strip()
            var_name = f"{name}_{current_type}{current_index}"

            if "fix" in parts:
                val = parts[1].strip() if len(parts) > 1 else "nan"
                evaluated_val = safe_eval(val)
                extraprior_fixed[var_name] = np.nan if str(evaluated_val).lower() == "nan" else evaluated_val
                # Only proceed with fallback logic if it's numeric and NaN
                if isinstance(extraprior_fixed[var_name], (int, float)) and np.isnan(extraprior_fixed[var_name]):
                    if "number" in var_name:
                        extraprior_fixed[var_name] = 1
                    elif "mass_percent" in var_name:
                        extraprior_fixed[var_name] = 10
                    elif "height" in var_name:
                        extraprior_fixed[var_name] = float(h_beg)
                    elif name in default_bounds:
                        extraprior_fixed[var_name] = float(np.mean(default_bounds[name]))
                    else:
                        extraprior_fixed[var_name] = 1
                continue

            min_val = parts[1].strip() if len(parts) > 1 else "nan"
            max_val = parts[2].strip() if len(parts) > 2 else "nan"
            flags = [flag.strip() for flag in parts[3:]] if len(parts) > 3 else []
            # Handle NaN values and default replacement
            min_val = safe_eval(min_val) if isinstance(min_val, str) and min_val.lower() != "nan" else default_bounds.get(name, (np.nan, np.nan))[0]
            max_val = safe_eval(max_val) if isinstance(max_val, str) and max_val.lower() != "nan" else default_bounds.get(name, (np.nan, np.nan))[1]

            min_val, max_val, flags = boundsMinMaxFlags(name, object_meteor, min_val, max_val, flags, default_bounds, default_flags)

            extraprior_flags[var_name] = flags
            extraprior_bounds.append((min_val, max_val))

    finalize_fragment_block()

    bounds.extend(extraprior_bounds)
    flags_dict.update(extraprior_flags)
    fixed_values.update(extraprior_fixed)

    return bounds, flags_dict, fixed_values


def boundsMinMaxFlags(name, object_meteor, min_val, max_val, flags, default_bounds, default_flags):
    """ Calculate the bounds, min, max, and flags for a given parameter.

    Arguments:
        name: [str] Name of the parameter.
        object_meteor: [object] Meteor object containing observational data.
        min_val: [float] Minimum value for the parameter.
        max_val: [float] Maximum value for the parameter.
        flags: [list] List of flags for the parameter.
        default_bounds: [dict] Dictionary of default bounds.
        default_flags: [dict] Dictionary of default flags.

    Return:
        min_val: [float] Calculated minimum value.
        max_val: [float] Calculated maximum value.
        flags: [list] List of flags for the parameter.
    
    """

    # check if min_val is a string and replace it with np.nan
    if isinstance(min_val, str):
        # raise an error
        raise ValueError(f"{name} : ERROR loading min_val = {min_val}")
        # print(f"{name} : ERROR loading max_val = {max_val}, converting to default value")
        # min_val = np.nan
    if isinstance(max_val, str):
        raise ValueError(f"{name} : ERROR loading max_val = {max_val}")
        # print(f"{name} : ERROR loading max_val = {max_val}, converting to default value")
        # max_val = np.nan

    #### vel, mass, zenith ####
    # check if name=='v_init' or zenith_angle or m_init or erosion_height_start values are np.nan and replace them with the object_meteor values
    if np.isnan(min_val) and name in object_meteor.__dict__ and ("norm" in flags or "invgamma" in flags):
        if "norm" in default_flags[name] or "invgamma" in default_flags[name]:
            min_val = default_bounds.get(name, (np.nan, np.nan))[0]
        else:
            min_val = object_meteor.__dict__[name]/10/2
    if np.isnan(min_val) and name in object_meteor.__dict__:
        # if norm in default_flags[name] then divide by 10
        if "norm" in default_flags[name] or "invgamma" in default_flags[name]:
            min_val = object_meteor.__dict__[name] + default_bounds.get(name, (np.nan, np.nan))[0]
        else:
            # min_val = object_meteor.__dict__[name] - 10**int(np.floor(np.log10(abs(object_meteor.__dict__[name]))))#object_meteor.__dict__[name]/10/2
            min_val = 10**int(np.floor(np.log10(abs(object_meteor.__dict__[name]))) - 1)#object_meteor.__dict__[name]/10/2

    if np.isnan(max_val) and name in object_meteor.__dict__ and ("norm" in flags or "invgamma" in flags):
        max_val = object_meteor.__dict__[name]
    if np.isnan(max_val) and name in object_meteor.__dict__:
        # if norm in default_flags[name] then divide by 10
        if "norm" in default_flags[name] or "invgamma" in default_flags[name]:
            max_val = object_meteor.__dict__[name] + default_bounds.get(name, (np.nan, np.nan))[0]
        else:
            # max_val = object_meteor.__dict__[name] + 10**int(np.floor(np.log10(abs(object_meteor.__dict__[name]))))#object_meteor.__dict__[name]/10/2
            max_val = 2*10**int(np.floor(np.log10(abs(object_meteor.__dict__[name]))) + 1)#object_meteor.__dict__[name]/10/2
                    
    #### rest of variables ####
    if np.isnan(min_val) and ("norm" in flags or "invgamma" in flags):
        if "norm" in default_flags[name] or "invgamma" in default_flags[name]:
            min_val = default_bounds.get(name, (np.nan, np.nan))[0]
        else:
            min_val = (default_bounds.get(name, (np.nan, np.nan))[1]-default_bounds.get(name, (np.nan, np.nan))[0])/10/2
    elif np.isnan(min_val):
        min_val = default_bounds.get(name, (np.nan, np.nan))[0]
    
    if np.isnan(max_val) and ("norm" in flags or "invgamma" in flags):
        if "norm" in default_flags[name] or "invgamma" in default_flags[name]:
            max_val = default_bounds.get(name, (np.nan, np.nan))[1]
        else:
            max_val = np.mean([default_bounds.get(name,(np.nan, np.nan))[1],default_bounds.get(name,(np.nan, np.nan))[0]])
    elif np.isnan(max_val):
        max_val = default_bounds.get(name, (np.nan, np.nan))[1]
    
    # check if min_val > max_val, then swap them cannot have negative values
    if min_val > max_val and "invgamma" not in flags:
        print(f"Min/sigma > MAX/mean : Swapping {min_val} and {max_val} for {name}")
        min_val, max_val = max_val, min_val
                
    # Apply log10 transformation if needed
    if "log" in flags:
        # check if any values is 0 and if it is, replace it with the default value
        if min_val == 0:
            min_val = 1/1e12
        # Apply log10 transformation
        min_val, max_val = np.log10(min_val), np.log10(max_val)

    # check if any of the values are np.nan raise an error
    if np.isnan(min_val) or np.isnan(max_val):
        raise ValueError(f"The value for {name} is np.nan and it is not in the dictionary")
    # check if inf in the values and raise an error
    if np.isinf(min_val) or np.isinf(max_val):
        raise ValueError(f"The value for {name} is inf and it is not in the dictionary")
    
    return min_val, max_val, flags


def findStrongestFlare(height, magnitude, preflare_points=-1, sigma_threshold=3):
    """ Detect all flares (onset to end) and select the strongest one based on envelope length.
    
    Arguments:
        height: [array-like] Heights (descending order).
        magnitude: [array-like] Absolute magnitude (brightness).

    Keyword arguments:
        preflare_points: [int] Number of points to fit baseline parabola. -1 by default (auto-detect).
        sigma_threshold: [float] Sigma threshold for flare detection. 3 by default.

    Return:
        min_height: [float] Minimum height of the strongest flare onset range.
        max_height: [float] Maximum height of the strongest flare onset range.

    """

    def parabola(x, a, b, c):
        return a*x**2 + b*x + c
    
    height_real = height

    height = np.array(height)
    magnitude = np.array(magnitude)

    # Normalize height for numerical stability
    height_norm = height/1e5

    if preflare_points <= 1:
        # preflare_points = int(len(height)/3)
        preflare_points = int(np.argmin(magnitude)/2)
        if preflare_points <= 1:
            print("No flares detected.")
            return np.max(height_real)+100, np.min(height_real)-100

    lower_bounds = [0, -np.inf, -np.inf]
    upper_bounds = [np.inf, np.inf, np.inf]

    # Fit parabola with bounds
    popt, _ = curve_fit(
        parabola,
        height_norm[:preflare_points],
        magnitude[:preflare_points],
        bounds=(lower_bounds, upper_bounds)
    )
    # Compute parabola fit
    parabola_fit = parabola(height_norm, *popt)

    # Compute residuals (observed - parabola)
    residuals = magnitude - parabola_fit

    # Baseline residual stats
    baseline_mean = np.mean(residuals[:preflare_points])
    baseline_std = np.std(residuals[:preflare_points])

    # Loop to find all flare onset and end pairs
    flares = []
    in_flare = False
    onset_idx = None

    for idx in range(preflare_points, len(residuals)):
        if not in_flare:
            # Flare onset detection
            if residuals[idx] < baseline_mean - sigma_threshold*baseline_std:
                onset_idx = idx
                in_flare = True
        else:
            # Flare end detection (merge-back to parabola)
            if residuals[idx] >= baseline_mean - sigma_threshold*baseline_std:
                end_idx = idx
                flares.append((onset_idx, end_idx))
                in_flare = False

    # Handle case where flare never merged back
    if in_flare and onset_idx is not None:
        flares.append((onset_idx, len(residuals)-1))

    if not flares:
        print("No flares detected.")
        return np.max(height_real)+100, np.min(height_real)-100

    # Select flare with largest envelope length
    flare_lengths = [abs(height[start] - height[end]) for start, end in flares]
    strongest_idx = np.argmax(flare_lengths)
    strongest_onset_idx, strongest_end_idx = flares[strongest_idx]

    strongest_flare_onset = height[strongest_onset_idx]
    strongest_flare_end = height[strongest_end_idx]

    # now find the peak of the flare in betwenen strongest_flare_onset and strongest_flare_end
    flare_peak_idx = np.argmin(magnitude[strongest_onset_idx:strongest_end_idx]) + strongest_onset_idx
    strongest_flare_peak = height[flare_peak_idx]

    # Compute search range around onset
    # envelope_length = flare_lengths[strongest_idx]
    # search_range = envelope_fraction*envelope_length
    search_range = abs(strongest_flare_peak-strongest_flare_onset)/2
    # flare_start_range = (strongest_flare_onset - search_range, strongest_flare_onset + search_range)
    print(f"Strongest flare detected: Onset={strongest_flare_onset}, End={strongest_flare_end}, Peak={strongest_flare_peak}")
    return strongest_flare_onset - search_range, strongest_flare_onset + search_range



###############################################################################
# Load observation data and create an object
###############################################################################

def integrateLuminosity(all_simulated_time, time_fps, luminosity_arr, dt, fps, P_0m):
    """ Integrate the luminosity over time and create a new luminosity array.

    Arguments:
        all_simulated_time: [ndarray] Array of all simulated time steps.
        time_fps: [ndarray] Array of time steps corresponding to the frames (observed time).
        luminosity_arr: [ndarray] Array of simulated luminosity values corresponding to `all_simulated_time`.
        dt: [float] Simulation time step (not actively used in simple mean calculation but kept for signature).
        fps: [float] Frames per second, defining the integration window (1/fps).
        P_0m: [float] Power of a zero-magnitude meteor (approx 840 W).

    Return:
        new_luminosity_arr: [ndarray] integrated/averaged luminosity at `time_fps` points.
        new_abs_magnitude: [ndarray] Calculated absolute magnitude from the new luminosity.

    """
    
    # make a copy of the luminosity_arr to make the new_luminosity_arr
    new_luminosity_arr = np.full(len(time_fps), np.nan) 
    new_abs_magnitude = np.full(len(time_fps), np.nan)
    # new_luminosity_arr = self.luminosity_arr
    for i in range(len(time_fps)):
        # pick all the self.time_arr between time_sampled_lum[i]-1/fps and time_sampled_lum[i] in self.time_arr
        mask = (all_simulated_time > time_fps[i]-1/fps) & (all_simulated_time <= time_fps[i])
        # sum them together and divide by 1/self.fps_lum
        # new_luminosity_arr[i] = (np.sum(luminosity_arr[mask])*dt)/abs(np.max(all_simulated_time[mask])-np.min(all_simulated_time[mask])) # 1/self.fps_lum
        # simply take the mean of the luminosity_arr[mask] and assign it to new_luminosity_arr[i]
        new_luminosity_arr[i] = np.mean(luminosity_arr[mask])
        new_abs_magnitude[i] = -2.5*np.log10(new_luminosity_arr[i]/P_0m)
    return new_luminosity_arr, new_abs_magnitude


class ObservationData:
    """ Class to load the observation data and create an object.

    Arguments:
        obs_file_path: [str or list] Path to the observation file(s) (pickle or json).

    Keyword arguments:
        use_all_cameras: [bool] Flag to use all cameras. False by default.
        lag_noise_prior: [float] Prior estimate for lag noise (in meters). 40 by default.
        lum_noise_prior: [float] Prior estimate for luminosity noise (in mag). 2.5 by default.
        fps_prior: [float] Prior override for FPS. np.nan by default.
        P_0m_prior: [float] Prior override for P_0m (power of zero magnitude, in Watts). np.nan by default.
        pick_position: [int] Index to pick specific position/station data. 0 by default.
        prior_file_path: [str] Path to a prior file. Empty string by default.

    """
    def __init__(self, obs_file_path, use_all_cameras=False, lag_noise_prior=40, lum_noise_prior=2.5, 
                 fps_prior=np.nan, P_0m_prior=np.nan, pick_position=0, prior_file_path=""):
        self.noise_lag = lag_noise_prior
        self.noise_lum = lum_noise_prior
        self.file_name = obs_file_path

        # check obs_file_path is a list if so take the first element
        if isinstance(obs_file_path, list):
            obs_file_path = obs_file_path[0]

        # check if the file is a json file
        if obs_file_path.endswith('.pickle'):
            self.loadPickleData(use_all_cameras,pick_position,prior_file_path, fps_prior, P_0m_prior)
        elif obs_file_path.endswith('.json'):
            self.loadJSONData(use_all_cameras)
        else:
            # file type not supported
            raise ValueError("File type not supported, only .json and .pickle files are supported")

    def saveToJSON(self, filepath="output.json"):
        """Save all attributes of self to a JSON file (numpy arrays converted to lists)."""

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj

        data = {k: convert(v) for k, v in self.__dict__.items()}
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved object state to {filepath}")

    def loadPickleData(self, use_all_cameras=False, pick_position=0, prior_file_path="", fps_prior=np.nan, 
                       P_0m_prior=np.nan):
        """ Load the pickle file(s) and create a dictionary keyed by each file name.

        Each file's data (e.g., list of station IDs, dens_co, zenith_angle, etc.) goes into a sub-dict.
        
        Keyword arguments:
            use_all_cameras: [bool] Flag to use all cameras (i.e. not filtering for specific types). False by default.
            pick_position: [int] Index to pick specific position/station data. 0 by default.
            prior_file_path: [str] Path to a prior file. Empty string by default.
            fps_prior: [float] Prior override for FPS. np.nan by default.
            P_0m_prior: [float] Prior override for P_0m (power of zero magnitude, in Watts). np.nan by default.

        """
        
        print('Loading pickle file(s):', self.file_name)
        
        # Top-level dictionary.
        combined_obs_dict = {}
        const = Constants()
        # check if it is not an array
        if not isinstance(self.file_name, list):
            self.file_name = [self.file_name]     

        obs_dict = []  # accumulates everything from all files   
        # Loop over each pickle file
        for current_file_name in self.file_name:
            traj = loadPickle(*os.path.split(current_file_name))

            # Skip if there's no .orbit attribute
            if not hasattr(traj, 'orbit'):
                print(f"Trajectory data not found in file: {current_file_name}")
                continue

            # self.v_init=traj.orbit.v_init+100
            # self.stations = []

            obs_data_dict = []
            for obs in traj.observations:
                # print('Station:', obs.station_id)

                # check that fps is not nan and P_0m is not nan
                if np.isnan(P_0m_prior):
                    # print('P_0m is not set, consider it as CAMO-narrowfield if 1 and 2 in camera station (i.e. output from .Met file)')
                    # check if the station_id is in the old format from .Met solution normally is CAMO narrow-field
                    if "1" == obs.station_id:
                        obs.station_id = obs.station_id.replace("1", "01T")
                    elif "2" == obs.station_id:
                        obs.station_id = obs.station_id.replace("2", "02T")
                    else:
                        P_0m = 840
                else:
                    P_0m = P_0m_prior

                # check if among obs.station_id there is one of the following 01T or 02T
                if "1T" in obs.station_id or "2T" in obs.station_id:
                    P_0m = 840
                elif "1K" in obs.station_id or "2K" in obs.station_id:
                    P_0m = 840
                elif "1G" in obs.station_id or "2G" in obs.station_id or "1F" in obs.station_id or "2F" in obs.station_id:
                    P_0m = 935
                # else:
                #     print(obs.station_id,'Station uknown\nMake sure the station is either EMCCD or CAMO (i.e. contains in the name 1T, 2T, 1K, 2K, 1G, 2G, 1F, 2F)')
                #     continue

                # check if obs.absolute_magnitudes is a 'NoneType' object
                if obs.absolute_magnitudes is None:
                    # create an array with the same length as obs.model_ht and fill it with 15
                    obs.absolute_magnitudes = np.array([15.5]*len(obs.model_ht))

                obs_data_camera = {
                    # make an array that is long as len(obs.model_ht) and has only obs.station_id
                    'flag_station': np.array([obs.station_id]*len(obs.model_ht)),
                    'flag_file': np.array([current_file_name]*len(obs.model_ht)),
                    'height': np.array(obs.model_ht), # m
                    'absolute_magnitudes': np.array(obs.absolute_magnitudes),
                    'luminosity': np.array(P_0m*(10 ** (obs.absolute_magnitudes/(-2.5)))), # const.P_0m)
                    'time': np.array(obs.time_data), # s
                    'ignore_list': np.array(obs.ignore_list),
                    'velocities': np.array(obs.velocities), # m/s
                    'lag': np.array(obs.lag), # m
                    'length': np.array(obs.state_vect_dist), # m
                    'time_lag': np.array(obs.time_data), # s
                    'height_lag': np.array(obs.model_ht), # m
                    'apparent_magnitudes': np.array(meteorAbsMagnitudeToApparent(np.array(obs.absolute_magnitudes), np.array(obs.meas_range))) # model_range
                    }
                obs_data_camera['velocities'][0] = obs.v_init
                obs_data_dict.append(obs_data_camera)
            
            obs_dict.extend(obs_data_dict)  # Add this file's data to the big list
            
        # ceck if obs_dict is empty
        if len(obs_dict) == 0:
            print('No valid station data found')
            return

        # Combine obs1 and obs2
        for key in obs_dict[0].keys():
            combined_obs_dict[key] = np.concatenate([obs[key] for obs in obs_dict])

        sorted_indices = np.argsort(combined_obs_dict['time'])
        for key in obs_dict[0].keys():
            combined_obs_dict[key] = combined_obs_dict[key][sorted_indices]

        # take all the unique values of the flag_station
        unique_stations = np.unique(combined_obs_dict['flag_station'])
        print('Unique stations:', unique_stations)

        # check if 1T 2T 1G 2G 1F 2F 1K 2K are present in unique_stations
        if not any(("1T" in station) or 
                ("2T" in station) or 
                ("1G" in station) or 
                ("2G" in station) or 
                ("1F" in station) or 
                ("2F" in station) or 
                ("1K" in station) or 
                ("2K" in station) 
                for station in unique_stations):
            print('Not EMCCD nor CAMO station found! Use all cameras since it is not known which one is best for luminosity or lag')
            use_all_cameras = True

        # check if among the unique_stations there is one of the following 01T or 02T
        if use_all_cameras==False:

            # check if among unique_stations there is one of the following 01G or 02G or 01F or 02F
            if any(("1G" in station) or ("2G" in station) or ("1F" in station) or ("2F" in station) for station in unique_stations):
                camera_name_lag = [camera for camera in unique_stations if "1G" in camera or "2G" in camera or "1F" in camera or "2F" in camera]
                lum_data, lum_files = self.extractLumData(combined_obs_dict, camera_name_lag)
                self.P_0m = 935
                self.fps_lum = 32
            elif any(("1K" in station) or ("2K" in station) or ("1T" in station) or ("2T" in station) for station in unique_stations):
                camera_name_lag = [camera for camera in unique_stations if "1K" in camera or "2K" in camera or "1T" in camera or "2T" in camera]
                lum_data, lum_files = self.extractLumData(combined_obs_dict, camera_name_lag)
                self.P_0m = 840
                self.fps_lum = 80
            else:
                # print the unique_stations
                print(unique_stations,'no known camera found')
                return

            # check if among unique_stations there is one of the following 01T or 02T  
            if any(("1T" in station) or ("2T" in station) for station in unique_stations):
                # find the name of the camera that has 1T or 2T
                camera_name_lag = [camera for camera in unique_stations if "1T" in camera or "2T" in camera]
                lag_data, lag_files = self.extractLagData(combined_obs_dict, camera_name_lag)
                self.fps_lag = 80
            elif any(("1G" in station) or ("2G" in station) or ("1F" in station) or ("2F" in station) for station in unique_stations):
                # find the name of the camera that has 1G or 2G or 1F or 2F
                camera_name_lag = [camera for camera in unique_stations if "1G" in camera or "2G" in camera or "1F" in camera or "2F" in camera]
                lag_data, lag_files = self.extractLagData(combined_obs_dict, camera_name_lag)
                self.fps_lag = 32
            elif any(("1K" in station) or ("2K" in station) for station in unique_stations):
                # find the name of the camera that has 1K or 2K
                camera_name_lag = [camera for camera in unique_stations if "1K" in camera or "2K" in camera]
                lag_data, lag_files = self.extractLagData(combined_obs_dict, camera_name_lag)
                self.fps_lag = 80
            else:
                # print the unique_stations
                print(unique_stations,'no known camera found')
                return
            
        else:
            lag_data = combined_obs_dict
            lum_data = combined_obs_dict
            lum_files = self.file_name
            lag_files = self.file_name

            # if it is a list of files consider a warning
            if len(lag_files) > 1:
                print('WARNING: Multiple files detected. Using all cameras for lag, the recorded data might have different starting time.')

            if any(("1G" in station) or ("2G" in station) or ("1F" in station) or ("2F" in station) for station in unique_stations):
                self.P_0m = 935
                self.fps_lum = 32
                self.fps_lag = 32
            elif any(("1K" in station) or ("2K" in station) or ("1T" in station) or ("2T" in station) for station in unique_stations):
                self.P_0m = 840
                self.fps_lum = 80
                self.fps_lag = 80
            else:
                # print(unique_stations,'no known camera found')
                # return
                if np.isnan(P_0m_prior):
                    print('P_0m is not set, consider it as P_0m=840')
                    self.P_0m = 840
                else:
                    self.P_0m = P_0m_prior
                if np.isnan(fps_prior):
                    # take for each of the station the average time in betwen the time and time lag with the same flag_station
                    for station in unique_stations:
                        indices = np.where(lum_data['flag_station'] == station)[0]
                        # Assign the average time lag to the station
                        time_diffs = np.diff(lum_data['time'][indices])
                        self.fps_lum = np.round(1/np.mean(time_diffs)) # if time_diffs.size > 0 or np.mean(time_diffs) != 0 else 32
                        self.fps_lag = np.round(1/np.mean(time_diffs)) # if time_diffs.size > 0 or np.mean(time_diffs) != 0 else 32
                    print('Computed fps_lag and fps_lum for each camera', np.round(1/np.mean(time_diffs)))
                else:
                    self.fps_lum = fps_prior
                    self.fps_lag = fps_prior

        # for the lum data delete all the keys that have values above 8
        if np.any(lum_data['absolute_magnitudes'] > 8):
            print(obs.station_id,'Found values below 8 absolute magnitudes :', lum_data['absolute_magnitudes'][lum_data['absolute_magnitudes'] > 8])
            # delete any values above 8 absolute_magnitudes and delete the corresponding values in the other arrays
            lum_data = {key: lum_data[key][lum_data['absolute_magnitudes'] < 8] for key in lum_data.keys()}

        # # print all the keys in the lag_data
        # print('Keys in lag_data:',lag_data.keys())
        # # print all the keys in the lum_data
        # print('Keys in lum_data:',lum_data.keys())
        # print(lum_files)
        # print(lag_files)
        
        self.lum_files = lum_files
        self.lag_files = lag_files

        v_init_list = []
        for curr_lag_file in lag_files:
            # take the v_init from the trajectory file
            traj=loadPickle(*os.path.split(curr_lag_file))
            # get the trajectory
            # v_avg = traj.v_avg
            v_init_list.append(traj.orbit.v_init+100)
        # do the mean of the v_init_list
        self.v_init = np.mean(v_init_list)

        # sort base on the lag_data['height'] from the biggest to the smallest
        sorted_indices_lag = np.argsort(lag_data['height'])[::-1]
        lag_data = {key: lag_data[key][sorted_indices_lag] for key in lag_data.keys()}

        # sort base on the lum_data['height'] from the biggest to the smallest
        sorted_indices_lum = np.argsort(lum_data['height'])[::-1]
        lum_data = {key: lum_data[key][sorted_indices_lum] for key in lum_data.keys()}

        if self.fps_lum > 80:
            offset_tol_use=0.2*(1/self.fps_lum)
        else:
            offset_tol_use=0.2*(1/80.0)
        lum_data, lum_aligned = buildGlobalTimeAxis(lum_data,offset_tol=offset_tol_use)
        if lum_aligned:
            print('Luminosity data aligned to global time axis.')

        if self.fps_lag > 80:
            offset_tol_use=0.2*(1/self.fps_lag)
        else:
            offset_tol_use=0.2*(1/80.0)
        lag_data, lag_aligned = buildGlobalTimeAxis(
            lag_data,
            offset_tol=offset_tol_use,
            v_init=self.v_init,          # important: recompute lag from length & time
            length_key="length",
            lag_key="lag",)
        if lag_aligned:
            print('Lag data aligned to global time axis.')

        # # Now these are safe for your integration windows:
        # assert np.all(np.diff(lum_data["time"]) >= 0)
        # assert np.all(np.diff(lag_data["time"]) >= 0)

        # put all the lag_data in the object
        self.velocities = lag_data['velocities']
        self.lag = lag_data['lag']
        self.length = lag_data['length']
        self.height_lag = lag_data['height']
        self.time_lag = lag_data['time']
        self.stations_lag = lag_data['flag_station']
        # put all the lum_data in the object
        self.height_lum = lum_data['height']
        self.absolute_magnitudes = lum_data['absolute_magnitudes']
        self.luminosity = lum_data['luminosity']
        self.time_lum = lum_data['time']
        self.stations_lum = lum_data['flag_station']
        self.apparent_magnitudes = lum_data['apparent_magnitudes']

        # adjusy the pick position to the leading edge of the luminosity if already leading edge useless
        unique_stations = np.unique(self.stations_lum)
        # for each station in unique_stations find the index of the first value of self.height_lum
        for station in unique_stations:
            # Get all indices where this station appears
            indices = np.where(self.stations_lum == station)[0]

            # Compute the differences between consecutive heights
            diffs = np.diff(self.height_lum[indices])
            
            # Prepend the first difference to match the array length
            diffheight = np.concatenate(([diffs[0]], diffs))
            
            # Subtract half of diffheight from the heights
            self.height_lum[indices] = self.height_lum[indices] - diffheight*pick_position

        # for lag measurements start from 0 for length and time
        self.length = self.length-self.length[0]
        self.time_lag = self.time_lag-np.min(self.time_lag)
        self.time_lum = self.time_lum-np.min(self.time_lum)

        # usually noise_lum = 2.5
        if np.isnan(self.noise_lum):
            self.noise_lum = self.fitLumNoise()
            print('Assumed Noise in luminosity based on SNR:',self.noise_lum)
        self.noise_mag = 0.1
        
        # usually noise_lag = 40 or 5 for CAMO
        if np.isnan(self.noise_lag):
            self.noise_lag = self.fitLagNoise()
            print('Assumed Noise in lag based on polynomial fit:',self.noise_lag)
        self.noise_vel = self.noise_lag*np.sqrt(2)/(1.0/self.fps_lag)

        zenith_angle_list = []
        m_init_list = []
        for curr_lum_file in lum_files:
            # take the m_init_list from the trajectory file
            traj=loadPickle(*os.path.split(curr_lum_file))

            # now find the zenith angle mass v_init and dens_co
            dens_fit_ht_beg = 180000
            dens_fit_ht_end = traj.rend_ele - 5000
            if dens_fit_ht_end < 14000:
                dens_fit_ht_end = 14000

            lat_mean = np.mean([traj.rbeg_lat, traj.rend_lat])
            lon_mean = meanAngle([traj.rbeg_lon, traj.rend_lon])
            jd_dat=traj.jdt_ref

            # Fit the polynomail describing the density
            self.dens_co = fitAtmPoly(lat_mean, lon_mean, dens_fit_ht_end, dens_fit_ht_beg, jd_dat)
            zenith_angle_list.append(zenithAngleAtSimulationBegin(const.h_init, traj.rbeg_ele, traj.orbit.zc, const.r_earth))
            time_mag_arr = []
            avg_t_diff_max = 0
            # take only the stations that are unique in the stations_lum
            lum_stations = np.unique(self.stations_lum)
            # Extract time vs. magnitudes from the trajectory pickle file
            for obs in traj.observations:
                # print('Station:', obs.station_id)

                if np.isnan(P_0m_prior):
                    # check if the station_id is in the old format from .Met solution
                    if "1" == obs.station_id:
                        obs.station_id = obs.station_id.replace("1", "01T")
                    elif "2" == obs.station_id:
                        obs.station_id = obs.station_id.replace("2", "02T")

                # check if the station_id is not in the lum_stations and continue
                if obs.station_id not in lum_stations:
                    continue

                # If there are not magnitudes for this site, skip it
                if obs.absolute_magnitudes is None:
                    continue

                # Compute average time difference
                avg_t_diff_max = max(avg_t_diff_max, np.median(obs.time_data[1:] - obs.time_data[:-1]))

                for t, mag in zip(obs.time_data[obs.ignore_list == 0], \
                    obs.absolute_magnitudes[obs.ignore_list == 0]):

                    if (mag is not None) and (not np.isnan(mag)) and (not np.isinf(mag)):
                        time_mag_arr.append([t, mag])

            

            # Sort array by time
            time_mag_arr = np.array(sorted(time_mag_arr, key=lambda x: x[0]))

            # check if not enough values to unpack (expected 2, got 0)
            if len(time_mag_arr) == 0:
                print('No valid luminosity data found')
                continue

            time_arr, mag_arr = time_mag_arr.T

            # Average out the magnitudes
            time_arr, mag_arr = mergeClosePoints(time_arr, mag_arr, avg_t_diff_max, method='avg')

            # # Calculate the radiated energy
            # radiated_energy = calcRadiatedEnergy(np.array(time_arr), np.array(mag_arr), P_0m=self.P_0m)
            lum_eff_type_val = None
            lum_eff_type_fixed = False
            luminous_efficiency = None
            if prior_file_path != "":

                with open(prior_file_path, 'r') as file:
                    for line in file:
                        stripped = line.strip().split('#')[0]  # Remove comments
                        if not stripped or ',' not in stripped:
                            continue

                        parts = [p.strip() for p in stripped.split(',')]

                        if parts[0] == 'lum_eff_type':
                            try:
                                lum_eff_type_val = int(parts[1])
                                lum_eff_type_fixed = 'fix' in parts[2:]
                            except (IndexError, ValueError):
                                continue

                        elif parts[0] == 'lum_eff':
                            try:
                                val1 = float(parts[1])
                                val2 = None

                                # Try parse second value if it exists and isn't 'fix'
                                if len(parts) > 2:
                                    second = parts[2].lower()
                                    if second != 'fix':
                                        val2 = float(second)
                                is_fixed = 'fix' in parts[2:]

                                if is_fixed:
                                    luminous_efficiency = val1
                                elif val2 is not None:
                                    luminous_efficiency = np.mean([val1, val2])
                                else:
                                    luminous_efficiency = val1  # fallback to first
                            except (IndexError, ValueError):
                                continue

                if not (lum_eff_type_val == 0 and lum_eff_type_fixed):
                    luminous_efficiency = 0.7  # fallback default

            else:
                luminous_efficiency = 0.7



            print("NOTE: The mass was computing using a constant luminous efficiency",luminous_efficiency,"%")

            # Compute the photometric mass
            photom_mass = calcMass(np.array(time_arr), np.array(mag_arr), traj.orbit.v_avg, tau=luminous_efficiency/100, P_0m=self.P_0m)

            # run in case we have a diffrent lum_eff_type than constant luminous efficiency to better estimate the correct mass range
            if lum_eff_type_val is not None and not (lum_eff_type_val == 0 and lum_eff_type_fixed):
                # Get the luminous efficiency
                tau_lum_eff_type = luminousEfficiency(lum_eff_type_val, luminous_efficiency, self.v_init, photom_mass)
                print("NOTE: Adjusting luminous efficiency for the given lum_eff_type new luminous efficiency =", tau_lum_eff_type)
                if tau_lum_eff_type <= 0:
                    print("WARNING: Computed non-physical luminous efficiency. Using constant value instead.")
                else:
                    # Recompute the photometric mass with the new luminous efficiency
                    photom_mass = calcMass(np.array(time_arr), np.array(mag_arr), traj.orbit.v_avg, tau=tau_lum_eff_type, P_0m=self.P_0m)

            m_init_list.append(photom_mass)

        # do the mean of the zenith_angle_list and m_init_list
        self.zenith_angle = np.mean(zenith_angle_list)
        self.m_init = np.mean(m_init_list)

        # ### DEBUG purposes ###
        # json_file_path = self.file_name[0]
        # # save the data to a json file base on self.obs_file_path but with .json extension for DEBUGGING purposes
        # if json_file_path.endswith('.pickle'):
        #     json_file_path = json_file_path.replace('.pickle', '.json')
        # self.saveToJSON(json_file_path)
                    
    def extractLagData(self, combined_obs_dict, camera_name_lag):
        """ Extract lag data from the combined observation dictionary. """

        lag_dict = []
        combined_lag_dict = {}
        lag_files = []
        # now for each of the camera_name save it in the lag_data
        for camera in camera_name_lag:
            lag_file=np.unique(combined_obs_dict['flag_file'][combined_obs_dict['flag_station'] == camera])[0]
            # the velocities, lag, height, time, flag_station and the unique flag_file
            lag_data = {
                'velocities': combined_obs_dict['velocities'][combined_obs_dict['flag_station'] == camera],
                'lag': combined_obs_dict['lag'][combined_obs_dict['flag_station'] == camera],
                'length': combined_obs_dict['length'][combined_obs_dict['flag_station'] == camera],
                'height': combined_obs_dict['height_lag'][combined_obs_dict['flag_station'] == camera],
                'time': combined_obs_dict['time_lag'][combined_obs_dict['flag_station'] == camera],
                'flag_station': combined_obs_dict['flag_station'][combined_obs_dict['flag_station'] == camera],
                'ignore_list': combined_obs_dict['ignore_list'][combined_obs_dict['flag_station'] == camera]
            }
            # Create a mask of all rows which have ignore_list == 0
            ignore_mask = (lag_data['ignore_list'] == 0)

            # Now rebuild lag_data so that each array is filtered by ignore_mask
            lag_data = {key: lag_data[key][ignore_mask] for key in lag_data.keys()}

            lag_dict.append(lag_data)
            lag_files.append(lag_file)

        # sort the indices by time in lag_dict
        for key in lag_dict[0].keys():
            combined_lag_dict[key] = np.concatenate([obs[key] for obs in lag_dict])

        sorted_indices = np.argsort(combined_lag_dict['time'])
        for key in lag_dict[0].keys():
            combined_lag_dict[key] = combined_lag_dict[key][sorted_indices]

        return combined_lag_dict, lag_files
    
    def extractLumData(self, combined_obs_dict, camera_name_lum):
        """ Extract luminosity data from the combined observation dictionary. """
        
        # consider that has to have height_lum absolute_magnitudes luminosity time_lum stations_lum apparent_magnitudes
        lum_dict = []
        combined_lum_dict = {}
        lum_files = []
        # now for each of the camera_name save it in the lum_data
        for camera in camera_name_lum:
            lum_file=np.unique(combined_obs_dict['flag_file'][combined_obs_dict['flag_station'] == camera])[0]
            # the velocities, lag, height, time, flag_station and the unique flag_file
            lum_data = {
                'height': combined_obs_dict['height'][combined_obs_dict['flag_station'] == camera],
                'absolute_magnitudes': combined_obs_dict['absolute_magnitudes'][combined_obs_dict['flag_station'] == camera],
                'luminosity': combined_obs_dict['luminosity'][combined_obs_dict['flag_station'] == camera],
                'time': combined_obs_dict['time'][combined_obs_dict['flag_station'] == camera],
                'flag_station': combined_obs_dict['flag_station'][combined_obs_dict['flag_station'] == camera],
                'apparent_magnitudes': combined_obs_dict['apparent_magnitudes'][combined_obs_dict['flag_station'] == camera]
            }
            # Extract heights
            heights = lum_data['height']
            # Compute differences between consecutive heights
            diffs = np.diff(heights)
            # Prepend the first difference to align array lengths
            diffheight = np.concatenate(([diffs[0]], diffs))
            # Compute middle heights
            lum_data['height_middle'] = heights - diffheight/2
            lum_dict.append(lum_data)
            lum_files.append(lum_file)
        # sort the indices by time in lum_dict
        for key in lum_dict[0].keys():
            combined_lum_dict[key] = np.concatenate([obs[key] for obs in lum_dict])

        sorted_indices = np.argsort(combined_lum_dict['time'])
        for key in lum_dict[0].keys():
            combined_lum_dict[key] = combined_lum_dict[key][sorted_indices]

        return combined_lum_dict, lum_files
        
    def fitLagNoise(self):
        ''' Define the lag fit noise '''
        # Fit a polynomial to the lag data

        def lagPoly(t, a, b, c, t0):
            """
            Quadratic lag function.
            """

            # Only take times <= t0
            t_before = t[t <= t0]

            # Only take times > t0
            t_after = t[t > t0]

            # Compute the lag linearly before t0
            l_before = np.zeros_like(t_before) # +c

            # Compute the lag quadratically after t0
            l_after = -abs(a)*(t_after - t0)**3 - abs(b)*(t_after - t0)**2

            c = 0

            lag_funct = np.concatenate((l_before, l_after))

            lag_funct = lag_funct - lag_funct[0]

            return lag_funct
        
        def lagPolyRes(params, t_time, l_data):
            """
            Residual function for the optimization.
            """

            return np.sum((l_data - lagPoly(t_time, *params))**2)

        # initial guess of deceleration decel equal to linear fit of velocity
        p0 = [np.mean(self.lag), 0, 0, np.mean(self.time_lag)]

        opt_res = minimize(lagPolyRes, p0, args=(np.array(self.time_lag), np.array(self.lag)), method='Nelder-Mead')

        # sample the fit for the velocity and acceleration
        a_t0, b_t0, c_t0, t0 = opt_res.x
        fitted_lag_t0 = lagPoly(self.time_lag, a_t0, b_t0, c_t0, t0)
        residuals_t0 = self.lag - fitted_lag_t0
        # avg_residual = np.mean(abs(residuals))
        rmsd_lag_polyn = np.sqrt(np.mean(residuals_t0**2))

        return rmsd_lag_polyn
    
    def fitLumNoise(self):
        ''' Define the SNR luminosity noise '''

        # Compute the SNR of the luminosity data
        shower_code = self.findIAUCode()
        # check if shower_code is None
        if shower_code is None:
            print("No IAU code found")
        else:
            print("Shower code:", shower_code)

        const = np.nan
        # if the last 3 letters of dir_pickle_files are DRA set const 8.0671
        if shower_code == 'DRA':
            const = 8.0671
        elif shower_code == 'CAP':
            const = 7.8009
        elif shower_code == 'ORI':
            const = 7.3346

        if np.isnan(const):
            # inverse polinomial fit
            velocities = np.array([20, 23, 66])*1000 # km/s
            offsets = np.array([8.0671, 7.8009, 7.3346]) # constant values
            
            log_velocities = np.log(velocities)
            log_offsets = np.log(offsets)

            b, log_a = np.polyfit(log_velocities, log_offsets, 1)
            a = np.exp(log_a)
        
            const = a*self.v_init**b

        apparent_mag = np.max(self.apparent_magnitudes)
        # find the index of the apparent magnitude in the list
        index = np.where(np.array(self.apparent_magnitudes) == apparent_mag)[0][0]

        lum_noise = self.luminosity[index]/10**((apparent_mag-const)/(-2.5))

        return lum_noise

    def findIAUCode(self):
        # check if self.file_name is a array or a string
        if not isinstance(self.file_name, str):
            # check if among the file_name there is one that contains _mir if so take it
            if any("_mir" in f for f in self.file_name):
                # take the first element of the array that contains _mir
                file_name_IAU = [f for f in self.file_name if "_mir" in f][0]
            else:
                # take the first element of the array
                file_name_IAU = self.file_name[0]
        else:
            file_name_IAU = self.file_name
        # Get the directory where self.file_name is stored
        file_dir = os.path.dirname(file_name_IAU)
        
        # Define the filenames to look for
        report_file = None
        for file_name in os.listdir(file_dir):
            if file_name.endswith("report.txt"):
                print("Found report.txt file to extract IAU code")
                report_file = file_name
                break
        if report_file is None:
            for file_name in os.listdir(file_dir):
                if file_name.endswith("report_sim.txt"):
                    print("Found report_sim.txt file to extract IAU code")
                    report_file = file_name
                    break
        
        # If no report file is found, return None
        if report_file is None:
            print("No report .txt file found in the directory")
            return None
        
        # Open and read the report file
        report_path = os.path.join(file_dir, report_file)
        with open(report_path, 'r', encoding='utf-8') as file:
            for line in file:
                match = re.search(r"IAU code =\s+(\S+)", line)
                if match:
                    return match.group(1)  # Extracted IAU code
        
        return None  # Return None if no match is found

    def _photometric_adjustment(self,unique_stations,peak_abs_mag_CAMO):
        
        print("NOTE: Applying photometric adjustment to mach the luminosity of the two stations")

        # Find indices of each station in self.stations_lum
        station_0_indices = np.where(self.stations_lum == unique_stations[0])[0]
        station_1_indices = np.where(self.stations_lum == unique_stations[1])[0]

        # Extract corresponding time arrays
        time_0 = self.time_lum[station_0_indices]
        time_1 = self.time_lum[station_1_indices]

        # Define a common time grid (union of both time arrays, sorted)
        common_time = np.linspace(max(time_0.min(), time_1.min()), min(time_0.max(), time_1.max()), num=100)

        # Interpolate absolute magnitudes onto the common time grid
        absolute_magnitudes_0 = np.interp(common_time, time_0, self.absolute_magnitudes[station_0_indices])
        absolute_magnitudes_1 = np.interp(common_time, time_1, self.absolute_magnitudes[station_1_indices])

        # Compute the mean
        avg_mag_0 = np.mean(absolute_magnitudes_0)
        avg_mag_1 = np.mean(absolute_magnitudes_1)
        # take the 4 smallest values from the absolute_magnitudes_0 and absolute_magnitudes_1
        small4_0 = np.mean(np.sort(absolute_magnitudes_0)[:4])
        small4_1 = np.mean(np.sort(absolute_magnitudes_1)[:4])
        # Compute the average difference between the two stations
        avg_diff = np.sqrt(np.mean((absolute_magnitudes_0 - absolute_magnitudes_1) ** 2))

        if peak_abs_mag_CAMO is not None:
            diff_0 = np.abs(peak_abs_mag_CAMO - small4_0)
            diff_1 = np.abs(peak_abs_mag_CAMO - small4_1)
            if diff_0 < diff_1 and avg_mag_0 < avg_mag_1:
                self.absolute_magnitudes[station_1_indices] -= avg_diff
                # Update luminosity for station_1
                self.luminosity[station_1_indices] = self.P_0m*(10 ** (self.absolute_magnitudes[station_1_indices]/(-2.5)))
            elif diff_0 < diff_1 and avg_mag_0 > avg_mag_1:
                self.absolute_magnitudes[station_1_indices] += avg_diff
                # Update luminosity for station_1
                self.luminosity[station_1_indices] = self.P_0m*(10 ** (self.absolute_magnitudes[station_1_indices]/(-2.5)))
            elif diff_0 > diff_1 and avg_mag_0 > avg_mag_1:
                self.absolute_magnitudes[station_0_indices] -= avg_diff
                # Update luminosity for station_0
                self.luminosity[station_0_indices] = self.P_0m*(10 ** (self.absolute_magnitudes[station_0_indices]/(-2.5)))
            elif diff_0 > diff_1 and avg_mag_0 < avg_mag_1:
                self.absolute_magnitudes[station_0_indices] += avg_diff
                # Update luminosity for station_0
                self.luminosity[station_0_indices] = self.P_0m*(10 ** (self.absolute_magnitudes[station_0_indices]/(-2.5)))
        else: 
            # Apply correction to station_0's absolute magnitudes
            if avg_mag_0 > avg_mag_1:
                self.absolute_magnitudes[station_0_indices] -= avg_diff
                # Update luminosity for station_0
                self.luminosity[station_0_indices] = self.P_0m*(10 ** (self.absolute_magnitudes[station_0_indices]/(-2.5)))
            else:
                self.absolute_magnitudes[station_1_indices] -= avg_diff
                # Update luminosity for station_1
                self.luminosity[station_1_indices] = self.P_0m*(10 ** (self.absolute_magnitudes[station_1_indices]/(-2.5)))



    def loadJSONData(self, use_all_cameras):

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

        print(f"Loading json file: {self.file_name}")

        # Read the JSON file
        with open(self.file_name, 'r') as f:
            data_dict = json.load(f)

        # check if data_dict has the key time_lag
        if 'time_lag' in data_dict.keys():

            # Convert lists back to numpy arrays where necessary
            def restore_data(obj):
                if isinstance(obj, dict):
                    return {k: restore_data(v) for k, v in obj.items()}

                elif isinstance(obj, list):
                    # If all items are numeric, convert to np.array of floats
                    if all(isinstance(i, (int, float)) for i in obj):
                        return np.array(obj)

                    # If all items are strings, convert to np.array of strings
                    if all(isinstance(i, str) for i in obj):
                        return np.array(obj, dtype=str)

                    # Otherwise, recurse in case it's a nested list
                    return [restore_data(v) for v in obj]

                else:
                    return obj

            restored_dict = restore_data(data_dict)
            self.__dict__.update(restored_dict)

            # check if self.fps exist for old version of the code
            if 'fps' in restored_dict.keys():
                if restored_dict['fps'] == 32:
                    self.fps_lum = 32
                    self.fps_lag = 32
                else:
                    self.fps_lum = 32
                    self.fps_lag = 80

        else:

            # Load the constants
            const, _ = loadConstants(self.file_name)
            const.dens_co = np.array(const.dens_co)

            # const_nominal.P_0m = 935

            # const.disruption_on = False

            const.lum_eff_type = 5

            if const.v_init < 30000:
                print('v_init < 30000 use 0.01 dt')
                const.dt = 0.01
                # const_nominal.erosion_bins_per_10mass = 5
            else:
                print('v_init > 30000 use 0.005 dt')
                const.dt = 0.005
                # const_nominal.erosion_bins_per_10mass = 10

            # Run the simulation
            frag_main, results_list, wake_results = runSimulation(const, compute_wake=False)
            simulation_MetSim_object = SimulationResults(const, frag_main, results_list, wake_results)

            # Store results in the object
            self.__dict__.update(simulation_MetSim_object.__dict__)

            # self.noise_lum = 2.5
            if np.isnan(self.noise_lum):
                self.noise_lum = 2.5
                print('Assumed default Noise in luminosity:',self.noise_lum)
            self.noise_mag = 0.1

            self.P_0m = self.const.P_0m
            self.fps_lum = 32
           
            if 1/self.fps_lum > self.const.dt:

                # integration time step lumionosity
                self.luminosity_arr, self.abs_magnitude = integrateLuminosity(self.time_arr, self.time_arr, 
                    self.luminosity_arr, self.const.dt, self.fps_lum, self.P_0m)

            # add a gausian noise to the luminosity of 2.5
            lum_obs_data = self.luminosity_arr + np.random.normal(loc=0, scale=self.noise_lum, size=len(self.luminosity_arr))

            # Identify indices where lum_obs_data > 0
            positive_indices = np.where(lum_obs_data > 0)[0]  # Get only valid indices

            # If no positive values exist, return empty list
            if len(positive_indices) == 0:
                indices_visible = []
                return
            else:
                # Find differences between consecutive indices
                diff = np.diff(positive_indices)

                # Identify breaks (where difference is more than 1)
                breaks = np.where(diff > 1)[0]

                # Split the indices into uninterrupted sequences
                sequences = np.split(positive_indices, breaks + 1)

                # Find the longest sequence
                indices_visible = max(sequences, key=len)

            # Store the constants
            self.v_init = self.const.v_init
            self.zenith_angle = self.const.zenith_angle
            self.m_init = self.const.m_init

            self.dens_co = np.array(self.const.dens_co) 

            # Compute absolute magnitudes
            absolute_magnitudes_check = -2.5*np.log10(lum_obs_data/self.P_0m)

            # Check if absolute_magnitudes_check exceeds 8
            if len(indices_visible) > 0:
                mask = absolute_magnitudes_check[indices_visible] > 8  # Only check relevant indices
                if np.any(mask):
                    print('Found values below 8 absolute magnitudes:', absolute_magnitudes_check[indices_visible][mask])

                    # Remove invalid indices
                    indices_visible = indices_visible[~mask]

                    # If gaps exist, extract longest continuous segment
                    indices_visible = np.sort(indices_visible)
                    diff = np.diff(indices_visible)
                    breaks = np.where(diff > 1)[0]
                    sequences = np.split(indices_visible, breaks + 1)
                    indices_visible = max(sequences, key=len) if sequences else []
            
            # extra filter that ensures no NaNs remain in the height array:
            finite_mask = np.isfinite(self.leading_frag_height_arr[indices_visible])
            indices_visible = indices_visible[finite_mask]

            # Select time, magnitude, height, and length above the visibility limit
            time_visible = self.time_arr[indices_visible]
            # the rest of the arrays are the same length as time_arr
            lum_visible = lum_obs_data[indices_visible]
            ht_visible   = self.leading_frag_height_arr[indices_visible]
            len_visible  = self.leading_frag_length_arr[indices_visible]
            vel_visible  = self.leading_frag_vel_arr[indices_visible]

            mag_visible  = self.abs_magnitude[indices_visible]
            maglim=8
            # check that all are below 8 mag_visible
            if np.any(mag_visible > maglim):
                print('Found values below',maglim,'absolute magnitudes:', mag_visible[mag_visible > maglim])
                # delete any values above 8 absolute_magnitudes and delete the corresponding values in the other arrays
                time_visible = time_visible[mag_visible < maglim]
                lum_visible = lum_visible[mag_visible < maglim]
                ht_visible = ht_visible[mag_visible < maglim]
                len_visible = len_visible[mag_visible < maglim]
                vel_visible = vel_visible[mag_visible < maglim]

            # Resample the time to the system FPS
            lum_interpol = scipy.interpolate.CubicSpline(time_visible, lum_visible)
            ht_interpol  = scipy.interpolate.CubicSpline(time_visible, ht_visible)
            len_interpol = scipy.interpolate.CubicSpline(time_visible, len_visible)
            vel_interpol = scipy.interpolate.CubicSpline(time_visible, vel_visible)

            fps_lum = 32
            if use_all_cameras == False:
                self.stations = ['01G','02G','01T','02T']
                self.fps_lag = 80
                # self.noise_lag = 5
                if np.isnan(self.noise_lag):
                    self.noise_lag = 5
                self.noise_vel = self.noise_lag*np.sqrt(2)/(1.0/self.fps_lag)
                # multiply by a number between 0.6 and 0.4 for the time to track for CAMO
                time_to_track = (time_visible[-1]-time_visible[0])*np.random.uniform(0.3,0.6)
                time_sampled_lag, stations_array_lag = self.mimicCameraFPS(time_visible, time_to_track, 
                    self.fps_lag, self.stations[2], self.stations[3])
                time_sampled_lum, stations_array_lum = self.mimicCameraFPS(time_visible, 0, fps_lum, 
                    self.stations[0], self.stations[1])
            else:
                self.stations = ['01F','02F']
                self.fps_lag = 32
                # self.noise_lag = 40
                if np.isnan(self.noise_lag):
                    self.noise_lag = 40
                self.noise_vel = self.noise_lag*np.sqrt(2)/(1.0/self.fps_lag)
                time_to_track = 0
                time_sampled_lag, stations_array_lag = self.mimicCameraFPS(time_visible, time_to_track, 
                    self.fps_lag, self.stations[0], self.stations[1])
                time_sampled_lum, stations_array_lum = time_sampled_lag, stations_array_lag

            # Create new mag, height and length arrays at FPS frequency
            self.stations_lum = stations_array_lum
            self.time_lum = time_sampled_lum - time_sampled_lum[0]
            self.height_lum = ht_interpol(time_sampled_lum)
            self.luminosity = lum_interpol(time_sampled_lum) # after the integration
            self.absolute_magnitudes = -2.5*np.log10(self.luminosity/self.P_0m) # P_0m*(10 ** (obs.absolute_magnitudes/(-2.5)))

            # mag_sampled = mag_interpol(time_sampled_lum)
            self.stations_lag = stations_array_lag
            self.height_lag = ht_interpol(time_sampled_lag)
            self.time_lag = time_sampled_lag - time_sampled_lag[0]

            # Find the index closest to the first height without NaNs
            index = np.argmin(np.abs(self.leading_frag_height_arr[~np.isnan(self.leading_frag_height_arr)]
                                      - self.height_lag[0]))

            # Compute the theoretical lag (without noise)
            lag_no_noise = (self.leading_frag_length_arr - self.leading_frag_length_arr[index]) - \
                           (self.v_init*(self.time_arr - self.time_arr[index]))
            lag_no_noise -= lag_no_noise[index]

            # Interpolate to align with observed height_lag
            self.lag = np.interp(self.height_lag, np.flip(self.leading_frag_height_arr), np.flip(lag_no_noise)) + np.random.normal(loc=0, scale=self.noise_lag, size=len(time_sampled_lag))

            self.length = len_interpol(time_sampled_lag) 
            self.length = self.length - self.length[0]
            self.length = self.length + np.random.normal(loc=0, scale=self.noise_lag, size=len(time_sampled_lag))

            # velocity noise
            self.velocities = vel_interpol(time_sampled_lag) + np.random.normal(loc=0, scale=self.noise_vel, size=len(time_sampled_lag))
            
            # Make const behave like a dict
            if hasattr(self, 'const'):
                self.const = self.const.__dict__

            self.new_json_file_save = self._save_json_data()


    def mimicCameraFPS(self, time_visible, time_to_track, fps, station1, station2):
        """ Mimic the camera FPS by sampling the time array.
        
        Arguments:
            time_visible: [ndarray] Array of time steps where the meteor is visible.
            time_to_track: [float] Duration of tracking time to skip at start (simulating tracking delay).
            fps: [float] Frames per second of the camera.
            station1: [str] Name of the first station.
            station2: [str] Name of the second station.

        Return:
            time_sampled: [ndarray] Array of time steps sampled at the camera FPS.
            stations: [ndarray] Array of station names corresponding to `time_sampled`.

        """

        # Sample the time according to the FPS from one camera
        time_sampled_cam1 = np.arange(np.min(time_visible)+time_to_track, np.max(time_visible), 1.0/fps)

        # Simulate sampling of the data from a second camera, with a random phase shift
        time_sampled_cam2 = time_sampled_cam1 + np.random.uniform(-1.0/fps, 1.0/fps)

        # Ensure second camera does not start before time_visible[0]
        time_sampled_cam2 = time_sampled_cam2[time_sampled_cam2 >= np.min(time_visible)]

        # The second camera will only capture 50 - 100% of the data, simulate this
        cam2_portion = np.random.uniform(0.5, 1.0)
        cam2_start = np.random.uniform(0, 1.0 - cam2_portion)
        cam2_start_index = int(cam2_start*len(time_sampled_cam2))
        cam2_end_index = int((cam2_start + cam2_portion)*len(time_sampled_cam2))

        # Cut the cam2 time to the portion of the data it will capture
        time_sampled_cam2 = time_sampled_cam2[cam2_start_index:cam2_end_index]

        # Cut the time array to the length of the visible data
        time_sampled_cam2 = time_sampled_cam2[(time_sampled_cam2 >= np.min(time_visible)) 
                                            & (time_sampled_cam2 <= np.max(time_visible))]

        # Combine the two camera time arrays
        time_sampled = np.sort(np.concatenate([time_sampled_cam1, time_sampled_cam2]))

        # # find the index of the time_sampled_cam1 in time_sampled
        # index_cam1 = np.searchsorted(time_sampled,time_sampled_cam1)
        # find the index of the time_sampled_cam2 in time_sampled
        index_cam2 = np.searchsorted(time_sampled,time_sampled_cam2)
        # create a array with self.station[-1] for the length of time_sampled
        stations = np.array([station1]*len(time_sampled))
        # replace the values of the index_cam1 with self.stations[0]
        stations[index_cam2] = station2
        
        return time_sampled,stations


    def _save_json_data(self):
        """Save the object to a JSON file."""

        # Deep copy to avoid modifying the original object
        json_self_save = copy.deepcopy(self)

        # Convert all numpy arrays in `self2` to lists
        def convertToSerializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convertToSerializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convertToSerializable(v) for v in obj]
            elif hasattr(obj, '__dict__'):  # Convert objects with __dict__
                return convertToSerializable(obj.__dict__)
            else:
                return obj  # Leave as is if it's already serializable

        serializable_dict = convertToSerializable(json_self_save.__dict__)

        # Define file path for saving
        json_file_save = os.path.splitext(self.file_name)[0] + "_with_noise.json"
        # check if the file exists if so give a _1, _2, _3, etc. at the end of the file name
        i_json = 1
        if os.path.exists(json_file_save):
            while os.path.exists(json_file_save):
                json_file_save = os.path.splitext(self.file_name)[0] + f"_{i_json}_with_noise.json"
                i_json += 1

        # update the file name
        self.file_name = json_file_save

        # Write to JSON file
        with open(json_file_save, 'w') as f:
            json.dump(serializable_dict, f, indent=4)

        print("Saved fit parameters with noise to:", json_file_save)

        return json_file_save



###############################################################################
# find dynestyfile and priors
###############################################################################

def setupDirAndRunDynesty(input_dir, output_dir='', prior='', resume=True, use_all_cameras=True, 
    only_plot=True, cores=None, pool_MPI=None, pick_position=0, extraprior_file='', save_backup=True):
    """ Create the output folder if it doesn't exist and run the Dynesty simulation.

    Arguments:
        input_dir: [str or list] Input directory path(s) or file path(s) containing observations.

    Keyword arguments:
        output_dir: [str] Path to the output directory. Empty string by default.
        prior: [str] Path to the prior file. Empty string by default.
        resume: [bool] Flag to resume a previous run. True by default.
        use_all_cameras: [bool] Flag to use all cameras. True by default.
        only_plot: [bool] Flag to only generate plots and skip the Dynesty run. True by default.
        cores: [int] Number of CPU cores to use. None by default (uses all available).
        pool_MPI: [object] MPI pool object for parallel execution. None by default.
        pick_position: [int] Index to pick specific position/station data. 0 by default.
        extraprior_file: [str] Path to an extra prior file. Empty string by default.
        save_backup: [bool] Flag to save a backup of the results. True by default.

    Return:
        None

    """

    if not DYNESTY_FOUND:
        print("Dynesty package not found. Install dynesty to use the Dynesty functions.")
        return

    # initlize cml_args
    class cml_args:
        pass

    cml_args.input_dir = input_dir
    cml_args.output_dir = output_dir
    cml_args.prior = prior
    cml_args.use_all_cameras = use_all_cameras
    cml_args.resume = resume
    cml_args.only_plot = only_plot
    cml_args.cores = cores
    cml_args.extraprior_file = extraprior_file

    # If no core count given, use all
    if cml_args.cores is None:
        cml_args.cores = multiprocessing.cpu_count()

    # If user specified a non-empty prior but the file doesn't exist, exit
    if cml_args.prior != "" and not os.path.isfile(cml_args.prior):
        print(f"File {cml_args.prior} not found.")
        print("Specify a valid .prior path or leave it empty.")
        sys.exit()

    # Handle comma-separated input paths
    if ',' in cml_args.input_dir:
        cml_args.input_dir = cml_args.input_dir.split(',')
        print('Number of input directories/files:', len(cml_args.input_dir))
    else:
        cml_args.input_dir = [cml_args.input_dir]

    # Process each input path
    for input_dirfile in cml_args.input_dir:
        print(f"Processing {input_dirfile} look for all files...")

        # Use the class to find .dynesty, load prior, and decide output folders
        finder = autoSetupDynestyFiles(
            input_dir_or_file=input_dirfile,
            prior_file=cml_args.prior,
            resume=cml_args.resume,
            output_dir=cml_args.output_dir,
            use_all_cameras=cml_args.use_all_cameras,
            pick_position=pick_position,
            extraprior_file=cml_args.extraprior_file
        )

        # check if finder is empty
        if not finder.base_names:
            print("No files found in the input directory.")
            continue

        # Each discovered or created .dynesty is in input_folder_file
        # with its matching prior info
        for i, (base_name, dynesty_info, prior_path, out_folder, report_txt) in enumerate(zip(
            finder.base_names,
            finder.input_folder_file,
            finder.priors,
            finder.output_folders,
            finder.report_txt)):

            dynesty_file, pickle_file, bounds, flags_dict, fixed_values = dynesty_info
            obs_data = finder.obsInstance(base_name)
            obs_data.file_name = pickle_file # update teh file name in the observation data object

            # update the log file with the error join out_folder,"log_"+base_name+".txt"
            log_file_path = os.path.join(out_folder, f"log_{base_name}.txt")

            dynesty_file = setupDynestyOutputDir(out_folder, obs_data, bounds, flags_dict, fixed_values, pickle_file, dynesty_file, prior_path, base_name, log_file_path, report_txt)
            
            ### set up obs_data const values to run same simultaions in run_simulation #################

            # if the real_event has an initial velocity lower than 30000 set "dt": 0.005 to "dt": 0.01
            if obs_data.v_init < 30000:
                obs_data.dt = 0.01
                # const_nominal.erosion_bins_per_10mass = 5
            else:
                obs_data.dt = 0.005
                # const_nominal.erosion_bins_per_10mass = 10

            obs_data.disruption_on = False

            obs_data.lum_eff_type = 5

            obs_data.h_kill = np.min([np.min(obs_data.height_lum),np.min(obs_data.height_lag)])-1000
            # check if the h_kill is smaller than 0
            if obs_data.h_kill < 0:
                obs_data.h_kill = 1
            # check if np.min(obs_data.velocity[-1]) is smaller than v_init-10000
            if np.min(obs_data.velocities) < obs_data.v_init-10000:
                obs_data.v_kill = obs_data.v_init-10000
            else:
                obs_data.v_kill = np.min(obs_data.velocities)-5000
            # check if the v_kill is smaller than 0
            if obs_data.v_kill < 0:
                obs_data.v_kill = 1

            ##################################################################################################
            if save_backup:
                print("NOTE: Saving backup of dynesty files restuls.")

            # Plot obs data vs json file if present
            plotJSONDataVsObs(obs_data, out_folder)

            obs_data_json_file = os.path.join(out_folder, f"obs_data_{base_name}.json")
            with open(obs_data_json_file, "w") as f:
                json.dump(vars(obs_data), f, indent=4, default=jsonDefault)


            if not cml_args.only_plot: 

                # start a timer to check how long it takes to run dynesty
                start_time = time.time()
                # Run dynesty
                try:
                    dynestyMainRun(dynesty_file, obs_data, bounds, flags_dict, fixed_values, cml_args.cores, output_folder=out_folder, file_name=base_name, log_file_path=log_file_path, pool_MPI=pool_MPI)
                    dsampler = dynesty.DynamicNestedSampler.restore(dynesty_file)
                    plotDynestyResults(dsampler.results, obs_data, flags_dict, fixed_values, out_folder, base_name,log_file_path,cml_args.cores,save_backup=save_backup)
                except Exception as e:
                    # Open the file in append mode and write the error message
                    with open(log_file_path, "a") as log_file:
                        log_file.write(f"\nError encountered in dynestsy run: {e}")
                    print(f"Error encountered in dynestsy run: {e}")
                    # now try and plot the dynesty file results
                    try:
                        dsampler = dynesty.DynamicNestedSampler.restore(dynesty_file)
                        plotDynestyResults(dsampler.results, obs_data, flags_dict, fixed_values, out_folder, base_name,log_file_path,cml_args.cores,save_backup=save_backup)
                    except Exception as e:
                        with open(log_file_path, "a") as log_file:
                            log_file.write(f"Error encountered in dynestsy plot: {e}")
                        print(f"Error encountered in dynestsy plot: {e}")
                        
                    # take only the name of the log file and the path
                    path_log_file, log_file_name = os.path.split(log_file_path)
                    # chenge the name log_file_name of the log_file_path to log_file_path_error adding error_ at the beginning
                    log_file_path_error = os.path.join(path_log_file, f"error_{log_file_name}")
                    # rename the log_file_path to log_file_path_error
                    os.rename(log_file_path, log_file_path_error)
                    print(f"Log file renamed to {log_file_path_error}")
                    log_file_path = log_file_path_error

                # Save the time it took to run dynesty
                end_time = time.time()
                elapsed_time = datetime.timedelta(seconds=end_time - start_time)

                # Add this time to the log file (use the correct log file path)
                with open(log_file_path, "a") as log_file:
                    log_file.write(f"\nTime to run dynesty: {elapsed_time}")

                # Print the time to run dynesty in hours, minutes, and seconds
                print(f"Time to run dynesty: {elapsed_time}")

            elif cml_args.only_plot and os.path.isfile(dynesty_file): 
                print("Only plotting requested. Skipping dynesty run.")
                dsampler = dynesty.DynamicNestedSampler.restore(dynesty_file)
                # dsampler = dynesty oad DynamicNestedSampler by restore(dynesty_file)
                plotDynestyResults(dsampler.results, obs_data, flags_dict, fixed_values, out_folder, base_name,log_file_path,cml_args.cores,save_backup=save_backup)

            else:
                print("Fail to generate dynesty plots, dynasty file not found:",dynesty_file)
                print("If you want to run the dynasty file set only_plot to False")


def setupDynestyOutputDir(out_folder, obs_data, bounds, flags_dict, fixed_values, pickle_files='', dynesty_file='', prior_path='', base_name='', log_file_path='', report_txt=''):
    """
    Create the output folder and set up the log file.
    """
    # check if the out_folder is empty and set it to the same folder as the dynesty_file
    if log_file_path == '':
        log_file_path = os.path.join(out_folder, f"log_{base_name}.txt")
        base_name_log = "log_{base_name}.txt"
    else:
        base_name_log = os.path.basename(log_file_path)

    print("--------------------------------------------------")
    # check if a file with the name "log"+n_PC_in_PCA+"_"+str(len(df_sel))+"ev.txt" already exist
    if os.path.exists(log_file_path):
        # clear the file
        with open(log_file_path, "w") as log_file:
            log_file.write("")
        # # remove the file
        # os.remove(log_file_path)
    sys.stdout = Logger(out_folder,base_name_log) # 
    print(f"Meteor:", base_name)
    print("  File name:    ", pickle_files)
    print("  Report file: ", report_txt)
    print("  Dynesty file: ", dynesty_file)
    print("  Prior file:   ", prior_path)
    print("  Output folder:", out_folder)
    print("  Bounds:")
    param_names = list(flags_dict.keys())
    for (low_val, high_val), param_name in zip(bounds, param_names):
        print(f"    {param_name}: [{low_val}, {high_val}] flags={flags_dict[param_name]}")
    print("  Fixed Values: ", fixed_values)
    # Close the Logger to ensure everything is written to the file STOP COPY in TXT file
    sys.stdout.close()
    # Reset sys.stdout to its original value if needed
    sys.stdout = sys.__stdout__
    print("--------------------------------------------------")

    # create output folder and put the image
    os.makedirs(out_folder, exist_ok=True)
    plotSimVsObsResiduals(obs_data, output_folder=out_folder, file_name=base_name)
    
    dynesty_file_in_output_path = os.path.join(out_folder,os.path.basename(dynesty_file))
    # copy the dynesty file to the output folder if not already there
    if not os.path.exists(dynesty_file_in_output_path) and os.path.isfile(dynesty_file):
        shutil.copy(dynesty_file, out_folder)
        print("dynesty file copied to output folder:", dynesty_file_in_output_path)

    if os.path.isfile(report_txt):
        # copy the report file to the output folder if not already there
        report_txt_in_output_path = os.path.join(out_folder,os.path.basename(report_txt))
        if not os.path.exists(report_txt_in_output_path):
            shutil.copy(report_txt, out_folder)
            print("report file copied to output folder:", report_txt_in_output_path)

    # chek if prior path is an array or a string
    if isinstance(prior_path, list):
        for prior_or_extra in prior_path:
            # add the prior pr
            if prior_or_extra != "":
                # check if there is a prior file with the same name in the output_folder
                prior_file_output = os.path.join(out_folder,os.path.basename(prior_or_extra))
                if not os.path.exists(prior_file_output):
                    shutil.copy(prior_or_extra, out_folder)
                    print("prior or extraprior file copied to output folder:", prior_file_output)
               
    else:
        # add the prior pr
        if prior_path != "":
            # check if there is a prior file with the same name in the output_folder
            prior_file_output = os.path.join(out_folder,os.path.basename(prior_path))
            if not os.path.exists(prior_file_output):
                shutil.copy(prior_path, out_folder)
                print("prior file copied to output folder:", prior_file_output)
            # # folder where is stored the pickle_file
            # prior_file_input = os.path.join(os.path.dirname(pickle_file),os.path.basename(prior_path))
            # if not os.path.exists(prior_file_input):
            #     shutil.copy(prior_path, os.path.dirname(pickle_file))
            #     print("prior file copied to input folder:", prior_file_input)

    # Dictionary to track count of each base name
    base_name_counter = {}

    for pickle_file in pickle_files:
        
        # Check that the file actually exists
        if not os.path.isfile(pickle_file):
            print("Original observation file not found, not copied:", pickle_file)
            continue
        
        # Extract the base filename
        base_name = os.path.basename(pickle_file)
        
        # Check if we've seen this filename before
        if base_name in base_name_counter:
            base_name_counter[base_name] += 1
            # Insert a suffix to differentiate
            root, ext = os.path.splitext(base_name)
            new_base_name = f"{root}_{base_name_counter[base_name]}{ext}"
        else:
            # First time seeing this base_name
            base_name_counter[base_name] = 0
            new_base_name = base_name

        # Compute the destination path
        dest_path = os.path.join(out_folder, new_base_name)

        # check if pickle_file and dest_path are the same
        if pickle_file != dest_path:
            # Copy the file
            shutil.copy(pickle_file, dest_path)
            print(f"Copied {pickle_file} to {dest_path}")

    return dynesty_file_in_output_path


def readSimParams(file_path, user_inputs=None, object_meteor=None):
    """
    check if present and read the prior file and return the bounds, flags, and fixed values.
    """

    # default values
    noise_lag_prior = np.nan
    noise_lum_prior = np.nan
    P_0m = np.nan
    fps = np.nan

    user_inputs = {} if user_inputs is None else dict(user_inputs)

    # Build optional eval context (so expressions like n_lag0 work)
    # NOTE: object_meteor is optional; if not given, h_* fall back to NaN.
    h_beg = np.nan
    h_end = np.nan
    h_peak = np.nan
    if object_meteor is not None and hasattr(object_meteor, "height_lum") and hasattr(object_meteor, "luminosity"):
        h_beg  = float(np.max(object_meteor.height_lum))
        h_end  = float(np.min(object_meteor.height_lum))
        h_peak = float(object_meteor.height_lum[np.argmax(object_meteor.luminosity)])

    # Allow aliases
    if "n_lag0" in user_inputs and "noise_lag" not in user_inputs:
        user_inputs["noise_lag"] = user_inputs["n_lag0"]
    if "n_lum0" in user_inputs and "noise_lum" not in user_inputs:
        user_inputs["noise_lum"] = user_inputs["n_lum0"]

    eval_ctx = {
        "h_beg": user_inputs.get("h_beg", h_beg),
        "h_end": user_inputs.get("h_end", h_end),
        "h_peak": user_inputs.get("h_peak", h_peak),

        "v_0": user_inputs.get("v_0", user_inputs.get("v_init", np.nan)),
        "m_0": user_inputs.get("m_0", user_inputs.get("m_init", np.nan)),
        "z_0": user_inputs.get("z_0", user_inputs.get("zenith_angle", np.nan)),

        "n_lag0": user_inputs.get("n_lag0", user_inputs.get("noise_lag", np.nan)),
        "n_lum0": user_inputs.get("n_lum0", user_inputs.get("noise_lum", np.nan)),
    }

    def safe_eval(value):
        """Evaluate numeric expressions safely; return np.nan if it can't be evaluated."""
        if not isinstance(value, str):
            return value
        v = value.strip()
        if v.lower() == "nan":
            return np.nan
        try:
            out = eval(v, {"__builtins__": {}, "np": np, **eval_ctx}, {})
            # ensure scalar numeric; otherwise treat as invalid here
            if isinstance(out, (int, float, np.number)):
                return float(out)
            return np.nan
        except Exception:
            return np.nan

    # Read .prior file, ignoring comment lines
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split('#')[0].strip().split(',')
            name = parts[0].strip()

            if name not in ["noise_lag", "noise_lum", "fps", "P_0m"]:
                continue

            # Handle fixed values
            if "fix" in parts:
                val = parts[1].strip() if len(parts) > 1 else "nan"
                val_fixed = safe_eval(val)
                if not np.isnan(val_fixed):
                    if name == "noise_lag":
                        noise_lag_prior = val_fixed
                    elif name == "noise_lum":
                        noise_lum_prior = val_fixed
                    elif name == "fps":
                        fps = val_fixed
                    elif name == "P_0m":
                        P_0m = val_fixed
                continue

            min_val = parts[1].strip() if len(parts) > 1 else "nan"
            max_val = parts[2].strip() if len(parts) > 2 else "nan"
            flags = [flag.strip() for flag in parts[3:]] if len(parts) > 3 else []

            min_val = safe_eval(min_val)
            max_val = safe_eval(max_val)

            if np.isnan(max_val):
                continue

            if ("norm" in flags or "invgamma" in flags):
                chosen = max_val
            else:
                chosen = np.mean([min_val, max_val]) if not np.isnan(min_val) else max_val

            if name == "noise_lag":
                noise_lag_prior = chosen
            elif name == "noise_lum":
                noise_lum_prior = chosen
            elif name == "fps":
                fps = chosen
            elif name == "P_0m":
                P_0m = chosen

    return noise_lag_prior, noise_lum_prior, fps, P_0m


class autoSetupDynestyFiles:
    """
    Automatically prepares .dynesty output filenames and associated prior configurations 
    for meteor observation data, supporting both single files and large directory trees 
    with multiple observations and multi-camera clusters.

    Key Features:
    1. Flexible Input Handling:
       - Accepts either a single file or a directory as input.
       - Recursively traverses directories to locate `.pickle` files containing trajectory data.
       - Groups files into meteor clusters based on time proximity (within 1 second) and camera station overlap.
       - Clusters are constructed with preference for files already containing all stations. Otherwise, unique subsets are used.

    2. Dynesty File and Output Management:
       - For each cluster or file, constructs a corresponding `.dynesty` output file.
       - If a `.dynesty` file exists:
         - It is reused if `resume=True`.
         - A new version with an incrementing suffix (_n1, _n2, ...) is created if `resume=False`.
       - Output directories are organized by meteor base name. If an `output_dir` is provided, a subfolder is created there; otherwise, results remain in the same folder.

    3. Prior Configuration:
       - If a valid `prior_file` is given, it is applied to all observations.
       - Otherwise, the code searches the input file's directory for a `.prior` file.
       - If no prior is found, default values are used.
       - Additional support for reading prior noise estimates (lag and luminosity) from `.prior` files.

    4. Multi-Camera and Observation Instance Management:
       - Supports merging multiple `.pickle` files from different cameras into a single observation instance.
       - For each meteor, stores:
         - `.dynesty` file path
         - Input pickle file(s)
         - Prior bounds and flags
         - Output folder
         - Path to associated `report.txt` (or `report_sim.txt`) for IAU identification
         - Fully initialized `ObservationData` instance, accessible via `obsInstance(base_name)`

    This class streamlines the processing of diverse meteor datasets by automatically grouping observations, assigning priors, and managing file I/O in a reproducible way, enabling large-scale nested sampling analysis on multi-camera meteor networks.
    """

    def __init__(self, input_dir_or_file, prior_file="", resume=False, output_dir="", use_all_cameras=False,
                 pick_position=0, extraprior_file=""):
        """ Initialize the autoSetupDynestyFiles class.

        Arguments:
            input_dir_or_file: [str] Path to the input directory or file.

        Keyword arguments:
            prior_file: [str] Path to the prior file. Empty string by default.
            resume: [bool] Flag to resume a previous run. False by default.
            output_dir: [str] Path to the output directory. Empty string by default.
            use_all_cameras: [bool] Flag to use all cameras. False by default.
            pick_position: [int] Index to pick specific position/station data. 0 by default.
            extraprior_file: [str] Path to an extra prior file. Empty string by default.

        """
        self.input_dir_or_file = input_dir_or_file
        self.prior_file = prior_file
        self.resume = resume
        self.output_dir = output_dir
        self.use_all_cameras = use_all_cameras
        self.pick_position = pick_position
        self.extraprior_file = extraprior_file  # to be filled if found

        # Prepare placeholders
        self.base_names = []        # [base_name, ...] (no extension)
        self.input_folder_file = [] # [(dynesty_file, input_file, bounds, flags_dict, fixed_values), ...]
        self.priors = []            # [used_prior_path_or_empty_string, ...]
        self.output_folders = []    # [output_folder_for_this_dynesty, ...]
        self.report_txt = []          # [report_txt_path, ...]
        self.observation_objects = {}  # {base_name: observation_instance}

        # Kick off processing
        self.processInput()

    def processInput(self):
        """Decide if input is file or directory, build .dynesty, figure out prior, and store results."""
        if os.path.isfile(self.input_dir_or_file):
            # Single file case
            input_file = self.input_dir_or_file
            root = os.path.dirname(input_file)
            # take all the files in the folder
            files = os.listdir(root)
            self.mainDirConstructor(input_file,root,files)

        else:

            all_pickle_files = []

            # Walk through all subdirectories and find pickle files
            for root, dirs, files in os.walk(self.input_dir_or_file):
                pickle_files = [f for f in files if f.endswith('.pickle')]
                if not pickle_files:
                    continue

                # Flatten list using extend (instead of appending a list inside a list)
                all_pickle_files.extend(os.path.join(root, f) for f in pickle_files)

            print(all_pickle_files)

            # Call function to process found pickle files
            clusters = self.combinePickleFiles(all_pickle_files)

            print(f"Found {len(clusters)} meteors in {self.input_dir_or_file}")
            for i, cluster_info in enumerate(clusters, start=1):
                # print(f"Cluster #{i}")
                print("meteor:", cluster_info['cluster_name'])
                # print("Filenames:", cluster_info['filenames'])
                print("Union stations:", cluster_info['union_stations'])
                # print("Time range:", cluster_info['jd_range'])
                # print("-----------------")

                # now check if only a single file is found in the cluster
                if len(cluster_info['filenames']) == 1:
                    input_file = cluster_info['filenames'][0]
                    root = os.path.dirname(input_file)
                    files = os.listdir(root)
                    self.mainDirConstructor(input_file,root,files)

                elif len(cluster_info['filenames']) > 1:
                    # Combine multiple files into a single observation instance
                    print("Multiple files found in the cluster. Combine them into a single observation instance.")
                    # chek if all cluster_info['filenames'] are in the same folder already
                    if all(os.path.dirname(cluster_info['filenames'][0]) == os.path.dirname(f) for f in cluster_info['filenames']):
                        root = os.path.dirname(cluster_info['filenames'][0])
                        files = os.listdir(root)
                        self.mainDirConstructor(cluster_info['filenames'],root,files,cluster_info['cluster_name'])
                    else:
                        # join self.input_dir_or_file and cluster_info['cluster_name'] to create a new folder
                        new_combined_input_folder = os.path.join(self.input_dir_or_file, cluster_info['cluster_name'])
                        # check if the folder exists
                        if os.path.exists(new_combined_input_folder):
                            root = new_combined_input_folder
                            files = os.listdir(root)
                            self.mainDirConstructor(cluster_info['filenames'],root,files,cluster_info['cluster_name'])
                        else:
                            self.mainDirConstructor(cluster_info['filenames'],new_combined_input_folder,[],cluster_info['cluster_name'])



    def combinePickleFiles(self, all_pickle_files, time_threshold=1/86400):
        """
        Group the given pickle files by time (within `time_threshold` in JD).
        
        For each group (cluster):
        1) Compute the union of all stations (cameras) in that cluster.
        2) If at least one file in the cluster already has ALL stations, 
            pick ONLY that/those file(s). 
            Otherwise, pick all unique station sets (removing exact duplicates).
        3) Return a list of dicts, each with:
            'filenames':       the .pickle files selected for that cluster
            'union_stations':  sorted list of all stations
            'jd_range':        (min_jd, max_jd)
            'cluster_name':    string like "YYYYMMDD_HHMMSS.sss" from avg(min_jd, max_jd)
        """
        data = []
        for fullpath in all_pickle_files:
            folder, fname = os.path.split(fullpath)

            ## more optional filtering of files
            # if '_sim.pickle' in fname:
            #     print(f"Skipping simulation file: {fullpath}")
            #     continue
            # if 'Monte Carlo' in fullpath:
            #     print(f"Skipping MonteCarlo file: {fullpath}")
            #     continue
            # if '_skyfit2' in fullpath:
            #     print(f"Skipping skyfit2 file: {fullpath}")
            #     continue

            try:
                traj = loadPickle(folder, fname)  # your existing load function
            except Exception as e:
                print(f"Cannot load pickle {fullpath}: {e}")
                continue
            
            if not hasattr(traj, 'orbit'):
                print(f"Trajectory data not found in {fullpath}")
                continue
            
            jdt_ref = getattr(traj, 'jdt_ref', None)
            if jdt_ref is None:
                print(f"No jdt_ref found in {fullpath}")
                continue
            
            station_ids = []
            for obs in getattr(traj, 'observations', []):
                station_ids.append(obs.station_id)
            
            data.append({
                'filename': fullpath,
                'jdt_ref': jdt_ref,
                'stations': frozenset(station_ids)
            })
        
        if not data:
            print("No valid trajectory data found.")
            return []

        # Sort by time
        df = pd.DataFrame(data).sort_values('jdt_ref').reset_index(drop=True)

        # ---------------------------------------------
        # Cluster by checking consecutive files' times
        # ---------------------------------------------
        clusters_raw = []
        current_cluster = [df.iloc[0]]

        for i in range(1, len(df)):
            curr_row = df.iloc[i]
            prev_row = current_cluster[-1]
            if abs(curr_row['jdt_ref'] - prev_row['jdt_ref']) <= time_threshold:
                current_cluster.append(curr_row)
            else:
                clusters_raw.append(current_cluster)
                current_cluster = [curr_row]

        if current_cluster:
            clusters_raw.append(current_cluster)

        # -------------------------------------------------
        # Build final clusters, check for "all cameras" file
        # -------------------------------------------------
        clusters_result = []
        for cluster_rows in clusters_raw:
            # 1) Compute the union of all stations in this cluster
            union_stations = set()
            jd_values = []
            for row in cluster_rows:
                union_stations |= row['stations']
                jd_values.append(row['jdt_ref'])

            min_jd = min(jd_values)
            max_jd = max(jd_values)
            cluster_time = []
            for jd_value in jd_values:
                # transform the jd_value to a datetime object
                timestamp = (jd_value - 2440587.5)*86400.0
                dt = datetime.datetime.utcfromtimestamp(timestamp)
                base_str = dt.strftime("%Y%m%d_%H%M%S")
                msec = dt.microsecond // 1000
                cluster_time.append(f"{base_str}.{msec:03d}")
            # put jd_values in 
            avg_jd = np.mean(jd_values)
            timestamp = (avg_jd - 2440587.5)*86400.0
            avg_dt = datetime.datetime.utcfromtimestamp(timestamp)

            base_str = avg_dt.strftime("%Y%m%d_%H%M%S")
            msec = avg_dt.microsecond // 1000
            # cluster_name = f"{base_str}-{msec:03d}_combined"
            cluster_name = f"{base_str}_combined"

            # 2) See if any file in cluster_rows has ALL stations
            #    i.e., row['stations'] == union_stations
            files_with_all = [r for r in cluster_rows if r['stations'] == union_stations]

            if files_with_all:
                # Keep only the FIRST file that has the entire station set
                # If you prefer the last, do [-1] instead
                chosen = files_with_all[0]  
                cluster_filenames = [chosen['filename']]
            else:
                # Otherwise, keep all unique station sets
                used_station_sets = set()
                cluster_filenames = []
                for row in cluster_rows:
                    if row['stations'] not in used_station_sets:
                        used_station_sets.add(row['stations'])
                        cluster_filenames.append(row['filename'])

            clusters_result.append({
                'cluster_name': cluster_name,
                'filenames': cluster_filenames,
                'union_stations': sorted(union_stations),
                'jd_range': (cluster_time)
            })

        return clusters_result


    def mainDirConstructor(self, input_file, root, files, base_name=""):
        """ Main function to return the observation instance """

        lag_noise_prior = np.nan
        lum_noise_prior = np.nan
        fps_prior = np.nan
        P_0m_prior = np.nan
        # If user gave a valid .prior path, read it once.
        if os.path.isfile(self.prior_file):
            prior_path_noise = self.prior_file
            lag_noise_prior, lum_noise_prior, fps_prior, P_0m_prior = readSimParams(self.prior_file)
        else:
            # Look for local .prior
            existing_prior_list = [f for f in files if f.endswith(".prior")]
            if existing_prior_list:
                prior_path_noise = os.path.join(root, existing_prior_list[0])
                lag_noise_prior, lum_noise_prior, fps_prior, P_0m_prior = readSimParams(prior_path_noise)

        if not (np.isnan(lag_noise_prior) or np.isnan(lum_noise_prior)):
            print("Found noise in prior file: lag",lag_noise_prior,"m, lum",lum_noise_prior,"J/s")

        # If user gave a valid .prior path, read it once.
        if os.path.isfile(self.prior_file):
            prior_path = self.prior_file
        else:
            # Look for local .prior
            existing_prior_list = [f for f in files if f.endswith(".prior")]
            if existing_prior_list:
                prior_path = os.path.join(root, existing_prior_list[0])
                # print the prior path has been found
                print(f"Take the first Prior file found in the same folder as the observation file: {prior_path}")
            else:
                # default
                prior_path = ""

        observation_instance = ObservationData(input_file, self.use_all_cameras, lag_noise_prior, 
            lum_noise_prior, fps_prior, P_0m_prior, self.pick_position, prior_path)

       # check if any camera was found if not return
        if not hasattr(observation_instance,'stations_lag'):
            print("No camera found in the observation file:", input_file)
            return

        # check if new_json_file_save is present in observation_instance
        if hasattr(observation_instance, 'new_json_file_save'):
            # change the input_file to the new_json_file_save
            input_file = observation_instance.new_json_file_save


        # check if input_file is a list
        if isinstance(input_file, list):
            # take the first file in the list
            file_name_no_ext = base_name
            input_files_save = input_file
            # check if among the file_name there is one that contains _mir if so take it
            if any("_mir" in f for f in input_file):
                # take the first element of the array that contains _mir
                file_name_IAU = [f for f in input_file if "_mir" in f][0]
            else:
                # take the first element of the array
                file_name_IAU = input_file[0]
        else:
            file_name_no_ext = os.path.splitext(input_file)[0]
            input_files_save = [input_file]
            file_name_IAU = input_file
        
        # Get the directory where self.file_name is stored
        file_dir_IAU = os.path.dirname(file_name_IAU)

        # Define the filenames to look for
        report_file = None
        for file_name in os.listdir(file_dir_IAU):
            if file_name.endswith("report.txt"):
                print("Found report.txt file to extract IAU code")
                report_file = file_name
                break
        if report_file is None:
            for file_name in os.listdir(file_dir_IAU):
                if file_name.endswith("report_sim.txt"):
                    print("Found report_sim.txt file to extract IAU code")
                    report_file = file_name
                    break
        # If no report file is found, return None
        if report_file is None:
            print("No report .txt file found in the directory")
            report_file = ''
        
        possible_dynesty = os.path.join(root, file_name_no_ext + ".dynesty")

        # Check for existing .dynesty in the same folder
        existing_dynesty_list = [f for f in files if f.endswith(".dynesty")]
        if existing_dynesty_list:
            # There is at least one .dynesty in this folder
            if os.path.exists(possible_dynesty) or (os.path.basename(possible_dynesty) in existing_dynesty_list):
                # Matches the .pickle base
                if self.resume:
                    dynesty_file = possible_dynesty
                else:
                    dynesty_file = self.constructNewDynestyName(possible_dynesty)
            else:
                dynesty_file = possible_dynesty
        else:
            # No .dynesty => create from .pickle base name
            dynesty_file = possible_dynesty

        # If user gave a valid .prior path, read it once.
        if os.path.isfile(self.prior_file):
            prior_path = self.prior_file
            bounds, flags_dict, fixed_values = loadPriorsAndGenerateBounds(observation_instance,self.prior_file)
        else:
            bounds, flags_dict, fixed_values = loadPriorsAndGenerateBounds(observation_instance,prior_path)

        # if given open the extra prior file and append the bounds, flags_dict, fixed_values
        if os.path.isfile(self.extraprior_file):
            bounds, flags_dict, fixed_values = appendExtraPriorsToBounds(observation_instance,bounds, flags_dict, fixed_values, file_path=self.extraprior_file)
            # if extraprior_file is find then add the path to the prior_path as an array of strings but since is a single file, it will need to be exteded
            prior_path = [prior_path, self.extraprior_file]
        else:
            # Look for local .extraprior
            existing_extraprior_list = [f for f in files if f.endswith(".extraprior")]
            if existing_extraprior_list:
                extraprior_path = os.path.join(root, existing_extraprior_list[0])
                # print the prior path has been found
                print(f"Take the first extraprior file found in the same folder as the observation file: {extraprior_path}")
                bounds, flags_dict, fixed_values = appendExtraPriorsToBounds(observation_instance,bounds, flags_dict, fixed_values, file_path=extraprior_path)
                prior_path = [prior_path, extraprior_path]

        if base_name == "":
            base_name = self.extractBaseName(input_file)
            
        if self.output_dir=="":
            # # if root do not exist create it
            # if not os.path.exists(root):
            #     os.makedirs(root)
            # Output folder is not specified
            output_folder = root
        else:
            # Output folder is specified
            output_folder = os.path.join(self.output_dir, base_name)
            # if not os.path.exists(output_folder):
            #     os.makedirs(output_folder)

        # Store results
        self.base_names.append(base_name)
        self.input_folder_file.append((dynesty_file, input_files_save, bounds, flags_dict, fixed_values))
        self.priors.append(prior_path)
        self.output_folders.append(output_folder)
        self.report_txt.append(os.path.join(file_dir_IAU, report_file))
        self.observation_objects[base_name] = observation_instance


    def obsInstance(self, base_name):
        """Return the observation instance corresponding to a specific base name."""
        return self.observation_objects.get(base_name, None)

    def extractBaseName(self, file_path):
        """ Extract the base name (timestamp) from a file path.

        Arguments:
            file_path: [str] Full path or filename of the file.

        Return:
            base_name: [str] Extracted base name (e.g. "YYYYMMDD_HHMMSS") or filename without extension.

        """
        filename = os.path.basename(file_path)  # Extract just the file name
        name_without_ext, _ = os.path.splitext(filename)  # Remove extension

        # Define regex pattern for YYYYMMDD_HHMMSS
        pattern = r"^(\d{8}_\d{6})"

        # Try to match the pattern
        match = re.match(pattern, name_without_ext)
        if match:
            return match.group(1)  # Return only the matched timestamp
        else:
            return name_without_ext  # Return the full name if no match

    def findPriorFile(self, folder):
        """Return the first .prior file found, or None if none."""
        for f in os.listdir(folder):
            if f.endswith(".prior"):
                return os.path.join(folder, f)
        return None

    def constructNewDynestyName(self, existing_dynesty_path):
        """ Generate a new unique filename to avoid overwriting an existing file.
        
        Appends _n1, _n2, etc. to the base name until no file collision occurs.

        Arguments:
            existing_dynesty_path: [str] The original path of the .dynesty file.

        Return:
            new_path: [str] A unique path that does not exist on the filesystem.

        """
        folder = os.path.dirname(existing_dynesty_path)
        base = os.path.splitext(os.path.basename(existing_dynesty_path))[0]
        ext = ".dynesty"

        counter = 1
        while True:
            new_name = f"{base}_n{counter}{ext}"
            new_path = os.path.join(folder, new_name)
            if not os.path.exists(new_path):
                return new_path
            counter += 1


###############################################################################
# Function: run simulations
###############################################################################

def constructConstants(parameter_guess, real_event, var_names, fix_var, dir_path="", file_name=""):
    """ Construct the constants object for the simulation from the input parameters.

    Arguments:
        parameter_guess: [list/ndarray] List of parameter values (sampled).
        real_event: [object] Object containing the real event data (e.g. initial velocity, density coeffs).
        var_names: [list] List of variable names corresponding to `parameter_guess`.
        fix_var: [dict] Dictionary of fixed variables.

    Keyword arguments:
        dir_path: [str] Directory path to save the constants. Empty string by default.
        file_name: [str] Filename to save the constants. Empty string by default.

    Return:
        const_nominal: [object] Constants object populated with the simulation parameters.

    """

    # Load the nominal simulation parameters
    const_nominal = Constants()
    const_nominal.dens_co = np.array(const_nominal.dens_co)

    dens_co=real_event.dens_co

    ### Calculate atmosphere density coeffs (down to the bottom observed height, limit to 15 km) ###

    # Assign the density coefficients
    const_nominal.dens_co = dens_co

    # # Turn on plotting of LCs of individual fragments 
    # const_nominal.fragmentation_show_individual_lcs = True

    # if the real_event has an initial velocity lower than 30000 set "dt": 0.005 to "dt": 0.01
    const_nominal.dt = real_event.dt
    # const_nominal.erosion_bins_per_10mass = 5

    const_nominal.P_0m = real_event.P_0m

    const_nominal.disruption_on = real_event.disruption_on

    const_nominal.lum_eff_type = real_event.lum_eff_type

    # Minimum height [m]
    const_nominal.h_kill = real_event.h_kill 

    # minim velocity [m/s]
    const_nominal.v_kill = real_event.v_kill
    
    # # Initial meteoroid height [m]
    # const_nominal.h_init = 180000

    # create a dictionary for var_names_frag and take the associated parameter_guess[i]
    var_dic_guess = {var_names: parameter_guess[i] for i, var_names in enumerate(var_names)}
    # print("var_dic_guess:", var_dic_guess)
    # regex for fragmentation variables
    fragmentation_regex = r'_([A-Z]{1,2})\d+'
    # from var_dic_guess take the variables that match the fragmentation_regex and create a new dictionary
    var_frag_dic = {var: var_dic_guess[var] for var in var_dic_guess if re.search(fragmentation_regex, var)}
    # print("var_frag_dic:", var_frag_dic)
    var_dic = {var: var_dic_guess[var] for var in var_dic_guess if var not in var_frag_dic}
    # print("var_dic:", var_dic)

    # Separate fragmentation fixed values from fix_var dictionary
    fix_var_frag_dic = {var: fix_var[var] for var in fix_var if re.search(fragmentation_regex, var)}
    # print("fix_var_frag_dic:", fix_var_frag_dic)
    fix_var = {var: fix_var[var] for var in fix_var if var not in fix_var_frag_dic}
    # print("fix_var:", fix_var)

    # # for loop for the var_cost that also give a number from 0 to the length of the var_cost
    # for i, var in enumerate(var_names):
    #     const_nominal.__dict__[var] = parameter_guess[i]

    var_guess_dic = list(var_dic.keys())
    # for loop for the fix_var that also give a number from 0 to the length of the fix_var
    for i, var in enumerate(var_guess_dic):
        const_nominal.__dict__[var] = var_dic[var]

    # first chack if fix_var is not {}
    if fix_var:
        var_names_fix = list(fix_var.keys())
        # for loop for the fix_var that also give a number from 0 to the length of the fix_var
        for i, var in enumerate(var_names_fix):
            const_nominal.__dict__[var] = fix_var[var]

    # fuse var_frag_dic and fix_var_frag_dic
    if var_frag_dic or fix_var_frag_dic:
        combined_frag_dic = {}
        combined_frag_dic.update(var_frag_dic)       # keep sampled variables (e.g., height_EF0)
        combined_frag_dic.update(fix_var_frag_dic)   # fixed values override if duplicated
        # use the function thaht add them in the const_nominal
        const_nominal = addFragToConst(const_nominal, combined_frag_dic)

    if dir_path!="" and file_name!="":
        _, _, _ = runSimulation(const_nominal, compute_wake=False) # completes the some fields in const_nominal that will be saved
        saveConstants(const_nominal, dir_path, file_name)

    return const_nominal

def runSimulationDynesty(parameter_guess, real_event, var_names, fix_var):
    """ Run the MetSim simulation with the given parameters.

    Arguments:
        parameter_guess: [list/ndarray] List of parameter values (sampled).
        real_event: [object] Object containing the real event data (e.g. observation data).
        var_names: [list] List of variable names corresponding to `parameter_guess`.
        fix_var: [dict] Dictionary of fixed variables.

    Return:
        simulation_MetSim_object: [object] SimulationResults object containing the simulation results.

    """

    # build the const to run the 
    const_nominal = constructConstants(parameter_guess, real_event, var_names, fix_var)

    try:
        # Run the simulation
        frag_main, results_list, wake_results = runSimulation(const_nominal, compute_wake=False)
        simulation_MetSim_object = SimulationResults(const_nominal, frag_main, results_list, wake_results)
    except ZeroDivisionError as e:
        print(f"Error during simulation: {e}")
        # run again with the nominal values to avoid the error
        const_nominal = Constants()
        # Run the simulation
        frag_main, results_list, wake_results = runSimulation(const_nominal, compute_wake=False)
        simulation_MetSim_object = SimulationResults(const_nominal, frag_main, results_list, wake_results)

    return simulation_MetSim_object

def addFragToConst(const_nominal, var_frag_dic):
    """ Add fragmentation variables from the dictionary into the constants object.

    Organizes variables by fragmentation type and index, creates FragmentationEntry objects,
    and appends them to the constants object.

    Arguments:
        const_nominal: [object] Constants object to be updated.
        var_frag_dic: [dict] Dictionary of fragmentation variables (e.g. {'height_M0': 90000, ...}).

    Return:
        const_nominal: [object] Updated Constants object.

    """

    frag_entries = {}

    # Group variables by fragment key (e.g., 'M0', 'EF1')
    for var_name, value in var_frag_dic.items():
        match = re.search(r'_([A-Z]{1,2})(\d+)$', var_name)
        if match:
            frag_type = match.group(1)  # 'M', 'A', 'EF', 'D', 'F'
            frag_index = int(match.group(2))
            frag_key = f"{frag_type}{frag_index}"

            if frag_key not in frag_entries:
                frag_entries[frag_key] = {"frag_type": frag_type}

            # Remove the suffix to get parameter name (e.g., height_M0 → height)
            match = re.match(r'^(.*)_([A-Z]{1,2}\d+)$', var_name)
            base_name = match.group(1)  # 'grain_mass_min'
            # frag_suffix = match.group(2)  # 'M0'
            # base_name = var_name[:var_name.rfind('_')] # wrong if there is _ in the base name
            frag_entries[frag_key][base_name] = value

    frag_entry_list = []

    for frag_key, frag_params in frag_entries.items():
        frag_type = frag_params["frag_type"]

        # Common parameters (initialize with None)
        height = frag_params.get("height", None)
        number = frag_params.get("number", None)
        mass_percent = frag_params.get("mass_percent", None)
        sigma = frag_params.get("sigma", None)
        gamma = frag_params.get("gamma", None)
        erosion_coeff = frag_params.get("erosion_coeff", None)
        grain_mass_min = frag_params.get("grain_mass_min", None)
        grain_mass_max = frag_params.get("grain_mass_max", None)
        mass_index = frag_params.get("mass_index", None)

        val = frag_params.get("mass_index", None)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            mass_index = const_nominal.erosion_mass_index
        else:
            mass_index = val
        val = frag_params.get("sigma", None)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            sigma = const_nominal.sigma
        else:
            sigma = val
        val = frag_params.get("gamma", None)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            gamma = const_nominal.gamma
        else:
            gamma = val
        val = frag_params.get("erosion_coeff", None)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            erosion_coeff = const_nominal.erosion_coeff
        else:
            erosion_coeff = val
        val = frag_params.get("grain_mass_min", None)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            grain_mass_min = const_nominal.erosion_mass_min
        else:
            grain_mass_min = val
        val = frag_params.get("grain_mass_max", None)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            grain_mass_max = const_nominal.erosion_mass_max
        else:
            grain_mass_max = val

        # check if number is not None
        if number is not None:
            # round it to the nearest integer
            number = int(round(number))

        # Now depending on frag_type, we validate required variables
        if frag_type == "M":
            # REQUIRED: height
            if height is None:
                raise ValueError(f"Missing height for Main Fragment {frag_key}")
            frag_entry = FragmentationEntry(frag_type="M",
                                            height=height,
                                            number=None,
                                            mass_percent=None,
                                            sigma=sigma,
                                            gamma=None,
                                            erosion_coeff=erosion_coeff,
                                            grain_mass_min=grain_mass_min,
                                            grain_mass_max=grain_mass_max,
                                            mass_index=mass_index)
            frag_entry_list.append(frag_entry)

        elif frag_type == "A":
            if height is None:
                raise ValueError(f"Missing height for All Fragments change {frag_key}")
            frag_entry = FragmentationEntry(frag_type="A",
                                            height=height,
                                            number=None,
                                            mass_percent=None,
                                            sigma=sigma,
                                            gamma=gamma,
                                            erosion_coeff=None,
                                            grain_mass_min=None,
                                            grain_mass_max=None,
                                            mass_index=None)
            frag_entry_list.append(frag_entry)

        elif frag_type == "F":
            if height is None or number is None or mass_percent is None:
                raise ValueError(f"Missing required parameters for Single Fragment {frag_key}")
            frag_entry = FragmentationEntry(frag_type="F",
                                            height=height,
                                            number=number,
                                            mass_percent=mass_percent,
                                            sigma=sigma,
                                            gamma=None,
                                            erosion_coeff=None,
                                            grain_mass_min=None,
                                            grain_mass_max=None,
                                            mass_index=None)
            frag_entry_list.append(frag_entry)

        elif frag_type == "EF":
            if height is None or number is None or mass_percent is None or erosion_coeff is None or grain_mass_min is None or grain_mass_max is None or mass_index is None:
                raise ValueError(f"Missing required parameters for Eroding Fragment {frag_key}")
            frag_entry = FragmentationEntry(frag_type="EF",
                                            height=height,
                                            number=number,
                                            mass_percent=mass_percent,
                                            sigma=sigma,
                                            gamma=None,
                                            erosion_coeff=erosion_coeff,
                                            grain_mass_min=grain_mass_min,
                                            grain_mass_max=grain_mass_max,
                                            mass_index=mass_index)
            frag_entry_list.append(frag_entry)

        elif frag_type == "D":
            if height is None or mass_percent is None or grain_mass_min is None or grain_mass_max is None:
                raise ValueError(f"Missing required parameters for Dust Release {frag_key}")
            frag_entry = FragmentationEntry(frag_type="D",
                                            height=height,
                                            number=None,
                                            mass_percent=mass_percent,
                                            sigma=None,
                                            gamma=None,
                                            erosion_coeff=None,
                                            grain_mass_min=grain_mass_min,
                                            grain_mass_max=grain_mass_max,
                                            mass_index=mass_index)
            frag_entry_list.append(frag_entry)

        else:
            raise ValueError(f"Unknown fragmentation type: {frag_type}")

    # Finalize into const_nominal
    const_nominal.fragmentation_on = True
    const_nominal.fragmentation_entries = frag_entry_list

    return const_nominal


###############################################################################
# Function: dynesty
###############################################################################

def logLikelihoodDynesty(guess_var, obs_metsim_obj, flags_dict, fix_var, timeout=20):
    """ Calculate the log-likelihood for Dynesty.

    Arguments:
        guess_var: [list/ndarray] List of parameter values (guess).
        obs_metsim_obj: [object] Observation object containing observed data.
        flags_dict: [dict] Dictionary of flags defining variable types and transformations.
        fix_var: [dict] Dictionary of fixed variables.

    Keyword arguments:
        timeout: [int] Timeout in seconds for the simulation. 20 by default.

    Return:
        log_likelihood: [float] Calculated log-likelihood (or -np.inf if invalid/timeout).

    """

    var_names = list(flags_dict.keys())
    # check for each var_name in flags_dict if there is "log" in the flags_dict
    for i, var_name in enumerate(var_names):
        if 'log' in flags_dict[var_name]:
            guess_var[i] = 10 ** guess_var[i]
        if var_name == 'noise_lag':
            obs_metsim_obj.noise_lag = guess_var[i]
        if var_name == 'noise_lum':
            obs_metsim_obj.noise_lum = guess_var[i]

    # check if among the var_names there is a "erosion_mass_max" and if there is a "erosion_mass_min"
    if 'erosion_mass_max' in var_names and 'erosion_mass_min' in var_names:
        # check if the guess_var of the erosion_mass_max is smaller than the guess_var of the erosion_mass_min
        if guess_var[var_names.index('erosion_mass_max')] < guess_var[var_names.index('erosion_mass_min')]:
            return -np.inf  # immediately return -np.inf if times out

    # check if among the var_names there is a "erosion_mass_max" and if there is a "erosion_mass_min"
    if 'disruption_mass_max_ratio' in var_names and 'disruption_mass_min_ratio' in var_names:
        # check if the guess_var of the erosion_mass_max is smaller than the guess_var of the erosion_mass_min
        if guess_var[var_names.index('disruption_mass_max_ratio')] < guess_var[var_names.index('disruption_mass_min_ratio')]:
            return -np.inf  # immediately return -np.inf if times out
    
    if 'erosion_mass_max' in var_names and 'm_init' in var_names:
        # check if the guess_var of the erosion_mass_max is smaller than the guess_var of the m_init
        if guess_var[var_names.index('erosion_mass_max')] > guess_var[var_names.index('m_init')]:
            return -np.inf

    if 'erosion_height_start' in var_names and 'erosion_height_change' in var_names:
        # check if the guess_var of the erosion_height_start is smaller than the guess_var of the erosion_height_change
        if guess_var[var_names.index('erosion_height_change')] > guess_var[var_names.index('erosion_height_start')]:
            return -np.inf
        
    if 'erosion_rho_change' in var_names and 'rho' in var_names:
        # check if the guess_var of the erosion_height_start is smaller than the guess_var of the erosion_height_change
        if guess_var[var_names.index('rho')] > guess_var[var_names.index('erosion_rho_change')]:
            return -np.inf

    ### ONLY on LINUX ###

    # check if the OS is not Linux
    if os.name != 'posix':
        # If not Linux, run the simulation without timeout
        simulation_results = runSimulationDynesty(guess_var, obs_metsim_obj, var_names, fix_var)
    else:
        # Set timeout handler
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)  # Start the timer for timeout
        # get simulated LC intensity onthe object
        try: # try to run the simulation
            simulation_results = runSimulationDynesty(guess_var, obs_metsim_obj, var_names, fix_var)
        except TimeoutException:
            print('timeout')
            return -np.inf  # immediately return -np.inf if times out
        finally:
            signal.alarm(0)  # Cancel alarm
        
    ### LUM CALC ###

    simulated_time = np.interp(obs_metsim_obj.height_lum, 
                                       np.flip(simulation_results.leading_frag_height_arr), 
                                       np.flip(simulation_results.time_arr))
    # check if the length of the lag_sim is the same as the length of the obs_metsim_obj.lag
    if np.sum(~np.isnan(simulated_time)) != np.sum(~np.isnan(obs_metsim_obj.time_lum)):
        return -np.inf

    # all simulated time is the time_arr subtract the first time of the simulation
    all_simulated_time = simulation_results.time_arr- simulated_time[0]#
    # find the integral of the luminosity in time in between FPS but not valid for CAMO narrowfield cameras as there is no smearing becuse it follows the meteor
    if (1/obs_metsim_obj.fps_lum > simulation_results.const.dt): # and (not any('1T' in station for station in obs_metsim_obj.stations_lum) or not any('2T' in station for station in obs_metsim_obj.stations_lum)): # FPS is lower than the simulation time step need to integrate the luminosity
        simulated_lc_intensity, _ = integrateLuminosity(all_simulated_time,obs_metsim_obj.time_lum,simulation_results.luminosity_arr,simulation_results.const.dt,obs_metsim_obj.fps_lum,obs_metsim_obj.P_0m)
    else:
        # too high frame rate, just interpolate the luminosity
        simulated_lc_intensity = np.interp(obs_metsim_obj.height_lum, 
                                           np.flip(simulation_results.leading_frag_height_arr), 
                                           np.flip(simulation_results.luminosity_arr))
        # check if the length of the simulated_lc_intensity is the same as the length of the obs_metsim_obj.luminosity
        if np.sum(~np.isnan(simulated_lc_intensity)) != np.sum(~np.isnan(obs_metsim_obj.luminosity)):
            return -np.inf

    ### LAG CALC ###

    lag_sim = simulation_results.leading_frag_length_arr - (obs_metsim_obj.v_init*simulation_results.time_arr)

    simulated_lag = np.interp(obs_metsim_obj.height_lag, 
                              np.flip(simulation_results.leading_frag_height_arr), 
                              np.flip(lag_sim))

    lag_sim = simulated_lag - simulated_lag[0]

    # check if the length of the lag_sim is the same as the length of the obs_metsim_obj.lag
    if np.sum(~np.isnan(lag_sim)) != np.sum(~np.isnan(obs_metsim_obj.lag)):
        return -np.inf

    # # create a plot with the obs_metsim_obj.luminosity and simulated_lc_intensity
    # plt.figure(figsize=(12,5))
    # plt.subplot(1,2,1)
    # plt.title("Luminosity")
    # plt.plot(obs_metsim_obj.height_lum, obs_metsim_obj.luminosity, 'o', label='Observed', markersize=4)
    # plt.plot(obs_metsim_obj.height_lum, simulated_lc_intensity, '-', label='Interpolated')
    # plt.plot(obs_metsim_obj.height_lum, np.interp(obs_metsim_obj.height_lum,np.flip(simulation_results.leading_frag_height_arr),np.flip(simulation_results.luminosity_arr)), 'r--', label='Not Integrated', markersize=4)
    # plt.xlabel("Height (m)")
    # plt.ylabel("Luminosity (J/s)")
    # plt.legend()
    # plt.subplot(1,2,2)
    # plt.title("Lag")
    # plt.plot(obs_metsim_obj.height_lag, obs_metsim_obj.lag, 'o', label='Observed', markersize=4)
    # plt.plot(obs_metsim_obj.height_lag, lag_sim, '-', label='Interpolated')
    # plt.xlabel("Height (m)")
    # plt.ylabel("Lag (m)")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    ### Log Likelihood ###

    log_likelihood_lum = np.nansum(-0.5*np.log(2*np.pi*obs_metsim_obj.noise_lum**2) - 0.5/(obs_metsim_obj.noise_lum**2)*(obs_metsim_obj.luminosity - simulated_lc_intensity) ** 2)
    # print("log_likelihood_lum:", log_likelihood_lum)
    log_likelihood_lag = np.nansum(-0.5*np.log(2*np.pi*obs_metsim_obj.noise_lag**2) - 0.5/(obs_metsim_obj.noise_lag**2)*(obs_metsim_obj.lag - lag_sim) ** 2)
    # print("log_likelihood_lag:", log_likelihood_lag)

    log_likelihood_tot = log_likelihood_lum + log_likelihood_lag

    ### Chi Square ###

    # chi_square_lum = - 0.5/(obs_metsim_obj.noise_lum**2)*np.nansum((obs_metsim_obj.luminosity_arr - simulated_lc_intensity) ** 2)  # add the error
    # chi_square_lag = - 0.5/(obs_metsim_obj.noise_lag**2)*np.nansum((obs_metsim_obj.lag - lag_sim) ** 2)  # add the error

    # log_likelihood_tot = chi_square_lum + chi_square_lag

    return log_likelihood_tot


def priorDynesty(cube, bounds, flags_dict):
    """ Transform the unit cube samples to the prior distribution.

    Arguments:
        cube: [list/ndarray] Unit hypercube samples (values between 0 and 1).
        bounds: [list] List of tuples defining bounds or parameters (min/max or sigma/mean) for each variable.
        flags_dict: [dict] Dictionary of flags specifying the distribution type (e.g., 'norm', 'invgamma') for each variable.

    Return:
        x: [ndarray] Transformed parameter values in the prior space.

    """
    x = np.array(cube)  # Copy u to avoid modifying it directly
    param_names = list(flags_dict.keys())
    i_prior=0
    for (min_or_sigma, MAX_or_mean), param_name in zip(bounds, param_names):
        # check if the flags_dict at index i is empty
        if 'norm' in flags_dict[param_name]:
            x[i_prior] = norm.ppf(cube[i_prior], loc=MAX_or_mean, scale=min_or_sigma)
        elif 'invgamma' in flags_dict[param_name]:
            x[i_prior] = invgamma.ppf(cube[i_prior], min_or_sigma, scale=MAX_or_mean*(min_or_sigma + 1))
        else:
            x[i_prior] = cube[i_prior]*(MAX_or_mean - min_or_sigma) + min_or_sigma  # Scale and shift
        i_prior += 1

    return x


def dynestyMainRun(dynesty_file, obs_data, bounds, flags_dict, fixed_values, n_core=1, output_folder="", 
                    file_name="",log_file_path="", pool_MPI=None):
    """ Main function to run the Dynesty nested sampling.

    Arguments:
        dynesty_file: [str] Path to the Dynesty checkpoint file.
        obs_data: [object] Observation object containing observed data.
        bounds: [list] List of tuples defining bounds or parameters for each variable.
        flags_dict: [dict] Dictionary of flags specifying the distribution type for each variable.
        fixed_values: [dict] Dictionary of fixed variables.

    Keyword arguments:
        n_core: [int] Number of CPU cores to use. 1 by default.
        output_folder: [str] Path to the output directory. Empty string by default.
        file_name: [str] Base name of the file (used for logging/naming). Empty string by default.
        log_file_path: [str] Full path to the log file. Empty string by default.
        pool_MPI: [object] MPI pool object for parallel execution. None by default.

    Return:
        None

    """

    print("Starting dynesty run...")  
    # get variable names
    var_names = list(flags_dict.keys())
    # get the number of parameters
    ndim = len(var_names)
    print("Number of parameters:", ndim)

    # first chack if fix_var is not {}
    if fixed_values:
        var_names_fix = list(fixed_values.keys())
        # check if among the noise_lum and noise_lag there is a "noise_lum" and if there is a "noise_lag"
        if 'noise_lum' in var_names_fix:
            # if so, set the noise_lum to the fixed value
            obs_data.noise_lum = fixed_values['noise_lum']
            print("Fixed noise in luminosity to:", fixed_values['noise_lum'])
        if 'noise_lag' in var_names_fix:
            # if so, set the noise_lag to the fixed value
            obs_data.noise_lag = fixed_values['noise_lag']
            print("Fixed noise in lag to:", fixed_values['noise_lag'])
    
    # Master-only: do any setup that requires file I/O
    if (pool_MPI is None) or (pool_MPI.is_master()):
        # e.g. check if dynesty_file exists, remove old logs, etc.
        pass

    # If using MPI, set up the dynamic sampler with the pool
    if pool_MPI is not None:
        print("Using MPI for parallelization of multiple nodes")
        # =============== MPI branch ==================

        # check if file exists
        if not os.path.exists(dynesty_file):
            print("Starting new run:")
            ### NEW RUN

            # Provide logl_args and ptform_args with a single argument
            dsampler = dynesty.DynamicNestedSampler(logLikelihoodDynesty, priorDynesty, ndim,
                                                    logl_args=(obs_data, flags_dict, fixed_values, 20),
                                                    ptform_args=(bounds, flags_dict),
                                                    sample='rslice', # nlive=1000,
                                                    pool = pool_MPI)
            dsampler.run_nested(print_progress=True, checkpoint_file=dynesty_file)
            # dlogz_init=0.001,
        else:
            print("Resuming previous run:")
            print('Warning: make sure the number of parameters and the bounds are the same as the previous run!')

            ### RESUME:
            dsampler = dynesty.DynamicNestedSampler.restore(dynesty_file,
                                                            pool = pool_MPI)
            dsampler.run_nested(resume=True, print_progress=True, checkpoint_file=dynesty_file)
                # dlogz_init=0.001,
               

    else:
        # =========== Normal multiprocessing ========== 

        # check if file exists
        if not os.path.exists(dynesty_file):
            print("Starting new run:")
            # Start new run
            with dynesty.pool.Pool(n_core, logLikelihoodDynesty, priorDynesty,
                                logl_args=(obs_data, flags_dict, fixed_values, 20),
                                ptform_args=(bounds, flags_dict)) as pool:
                ### NEW RUN
                dsampler = dynesty.DynamicNestedSampler(pool.loglike, 
                                                        pool.prior_transform, ndim,
                                                        sample='rslice', # nlive=1000,
                                                        pool = pool)
                dsampler.run_nested(print_progress=True, checkpoint_file=dynesty_file) #  dlogz_init=0.001,

        else:
            print("Resuming previous run:")
            print('Warning: make sure the number of parameters and the bounds are the same as the previous run!')
            # Resume previous run
            with dynesty.pool.Pool(n_core, logLikelihoodDynesty, priorDynesty,
                                logl_args=(obs_data, flags_dict, fixed_values, 20),
                                ptform_args=(bounds, flags_dict)) as pool:
                ### RESUME:
                dsampler = dynesty.DynamicNestedSampler.restore(dynesty_file,
                                                                pool = pool)
                dsampler.run_nested(resume=True, print_progress=True, checkpoint_file=dynesty_file) # dlogz_init=0.001,

    print('SUCCESS: dynesty results ready!\n')

    # check if output_folder is different from the dynesty_file folder
    if output_folder != os.path.dirname(dynesty_file):
        print("Copying dynesty file to output folder...")
        shutil.copy(dynesty_file, output_folder)
        print("dynesty file copied to:", output_folder)







###############################################################################
if __name__ == "__main__":

    import argparse

    ### COMMAND LINE ARGUMENTS
    arg_parser = argparse.ArgumentParser(description="Run dynesty with optional .prior file.")
    
    arg_parser.add_argument('input_dir', metavar='INPUT_PATH', type=str,
        help="Path to walk and find .pickle file or specific single file .pickle or .json file."
        "If you want multiple specific folder or files just divided them by ',' in between.")
    
    arg_parser.add_argument('--output_dir', metavar='OUTPUT_DIR', type=str,
        default=r"",
        help="Where to store results. If empty, store in the input directory.")
    
    arg_parser.add_argument('--prior', metavar='PRIOR', type=str,
        default=r"",
        help="Path to a .prior file. If blank, we look in the .dynesty folder for other .prior files. " \
        "If no data given and none found not present resort to default built-in bounds.")

    arg_parser.add_argument('--extraprior', metavar='EXTRAPRIOR', type=str, 
        default=r"",
        help="Path to an .extraprior file these are used to add more FragmentationEntry or diferent types of fragmentations. " \
        "If blank, no extraprior file will be used so will only use the prior file.")

    arg_parser.add_argument('-all','--all_cameras',
        help="If active use all data, if not only CAMO data for lag if present in pickle file. " \
        "If False, use CAMO data only for deceleration (by default is False). " \
        "When gnerating json simulations filr if False create a combination EMCCD CAMO data and if True EMCCD only",
        action="store_true")

    arg_parser.add_argument('-new','--new_dynesty',
        help="If active restart a new dynesty run if not resume from existing .dynesty if found. " \
        "If False, create a new dynesty version.",
        action="store_false")
    
    arg_parser.add_argument('-NoBackup','--save_backup',
        help="Run all the simulation agin at th end saves the weighted mass bulk density and save a back with all the data" \
        "and creates the distribution plot takes, in general 10 more minute or more base on the number of cores available.",
        action="store_false")
    
    arg_parser.add_argument('-plot','--only_plot',
        help="If active only plot the results of the dynesty run, if not run dynesty and then plot all when finish.", 
        action="store_true")

    arg_parser.add_argument('--pick_pos', metavar='PICK_POSITION_REAL', type=int, default=0,
        help="corretion for pick postion in the meteor frame raging from from 0 to 1, " \
        "for leading edge picks is 0 for the centroid on the entire meteor is 0.5.")

    arg_parser.add_argument('--cores', metavar='CORES', type=int, default=None,
        help="Number of cores to use. Default = all available.")

    # Optional: suppress warnings
    # warnings.filterwarnings('ignore')

    # Parse
    cml_args = arg_parser.parse_args()

    # check if the pick position is between 0 and 1
    if cml_args.pick_pos < 0 or cml_args.pick_pos > 1:
        raise ValueError("pick_position must be between 0 and 1, 0 leading edge, 0.5 centroid full meteor, 1 trailing edge.")

    setupDirAndRunDynesty(cml_args.input_dir, output_dir=cml_args.output_dir, prior=cml_args.prior, resume=cml_args.new_dynesty, use_all_cameras=cml_args.all_cameras, only_plot=cml_args.only_plot, cores=cml_args.cores, pick_position=cml_args.pick_pos, extraprior_file=cml_args.extraprior, save_backup=cml_args.save_backup)

    print("\nDONE: Completed processing of all files in the input directory.\n")
