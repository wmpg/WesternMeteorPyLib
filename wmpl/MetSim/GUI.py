""" GUI interface to meteor ablation models which enables manual modelling of meteors. """


import os
import sys
import copy
import argparse
import time
import datetime
import json
import glob
import multiprocessing as mp
import importlib.machinery

import numpy as np
import scipy.stats
import scipy.interpolate
import scipy.optimize
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines

import PyQt5.QtCore
from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from PyQt5.uic import loadUi

from wmpl.Formats.Met import loadMet
from wmpl.MetSim.GUITools import MatplotlibPopupWindow
from wmpl.MetSim.MetSimErosion import runSimulation, Constants, zenithAngleAtSimulationBegin
from wmpl.Trajectory.Trajectory import Trajectory, ObservedPoints
from wmpl.Trajectory.Orbit import calcOrbit, Orbit
from wmpl.Utils.AtmosphereDensity import fitAtmPoly, getAtmDensity, atmDensPoly
from wmpl.Utils.Math import mergeClosePoints, findClosestPoints, vectMag, vectNorm, lineFunc, meanAngle
from wmpl.Utils.Physics import calcMass, dynamicPressure, calcRadiatedEnergy
from wmpl.Utils.Pickling import loadPickle, savePickle
from wmpl.Utils.Plotting import saveImage
from wmpl.Utils.TrajConversions import unixTime2JD, datetime2JD, jd2Date, geo2Cartesian, cartesian2Geo, \
    altAz2RADec, altAz2RADec_vect, raDec2ECI, EARTH




### CONSTANTS ###

# Height padding for top plots
BEG_HT_PAD = +5
END_HT_PAD = -2

# Text label height padding
TEXT_LABEL_HT_PAD = 0.1

# Legend text size
LEGEND_TEXT_SIZE  = 8

# Fragmentation file
FRAG_FILE_NAME = "metsim_fragmentation.txt"

# Simulation output file name
SIM_RESULTS_CSV = "metsim_results.csv"

### ###



class SimulationResults(object):
    def __init__(self, const, frag_main, results_list, wake_results):
        """ Container for simulation results. """

        # Save the constants used to compute the results
        self.const = copy.deepcopy(const)

        # Final physical parameters of the main fragment
        self.frag_main = frag_main

        # Unpack the results
        results_list = np.array(results_list).astype(np.float64)
        self.time_arr, self.luminosity_arr, self.luminosity_main_arr, self.luminosity_eroded_arr, \
            self.electron_density_total_arr, self.tau_total_arr, self.tau_main_arr, self.tau_eroded_arr, \
            self.brightest_height_arr, self.brightest_length_arr, self.brightest_vel_arr, \
            self.leading_frag_height_arr, self.leading_frag_length_arr, self.leading_frag_vel_arr, \
            self.leading_frag_dyn_press_arr, self.mass_total_active_arr, \
            self.main_mass_arr, self.main_height_arr, self.main_length_arr, self.main_vel_arr, \
            self.main_dyn_press_arr = results_list.T


        # Calculate the total absolute magnitude (apparent @100km), and fix possible NaN values (replace them 
        #   with the faintest magnitude)
        self.abs_magnitude = -2.5*np.log10(self.luminosity_arr/self.const.P_0m)
        self.abs_magnitude[np.isnan(self.abs_magnitude)] = np.nanmax(self.abs_magnitude)

        # Compute the absolute magnitude of the main fragment
        self.abs_magnitude_main = -2.5*np.log10(self.luminosity_main_arr/self.const.P_0m)
        self.abs_magnitude_main[np.isnan(self.abs_magnitude_main)] = np.nanmax(self.abs_magnitude_main)

        # Compute the absolute magnitude of the eroded and disruped grains
        self.abs_magnitude_eroded = -2.5*np.log10(self.luminosity_eroded_arr/self.const.P_0m)
        self.abs_magnitude_eroded[np.isnan(self.abs_magnitude_eroded)] = np.nanmax(self.abs_magnitude_eroded)   


        # Interpolate time vs leading fragment height
        leading_frag_ht_interpol = scipy.interpolate.interp1d(self.time_arr, self.leading_frag_height_arr)

        # Compute the absolute magnitude of individual fragmentation entries, and join them a height of the
        #   leading fragment
        if self.const.fragmentation_show_individual_lcs:
            for frag_entry in self.const.fragmentation_entries:

                # Compute values for the main fragment
                if len(frag_entry.main_time_data):

                    # Find the corresponding height for every time
                    frag_entry.main_height_data = leading_frag_ht_interpol(np.array(frag_entry.main_time_data))

                    # Compute the magnitude
                    frag_entry.main_abs_mag = -2.5*np.log10(np.array(frag_entry.main_luminosity)
                                                                /self.const.P_0m)

                    # Compute the luminosity weigthed tau
                    frag_entry.main_tau = np.array(frag_entry.main_tau_over_lum)\
                                            /np.array(frag_entry.main_luminosity)


                # Compute values for the grains
                if len(frag_entry.grains_time_data):

                    # Find the corresponding height for every time
                    frag_entry.grains_height_data = leading_frag_ht_interpol(
                        np.array(frag_entry.grains_time_data)
                        )

                    # Compute the magnitude
                    frag_entry.grains_abs_mag = -2.5*np.log10(np.array(frag_entry.grains_luminosity)
                                                                /self.const.P_0m)

                    # Compute the luminosity weigthed tau
                    frag_entry.grains_tau = np.array(frag_entry.grains_tau_over_lum) \
                                                /np.array(frag_entry.grains_luminosity)


        ### Wake simulation ###

        self.wake_results = wake_results
        self.wake_max_lum = 0

        if np.any(wake_results):
            
            # Determine the wake plot upper limit
            self.wake_max_lum = max([max(wake.wake_luminosity_profile) for wake in wake_results \
                if wake is not None])


        ###


    def writeCSV(self, dir_path, file_name):

        # Combine data into one array
        out_arr = np.c_[
            self.time_arr,
            self.brightest_height_arr/1000, self.brightest_length_arr/1000, self.brightest_vel_arr/1000, 
            self.leading_frag_height_arr/1000, self.leading_frag_length_arr/1000, 
            self.leading_frag_vel_arr/1000, self.leading_frag_dyn_press_arr/1e6,
            self.main_height_arr/1000, self.main_length_arr/1000, self.main_vel_arr/1000, \
            self.main_dyn_press_arr/1e6,
            self.tau_total_arr, self.tau_main_arr, self.tau_eroded_arr,
            self.abs_magnitude, self.abs_magnitude_main, self.abs_magnitude_eroded,
            np.log10(self.luminosity_arr), np.log10(self.luminosity_main_arr), np.log10(self.luminosity_eroded_arr), 
            np.log10(self.electron_density_total_arr),
            self.mass_total_active_arr, self.main_mass_arr
            ]

        header  = "B = brightest mass bin, L = leading fragment, M = main\n"
        header += "Time (s), B ht (km), B len (km), B vel (km/s), " + \
                  "L ht (km), L len (km), L vel (km/s), L dyn press (Gamma = 1.0; MPa), " + \
                  "M ht (km), M len (km), M vel (km/s), M dyn press (Gamma = 1.0; MPa), " + \
                  "Tau total (%), Tau main (%), Tau grain (%), " + \
                  "Abs mag total, Abs mag main, Abs mag grain, " + \
                  "log10 Lum total (W), log10 Lum main (W), log10 Lum grain (W), "+\
                  "log10 Electron line density (-/m), Mass total (kg), Mass main (kg)"


        # If the file cannot be opened, throw an error message
        try:
            with open(os.path.join(dir_path, file_name), 'w') as f:
                pass

        except PermissionError:

            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("File permission error")
            msg.setText("Cannot save file " + os.path.join(dir_path, file_name))
            msg.setInformativeText("Make sure you have write permissions, or close the file if it's open in another program.")
            msg.exec_()

            return None


        with open(os.path.join(dir_path, file_name), 'w') as f:

            # Write the data
            np.savetxt(f, out_arr, fmt='%.5e', delimiter=',', newline='\n', header=header, comments="# ")




class MetObservations(object):
    def __init__(self, met, traj):
        """ Container for additional observations from a .met file. Computes the lag and apparent magnitude
            given the base trajectory solution. 
        """

        self.met = met
        self.traj = traj

        self.sites = [site for site in met.sites]


        self.time_data = {}
        self.height_data = {}
        self.state_vect_dist_data = {}
        self.lag_data = {}
        self.abs_mag_data = {}


        # Compute the lag and the magnitude using the given observations
        if self.met is not None:

            # Project additional observations from the .met file to the trajectory and compute the lag
            # Go through all sites
            for site in met.sites:

                jd_picks = []
                time_rel_picks = []
                theta_picks = []
                phi_picks = []
                mag_picks = []

                # Go through all picks
                for pick in met.picks_objs[site]:

                    # Add the pick to the picks list
                    theta_picks.append(pick.theta)
                    phi_picks.append(pick.phi)

                    # Convert the reference Unix time to Julian date
                    ts = int(pick.unix_time)
                    tu = (pick.unix_time - ts)*1e6
                    jd = unixTime2JD(ts, tu)

                    # Add the time of the pick to a list
                    jd_picks.append(jd)

                    # Compute relative time in seconds since the beginning of the meteor
                    time_rel_picks.append((jd - self.traj.jdt_ref)*86400)

                    # Add magnitude
                    mag_picks.append(pick.mag)


                jd_picks = np.array(jd_picks).ravel()
                time_rel_picks = np.array(time_rel_picks).ravel()
                theta_picks = np.array(theta_picks).ravel()
                phi_picks = np.array(phi_picks).ravel()
                mag_picks = np.array(mag_picks).ravel()


                # Compute RA/Dec of observations
                ra_picks, dec_picks = altAz2RADec_vect(np.pi/2 - phi_picks, np.pi/2 - theta_picks, \
                    jd_picks, met.lat[site], met.lon[site])

                
                # List of distances from the state vector (m)
                state_vect_dist = []

                # List of heights (m)
                height_data = []

                # List of absolute magnitudes
                abs_mag_data = []

                # Project rays to the trajectory line
                for i, (jd, ra, dec, mag) in enumerate(np.c_[jd_picks, ra_picks, dec_picks, mag_picks]):

                    # Compute the station coordinates at the given time
                    stat = geo2Cartesian(met.lat[site], met.lon[site], met.elev[site], jd)

                    # Compute measurement rays in cartesian coordinates
                    meas = np.array(raDec2ECI(ra, dec))


                    # Calculate closest points of approach (observed line of sight to radiant line)
                    obs_cpa, rad_cpa, d = findClosestPoints(stat, meas, self.traj.state_vect_mini, \
                        self.traj.radiant_eci_mini)

                    # Compute the height (meters)
                    _, _, ht = cartesian2Geo(jd, *rad_cpa)
                    height_data.append(ht)


                    # If the height is higher than the beginning height, make the state vector distance 
                    #   negative
                    state_vect_dist_sign = 1
                    if ht > self.traj.rbeg_ele:
                        state_vect_dist_sign = -1

                    # Distance from the state vector to the projected point on the radiant line
                    state_vect_dist.append(state_vect_dist_sign*vectMag(self.traj.state_vect_mini - rad_cpa))

                    # Compute the range to the station and the absolute magnitude
                    if mag is not None:
                        r = vectMag(rad_cpa - stat)
                        abs_mag = mag + 5*np.log10(100000/r)
                    else:
                        abs_mag = np.nan

                    abs_mag_data.append(abs_mag)


                state_vect_dist = np.array(state_vect_dist)

                # Store the state vector distance
                self.state_vect_dist_data[site] = state_vect_dist

                # Compute the lag
                self.lag_data[site] = state_vect_dist - lineFunc(time_rel_picks, *self.traj.velocity_fit)

                # Store the time, heights, magnitudes
                self.time_data[site] = np.array(time_rel_picks)
                self.height_data[site] = np.array(height_data)
                self.abs_mag_data[site] = np.array(abs_mag_data)
                


class LightCurveContainer(object):
    def __init__(self, dir_path, file_name):
        """ Loads the light curve data form an CSV file. 
        Arguments:
            dir_path: [str] Path to the directory with the LC file.
            file_name: [str] Name of the LC CSV file.
        """

        self.sites = []

        self.time_data = {}
        self.height_data = {}
        self.abs_mag_data = {}

        with open(os.path.join(dir_path, file_name)) as f:
            
            time_data = []
            height_data = []
            abs_mag_data = []
            current_station = None

            station_label = "# Station:"

            for line in f:

                line = line.replace('\n', '').replace('\r', '')

                if not len(line):
                    continue

                # Start a new station
                if line.startswith(station_label):

                    # Add the previous data to the dictionary
                    if current_station is not None:
                        self.sites.append(current_station)
                        print("Loaded {:d} mag points from station: {:s}".format(len(time_data), current_station))
                        self.time_data[current_station] = np.array(time_data)
                        self.height_data[current_station] = np.array(height_data)
                        self.abs_mag_data[current_station] = np.array(abs_mag_data)

                    time_data = []
                    height_data = []
                    abs_mag_data = []
                    current_station = line.strip(station_label).strip().replace(',', '')

                    continue


                # Skip comments
                if line.startswith('#'):
                    continue

                # Read the data line (convert height to meters)
                t, ht, mag = line.split(',')
                time_data.append(float(t))
                height_data.append(1000*float(ht))
                abs_mag_data.append(float(mag))


            # Add the final station to the list
            if current_station is not None:
                print("Loaded {:d} mag points from station: {:s}".format(len(time_data), current_station))
                self.sites.append(current_station)
                self.time_data[current_station] = np.array(time_data)
                self.height_data[current_station] = np.array(height_data)
                self.abs_mag_data[current_station] = np.array(abs_mag_data)



class WakePoint(object):
    def __init__(self, n, th, phi, intens_sum, amp, r, b, c, state_vect_dist, ht):
        """ Container for wake points. """

        # Distance along the trail (pixels)
        self.n = n

        # Zenith angle (deg)
        self.th = th

        # Azimuth +N of due E (deeg)
        self.phi = phi

        # Intensity sum
        self.intens_sum = intens_sum

        # Peak pixel intensity
        self.amp = amp

        # Raw trail width (pixels above the background)
        self.r = r

        # Standard deviation of the Gaussian fit (meters)
        self.b = b

        # Corrected trail width (raw - stellar width) (meters)
        self.c = c

        # Distance from the trajectory state vector (m)
        self.state_vect_dist = state_vect_dist

        # Height (m)
        self.ht = ht

        # Distance from the leading fragment (m)
        self.leading_frag_length = 0



class WakeContainter(object):
    def __init__(self, site_id, frame_n):
        """ Containter for wake profile data extracted from mirfit wid files. 
        
        Arguments:
            site_id: [str] Name of the site.
            frame_n: [str] Frame number of the measurement.
        """

        self.site_id = int(site_id)
        self.frame_n = frame_n

        self.points = []


    def addPoint(self, n, th, phi, intens_sum, amp, r, b, c, state_vect_dist, ht):
        self.points.append(WakePoint(n, th, phi, intens_sum, amp, r, b, c, state_vect_dist, ht))




class FragmentationEntry(object):
    def __init__(self, frag_type, height, number, mass_percent, sigma, gamma, erosion_coeff, grain_mass_min, \
        grain_mass_max, mass_index, normalize_units=False):
        """ A container for every fragmentation entry. 

        normalize_units: [bool] Convert the units of sigma and erosion coeff from s^2 km^-2 to s^2 m^-2,
            and convert the height to meters from km.
        """

        norm = 0.0
        if normalize_units:
            norm = 1.0


        ### STATUS FLAGS ###


        # Unique identifer of the fragmentation entry, will be assigned at the beginning of simulation
        self.id = None

        # Indicates that the fragmentation was not performed yet
        self.done = False


        ### ###

        ### INPUT parameters ###


        self.frag_type = frag_type.strip()

        self.height = (1000**norm)*float(height)

        try:
            self.number = int(number)
        except:
            if (self.frag_type == "F") or (self.frag_type == "EF"):
                self.number = 1
            else:
                self.number = None

        try:
            self.mass_percent = float(mass_percent)

            # Limit mass loss to 100%
            if self.mass_percent > 100:
                mass_percent = 100.0

        except:
            self.mass_percent = None

        try:
            self.sigma = float(sigma)/(1e6**norm)
        except:
            self.sigma = None

        try:
            self.gamma = float(gamma)
        except:
            self.gamma = None

        try:
            self.erosion_coeff = float(erosion_coeff)/(1e6**norm)
        except:
            self.erosion_coeff = None

        try:
            self.grain_mass_min = float(grain_mass_min)
        except:
            self.grain_mass_min = None

        try:
            self.grain_mass_max = float(grain_mass_max)
        except:
            self.grain_mass_max = None

        try:
            self.mass_index = float(mass_index)
        except:
            self.mass_index = None

        ### ###


        ### List of fragments generated by the fragmentation ###

        self.fragments = []

        ###


        ### Output parameters ###

        # Time of fragmentation
        self.time = None

        # Dynamic pressure of fragmentation
        self.dyn_pressure = None

        # Velocity of fragmentation
        self.velocity = None

        # Parent mass at the moment of fragmentation
        self.parent_mass = None

        # Initial mass of the fragments
        self.mass = None

        # Final mass of the fragments
        self.final_mass = None


        self.resetOutputParameters()
        

        ### ###


    def resetOutputParameters(self):
        """ Reset lists with output parameters. """

        # Time data of the main parent fragment (not the grains)
        self.main_time_data = []

        # Height (will be joined in postprocessing to the height of the leading fragment)
        self.main_height_data = []

        # Luminosity of the main parent
        self.main_luminosity = []

        # Absolute magnitude (will be computed after the simulation)
        self.main_abs_mag = []

        # Luminous efficiency of the main parent divided by luminosity
        self.main_tau_over_lum = []

        # Lumnous efficiency weighted by the luminosity
        self.main_tau = []


        # Time data for the grains
        self.grains_time_data = []

        # Height (will be joined in postprocessing to the height of the leading fragment)
        self.grains_height_data = []

        # Luminosity of the grains
        self.grains_luminosity = []

        # Absolute magnitude (will be computed after the simulation)
        self.grains_abs_mag = []

        # Lumionus efficiency of grains divided by luminsotiy
        self.grains_tau_over_lum = []

        # Lumnous efficiency of grains weighted by the luminosity
        self.grains_tau = []



    def toString(self):
        """ Convert the entry to a string that can be written to a text file. """

        line_entries = []

        line_entries.append("{:>6s}".format(self.frag_type))

        line_entries.append("{:11.3f}".format(self.height/1000))

        if self.number is not None:
            line_entries.append("{:6d}".format(self.number))
        else:
            line_entries.append(6*" ")

        if self.mass_percent is not None:
            line_entries.append("{:8.3f}".format(self.mass_percent))
        else:
            line_entries.append(8*" ")

        if self.sigma is not None:
            line_entries.append("{:14.4f}".format(1e6*self.sigma))
        else:
            line_entries.append(14*" ")

        if self.gamma is not None:
            line_entries.append("{:5.2f}".format(self.gamma))
        else:
            line_entries.append(5*" ")

        if self.erosion_coeff is not None:
            line_entries.append("{:13.4f}".format(1e6*self.erosion_coeff))
        else:
            line_entries.append(13*" ")

        if self.grain_mass_min is not None:
            line_entries.append("{:9.2e}".format(self.grain_mass_min))
        else:
            line_entries.append(9*" ")

        if self.grain_mass_max is not None:
            line_entries.append("{:9.2e}".format(self.grain_mass_max))
        else:
            line_entries.append(9*" ")

        if self.mass_index is not None:
            line_entries.append("{:5.2f}".format(self.mass_index))
        else:
            line_entries.append(5*" ")


        # Join the entries to one string
        out_str = ", ".join(line_entries)

        # Add separator from inputs and outputs
        out_str += " # "


        line_entries = []

        if self.time is not None:
            line_entries.append("{:9.6f}".format(self.time))
        else:
            line_entries.append(9*" ")

        if self.dyn_pressure is not None:
            line_entries.append("{:9.3f}".format(self.dyn_pressure/1000))
        else:
            line_entries.append(9*" ")

        if self.velocity is not None:
            line_entries.append("{:8.3f}".format(self.velocity/1000))
        else:
            line_entries.append(8*" ")

        if self.parent_mass is not None:
            line_entries.append("{:11.2e}".format(self.parent_mass))
        else:
            line_entries.append(11*" ")

        if self.mass is not None:
            line_entries.append("{:9.2e}".format(self.mass))
        else:
            line_entries.append(9*" ")

        if self.final_mass is not None:
            line_entries.append("{:10.2e}".format(self.final_mass))
        else:
            line_entries.append(10*" ")

        out_str += ", ".join(line_entries)

        # Add final separator
        out_str += " #"


        return out_str



            



class FragmentationContainer(object):
    def __init__(self, gui, fragmentation_file_path):
        """ Class which handles fragmentation file I/O. 
    
        Arguments:
            gui: [MetSimGUI object] A handle to the parent GUI object.
            fragmentation_file_path: [str] Path to the main fragmentation file.

        """

        self.gui = gui

        self.fragmentation_file_path = fragmentation_file_path

        self.fragmentation_entries = []


    def sortByHeight(self):
        """ Sort the fragmentation entries by height, from highest to lowest. """

        self.fragmentation_entries = sorted(self.fragmentation_entries, key=lambda x: x.height, reverse=True)


    def resetAll(self):
        """ Reset the 'done' flag of all fragments to False, which means that the fragmentation should run.
        """

        for frag_entry in self.fragmentation_entries:
            frag_entry.done = False


    def loadFromString(self, string):
        """ Load parameters from a string. """

        # Reset fragmentation entries and reload
        self.fragmentation_entries = []

        for line in string.split('\n'):

            # Skip comment lines
            if line.startswith("#"):
                continue

            line = line.replace('\n', '').replace('\r', '')

            if not len(line):
                continue

            # Strip the output part
            line = line.split("#")[0]

            entries = line.split(',')

            # There need to be exactly 10 entries for every fragmentation
            if len(entries) != 10:
                print("ERROR! Cannot read fragmentation line:")
                print(line)
                continue

            # Create a new fragmentation entry
            frag_entry = FragmentationEntry(*entries, normalize_units=True)

            self.fragmentation_entries.append(frag_entry)


        # Set the fragmentation entries to constants
        self.gui.const.fragmentation_entries = self.fragmentation_entries


    def loadFragmentationFile(self):
        """ Load fragmentation paramters from the fragmentation file. """

        string = ""
        with open(self.fragmentation_file_path) as f:
            for line in f:
                string += line


        self.loadFromString(string)


    def toString(self):
        """ Convert the container to a string. """

        # Write the header
        out_str  = ""
        out_str += """# MetSim fragmentation file.
#
# Types of entries:
#   - *INIT - Initial parameters taken from the GUI. Ready only, cannot be set in this file.
#   - M    - Main fragment - parameter change.
#           - REQUIRED: Height.
#           - Possible: Ablation coeff, Erosion coeff, Grain masses, Mass index.
#   - A    - All fragments - parameter change.
#           - REQUIRED: Height.
#           - Possible: Ablation coeff, Gamma.
#   - F    - New single-body fragment.
#           - REQUIRED: Height, Number, Mass (%).
#           - Possible: Ablation coeff.
#   - EF   - New eroding fragment. A mass index of 2.0 will be assumed if not given.
#           - REQUIRED: Height, Number, Mass (%), Erosion coeff, Grain masses.
#           - Possible: Ablation coeff, Mass index.
#   - D    - Dust release. Only the grain mass range needs to be specified. A mass index of 2.0 will be assumed if not given.
#           - REQUIRED: Height, Mass (%), Grain MIN mass, Grain MAX mass.
#           - Possible: Mass index.
#
#                             INPUTS (leave unchanged fields empty)                                      #        OUTPUTS  (do not fill in!)                                  #
# ------------------------------------------------------------------------------------------------------ # ------------------------------------------------------------------ #
# Type, Height (km), Number, Mass (%), Ablation coeff, Gamma, Erosion coeff, Grain MIN, Grain MAX, Mass  #  Time (s),  Dyn pres, Velocity, Parent mass, Mass (kg), Final mass #
#     ,            ,       ,         , (s^2 km^-2)   ,      , (s^2 km^-2)  , mass (kg), mass (kg), index #          ,  (kPa)   , (km/s)  , (kg)       ,          , (kg)       #
#-----,------------,-------,---------,---------------,------,--------------,----------,----------,-------#----------,----------,---------,------------,----------,----------- #
"""
    
    
        frag_entry_list = []

        # Write the initial parameters
        initial_entry = FragmentationEntry(frag_type="# INIT",
                                            height=self.gui.const.h_init,
                                            number=1,
                                            mass_percent=100,
                                            sigma=self.gui.const.sigma,
                                            gamma=self.gui.const.gamma,
                                            erosion_coeff=0.0,
                                            grain_mass_min=None,
                                            grain_mass_max=None,
                                            mass_index=None
                                            )

        frag_entry_list.append(initial_entry)
        frag_entry_list += self.fragmentation_entries

        # Write the last parameters of the main fragment
        if self.gui.simulation_results is not None:

            final_fragment = FragmentationEntry(frag_type="# END",
                                                height=self.gui.simulation_results.frag_main.h,
                                                number=1,
                                                mass_percent=None,
                                                sigma=None,
                                                gamma=None,
                                                erosion_coeff=None,
                                                grain_mass_min=None,
                                                grain_mass_max=None,
                                                mass_index=None
                )

            final_fragment.final_mass = self.gui.simulation_results.frag_main.m

            frag_entry_list.append(final_fragment)


        # Write fragmentation entries
        for frag_entry in frag_entry_list:
            out_str += frag_entry.toString() + "\n"


        return out_str



    def writeFragmentationFile(self):

        with open(self.fragmentation_file_path, 'w') as f:
            f.write(self.toString())

        print("Fragmentation file saved:", self.fragmentation_file_path)


    def newFragmentationFile(self):
        """ Open a new fragmentation file. """

        # Reset fragmentation entries
        self.fragmentation_entries = []

        # Write an empty fragmentation file to disk
        self.writeFragmentationFile()

        self.gui.const.fragmentation_file_name = os.path.basename(self.fragmentation_file_path)


    def addFragmentation(self, frag_type):
        """ Add a new fragmentation entry. """

        # Load the fragmentation file
        self.loadFragmentationFile()


        # Change paramters of the main fragment
        if frag_type == "M":
            frag_entry = FragmentationEntry(frag_type=frag_type,
                                            height=100000.0000,
                                            number=None,
                                            mass_percent=None,
                                            sigma=None,
                                            gamma=None,
                                            erosion_coeff=self.gui.const.erosion_coeff,
                                            grain_mass_min=None,
                                            grain_mass_max=None,
                                            mass_index=None
                                            )

        # Change paramters of all fragments
        elif frag_type == "A":
            frag_entry = FragmentationEntry(frag_type=frag_type,
                                            height=100000.0,
                                            number=None,
                                            mass_percent=None,
                                            sigma=self.gui.const.sigma,
                                            gamma=self.gui.const.gamma,
                                            erosion_coeff=None,
                                            grain_mass_min=None,
                                            grain_mass_max=None,
                                            mass_index=None
                                            )

        # New single-body fragment
        elif frag_type == "F":
            frag_entry = FragmentationEntry(frag_type=frag_type,
                                            height=100000.0,
                                            number=1,
                                            mass_percent=10,
                                            sigma=self.gui.const.sigma,
                                            gamma=None,
                                            erosion_coeff=None,
                                            grain_mass_min=None,
                                            grain_mass_max=None,
                                            mass_index=None
                                            )

        elif frag_type == "EF":
            frag_entry = FragmentationEntry(frag_type=frag_type,
                                            height=100000.0,
                                            number=1,
                                            mass_percent=10,
                                            sigma=None,
                                            gamma=None,
                                            erosion_coeff=self.gui.const.erosion_coeff,
                                            grain_mass_min=self.gui.const.erosion_mass_min,
                                            grain_mass_max=self.gui.const.erosion_mass_max,
                                            mass_index=self.gui.const.erosion_mass_index
                                            )

        elif frag_type == "D":
            frag_entry = FragmentationEntry(frag_type=frag_type,
                                            height=100000.0,
                                            number=None,
                                            mass_percent=10,
                                            sigma=None,
                                            gamma=None,
                                            erosion_coeff=None,
                                            grain_mass_min=self.gui.const.erosion_mass_min,
                                            grain_mass_max=self.gui.const.erosion_mass_max,
                                            mass_index=self.gui.const.erosion_mass_index
                                            )

        else:
            print("ERROR! Unknown fragmentation type:", frag_type)


        # Add the fragment entry
        self.fragmentation_entries.append(frag_entry)
        self.gui.const.fragmentation_entries = self.fragmentation_entries


        # Save the fragmentation file
        self.writeFragmentationFile()






class MinimizationParameterNormalization(object):
    def __init__(self, params):
        """ Normalize the fit parameters so that they are approx in the 0 to 1 range.

            The normalization can be done by giving the fit bounds, in which case all parameters will be
            normalized to the 0 to 1 range. Or, a fixed value can be given for every parameter which should
            bring it approx to the 0 to 1 range.

        Arguments:
            params: [list] A list of fit parameters.


        """

        self.params = params

        self.normalization_method = None
        self.bounds = None
        self.scaling_list = None



    def normalizeBounds(self, bounds):
        """ Normalize the parameters given the fit bounds. 
        
        Arguments:
            bounds: [list] A list of bounds for every parameter.

        Return:
            params_normed: [list] A list of normalized parameters.
            bounds_normed: [list] A list of normalized boundaries to the [0, 1] range.
        """

        self.normalization_method = 'bounds'
        self.bounds = bounds

        # Normalize every parameter
        params_normed = []
        bounds_normed = []
        for p, bound in zip(self.params, self.bounds):

            bound_min, bound_max = bound

            # Normalize the parameter to [0, 1] range
            p_normed = (p - bound_min)/(bound_max - bound_min)

            params_normed.append(p_normed)

            bounds_normed.append([0, 1])


        return params_normed, bounds_normed



    def normalizeScaling(self, scaling_list, bounds=None):
        """ Normalize the parameters by specifying a scale for every paramter. 
    
        Arguments:
            scaling_list: [list] A list of values used for scaling each parameter to approx the 0 to 1 range
                using multiplication.

        Keyword arguments:
            bounds: [list] A list of bounds which will be scaled. None by default.


        Return;
            if bounds is None:
                params_normed: [list] A list of scaled paramters.
            else:
                params_normed: [list] A list of scaled paramters.
                bounds_normed: [list] A list of scaled boundaries.
        """

        self.normalization_method = 'scaling'
        self.scaling_list = scaling_list
        self.bounds = bounds

        params_normed = []
        bounds_normed = []

        for i, (p, scale) in enumerate(zip(self.params, self.scaling_list)):

            # Scale the parameter
            p_normed = p*scale

            params_normed.append(p_normed)

            # Scale the bounds, if given
            if bounds is not None:
                bound = bounds[i]
                bound_normed = [b*scale for b in bound]
                bounds_normed.append(bound_normed)

        
        if bounds is None:
            return params_normed

        else:
            return params_normed, bounds_normed



    def denormalize(self, params_normed):
        """ Denormalize the given normalized parameters. 
        
        Arguments:
            params_normed: [list] A list of normalized parameters.

        Return:
            params: [list] A list of denormalized parameters.
        """

        # Denormalize using bounds
        if self.normalization_method == 'bounds':

            if self.bounds is None:
                raise ValueError("Boundaries for parameter denormalization not specified!")


            params = []

            for p_normed, bound in zip(params_normed, self.bounds):

                bound_min, bound_max = bound

                # Denormalize the paramer to original bounds
                p = p_normed*(bound_max - bound_min) + bound_min

                params.append(p)


        # Denormalize using scaling
        elif self.normalization_method == 'scaling':

            if self.scaling_list is None:
                raise ValueError("Scaling list for parameter denormalization not specified!")

            params = []

            for p_normed, scale in zip(params_normed, self.scaling_list):

                p = p_normed/scale

                params.append(p)


        return params




def extractConstantParams(const_original, params, param_string, mini_norm_handle):
    """ Assign parameters to a Constants object given an array of parameters. """

    # Create a copy of fit parameters
    const = copy.deepcopy(const_original)


    ### Extract fit parameters ###


    # Denormalize the fit parameters from the [0, 1] range
    params = mini_norm_handle.denormalize(params)

    # Read meteoroid properties
    param_index = 4
    const.m_init, const.v_init, const.rho, const.sigma = params[:param_index]

    # Read erosion parameters if the erosion is on
    if 'e' in param_string:
        const.erosion_height_start, const.erosion_coeff, const.erosion_height_change, \
            const.erosion_coeff_change, const.erosion_mass_index, const.erosion_mass_min, \
            const.erosion_mass_max = params[param_index:param_index + 7]

        param_index += 7


    # Read distuption parameters if the disruption is on
    if 'd' in param_string:
        const.compressive_strength, const.disruption_erosion_coeff, const.disruption_mass_index, \
            const.disruption_mass_min_ratio, const.disruption_mass_max_ratio, \
            const.disruption_mass_grain_ratio = params[param_index:param_index + 6]

        param_index += 6

    ### ###
    
    return const




def fitResiduals(params, fit_input_data, param_string, const_original, traj, mini_norm_handle, 
    mag_weight=10.0, lag_weights=None, lag_weight_ht_change=0.0, verbose=True, gui_handle=None):
    """ Compute the fit residual. """

    if verbose:
        print()
        print('Params:')
        # print(params)
        print(mini_norm_handle.denormalize(params))


    if lag_weights is None:
        lag_weights = [1.0, 1.0]


    # Assign fit parameters to a constants object
    const = extractConstantParams(const_original, params, param_string, mini_norm_handle)


    # Run the simulation
    frag_main, results_list, wake_results = runSimulation(const, compute_wake=False)

    # Store simulation results
    sr = SimulationResults(const, frag_main, results_list, wake_results)


    ### Compute the residuals ###

    # Count the total number of length and magnitude points
    mag_point_count = len(~np.isnan(fit_input_data[:, 1]))
    lag_point_count = len(~np.isnan(fit_input_data[:, 2]))

    # Interpolate simulated magnitude and length
    len_sim_interpol = scipy.interpolate.interp1d(sr.brightest_height_arr, sr.brightest_length_arr, \
        bounds_error=False, fill_value=0)
    mag_sim_interpol = scipy.interpolate.interp1d(sr.brightest_height_arr, sr.abs_magnitude, \
        bounds_error=False, fill_value=10)
    time_sim_interpol = scipy.interpolate.interp1d(sr.brightest_height_arr, sr.time_arr, \
        bounds_error=False, fill_value=0)

    
    # Find the length and time at the meteor begin point
    begin_length_sim = len_sim_interpol(traj.rbeg_ele)
    begin_time_sim = time_sim_interpol(traj.rbeg_ele)


    # Go through every height point and compute the residual contribution
    total_mag_residual = 0
    total_lag_residual = 0
    for entry in fit_input_data:

        height, mag, lag = entry


        # Find the corresponding magnitude and length from the simulation to the observation
        mag_sim = mag_sim_interpol(height)
        lag_sim = len_sim_interpol(height) - begin_length_sim \
            - traj.orbit.v_init*(time_sim_interpol(height) - begin_time_sim)


        # print("{:.2f}, {:.2f}, {:.2f}, {:.0f}, {:.0f}".format(height, mag, mag_sim, lag, lag_sim))


        # # Compute the magnitude residual 
        # mag_res = 0
        # if not np.isnan(mag):
        #     mag_res = abs((mag - mag_sim)*mag_weight)

        # if np.isnan(mag_res):
        #     mag_res = 10.0

        # Compute the magnitude residual in linear luminosity units
        if not np.isnan(mag):
            lum_obs = const.P_0m*10**(-0.4*mag)
            lum_sim = const.P_0m*10**(-0.4*mag_sim)

            mag_res = mag_weight*(lum_obs - lum_sim)**2

        if np.isnan(mag_res):
            mag_res = 1e6**2


        # Choose the lag weight
        if height > lag_weight_ht_change:
            lag_weight = lag_weights[0]
        else:
            lag_weight = lag_weights[1]


        # Compute the length residual
        lag_res = 0
        if not np.isnan(lag):
            lag_res = lag_weight*(lag - lag_sim)**2

        if np.isnan(lag_res):
            lag_res = 10000**2


        total_mag_residual += mag_res
        total_lag_residual += lag_res

    total_residual = total_mag_residual/mag_point_count + total_lag_residual/lag_point_count


    # If the total residual is zero, make it very high
    if total_residual == 0:
        total_residual = 1e8


    if verbose:
        print('Total residual:', total_residual)

        if gui_handle is not None:

            # Plot the results of the current fit
            gui_handle.simulation_results = sr
            gui_handle.const = const
            gui_handle.showCurrentResults()
            gui_handle.repaint()


    return total_residual



# Modify the residuals function so that it takes a list of arguments
def fitResidualsListArguments(params, *args, **kwargs):
    return [fitResiduals(param_line, *args, **kwargs) for param_line in params]



def loadConstants(sim_fit_json):
    """ Load the simulation constants from a JSON file. 
        
    Arguments:
        sim_fit_json: [str] Path to the sim_fit JSON file.

    Return:
        (const, const_json): 
            - const: [Constants object]
            - const_json: [dict]

    """

    # Init the constants
    const = Constants()


    # Load the nominal simulation
    with open(sim_fit_json) as f:
        const_json = json.load(f)


        # Fill in the constants
        for key in const_json:
            setattr(const, key, const_json[key])


    if 'fragmentation_entries' in const_json:

        # Convert fragmentation entries from dictionaties to objects
        frag_entries = []
        if len(const_json['fragmentation_entries']) > 0:
            for frag_entry_dict in const_json['fragmentation_entries']:

                # Only take entries which are variable names for the FragmentationEntry class
                frag_entry_dict = {key:frag_entry_dict[key] for key in frag_entry_dict \
                    if key in FragmentationEntry.__init__.__code__.co_varnames}

                frag_entry = FragmentationEntry(**frag_entry_dict)
                frag_entries.append(frag_entry)

        const.fragmentation_entries = frag_entries


    return const, const_json



def saveConstants(const, dir_path, file_name):
    """ Save the simulation constants to a JSON file.

    Arguments:
        const: [Constants object] The constants to save.
        dir_path: [str] The directory path to save the file to.
        file_name: [str] The name of the file to save.

    """


    # Create a copy of the fit parameters
    const = copy.deepcopy(const)


    # Convert the density parameters to a list
    if isinstance(const.dens_co, np.ndarray):
        const.dens_co = const.dens_co.tolist()

    # Remove fragments from entries becuase they can't be saved in JSON
    for frag_entry in const.fragmentation_entries:
        del frag_entry.fragments
        frag_entry.resetOutputParameters()


    file_path = os.path.join(dir_path, file_name)
    with open(file_path, 'w') as f:
        json.dump(const, f, default=lambda o: o.__dict__, indent=4)


def loadUSGInputFile(dir_path, usg_file):
    """ Load the USG input file into a trajectory. """


    # Load the file data
    loader = importlib.machinery.SourceFileLoader('data', os.path.join(dir_path, usg_file))
    data = loader.load_module()

    # Convert time from string to datetime
    try:
        data.dt = datetime.datetime.strptime(data.time, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        data.dt = datetime.datetime.strptime(data.time, "%Y-%m-%d %H:%M:%S")

    # Convert time to JD
    data.jd = datetime2JD(data.dt)


    ### Compute light curve ###
    
    data.usg_intensity_data = np.array(data.usg_intensity_data)

    # Extract the time
    data.time_data = data.usg_intensity_data[:,0]

    # Compute the light curve (see Brown et al. 1996, St. Robert paper)
    data.absolute_magnitudes = -2.5*np.log10(data.usg_intensity_data[:,1]/248)

    ### ###


    # Align the time so that the reference time 0 is at the height of the peak magnitude
    peak_time_offset = data.time_data[np.nanargmin(data.absolute_magnitudes)] 
    data.time_data -= peak_time_offset


    ### Prepare all light curves as input ###

    # Add USG/CNEOS data
    lc_data = {}
    lc_data["CNEOS"] = np.c_[np.array(data.time_data), np.array(data.absolute_magnitudes)]

    # Add any additional light curves (note that the time base needs to be the same)
    if hasattr(data, "additional_lcs"):

        # Apply time offset to the additional LCs
        for obs in data.additional_lcs:

            additional_time, additional_mag = np.array(data.additional_lcs[obs]).T
            additional_time -= peak_time_offset

            data.additional_lcs[obs] = np.c_[additional_time, additional_mag]
            

        lc_data.update(data.additional_lcs)


    ### ###


    

    # Get the reference point in ECI
    eci_ref = np.array(geo2Cartesian(np.radians(data.lat), np.radians(data.lon), 1000*data.ht, data.jd))

    # Get the radiant in ECI coordiantes
    ra_rad, dec_rad = altAz2RADec(np.radians(data.azimuth), np.radians(data.entry_angle), data.jd, \
        np.radians(data.lat), np.radians(data.lon))
    eci_rad = vectNorm(np.array(raDec2ECI(ra_rad, dec_rad)))



    # Init the trajectory structure
    traj = Trajectory(data.jd, output_dir=dir_path, meastype=2)


    # Keep track of the beg/end point
    rbeg_lat = 0
    rend_lat = 0
    rbeg_lon = 0
    rend_lon = 0
    rbeg_ele = 0
    rend_ele = np.inf

    # Go though all available light curves
    for station_id in lc_data:

        time_data, abs_mag_data = np.array(lc_data[station_id]).T


        ### Compute the height array ###

        # Compute height of the fireball over time
        lat_data = []
        lon_data = []
        height_data = []
        for t in time_data:

            # Compute the length relative to the reference point (in meters)
            l = t*1000*data.v

            # Compute the ECI coordinates of the fireball
            eci = eci_ref - l*eci_rad

            # Compute reference julian date
            jd = data.jd + t/86400

            # Compute geo coordinates of the point on the trajectory at the given time
            lat, lon, h = cartesian2Geo(jd, *eci)

            lat_data.append(lat)
            lon_data.append(lon)
            height_data.append(h)

        lat_data = np.array(lat_data)
        lon_data = np.array(lon_data)
        height_data = np.array(height_data)

        ### ###


        # Init the observations
        obs = ObservedPoints(data.jd, np.zeros_like(abs_mag_data), \
            np.zeros_like(abs_mag_data), time_data, np.radians(data.lat), np.radians(data.lon), \
            1000*data.ht, 2, station_id=station_id)

        # Assign magnitude data
        obs.absolute_magnitudes = abs_mag_data

        # Assign dynamics
        obs.velocities = np.zeros_like(obs.absolute_magnitudes) + 1000*data.v
        obs.lag = np.zeros_like(obs.absolute_magnitudes)

        # Assign computed heights
        obs.model_ht = obs.meas_ht = height_data

        # Assign computed geo coordinates
        obs.model_lat = obs.meas_lat = lat_data
        obs.model_lon = obs.meas_lon = lon_data

        # Add the observations to the trajectory
        traj.observations.append(obs)


        # Keep track of the begin/final point
        top_index = np.argmax(height_data)
        bottom_index = np.argmin(height_data)
        if height_data[top_index] > rbeg_ele:
            rbeg_lat = lat_data[top_index]
            rbeg_lon = lon_data[top_index]
            rbeg_ele = height_data[top_index]

        if height_data[bottom_index] < rend_ele:
            rend_lat = lat_data[bottom_index]
            rend_lon = lon_data[bottom_index]
            rend_ele = height_data[bottom_index]



    # Assign trajectory parameters
    traj.rbeg_lat = rbeg_lat
    traj.rend_lat = rend_lat
    traj.rbeg_lon = rbeg_lon
    traj.rend_lon = rend_lon
    traj.rbeg_ele = rbeg_ele
    traj.rend_ele = rend_ele
    traj.orbit = Orbit()
    traj.orbit.zc = np.radians(90 - data.entry_angle)
    traj.v_avg = traj.orbit.v_avg = traj.orbit.v_avg_norot = 1000*data.v
    traj.v_init = traj.orbit.v_init = traj.orbit.v_init_norot = 1000*data.v
    traj.radiant_eci_mini = eci_rad
    traj.state_vect_mini = eci_ref
    traj.rbeg_jd = data.jd


    return data, traj


def loadWakeFile(traj, file_path):
    """ Load a mirfit wake "wid" file. 
    
    Arguments:
        traj: Trajectory object.
        file_path: Path to the wid file.

    Return:
        wake_container: WakeContainer object.
    
    """


    # Extract the site ID and the frame number from the file name
    site_id, frame_n = os.path.basename(file_path).replace('.txt', '').split('_')[1:]
    site_id = str(int(site_id))
    frame_n = int(frame_n)

    print('wid file: ', site_id, frame_n, end='')


    # Extract geo coordinates of sites
    lat_dict = {obs.station_id:obs.lat for obs in traj.observations}
    lon_dict = {obs.station_id:obs.lon for obs in traj.observations}
    ele_dict = {obs.station_id:obs.ele for obs in traj.observations}

    wake_container = None
    leading_state_vect_dist = 0
    with open(file_path) as f:
        for line in f:

            if line.startswith('#'):
                continue

            line = line.replace('\n', '').replace('\r', '').split()

            if not line:
                continue


            # Init the wake container
            if wake_container is None:
                wake_container = WakeContainter(site_id, frame_n)

            # Read the wake point
            n, th, phi, _, _, _, _, intens_sum, amp, r, b, c = list(map(float, line))

            # Skip bad measurements
            if np.any(np.isnan([th, phi])):
                continue

            ### Compute the projection of the wake line of sight to the trajectory ###

            # Calculate RA/Dec
            ra, dec = altAz2RADec(np.pi/2 - np.radians(phi), np.pi/2 - np.radians(th), \
                traj.jdt_ref, lat_dict[site_id], lon_dict[site_id])

            # Compute the station coordinates at the given time
            stat = geo2Cartesian(lat_dict[site_id], lon_dict[site_id], ele_dict[site_id], \
                traj.jdt_ref)

            # Compute measurement rays in cartesian coordinates
            meas = np.array(raDec2ECI(ra, dec))

            # Calculate closest points of approach (observed line of sight to radiant line)
            obs_cpa, rad_cpa, d = findClosestPoints(stat, meas, traj.state_vect_mini, \
                traj.radiant_eci_mini)

            # If the projected point is above the state vector, use negative lengths
            state_vect_dist_sign = 1.0
            if vectMag(rad_cpa) > vectMag(traj.state_vect_mini):
                state_vect_dist_sign = -1.0


            # Compute Distance from the state vector to the projected point on the radiant line
            state_vect_dist = state_vect_dist_sign*vectMag(traj.state_vect_mini - rad_cpa)

            # Compute the height (meters)
            _, _, ht = cartesian2Geo(traj.jdt_ref, *rad_cpa)

            ### ###

            # Record the state vector distance of the leading fragment
            if int(n) == 0:
                leading_state_vect_dist = state_vect_dist

            wake_container.addPoint(n, th, phi, intens_sum, amp, r, b, c, state_vect_dist, ht)


        # If there are no points in the wake container, don't use it
        if wake_container is not None:
            if len(wake_container.points) == 0:
                wake_container = None

        # Compute lengths of the leading fragment
        if wake_container is not None:
            for wake_pt in wake_container.points:
                wake_pt.leading_frag_length = wake_pt.state_vect_dist - leading_state_vect_dist

    if wake_container is None:
        print("... rejected")
    else:
        print("... loaded!")

    return wake_container



def extractWake(sr, wake_containers, wake_fraction=0.5, peak_region=20, site_id=None, max_len_shift=50):
    """ Extract the wake from the simulation results. 
    
    Arguments:
        sr: [SimulationResults object] Simulation results.
        wake_containers: [list of WakeContainer objects] List of wake containers.

    Keyword arguments:
        wake_fraction: [float] Fraction of the wake height to probe. 0 is the height when tracking began and
            1 is the height when the tracking stopped.
        peak_region: [float] Region around the peak to use for the wake normalization (m). If None, the whole
            wake will be used.
        site_id: [int] Name of the site where the meteor was observed. 1 for Tavistock and 2 for Elginfield. 
            If None, both will be taken.
        max_len_shift: [float] Maximum length shift allowed when aligning the observed and simulated wakes (m).
            If the shift is larger than this, the wake will not be aligned.
    """

    if peak_region is None:
        peak_region = np.inf

    # Filter wake containers by site
    if site_id is not None:
        wake_containers = [wake_container for wake_container in wake_containers 
                           if wake_container.site_id == site_id]

    # Get a list of all heights in the wake
    wake_heights = [wake_container.points[0].ht for wake_container in wake_containers]

    # Compute the range of heights
    ht_range = np.max(wake_heights) - np.min(wake_heights)

    # Compute the probing heights
    ht_ref = np.max(wake_heights) - wake_fraction*ht_range

    # Find the container which are closest to reference height of the wake fraction
    ht_ref_idx = np.argmin(np.abs(np.array(wake_heights) - ht_ref))

    # Get the two containers with observations
    wake_container_ref = wake_containers[ht_ref_idx]

    # Find the wake index closest to the given wake height, ignoring nana
    wake_res_indx_ref =  np.nanargmin(np.abs(ht_ref - sr.leading_frag_height_arr))

    # Extract the wake results
    wake_ref = sr.wake_results[wake_res_indx_ref]

    # Extract the wake points from the containers
    len_ref_array = []
    wake_ref_intensity_array = []
    for wake_pt in wake_container_ref.points:
        len_ref_array.append(wake_pt.leading_frag_length)
        wake_ref_intensity_array.append(wake_pt.intens_sum)

    len_ref_array = np.array(len_ref_array)
    wake_ref_intensity_array = np.array(wake_ref_intensity_array)

    # Normalize the wake intensity so the areas are equal between the observed and simulated wakes
    # Only take the points +/- peak_region m from the maximum intensity
    obslen_ref_max_intens = len_ref_array[np.argmax(wake_ref_intensity_array)]
    obslen_ref_filter = np.abs(len_ref_array - obslen_ref_max_intens) < peak_region
    obs_area_ref = np.trapz(wake_ref_intensity_array[obslen_ref_filter], len_ref_array[obslen_ref_filter])
    simlen_ref_max_intens = wake_ref.length_array[np.argmax(wake_ref.wake_luminosity_profile)]
    simlen_ref_filter = np.abs(wake_ref.length_array - simlen_ref_max_intens) < peak_region
    sim_area_ref = np.trapz(wake_ref.wake_luminosity_profile[simlen_ref_filter], wake_ref.length_array[simlen_ref_filter])

    # Normalize the observed wake intensity
    wake_ref_intensity_array = wake_ref_intensity_array*sim_area_ref/obs_area_ref



    ### Align the observed and simulated wakes by correlation ###
    
    # Interpolate the model values and sample them at observed points
    sim_wake_interp = scipy.interpolate.interp1d(wake_ref.length_array, \
        wake_ref.wake_luminosity_profile, bounds_error=False, fill_value=0)
    model_wake_obs_len_sample = sim_wake_interp(-len_ref_array)

    # Correlate the wakes and find the shift
    wake_shift = np.argmax(np.correlate(model_wake_obs_len_sample, wake_ref_intensity_array, \
        "full")) + 1

    # Find the index of the zero observed length
    obs_len_zero_indx = np.argmin(np.abs(len_ref_array))

    # Compute the length shift
    len_shift = len_ref_array[(obs_len_zero_indx + wake_shift)%len(model_wake_obs_len_sample)]

    # If the shift is larger than the maximum allowed, do not align the wakes
    if np.abs(len_shift) > max_len_shift:
        len_shift = 0

    # Add the offset to the observed length
    len_ref_array += len_shift

    ### ###


    return (
        ht_ref, # Return the reference height
        wake_ref.length_array, wake_ref.wake_luminosity_profile, # Return the simulated wake at the ref ht
        -len_ref_array, wake_ref_intensity_array # Return the observed wake at the ref ht
    )


def plotWakeOverview(sr, wake_containers, plot_dir, event_name, site_id=None, wake_samples=8,
                     first_height_ratio=0.1, final_height_ratio=0.75, peak_region=20):
    """ Plot the wake at a range of heights showing the match between the observed and simulated wake. 

    Arguments:
        sr: [SimulationResults object] Simulation results.
        wake_containers: [list of WakeContainer objects] List of wake containers.
        plot_dir: [str] Path to the directory where the plots will be saved.
        event_name: [str] Name of the event.

    Keyword arguments:
        site_id: [int] Name of the site where the meteor was observed. 1 for Tavistock and 2 for Elginfield.
            If None, both will be taken.
        wake_samples: [int] Number of wake samples to plot.
        first_height_ratio: [float] Fraction of the wake height to probe for the first sample. 0 is the height
            when tracking began and 1 is the height when the tracking stopped.
        final_height_ratio: [float] Fraction of the wake height to probe for the last sample. 0 is the height
            when tracking began and 1 is the height when the tracking stopped.
        peak_region: [float] Region around the peak to use for the wake normalization (m). If None, the whole
            wake will be used.

    """

    # Make N plots for wake_samples heights
    height_fractions = np.linspace(first_height_ratio, final_height_ratio, wake_samples)

    # Set up the plot
    fig, axes = plt.subplots(figsize=(8, 8), nrows=wake_samples, sharex=True)

    # Length at which text is plotted
    txt_len_coord = 50 # m

    # Loop through the heights
    for i, height_fraction in enumerate(height_fractions):
            
        # Get the wake results
        (
            ht_ref, # Return the reference height
            wake_len_array, wake_lum_array, # Return the simulated wake at the ref ht
            obs_len_array, obs_lum_array # Return the observed wake at the ref ht
        ) = extractWake(sr, wake_containers, wake_fraction=height_fraction, site_id=site_id, 
                        peak_region=peak_region)

        # Plot the observed wake
        axes[i].plot(obs_len_array, obs_lum_array, color="black", linestyle="--", linewidth=1, alpha=0.75)

        # Plot the simulated wake
        axes[i].plot(wake_len_array, wake_lum_array, color="black", linestyle="solid", linewidth=1, alpha=0.75)

        # Get the height label as halfway between the peak model wake and 0
        txt_ht = np.max(wake_lum_array)/2

        # Set the height label
        axes[i].text(txt_len_coord, txt_ht, "{:.1f} km".format(ht_ref/1000), fontsize=8, ha="right", va="center")

    # Remove Y ticks on all axes
    for ax in axes:
        ax.set_yticks([])

    # Set the X label
    axes[-1].set_xlabel("Length (m)", fontsize=12)

    # Set X axis limits
    axes[-1].set_xlim(-200, 80)

    # Invert X axis
    axes[-1].invert_xaxis()

    # Remove vertical space between subplots
    plt.subplots_adjust(hspace=0)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Save the plot
    plt.savefig(os.path.join(plot_dir, "{:s}_wake_overview.png".format(event_name)), dpi=300, 
                bbox_inches="tight")

    # Close the plot
    plt.close(fig)



def plotObsAndSimComparison(traj, sr, met_obs, plot_dir, wake_containers=None, plot_lag=False, camo=False):
    """ Plot a comparison between observed and simulated data. Plot the light curve and velocity comparison
        and wake if given.

        TO DO:
            - Include arbitrary light curves and USG data.
            - Show LCs of individual fragments.

    Arguments:
        traj: [Trajectory object] Trajectory object.
        sr: [SimulationResults object] Simulation results.
        met_obs: [MetObservations object] Meteor observations.
        plot_dir: [str] Path to the directory where the plots will be saved.

    Keyword arguments:
        wake_containers: [list of WakeContainer objects] List of wake containers.
        plot_lag: [bool] If True, plot the lag comparison instead of length residuals.
        camo: [bool] Should be True if CAMO mirror tracking data is used and False otherwise.
    """

    # Define plot properties for CAMO (wake_containers are given or "camo" is True)
    if (wake_containers is not None) or camo:

        # Set the camo varaible to True if any wake is given, so the correct plot parameters are used
        camo = True

        plot_params_dict = {
            "sites-narrow":
                {   
                    # Blue empty circles for Tavistock
                    "1": {"color": "blue", "marker": "o", "markerfacecolor": "none", "markersize": 5, 
                          "linestyle": "", "label": "NF - Tavistock"},
                    # Green empty squares for Elginfield
                    "2": {"color": "green", "marker": "s", "markerfacecolor": "none", "markersize": 5, 
                          "linestyle": "", "label": "NF - Elginfield"},
                },
            "sites-wide":
                {
                    # Small blue full circles for Tavistock
                    "1": {"color": "blue", "marker": "o", "markersize": 3, 
                          "linestyle": "", "label": "WF - Tavistock"},
                    # Small green full squares for Elginfield
                    "2": {"color": "green", "marker": "s", "markersize": 3, 
                          "linestyle": "", "label": "WF - Elginfield"},
                },
            # Black line for simulated data (no marker)
            "sim": {"color": "black", "marker": None, "linewidth": 1, "label": "Simulated"}
        }

    # For other data, define different plot parameters
    else:
        
        plot_params_dict = {
            # Black line for simulated data (no marker)
            "sim": {"color": "black", "marker": None, "linewidth": 1, "label": "Simulated"}
        }

        # Generate a unique list of station codes using the trajectory and .met file
        station_codes  = [str(obs.station_id) for obs in traj.observations]
        if met_obs is not None:
            station_codes += [str(site) for site in met_obs.sites]

        station_codes = list(set(station_codes))

        # For each station, generate plot parameters with a unique color and marker
        markers = ["o", "o", "s", "s", "v", "v", "D", "D", "X", "+", "*"]
        colors = ["blue", "green", "red", "orange", "purple", "brown", "pink", "cyan"]
        for i, station_code in enumerate(station_codes):
            
            # Get the color
            color = colors[i%len(colors)]

            # Get the marker
            marker = markers[i%len(markers)]

            # Alternate between empty and full markers
            if (i%len(colors) == 0) and (marker not in ["X", "+", "*"]):
                markerfacecolor = "none"
            else:
                markerfacecolor = color

            # Define the plot parameters
            plot_params_dict[station_code] = {
                "color": color, "marker": marker, "markerfacecolor": markerfacecolor,
                "markersize": 3, "linestyle": "", "label": station_code
            }



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
    
    # Interpolate the simulated height by time
    sim_norm_ht_interp = scipy.interpolate.interp1d(norm_sim_time, norm_sim_ht,
                                                    bounds_error=False, fill_value='extrapolate')



    #fig, (ax_mag, ax_magres, ax_lag, ax_lenres) = plt.subplots(ncols=4, sharey=True)
    #fig, (ax_mag, ax_lenres) = plt.subplots(ncols=2, sharey=True)

    # Make four subplots in one row. Two for magnitude and two for length residuals. The other
    # two are for wake1 and wake2. These should be in the third panel to the right, but one on top of 
    # the other. Use gridspec.

    # If the wake is shown, add the thried column for the wake plots
    if wake_containers is not None:
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(nrows=2, ncols=3, width_ratios=[1, 1, 1], height_ratios=[1, 1])
        ax_mag = fig.add_subplot(gs[:, 0])
        ax_lenres = fig.add_subplot(gs[:, 1], sharey=ax_mag)
        ax_wake1 = fig.add_subplot(gs[0, 2])
        ax_wake2 = fig.add_subplot(gs[1, 2], sharex=ax_wake1)
    else:

        # Just have a two column plot with magnitudes and lenres
        fig, (ax_mag, ax_lenres) = plt.subplots(ncols=2, sharey=True, figsize=(8, 6))


    # Compute the magnitude and length residuals from the trajectory object
    ht_obs_max = -np.inf
    ht_obs_min = np.inf
    mag_obs_faintest = -np.inf
    mag_obs_brightest = np.inf
    time_obs_min = np.inf
    time_obs_max = -np.inf
    for obs in traj.observations:

        # Set a default filter which takes all observations
        mag_filter = np.ones(len(obs.model_ht), dtype=bool)

        if obs.absolute_magnitudes is not None:

            # Filter out observations with magnitude fainter than +9 or with NaN magnitudes
            mag_filter = (obs.absolute_magnitudes < 9) & (~np.isnan(obs.absolute_magnitudes))

            # # Sample the simulated magnitude at the observation heights
            # sim_mag_sampled = sim_mag_interp(obs.model_ht[mag_filter])

        # Sample the simulated normalized length at observed times
        sim_norm_len_sampled = sim_norm_len_interp(obs.time_data[mag_filter])


        # Select plot parameters for the current station
        if camo:
            plot_params = plot_params_dict["sites-narrow"][str(obs.station_id)]
        else:
            plot_params = plot_params_dict[str(obs.station_id)]


        if obs.absolute_magnitudes is not None:

            # Plot the observations
            ax_mag.plot(obs.absolute_magnitudes[mag_filter], obs.model_ht[mag_filter]/1000,
                        **plot_params)
            
            # # Plot the simulated magnitude
            # ax_mag.plot(sim_mag_sampled, obs.model_ht[mag_filter]/1000, 
            #             **plot_params_dict["sim"])

            # ax_magres.scatter(obs.absolute_magnitudes[mag_filter] - sim_mag_sampled,
            #                 obs.model_ht[mag_filter]/1000)

            # Keep track of the observed magnitude range
            mag_obs_faintest = max(mag_obs_faintest, np.max(obs.absolute_magnitudes[mag_filter]))
            mag_obs_brightest = min(mag_obs_brightest, np.min(obs.absolute_magnitudes[mag_filter]))

        # ax_len.scatter(obs.state_vect_dist[mag_filter], obs.model_ht[mag_filter]/1000,
        #   label=obs.station_id)
        # ax_len.plot(sim_norm_len_sampled, obs.model_ht[mag_filter]/1000)


        if plot_lag:
            
            # # Compute the simulated lag
            # sim_lag = sim_norm_len_sampled - sim_vel_beg*obs.time_data[mag_filter]

            # Compute the observed lag using the simulated velocity
            obs_lag = obs.state_vect_dist[mag_filter] - sim_vel_beg*obs.time_data[mag_filter]

            # Plot the observed lag
            ax_lenres.plot(obs_lag, obs.model_ht[mag_filter]/1000, **plot_params)
            
            # ax_lenres.plot(sim_lag, obs.model_ht[mag_filter]/1000, **plot_params_dict["sim"])

        else:

            # Compute the length residuals
            len_residuals = obs.state_vect_dist[mag_filter] - sim_norm_len_sampled

            # Demean the residuals
            len_residuals = len_residuals - np.median(len_residuals)

            # Plot the length residuals
            ax_lenres.plot(len_residuals, obs.model_ht[mag_filter]/1000, **plot_params)


        # Keep track of the observed height range
        ht_obs_max = max(ht_obs_max, np.max(obs.model_ht[mag_filter]))
        ht_obs_min = min(ht_obs_min, np.min(obs.model_ht[mag_filter]))

        # Keep track of the observed time range
        time_obs_max = max(time_obs_max, np.max(obs.time_data[mag_filter]))
        time_obs_min = min(time_obs_min, np.min(obs.time_data[mag_filter]))



    # Compute magnitude residuals from the .met file
    if met_obs is not None:

        # If 2 and 4 are the sites, rename them to 1 and 2
        if camo and ("2" in met_obs.sites) and ("4" in met_obs.sites):

            # Rename the list entries to 1 and 2
            temp_sites = []
            for site in met_obs.sites:
                if site == "2":
                    temp_sites.append("1")
                elif site == "4":
                    temp_sites.append("2")
                else:
                    temp_sites.append(site)
            
            # Replace the sites list
            met_obs.sites = temp_sites

            # Swap keys in height_data and abs_mag_data
            temp_height_data = met_obs.height_data["2"]
            temp_abs_mag_data = met_obs.abs_mag_data["2"]
            met_obs.height_data["2"] = met_obs.height_data["4"]
            met_obs.abs_mag_data["2"] = met_obs.abs_mag_data["4"]
            met_obs.height_data["1"] = temp_height_data
            met_obs.abs_mag_data["1"] = temp_abs_mag_data


        # Plot magnitudes for all sites
        for site in met_obs.sites:

            # Extract data (filter out inf values)
            height_data = met_obs.height_data[site][~np.isinf(met_obs.abs_mag_data[site])]
            abs_mag_data = met_obs.abs_mag_data[site][~np.isinf(met_obs.abs_mag_data[site])]

            # # Sample the simulated magnitude at the observation heights
            # sim_mag_sampled = sim_mag_interp(height_data)

            # Select plot parameters for the current station
            if camo:
                plot_params = plot_params_dict["sites-wide"][str(site)]
            else:
                plot_params = plot_params_dict[str(site)]

            # Plot the observed and simulated magnitudes
            ax_mag.plot(abs_mag_data, height_data/1000, **plot_params)
            # ax_mag.plot(sim_mag_sampled, height_data/1000, **plot_params_dict["sim"])

            # ax_magres.scatter(abs_mag_data - sim_mag_sampled, height_data/1000)

            # Keep track of the observed height range
            ht_obs_max = max(ht_obs_max, np.max(height_data))
            ht_obs_min = min(ht_obs_min, np.min(height_data))

            # Keep track of the observed magnitude range
            mag_obs_faintest = max(mag_obs_faintest, np.max(abs_mag_data))
            mag_obs_brightest = min(mag_obs_brightest, np.min(abs_mag_data))

            # Keep track of the observed time range
            time_obs_max = max(time_obs_max, np.max(met_obs.time_data[site]))
            time_obs_min = min(time_obs_min, np.min(met_obs.time_data[site]))


    ### Plot the simulated magnitude ###

    # Plot the simulated magnitude +/- 2 km from the observed height range
    ht_sim_min = ht_obs_min - 2000
    ht_sim_max = ht_obs_max + 2000
    ht_sim_arr = np.linspace(ht_sim_min, ht_sim_max, 1000)
    mag_sim_arr = sim_mag_interp(ht_sim_arr)

    # Limit the plotted simulated magnitude to +/- 2 mag from the observed magnitude range
    mag_sim_min = mag_obs_brightest - 2
    mag_sim_max = mag_obs_faintest + 2
    mag_range_filter = (mag_sim_arr > mag_sim_min) & (mag_sim_arr < mag_sim_max)
    ht_sim_arr = ht_sim_arr[mag_range_filter]
    mag_sim_arr = mag_sim_arr[mag_range_filter]

    # Plot the simulated magnitude
    ax_mag.plot(mag_sim_arr, ht_sim_arr/1000, **plot_params_dict["sim"])

    ### ###

    # If lag is plotted, sample the simulated length between the observed time range
    if plot_lag:

        time_sim_arr = np.linspace(time_obs_min, time_obs_max, 1000)
        len_sim_arr = sim_norm_len_interp(time_sim_arr)
        ht_sim_arr = sim_norm_ht_interp(time_sim_arr)

        # Compute the lag
        lag_sim_arr = len_sim_arr - sim_vel_beg*time_sim_arr

        # Plot the simulated lag
        ax_lenres.plot(lag_sim_arr, ht_sim_arr/1000, **plot_params_dict["sim"])



    
    ### Plot the wake ###
    if wake_containers is not None:

        # Determine where the plots along the range of wake heights should be made (at the beginning 
        # and the middle, 0 and 0.5)
        wake_1_fraction = 0
        wake_2_fraction = 0.5

        # Extract the observed and simulated wakes at the given reference heights
        (
            ht_top1,
            sim_len_top1, sim_wake_top1, 
            obs_len_top1, obs_wake_top1
        ) = extractWake(sr, wake_containers, wake_fraction=wake_1_fraction, peak_region=20)
        (
            ht_btm2,
            sim_len_btm2, sim_wake_btm2, 
            obs_len_btm2, obs_wake_btm2
        ) = extractWake(sr, wake_containers, wake_fraction=wake_2_fraction, peak_region=20)


        # Plot the simulated wakes
        ax_wake1.plot(sim_len_top1, sim_wake_top1, label='Simulated', color='k')
        ax_wake2.plot(sim_len_btm2, sim_wake_btm2, color='k')

        # Plot the observed wakes
        ax_wake1.plot(obs_len_top1, obs_wake_top1, label='Observed', color='k', linestyle='--')
        ax_wake2.plot(obs_len_btm2, obs_wake_btm2, color='k', linestyle='--')
        
        ax_wake1.legend(loc='upper right', fontsize=12)
        
        # Add horizontal lines to magnitude and lenres plots to show the top1% and btm2% limits of heights
        ax_mag.axhline(y=ht_top1/1000, color='k', linestyle='dashed', linewidth=1)
        ax_mag.axhline(y=ht_btm2/1000, color='k', linestyle='dotted', linewidth=1)
        ax_lenres.axhline(y=ht_top1/1000, color='k', linestyle='dashed', linewidth=1)
        ax_lenres.axhline(y=ht_btm2/1000, color='k', linestyle='dotted', linewidth=1)

        # Add the text labels above the horizontal lines indicating the heights of the wakes
        axmag_min, axmag_max = ax_mag.get_xlim()
        ax_mag.text(axmag_max - 0.05, ht_top1/1000 + 0.1, "Wake at {:.2f} km".format(ht_top1/1000), size=8)
        ax_mag.text(axmag_max - 0.05, ht_btm2/1000 + 0.1, "Wake at {:.2f} km".format(ht_btm2/1000), size=8)

        # Add titles to the wake plots indicating the height
        ax_wake1.set_title("Height = {:.2f} km".format(ht_top1/1000))
        ax_wake2.set_title("Height = {:.2f} km".format(ht_btm2/1000))


        # invert x axis
        ax_wake1.invert_xaxis()

        # Set labels
        ax_wake1.set_ylabel("Intensity")
        ax_wake2.set_xlabel("Distance behind leading fragment (m)")
        ax_wake2.set_ylabel("Intensity")
        
        ###


    # # Add a title with the meteor number and timestamp
    # met_timestamp = jd2Date(traj.jdt_ref, dt_obj=True).strftime("%Y-%m-%d %H:%M:%S")
    # #fig.suptitle("#{} - {}".format(met_num, met_timestamp))
    # fig.suptitle("{} - {}".format(formatDirNameToEventName(entry), met_timestamp))

    ax_mag.set_ylabel("Height (km)")
    ax_mag.set_xlabel("Magnitude")

    #ax_len.set_xlabel("Length (m)")
    
    if plot_lag:
        ax_lenres.set_xlabel("Lag (m)")
    else:
        ax_lenres.set_xlabel("Length residuals (m)")

    # Plot a vertical line at 0 on the length residual plot
    ax_lenres.axvline(0, color='0.5', linewidth=1, linestyle='dashed')

    # Invert magnitude axes
    ax_mag.invert_xaxis()
    #ax_magres.invert_xaxis()

    # # Invert length axes
    # ax_len.invert_yaxis()
    # ax_lenres.invert_yaxis()
    # ax_lag.invert_yaxis()

    plt.tight_layout()

    # Remove current legent entries and make new ones based on the plot_params_dict
    if camo:

        ax_mag.legend().remove()
        handles = []

        handles.append(mlines.Line2D([0], [0], **plot_params_dict["sim"]))
        for site_id in plot_params_dict["sites-wide"]:
            handles.append(mlines.Line2D([0], [0], **plot_params_dict["sites-wide"][site_id]))
        for site_id in plot_params_dict["sites-narrow"]:
            handles.append(mlines.Line2D([0], [0], **plot_params_dict["sites-narrow"][site_id]))
        
        # Add the legend
        ax_mag.legend(handles=handles, fontsize=8)

    else:
        ax_mag.legend(fontsize=8)




    # Save the figure
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Generate a plot name using the timestamp
    met_timestamp = jd2Date(traj.jdt_ref, dt_obj=True).strftime("%Y%m%d_%H%M%S")
    plot_name = "{:s}_metsim_fit".format(met_timestamp)

    plt.savefig(os.path.join(plot_dir, plot_name + ".png"), dpi=300)

    # plt.show()

    plt.close(fig)


class MetSimGUI(QMainWindow):
    def __init__(self, traj_path, const_json_file=None, met_path=None, lc_path=None, wid_files=None, \
        usg_input=False):
        """ GUI tool for MetSim. 
    
        Arguments:
            traj_path: [str] Path to the trajectory pickle file.

        Keyword arguments:
            const_json_file: [str] Path to the JSON file with simulation parameters.
            met_path: [str] Path to the METAL or mirfit .met file with additional magnitude or lag information.
            lc_path: [str] Path to the light curve CSV file.
            wid_files: [str] Mirfit wid files containing the meteor wake information.
            usg_input: [bool] Instead of a trajectory pickle file, a special input file for CNEOS data is 
                given.
        """
        

        self.traj_path = traj_path

        self.usg_data = None
        if not usg_input:

            # Load the trajectory pickle file
            self.traj = loadPickle(*os.path.split(traj_path))

        else:

            # Load the trajectory from USG input file
            self.usg_data, self.traj = loadUSGInputFile(*os.path.split(traj_path))


        # Extract the directory path
        self.dir_path = os.path.dirname(traj_path)


        ### LOAD .met FILE ###

        # Load a METAL .met file if given
        self.met = None
        if met_path is not None:
            if os.path.isfile(met_path):
                print()
                print("Loading .met file: {:s}".format(met_path))
                self.met = loadMet(*os.path.split(os.path.abspath(met_path)))
            else:
                print('The .met file does not exist:', met_path)
                sys.exit()

        # Init a container for .met results, compute lag and magnitude
        if self.met is not None:
            self.met_obs = MetObservations(self.met, self.traj)
        else:
            self.met_obs = None

        ### ###



        ### LOAD the light curve file ###
        self.lc_data = None
        if lc_path is not None:
            if os.path.isfile(lc_path):

                # Read the light curve data
                self.lc_data = LightCurveContainer(*os.path.split(os.path.abspath(lc_path)))

            else:
                print("The light curve file does not exist:", lc_path)
                sys.exit()


        ### ###



        ### LOAD WAKE FILES ###

        # Load mirfit wake "wid" file
        self.wake_meas = []
        if wid_files is not None:
            for wid_path in wid_files:
                for wid_file in glob.glob(wid_path):

                    # Load wake information from wid files
                    wake_container = self.loadWakeFile(wid_file)

                    if wake_container is not None:
                        self.wake_meas.append(wake_container)

        
        # Get a list of heights where lag measurements were taken
        self.wake_heights = None
        if self.wake_meas:
            self.wake_heights = []
            for wake_container in self.wake_meas:
                for wake_pt in wake_container.points:
                    if int(wake_pt.n ) == 0:
                        self.wake_heights.append([wake_pt.ht, wake_container])
                        break

            # Sort wake height list by height
            self.wake_heights = sorted(self.wake_heights, key=lambda x: x[0])

        ### ###



        ### Init GUI ###

        QMainWindow.__init__(self)

        # Load the GUI design file
        loadUi(os.path.join(os.path.dirname(__file__), "GUI.ui"), self)

        self.setWindowTitle("MetSim")

        ### ###


        ### ### Define GUI and simulation attributes ### ###


        # Init an axis for the electron line density
        self.electronDensityPlot = self.magnitudePlot.canvas.axes.twiny()
        self.electron_density_plot_show = False


        ### Wake parameters ###
        self.wake_on = False
        self.wake_show_mass_bins = False
        self.wake_ht_current_index = 0
        self.current_wake_container = None

        if self.wake_heights is not None:
            self.wake_plot_ht, self.current_wake_container = self.wake_heights[self.wake_ht_current_index]
        else:
            self.wake_plot_ht = self.traj.rbeg_ele # m

        self.wake_normalization_method = 'area'
        self.wake_align_method = 'none'


        self.magnitudePlotWakeLines = None
        self.magnitudePlotWakeLineLabels = None
        self.velocityPlotWakeLines = None
        self.lagPlotWakeLines = None

        ### ###


        ### Autofit parameters ###


        self.autofit_method = "Local"
        self.autoFitMethodToggle(self.autofit_method)

        self.pso_iterations = 10
        self.pso_particles = 100

        self.autofit_mag_weight = 0.1
        self.autofit_lag_weight_ht_change = self.traj.rend_ele - 10000
        self.autofit_lag_weights = [1.0, 1.0]

        ### ###
    

        # Disable different density after erosion change
        self.erosion_different_rho = False

        # Disable different ablation coeff after erosion change
        self.erosion_different_sigma = False

        # Disable different erosion coeff after disruption at the beginning
        self.disruption_different_erosion_coeff = False


        # Fragmentation object
        self.fragmentation = None


        self.simulation_results = None

        self.const_prev = None
        self.simulation_results_prev = None


        ### ### ### ###


        ### Init simulation parameters ###

        # Init the constants
        self.const = Constants()

        # Assign USG values from the input file, if given
        if self.usg_data is not None:

            self.const.P_0m = self.usg_data.P_0m_bolo


        # If a JSON file with constant was given, load them instead of initing from scratch
        if const_json_file is not None:

            # Load the constants from the JSON files
            self.const, const_json = loadConstants(const_json_file)


            # Init the fragmentation container for the GUI
            if len(self.const.fragmentation_entries):
            
                self.fragmentation = FragmentationContainer(self, \
                    os.path.join(self.dir_path, self.const.fragmentation_file_name))
                self.fragmentation.fragmentation_entries = self.const.fragmentation_entries

                # Overwrite the existing fragmentatinon file
                self.fragmentation.writeFragmentationFile()




            # Check if the disruption erosion coefficient is different than the main erosion coeff
            if const_json['disruption_erosion_coeff'] != const_json['erosion_coeff']:
                self.disruption_different_erosion_coeff = True


            # Check if the density is changed after Hchange
            if 'erosion_rho_change' in const_json:
                if const_json['erosion_rho_change'] != const_json['rho']:
                    self.erosion_different_rho = True

            # Check if the ablation coeff is changed after Hchange
            if 'erosion_sigma_change' in const_json:
                if const_json['erosion_sigma_change'] != const_json['sigma']:
                    self.erosion_different_sigma = True

            # if 'fragmentation_string' in const_json:
            #     if const_json['fragmentation_string'] is not None:
            #         self.fragmentation = FragmentationContainer(self, os.path.join(self.dir_path, \
            #             FRAG_FILE_NAME))
            #         self.fragmentation.loadFromString(const_json['fragmentation_string'])
            #         self.fragmentation.writeFragmentationFile()


            # # Convert the density coefficients into a numpy array
            # self.const.dens_co = np.array(self.const.dens_co)


        else:

            # Compute the radius of the Earth at the latitude of the observer
            self.const.r_earth = \
                EARTH.EQUATORIAL_RADIUS/np.sqrt(1.0 - (EARTH.E**2)*np.sin(self.traj.rbeg_lat)**2)

            # Set the constants value from the trajectory
            self.const.v_init = self.traj.orbit.v_init

            # Set kill height to the observed end height
            self.const.h_kill = self.traj.rend_ele - 3000

            # Set erosion heights to the beginning/end height
            self.const.erosion_height_start = self.traj.rbeg_ele
            self.const.erosion_height_change = self.traj.rend_ele

            # Disable erosion and disruption at the beginning
            self.const.erosion_on = False
            self.const.disruption_on = False

            # Compute the zenith angle at the beginning of the simulation, taking Earth's curvature into 
            # account
            self.computeSimZenithAngle()

            # Calculate the photometric mass
            _, self.const.m_init = self.calcPhotometricMass()
            print("Using initial mass: {:.2e} kg".format(self.const.m_init))


        ### ###



        ### Calculate atmosphere density coeffs (down to the bottom observed height, limit to 15 km) ###

        # Determine the height range for fitting the density
        self.dens_fit_ht_beg = self.const.h_init
        self.dens_fit_ht_end = self.traj.rend_ele - 5000
        if self.dens_fit_ht_end < 14000:
            self.dens_fit_ht_end = 14000

        # Fit the polynomail describing the density
        dens_co = self.fitAtmosphereDensity(self.dens_fit_ht_beg, self.dens_fit_ht_end)
        self.const.dens_co = dens_co

        print("Atmospheric mass density fit for the range of heights: {:.2f} - {:.2f} km".format(\
            self.dens_fit_ht_end/1000, self.dens_fit_ht_beg/1000))

        ### ###



        # Update the values in the input boxes
        self.updateInputBoxes()



        ### Add key bindings ###

        # Update shown grain diameters when the grain mass is updated
        self.inputErosionMassMin.editingFinished.connect(self.updateGrainDiameters)
        self.inputErosionMassMax.editingFinished.connect(self.updateGrainDiameters)

        # Update the photometric mass when the luminous efficiency is changed
        self.inputLumEff.editingFinished.connect(self.updateInitialMass)

        # Update boxes when the luminous efficiency type is changes
        self.lumEffComboBox.currentIndexChanged.connect(self.updateLumEffType)

        self.wakePlotUpdateButton.clicked.connect(self.updateWakePlot)
        self.wakeIncrementPlotHeightButton.clicked.connect(self.incrementWakePlotHeight)
        self.wakeDecrementPlotHeightButton.clicked.connect(self.decrementWakePlotHeight)
        self.wakeSaveVideoButton.clicked.connect(self.saveVideo)

        self.radioButtonWakeNormalizationPeak.toggled.connect(self.toggleWakeNormalizationMethod)
        self.radioButtonWakeNormalizationArea.toggled.connect(self.toggleWakeNormalizationMethod)
        self.radioButtonWakeNormalizationArea.setChecked(self.wake_normalization_method == 'area')
        self.radioButtonWakeNormalizationPeak.setChecked(self.wake_normalization_method == 'peak')
        
        self.radioButtonWakeAlignNone.toggled.connect(self.toggleWakeAlignMethod)
        self.radioButtonWakeAlignPeak.toggled.connect(self.toggleWakeAlignMethod)
        self.radioButtonWakeAlignCorrelate.toggled.connect(self.toggleWakeAlignMethod)
        self.radioButtonWakeAlignNone.setChecked(self.wake_align_method == 'none')
        self.radioButtonWakeAlignPeak.setChecked(self.wake_align_method == 'peak')
        self.radioButtonWakeAlignCorrelate.setChecked(self.wake_align_method == 'correlate')

        #self.addToolBar(NavigationToolbar(self.magnitudePlot.canvas, self))


        self.autoFitMethodComboBox.activated[str].connect(self.autoFitMethodToggle)


        self.checkBoxWake.stateChanged.connect(self.checkBoxWakeSignal)
        self.checkBoxWakeMassBins.stateChanged.connect(self.checkBoxWakeMassBinsSignal)
        self.checkBoxErosion.stateChanged.connect(self.checkBoxErosionSignal)
        self.checkBoxErosionRhoChange.stateChanged.connect(self.checkBoxErosionRhoSignal)
        self.checkBoxErosionAblationCoeffChange.stateChanged.connect(self.checkBoxErosionAblationCoeffSignal)
        self.checkBoxDisruption.stateChanged.connect(self.checkBoxDisruptionSignal)
        self.checkBoxDisruptionErosionCoeff.stateChanged.connect(self.checkBoxDisruptionErosionCoeffSignal)


        self.runSimButton.clicked.connect(self.runSimulationGUI)
        self.autoFitButton.clicked.connect(self.autoFit)


        self.fragmentationGroup.toggled.connect(self.toggleFragmentation)
        self.newFragmentationFileButton.clicked.connect(self.newFragmentationFile)
        self.checkBoxFragmentationShowIndividualLCs.clicked.connect(self.checkBoxFragmentationShowIndividualLCsSignal)
        self.loadFragmentationFileButton.clicked.connect(self.loadFragmentationFile)
        self.mainFragmentStatusChangeButton.clicked.connect(lambda x: self.addFragmentation("M"))
        self.allFragmentsStatusChangeButton.clicked.connect(lambda x: self.addFragmentation("A"))
        self.newSingleBodyFragmentButton.clicked.connect(lambda x: self.addFragmentation("F"))
        self.newErodingFragmentButton.clicked.connect(lambda x: self.addFragmentation("EF"))
        self.newDustReleaseButton.clicked.connect(lambda x: self.addFragmentation("D"))

        
        self.showPreviousButton.pressed.connect(self.showPreviousResults)
        self.showPreviousButton.released.connect(self.showCurrentResults)

        self.saveUpdatedOrbitButton.clicked.connect(self.saveUpdatedOrbit)
        self.saveFitParametersButton.clicked.connect(self.saveFitParameters)

        # Electron line density
        self.checkBoxPlotElectronLineDensity.toggled.connect(self.checkBoxPlotElectronLineDensitySignal)
        self.inputElectronDensityHt.editingFinished.connect(self.electronDensityMeasChanged)
        self.inputElectronDensity.editingFinished.connect(self.electronDensityMeasChanged)

        # Plots
        self.plotAirDensityButton.clicked.connect(self.plotAtmDensity)
        self.plotDynamicPressureButton.clicked.connect(self.plotDynamicPressure)
        self.plotMassLossButton.clicked.connect(self.plotMassLoss)
        self.plotObsVsSimComparisonButton.clicked.connect(self.plotObsVsSimComparison)
        
        self.plotLumEffButton.clicked.connect(self.plotLumEfficiency)
        self.plotLumEffButton.setDisabled(True)


        ### ###

        # Update checkboxes
        self.checkBoxWakeSignal(None)
        self.checkBoxErosionSignal(None)
        self.checkBoxErosionRhoSignal(None)
        self.checkBoxErosionAblationCoeffSignal(None)
        self.checkBoxDisruptionSignal(None)
        self.checkBoxDisruptionErosionCoeffSignal(None)
        self.checkBoxDisruptionErosionCoeffSignal(None)
        self.toggleWakeNormalizationMethod(None)
        self.toggleWakeAlignMethod(None)
        self.toggleFragmentation(None)


        # Compute plot height limits
        self.plot_beg_ht, self.plot_end_ht = self.calcHeightLimits()

        # Update plots
        self.showCurrentResults()



    def fitAtmosphereDensity(self, dens_fit_ht_beg, dens_fit_ht_end):
        """ Fit the atmosphere density coefficients for the given day and location. 
        
        Arguments:
            dens_fit_ht_beg: [float] Begin height (top) for which the fit is valid (meters).
            dens_fit_ht_end: [float] End height - bottom (meters).

        """

        # Take mean meteor lat/lon as reference for the atmosphere model
        lat_mean = np.mean([self.traj.rbeg_lat, self.traj.rend_lat])
        lon_mean = meanAngle([self.traj.rbeg_lon, self.traj.rend_lon])

        return fitAtmPoly(lat_mean, lon_mean, dens_fit_ht_end, dens_fit_ht_beg, self.traj.jdt_ref)
    
    def computeSimZenithAngle(self):
        """ Compute the zenith angle at the beginning of the simulation, taking Earth's curvature into 
        account.
        
        """

        # Compute the zenith angle at the beginning of the simulation
        self.const.zenith_angle = zenithAngleAtSimulationBegin(self.const.h_init, self.traj.rbeg_ele, 
            self.traj.orbit.zc, self.const.r_earth)


    def loadWakeFile(self, file_path):
        """ Load a mirfit wake "wid" file. """

        return loadWakeFile(self.traj, file_path)


    def calcPhotometricMass(self):
        """ Calculate photometric mass from given magnitude data. """

        print()

        # If the magnitudes are given from the met file, use them instead of the trajectory file
        time_mag_arr = []
        avg_t_diff_max = 0
        if self.met_obs is not None:

            print("Photometric mass computed from the .met file!")

            # Extract time vs. magnitudes from the met file
            for site in self.met_obs.sites:

                # Extract data
                abs_mag_data = self.met_obs.abs_mag_data[site]
                time_data = self.met_obs.time_data[site]

                # Compute the average time difference
                avg_t_diff_max = max(avg_t_diff_max, np.median(time_data[1:] - time_data[:-1]))

                for t, mag in zip(time_data, abs_mag_data):
                    if (mag is not None) and (not np.isnan(mag)):
                        time_mag_arr.append([t, mag])

        # If the magnitudes are given
        elif self.lc_data is not None:

            print("Photometric mass computed from the light curve file!")

            # Plot additional magnitudes for all sites
            for site in self.lc_data.sites:

                # Extract data
                abs_mag_data = self.lc_data.abs_mag_data[site]
                time_data = self.lc_data.time_data[site]

                # Compute the average time difference
                avg_t_diff_max_tmp = max(avg_t_diff_max, np.median(time_data[1:] - time_data[:-1]))
                if avg_t_diff_max == 0:
                    avg_t_diff_max = avg_t_diff_max_tmp
                else:
                    avg_t_diff_max = max([avg_t_diff_max, avg_t_diff_max_tmp])

                for t, mag in zip(time_data, abs_mag_data):
                    if (mag is not None) and (not np.isnan(mag)):
                        time_mag_arr.append([t, mag])

        else:

            print("Photometric mass computed from the trajectory pickle file!")

            # Extract time vs. magnitudes from the trajectory pickle file
            for obs in self.traj.observations:

                # If there are not magnitudes for this site, skip it
                if obs.absolute_magnitudes is None:
                    continue

                # Compute average time difference
                avg_t_diff_max = max(avg_t_diff_max, np.median(obs.time_data[1:] - obs.time_data[:-1]))

                for t, mag in zip(obs.time_data[obs.ignore_list == 0], \
                    obs.absolute_magnitudes[obs.ignore_list == 0]):

                    if (mag is not None) and (not np.isnan(mag)) and (not np.isinf(mag)):
                        time_mag_arr.append([t, mag])

        
        # If there are no magnitudes, assume that the initial mass is 0.2 grams
        if not time_mag_arr:
            print("No photometry, assuming default mass:", self.const.m_init)
            return 0, self.const.m_init

        print("NOTE: The mass was computing using a constant luminous efficiency defined in the GUI!")

        # Sort array by time
        time_mag_arr = np.array(sorted(time_mag_arr, key=lambda x: x[0]))

        time_arr, mag_arr = time_mag_arr.T

        
        # Average out the magnitudes
        time_arr, mag_arr = mergeClosePoints(time_arr, mag_arr, avg_t_diff_max, method='avg')


        # Calculate the radiated energy
        radiated_energy = calcRadiatedEnergy(np.array(time_arr), np.array(mag_arr), P_0m=self.const.P_0m)

        # Compute the photometric mass
        photom_mass = calcMass(np.array(time_arr), np.array(mag_arr), self.traj.orbit.v_avg, \
            tau=self.const.lum_eff/100, P_0m=self.const.P_0m)

        
        return radiated_energy, photom_mass



    def updateInputBoxes(self, show_previous=False):
        """ Update input boxes with values from the Constants object. """

        # Choose to show current or previous simulation parameters
        if show_previous and (self.const_prev is not None):
            const = self.const_prev
        else:
            const = self.const

        
        ### Simulation params ###

        self.inputTimeStep.setText(str(const.dt))
        self.inputHtInit.setText("{:.3f}".format(const.h_init/1000))
        self.inputP0M.setText("{:d}".format(int(const.P_0m)))
        self.inputMassKill.setText("{:.1e}".format(const.m_kill))
        self.inputVelKill.setText("{:.3f}".format(const.v_kill/1000))
        self.inputHtKill.setText("{:.3f}".format(const.h_kill/1000))

        ### ###


        ### Meteoroid physical properties ###

        self.inputRho.setText("{:d}".format(int(const.rho)))
        self.inputRhoGrain.setText("{:d}".format(int(const.rho_grain)))
        self.inputMassInit.setText("{:.2e}".format(const.m_init))
        self.inputAblationCoeff.setText("{:.4f}".format(const.sigma*1e6))
        self.inputVelInit.setText("{:.3f}".format(const.v_init/1000))
        self.inputShapeFact.setText("{:.2f}".format(const.shape_factor))
        self.inputGamma.setText("{:.2f}".format(const.gamma))
        self.inputZenithAngle.setText("{:.3f}".format(np.degrees(const.zenith_angle)))
        self.inputLumEff.setText("{:.2f}".format(const.lum_eff))
        
        self.lumEffComboBox.setCurrentIndex(const.lum_eff_type)

        # Enable/disable the constant tau box (index 0 = constant tau)
        self.inputLumEff.setDisabled(self.const.lum_eff_type != 0)

        ### ###


        ### Wake parameters ###

        self.checkBoxWake.setChecked(self.wake_on)
        self.checkBoxWakeMassBins.setChecked(self.wake_show_mass_bins)

        self.inputWakePSF.setText("{:.1f}".format(const.wake_psf))
        self.inputWakeExt.setText("{:d}".format(int(const.wake_extension)))
        self.inputWakePlotHt.setText("{:.3f}".format(self.wake_plot_ht/1000))

        ### ###


        ### Erosion parameters ###

        self.checkBoxErosion.setChecked(const.erosion_on)
        self.checkBoxErosionRhoChange.setChecked(self.erosion_different_rho)
        self.checkBoxErosionAblationCoeffChange.setChecked(self.erosion_different_sigma)
        self.inputErosionHtStart.setText("{:.3f}".format(const.erosion_height_start/1000))
        self.inputBinsPer10m.setText("{:d}".format(const.erosion_bins_per_10mass))
        self.inputErosionCoeff.setText("{:.3f}".format(const.erosion_coeff*1e6))
        self.inputErosionHtChange.setText("{:.3f}".format(const.erosion_height_change/1000))
        self.inputErosionCoeffChange.setText("{:.3f}".format(const.erosion_coeff_change*1e6))
        self.inputErosionMassIndex.setText("{:.2f}".format(const.erosion_mass_index))
        self.inputErosionMassMin.setText("{:.2e}".format(const.erosion_mass_min))
        self.inputErosionMassMax.setText("{:.2e}".format(const.erosion_mass_max))
        self.inputErosionRhoChange.setText("{:d}".format(int(const.erosion_rho_change)))
        self.inputErosionAblationCoeffChange.setText("{:.4f}".format(const.erosion_sigma_change*1e6))

        ### ###


        ### Disruption parameters ###

        self.checkBoxDisruption.setChecked(const.disruption_on)
        self.checkBoxDisruptionErosionCoeff.setChecked(self.disruption_different_erosion_coeff)
        self.inputDisruptionErosionCoeff.setText("{:.3f}".format(const.disruption_erosion_coeff*1e6))
        self.inputCompressiveStrength.setText("{:.1f}".format(const.compressive_strength/1000))
        self.inputDisruptionMassGrainRatio.setText("{:.2f}".format(const.disruption_mass_grain_ratio*100))
        self.inputDisruptionMassIndex.setText("{:.2f}".format(const.disruption_mass_index))
        self.inputDisruptionMassMinRatio.setText("{:.2f}".format(const.disruption_mass_min_ratio*100))
        self.inputDisruptionMassMaxRatio.setText("{:.2f}".format(const.disruption_mass_max_ratio*100))

        ### ###


        ### Autofit parameters ###

        self.inputAutoFitPSOIterations.setText("{:d}".format(self.pso_iterations))
        self.inputAutoFitPSOParticles.setText("{:d}".format(self.pso_particles))

        self.inputAutoFitMagWeight.setText("{:.1f}".format(self.autofit_mag_weight))
        self.inputAutoFitLagWeightHtChange.setText("{:.2f}".format(self.autofit_lag_weight_ht_change/1000))
        self.inputAutoFitLagWeights.setText("{:.1f}, {:.2f}".format(*self.autofit_lag_weights))

        ### ###


        ### Fragmentation parameters ###
        self.fragmentationGroup.setChecked(const.fragmentation_on)
        self.checkBoxFragmentationShowIndividualLCs.setChecked(const.fragmentation_show_individual_lcs)
        self.checkBoxDisruptionErosionCoeff.setChecked(const.fragmentation_show_individual_lcs)

        ###


        ### Radar measurements ###

        # Height of electron line density measurement
        self.inputElectronDensityHt.setText("{:.3f}".format(self.const.electron_density_meas_ht/1000))

        # Electron line density measurement
        self.inputElectronDensity.setText("{:.2e}".format(self.const.electron_density_meas_q))

        ###


        self.updateGrainDiameters()


    def updateInitialMass(self):
        """ Update the value of the photometric mass. """

        # Keep the old luminous efficiency
        lum_eff_old = self.const.lum_eff

        # Read the new value
        self.const.lum_eff = self._tryReadBox(self.inputLumEff, self.const.lum_eff)

            
        # Rescale the initial mass using the new tau value
        self.const.m_init *= lum_eff_old/self.const.lum_eff


        self.updateInputBoxes()


    def updateLumEffType(self):
        """ Update the luminous efficiency type. """

        self.const.lum_eff_type = self.lumEffComboBox.currentIndex()

        # Enable/disable the constant tau box (index 0 = constant tau)
        self.inputLumEff.setDisabled(self.const.lum_eff_type != 0)


    def updateGrainDiameters(self):
        """ Update the grain diameter labels. """

        # Read minimum and maximum grain masses
        self.const.erosion_mass_min = self._tryReadBox(self.inputErosionMassMin, \
            self.const.erosion_mass_min)
        self.const.erosion_mass_max = self._tryReadBox(self.inputErosionMassMax, \
            self.const.erosion_mass_max)


        # Compute minimum grain diameter (m)
        grain_diam_min = 2*(3/(4.0*np.pi)*self.const.erosion_mass_min/self.const.rho_grain)**(1/3.0)

        # Update diameter label
        self.erosionMinMassDiameter.setText("""d<span style="vertical-align:sub;">min</span> = """ \
            + "{:.0f}".format(1e6*grain_diam_min))


        # Compute maximum grain diameter (m)
        grain_diam_max = 2*(3/(4.0*np.pi)*self.const.erosion_mass_max/self.const.rho_grain)**(1/3.0)

        # Update diameter label
        self.erosionMaxMassDiameter.setText("""d<span style="vertical-align:sub;">max</span> = """ \
            + "{:.0f}".format(1e6*grain_diam_max))




    def checkBoxWakeSignal(self, event):
        """ Control what happens when the wake checkbox is toggled. """

        # Read the wake checkbox
        self.wake_on = self.checkBoxWake.isChecked()

        # Read the mass bins show checkbox
        self.wake_show_mass_bins = self.checkBoxWakeMassBins.isChecked()

        # Disable/enable inputs if the checkbox is checked/unchecked
        self.checkBoxWakeMassBins.setDisabled(not self.wake_on)
        self.inputWakePlotHt.setDisabled(not self.wake_on)
        self.inputWakePSF.setDisabled(not self.wake_on)
        self.inputWakeExt.setDisabled(not self.wake_on)
        self.radioButtonWakeNormalizationPeak.setDisabled(not self.wake_on)
        self.radioButtonWakeNormalizationArea.setDisabled(not self.wake_on)
        self.wakeSaveVideoButton.setDisabled(not self.wake_on)
        self.radioButtonWakeAlignNone.setDisabled(not self.wake_on)
        self.radioButtonWakeAlignPeak.setDisabled(not self.wake_on)
        self.radioButtonWakeAlignCorrelate.setDisabled(not self.wake_on)

        # Read inputs
        self.readInputBoxes()


    def checkBoxWakeMassBinsSignal(self, event):
        """ Control what happens when the wake checkbox for showing mass bins is toggled. """

        # Read the mass bins show checkbox
        self.wake_show_mass_bins = self.checkBoxWakeMassBins.isChecked()

        # Update the wake plot
        self.updateWakePlot()



    def checkBoxErosionSignal(self, event):
        """ Control what happens when the erosion checkbox is pressed. """

        # Read the wake checkbox
        self.const.erosion_on = self.checkBoxErosion.isChecked()

        # Disable/enable inputs if the checkbox is checked/unchecked
        self.inputErosionHtStart.setDisabled(not self.const.erosion_on)
        self.inputErosionCoeff.setDisabled(not self.const.erosion_on)
        self.inputErosionHtChange.setDisabled(not self.const.erosion_on)
        self.inputErosionCoeffChange.setDisabled(not self.const.erosion_on)
        self.inputErosionRhoChange.setDisabled(not self.const.erosion_on)
        self.inputErosionAblationCoeffChange.setDisabled(not self.const.erosion_on)
        self.inputErosionMassIndex.setDisabled(not self.const.erosion_on)
        self.inputErosionMassMin.setDisabled(not self.const.erosion_on)
        self.inputErosionMassMax.setDisabled(not self.const.erosion_on)

        self.checkBoxErosionRhoSignal(None)
        self.checkBoxErosionAblationCoeffSignal(None)

        # Read inputs
        self.readInputBoxes()



    def checkBoxErosionRhoSignal(self, event):
        """ Use a different erosion coefficient after disruption. """

        self.erosion_different_rho = self.checkBoxErosionRhoChange.isChecked()

        # Disable/enable different density coefficient checkbox
        self.inputErosionRhoChange.setDisabled((not self.erosion_different_rho) \
            or (not self.const.erosion_on))

        # Read inputs
        self.readInputBoxes()


    def checkBoxErosionAblationCoeffSignal(self, event):
        """ Use a different erosion coefficient after disruption. """

        self.erosion_different_sigma = self.checkBoxErosionAblationCoeffChange.isChecked()

        # Disable/enable different ablation coeff checkbox
        self.inputErosionAblationCoeffChange.setDisabled((not self.erosion_different_sigma) \
            or (not self.const.erosion_on))

        # Read inputs
        self.readInputBoxes()



    def checkBoxDisruptionSignal(self, event):
        """ Control what happens when the disruption checkbox is pressed. """

        # Read the wake checkbox
        self.const.disruption_on = self.checkBoxDisruption.isChecked()

        # Disable/enable inputs if the checkbox is checked/unchecked
        self.checkBoxDisruptionErosionCoeff.setDisabled(not self.const.disruption_on)
        self.inputDisruptionErosionCoeff.setDisabled(not self.const.disruption_on)
        self.inputCompressiveStrength.setDisabled(not self.const.disruption_on)
        self.inputDisruptionMassGrainRatio.setDisabled(not self.const.disruption_on)
        self.inputDisruptionMassIndex.setDisabled(not self.const.disruption_on)
        self.inputDisruptionMassMinRatio.setDisabled(not self.const.disruption_on)
        self.inputDisruptionMassMaxRatio.setDisabled(not self.const.disruption_on)

        self.checkBoxDisruptionErosionCoeffSignal(None)

        # Read inputs
        self.readInputBoxes()


    def checkBoxDisruptionErosionCoeffSignal(self, event):
        """ Use a different erosion coefficient after disruption. """

        self.disruption_different_erosion_coeff = self.checkBoxDisruptionErosionCoeff.isChecked()

        # Disable/enable different erosion coefficient checkbox
        self.inputDisruptionErosionCoeff.setDisabled((not self.disruption_different_erosion_coeff) \
            or (not self.const.disruption_on))

        # Read inputs
        self.readInputBoxes()


    def checkBoxFragmentationShowIndividualLCsSignal(self, event):
        """ Toggle computing light curves of individual fragments during complex fragmentation. """

        self.const.fragmentation_show_individual_lcs = self.checkBoxFragmentationShowIndividualLCs.isChecked()


    def checkBoxPlotElectronLineDensitySignal(self, event):
        """ Toggle plotting the electron line density on the magnitude plot. """

        self.electron_density_plot_show = self.checkBoxPlotElectronLineDensity.isChecked()



    def toggleWakeNormalizationMethod(self, event):
        """ Toggle methods of wake plot normalization. """

        if self.radioButtonWakeNormalizationPeak.isChecked():
            
            # Set the normalization method
            self.wake_normalization_method = 'peak'



        if self.radioButtonWakeNormalizationArea.isChecked():
            
            # Set the normalization method
            self.wake_normalization_method = 'area'



        self.updateWakePlot()



    def toggleWakeAlignMethod(self, event):
        """ Toggle method of wake horizontal alignment. """

        if self.radioButtonWakeAlignNone.isChecked():

            # Set the align methods
            self.wake_align_method = 'none'


        if self.radioButtonWakeAlignPeak.isChecked():

            # Set the align methods
            self.wake_align_method = 'peak'


        if self.radioButtonWakeAlignCorrelate.isChecked():

            # Set the align methods
            self.wake_align_method = 'correlate'


        self.updateWakePlot()



    def _tryReadBox(self, input_box, value, value_type=float):
        try:
            value = value_type(input_box.text())
        except:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Input parsing error")
            msg.setText("Error reading input box " + input_box.objectName())
            msg.setInformativeText("Setting it back to: " + str(value))
            msg.exec_()

        return value



    def readInputBoxes(self):
        """ Read input boxes and set values to the Constants object. """

        
        ### Simulation params ###

        self.const.dt = self._tryReadBox(self.inputTimeStep, self.const.dt)
        self.const.P_0m = self._tryReadBox(self.inputP0M, self.const.P_0m)

        self.const.h_init = 1000*self._tryReadBox(self.inputHtInit, self.const.h_init/1000)
        self.const.m_kill = self._tryReadBox(self.inputMassKill, self.const.m_kill)
        self.const.v_kill = 1000*self._tryReadBox(self.inputVelKill, self.const.v_kill/1000)
        self.const.h_kill = 1000*self._tryReadBox(self.inputHtKill, self.const.h_kill/1000)

        ### ###


        ### Meteoroid physical properties ###

        self.const.rho = self._tryReadBox(self.inputRho, self.const.rho)
        self.const.rho_grain = self._tryReadBox(self.inputRhoGrain, self.const.rho_grain)
        self.const.m_init = self._tryReadBox(self.inputMassInit, self.const.m_init)
        self.const.sigma = self._tryReadBox(self.inputAblationCoeff, self.const.sigma*1e6)/1e6
        self.const.v_init = 1000*self._tryReadBox(self.inputVelInit, self.const.v_init/1000)
        self.const.shape_factor = self._tryReadBox(self.inputShapeFact, self.const.shape_factor)
        self.const.gamma = self._tryReadBox(self.inputGamma, self.const.gamma)
        self.const.zenith_angle = np.radians(self._tryReadBox(self.inputZenithAngle, \
            np.degrees(self.const.zenith_angle)))

        # If the bulk density is higher than the grain density, set the grain density to the bulk denisty
        if self.const.rho > self.const.rho_grain:
            self.const.rho_grain = self.const.rho


        self.const.lum_eff = self._tryReadBox(self.inputLumEff, self.const.lum_eff)
        self.const.lum_eff_type = self.lumEffComboBox.currentIndex()

        ### ###


        ### Wake parameters ###

        self.const.wake_psf = self._tryReadBox(self.inputWakePSF, self.const.wake_psf)
        self.const.wake_extension = self._tryReadBox(self.inputWakeExt, self.const.wake_extension)
        self.wake_plot_ht = 1000*self._tryReadBox(self.inputWakePlotHt, self.wake_plot_ht/1000)

        ### ###


        ### Erosion parameters ###

        self.const.erosion_height_start = 1000*self._tryReadBox(self.inputErosionHtStart, \
            self.const.erosion_height_start/1000)
        self.const.erosion_bins_per_10mass = int(self._tryReadBox(self.inputBinsPer10m, \
            self.const.erosion_bins_per_10mass))
        self.const.erosion_coeff = self._tryReadBox(self.inputErosionCoeff, self.const.erosion_coeff*1e6)/1e6
        self.const.erosion_height_change = 1000*self._tryReadBox(self.inputErosionHtChange, \
            self.const.erosion_height_change/1000)
        self.const.erosion_coeff_change = self._tryReadBox(self.inputErosionCoeffChange, \
            self.const.erosion_coeff_change*1e6)/1e6
        self.const.erosion_mass_index = self._tryReadBox(self.inputErosionMassIndex, \
            self.const.erosion_mass_index)
        self.const.erosion_mass_min = self._tryReadBox(self.inputErosionMassMin, self.const.erosion_mass_min)
        self.const.erosion_mass_max = self._tryReadBox(self.inputErosionMassMax, self.const.erosion_mass_max)


        # If a different density value after the change of erosion is used, read it
        if self.erosion_different_rho:
            self.const.erosion_rho_change = self._tryReadBox(self.inputErosionRhoChange, \
                self.const.erosion_rho_change)
        else:
            # Otherwise, use the same bulk density value
            self.const.erosion_rho_change = self.const.rho


        # If a different ablation coeff value after the change of erosion is used, read it
        if self.erosion_different_sigma:
            self.const.erosion_sigma_change = self._tryReadBox(self.inputErosionAblationCoeffChange, \
                self.const.erosion_sigma_change*1e6)/1e6
        else:
            # Otherwise, use the same bulk density value
            self.const.erosion_sigma_change = self.const.sigma

        ### ###



        ### Disruption parameters ###

        self.const.compressive_strength = 1000*self._tryReadBox(self.inputCompressiveStrength, \
            self.const.compressive_strength/1000)

        # If a different value for erosion coefficient after disruption should be used, read it
        if self.disruption_different_erosion_coeff:
            self.const.disruption_erosion_coeff = self._tryReadBox(self.inputDisruptionErosionCoeff, \
                self.const.disruption_erosion_coeff*1e6)/1e6
        else:
            # Otherwise, use the same value
            self.const.disruption_erosion_coeff = self.const.erosion_coeff


        self.const.disruption_mass_grain_ratio = self._tryReadBox(self.inputDisruptionMassGrainRatio, \
            self.const.disruption_mass_grain_ratio*100)/100
        self.const.disruption_mass_index = self._tryReadBox(self.inputDisruptionMassIndex, \
            self.const.disruption_mass_index)
        self.const.disruption_mass_min_ratio = self._tryReadBox(self.inputDisruptionMassMinRatio, \
            self.const.disruption_mass_min_ratio*100)/100
        self.const.disruption_mass_max_ratio = self._tryReadBox(self.inputDisruptionMassMaxRatio, \
            self.const.disruption_mass_max_ratio*100)/100

        ### ###


        ### Autofit parameters ###

        self.pso_iterations = self._tryReadBox(self.inputAutoFitPSOIterations, self.pso_iterations, \
            value_type=int)
        self.pso_particles = self._tryReadBox(self.inputAutoFitPSOParticles, self.pso_particles, \
            value_type=int)

        self.autofit_mag_weight = self._tryReadBox(self.inputAutoFitMagWeight, self.autofit_mag_weight)
        self.autofit_lag_weight_ht_change = 1000*self._tryReadBox(self.inputAutoFitLagWeightHtChange, \
            self.autofit_lag_weight_ht_change/1000)
        self.autofit_lag_weights = self._tryReadBox(self.inputAutoFitLagWeights, self.autofit_lag_weights, \
            value_type=lambda x: list(map(float, x.split(","))))

        ###


        ### Radar measurements ###

        # Height of electron line density measurement
        self.const.electron_density_meas_ht = 1000*self._tryReadBox(self.inputElectronDensityHt, \
            self.const.electron_density_meas_ht/1000)

        # Electron line density measurement
        self.const.electron_density_meas_q = self._tryReadBox(self.inputElectronDensity, \
            self.const.electron_density_meas_q)

        ### ###


        # Update the boxes with read values
        self.updateInputBoxes()



    def calcHeightLimits(self):
        """ Find the upper and lower height limit for all plots. """


        # Track plot limits
        plot_beg_ht = -np.inf
        plot_end_ht = np.inf


        # Keep track of the height range from the original observations
        for obs in self.traj.observations:

            height_data = obs.model_ht[obs.ignore_list == 0]/1000

            # Keep track of the height limits
            plot_beg_ht = max(plot_beg_ht, np.max(height_data))
            plot_end_ht = min(plot_end_ht, np.min(height_data))


        # Track heihgts of additional observations from the .met file (if available)
        if self.met_obs is not None:

            # Plot additional lags for all sites (plot only mirfit lags)
            for site in self.met_obs.sites:

                height_data = self.met_obs.height_data[site]/1000

                # Keep track of the height limits
                plot_beg_ht = max(plot_beg_ht, np.max(height_data))
                plot_end_ht = min(plot_end_ht, np.min(height_data))


        return plot_beg_ht, plot_end_ht





    def updateCommonPlotFeatures(self, plt_handle, sr, plot_text=False):
        """ Update common features on all plots such as the erosion start. 

        Arguments:
            plt_handle: [axis handle]
            sr: [object] Simulation results.
        """


        # Get the plot X limits
        x_min, x_max = plt_handle.get_xlim()

        # Get the plot Y limits
        y_min, y_max = plt_handle.get_ylim()


        # Generate array for horizontal line plotting
        x_arr = np.linspace(x_min, x_max, 10)


        # Plot the beginning only if it's inside the plot and the erosion is on
        if self.const.erosion_on and (self.const.erosion_height_start/1000 >= y_min) \
            and (self.const.erosion_height_start/1000 <= y_max):
            
            # Plot a line marking erosion beginning
            plt_handle.plot(x_arr, np.zeros_like(x_arr) + self.const.erosion_height_start/1000, \
                linestyle='dashed', color='k', alpha=0.25)

            # Add the text about erosion begin
            if plot_text:
                plt_handle.text(x_min, TEXT_LABEL_HT_PAD + self.const.erosion_height_start/1000,\
                    "Erosion beg", size=7, alpha=0.5)



        # Only plot the erosion change if it's above the meteor end and inside the plot
        if (self.const.erosion_height_change > self.traj.rend_ele) and (self.const.erosion_height_change/1000\
            >= y_min) and (self.const.erosion_height_change/1000 <= y_max):

            # Plot a line marking erosion change
            plt_handle.plot(x_arr, np.zeros_like(x_arr) + self.const.erosion_height_change/1000,\
                linestyle='dashed', color='k', alpha=0.25)

            # Add the text about erosion change
            if plot_text:
                plt_handle.text(x_min, TEXT_LABEL_HT_PAD \
                    + self.const.erosion_height_change/1000, "Erosion change", size=7, alpha=0.5)



        # Plot the disruption height
        if self.const.disruption_on and (self.const.disruption_height is not None):

            # Check that the disruption height is inside the plot
            if (self.const.disruption_height/1000 >= y_min) and (self.const.disruption_height/1000 <=y_max):

                plt_handle.plot(x_arr, np.zeros_like(x_arr) + self.const.disruption_height/1000,\
                    linestyle='dotted', color='k', alpha=0.5)

                # Add the text about disruption
                if plot_text:
                    plt_handle.text(x_min, TEXT_LABEL_HT_PAD \
                        + self.const.disruption_height/1000, "Disruption", size=7, alpha=0.5)



        # Force the plot X limits back
        plt_handle.set_xlim([x_min, x_max])




    def updateMagnitudePlot(self, show_previous=False):
        """ Update the magnitude plot. 

        Keyword arguments:
            show_previous: [bool] Show the previous simulation. False by default.

        """

        # Choose to show current or previous results
        if show_previous:
            sr = self.simulation_results_prev
        else:
            sr = self.simulation_results


        self.magnitudePlot.canvas.axes.clear()
        self.electronDensityPlot.clear()

        
        # Track plot limits
        mag_brightest = np.inf
        mag_faintest = -np.inf

        # Plot observed magnitudes from different stations (only if additional LC data is not given)
        if self.lc_data is None:
            for obs in self.traj.observations:

                # Skip instances when no magnitudes are present
                if obs.absolute_magnitudes is None:
                    continue

                # Extract data
                abs_mag_data = obs.absolute_magnitudes[obs.ignore_list == 0]
                height_data = obs.model_ht[obs.ignore_list == 0]/1000

                # Don't plot magnitudes fainter than 8
                mag_filter = abs_mag_data < 8
                height_data = height_data[mag_filter]
                abs_mag_data = abs_mag_data[mag_filter]

                if len(height_data) == 0:
                    continue

                self.magnitudePlot.canvas.axes.plot(abs_mag_data, height_data, marker='x',
                    linestyle='dashed', label=obs.station_id, markersize=5, linewidth=1)

                # Keep track of the faintest and the brightest magnitude
                if len(abs_mag_data):
                    mag_brightest = min(mag_brightest, np.min(abs_mag_data[~np.isinf(abs_mag_data)]))
                    mag_faintest = max(mag_faintest, np.max(abs_mag_data[~np.isinf(abs_mag_data)]))
                else:
                    mag_faintest = 6.0
                    if sr is not None:
                        mag_brightest = np.min(sr.abs_magnitude)
                    else:
                        mag_brightest = -2.0
            

        # Update the radiated energy label
        obs_radiated_energy, _ = self.calcPhotometricMass()
        obs_radiated_energy_text = "Radiated energy (obs) = {:.2e} J".format(obs_radiated_energy)

        # Print TNT tonnes equivalent if energy > 0.01 T TNT
        if obs_radiated_energy > 4e7:
            obs_radiated_energy_text += " ({:.2f} T TNT)".format(obs_radiated_energy*2.39006e-10)
        self.obsRadiatedEnergyLabel.setText(obs_radiated_energy_text)

        # Compute the simulated energy
        if sr is not None:

            # Filter out NaNs and Infs from magnitude array
            mag_filter = ~np.isnan(sr.abs_magnitude) & ~np.isinf(sr.abs_magnitude)

            sim_radiated_energy = calcRadiatedEnergy(sr.time_arr[mag_filter], sr.abs_magnitude[mag_filter], \
                P_0m=self.const.P_0m)
            sim_radiated_energy_text = "Radiated energy (sim) = {:.2e} J".format(sim_radiated_energy)

            if sim_radiated_energy > 4e7:
                sim_radiated_energy_text += " ({:.2f} T TNT)".format(sim_radiated_energy*2.39006e-10)

        else:
            sim_radiated_energy_text = "Radiated energy (sim) = ?"

        self.simRadiatedEnergyLabel.setText(sim_radiated_energy_text)


        # Plot additional observations from the .met file (if available)
        if self.met_obs is not None:

            # Plot additional magnitudes for all sites
            for site in self.met_obs.sites:

                # Extract data
                abs_mag_data = self.met_obs.abs_mag_data[site]
                height_data = self.met_obs.height_data[site]/1000

                self.magnitudePlot.canvas.axes.plot(abs_mag_data, \
                    height_data, marker='x', linestyle='dashed', label=str(site), markersize=5, linewidth=1)

                # Keep track of the faintest and the brightest magnitude
                mag_brightest = min(mag_brightest, np.min(abs_mag_data[~np.isinf(abs_mag_data)]))
                mag_faintest = max(mag_faintest, np.max(abs_mag_data[~np.isinf(abs_mag_data)]))


        # Plot additional observations from the light curve CSV file (if available)
        if self.lc_data is not None:

            # Plot additional magnitudes for all sites
            for site in self.lc_data.sites:

                # Extract data
                abs_mag_data = self.lc_data.abs_mag_data[site]
                height_data = self.lc_data.height_data[site]/1000

                # self.magnitudePlot.canvas.axes.plot(abs_mag_data, \
                #     height_data, marker='x', linestyle='dashed', label=str(site), markersize=1, linewidth=1)

                self.magnitudePlot.canvas.axes.plot(abs_mag_data, \
                    height_data, marker='x', linestyle='none', label=str(site), markersize=2, linewidth=1)

                # Keep track of the faintest and the brightest magnitude
                mag_brightest = min(mag_brightest, np.min(abs_mag_data[~np.isinf(abs_mag_data)]))
                mag_faintest = max(mag_faintest, np.max(abs_mag_data[~np.isinf(abs_mag_data)]))


        # Plot simulated magnitudes
        q_sim_plot = None
        q_meas_plot = None
        if sr is not None:

            # # Cut the part with same beginning heights as observations
            # temp_arr = np.c_[sr.brightest_height_arr, sr.abs_magnitude]
            # temp_arr = temp_arr[(sr.brightest_height_arr < plot_beg_ht*1000) \
            #     & (sr.brightest_height_arr > plot_end_ht*1000)]
            # ht_arr, abs_mag_arr = temp_arr.T

            # Plot the simulated magnitudes
            self.magnitudePlot.canvas.axes.plot(sr.abs_magnitude, sr.leading_frag_height_arr/1000, \
                label='Simulated', color='k', alpha=0.5)


            if self.electron_density_plot_show:

                # Plot the simulated electron line density
                q_sim_plot = self.electronDensityPlot.plot(np.log10(sr.electron_density_total_arr), \
                    sr.leading_frag_height_arr/1000, color='b', alpha=0.5, linestyle='dashed')

                # Plot the measured electron line density
                if self.const.electron_density_meas_ht > 0:

                    q_meas_plot = self.electronDensityPlot.scatter( \
                        np.log10(self.const.electron_density_meas_q), \
                        self.const.electron_density_meas_ht/1000, c='b', s=10, marker='o')



            # Plot magnitudes of individual fragments
            if self.const.fragmentation_show_individual_lcs:

                # Plot the magnitude of the initial/main fragment
                self.magnitudePlot.canvas.axes.plot(sr.abs_magnitude_main, \
                    sr.leading_frag_height_arr/1000, color='blue', linestyle='solid', linewidth=1, \
                    alpha=0.5)

                # Plot the magnitude of the eroded and disrupted fragments
                self.magnitudePlot.canvas.axes.plot(sr.abs_magnitude_eroded, \
                    sr.leading_frag_height_arr/1000, color='purple', linestyle='dashed', linewidth=1, \
                    alpha=0.5)


                # Plot magnitudes for every fragmentation entry
                for frag_entry in sr.const.fragmentation_entries:

                    # Plot magnitude of the main fragment in the fragmentation (not the grains)
                    if len(frag_entry.main_height_data):

                        # Choose the color of the main fragment depending on the fragmentation type
                        if frag_entry.frag_type == "F":
                            line_color = 'blue'
                        elif frag_entry.frag_type == "EF":
                            line_color = 'green'

                        self.magnitudePlot.canvas.axes.plot(frag_entry.main_abs_mag, \
                            frag_entry.main_height_data/1000, color=line_color, linestyle='dashed', \
                            linewidth=1, alpha=0.5)


                    # Plot magnitude of the grains
                    if len(frag_entry.grains_height_data):

                        # Eroding grains are purple, dust grains as red
                        if frag_entry.frag_type == "EF":
                            line_color = 'purple'
                        elif frag_entry.frag_type == "D":
                            line_color = 'red'

                        self.magnitudePlot.canvas.axes.plot(frag_entry.grains_abs_mag, \
                            frag_entry.grains_height_data/1000, color=line_color, linestyle='dashed', \
                            linewidth=1, alpha=0.5)




        # Label for magnitude plot
        self.magnitudePlot.canvas.axes.set_ylabel('Height (km)')
        self.magnitudePlot.canvas.axes.set_xlabel('Abs magnitude')

        # Get legend entries
        handles, labels = self.magnitudePlot.canvas.axes.get_legend_handles_labels()

        # Label for electron line density
        if self.electron_density_plot_show:
            
            self.electronDensityPlot.set_xlabel("log$_{10}$ q (e$^{-}$/m)")

            # Scale the electron line density so it's not in the way of magnitude
            q_min, q_max = self.electronDensityPlot.get_xlim()
            self.electronDensityPlot.set_xlim(q_min, q_max)
            self.electronDensityPlot.set_visible(True)

            # Add to the legend entry
            if q_sim_plot is not None:
                handles += [q_sim_plot[0]]
                labels += ["Sim q"]

            if q_meas_plot is not None:
                handles += [q_meas_plot]
                labels += ["Meas q"]


        else:
            self.electronDensityPlot.set_visible(False)
            self.magnitudePlot.canvas.axes.set_title('Magnitude')


        self.magnitudePlot.canvas.axes.set_ylim([self.plot_end_ht + END_HT_PAD, \
            self.plot_beg_ht + BEG_HT_PAD])
        self.magnitudePlot.canvas.axes.set_xlim([mag_brightest - 1, mag_faintest + 1])
        self.magnitudePlot.canvas.axes.invert_xaxis()

        # Plot common features across all plots
        self.updateCommonPlotFeatures(self.magnitudePlot.canvas.axes, sr, plot_text=True)

        #self.magnitudePlot.canvas.axes.legend(prop={'size': LEGEND_TEXT_SIZE}, loc='upper right')
        self.magnitudePlot.canvas.axes.legend(handles, labels, prop={'size': LEGEND_TEXT_SIZE}, loc='upper right')

        self.magnitudePlot.canvas.axes.grid(color="k", linestyle='dotted', alpha=0.3)

        self.magnitudePlot.canvas.figure.tight_layout()

        self.magnitudePlot.canvas.draw()


    def updateInterpolations(self, show_previous=False):
        """ Update variables interpolating simulation results and interfacing with the trajectory.
            This is used to correctly compute the lag and to accurately compare simulated and observed 
            heights.
        """

        # Choose to show current or previous results
        if show_previous:
            sr = self.simulation_results_prev
        else:
            sr = self.simulation_results


        if sr is not None:

            # Interpolate the simulated length by height
            self.sim_len_interp = scipy.interpolate.interp1d(sr.leading_frag_height_arr, sr.leading_frag_length_arr,
                bounds_error=False, fill_value='extrapolate')

            # Interpolate the simulated time
            self.sim_time_interp = scipy.interpolate.interp1d(sr.leading_frag_height_arr, sr.time_arr,
                bounds_error=False, fill_value='extrapolate')

            # Interpolate the velocity
            self.sim_vel_interp = scipy.interpolate.interp1d(sr.leading_frag_height_arr, sr.leading_frag_vel_arr,
                bounds_error=False, fill_value='extrapolate')


            # Find the simulated length at the trajectory begining
            self.sim_len_beg = self.sim_len_interp(self.traj.rbeg_ele)

            # Find the simulated time at the trajectory begining
            self.sim_time_beg = self.sim_time_interp(self.traj.rbeg_ele)

            # Find the simulated velocity at the trajectory begining
            self.sim_vel_beg = self.sim_vel_interp(self.traj.rbeg_ele)

            # Set the simulated length at the beginning of observations to zero
            norm_sim_len = sr.leading_frag_length_arr - self.sim_len_beg

            # Compute the normalized time
            norm_sim_time = sr.time_arr - self.sim_time_beg


            self.norm_sim_ht = sr.leading_frag_height_arr[norm_sim_len > 0]
            self.norm_sim_time = norm_sim_time[norm_sim_len > 0]
            self.norm_sim_len = norm_sim_len[norm_sim_len > 0]

            # Interpolate the normalized length by time
            self.sim_norm_len_interp = scipy.interpolate.interp1d(self.norm_sim_time, self.norm_sim_len,
                bounds_error=False, fill_value='extrapolate')

            # Interpolate the height by normalized length
            self.sim_norm_ht_interp = scipy.interpolate.interp1d(self.norm_sim_len, self.norm_sim_ht,
                bounds_error=False, fill_value='extrapolate')


    def updateVelocityPlot(self, show_previous=False):
        """ Update the velocity plot. 
        
        Arguments:
            None

        Keyword arguments:
            show_previous: [bool] Show the previous simulation. False by default.

        """

        # Choose to show current or previous results
        if show_previous:
            sr = self.simulation_results_prev
        else:
            sr = self.simulation_results


        self.velocityPlot.canvas.axes.clear()


        vel_min = np.inf
        vel_max = -np.inf

        # Plot observed velocities from different stations
        for obs in self.traj.observations:

            # Extract data
            vel_data = obs.velocities[obs.ignore_list == 0][1:]/1000
            height_data = obs.model_ht[obs.ignore_list == 0][1:]/1000

            # If there is a simulation, correct the heights
            if (sr is not None) and (obs.state_vect_dist is not None):

                # Compute the corrected heights, so the simulations and the observations match
                height_data = self.sim_norm_ht_interp(obs.state_vect_dist[obs.ignore_list == 0][1:])/1000

            self.velocityPlot.canvas.axes.plot(vel_data, height_data, marker='o', label=obs.station_id, \
                markersize=1, linestyle='none')

            # Keep track of the faintest and the brightest magnitude
            vel_min = min(vel_min, np.min(vel_data))
            vel_max = max(vel_max, np.max(vel_data))



        # Plot the observed initial velocity
        self.velocityPlot.canvas.axes.plot(self.traj.orbit.v_init/1000, self.traj.rbeg_ele/1000, \
            marker='x', label="Vinit obs", markersize=5, linestyle='none', color='k')

        # Plot the observed average velocity
        avg_vel_ht_plot_arr = np.linspace(self.traj.rbeg_ele/1000, self.traj.rend_ele/1000, 10)
        self.velocityPlot.canvas.axes.plot(np.zeros_like(avg_vel_ht_plot_arr) \
            + self.traj.orbit.v_avg/1000, avg_vel_ht_plot_arr, label="Vavg obs", linestyle='dashed', \
            color='k', alpha=0.5)



        # Plot simulated velocity
        if sr is not None:

            # Plot the simulated velocity at the brightest point
            self.velocityPlot.canvas.axes.plot(sr.brightest_vel_arr/1000, sr.brightest_height_arr/1000, \
                label='Simulated - brightest', color='k', alpha=0.5)

            # Plot the simulated velocity at the leading fragment
            self.velocityPlot.canvas.axes.plot(sr.leading_frag_vel_arr/1000, sr.leading_frag_height_arr/1000,\
                label='Simulated - leading', color='k', alpha=0.5, linestyle="dashed")



            ### Compute the simulated average velocity ###

            # Select only the height range from observations
            sim_vel_obs_range = sr.leading_frag_vel_arr[(sr.leading_frag_height_arr <= self.traj.rbeg_ele) \
                & (sr.leading_frag_height_arr >= self.traj.rend_ele)]

            # Compute the simulated average velocity
            v_avg_sim = np.mean(sim_vel_obs_range)

            ### ###


            # Plot the simulated average velocity
            self.velocityPlot.canvas.axes.plot(np.zeros_like(avg_vel_ht_plot_arr) \
                + v_avg_sim/1000, avg_vel_ht_plot_arr, label="Vavg sim", linestyle='dotted', \
                color='k', alpha=0.5)



        self.velocityPlot.canvas.axes.set_ylabel('Height (km)')
        self.velocityPlot.canvas.axes.set_xlabel('Velocity (km/s)')

        self.velocityPlot.canvas.axes.set_ylim([self.plot_end_ht + END_HT_PAD, self.plot_beg_ht + BEG_HT_PAD])
        self.velocityPlot.canvas.axes.set_xlim([vel_min - 1, vel_max + 1])

        # Plot common features across all plots
        self.updateCommonPlotFeatures(self.velocityPlot.canvas.axes, sr)

        self.velocityPlot.canvas.axes.legend(prop={'size': LEGEND_TEXT_SIZE}, loc='upper left')

        self.velocityPlot.canvas.axes.grid(color="k", linestyle='dotted', alpha=0.3)

        self.velocityPlot.canvas.axes.set_title('Velocity')

        self.velocityPlot.canvas.figure.tight_layout()

        self.velocityPlot.canvas.draw()



    def updateLagPlot(self, show_previous=False):
        """ Update the lag plot. 
        
        Arguments:
            None

        Keyword arguments:
            show_previous: [bool] Show the previous simulation. False by default.

        """

        # Choose to show current of previous results
        if show_previous:
            sr = self.simulation_results_prev
        else:
            sr = self.simulation_results


        self.lagPlot.canvas.figure.clear()


        # Update the observed initial velocity label
        # NOTE: The ECI, not the ground-fixed velocity needs to be used, as the meteor model does not
        #   include Earth's rotation!
        self.vInitObsLabel.setText("Vinit obs = {:.3f} km/s".format(self.traj.orbit.v_init/1000))


        # Compute things needed for the simulated lag
        if sr is not None:

            # Compute the simulated lag
            sim_lag = self.sim_norm_len_interp(self.norm_sim_time) - self.sim_vel_beg*self.norm_sim_time

            # Compute the height for the simulated lag
            sim_lag_ht = self.norm_sim_ht



            # Update the simulated initial velocity label
            self.vInitSimLabel.setText("Vinit sim = {:.3f} km/s".format(self.sim_vel_beg/1000))


            # ### Compute parameters for the brightest point on the trajectory ###

            # # Cut the part with same beginning heights as observations
            # temp_arr = np.c_[sr.brightest_height_arr, sr.brightest_length_arr]
            # temp_arr = temp_arr[(sr.brightest_height_arr <= self.traj.rbeg_ele) \
            #     & (sr.brightest_height_arr >= self.plot_end_ht)]
            # brightest_ht_arr, brightest_len_arr = temp_arr.T

            # if len(brightest_len_arr):

            #     # Compute the simulated lag using the observed velocity
            #     brightest_lag_sim = brightest_len_arr - brightest_len_arr[0] \
            #         - self.traj.orbit.v_init*np.arange(0, self.const.dt*len(brightest_len_arr), \
            #                                            self.const.dt)[:len(brightest_len_arr)]

            #     ###


            #     ### Compute parameters for the leading point on the trajectory ###

            #     # Cut the part with same beginning heights as observations
            #     temp_arr = np.c_[sr.leading_frag_height_arr, sr.leading_frag_length_arr]
            #     temp_arr = temp_arr[(sr.leading_frag_height_arr <= self.traj.rbeg_ele) \
            #         & (sr.leading_frag_height_arr >= self.plot_end_ht)]
            #     leading_ht_arr, leading_frag_len_arr = temp_arr.T

            #     # Compute the simulated lag using the observed velocity
            #     leading_lag_sim = leading_frag_len_arr - leading_frag_len_arr[0] \
            #                       - self.traj.orbit.v_init*np.arange(0, self.const.dt*len(leading_frag_len_arr), \
            #                                                          self.const.dt)[:len(leading_frag_len_arr)]

            #     ###

            # else:
            #     sr = None



        ### Lag plot ###

        # Init the lag plot
        lag_plot = self.lagPlot.canvas.figure.add_subplot(1, 2, 1, label='lag')


        # Init the lag residuals plot
        lag_residuals_plot = self.lagPlot.canvas.figure.add_subplot(1, 2, 2, label='lag residuals', \
            sharey=lag_plot)


        # Plot the lag from observations
        for obs in self.traj.observations:

            if (sr is None) or (obs.state_vect_dist is None):

                # Get observed heights
                height_data = obs.model_ht[obs.ignore_list == 0]
                
                # Plot observed lag directly from observations if no simulations are available
                lag_handle = lag_plot.plot(obs.lag[obs.ignore_list == 0], height_data/1000, marker='x', \
                    linestyle='dashed', label=obs.station_id, markersize=3, linewidth=0.5)


            # Recompute the lag using simulated parameters
            else:

                # Compute the observed lag
                obs_lag = obs.state_vect_dist[obs.ignore_list == 0] \
                    - self.sim_vel_beg*obs.time_data[obs.ignore_list == 0]

                # Compute the corrected heights, so the simulations and the observations match
                obs_ht = self.sim_norm_ht_interp(obs.state_vect_dist[obs.ignore_list == 0])
                
                # # Take the observed heights
                # obs_ht = obs.model_ht[obs.ignore_list == 0]

                # Plot the observed lag
                lag_handle = lag_plot.plot(obs_lag, obs_ht/1000, marker='x', \
                    linestyle='dashed', label=obs.station_id, markersize=3, linewidth=0.5)
                

                # Sample the simulated normalized length at observed times
                sim_norm_len_sampled = self.sim_norm_len_interp(obs.time_data[obs.ignore_list == 0])

                # Compute the length residuals
                len_res = obs.state_vect_dist[obs.ignore_list == 0] - sim_norm_len_sampled

                # Plot the length residuals
                lag_residuals_plot.scatter(len_res, obs_ht/1000, marker='+', \
                    c=lag_handle[0].get_color(), label="Leading, {:s}".format(obs.station_id), s=6)
                


                # ### Compute the residuals from simulated, brightest point on the trajectory ###

                # # Get simulated lags at the same height as observed
                # brightest_interp =  scipy.interpolate.interp1d(-brightest_ht_arr, brightest_lag_sim, \
                #     bounds_error=False, fill_value=0)

                # obs_height_indices = height_data > np.min(brightest_ht_arr)
                # obs_hts = height_data[obs_height_indices]
                # brightest_residuals = obs.lag[obs.ignore_list == 0][obs_height_indices] \
                #     - brightest_interp(-obs_hts)

                # # Plot the lag residuals
                # lag_residuals_plot.scatter(brightest_residuals, obs_hts/1000, marker='+', \
                #     c=lag_handle[0].get_color(), label="Brightest, {:s}".format(obs.station_id), s=6)

                # ### ###


                # ### Compute the residuals from simulated, leading point on the trajectory ###

                # # Get simulated lags at the same height as observed
                # leading_interp =  scipy.interpolate.interp1d(-leading_ht_arr, leading_lag_sim, \
                #     bounds_error=False, fill_value=0)
                # obs_height_indices = height_data > np.min(leading_ht_arr)
                # obs_hts = height_data[obs_height_indices]
                # leading_residuals = obs.lag[obs.ignore_list == 0][obs_height_indices] \
                #     - leading_interp(-obs_hts)

                # # Plot the lag residuals
                # lag_residuals_plot.scatter(leading_residuals, obs_hts/1000, marker='s', \
                #     c=lag_handle[0].get_color(), label="Leading, {:s}".format(obs.station_id), s=6)

                # ### ###



        # Plot additional observations from the .met file (if available)
        if self.met_obs is not None:

            # Plot additional lags for all sites (plot only mirfit lags)
            for site in self.met_obs.sites:

                time_data = self.met_obs.time_data[site]
                state_vect_dist_data = self.met_obs.state_vect_dist_data[site]

                # Only plot mirfit lags
                if self.met.mirfit:

                    # Compute the observed lag
                    obs_lag = state_vect_dist_data - self.sim_vel_beg*time_data

                    # Compute the corrected heights, so the simulations and the observations match
                    obs_ht = self.sim_norm_ht_interp(state_vect_dist_data)

                    # # Get observed heights
                    # obs_ht = self.met_obs.height_data[site]

                    # Plot the observed lag
                    lag_handle = lag_plot.plot(obs_lag, obs_ht/1000, marker='x', \
                        linestyle='dashed', label=str(site),  markersize=5, linewidth=1)
                    

                    # Sample the simulated normalized length at observed times
                    sim_norm_len_sampled = self.sim_norm_len_interp(time_data)

                    # Compute the length residuals
                    len_res = state_vect_dist_data - sim_norm_len_sampled

                    # Plot the length residuals
                    lag_residuals_plot.scatter(len_res, obs_ht/1000, marker='+', \
                        c=lag_handle[0].get_color(), label="Leading, {:s}".format(str(site)), s=6)


        # Get X plot limits before the simulated lag is plotted
        x_min, x_max = lag_plot.get_xlim()



        # Plot simulated lag
        if sr is not None:

            # Plot the lag
            lag_plot.plot(sim_lag, sim_lag_ht/1000, 
                          label='Simulated - leading', color='k', alpha=0.5, linestyle='dashed')


            # # Plot lag of the brightest point on the trajectory
            # lag_plot.plot(brightest_lag_sim[:len(brightest_ht_arr)], \
            #              (brightest_ht_arr/1000)[:len(brightest_lag_sim)], \
            #              label='Simulated - brightest', color='k', alpha=0.5)


            # # Plot lag of the leading fragment
            # lag_plot.plot(leading_lag_sim[:len(leading_ht_arr)], (leading_ht_arr/1000)[:len(leading_lag_sim)], 
            #     label='Simulated - leading', color='k', alpha=0.5, linestyle='dashed')



        # Set parameters for the lag plot
        lag_plot.set_ylim([self.plot_end_ht + END_HT_PAD, self.plot_beg_ht + BEG_HT_PAD])
        lag_plot.set_xlim([x_min, x_max])

        lag_plot.set_xlabel('Lag (m)')
        lag_plot.set_ylabel('Height (km)')
        
        # Plot common features across all plots
        self.updateCommonPlotFeatures(lag_plot, sr)
        self.updateCommonPlotFeatures(lag_residuals_plot, sr)

        lag_plot.legend(prop={'size': LEGEND_TEXT_SIZE})
        lag_plot.grid(color="k", linestyle='dotted', alpha=0.3)
        lag_plot.set_title('Lag')


        lag_residuals_plot.set_xlabel('Residuals (m)')
        lag_residuals_plot.get_yaxis().set_visible(False)
        lag_residuals_plot.legend(prop={'size': LEGEND_TEXT_SIZE})
        lag_residuals_plot.grid(color="k", linestyle='dotted', alpha=0.3)
        lag_residuals_plot.set_title('Lag residuals')


        self.lagPlot.canvas.figure.tight_layout()
        self.lagPlot.canvas.figure.subplots_adjust(wspace=0)
        self.lagPlot.canvas.draw()



    def incrementWakePlotHeight(self):
        """ Increment wake plot height by 100 m. """

        # If the wake files are loaded, use the wake heights
        if self.wake_heights is not None:
            
            self.wake_ht_current_index = (self.wake_ht_current_index + 1)%len(self.wake_heights)

            # Extract the wake height and observations
            self.wake_plot_ht, self.current_wake_container = self.wake_heights[self.wake_ht_current_index]


        # Otherwise, increment the wake height by 100 m
        else:
            self.wake_plot_ht += 100

        self.updateInputBoxes()
        self.updateWakePlot()



    def decrementWakePlotHeight(self):
        """ Decrement wake plot height by 100 m. """

        # If the wake files are loaded, use the wake heights
        if self.wake_heights is not None:
            
            self.wake_ht_current_index = (self.wake_ht_current_index - 1)%len(self.wake_heights)

            # Extract the wake height and observations
            self.wake_plot_ht, self.current_wake_container = self.wake_heights[self.wake_ht_current_index]


        # Otherwise, decrement the wake height by 100 m
        else:
            self.wake_plot_ht -= 100

        self.updateInputBoxes()
        self.updateWakePlot()




    def redrawWakeHeightLine(self, plt_handle, line_handles, label_handles=None, draw_label=False):
        """ Plot lines on mag, vel, lag plots indicating where the wake is shown. The line will be just 
            refreshed to prevent redrawing all plots from scratch.
        """

        new_line_handles = []
        new_label_handles = []

        # Go through all axes on the image
        for i, ax in enumerate(plt_handle.canvas.figure.get_axes()):

            # Extract line and label handles
            line_handle = None
            if line_handles is not None:
                if i <= (len(line_handles) - 1):
                    line_handle = line_handles[i]

            label_handle = None
            if label_handles is not None:
                if i <= (len(label_handles) - 1):
                    label_handle = label_handles[i]


            # Remove the line from the plot
            if line_handle is not None:
                try:
                    line_handle.remove()
                except ValueError:
                    pass



            # Get the plot limits
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()

            # Construct X array
            x_arr = np.linspace(x_min, x_max)

            # Plot the wake line
            line_handle = ax.plot(x_arr, np.zeros_like(x_arr) + self.wake_plot_ht/1000, \
                linestyle='dashed', color='green', alpha=0.5)[0]

            new_line_handles.append(line_handle)



            # Plot the label if given
            if draw_label:

                # Remove the old label handle
                if label_handle is not None:
                    try:
                        label_handle.remove()
                    except ValueError:
                        pass

                # Draw the wake line label
                label_handle = ax.text(x_min, TEXT_LABEL_HT_PAD + self.wake_plot_ht/1000, \
                    "Wake plot", size=7, color='green', alpha=0.5)

                new_label_handles.append(label_handle)


            # Keep the plot limits
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])


        # Redraw the plot
        plt_handle.canvas.draw_idle()


        return new_line_handles, new_label_handles



    def normalizeObservedWake(self, len_array, wake_intensity_array, wake_sim, simulated_peak_luminosity, 
        simulated_integrated_luminosity, simulated_peak_length, sim_wake_exists=False):
        """ Normalize the observed wake and match it to the simulated wake. """

        # Normalize the wake by areas under both curves
        if self.wake_normalization_method == "area":

            # Compute the area under the observed wake curve that is within the simulated range
            #   (take only the part after the leading fragment)
            if sim_wake_exists:
                selected_indices = (len_array > -50) | (len_array < self.const.wake_extension)
                wake_intensity_array_trunc = wake_intensity_array[selected_indices]
                len_array_trunc = len_array[selected_indices]
            else:
                wake_intensity_array_trunc = wake_intensity_array
                len_array_trunc = len_array

            observed_integrated_luminosity = np.trapz(wake_intensity_array_trunc, len_array_trunc)


            # Normalize the wake intensity by the area under the intensity curve
            wake_intensity_array *= simulated_integrated_luminosity/observed_integrated_luminosity


        # Normalize the wake by the peak intensity
        else:

            # Handle cases when the simulation doesn't exist
            if simulated_peak_luminosity is None:
                simulated_peak_luminosity = 0

            wake_intensity_array *= simulated_peak_luminosity/np.max(wake_intensity_array)


        # Perform alignments when simulations are available
        if wake_sim is not None:

            # Align wake by peaks
            if self.wake_align_method == 'peak':

                # Find the length of the peak intensity
                peak_len = len_array[np.argmax(wake_intensity_array)]

                # Offset lengths
                len_array -= peak_len + simulated_peak_length


            # Align the wake by cross correlation
            elif self.wake_align_method == 'correlate':


                # Interpolate the model values and sample them at observed points
                sim_wake_interp = scipy.interpolate.interp1d(wake_sim.length_array, \
                    wake_sim.wake_luminosity_profile, bounds_error=False, fill_value=0)
                model_wake_obs_len_sample = sim_wake_interp(-len_array)

                # Correlate the wakes and find the shift
                wake_shift = np.argmax(np.correlate(model_wake_obs_len_sample, wake_intensity_array, \
                    "full")) + 1

                # Find the index of the zero observed length
                obs_len_zero_indx = np.argmin(np.abs(len_array))

                # Add the offset to the observed length
                len_array += len_array[(obs_len_zero_indx + wake_shift)%len(model_wake_obs_len_sample)]


        return len_array, wake_intensity_array



    def updateWakePlot(self, show_previous=False):
        """ Plot the wake. """

        if not show_previous:
            self.readInputBoxes()


        # Choose to show current of previous results
        if show_previous:
            sr = self.simulation_results_prev
        else:
            sr = self.simulation_results


        wake = None


        # Plot lines on mag, vel, lag plots indicating where the wake is shown
        self.magnitudePlotWakeLines, self.magnitudePlotWakeLineLabels \
            = self.redrawWakeHeightLine(self.magnitudePlot, self.magnitudePlotWakeLines, \
                label_handles=self.magnitudePlotWakeLineLabels, draw_label=True)
        self.velocityPlotWakeLines, _ = self.redrawWakeHeightLine(self.velocityPlot, \
            self.velocityPlotWakeLines)
        self.lagPlotWakeLines, _ = self.redrawWakeHeightLine(self.lagPlot, self.lagPlotWakeLines)



        # Find the appropriate observed wake to plot for the given height
        if self.wake_heights is not None:

            # Find the index of the observed wake that's closest to the given plot height
            self.wake_ht_current_index = np.argmin(np.abs(np.array([w[0] for w in self.wake_heights]) \
                - self.wake_plot_ht))

            # Extract the wake height and observations
            self.wake_plot_ht, self.current_wake_container = self.wake_heights[self.wake_ht_current_index]



        self.wakePlot.canvas.figure.clf()


        # Create one large wake plot with movable height (first column)
        wake_ht_plot = self.wakePlot.canvas.figure.add_subplot(1, 2, 1, label='wake ht')


        ### PLOT SIMULATED WAKE ###
        simulated_integrated_luminosity = 1.0
        simulated_peak_luminosity = 1.0
        simulated_peak_length = 0.0
        sim_wake_exists = False
        if sr is not None:

            # Find the wake index closest to the given wake height
            wake_res_indx =  np.argmin(np.abs(self.wake_plot_ht - sr.brightest_height_arr))

            # Get the approprate wake results
            wake = sr.wake_results[wake_res_indx]

            if wake is not None:

                sim_wake_exists = True

                ### Plot the simulated wake ###
                wake_ht_plot.plot(wake.length_array, wake.wake_luminosity_profile, \
                    label='Simulated', color='k', alpha=0.5)

                # Compute the area under the simulated wake curve (take only the part after the leading fragment)
                selected_indices = (wake.length_array > 0) | (wake.length_array < self.const.wake_extension)
                wake_intensity_array_trunc = wake.wake_luminosity_profile[selected_indices]
                len_array_trunc = wake.length_array[selected_indices]
                simulated_integrated_luminosity = np.trapz(wake_intensity_array_trunc, len_array_trunc)

                # Store the max values
                simulated_peak_luminosity = np.max(wake_intensity_array_trunc)
                simulated_peak_length = len_array_trunc[np.argmax(wake_intensity_array_trunc)]

                ### ###


                ### Plot fragment masses ###
                if self.wake_show_mass_bins:

                    # Add second scale for masses (log)
                    mass_ax = wake_ht_plot.twinx()
                    mass_ax.set_yscale("log")

                    frag_main = None
                    frag_len_list = []
                    #m_init_list = []
                    m_list = []
                    for frag in wake.frag_list:

                        if frag.main:
                            frag_main = frag

                        if frag.m > 1.0e-11:

                            # Compute the fragment length on the wake plot
                            frag_wake_len = frag.length - wake.leading_frag_length

                            frag_len_list.append(frag_wake_len)
                            #m_init_list.append(frag.m_init)
                            m_list.append(frag.m)


                    # Plot the mass of the main fragment
                    if frag_main is not None:
                        # mass_ax.scatter(frag_main.length - wake.leading_frag_length, frag_main.m_init, c='k', \
                        #     s=10, alpha=0.5)
                        mass_ax.scatter(frag_main.length - wake.leading_frag_length, frag_main.m, c='r', \
                            s=20, alpha=0.5)


                    # # Plot the fragment initial mass at the position of the fragment length
                    # mass_ax.scatter(frag_len_list, m_init_list, c='k', s=0.5, label="$m_0$")

                    # Plot the instantaneous fragment mass
                    mass_ax.scatter(frag_len_list, m_list, c='r', s=0.5, label="$m$")


                    # Set the mass range (from ablation limit to initial mass)
                    mass_ax.set_ylim([1.0e-11, self.const.m_init])

                    mass_ax.set_ylabel("Fragment mass (kg)")



                ### ###


        ### ###


        ### PLOT OBSERVED WAKE ###

        if self.current_wake_container is not None:
            
            # Extract array of length and intensity 
            len_array = []
            wake_intensity_array = []
            for wake_pt in self.current_wake_container.points:
                len_array.append(wake_pt.leading_frag_length)
                wake_intensity_array.append(wake_pt.intens_sum)

            len_array = np.array(len_array)
            wake_intensity_array = np.array(wake_intensity_array)


            # # Smooth the wake intensity to reduce noise
            # wake_intensity_array = scipy.signal.savgol_filter(wake_intensity_array, 5, 3)


            # Normalize and align the observed wake with simulations
            len_array, wake_intensity_array = self.normalizeObservedWake(len_array, wake_intensity_array, \
                wake, simulated_peak_luminosity, simulated_integrated_luminosity, simulated_peak_length, \
                sim_wake_exists=sim_wake_exists)



            # Plot the observed wake
            wake_ht_plot.plot(-len_array, wake_intensity_array,
                label='Observed, site: {:s}'.format(str(self.current_wake_container.site_id)), color='k', \
                linestyle='dotted')

        ### ###

        if wake_ht_plot.lines or wake_ht_plot.collections:
            wake_ht_plot.legend(prop={'size': LEGEND_TEXT_SIZE})


        wake_ht_plot.set_xlabel('Length behind leading fragment (m)')
        wake_ht_plot.set_ylabel('Intensity')

        wake_ht_plot.invert_xaxis()

        wake_ht_plot.set_ylim(bottom=0)

        wake_ht_plot.set_title('Wake at {:.2f} km'.format(self.wake_plot_ht/1000))




        ### PLOT WAKE OVERVIEW SUBPLOT ###

        # Init second subplot
        wake_overview_plot = self.wakePlot.canvas.figure.add_subplot(1, 2, 2, label='wake overview')

        # Number of wakes shown on the plot
        n_plots = 10


        # Generate a range of heights used for plotting
        if self.wake_heights is not None:

            # Choose heights from observed list of heights
            step = len(self.wake_heights)//n_plots
            wake_plotting_heights = np.array(self.wake_heights[::-step])[:, 0]

        else:
            # Generate a list of heights between the begin and end of the trajectory
            wake_plotting_heights = np.linspace(self.traj.rbeg_ele, self.traj.rend_ele, n_plots)
        

        # Go through different heights
        for i, plot_ht in enumerate(sorted(wake_plotting_heights)):

            plot_shift = 0.75*i

            wake = None
            if sr is not None:

                # Find the wake index closest to the given wake height
                wake_res_indx =  np.argmin(np.abs(plot_ht - sr.brightest_height_arr))

                # Get the approprate wake results
                wake = sr.wake_results[wake_res_indx]


            # Plot the simulated wake
            simulated_integrated_luminosity = 1.0
            simulated_peak_luminosity = None
            simulated_peak_length = 0.0
            sim_wake_exists = False
            if wake is not None:

                sim_wake_exists = True

                # Scale the luminosity profile to 1 and shift
                luminosity_profile_scaled = plot_shift \
                    + wake.wake_luminosity_profile/np.max(wake.wake_luminosity_profile)

                # Plot the simulated wake and scale it to the maximum lumino
                wake_overview_plot.plot(wake.length_array, luminosity_profile_scaled, \
                    label='Simulated', color='k', alpha=0.5, linewidth=1)


                # Add text indicating the height of the wake
                wake_overview_plot.text(wake.length_array[0], luminosity_profile_scaled[0] - 0.1/n_plots,\
                    "{:.2f} km".format(plot_ht/1000), ha='center', va='top', size=6)

                # Compute the area under the simulated wake curve (take only the part after the leading fragment)
                selected_indices = (wake.length_array > 0) | (wake.length_array < self.const.wake_extension)
                wake_intensity_array_trunc = wake.wake_luminosity_profile[selected_indices]
                len_array_trunc = wake.length_array[selected_indices]
                simulated_integrated_luminosity = np.trapz(wake_intensity_array_trunc, len_array_trunc)

                # Store the max values
                simulated_peak_luminosity = np.max(wake_intensity_array_trunc)
                simulated_peak_length = len_array_trunc[np.argmax(wake_intensity_array_trunc)]

                ### ###



            # Plot observed wake
            if self.wake_heights is not None:

                # Find the index of the observed wake that's closest to the given plot height
                wake_ht_index = np.argmin(np.abs(np.array([w[0] for w in self.wake_heights]) \
                    - plot_ht))

                # Extract the wake observations
                _, wake_container = self.wake_heights[wake_ht_index]

                len_array = []
                wake_intensity_array = []
                for wake_pt in wake_container.points:
                    len_array.append(wake_pt.leading_frag_length)
                    wake_intensity_array.append(wake_pt.intens_sum)

                len_array = np.array(len_array)
                wake_intensity_array = np.array(wake_intensity_array)


                # Normalize and align the observed wake with simulations
                len_array, wake_intensity_array = self.normalizeObservedWake(len_array, \
                    wake_intensity_array, wake, simulated_peak_luminosity, \
                    simulated_integrated_luminosity, simulated_peak_length, \
                    sim_wake_exists=sim_wake_exists)


                # Normalize the observed wake to 1
                if simulated_peak_luminosity is not None:
                    obs_wake_scale = simulated_peak_luminosity
                else:
                    obs_wake_scale = np.max(wake_intensity_array)

                wake_intensity_array_scaled = plot_shift + wake_intensity_array/obs_wake_scale

                # Plot the observed wake
                wake_overview_plot.plot(-len_array, wake_intensity_array_scaled, color='k', \
                    linestyle='dotted', linewidth=1)


                # If the simulations are not shown, plot the wake heights
                if wake is None:
                    wake_overview_plot.text(-self.const.wake_extension, wake_intensity_array_scaled[-1] \
                        - 0.1/n_plots, "{:.2f} km".format(plot_ht/1000), ha='center', va='top', size=6)



        wake_overview_plot.set_xlabel('Length behind leading fragment (m)')
        wake_overview_plot.get_yaxis().set_visible(False)

        wake_overview_plot.set_ylim(bottom=0)
        wake_overview_plot.set_xlim(-self.const.wake_extension - 50, 50)

        wake_overview_plot.invert_xaxis()

        wake_overview_plot.set_title('Wake overview')

        ###



        self.wakePlot.canvas.figure.tight_layout()
        self.wakePlot.canvas.draw()



        # Enable/disable wake normalization and alignment dpending on availability of simulated data
        self.wakeNormalizeGroup.setDisabled(wake is None)
        self.wakeAlignGroup.setDisabled(wake is None)



    def showPreviousResults(self):
        """ Show previous simulation results and parameters. """

        if self.simulation_results_prev is not None:

            self.updateInputBoxes(show_previous=True)
            self.updateInterpolations(show_previous=True)
            self.updateMagnitudePlot(show_previous=True)
            self.updateVelocityPlot(show_previous=True)
            self.updateLagPlot(show_previous=True)
            self.updateWakePlot(show_previous=True)



    def showCurrentResults(self):
        """ Show current simulation results and parameters. """

        self.updateInputBoxes(show_previous=False)
        self.updateInterpolations(show_previous=False)
        self.updateMagnitudePlot(show_previous=False)
        self.updateVelocityPlot(show_previous=False)
        self.updateLagPlot(show_previous=False)
        self.updateWakePlot(show_previous=False)



    def plotAtmDensity(self):
        """ Open a separate plot showing the MSISE air density and the polynomial fit. """

        # Init a matplotlib popup window
        self.mpw = MatplotlibPopupWindow()

        # Set the window title
        self.mpw.setWindowTitle("Air density fit")



        # Take mean meteor lat/lon as reference for the atmosphere model
        lat_mean = np.mean([self.traj.rbeg_lat, self.traj.rend_lat])
        lon_mean = meanAngle([self.traj.rbeg_lon, self.traj.rend_lon])


        # Generate a height array
        height_arr = np.linspace(self.dens_fit_ht_beg, self.dens_fit_ht_end, 200)

        # Get atmosphere densities from NRLMSISE-00 (use log values for the fit)
        atm_densities = np.array([getAtmDensity(lat_mean, lon_mean, ht, self.traj.jdt_ref) \
            for ht in height_arr])


        # Get atmosphere densities from the fitted polynomial
        atm_densities_poly = atmDensPoly(height_arr, self.const.dens_co)


        # Plot the MSISE density
        self.mpw.canvas.axes.semilogx(atm_densities, height_arr/1000, \
            label="NRLMSISE-00", color='k')

        # Poly poly fit
        self.mpw.canvas.axes.semilogx(atm_densities_poly, height_arr/1000, \
            label="Polynomial fit", color='red', linestyle="dashed")


        self.mpw.canvas.axes.legend()

        self.mpw.canvas.axes.set_ylabel("Height (km)")
        self.mpw.canvas.axes.set_xlabel("Atmosphere mass density (kg/m^3)")

        self.mpw.canvas.axes.set_ylim([self.dens_fit_ht_end/1000, self.dens_fit_ht_beg/1000])

        self.mpw.canvas.axes.grid()

        self.mpw.canvas.figure.tight_layout()
            
        self.mpw.show()


    def plotDynamicPressure(self):
        """ Open a separate plot with the simulated dynamic pressure. """


        if self.simulation_results is not None:

            # Init a matplotlib popup window
            self.mpw = MatplotlibPopupWindow()

            # Set the window title
            self.mpw.setWindowTitle("Dynamic pressure")



            sr = self.simulation_results

            # Take mean meteor lat/lon as reference for the atmosphere model
            lat_mean = np.mean([self.traj.rbeg_lat, self.traj.rend_lat])
            lon_mean = meanAngle([self.traj.rbeg_lon, self.traj.rend_lon])

            # Compute the dynamic pressure
            brightest_dyn_pressure = dynamicPressure(lat_mean, lon_mean, sr.brightest_height_arr, \
                self.traj.jdt_ref, sr.brightest_vel_arr, gamma=self.const.gamma)
            leading_frag_dyn_pressure = dynamicPressure(lat_mean, lon_mean, sr.leading_frag_height_arr, \
                self.traj.jdt_ref, sr.leading_frag_vel_arr, gamma=self.const.gamma)


            # Plot dyn pressure
            self.mpw.canvas.axes.plot(brightest_dyn_pressure/1e6, sr.brightest_height_arr/1000, \
                label="Brightest", color='k', alpha=0.5)
            self.mpw.canvas.axes.plot(leading_frag_dyn_pressure/1e6, sr.leading_frag_height_arr/1000, \
                label="Leading", color='k', alpha=0.5, linestyle="dashed")


            # Compute and mark peak on the graph
            brightest_peak_dyn_pressure_index = np.argmax(brightest_dyn_pressure)
            peak_dyn_pressure = brightest_dyn_pressure[brightest_peak_dyn_pressure_index]/1e6
            peak_dyn_pressure_ht = sr.brightest_height_arr[brightest_peak_dyn_pressure_index]/1000
            self.mpw.canvas.axes.scatter(peak_dyn_pressure, peak_dyn_pressure_ht, \
                label="Peak P = {:.2f} MPa\nHt = {:.2f} km".format(peak_dyn_pressure, peak_dyn_pressure_ht))



            self.mpw.canvas.axes.legend()

            self.mpw.canvas.axes.set_ylabel("Height (km)")
            self.mpw.canvas.axes.set_xlabel("Dynamic pressure (MPa)")

            self.mpw.canvas.axes.set_ylim([self.plot_end_ht + END_HT_PAD, self.plot_beg_ht + BEG_HT_PAD])

            self.mpw.canvas.axes.grid()

            self.mpw.canvas.figure.tight_layout()
                
            self.mpw.show()


        else:
            print("No simulation results to show!")


    def plotObsVsSimComparison(self):
        """ Plot the comparison between the simulations and observations + wake overview and save to data
            directory.
        """

        print()

        if self.simulation_results is not None:
            
            # Plot the comparison between the simulations and observations (excluding the wake)
            plotObsAndSimComparison(self.traj, self.simulation_results, self.met_obs, self.dir_path, 
                                    wake_containers=None, plot_lag=True)

            # Show a message box
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Saved obs vs sim comparison plots to {:s}".format(self.dir_path))
            msg.setWindowTitle("Saved")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

            print("Saved obs vs sim comparison plots to {:s}".format(self.dir_path))         

            # Plot the wake overview if the wake simulation is available and wake measurements are given
            if self.wake_meas is not None:

                # Run only if not all elements in the wake results are None
                if not all([w is None for w in self.simulation_results.wake_results]):

                    event_name = jd2Date(self.traj.jdt_ref, dt_obj=True).strftime("%Y%m%d_%H%M%S")
                    plotWakeOverview(self.simulation_results, self.wake_meas, self.dir_path, event_name)

                else:
                    
                    # Show warning message
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("No simulated wake to plot! Run the wake simulation first and press the button again.")
                    msg.setWindowTitle("Warning")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()

                    print("No simulated wake to plot! Run the wake simulation first and press the button again.")

        else:
            
            # Show warning
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("No simulation results to show! Run the simulation first and press the button again.")
            msg.setWindowTitle("Warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

            print("No simulation results to show! Run the simulation first and press the button again.")




    def plotMassLoss(self):
        """ Open a separate plot with the simulated mass loss. """


        if self.simulation_results is not None:

            # Init a matplotlib popup window
            self.mpw = MatplotlibPopupWindow()

            # Set the window title
            self.mpw.setWindowTitle("Mass loss")



            sr = self.simulation_results

            # Take mean meteor lat/lon as reference for the atmosphere model
            lat_mean = np.mean([self.traj.rbeg_lat, self.traj.rend_lat])
            lon_mean = meanAngle([self.traj.rbeg_lon, self.traj.rend_lon])

            # Compute the dynamic pressure
            brightest_dyn_pressure = dynamicPressure(lat_mean, lon_mean, sr.brightest_height_arr, \
                self.traj.jdt_ref, sr.brightest_vel_arr, gamma=self.const.gamma)
            main_dyn_pressure = dynamicPressure(lat_mean, lon_mean, sr.main_height_arr, \
                self.traj.jdt_ref, sr.main_vel_arr, gamma=self.const.gamma)


            # Filter out very small masses
            mass_filter = sr.mass_total_active_arr > 0.001


            # Plot mass of all fragments
            self.mpw.canvas.axes.loglog(brightest_dyn_pressure[mass_filter]/1e6, \
                sr.mass_total_active_arr[mass_filter], label="Total mass", color='k', linewidth=1, \
                linestyle='dashed')

            # Plot mass of the main fragment
            self.mpw.canvas.axes.loglog(main_dyn_pressure[mass_filter]/1e6, sr.main_mass_arr[mass_filter], \
                label="Main mass", color='k', linewidth=2)

            self.mpw.canvas.axes.set_xlim(xmin=1e-3)


            self.mpw.canvas.axes.legend()

            self.mpw.canvas.axes.set_ylabel("Mass (kg)")
            self.mpw.canvas.axes.set_xlabel("Dynamic pressure (MPa)")

            self.mpw.canvas.axes.grid()

            self.mpw.canvas.figure.tight_layout()
                
            self.mpw.show()


        else:
            print("No simulation results to show!")




    def plotLumEfficiency(self):
        """ Open a separate plot showing the tau of individual fragments. """

        # Init a matplotlib popup window
        self.mpw = MatplotlibPopupWindow()

        # Set the window title
        self.mpw.setWindowTitle("Lumionus efficiency")


        sr = self.simulation_results


        
        # Plot the magnitude of the total tau
        self.mpw.canvas.axes.plot(100*sr.tau_total_arr, sr.leading_frag_height_arr/1000, color='k', \
            alpha=0.5, label="Total")

        # Plot the magnitude of the initial/main fragment
        self.mpw.canvas.axes.plot(100*sr.tau_main_arr, sr.leading_frag_height_arr/1000, color='blue', 
            linestyle='solid', linewidth=1, alpha=0.5, label="Main")

        # Plot the magnitude of the eroded and disrupted fragments
        self.mpw.canvas.axes.plot(100*sr.tau_eroded_arr, sr.leading_frag_height_arr/1000, color='purple', \
            linestyle='dashed', linewidth=1, alpha=0.5, label="Erosion")


        # Plot magnitudes for every fragmentation entry
        main_f_label = False
        main_ef_label = False
        grains_ef_label = False
        grains_d_label = False

        for frag_entry in self.const.fragmentation_entries:

            label = None

            # Plot magnitude of the main fragment in the fragmentation (not the grains)
            if len(frag_entry.main_height_data):

                # Choose the color of the main fragment depending on the fragmentation type
                if frag_entry.frag_type == "F":
                    line_color = 'blue'
                    if not main_f_label:
                        label = "Main F"
                        main_f_label = True
                        
                elif frag_entry.frag_type == "EF":
                    line_color = 'green'
                    if not main_ef_label:
                        label = "Main EF"
                        main_ef_label = True

                self.mpw.canvas.axes.plot(100*frag_entry.main_tau, frag_entry.main_height_data/1000, \
                    color=line_color, linestyle='dashed', linewidth=1, alpha=0.5, label=label)


            # Plot magnitude of the grains
            if len(frag_entry.grains_height_data):

                # Eroding grains are purple, dust grains as red
                if frag_entry.frag_type == "EF":
                    line_color = 'purple'
                    if not grains_ef_label:
                        label = "Grains EF"
                        grains_ef_label = True

                elif frag_entry.frag_type == "D":
                    line_color = 'red'
                    if not grains_d_label:
                        label = "Grains D"
                        grains_d_label = True

                self.mpw.canvas.axes.plot(100*frag_entry.grains_tau, frag_entry.grains_height_data/1000, \
                    color=line_color, linestyle='dashed', linewidth=1, alpha=0.5, label=label)


        self.mpw.canvas.axes.legend()

        self.mpw.canvas.axes.set_ylabel("Height (km)")
        self.mpw.canvas.axes.set_xlabel("Luminous efficiency (%)")

        self.mpw.canvas.axes.set_ylim([self.plot_end_ht + END_HT_PAD, \
            self.plot_beg_ht + BEG_HT_PAD])

        self.mpw.canvas.axes.grid()

        self.mpw.canvas.figure.tight_layout()
            
        self.mpw.show()



    def runSimulationGUI(self):
        """ Run the simulation and show the results. """

        # If the fragmentation is turned on and no fragmentation data is given, notify the user
        if self.const.fragmentation_on and (self.fragmentation is None):
            frag_error_message = QMessageBox(QMessageBox.Critical, "Fragmentation file error", \
                "Fragmentation is enabled but no fragmentation file is set.")
            frag_error_message.setInformativeText("Either load an existing fragmentation file or create a new one.")
            frag_error_message.exec_()
            return None


        # Load fragmentation entries if fragmentation is enabled
        if self.const.fragmentation_on:

            # Load the file
            self.fragmentation.loadFragmentationFile()

            # Sort entries by height
            self.fragmentation.sortByHeight()

            # Reset the status of all fragmentations
            self.fragmentation.resetAll()

            # Write the fragmentation file
            self.fragmentation.writeFragmentationFile()



        # Store previous run results
        self.const_prev = copy.deepcopy(self.const)
        self.simulation_results_prev = copy.deepcopy(self.simulation_results)


        # Read the values from the input boxes
        self.readInputBoxes()


        # Disable the simulation button (have to force update by calling "repaint")
        self.runSimButton.setStyleSheet("background-color: red")
        self.runSimButton.setDisabled(True)
        self.repaint()


        print('Running simulation...')
        t1 = time.time()

        # Run the simulation
        frag_main, results_list, wake_results = runSimulation(self.const, compute_wake=self.wake_on)

        sim_runtime = time.time() - t1

        if sim_runtime < 0.5:
            print('Simulation runtime: {:d} ms'.format(int(1000*sim_runtime)))
        elif sim_runtime < 100:
            print('Simulation runtime: {:.2f} s'.format(sim_runtime))
        else:
            print('Simulation runtime: {:.2f} min'.format(sim_runtime/60))

        # Store simulation results
        self.simulation_results = SimulationResults(self.const, frag_main, results_list, wake_results)

        # Save simulated parametrs to file
        self.simulation_results.writeCSV(self.dir_path, SIM_RESULTS_CSV)
        print("Saved simulation results to:", os.path.join(self.dir_path, SIM_RESULTS_CSV))

        # Toggle lum eff button to only be available if lum eff was computed
        self.plotLumEffButton.setDisabled(not self.const.fragmentation_show_individual_lcs)

        # Write results in the fragmentation file
        if self.const.fragmentation_on:
            self.fragmentation.writeFragmentationFile()

        # Update the plots
        self.showCurrentResults()

        # Save the latest run parameters
        self.saveFitParameters(False, suffix="_latest")

        # Enable the simulation button
        self.runSimButton.setDisabled(False)
        self.runSimButton.setStyleSheet("background-color: #b1eea6")



    def autoFitMethodToggle(self, text):
        """ Select method of auto fit. """

        self.autofit_method = text        

        # Enable/disable PSO options
        self.inputAutoFitPSOIterations.setDisabled("PSO" not in text)
        self.inputAutoFitPSOParticles.setDisabled("PSO" not in text)




    def autoFit(self):
        """ Run the auto fit procedure. """

        # Read inputs
        self.readInputBoxes()


        # Disable the fit button (have to force update by calling "repaint")
        self.autoFitButton.setStyleSheet("background-color: red")
        self.autoFitButton.setDisabled(True)
        self.repaint()


        print()
        print("===================")
        print("AUTO FIT REFINEMENT")
        print("===================")
        print()


        # Store original constants
        const_original = copy.deepcopy(self.const)

        # Save current simulation results
        simulation_results_prefit = copy.deepcopy(self.simulation_results)

        ### Construct input data matrix ###

        # Construct 2D array of height vs. absolute magnitude vs. lag
        temp_list = []
        for obs in self.traj.observations:


            # Extract data
            height_data = obs.model_ht[obs.ignore_list == 0]
            lag_data = obs.lag[obs.ignore_list == 0]


            # If the magnitudes are None, only fit the lag
            if obs.absolute_magnitudes is None:
                abs_mag_data = np.zeros_like(height_data) + np.nan

            else:
                abs_mag_data = obs.absolute_magnitudes[obs.ignore_list == 0]    


            temp_arr = np.c_[height_data, abs_mag_data, lag_data]

            # Skip infinite magnitude
            temp_arr = temp_arr[~np.isinf(abs_mag_data)]

            temp_list.append(temp_arr)


        # Add data from the met file (if available)
        if self.met_obs is not None:

            # Plot additional magnitudes for all sites
            for site in self.met_obs.sites:

                # Extract data
                height_data = self.met_obs.height_data[site]
                abs_mag_data = self.met_obs.abs_mag_data[site]                


                # Add dummy lag data
                lag_data = np.zeros_like(height_data) + np.nan

                temp_arr = np.c_[height_data, abs_mag_data, lag_data]

                # Skip infinite magnitude
                temp_arr = temp_arr[~np.isinf(abs_mag_data)]

                temp_list.append(temp_arr)


        # Construct single data matrix and sort by reverse height
        fit_input_data = np.vstack(temp_list)
        fit_input_data = fit_input_data[np.argsort(fit_input_data[:, 0])[::-1]]


        ### ###


        ### Only fit parameters of fragmentation processes which are used ###

        p0 = []
        bounds = []
        param_string = 'm'
        const = copy.deepcopy(self.const)


        # Select meteoroid parameters
        meteoroid_params = [const.m_init, const.v_init, const.rho, const.sigma]
        meteoroid_bounds = [[0.1*const.m_init, 2*const.m_init],
                            [0.9*const.v_init, 1.1*const.v_init],
                            [100, 6000],
                            [0.25*const.sigma, 4*const.sigma]]

        p0 += meteoroid_params
        bounds += meteoroid_bounds


        # Add erosion parameters if the erosion is on
        if const.erosion_on:

            erosion_params = [const.erosion_height_start, const.erosion_coeff, \
                const.erosion_height_change, const.erosion_coeff_change, \
                const.erosion_mass_index, const.erosion_mass_min, const.erosion_mass_max]

            erosion_bounds = [[70000, 130000],
                              [0.0, 5.0/1e6],
                              [70000, 130000],
                              [0.0, 5.0/1e6],
                              [1.0, 3.0],
                              [1e-12, 1e-9],
                              [1e-11, 1e-7]]

            p0 += erosion_params
            bounds += erosion_bounds

            param_string += 'e'


        # Add distuption parameters if the disruption is on
        if const.disruption_on:

            disruption_params = [const.compressive_strength, const.disruption_erosion_coeff, \
                const.disruption_mass_index, const.disruption_mass_min_ratio, \
                const.disruption_mass_max_ratio, const.disruption_mass_grain_ratio]

            disruption_bounds = [[0.1*const.compressive_strength, 10*const.compressive_strength],
                                 [0.0, 1.0/1e6],
                                 [1.0, 3.0],
                                 [0.01, 50.0],
                                 [0.1, 80.0],
                                 [0.0, 100.0]]

            p0 += disruption_params
            bounds += disruption_bounds

            param_string += 'd'


        ### ###



        # Normalize the fit parameters to the [0, 1] range
        mini_norm_handle = MinimizationParameterNormalization(p0)
        p0_normed, bounds_normed = mini_norm_handle.normalizeBounds(bounds)


        # Print residual value
        print("Starting residual value: {:.5f}".format(fitResiduals(p0_normed, fit_input_data, \
            param_string, const, self.traj, mini_norm_handle, mag_weight=self.autofit_mag_weight, \
            lag_weights=self.autofit_lag_weights, lag_weight_ht_change=self.autofit_lag_weight_ht_change, \
            verbose=False)))
        print()



        print("Method:", self.autofit_method)


        if self.autofit_method == "Local":

            ### scipy minimize ###

            # Run the fit
            res = scipy.optimize.minimize(fitResiduals, p0_normed, args=(fit_input_data, param_string, \
                const, self.traj, mini_norm_handle, self.autofit_mag_weight, self.autofit_lag_weights, \
                self.autofit_lag_weight_ht_change, True, self), bounds=bounds_normed, tol=0.001)

            print(res)

            fit_params = res.x


            ### ###


        # PSO optimization
        else:

            print()
            print("N particles:", self.pso_particles)
            print("Iterations:", self.pso_iterations)

            ### pyswarms ###

            import pyswarms as ps
            from pyswarms.utils.plotters import plot_cost_history


            # Set up hyperparameters
            #options = {'c1': 0.5, 'c2': 0.7, 'w':0.9}
            options = {'c1': 0.6, 'c2': 0.3, 'w': 0.9, 'k': 10, 'p': 1}


            # Set up bounds (min, max) are (0, 1)
            pso_bounds = (np.zeros(len(p0_normed)), np.ones(len(p0_normed)))


            init_pos = None


            # If PSO local optimization is desired, create a tight cluster of particles around the initial
            #   parameters
            if self.autofit_method == "PSO local":

                # Create particles in a tight Gaussian around the initial parameters
                init_pos = np.random.normal(loc=p0_normed, scale=0.2 + np.zeros_like(p0_normed), \
                    size=(self.pso_particles - 1, len(p0_normed)))
                init_pos[init_pos < 0] = abs(init_pos[init_pos < 0])
                init_pos[init_pos > 1] = 1 - init_pos[init_pos > 1] + 1

                # Add manual fit to initial positions
                init_pos = np.append(init_pos, np.array([p0_normed]), axis=0)


            # Call instance of PSO with bounds argument
            optimizer = ps.single.LocalBestPSO(n_particles=self.pso_particles, dimensions=len(p0_normed), \
                options=options, bounds=pso_bounds, bh_strategy='reflective', vh_strategy='invert', \
                init_pos=init_pos)


            # Run PSO
            cost, pos = optimizer.optimize(fitResidualsListArguments, iters=self.pso_iterations, \
                n_processes=mp.cpu_count() - 1, fit_input_data=fit_input_data, param_string=param_string, \
                const_original=const, traj=self.traj, mini_norm_handle=mini_norm_handle, \
                mag_weight=self.autofit_mag_weight, lag_weights=self.autofit_lag_weights, \
                lag_weight_ht_change=self.autofit_lag_weight_ht_change, verbose=False)

            print(cost, pos)

            fit_params = pos


            # Plot the cost history
            plot_cost_history(optimizer.cost_history)
            plt.show()

            ### ###



        # Enable the fit button
        self.autoFitButton.setDisabled(False)
        self.autoFitButton.setStyleSheet("background-color: #efebe7")


        # Init a Constants instance with fitted parameters
        const_fit = extractConstantParams(const_original, fit_params, param_string, mini_norm_handle)

        # Assign fitted parameters and run the Simulation
        self.const = const_fit
        self.updateInputBoxes()
        self.runSimulationGUI()

        # Store the simulation results prior to auto fit as the previous simulation results
        self.const_prev = const_original
        self.simulation_results_prev = simulation_results_prefit


    def toggleFragmentation(self, event):

        if self.fragmentationGroup.isChecked():

            print("Fragmentation ENABLED!")

            self.const.fragmentation_on = True

        else:

            print("Fragmentation DISABLED!")

            self.const.fragmentation_on = False


    def newFragmentationFile(self):
        """ Choose the location of a new fragmentation file. """

        # Choose the location of the new file
        fragmentation_file_path = QFileDialog.getSaveFileName(self, "Choose the fragmentation file", \
            os.path.join(self.dir_path, FRAG_FILE_NAME), "Fragmentation file (*.txt)")[0]

        # If it was cancelled, end function
        if not fragmentation_file_path:
            self.fragmentationGroup.setChecked(False)
            return None

        # Add a file extension if it was not given
        if len(os.path.basename(fragmentation_file_path).split(".")) == 1:
            fragmentation_file_path += ".txt"

        print("New fragmentation file:", fragmentation_file_path)

        self.fragmentation = FragmentationContainer(self, fragmentation_file_path)
        self.fragmentation.newFragmentationFile()
        

    def loadFragmentationFile(self):
        """ Load the fragmentation file for disk. """

        fragmentation_file_path = QFileDialog.getOpenFileName(self, "Choose the fragmentation file", \
            self.dir_path, "Fragmentation file (*.txt)")[0]


        print("Loading fragmentation file:", fragmentation_file_path)


        # Load the fragmentation data
        self.fragmentation = FragmentationContainer(self, fragmentation_file_path)
        self.fragmentation.loadFragmentationFile()

        print("Loaded fragments:")
        for frag_entry in self.fragmentation.fragmentation_entries:
            print(frag_entry.toString())


    def addFragmentation(self, frag_type):
        """ Add a fragmentation line."""

        # Open a few fragmentation file, if one doesn't exist
        if self.fragmentation is None:
            self.newFragmentationFile()

        self.fragmentation.addFragmentation(frag_type)


    def electronDensityMeasChanged(self):
        """ Handle what happens when the electron density measurements are changed. """

        self.readInputBoxes()
        self.showCurrentResults()


    def saveUpdatedOrbit(self):
        """ Save updated orbit and trajectory to file. """

        # Compute the difference between the model and the measured initial velocity
        # NOTE: The ECI, not the ground-fixed velocity needs to be used, as the meteor model does not
        #   include Earth's rotation!
        v_init_diff = self.const.v_init - self.traj.orbit.v_init

        # Recompute the orbit with an increased initial velocity
        orb = calcOrbit(self.traj.radiant_eci_mini, self.traj.v_init + v_init_diff, self.traj.v_avg \
            + v_init_diff, self.traj.state_vect_mini, self.traj.rbeg_jd)


        print(orb)

        
        # Make a file name for the report
        traj_updated = copy.deepcopy(self.traj)
        traj_updated.orbit = orb
        dir_path, file_name = os.path.split(self.traj_path)
        report_file_name = file_name.replace('trajectory.pickle', '') + 'report_sim.txt'

        # If USG data is used, only the orbit can be saved
        if self.usg_data is not None:

            # Save orbit report using USG data
            with open(os.path.join(dir_path, "updated_orbit.txt"), 'w') as f:
                f.write(traj_updated.orbit.__repr__())

        else:
            # Save the report with updated orbit
            traj_updated.saveReport(dir_path, report_file_name, uncertainties=self.traj.uncertainties,
                verbose=False)

            updated_pickle_file_name = file_name.replace('trajectory.pickle', 'trajectory_sim.pickle')

            # Save the updated pickle file
            savePickle(traj_updated, dir_path, updated_pickle_file_name)



    def saveFitParameters(self, event, suffix=None):
        """ Save fit parameters to a JSON file. """

        if suffix is None:
            suffix = str("")

        dir_path, file_name = os.path.split(self.traj_path)
        file_name = file_name.replace('trajectory.pickle', '').replace(".txt", "_") + "sim_fit{:s}.json".format(suffix)

        # Save the fit parameters to disk in JSON format
        saveConstants(self.const, dir_path, file_name)

        print("Saved fit parameters to:", os.path.join(dir_path, file_name))




    def saveVideo(self, event):
        """ Generate video frames using the simulated wake and PSF. """


        # Skip saving video if there is no wake results
        if self.simulation_results.wake_results is None:
            return False


        # Disable the video button
        self.wakeSaveVideoButton.setStyleSheet("background-color: red")
        self.wakeSaveVideoButton.setDisabled(True)
        self.repaint()


        # # (CAMO) The plate scale is fixed at 0.5 meters per pixel at 100 km, so the animation is better visible
        # plate_scale = 0.5

        # Moderate FOV plate scale (m/px @ 100 km)
        plate_scale = 10

        # Frame nackground intensity
        background_intensity = 30


        # Create meshgrid for Gaussian evaluation
        grid_size = 20
        x = np.linspace(-grid_size//2, grid_size//2, grid_size)
        y = np.linspace(-grid_size//2, grid_size//2, grid_size)
        X, Y = np.meshgrid(x, y)
        mesh_list = np.c_[X.flatten(), Y.flatten()]


        # Compute the video size from the pixel scale and wake extent and make sure it's an even number
        frame_ht = frame_wid = int(np.ceil(self.const.wake_extension/plate_scale) + grid_size)
        if frame_ht%2 == 1:
            frame_ht = frame_wid = frame_ht + 1


        # Init the Gaussian
        gauss = scipy.stats.multivariate_normal([0.0, 0.0], [[self.const.wake_psf/np.sqrt(2)/plate_scale, 0 ], 
                                                 [        0,     self.const.wake_psf/np.sqrt(2)/plate_scale]])


        # Get the directory path
        dir_path, _ = os.path.split(self.traj_path)



        # Find the wake index starting at 5 km above the beginning height
        video_beg_ht = self.traj.rbeg_ele + 5000
        wake_beg_indx =  np.argmin(np.abs(video_beg_ht - self.simulation_results.brightest_height_arr))

        # Find the wake index ending at the end height
        video_end_ht = self.traj.rend_ele
        wake_end_indx =  np.argmin(np.abs(video_end_ht - self.simulation_results.brightest_height_arr))

        # Go through all wake points
        for i, (wake, ht) in enumerate(zip(self.simulation_results.wake_results[wake_beg_indx:wake_end_indx],\
            self.simulation_results.brightest_height_arr[wake_beg_indx:wake_end_indx])):

            if wake is None:
                continue

            print('Height: {:.3f} km'.format(ht/1000))


            # Init a new video frame
            frame = np.zeros((frame_ht, frame_wid), dtype=np.float64)


            # Compute pixel scale from length
            pixel_length = wake.length_points/plate_scale


            # Normalize the luminosity by the maximum luminosity (oversaturate the peak so the fainter parts
            #   are better visible)
            luminosities = 2*255*wake.luminosity_points/self.simulation_results.wake_max_lum

            
            # Evaluate the gaussian of every fragment
            for px, lum in zip(pixel_length, luminosities):

                # Compute the gaussian centre
                center = int(px) + frame_wid//2

                # Skip those fragments exiting the FOV
                if center < -grid_size//2:
                    continue

                # Evaluate the gaussian, normalize so that the brightest peak of the meteor is saturating
                gauss_eval = self.const.wake_psf/plate_scale*lum*gauss.pdf(mesh_list).reshape(grid_size, \
                    grid_size)


                ### Add the evaluated gaussian to the frame ###

                # Compute range of pixel coordinates where to add the evaluated window
                x_min = center - grid_size//2
                x_min_eval = 0
                if x_min < 0: 
                    x_min_eval = -x_min
                    x_min = 0

                x_max = center + grid_size//2
                x_max_eval = grid_size
                if x_max >= frame_wid: 
                    x_max_eval = grid_size + frame_wid - x_max
                    x_max = frame_wid - 1

                y_min = center - grid_size//2
                y_min_eval = 0
                if y_min < 0: 
                    y_min_eval = -y_min
                    y_min = 0


                y_max = center + grid_size//2
                y_max_eval = grid_size
                if y_max >= frame_wid: 
                    y_max_eval = grid_size + frame_ht - y_max
                    y_max = frame_ht - 1

                ### ###

                # print()
                # print(center)
                # print(y_min,y_max, x_min,x_max)
                # print(y_min_eval,y_max_eval, x_min_eval,x_max_eval)


                # Add the evaluated gaussian to the frame
                frame[y_min:y_max, x_min:x_max] += gauss_eval[y_min_eval:y_max_eval, x_min_eval:x_max_eval]

            # Add frame background intensity
            frame += background_intensity

            # Save the image to disk
            saveImage(os.path.join(dir_path, "{:04d}_{:7.3f}km.png".format(i, ht/1000)), frame, cmap='gray',
                vmin=0, vmax=255)


        print("Video frame saving done!")
            

        # Enable the video button
        self.wakeSaveVideoButton.setDisabled(False)
        self.wakeSaveVideoButton.setStyleSheet("background-color: #efebe7")







if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Run meteor ablation modelling using the given trajectory file.")

    arg_parser.add_argument('traj_pickle', metavar='TRAJ_PICKLE', type=str, \
        help="Either the .pickle file with the trajectory solution and the magnitudes, or the path to the \
        folder when the --all option is given. If the --usg option is given, the trajectory will be loaded \
        from a file specifing a reference points and a light curve.")

    arg_parser.add_argument('--usg', \
        help=""" Flag for US government (CNEOS) data. The trajectory is given in a special input file \
        instead of a pickle file.
        """, \
        action="store_true")

    arg_parser.add_argument('-l', '--load', metavar='LOAD_JSON', \
        help="Load JSON file with fit parameters. Instead of giving the full path to the file, you can call \
        it as '--load .' and it will automatically find the file if it exists.", type=str)

    arg_parser.add_argument('-m', '--met', metavar='MET_FILE', \
        help='Load additional observations from a METAL or mirfit .met file.', type=str)

    arg_parser.add_argument('-c', '--lc', metavar='LIGHT_CURVE', \
        help="""Additional light curve given in a comma-separated CSV file. The beginning of every station \
        entry should start with 'Station: Name', followed by the height in km and the absolute magnitude. E.g.
        # Station: XX0001
        # Height (km), Abs Magnitude
          102.3996248,  -8.021
          101.2172498,  -8.036
          100.0348748,  -8.142
           98.8524998,  -8.607
           97.6701248,  -8.703
           96.4877498,  -8.788
           95.3053748,  -9.084
           94.1229998,  -9.399
           92.9406248   -9.475
           91.7582498,  -9.671
           90.5758747,  -9.956
           89.3934997, -10.061
           88.2111247, -10.227
           87.0287497, -10.412
           85.8463747, -10.588
           84.6639997, -10.763
        """, type=str)

    arg_parser.add_argument('-w', '--wid', metavar='WID_FILES', \
        help='Load mirfit wid files which will be used for wake plotting. Wildchars can be used, e.g. /path/wid*.txt.', 
        type=str, nargs='+')

    arg_parser.add_argument('-a', '--all', \
        help="""Automatically find and load the trajectory pickle file, the metal met file, and the wid \
        files. Instead of the path to the trajectory pickle file, a path to folders with data files should \
        be given. For example, if the data is in /home/user/data/20200526_012345_mir and \
        /home/user/data/20200526_012345_met, you should call this module as as: \
        python -m wmpl.MetSim.GUI /home/user/data/20200526_012345 --all --load . \
        """, \
        action="store_true")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    # os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1" # bool


    ### Compute the window scaling factor for high resolution displays ###

    # Get the screen resolution
    app = QApplication([])
    screen = app.primaryScreen()
    screen_size = screen.size()
    screen_width = screen_size.width()
    screen_height = screen_size.height()

    # Only scale the window if the screen resolution is not 1080p
    if screen_height != 1080:

        # Compute the scaling factor, taking 1080p as the reference resolution (compute the ratio of 
        # diagonals)
        # Use the screen height as the reference to avoid issues with very wide screens, and assume that the 
        # screen size ratio is 1.6 (16:10)
        screen_width_calc = int(screen_height*1.6)
        scaling_factor = np.sqrt(screen_width_calc**2 + screen_height**2) / np.sqrt(1920**2 + 1080**2)

        # If the scaling factor is > 1, reduce it by 2% to avoid too large fonts
        if scaling_factor > 1:
            scaling_factor *= 0.98

            if scaling_factor < 1:
                scaling_factor = 1

        os.environ["QT_SCALE_FACTOR"] = str(scaling_factor)

    else:    
        scaling_factor = 1

    # Destroy the QApplication object
    app.quit()
    del app

    ### ###
    

    # Init PyQt5 window
    app = QApplication([])

    # Set a font for the whole application
    font = QFont()
    font.setFamily("Arial")
    font.setPointSize(int(np.ceil(8/scaling_factor)))
    app.setFont(font)

    

    # Automatically find all input files if the --all option is given
    traj_pickle_file = None
    met_file = None
    wid_files = None
    if cml_args.all and not cml_args.usg:

        # Find all folders that match the given regex
        for dir_path in glob.glob(os.path.abspath(cml_args.traj_pickle.rstrip(os.sep) + "*")):

            # Only take directories
            if not os.path.isdir(dir_path):
                continue

            # Try finding the needed files
            for file_name in sorted(os.listdir(dir_path)):

                file_path = os.path.join(dir_path, file_name)

                # If the folder is the metal folder, don't load the trajectory pickle nor wake files
                if not os.path.basename(dir_path).endswith("_met"):

                    # If it's a trajectory pickle file, assign it
                    if traj_pickle_file is None:
                        if file_name.endswith("_trajectory.pickle"):
                            traj_pickle_file = file_path
                            continue

                    # Look for wid files
                    if wid_files is None:
                        if file_name.startswith("wid_") and file_name.endswith(".txt"):
                            wid_files = sorted(glob.glob(os.path.join(dir_path, "wid_*.txt")))
                            continue


                # Look for state files with magnitude measurements
                # Note that there is no check if the met_file has already been found, which means that it
                #   will keep loading state files and take the last one in the end (i.e. the latest one)
                if file_name.startswith("state") and file_name.endswith(".met"):

                    # Try loading the state file and check if it's the METAL state file
                    met = loadMet(dir_path, file_name)

                    # Take the file if it's a METAL state file
                    if not met.mirfit:
                        met_file = file_path


        # Print notices if certain files were not found
        if traj_pickle_file is None:
            print("No trajectory .pickle files found in the given path: {:s}".format(cml_args.traj_pickle))
            sys.exit()

        if met_file is None:
            print("No .met files found in {:s}".format(cml_args.traj_pickle))

        if wid_files is None:
            print("No wid files found in {:s}".format(cml_args.traj_pickle))


    else:

        # Assign individual files
        traj_pickle_file = os.path.abspath(cml_args.traj_pickle)
        met_file = cml_args.met
        wid_files = cml_args.wid



    # Handle auto loading fit parameters
    load_file = None
    if cml_args.load == '.':

        # Extract the directory (or more) where the JSON file with fit parametrs could be
        if os.path.isfile(cml_args.traj_pickle):
            dir_path = os.path.abspath(os.path.dirname(cml_args.traj_pickle))

        else:
            dir_path = os.path.abspath(cml_args.traj_pickle.rstrip(os.sep) + "*")


        # Find the JSON file with the fit parameters
        for dir_path in glob.glob(dir_path):

            # Skip files
            if not os.path.isdir(dir_path):
                continue

            # Find the file with fit parametres (but not the latest file, just the saved parameters)
            for file_name in sorted(os.listdir(dir_path)):

                if file_name.endswith("sim_fit.json"):
                    load_file = os.path.join(dir_path, file_name)
                    break

    else:
        load_file = cml_args.load


    print("Loading trajectory pickle file:")
    print(traj_pickle_file)

    if met_file is not None:
        print("Loading additional magnitudes from .met file:")
        print(met_file)

    if cml_args.lc is not None:
        print("Loading additional light curve from a light curve CSV file:")
        print(cml_args.lc)

    if wid_files is not None:
        print("Loading wake data from:")
        for wfile in wid_files:
            print(wfile)


    if load_file is not None:
        print("Loading fit parameters from:")
        print(load_file)


    # Init the MetSimGUI application
    main_window = MetSimGUI(traj_pickle_file, const_json_file=load_file, \
        met_path=met_file, lc_path=cml_args.lc, wid_files=wid_files, usg_input=cml_args.usg)


    main_window.show()

    sys.exit(app.exec_())
