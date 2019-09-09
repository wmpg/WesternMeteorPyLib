""" GUI interface to meteor ablation models which enables manual modelling of meteors. """


import os
import sys
import copy
import argparse
import time
import json
import glob

import numpy as np
import scipy.stats
import scipy.interpolate
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.uic import loadUi


from wmpl.Formats.Met import loadMet
from wmpl.MetSim.MetSimErosion import runSimulation, Constants
from wmpl.Trajectory.Orbit import calcOrbit
from wmpl.Utils.Math import averageClosePoints, findClosestPoints, vectMag, lineFunc
from wmpl.Utils.Physics import calcMass
from wmpl.Utils.Pickling import loadPickle
from wmpl.Utils.Plotting import saveImage
from wmpl.Utils.TrajConversions import unixTime2JD, geo2Cartesian, cartesian2Geo, altAz2RADec, \
    altAz2RADec_vect, raDec2ECI




### CONSTANTS ###

# Height padding for top plots
BEG_HT_PAD = +5
END_HT_PAD = -2

# Text label height padding
TEXT_LABEL_HT_PAD = 0.1

### ###




class SimulationResults(object):
    def __init__(self, const, results_list, wake_results):
        """ Container for simulation results. """


        # Unpack the results
        results_list = np.array(results_list).astype(np.float64)
        self.time_arr, self.luminosity_arr, self.brightest_height_arr, self.brightest_length_arr, \
            self.brightest_vel_arr, self.leading_frag_length_arr, self.mass_total_arr = results_list.T


        # Calculate absolute magnitude (apparent @100km)
        self.abs_magnitude = -2.5*np.log10(self.luminosity_arr/const.P_0m)



        ### Wake simulation ###

        self.wake_results = wake_results
        self.wake_max_lum = 0

        if np.any(wake_results):
            
            # Determine the wake plot upper limit
            self.wake_max_lum = max([max(wake.wake_luminosity_profile) for wake in wake_results \
                if wake is not None])


        ###



class MetObservations(object):
    def __init__(self, met, traj):
        """ Container for additional observations from a .met file. Computes the lag and apparent magnitude
            given the base trajectory solution. 
        """

        self.met = met
        self.traj = traj

        self.sites = [site for site in met.sites]


        self.height_data = {}
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

                    # Distance from the state vector to the projected point on the radiant line
                    state_vect_dist.append(vectMag(self.traj.state_vect_mini - rad_cpa))

                    # Compute the height (meters)
                    _, _, ht = cartesian2Geo(jd, *rad_cpa)
                    height_data.append(ht)

                    # Compute the range to the station and the absolute magnitude
                    if mag is not None:
                        r = vectMag(rad_cpa - stat)
                        abs_mag = mag + 5*np.log10(100000/r)
                    else:
                        abs_mag = np.nan

                    abs_mag_data.append(abs_mag)


                state_vect_dist = np.array(state_vect_dist)

                # Compute the lag
                self.lag_data[site] = state_vect_dist - lineFunc(time_rel_picks, *self.traj.velocity_fit)

                # Store the heights and magnitudes
                self.height_data[site] = np.array(height_data)
                self.abs_mag_data[site] = np.array(abs_mag_data)





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



class MetSimGUI(QMainWindow):
    
    def __init__(self, traj_path, const_json_file=None, met_path=None, wid_files=None):
        """ GUI tool for MetSim. 
    
        Arguments:
            traj_path: [str] Path to the trajectory pickle file.

        Keyword arguments:
            const_json_file: [str] Path to the JSON file with simulation parameters.
            met: [str] Path to the METAL or mirfit .met file with additional magnitude or lag information.
            wid_files: [str] Mirfit wid files containing the meteor wake information.
        """
        

        self.traj_path = traj_path

        # Load the trajectory pickle file
        self.traj = loadPickle(*os.path.split(traj_path))


        ### LOAD .met FILE ###

        # Load a METAL .met file if given
        self.met = None
        if met_path is not None:
            if os.path.isfile(met_path):
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


        ### Wake parameters
        self.wake_on = False
        self.wake_ht_current_index = 0
        self.current_wake_container = None

        if self.wake_heights is not None:
            self.wake_plot_ht, self.current_wake_container = self.wake_heights[self.wake_ht_current_index]
        else:
            self.wake_plot_ht = self.traj.rbeg_ele # m

        self.wake_normalization_method = 'area'
        self.wake_align_method = 'none'


        self.magnitudePlotWakeLine = None
        self.magnitudePlotWakeLineLabel = None
        self.velocityPlotWakeLine = None
        self.lagPlotWakeLine = None

        ### ###


        # Disable different erosion coeff after disruption at the beginning
        self.disruption_different_erosion_coeff = False

        self.simulation_results = None

        self.const_prev = None
        self.simulation_results_prev = None


        ### ### ### ###


        ### Init simulation parameters ###

        # Init the constants
        self.const = Constants()

        # If a JSON file with constant was given, load them instead of initing from scratch
        if const_json_file is not None:

            with open(const_json_file) as f:
                const_json = json.load(f)

            # Fill in the constants
            for key in const_json:
                setattr(self.const, key, const_json[key])


            # Check if the disruption erosion coefficient is different than the main erosion coeff
            if const_json['disruption_erosion_coeff'] != const_json['erosion_coeff']:
                self.disruption_different_erosion_coeff = True

        else:

            # Set the constants value from the trajectory
            self.const.zenith_angle = self.traj.orbit.zc
            self.const.v_init = self.traj.orbit.v_init

            # Set kill height to the observed end height
            self.const.h_kill = self.traj.rend_ele - 3000

            # Set erosion heights to the beginning/end height
            self.const.erosion_height_start = self.traj.rbeg_ele
            self.const.erosion_height_change = self.traj.rend_ele

            # Disable erosion and disruption at the beginning
            self.const.erosion_on = False
            self.const.disruption_on = False


            # Calculate the photometric mass
            self.const.m_init = self.calcPhotometricMass()

        ### ###



        # Update the values in the input boxes
        self.updateInputBoxes()



        ### Add key bindings ###

        self.runSimButton.clicked.connect(self.runSimulationGUI)
        
        self.showPreviousButton.pressed.connect(self.showPreviousResults)
        self.showPreviousButton.released.connect(self.showCurrentResults)

        self.saveUpdatedOrbitButton.clicked.connect(self.saveUpdatedOrbit)

        self.saveFitParametersButton.clicked.connect(self.saveFitParameters)


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


        self.checkBoxWake.stateChanged.connect(self.checkBoxWakeSignal)
        self.checkBoxErosion.stateChanged.connect(self.checkBoxErosionSignal)
        self.checkBoxDisruption.stateChanged.connect(self.checkBoxDisruptionSignal)
        self.checkBoxDisruptionErosionCoeff.stateChanged.connect(self.checkBoxDisruptionErosionCoeffSignal)

        ### ###

        # Update checkboxes
        self.checkBoxWakeSignal(None)
        self.checkBoxErosionSignal(None)
        self.checkBoxDisruptionSignal(None)
        self.checkBoxDisruptionErosionCoeffSignal(None)
        self.toggleWakeNormalizationMethod(None)
        self.toggleWakeAlignMethod(None)

        # Update plots
        self.showCurrentResults()




    def loadWakeFile(self, file_path):
        """ Load a mirfit wake "wid" file. """


        # Extract the site ID and the frame number from the file name
        site_id, frame_n = os.path.basename(file_path).replace('.txt', '').split('_')[1:]
        site_id = str(int(site_id))
        frame_n = int(frame_n)

        print('wid file: ', site_id, frame_n)


        # Extract geo coordinates of sites
        lat_dict = {obs.station_id:obs.lat for obs in self.traj.observations}
        lon_dict = {obs.station_id:obs.lon for obs in self.traj.observations}
        ele_dict = {obs.station_id:obs.ele for obs in self.traj.observations}

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
                    self.traj.jdt_ref, lat_dict[site_id], lon_dict[site_id])

                # Compute the station coordinates at the given time
                stat = geo2Cartesian(lat_dict[site_id], lon_dict[site_id], ele_dict[site_id], \
                    self.traj.jdt_ref)

                # Compute measurement rays in cartesian coordinates
                meas = np.array(raDec2ECI(ra, dec))

                # Calculate closest points of approach (observed line of sight to radiant line)
                obs_cpa, rad_cpa, d = findClosestPoints(stat, meas, self.traj.state_vect_mini, \
                    self.traj.radiant_eci_mini)

                # If the projected point is above the state vector, use negative lengths
                state_vect_dist_sign = 1.0
                if vectMag(rad_cpa) > vectMag(self.traj.state_vect_mini):
                    state_vect_dist_sign = -1.0


                # Compute Distance from the state vector to the projected point on the radiant line
                state_vect_dist = state_vect_dist_sign*vectMag(self.traj.state_vect_mini - rad_cpa)

                # Compute the height (meters)
                _, _, ht = cartesian2Geo(self.traj.jdt_ref, *rad_cpa)

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


        return wake_container



    def calcPhotometricMass(self):
        """ Calculate photometric mass from given magnitude data. """

        time_mag_arr = []
        avg_t_diff_max = 0
        for obs in self.traj.observations:

            # If there are not magnitudes for this site, skip it
            if obs.absolute_magnitudes is None:
                continue

            # Compute average time difference
            avg_t_diff_max = max(avg_t_diff_max, np.median(obs.time_data[1:] - obs.time_data[:-1]))

            for t, mag in zip(obs.time_data, obs.absolute_magnitudes):
                if (mag is not None) and (not np.isnan(mag)):
                    time_mag_arr.append([t, mag])

        
        # If there are no magnitudes, assume that the initial mass is 0.1 grams
        if not time_mag_arr:
            return self.const.m_init


        # Sort array by time
        time_mag_arr = np.array(sorted(time_mag_arr, key=lambda x: x[0]))

        time_arr, mag_arr = time_mag_arr.T

        
        # Average out the magnitudes
        time_arr, mag_arr = averageClosePoints(time_arr, mag_arr, avg_t_diff_max)

        # Compute the photometry mass
        return calcMass(np.array(time_arr), np.array(mag_arr), self.traj.orbit.v_avg, P_0m=self.const.P_0m)



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
        self.inputMassInit.setText("{:.1e}".format(const.m_init))
        self.inputAblationCoeff.setText("{:.3f}".format(const.sigma*1e6))
        self.inputVelInit.setText("{:.3f}".format(const.v_init/1000))
        self.inputShapeFact.setText("{:.2f}".format(const.shape_factor))
        self.inputGamma.setText("{:.1f}".format(const.gamma))
        self.inputZenithAngle.setText("{:.3f}".format(np.degrees(const.zenith_angle)))

        ### ###


        ### Wake parameters ###

        self.checkBoxWake.setChecked(self.wake_on)

        self.inputWakePSF.setText("{:.1f}".format(const.wake_psf))
        self.inputWakeExt.setText("{:d}".format(int(const.wake_extension)))
        self.inputWakePlotHt.setText("{:.3f}".format(self.wake_plot_ht/1000))

        ### ###


        ### Erosion parameters ###

        self.checkBoxErosion.setChecked(const.erosion_on)
        self.checkBoxDisruptionErosionCoeff.setChecked(self.disruption_different_erosion_coeff)

        self.inputErosionHtStart.setText("{:.3f}".format(const.erosion_height_start/1000))
        self.inputErosionCoeff.setText("{:.3f}".format(const.erosion_coeff*1e6))
        self.inputErosionHtChange.setText("{:.3f}".format(const.erosion_height_change/1000))
        self.inputErosionCoeffChange.setText("{:.3f}".format(const.erosion_coeff_change*1e6))
        self.inputErosionMassIndex.setText("{:.2f}".format(const.erosion_mass_index))
        self.inputErosionMassMin.setText("{:.2e}".format(const.erosion_mass_min))
        self.inputErosionMassMax.setText("{:.2e}".format(const.erosion_mass_max))

        ### ###


        ### Disruption parameters ###

        self.checkBoxDisruption.setChecked(const.disruption_on)
        self.inputDisruptionErosionCoeff.setText("{:.3f}".format(const.disruption_erosion_coeff*1e6))
        self.inputCompressiveStrength.setText("{:.1f}".format(const.compressive_strength/1000))
        self.inputDisruptionMassGrainRatio.setText("{:.2f}".format(const.disruption_mass_grain_ratio*100))
        self.inputDisruptionMassIndex.setText("{:.2f}".format(const.disruption_mass_index))
        self.inputDisruptionMassMinRatio.setText("{:.2f}".format(const.disruption_mass_min_ratio*100))
        self.inputDisruptionMassMaxRatio.setText("{:.2f}".format(const.disruption_mass_max_ratio*100))

        ### ###



    def checkBoxWakeSignal(self, event):
        """ Control what happens when the wake checkbox is pressed. """

        # Read the wake checkbox
        self.wake_on = self.checkBoxWake.isChecked()

        # Disable/enable inputs if the checkbox is checked/unchecked
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



    def checkBoxErosionSignal(self, event):
        """ Control what happens when the erosion checkbox is pressed. """

        # Read the wake checkbox
        self.const.erosion_on = self.checkBoxErosion.isChecked()

        # Disable/enable inputs if the checkbox is checked/unchecked
        self.inputErosionHtStart.setDisabled(not self.const.erosion_on)
        self.inputErosionCoeff.setDisabled(not self.const.erosion_on)
        self.inputErosionHtChange.setDisabled(not self.const.erosion_on)
        self.inputErosionCoeffChange.setDisabled(not self.const.erosion_on)
        self.inputErosionMassIndex.setDisabled(not self.const.erosion_on)
        self.inputErosionMassMin.setDisabled(not self.const.erosion_on)
        self.inputErosionMassMax.setDisabled(not self.const.erosion_on)

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



    def readInputBoxes(self):
        """ Read input boxes and set values to the Constants object. """


        def _tryReadFloat(input_box, value):
            try:
                value = float(input_box.text())
            except:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setWindowTitle("Input parsing error")
                msg.setText("Error reading input box " + input_box.objectName())
                msg.setInformativeText("Setting it back to: " + str(value))
                msg.exec_()

            return value

        
        ### Simulation params ###

        self.const.dt = _tryReadFloat(self.inputTimeStep, self.const.dt)
        self.const.P_0m = _tryReadFloat(self.inputP0M, self.const.P_0m)

        self.const.h_init = 1000*_tryReadFloat(self.inputHtInit, self.const.h_init/1000)
        self.const.m_kill = _tryReadFloat(self.inputMassKill, self.const.m_kill)
        self.const.v_kill = 1000*_tryReadFloat(self.inputVelKill, self.const.v_kill/1000)
        self.const.h_kill = 1000*_tryReadFloat(self.inputHtKill, self.const.h_kill/1000)

        ### ###


        ### Meteoroid physical properties ###

        self.const.rho = _tryReadFloat(self.inputRho, self.const.rho)
        self.const.rho_grain = _tryReadFloat(self.inputRhoGrain, self.const.rho_grain)
        self.const.m_init = _tryReadFloat(self.inputMassInit, self.const.m_init)
        self.const.sigma = _tryReadFloat(self.inputAblationCoeff, self.const.sigma*1e6)/1e6
        self.const.v_init = 1000*_tryReadFloat(self.inputVelInit, self.const.v_init/1000)
        self.const.shape_factor = _tryReadFloat(self.inputShapeFact, self.const.shape_factor)
        self.const.gamma = _tryReadFloat(self.inputGamma, self.const.gamma)
        self.const.zenith_angle = np.radians(_tryReadFloat(self.inputZenithAngle, \
            np.degrees(self.const.zenith_angle)))

        ### ###


        ### Wake parameters ###

        self.const.wake_psf = _tryReadFloat(self.inputWakePSF, self.const.wake_psf)
        self.const.wake_extension = _tryReadFloat(self.inputWakeExt, self.const.wake_extension)
        self.wake_plot_ht = 1000*_tryReadFloat(self.inputWakePlotHt, self.wake_plot_ht/1000)

        ### ###


        ### Erosion parameters ###

        self.const.erosion_height_start = 1000*_tryReadFloat(self.inputErosionHtStart, \
            self.const.erosion_height_start/1000)
        self.const.erosion_coeff = _tryReadFloat(self.inputErosionCoeff, self.const.erosion_coeff*1e6)/1e6
        self.const.erosion_height_change = 1000*_tryReadFloat(self.inputErosionHtChange, \
            self.const.erosion_height_change/1000)
        self.const.erosion_coeff_change = _tryReadFloat(self.inputErosionCoeffChange, \
            self.const.erosion_coeff_change*1e6)/1e6
        self.const.erosion_mass_index = _tryReadFloat(self.inputErosionMassIndex, \
            self.const.erosion_mass_index)
        self.const.erosion_mass_min = _tryReadFloat(self.inputErosionMassMin, self.const.erosion_mass_min)
        self.const.erosion_mass_max = _tryReadFloat(self.inputErosionMassMax, self.const.erosion_mass_max)

        ### ###



        ### Disruption parameters ###

        self.const.compressive_strength = 1000*_tryReadFloat(self.inputCompressiveStrength, \
            self.const.compressive_strength/1000)

        # If a different value for erosion coefficient after disruption should be used, read it
        if self.disruption_different_erosion_coeff:
            self.const.disruption_erosion_coeff = _tryReadFloat(self.inputDisruptionErosionCoeff, \
                self.const.disruption_erosion_coeff*1e6)/1e6
        else:
            # Otherwise, use the same value
            self.const.disruption_erosion_coeff = self.const.erosion_coeff


        self.const.disruption_mass_grain_ratio = _tryReadFloat(self.inputDisruptionMassGrainRatio, \
            self.const.disruption_mass_grain_ratio*100)/100
        self.const.disruption_mass_index = _tryReadFloat(self.inputDisruptionMassIndex, \
            self.const.disruption_mass_index)
        self.const.disruption_mass_min_ratio = _tryReadFloat(self.inputDisruptionMassMinRatio, \
            self.const.disruption_mass_min_ratio*100)/100
        self.const.disruption_mass_max_ratio = _tryReadFloat(self.inputDisruptionMassMaxRatio, \
            self.const.disruption_mass_max_ratio*100)/100

        ### ###


        # Update the boxes with read values
        self.updateInputBoxes()




    def updateCommonPlotFeatures(self, plot_handle, sr, plot_text=False):
        """ Update common features on all plots such as the erosion start. 

        Arguments:
            plot_handle: [PyQt5 plot handle]
            sr: [object] Simulation results.
        """


        # Get the plot X limits
        x_min, x_max = plot_handle.canvas.axes.get_xlim()


        # Generate array for horizontal line plotting
        x_arr = np.linspace(x_min, x_max, 10)


        # Plot a line marking erosion beginning
        plot_handle.canvas.axes.plot(x_arr, np.zeros_like(x_arr) + self.const.erosion_height_start/1000, \
            linestyle='dashed', color='k', alpha=0.25)

        # Add the text about erosion begin
        if plot_text:
            plot_handle.canvas.axes.text(x_min, TEXT_LABEL_HT_PAD + self.const.erosion_height_start/1000, \
                "Erosion beg", size=7, alpha=0.5)



        # Only plot the erosion change if it's above the meteor end
        if self.const.erosion_height_change > self.traj.rend_ele:

            # Plot a line marking erosion change
            plot_handle.canvas.axes.plot(x_arr, np.zeros_like(x_arr) + self.const.erosion_height_change/1000,\
                linestyle='dashed', color='k', alpha=0.25)

            # Add the text about erosion change
            if plot_text:
                plot_handle.canvas.axes.text(x_min, TEXT_LABEL_HT_PAD \
                    + self.const.erosion_height_change/1000, "Erosion change", size=7, alpha=0.5)



        # Plot the disruption height
        if self.const.disruption_on and (self.const.disruption_height is not None):

            plot_handle.canvas.axes.plot(x_arr, np.zeros_like(x_arr) + self.const.disruption_height/1000, \
                linestyle='dotted', color='k', alpha=0.5)

            # Add the text about disruption
            if plot_text:
                plot_handle.canvas.axes.text(x_min, TEXT_LABEL_HT_PAD + self.const.disruption_height/1000, \
                    "Disruption", size=7, alpha=0.5)



        # Force the plot X limits back
        plot_handle.canvas.axes.set_xlim([x_min, x_max])




    def updateMagnitudePlot(self, show_previous=False):
        """ Update the magnitude plot. 

        Arguments:
            None

        Keyword arguments:
            show_previous: [bool] Show the previous simulation. False by default.


        Return: 
            (plot_beg_ht, plot_end_ht): [tuple] A range of heights used for plotting the magnitude plot.
        """

        # Choose to show current or previous results
        if show_previous:
            sr = self.simulation_results_prev
        else:
            sr = self.simulation_results


        self.magnitudePlot.canvas.axes.clear()

        
        # Track plot limits
        plot_beg_ht = -np.inf
        plot_end_ht = np.inf

        mag_brightest = np.inf
        mag_faintest = -np.inf

        # Plot observed magnitudes from different stations
        for obs in self.traj.observations:

            # Extract data
            abs_mag_data = obs.absolute_magnitudes
            height_data = obs.model_ht/1000

            # Keep track of the height limits
            plot_beg_ht = max(plot_beg_ht, np.max(height_data))
            plot_end_ht = min(plot_end_ht, np.min(height_data))

            # Skip instances when no magnitudes are present
            if abs_mag_data is None:
                continue

            self.magnitudePlot.canvas.axes.plot(abs_mag_data, height_data, marker='x',
                linestyle='dashed', label=obs.station_id, markersize=5, linewidth=1)

            # Keep track of the faintest and the brightest magnitude
            mag_brightest = min(mag_brightest, np.min(abs_mag_data[~np.isinf(abs_mag_data)]))
            mag_faintest = max(mag_faintest, np.max(abs_mag_data[~np.isinf(abs_mag_data)]))
            


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

                # Keep track of the height limits
                plot_beg_ht = max(plot_beg_ht, np.max(height_data))
                plot_end_ht = min(plot_end_ht, np.min(height_data))



        # Plot simulated magnitudes
        if sr is not None:

            # # Cut the part with same beginning heights as observations
            # temp_arr = np.c_[sr.brightest_height_arr, sr.abs_magnitude]
            # temp_arr = temp_arr[(sr.brightest_height_arr < plot_beg_ht*1000) \
            #     & (sr.brightest_height_arr > plot_end_ht*1000)]
            # ht_arr, abs_mag_arr = temp_arr.T

            # Plot the simulated magnitudes
            self.magnitudePlot.canvas.axes.plot(sr.abs_magnitude, sr.brightest_height_arr/1000, \
                label='Simulated', color='k', alpha=0.5)



        self.magnitudePlot.canvas.axes.set_ylabel('Height (km)')
        self.magnitudePlot.canvas.axes.set_xlabel('Abs magnitude')

        self.magnitudePlot.canvas.axes.set_ylim([plot_end_ht + END_HT_PAD, plot_beg_ht + BEG_HT_PAD])
        self.magnitudePlot.canvas.axes.set_xlim([mag_brightest - 1, mag_faintest + 1])
        self.magnitudePlot.canvas.axes.invert_xaxis()

        # Plot common features across all plots
        self.updateCommonPlotFeatures(self.magnitudePlot, sr, plot_text=True)

        self.magnitudePlot.canvas.axes.legend()

        self.magnitudePlot.canvas.axes.grid(color="k", linestyle='dotted', alpha=0.3)

        self.magnitudePlot.canvas.axes.set_title('Magnitude')

        self.magnitudePlot.canvas.figure.tight_layout()

        self.magnitudePlot.canvas.draw()


        return plot_beg_ht, plot_end_ht



    def updateVelocityPlot(self, show_previous=False, plot_beg_ht=None, plot_end_ht=None):
        """ Update the velocity plot. 
        
        Arguments:
            None

        Keyword arguments:
            show_previous: [bool] Show the previous simulation. False by default.
            plot_beg_ht: [float] Beginning height for the plot before adding the padding.
            plot_end_ht: [float] End height for the plot before adding the padding.

        """

        # Choose to show current or previous results
        if show_previous:
            sr = self.simulation_results_prev
        else:
            sr = self.simulation_results


        self.velocityPlot.canvas.axes.clear()

        
        # Track plot limits
        if plot_beg_ht is None:
            plot_beg_ht = -np.inf

        if plot_end_ht is None:
            plot_end_ht = np.inf


        vel_min = np.inf
        vel_max = -np.inf

        # Plot observed velocities from different stations
        for obs in self.traj.observations:

            # Extract data
            vel_data = obs.velocities[1:]/1000
            height_data = obs.model_ht[1:]/1000

            # Keep track of the height limits
            plot_beg_ht = max(plot_beg_ht, np.max(height_data))
            plot_end_ht = min(plot_end_ht, np.min(height_data))

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

            # Plot the simulated velocity
            self.velocityPlot.canvas.axes.plot(sr.brightest_vel_arr/1000, sr.brightest_height_arr/1000, \
                label='Simulated', color='k', alpha=0.5)



            ### Compute the simulated average velocity ###

            # Select only the height range from observations
            sim_vel_obs_range = sr.brightest_vel_arr[(sr.brightest_height_arr <= self.traj.rbeg_ele) \
                & (sr.brightest_height_arr >= self.traj.rend_ele)]

            # Compute the simulated average velocity
            v_avg_sim = np.mean(sim_vel_obs_range)

            ### ###


            # Plot the simulated average velocity
            self.velocityPlot.canvas.axes.plot(np.zeros_like(avg_vel_ht_plot_arr) \
                + v_avg_sim/1000, avg_vel_ht_plot_arr, label="Vavg sim", linestyle='dotted', \
                color='k', alpha=0.5)



        self.velocityPlot.canvas.axes.set_ylabel('Height (km)')
        self.velocityPlot.canvas.axes.set_xlabel('Velocity (km/s)')

        self.velocityPlot.canvas.axes.set_ylim([plot_end_ht + END_HT_PAD, plot_beg_ht + BEG_HT_PAD])
        self.velocityPlot.canvas.axes.set_xlim([vel_min - 1, vel_max + 1])

        # Plot common features across all plots
        self.updateCommonPlotFeatures(self.velocityPlot, sr)

        self.velocityPlot.canvas.axes.legend()

        self.velocityPlot.canvas.axes.grid(color="k", linestyle='dotted', alpha=0.3)

        self.velocityPlot.canvas.axes.set_title('Velocity')

        self.velocityPlot.canvas.figure.tight_layout()

        self.velocityPlot.canvas.draw()



    def updateLagPlot(self, show_previous=False, plot_beg_ht=None, plot_end_ht=None):
        """ Update the lag plot. 
        
        Arguments:
            None

        Keyword arguments:
            show_previous: [bool] Show the previous simulation. False by default.
            plot_beg_ht: [float] Beginning height for the plot before adding the padding.
            plot_end_ht: [float] End height for the plot before adding the padding.

        """

        # Choose to show current of previous results
        if show_previous:
            sr = self.simulation_results_prev
        else:
            sr = self.simulation_results


        self.lagPlot.canvas.axes.clear()


        # Update the observed initial velocity label
        # NOTE: The ECI, not the ground-fixed velocity needs to be used, as the meteor model does not
        #   include Earth's rotation!
        self.vInitObsLabel.setText("Vinit obs = {:.3f} km/s".format(self.traj.orbit.v_init/1000))


        # Track plot limits
        if plot_beg_ht is None:
            plot_beg_ht = -np.inf

        if plot_end_ht is None:
            plot_end_ht = np.inf


        # Keep track of the height range
        for obs in self.traj.observations:

            height_data = obs.model_ht/1000

            self.lagPlot.canvas.axes.plot(obs.lag, height_data, marker='x',
                linestyle='dashed', label=obs.station_id, markersize=5, linewidth=1)

            # Keep track of the height limits
            plot_beg_ht = max(plot_beg_ht, np.max(height_data))
            plot_end_ht = min(plot_end_ht, np.min(height_data))


        # Plot additional observations from the .met file (if available)
        if self.met_obs is not None:

            # Plot additional lags for all sites (plot only mirfit lags)
            for site in self.met_obs.sites:

                height_data = self.met_obs.height_data[site]/1000

                # Only plot mirfit lags
                if self.met.mirfit:
                    self.lagPlot.canvas.axes.plot(self.met_obs.lag_data[site], height_data, marker='x',
                        linestyle='dashed', label=str(site),  markersize=5, linewidth=1)

                # Keep track of the height limits
                plot_beg_ht = max(plot_beg_ht, np.max(height_data))
                plot_end_ht = min(plot_end_ht, np.min(height_data))


        # Get X plot limits
        x_min, x_max = self.lagPlot.canvas.axes.get_xlim()


        # Plot simulated lag of the brightest point on the trajectory
        if sr is not None:

            # Get the model velocity at the observed beginning height
            sim_beg_ht_indx = np.argmin(np.abs(self.traj.rbeg_ele - sr.brightest_height_arr))
            v_init_sim = sr.brightest_vel_arr[sim_beg_ht_indx]

            # Update the simulated initial velocity label
            self.vInitSimLabel.setText("Vinit sim = {:.3f} km/s".format(v_init_sim/1000))


            # Cut the part with same beginning heights as observations
            temp_arr = np.c_[sr.brightest_height_arr, sr.brightest_length_arr]
            temp_arr = temp_arr[(sr.brightest_height_arr <= self.traj.rbeg_ele) \
                & (sr.brightest_height_arr >= plot_end_ht)]
            ht_arr, brightest_len_arr = temp_arr.T

            # Compute the simulated lag using the observed velocity
            lag_sim = brightest_len_arr - brightest_len_arr[0] - self.traj.orbit.v_init*np.arange(0, \
                self.const.dt*len(brightest_len_arr), self.const.dt)[:len(brightest_len_arr)]

            self.lagPlot.canvas.axes.plot(lag_sim[:len(ht_arr)], (ht_arr/1000)[:len(lag_sim)], 
                label='Simulated', color='k', alpha=0.5)



        self.lagPlot.canvas.axes.set_ylim([plot_end_ht + END_HT_PAD, plot_beg_ht + BEG_HT_PAD])
        self.lagPlot.canvas.axes.set_xlim([x_min, x_max])

        self.lagPlot.canvas.axes.set_xlabel('Lag (m)')
        self.lagPlot.canvas.axes.set_ylabel('Height (km)')
        
        # Plot common features across all plots
        self.updateCommonPlotFeatures(self.lagPlot, sr)

        self.lagPlot.canvas.axes.legend()

        self.lagPlot.canvas.axes.grid(color="k", linestyle='dotted', alpha=0.3)

        self.lagPlot.canvas.axes.set_title('Lag')

        self.lagPlot.canvas.figure.tight_layout()

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




    def redrawWakeHeightLine(self, plot_handle, line_handle, label_handle=None, draw_label=False):
        """ Plot lines on mag, vel, lag plots indicating where the wake is shown. The line will be just 
            refreshed to prevent redrawing all plots from scratch.
        """

        # Remove the line from the plot
        if line_handle is not None:
            line_handle.remove()


        # Get the plot limits
        x_min, x_max = plot_handle.canvas.axes.get_xlim()
        y_min, y_max = plot_handle.canvas.axes.get_ylim()

        # Construct X array
        x_arr = np.linspace(x_min, x_max)

        # Plot the wake line
        line_handle = plot_handle.canvas.axes.plot(x_arr, np.zeros_like(x_arr) + self.wake_plot_ht/1000, \
            linestyle='dashed', color='green', alpha=0.5)[0]



        # Plot the label if given
        if draw_label:

            # Remove the old label handle
            if label_handle is not None:
                label_handle.remove()

            # Draw the wake line label
            label_handle = plot_handle.canvas.axes.text(x_min, TEXT_LABEL_HT_PAD + self.wake_plot_ht/1000, \
                "Wake plot", size=7, color='green', alpha=0.5)


        # Keep the plot limits
        plot_handle.canvas.axes.set_xlim([x_min, x_max])
        plot_handle.canvas.axes.set_ylim([y_min, y_max])

        # Redraw the plot
        plot_handle.canvas.draw_idle()


        return line_handle, label_handle




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
        self.magnitudePlotWakeLine, self.magnitudePlotWakeLineLabel = self.redrawWakeHeightLine(self.magnitudePlot, \
            self.magnitudePlotWakeLine, label_handle=self.magnitudePlotWakeLineLabel, draw_label=True)
        self.velocityPlotWakeLine, _ = self.redrawWakeHeightLine(self.velocityPlot, self.velocityPlotWakeLine)
        self.lagPlotWakeLine, _ = self.redrawWakeHeightLine(self.lagPlot, self.lagPlotWakeLine)



        # Find the appropriate observed wake to plot for the given height
        if self.wake_heights is not None:

            # Find the index of the observed wake that's closest to the given plot height
            self.wake_ht_current_index = np.argmin(np.abs(np.array([w[0] for w in self.wake_heights]) \
                - self.wake_plot_ht))

            # Extract the wake height and observations
            self.wake_plot_ht, self.current_wake_container = self.wake_heights[self.wake_ht_current_index]



        self.wakePlot.canvas.axes.clear()


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

                # Plot the simulated wake
                self.wakePlot.canvas.axes.plot(wake.length_array, wake.wake_luminosity_profile, \
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
                wake_intensity_array *= simulated_peak_luminosity/np.max(wake_intensity_array)


            # Perform alignments when simulations are available
            if wake is not None:

                # Align wake by peaks
                if self.wake_align_method == 'peak':

                    # Find the length of the peak intensity
                    peak_len = len_array[np.argmax(wake_intensity_array)]

                    # Offset lengths
                    len_array -= peak_len + simulated_peak_length


                # Align the wake by cross correlation
                elif self.wake_align_method == 'correlate':


                    # Interpolate the model values and sample them at observed points
                    sim_wake_interp = scipy.interpolate.interp1d(wake.length_array, \
                        wake.wake_luminosity_profile, bounds_error=False, fill_value=0)
                    model_wake_obs_len_sample = sim_wake_interp(-len_array)

                    # Correlate the wakes and find the shift
                    wake_shift = np.argmax(np.correlate(model_wake_obs_len_sample, wake_intensity_array, \
                        "full")) + 1

                    # Find the index of the zero observed length
                    obs_len_zero_indx = np.argmin(np.abs(len_array))

                    # Add the offset to the observed length
                    len_array += len_array[(obs_len_zero_indx + wake_shift)%len(model_wake_obs_len_sample)]



            # Plot the observed wake
            self.wakePlot.canvas.axes.plot(-len_array, wake_intensity_array,
                label='Observed, site: {:s}'.format(str(self.current_wake_container.site_id)), color='k', \
                linestyle='dotted')

        ### ###

        self.wakePlot.canvas.axes.legend()


        self.wakePlot.canvas.axes.set_xlabel('Length behind leading fragment')
        self.wakePlot.canvas.axes.set_ylabel('Intensity')

        self.wakePlot.canvas.axes.invert_xaxis()

        self.wakePlot.canvas.axes.set_ylim(bottom=0)

        self.wakePlot.canvas.axes.set_title('Wake')

        self.wakePlot.canvas.figure.tight_layout()

        self.wakePlot.canvas.draw()


        # Enable/disable wake normalization and alignment dpending on availability of simulated data
        self.wakeNormalizeGroup.setDisabled(wake is None)
        self.wakeAlignGroup.setDisabled(wake is None)



    def showPreviousResults(self):
        """ Show previous simulation results and parameters. """

        if self.simulation_results_prev is not None:

            self.updateInputBoxes(show_previous=True)
            plot_beg_ht, plot_end_ht = self.updateMagnitudePlot(show_previous=True)
            self.updateVelocityPlot(show_previous=True, plot_beg_ht=plot_beg_ht, plot_end_ht=plot_end_ht)
            self.updateLagPlot(show_previous=True, plot_beg_ht=plot_beg_ht, plot_end_ht=plot_end_ht)
            self.updateWakePlot(show_previous=True)



    def showCurrentResults(self):
        """ Show current simulation results and parameters. """

        self.updateInputBoxes(show_previous=False)
        plot_beg_ht, plot_end_ht = self.updateMagnitudePlot(show_previous=False)
        self.updateVelocityPlot(show_previous=False, plot_beg_ht=plot_beg_ht, plot_end_ht=plot_end_ht)
        self.updateLagPlot(show_previous=False, plot_beg_ht=plot_beg_ht, plot_end_ht=plot_end_ht)
        self.updateWakePlot(show_previous=False)




    def runSimulationGUI(self):

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
        results_list, wake_results = runSimulation(self.const, compute_wake=self.wake_on)

        sim_runtime = time.time() - t1

        if sim_runtime < 0.5:
            print('Simulation runtime: {:d} ms'.format(int(1000*sim_runtime)))
        elif sim_runtime < 100:
            print('Simulation runtime: {:.2f} s'.format(sim_runtime))
        else:
            print('Simulation runtime: {:.2f} min'.format(sim_runtime/60))

        # Store simulation results
        self.simulation_results = SimulationResults(self.const, results_list, wake_results)

        # Update the plots
        self.showCurrentResults()

        # Save the latest run parameters
        self.saveFitParameters(False, suffix="_latest")

        # Enable the simulation button
        self.runSimButton.setDisabled(False)
        self.runSimButton.setStyleSheet("background-color: #b1eea6")



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
        file_name = file_name.replace('trajectory.pickle', '') + 'report_sim.txt'

        # Save the report with updated orbit
        traj_updated.saveReport(dir_path, file_name, uncertanties=self.traj.uncertanties, verbose=False)



    def saveFitParameters(self, event, suffix=None):
        """ Save fit parameters to a JSON file. """

        if suffix is None:
            suffix = str("")

        dir_path, file_name = os.path.split(self.traj_path)
        file_name = file_name.replace('trajectory.pickle', '') + "sim_fit{:s}.json".format(suffix)

        file_path = os.path.join(dir_path, file_name)
        with open(file_path, 'w') as f:
            json.dump(self.const, f, default=lambda o: o.__dict__, indent=4)

        print("Saved fit parameters to:", file_path)




    def saveVideo(self, event):
        """ Generate video frames using the simulated wake and PSF. """


        # Skip saving video if there is no wake results
        if self.simulation_results.wake_results is None:
            return False


        # Disable the video button
        self.wakeSaveVideoButton.setStyleSheet("background-color: red")
        self.wakeSaveVideoButton.setDisabled(True)
        self.repaint()


        # The plate scale is fixed at 0.5 meters per pixel at 100 km, so the animation is better visible
        plate_scale = 0.5

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
        help=".pickle file with the trajectory solution and the magnitudes.")

    arg_parser.add_argument('-l', '--load', metavar='LOAD_JSON', \
        help='Load JSON file with fit parameters.', type=str)

    arg_parser.add_argument('-m', '--met', metavar='MET_FILE', \
        help='Load additional observations from a METAL or mirfit .met file.', type=str)

    arg_parser.add_argument('-w', '--wid', metavar='WID_FILES', \
        help='Load mirfit wid files which will be used for wake plotting. Wildchars can be used, e.g. /path/wid*.txt.', 
        type=str, nargs='+')

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    # Init PyQt5 window
    app = QApplication([])


    # Init the MetSimGUI application
    main_window = MetSimGUI(os.path.abspath(cml_args.traj_pickle), const_json_file=cml_args.load, \
        met_path=cml_args.met, wid_files=cml_args.wid)


    main_window.show()

    sys.exit(app.exec_())