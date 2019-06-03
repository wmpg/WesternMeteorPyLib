""" GUI interface to meteor ablation models which enables manual modelling of meteors. """


import os
import sys
import copy
import argparse
import time

import numpy as np
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.uic import loadUi


from wmpl.MetSim.MetSimErosion import runSimulation, Constants
from wmpl.Utils.Math import averageClosePoints
from wmpl.Utils.Physics import calcMass
from wmpl.Utils.Pickling import loadPickle



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




class MetSimGUI(QMainWindow):
    
    def __init__(self, traj):
        """ GUI tool for MetSim. """
            

        self.traj = traj


        ### Init GUI ###

        QMainWindow.__init__(self)

        # Load the GUI design file
        loadUi(os.path.join(os.path.dirname(__file__), "GUI.ui"), self)

        self.setWindowTitle("MetSim")

        ### ###




        ### Add key bindings ###

        self.runSimButton.clicked.connect(self.runSimulationGUI)
        self.showPreviousButton.pressed.connect(self.showPreviousResults)
        self.showPreviousButton.released.connect(self.showCurrentResults)

        self.wakePlotUpdateButton.clicked.connect(self.updateWakePlot)
        self.wakeIncrementPlotHeightButton.clicked.connect(self.incrementWakePlotHeight)
        self.wakeDecrementPlotHeightButton.clicked.connect(self.decrementWakePlotHeight)

        #self.addToolBar(NavigationToolbar(self.magnitudePlot.canvas, self))


        self.checkBoxWake.stateChanged.connect(self.checkBoxWakeSignal)
        self.checkBoxErosion.stateChanged.connect(self.checkBoxErosionSignal)
        self.checkBoxDisruption.stateChanged.connect(self.checkBoxDisruptionSignal)

        ### ###



        ### Init simulation parameters ###

        # Init the constants
        self.const = Constants()


        # Set the constants value from the trajectory
        self.const.zenith_angle = traj.orbit.zc
        self.const.v_init = traj.orbit.v_init_norot

        # Set kill height to the observed end height
        self.const.h_kill = traj.rend_ele - 3000

        # Set erosion height to the beginning height
        self.const.erosion_height = traj.rbeg_ele


        # Calculate the photometric mass
        self.const.m_init = self.calcPhotometricMass()

        ### ###


        ### Define GUI and simulation attributes ###

        self.wake_on = False
        self.wake_plot_ht = self.traj.rbeg_ele # m

        # Disable erosion and disruption at the beginning
        self.const.erosion_on = False
        self.const.disruption_on = False


        self.simulation_results = None

        self.const_prev = None
        self.simulation_results_prev = None

        ### ###


        # Update the values in the input boxes
        self.updateInputBoxes()

        # Update checkboxes
        self.checkBoxWakeSignal(None)
        self.checkBoxErosionSignal(None)
        self.checkBoxDisruptionSignal(None)

        # Update plots
        self.updateMagnitudePlot()
        self.updateLagPlot()
        self.updateWakePlot()



    def calcPhotometricMass(self):
        """ Calculate photometric mass from given magnitude data. """

        time_mag_arr = []
        avg_t_diff_max = 0
        for obs in self.traj.observations:

            # Compute average time difference
            avg_t_diff_max = max(avg_t_diff_max, np.median(obs.time_data[1:] - obs.time_data[:-1]))

            for t, mag in zip(obs.time_data, obs.absolute_magnitudes):
                if mag is not None:
                    time_mag_arr.append([t, mag])

        # Sort array by time
        time_mag_arr = np.array(sorted(time_mag_arr, key=lambda x: x[0]))

        time_arr, mag_arr = time_mag_arr.T

        
        # Average out the magnitudes
        time_arr, mag_arr = averageClosePoints(time_arr, mag_arr, avg_t_diff_max)

        # Compute the photometry mass
        return calcMass(np.array(time_arr), np.array(mag_arr), self.traj.orbit.v_avg_norot, P_0m=self.const.P_0m)


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

        self.inputWakePSF.setText("{:d}".format(int(const.wake_psf)))
        self.inputWakeExt.setText("{:d}".format(int(const.wake_extension)))
        self.inputWakePlotHt.setText("{:.3f}".format(self.wake_plot_ht/1000))

        ### ###


        ### Erosion parameters ###

        self.checkBoxErosion.setChecked(const.erosion_on)

        self.inputErosionHt.setText("{:.3f}".format(const.erosion_height/1000))
        self.inputErosionCoeff.setText("{:.3f}".format(const.erosion_coeff*1e6))
        self.inputErosionMassIndex.setText("{:.2f}".format(const.erosion_mass_index))
        self.inputErosionMassMin.setText("{:.2e}".format(const.erosion_mass_min))
        self.inputErosionMassMax.setText("{:.2e}".format(const.erosion_mass_max))

        ### ###


        ### Disruption parameters ###

        self.checkBoxDisruption.setChecked(const.disruption_on)

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

        # Read inputs
        self.readInputBoxes()



    def checkBoxErosionSignal(self, event):
        """ Control what happens when the erosion checkbox is pressed. """

        # Read the wake checkbox
        self.const.erosion_on = self.checkBoxErosion.isChecked()

        # Disable/enable inputs if the checkbox is checked/unchecked
        self.inputErosionHt.setDisabled(not self.const.erosion_on)
        self.inputErosionCoeff.setDisabled(not self.const.erosion_on)
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
        self.inputCompressiveStrength.setDisabled(not self.const.disruption_on)
        self.inputDisruptionMassGrainRatio.setDisabled(not self.const.disruption_on)
        self.inputDisruptionMassIndex.setDisabled(not self.const.disruption_on)
        self.inputDisruptionMassMinRatio.setDisabled(not self.const.disruption_on)
        self.inputDisruptionMassMaxRatio.setDisabled(not self.const.disruption_on)

        # Read inputs
        self.readInputBoxes()




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

        self.const.erosion_height = 1000*_tryReadFloat(self.inputErosionHt, self.const.erosion_height/1000)
        self.const.erosion_coeff = _tryReadFloat(self.inputErosionCoeff, self.const.erosion_coeff*1e6)/1e6
        self.const.erosion_mass_index = _tryReadFloat(self.inputErosionMassIndex, \
            self.const.erosion_mass_index)
        self.const.erosion_mass_min = _tryReadFloat(self.inputErosionMassMin, self.const.erosion_mass_min)
        self.const.erosion_mass_max = _tryReadFloat(self.inputErosionMassMax, self.const.erosion_mass_max)

        ### ###



        ### Disruption parameters ###

        self.const.compressive_strength = 1000*_tryReadFloat(self.inputCompressiveStrength, \
            self.const.compressive_strength/1000)
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



    def updateMagnitudePlot(self, show_previous=False):
        """ Update the magnitude plot. """

        # Choose to show current or previous results
        if show_previous:
            sr = self.simulation_results_prev
        else:
            sr = self.simulation_results


        self.magnitudePlot.canvas.axes.clear()

        
        # Set height plot limits
        plot_beg_ht = self.traj.rbeg_ele + 5000
        plot_end_ht = self.traj.rend_ele - 2000


        mag_brightest = np.inf
        mag_faintest = -np.inf

        # Plot observed magnitudes from different stations
        for obs in traj.observations:

            self.magnitudePlot.canvas.axes.plot(obs.absolute_magnitudes, obs.model_ht/1000, marker='x',
                linestyle='dashed', label=obs.station_id)

            # Keep track of the faintest and the brightest magnitude
            mag_brightest = min(mag_brightest, np.min(obs.absolute_magnitudes))
            mag_faintest = max(mag_faintest, np.max(obs.absolute_magnitudes))


        # Plot simulated magnitudes
        if sr is not None:

            # Cut the part with same beginning heights as observations
            temp_arr = np.c_[sr.brightest_height_arr, sr.abs_magnitude]
            temp_arr = temp_arr[(sr.brightest_height_arr < plot_beg_ht) \
                & (sr.brightest_height_arr > plot_end_ht)]
            ht_arr, abs_mag_arr = temp_arr.T

            # Plot the simulated magnitudes
            self.magnitudePlot.canvas.axes.plot(abs_mag_arr, ht_arr/1000, label='Simulated')



        self.magnitudePlot.canvas.axes.set_ylabel('Height (km)')
        self.magnitudePlot.canvas.axes.set_xlabel('Abs magnitude')

        self.magnitudePlot.canvas.axes.set_ylim([plot_end_ht/1000, plot_beg_ht/1000])
        self.magnitudePlot.canvas.axes.set_xlim([mag_faintest + 1, mag_brightest - 1])

        self.magnitudePlot.canvas.axes.legend()

        self.magnitudePlot.canvas.axes.set_title('Magnitude')

        self.magnitudePlot.canvas.draw()



    def updateLagPlot(self, show_previous=False):
        """ Update the lag plot. """

        # Choose to show current of previous results
        if show_previous:
            sr = self.simulation_results_prev
        else:
            sr = self.simulation_results


        self.lagPlot.canvas.axes.clear()

        # Set height plot limits
        plot_beg_ht = self.traj.rbeg_ele + 5000
        plot_end_ht = self.traj.rend_ele - 2000


        # Update the observed initial velocity label
        self.vInitObsLabel.setText("Vinit obs = {:.3f} km/s".format(traj.orbit.v_init_norot/1000))


        # Plot observed magnitudes from different stations
        for obs in traj.observations:

            self.lagPlot.canvas.axes.plot(obs.lag, obs.model_ht/1000, marker='x',
                linestyle='dashed', label=obs.station_id)


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
            lag_sim = brightest_len_arr - brightest_len_arr[0] - traj.orbit.v_init_norot*np.arange(0, \
                self.const.dt*len(brightest_len_arr), self.const.dt)

            self.lagPlot.canvas.axes.plot(lag_sim, ht_arr/1000, label='Simulated')


        self.lagPlot.canvas.axes.set_ylim([plot_end_ht/1000, plot_beg_ht/1000])


        self.lagPlot.canvas.axes.set_xlabel('Lag (m)')
        self.lagPlot.canvas.axes.set_ylabel('Height (km)')
        

        self.lagPlot.canvas.axes.legend()

        self.lagPlot.canvas.axes.set_title('Lag')

        self.lagPlot.canvas.draw()


    def incrementWakePlotHeight(self):
        """ Increment wake plot height by 100 m. """

        self.wake_plot_ht += 100
        self.updateInputBoxes()
        self.updateWakePlot()


    def decrementWakePlotHeight(self):
        """ Decrement wake plot height by 100 m. """

        self.wake_plot_ht -= 100
        self.updateInputBoxes()
        self.updateWakePlot()



    def updateWakePlot(self, show_previous=False):
        """ Plot the wake. """

        self.readInputBoxes()


        # Choose to show current of previous results
        if show_previous:
            sr = self.simulation_results_prev
        else:
            sr = self.simulation_results


        self.wakePlot.canvas.axes.clear()


        # Plot observed wake
        # ...


        # Plot simulated wake
        if sr is not None:

            # Find the wake index closest to the given wake height
            wake_res_indx =  np.argmin(np.abs(self.wake_plot_ht - sr.brightest_height_arr))

            # Get the approprate wake results
            wake = sr.wake_results[wake_res_indx]

            if wake is not None:

                self.wakePlot.canvas.axes.plot(wake.length_array, wake.wake_luminosity_profile, label='Simulated')

                self.lagPlot.canvas.axes.set_ylim([0, sr.wake_max_lum])


        


        self.wakePlot.canvas.axes.set_xlabel('Length behind leading fragment')
        self.wakePlot.canvas.axes.set_ylabel('Intensity')

        # self.wakePlot.canvas.axes.legend()

        self.wakePlot.canvas.axes.set_title('Wake')

        self.wakePlot.canvas.draw()



    def showPreviousResults(self):
        """ Show previous simulation results and parameters. """

        if self.simulation_results_prev is not None:

            self.updateInputBoxes(show_previous=True)
            self.updateMagnitudePlot(show_previous=True)
            self.updateLagPlot(show_previous=True)
            self.updateWakePlot(show_previous=True)



    def showCurrentResults(self):
        """ Show current simulation results and parameters. """

        self.updateInputBoxes(show_previous=False)
        self.updateMagnitudePlot(show_previous=False)
        self.updateLagPlot(show_previous=False)
        self.updateWakePlot(show_previous=False)




    def runSimulationGUI(self):

        # Store previous run results
        self.const_prev = copy.deepcopy(self.const)
        self.simulation_results_prev = copy.deepcopy(self.simulation_results)


        # Read the values from the input boxes
        self.readInputBoxes()


        print('Running simulation...')
        t1 = time.time()

        # Run the simulation
        results_list, wake_results = runSimulation(self.const, compute_wake=self.wake_on)

        print('Simulation runtime: {:d} ms'.format(int(1000*(time.time() - t1))))

        # Store simulation results
        self.simulation_results = SimulationResults(self.const, results_list, wake_results)


        self.updateMagnitudePlot()
        self.updateLagPlot()
        self.updateWakePlot()




if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Run meteor ablation modelling using the given trajectory file.")

    arg_parser.add_argument('traj_pickle', metavar='TRAJ_PICKLE', type=str, \
        help=".pickle file with the trajectory solution and the magnitudes.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################



    # Load the trajectory pickle file
    traj = loadPickle(*os.path.split(os.path.abspath(cml_args.traj_pickle)))


    app = QApplication([])

    main_window = MetSimGUI(traj)

    main_window.show()

    sys.exit(app.exec_())