""" GUI interface to meteor ablation models which enables manual modelling of meteors. """


import os
import sys
import argparse
import time

import numpy as np
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.uic import loadUi


from wmpl.MetSim.MetSimErosion import runSimulation, Constants
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
        self.const.v_init = traj.v_init


        ### ###


        ### Define GUI and simulation attributes ###

        self.wake_on = False
        self.wake_plot_ht = 100000 # m

        # Disable erosion and disruption at the beginning
        self.const.erosion_on = False
        self.const.disruption_on = False


        self.simulation_results = None

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




    def updateInputBoxes(self):
        """ Update input boxes with values from the Constants object. """

        
        ### Simulation params ###

        self.inputTimeStep.setText(str(self.const.dt))
        self.inputHtInit.setText("{:.3f}".format(self.const.h_init/1000))
        self.inputP0M.setText("{:d}".format(int(self.const.P_0m)))
        self.inputMassKill.setText("{:.1e}".format(self.const.m_kill))
        self.inputVelKill.setText("{:.3f}".format(self.const.v_kill/1000))
        self.inputHtKill.setText("{:.3f}".format(self.const.h_kill/1000))

        ### ###


        ### Meteoroid physical properties ###

        self.inputRho.setText("{:d}".format(int(self.const.rho)))
        self.inputRhoGrain.setText("{:d}".format(int(self.const.rho_grain)))
        self.inputMassInit.setText("{:.1e}".format(self.const.m_init))
        self.inputAblationCoeff.setText("{:.3f}".format(self.const.sigma*1e6))
        self.inputVelInit.setText("{:.3f}".format(self.const.v_init/1000))
        self.inputShapeFact.setText("{:.2f}".format(self.const.shape_factor))
        self.inputGamma.setText("{:.1f}".format(self.const.gamma))
        self.inputZenithAngle.setText("{:.3f}".format(np.degrees(self.const.zenith_angle)))

        ### ###


        ### Wake parameters ###

        self.checkBoxWake.setChecked(self.wake_on)

        self.inputWakePSF.setText("{:d}".format(int(self.const.wake_psf)))
        self.inputWakeExt.setText("{:d}".format(int(self.const.wake_extension)))
        self.inputWakePlotHt.setText("{:.3f}".format(self.wake_plot_ht/1000))

        ### ###


        ### Erosion parameters ###

        self.checkBoxErosion.setChecked(self.const.erosion_on)

        self.inputErosionHt.setText("{:.3f}".format(self.const.erosion_height/1000))
        self.inputErosionCoeff.setText("{:.3f}".format(self.const.erosion_coeff*1e6))
        self.inputErosionMassIndex.setText("{:.2f}".format(self.const.erosion_mass_index))
        self.inputErosionMassMin.setText("{:.2e}".format(self.const.erosion_mass_min))
        self.inputErosionMassMax.setText("{:.2e}".format(self.const.erosion_mass_max))

        ### ###


        ### Disruption parameters ###

        self.checkBoxDisruption.setChecked(self.const.disruption_on)

        self.inputCompressiveStrength.setText("{:.1f}".format(self.const.compressive_strength/1000))
        self.inputDisruptionMassGrainRatio.setText("{:.2f}".format(self.const.disruption_mass_grain_ratio))
        self.inputDisruptionMassIndex.setText("{:.2f}".format(self.const.disruption_mass_index))
        self.inputDisruptionMassMinRatio.setText("{:.4f}".format(self.const.disruption_mass_min_ratio))
        self.inputDisruptionMassMaxRatio.setText("{:.4f}".format(self.const.disruption_mass_max_ratio))

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
            self.const.disruption_mass_grain_ratio)
        self.const.disruption_mass_index = _tryReadFloat(self.inputDisruptionMassIndex, \
            self.const.disruption_mass_index)
        self.const.disruption_mass_min_ratio = _tryReadFloat(self.inputDisruptionMassMinRatio, \
            self.const.disruption_mass_min_ratio)
        self.const.disruption_mass_max_ratio = _tryReadFloat(self.inputDisruptionMassMaxRatio, \
            self.const.disruption_mass_max_ratio)

        ### ###


        # Update the boxes with read values
        self.updateInputBoxes()



    def updateMagnitudePlot(self):
        """ Update the magnitude plot. """



        self.magnitudePlot.canvas.axes.clear()

        
        # Set height plot limits
        plot_beg_ht = self.traj.rbeg_ele + 5000
        plot_end_ht = self.traj.rend_ele - 2000


        mag_brightest = np.inf
        mag_faintest = -np.inf

        # Plot observed magnitudes from different stations
        for obs in traj.observations:

            self.magnitudePlot.canvas.axes.plot(obs.model_ht/1000, obs.absolute_magnitudes, marker='x',
                linestyle='dashed', label=obs.station_id)

            # Keep track of the faintest and the brightest magnitude
            mag_brightest = min(mag_brightest, np.min(obs.absolute_magnitudes))
            mag_faintest = max(mag_faintest, np.max(obs.absolute_magnitudes))


        # Plot simulated magnitudes
        if self.simulation_results is not None:

            sr = self.simulation_results

            # Cut the part with same beginning heights as observations
            temp_arr = np.c_[sr.brightest_height_arr, sr.abs_magnitude]
            temp_arr = temp_arr[(sr.brightest_height_arr < plot_beg_ht) \
                & (sr.brightest_height_arr > plot_end_ht)]
            ht_arr, abs_mag_arr = temp_arr.T

            # Plot the simulated magnitudes
            self.magnitudePlot.canvas.axes.plot(ht_arr/1000, abs_mag_arr, label='Simulated')



        self.magnitudePlot.canvas.axes.invert_yaxis()

        self.magnitudePlot.canvas.axes.set_xlabel('Height (km)')
        self.magnitudePlot.canvas.axes.set_ylabel('Abs magnitude')

        self.magnitudePlot.canvas.axes.set_xlim([plot_beg_ht/1000, plot_end_ht/1000])
        self.magnitudePlot.canvas.axes.set_ylim([mag_faintest + 1, mag_brightest - 1])

        self.magnitudePlot.canvas.axes.legend()

        self.magnitudePlot.canvas.axes.set_title('Magnitude')

        self.magnitudePlot.canvas.draw()



    def updateLagPlot(self):
        """ Update the lag plot. """


        self.lagPlot.canvas.axes.clear()

        # Set height plot limits
        plot_beg_ht = self.traj.rbeg_ele + 5000
        plot_end_ht = self.traj.rend_ele - 2000


        # Update the observed initial velocity label
        self.vInitObsLabel.setText("Vinit obs = {:.3f} km/s".format(traj.v_init/1000))


        # Plot observed magnitudes from different stations
        for obs in traj.observations:

            self.lagPlot.canvas.axes.plot(obs.lag, obs.model_ht/1000, marker='x',
                linestyle='dashed', label=obs.station_id)


        # Plot simulated lag of the brightest point on the trajectory
        if self.simulation_results is not None:

            sr = self.simulation_results

            # Get the model velocity at the observed beginning height
            sim_beg_ht_indx = np.argmin(np.abs(self.traj.rbeg_ele - sr.brightest_height_arr))
            v_init_sim = sr.brightest_vel_arr[sim_beg_ht_indx]

            # Update the simulated initial velocity label
            self.vInitSimLabel.setText("Vinit sim = {:.3f} km/s".format(v_init_sim/1000))


            # Cut the part with same beginning heights as observations
            temp_arr = np.c_[sr.brightest_height_arr, sr.brightest_length_arr]
            temp_arr = temp_arr[(sr.brightest_height_arr < plot_beg_ht) \
                & (sr.brightest_height_arr > plot_end_ht)]
            ht_arr, brightest_len_arr = temp_arr.T

            # Find the 

            # Compute the simulated lag using observed initial velocity




        #self.lagPlot.canvas.axes.invert_yaxis()

        self.lagPlot.canvas.axes.set_ylim([plot_end_ht/1000, plot_beg_ht/1000])


        self.lagPlot.canvas.axes.set_xlabel('Lag (m)')
        self.lagPlot.canvas.axes.set_ylabel('Height (km)')
        

        self.lagPlot.canvas.axes.legend()

        self.lagPlot.canvas.axes.set_title('Lag')

        self.lagPlot.canvas.draw()




    def runSimulationGUI(self):


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