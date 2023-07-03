""" SimulationResults class used by GUI interface to meteor ablation models which enables manual modelling of meteors. """
import os
import numpy as np
import copy
import scipy.stats
import scipy.interpolate
import scipy.optimize
import scipy.signal
try:
    from PyQt5.QtWidgets import QMessageBox
    gotQT = True
except:
    gotQT = False


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
                        np.array(frag_entry.grains_time_data))

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
            self.wake_max_lum = max([max(wake.wake_luminosity_profile) for wake in wake_results 
                if wake is not None])


        ###


    def writeCSV(self, dir_path, file_name):

        # Combine data into one array
        out_arr = np.c_[
            self.time_arr,
            self.brightest_height_arr/1000, self.brightest_length_arr/1000, self.brightest_vel_arr/1000, 
            self.leading_frag_height_arr/1000, self.leading_frag_length_arr/1000, 
            self.leading_frag_vel_arr/1000, self.leading_frag_dyn_press_arr/1e6,
            self.main_height_arr/1000, self.main_length_arr/1000, self.main_vel_arr/1000, 
            self.main_dyn_press_arr/1e6,
            self.tau_total_arr, self.tau_main_arr, self.tau_eroded_arr,
            self.abs_magnitude, self.abs_magnitude_main, self.abs_magnitude_eroded,
            np.log10(self.luminosity_arr), np.log10(self.luminosity_main_arr), np.log10(self.luminosity_eroded_arr), 
            np.log10(self.electron_density_total_arr),
            self.mass_total_active_arr, self.main_mass_arr]

        header = "B = brightest mass bin, L = leading fragment, M = main\n"
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
            if gotQT:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setWindowTitle("File permission error")
                msg.setText("Cannot save file " + os.path.join(dir_path, file_name))
                msg.setInformativeText("Make sure you have write permissions, or close the file if it's open in another program.")
                msg.exec_()
            else:
                print("File permission error")
                print("Cannot save file " + os.path.join(dir_path, file_name))
                print("Make sure you have write permissions, or close the file if it's open in another program.")
            return None


        with open(os.path.join(dir_path, file_name), 'w') as f:

            # Write the data
            np.savetxt(f, out_arr, fmt='%.5e', delimiter=',', newline='\n', header=header, comments="# ")
