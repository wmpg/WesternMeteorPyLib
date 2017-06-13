""" Runs Romulan data through Gural and MC solver. """

from __future__ import print_function, division, absolute_import


import os
import shutil


from Formats.Met import loadMet, solveTrajectoryMet
from Utils.OSTools import mkdirP


if __name__ == "__main__":

    romulan_data_dir = "../Romulan2012Geminids"


    started = False

    for romulan_state_file in sorted(os.listdir(romulan_data_dir)):

        romulan_state_path = os.path.abspath(os.path.join(romulan_data_dir, romulan_state_file))

        if ('.met' in romulan_state_file) and os.path.isfile(romulan_state_path):

            # # Continue from the given .met file
            # if (not '20121215_072101_A_RR.met' in romulan_state_file) and not started:
            #     continue

            # Do just the given .met file
            if not "20121213_001806_A_RR" in romulan_state_file:
                continue


            started = True

            print("#########################################################################################")
            print('Solving:', romulan_state_file)
            print("#########################################################################################")

            # Make a directory for the solution
            romulan_solution_path = os.path.splitext(romulan_state_path)[0]
            mkdirP(romulan_solution_path)

            # Copy the met file to the solution directory
            shutil.copy2(romulan_state_path, os.path.join(romulan_solution_path, romulan_state_file))

            # Load data from the .met file
            met = loadMet(romulan_solution_path, romulan_state_file, mirfit=False)

            # Run the trajectory solver
            solveTrajectoryMet(met, solver='original', show_plots=False, mc_pick_multiplier=2)
            
            
