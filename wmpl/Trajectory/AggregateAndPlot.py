""" Aggregate trajectory results into one results file and generate plots of trajectory results. """

from __future__ import print_function, division, absolute_import


import os

import numpy as np
import matplotlib.pyplot as plt


from wmpl.Utils.Pickling import loadPickle
from wmpl.Utils.PlotCelestial import CelestialPlot


def writeOrbitSummaryFile(dir_path, traj_list):
    """ Given a list of trajectory files, generate CSV file with the orbit summary. """

    pass



def generateTrajectoryPlots(dir_path, traj_list):
    """ Given a path with trajectory .pickle files, generate orbits plots. """


    ### PLOT SUN-CENTERED GEOCENTRIC ECLIPTIC RADIANTS ###

    fig = plt.figure(figsize=(16, 8), facecolor='k')

    lambda_list = []
    beta_list = []
    vg_list = []

    for traj in traj_list:

        # Compute Sun-centered longitude
        lambda_list.append(traj.orbit.L_g - traj.orbit.la_sun)

        beta_list.append(traj.orbit.B_g)
        vg_list.append(traj.orbit.v_g/1000)


    # Init the allsky plot
    celes_plot = CelestialPlot(lambda_list, beta_list, projection='sinu', lon_0=270, ax=fig.gca())

    # Mark sources
    sources_lg = np.radians([0, 270, 180])
    sources_bg = np.radians([0, 0, 0])
    sources_labels = ["Helion", "Apex", "Antihelion"]
    celes_plot.scatter(sources_lg, sources_bg, marker='x', s=15, c='0.75')

    # Convert angular coordinates to image coordinates
    x_list, y_list = celes_plot.m(np.degrees(sources_lg), np.degrees(sources_bg + np.radians(2.0)))
    for x, y, lbl in zip(x_list, y_list, sources_labels):
        plt.text(x, y, lbl, color='0.5', ha='center')


    # Compute the dot size which varies by the number of data points
    dot_size = 40*(1.0/np.sqrt(len(lambda_list)))

    # Plot the data
    scat = celes_plot.scatter(lambda_list, beta_list, vg_list, s=dot_size, vmin=11, vmax=71)
    
    # Plot the colorbar
    cb = fig.colorbar(scat)
    fg_color = 'white'
    cb.set_label("$V_g$ (km/s)", color=fg_color)
    cb.ax.yaxis.set_tick_params(color=fg_color)
    cb.outline.set_edgecolor(fg_color)
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=fg_color)

    plt.title("Sun-centered geocentric ecliptic coordinates", color=fg_color)

    plt.tight_layout()

    plt.savefig(os.path.join(dir_path, "scecliptic.png"), dpi=100, facecolor=fig.get_facecolor(), \
        edgecolor='none')

    plt.close()



    ### ###

    pass




if __name__ == "__main__":

    import argparse

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="""Given a folder with trajectory .pickle files, generate an orbit summary CSV file and orbital graphs.""",
        formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('dir_path', type=str, help='Path to the data directory. Trajectory pickle files are found in all subdirectories.')

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    ############################


    # Get a list of paths of all trajectory pickle files
    traj_list = []
    for entry in os.walk(cml_args.dir_path):

        dir_path, _, file_names = entry

        # Go through all files
        for file_name in file_names:

            # Check if the file is a pickel file
            if file_name.endswith("_trajectory.pickle"):

                # Load the pickle file
                traj = loadPickle(dir_path, file_name)

                traj_list.append(traj)




    # Generate the orbit summary file
    writeOrbitSummaryFile(cml_args.dir_path, traj_list)

    # Generate plots
    generateTrajectoryPlots(cml_args.dir_path, traj_list)




