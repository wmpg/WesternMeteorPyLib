""" Aggregate trajectory results into one results file and generate plots of trajectory results. """

from __future__ import print_function, division, absolute_import


import os
import datetime

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import shiftgrid
import scipy.stats


from wmpl.Utils.Pickling import loadPickle
from wmpl.Utils.PlotCelestial import CelestialPlot
from wmpl.Utils.SolarLongitude import jd2SolLonSteyaert


def writeOrbitSummaryFile(dir_path, traj_list):
    """ Given a list of trajectory files, generate CSV file with the orbit summary. """

    pass





def plotSCE(x_data, y_data, color_data, sol_range, plot_title, colorbar_title, dir_path, file_name, \
    density_plot=False):

    ### PLOT SUN-CENTERED GEOCENTRIC ECLIPTIC RADIANTS ###

    fig = plt.figure(figsize=(16, 8), facecolor='k')


    # Init the allsky plot
    celes_plot = CelestialPlot(x_data, y_data, projection='sinu', lon_0=270, ax=fig.gca())

    # ### Mark the sources ###
    # sources_lg = np.radians([0, 270, 180])
    # sources_bg = np.radians([0, 0, 0])
    # sources_labels = ["Helion", "Apex", "Antihelion"]
    # celes_plot.scatter(sources_lg, sources_bg, marker='x', s=15, c='0.75')

    # # Convert angular coordinates to image coordinates
    # x_list, y_list = celes_plot.m(np.degrees(sources_lg), np.degrees(sources_bg + np.radians(2.0)))
    # for x, y, lbl in zip(x_list, y_list, sources_labels):
    #     plt.text(x, y, lbl, color='w', ha='center', alpha=0.5)

    # ### ###

    fg_color = 'white'


    ### Do a KDE density plot
    if density_plot:

        # Init the KDE
        data = np.vstack([np.degrees(np.array(x_data)), np.degrees(np.array(y_data))])
        kde = scipy.stats.gaussian_kde(data, bw_method=0.01)

        # Get lat/lons of ny by nx evenly space grid.
        #lons, lats = celes_plot.m.makegrid(n_lon, n_lat) 
        delta = 0.2
        lons = np.arange(0, 360, delta)
        lats = np.arange(-90, 90 + delta, delta)


        # Compute map proj coordinates.
        LONS, LATS = np.meshgrid(lons, lats)

        # Get KDE values
        kde_values = kde(np.vstack([LONS.ravel(), LATS.ravel()]))

        # Shift values so they can be plotted
        kde_values, LONS_ravelled = shiftgrid(90., kde_values, LONS.ravel(), start=True)
        LONS = np.reshape(LONS_ravelled, LONS.shape)
        x, y = celes_plot.m(LONS, LATS)
        kde_values = np.reshape(kde_values.T, x.shape)

        # Scale KDE values to counts
        kde_values *= len(x_data)

        # Plot the countures
        plt_handle = celes_plot.m.contourf(x, y, kde_values, levels=30, cmap='inferno')
        
        # # add colorbar.
        # cbar = celes_plot.m.colorbar(cs, location='bottom', pad="5%")
        # cbar.set_label('Count')


    else:
        
        ### Do a scatter plot

        # Compute the dot size which varies by the number of data points
        dot_size = 40*(1.0/np.sqrt(len(x_data)))

        # Plot the data
        plt_handle = celes_plot.scatter(x_data, y_data, color_data, s=dot_size)
        
    # Plot the colorbar
    cb = fig.colorbar(plt_handle)
    cb.set_label(colorbar_title, color=fg_color)
    cb.ax.yaxis.set_tick_params(color=fg_color)
    cb.outline.set_edgecolor(fg_color)
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=fg_color)


    plt.title(plot_title, color=fg_color)

    # Plot solar longitude range and count
    sol_min, sol_max = sol_range
    # plt.annotate(u"$\lambda_{\u2609 min} =$" + u"{:>5.2f}\u00b0".format(sol_min) \
    #     + u"\n$\lambda_{\u2609 max} =$" + u"{:>5.2f}\u00b0".format(sol_max), \
    #     xy=(0, 1), xycoords='axes fraction', color='w', size=12, family='monospace')
    plt.annotate(u"Sol min = {:>6.2f}\u00b0".format(sol_min) \
        + u"\nSol max = {:>6.2f}\u00b0".format(sol_max)
        + "\nCount = {:d}".format(len(x_data)), \
        xy=(0, 1), xycoords='axes fraction', color='w', size=10, family='monospace')

    plt.tight_layout()

    plt.savefig(os.path.join(dir_path, file_name), dpi=100, facecolor=fig.get_facecolor(), \
        edgecolor='none')

    plt.close()

    ### ###



def generateTrajectoryPlots(dir_path, traj_list):
    """ Given a path with trajectory .pickle files, generate orbits plots. """



    ### Plot Sun-centered geocentric ecliptic plots ###
    lambda_list = []
    beta_list = []
    vg_list = []
    sol_list = []

    hypo_count = 0
    jd_min = np.inf
    jd_max = 0
    for traj in traj_list:

        # Reject all hyperbolic orbits
        if traj.orbit.e > 1:
            hypo_count += 1
            continue

        # Compute Sun-centered longitude
        lambda_list.append(traj.orbit.L_g - traj.orbit.la_sun)

        beta_list.append(traj.orbit.B_g)
        vg_list.append(traj.orbit.v_g/1000)
        sol_list.append(np.degrees(traj.orbit.la_sun))

        # Track first and last observation
        jd_min = min(jd_min, traj.jdt_ref)
        jd_max = max(jd_max, traj.jdt_ref)


    print("Hyperbolic percentage: {:.2f}%".format(100*hypo_count/len(traj_list)))

    # Compute the range of solar longitudes
    sol_min = np.degrees(jd2SolLonSteyaert(jd_min))
    sol_max = np.degrees(jd2SolLonSteyaert(jd_max))



    # Plot SCE vs Vg
    plotSCE(lambda_list, beta_list, vg_list, (sol_min, sol_max), 
        "Sun-centered geocentric ecliptic coordinates", "$V_g$ (km/s)", dir_path, "scecliptic_vg.png")


    # Plot SCE vs Sol
    plotSCE(lambda_list, beta_list, sol_list, (sol_min, sol_max), \
        "Sun-centered geocentric ecliptic coordinates", "Solar longitude (deg)", dir_path, \
        "scecliptic_sol.png")
    

    
    # Plot SCE orbit density
    plotSCE(lambda_list, beta_list, None, (sol_min, sol_max), 
        "Sun-centered geocentric ecliptic coordinates", "Count", dir_path, "scecliptic_density.png", \
        density_plot=True)




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



    ### FILTERS ###

    # Minimum number of points on the trajectory for the station with the most points
    min_traj_points = 6


    ### ###


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


                ### Reject all trajectories with small number of used points ###
                points_count = [len(obs.time_data[obs.ignore_list == 0]) for obs in traj.observations \
                    if obs.ignore_station == False]

                if not points_count:
                    continue

                max_points = max(points_count)

                if max_points < min_traj_points:
                    print("Skipping {:.2} due to the small number of points...".format(traj.jdt_ref))
                    continue

                ###



                traj_list.append(traj)




    # Generate the orbit summary file
    writeOrbitSummaryFile(cml_args.dir_path, traj_list)

    # Generate plots
    generateTrajectoryPlots(cml_args.dir_path, traj_list)




