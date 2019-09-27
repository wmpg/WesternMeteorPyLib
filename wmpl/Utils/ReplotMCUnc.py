
import os

import numpy as np
import matplotlib.pyplot as plt

from wmpl.Utils.Pickling import loadPickle
from wmpl.Utils.Plotting import savePlot


if __name__ == "__main__":


    import argparse

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description=""" Replot the trajectory uncertainties.""",
        formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('mc_uncertainties_path', type=str, help='Path to the MC uncertainties file.')

    arg_parser.add_argument('-n', '--nbins', metavar="NUM_BINS", nargs=1, \
        help='Number of bins for the histogram.', type=int)

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    ############################


    dir_path_mc, mc_unc_file = os.path.split(cml_args.mc_uncertainties_path)


    ### Load trajectory pickles

    # Load uncertainties
    traj_unc = loadPickle(dir_path_mc, mc_unc_file)


    # Extract file name core
    traj_file_name_core = mc_unc_file.replace('_mc_uncertainties.pickle', '')

    # Load geometrical trajectory
    dir_path_parent = os.path.abspath(os.path.join(dir_path_mc, os.pardir))
    traj = loadPickle(dir_path_parent, traj_file_name_core + '_trajectory.pickle')

    # Load MC trajectory
    traj_best = loadPickle(dir_path_mc, traj_file_name_core + '_mc_trajectory.pickle')


    ###




    mc_results = traj_unc.mc_traj_list


    a_list = np.array([traj_temp.orbit.a for traj_temp in mc_results])
    incl_list = np.array([traj_temp.orbit.i for traj_temp in mc_results])
    e_list = np.array([traj_temp.orbit.e for traj_temp in mc_results])
    peri_list = np.array([traj_temp.orbit.peri for traj_temp in mc_results])
    q_list = np.array([traj_temp.orbit.q for traj_temp in mc_results])

    fig = plt.figure()

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2, sharey=ax1)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4, sharey=ax3)

    # Compute the number of bins
    if cml_args.nbins is not None:
        nbins = cml_args.nbins[0]

    else:
        nbins = np.ceil(np.sqrt(len(a_list)))
        if nbins < 10:
            nbins = 10

    # Semimajor axis vs. inclination
    ax1.hist2d(a_list, np.degrees(incl_list), bins=nbins)
    ax1.set_xlabel('a (AU)')
    ax1.set_ylabel('Inclination (deg)')
    plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')
    #ax1.get_xaxis().get_major_formatter().set_useOffset(False)
    ax1.ticklabel_format(useOffset=False)

    # Plot the first solution and the MC solution
    if traj is not None:
        if traj.orbit.a is not None:
            ax1.scatter(traj.orbit.a, np.degrees(traj.orbit.i), c='r', linewidth=1, edgecolors='w')

    if traj_best is not None:
        if traj_best.orbit.a is not None:
            ax1.scatter(traj_best.orbit.a, np.degrees(traj_best.orbit.i), c='g', linewidth=1, edgecolors='w')



    # Plot argument of perihelion vs. inclination
    ax2.hist2d(np.degrees(peri_list), np.degrees(incl_list), bins=nbins)
    ax2.set_xlabel('peri (deg)')
    plt.setp(ax2.get_xticklabels(), rotation=30, horizontalalignment='right')
    #ax2.get_xaxis().get_major_formatter().set_useOffset(False)
    ax2.ticklabel_format(useOffset=False)

    # Plot the first solution and the MC solution
    if traj is not None:
        if traj.orbit.peri is not None:
            ax2.scatter(np.degrees(traj.orbit.peri), np.degrees(traj.orbit.i), c='r', linewidth=1, \
                edgecolors='w')

    if traj_best is not None:
        if traj_best.orbit.peri is not None:
            ax2.scatter(np.degrees(traj_best.orbit.peri), np.degrees(traj_best.orbit.i), c='g', linewidth=1, \
                edgecolors='w')

    ax2.tick_params(
        axis='y',          # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',        # ticks along the left edge are off
        labelleft='off')   # labels along the left edge are off


    # Plot eccentricity vs. perihelion distance
    ax3.hist2d(e_list, q_list, bins=nbins)
    ax3.set_xlabel('Eccentricity')
    ax3.set_ylabel('q (AU)')
    plt.setp(ax3.get_xticklabels(), rotation=30, horizontalalignment='right')
    #ax3.get_xaxis().get_major_formatter().set_useOffset(False)
    ax3.ticklabel_format(useOffset=False)

    # Plot the first solution and the MC solution
    if traj is not None:
        if traj.orbit.e is not None:
            ax3.scatter(traj.orbit.e, traj.orbit.q, c='r', linewidth=1, edgecolors='w')

    if traj_best is not None:
        if traj_best.orbit.e is not None:
            ax3.scatter(traj_best.orbit.e, traj_best.orbit.q, c='g', linewidth=1, edgecolors='w')

    # Plot argument of perihelion vs. perihelion distance
    ax4.hist2d(np.degrees(peri_list), q_list, bins=nbins)
    ax4.set_xlabel('peri (deg)')
    plt.setp(ax4.get_xticklabels(), rotation=30, horizontalalignment='right')
    #ax4.get_xaxis().get_major_formatter().set_useOffset(False)
    ax4.ticklabel_format(useOffset=False)

    # Plot the first solution and the MC solution
    if traj is not None:
        if traj.orbit.peri is not None:
            ax4.scatter(np.degrees(traj.orbit.peri), traj.orbit.q, c='r', linewidth=1, edgecolors='w')

    if traj_best is not None:
        if traj_best.orbit.peri is not None:
            ax4.scatter(np.degrees(traj_best.orbit.peri), traj_best.orbit.q, c='g', linewidth=1, \
                edgecolors='w')
        

    ax4.tick_params(
        axis='y',          # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',        # ticks along the left edge are off
        labelleft='off')   # labels along the left edge are off
    

    plt.tight_layout()
    plt.subplots_adjust(wspace=0)

    savePlot(plt, traj_file_name_core + '_monte_carlo_orbit_elems.png', output_dir=dir_path_mc)

    plt.show()