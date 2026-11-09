""" Given the trajectory, beginning and ending height (or time) of sampling and the step, this code will
    sample the trajectory and produce time, height, geo coordinates and velocity for every sample.

    This is used as a step in dark flight modelling.
"""

from __future__ import print_function, division, absolute_import

import os
import argparse

import numpy as np

from wmpl.Utils.DragFit import DragFit, plotDragFit
from wmpl.Utils.TrajConversions import cartesian2Geo, derotatedRadiantAltAz
from wmpl.Utils.Math import lineAndSphereIntersections
from wmpl.Utils.Pickling import loadPickle


def _plotSphereAndArrow(centre, radius, origin, direction, intersection_list):
    from itertools import product, combinations
    
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    from wmpl.Utils.Plotting import Arrow3D


    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect("equal")


    # draw sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = centre[0] + radius*np.cos(u)*np.sin(v)
    y = centre[1] + radius*np.sin(u)*np.sin(v)
    z = centre[2] + radius*np.cos(v)
    ax.plot_wireframe(x, y, z, color="b")

    # draw the origin
    ax.scatter(*origin, color="g", s=100)

    # draw a vector
    xa, ya, za = np.c_[origin, origin + direction]

    a = Arrow3D(xa, ya, za, mutation_scale=20,
                lw=1, arrowstyle="-|>", color="k")
    ax.add_artist(a)


    # if intersection:

    #     for point in intersection_list:
    #         # draw the intersections
    #         ax.scatter(*point, color="r", s=100)

    plt.show()




class TrajectorySamples(object):
    def __init__(self, traj, drag_fit=None):
        """ Container for the sampled positions on the trajectory.

        Arguments:
            traj: [Trajectory object] Sampled trajectory.

        Keyword arguments:
            drag_fit: [DragFit object] Drag fit of the motion along the trajectory used for sampling.
                None by default.
        """

        self.traj = traj
        self.drag_fit = drag_fit

        self.t_est = []
        self.ht = []
        self.lat = []
        self.lon = []
        self.ele_geo = []
        self.azim_norot = []
        self.elev_norot = []
        self.vel = []
        self.vel_norot = []



    def addSample(self, t_est, ht, lat, lon, ele_geo, azim_norot, elev_norot, vel=None, vel_norot=None):

        self.t_est.append(t_est)
        self.ht.append(ht)
        self.lat.append(lat)
        self.lon.append(lon)
        self.ele_geo.append(ele_geo)
        self.azim_norot.append(azim_norot)
        self.elev_norot.append(elev_norot)
        self.vel.append(vel)
        self.vel_norot.append(vel_norot)



def sampleTrajectory(traj, beg_ht, end_ht, sample_step, show_plots=False, by_time=False, drag_above=False):
    """ Given the trajectory, beginning, end and step in meters, this function will sample the positions on
        the trajectory and compute the geographical coordinates, the ground-fixed azimuth and elevation of the
        radiant and the velocity for every sample. The times and velocities are computed from a drag model
        fitted to the observations (see wmpl.Utils.DragFit), which also extrapolates them
        above the beginning and below the end of the observed part of the trajectory.

    Arguments:
        traj: [Trajectory object] Trajectory to sample.
        beg_ht: [float] Beginning height of sampling (m). -1 to use the observed beginning height.
        end_ht: [float] End height of sampling (m). -1 to use the observed end height.
        sample_step: [float] Sampling step (m).

    Keyword arguments:
        show_plots: [bool] Show the model of the motion and print the samples. False by default.
        by_time: [bool] If True, beg_ht, end_ht and sample_step are times (s, relative to traj.jdt_ref)
            instead of heights, and -1 does not mean the observed beginning and end. False by default.
        drag_above: [bool] If True, the drag above the beginning is computed with the ballistic coefficient
            at the beginning. If False, the meteor moves without drag above the beginning. False by default.

    Return:
        traj_samples: [TrajectorySamples object] Sampled positions.
    """

    # Fit the drag model of the motion along the trajectory
    drag_fit = DragFit(traj, drag_above=drag_above)

    if by_time:

        # Generate times for sampling and compute the positions along the trajectory
        time_array = np.arange(beg_ht, end_ht + sample_step/2, sample_step)
        len_array = drag_fit.lengthAtTime(time_array)
        height_array = drag_fit.heightAtLength(len_array)

    else:

        # Set begin and end heights, if not given
        if beg_ht < 0:
            beg_ht = traj.rbeg_ele

        if end_ht < 0:
            end_ht = traj.rend_ele

        # Generate heights for sampling and compute the positions along the trajectory
        height_array = np.flipud(np.arange(end_ht, beg_ht + sample_step, sample_step))
        len_array = drag_fit.lengthAtHeight(height_array)
        time_array = drag_fit.timeAtLength(len_array)


    vel_array = drag_fit.velocityAtLength(len_array)

    if show_plots:

        plotDragFit(drag_fit)

        print('Drag fit: residuals RMS = {:.1f} m, initial velocity = {:.3f} km/s'.format( \
            drag_fit.res_rms, drag_fit.params[1]/1000))
        print()
        print('Beginning coordinates (observed):')
        print('    Lat: {:.6f}'.format(np.degrees(traj.rbeg_lat)))
        print('    Lon: {:.6f}'.format(np.degrees(traj.rbeg_lon)))
        print('    Elev: {:.1f}'.format(traj.rbeg_ele))
        print()
        print("Ground-fixed azimuth and altitude:")
        print(' Time(s), Sample ht (m),  Lat (deg),   Lon (deg), Height (m), Azim (deg), Elev (deg), ' \
            + 'Vel (km/s), Vgnd (km/s)')


    # Open a trajectory sample container
    traj_samples = TrajectorySamples(traj, drag_fit)

    # Compute the geo coordinates, the ground-fixed azimuth and elevation and the velocity of every sample
    n_skipped = 0
    for t_est, ht, len_val, vel in zip(time_array, height_array, len_array, vel_array):

        # Skip the samples outside the modelled part of the trajectory
        if np.isnan(t_est) or np.isnan(len_val) or np.isnan(vel):
            n_skipped += 1
            continue

        # Compute geographical coordinates
        height_eci = drag_fit.eciAtLength(len_val)[0]
        jd = traj.jdt_ref + t_est/86400.0
        lat, lon, ele_geo = cartesian2Geo(jd, *height_eci)

        # Compute the ground-fixed alt/az of the radiant with the instantaneous velocity, along the path bent
        #   by the gravity drop. The ECI coordinates are already in the epoch of date, so no precession is
        #   applied
        azim_norot, elev_norot, vel_norot = derotatedRadiantAltAz(vel*drag_fit.radiantAtLength(len_val)[0], \
            height_eci, jd, lat, lon)

        # Mark the samples extrapolated outside the observed part
        if (len_val < drag_fit.len_first) or (len_val > drag_fit.len_last):
            time_marker = "*"
        else:
            time_marker = " "


        # Add point parameters
        traj_samples.addSample(t_est, ht, lat, lon, ele_geo, azim_norot, elev_norot, vel, vel_norot)

        if show_plots:
            print(("{:s}{:7.3f}, {:13.1f}, {:10.6f}, {:11.6f}, {:10.1f}, {:10.6f}, {:10.6f}, {:10.3f}, " \
                + "{:11.3f}").format(time_marker, t_est, ht, np.degrees(lat), np.degrees(lon), ele_geo, \
                np.degrees(azim_norot), np.degrees(elev_norot), vel/1000, vel_norot/1000))


    # Warn about the samples outside the modelled part of the trajectory, which goes from the top (or where
    #   the velocity above the beginning has grown by 10% with drag above) down to where gravity bends the
    #   path by 5 deg after the end
    if n_skipped:

        valid = drag_fit._valid
        ht_valid = drag_fit.ht_grid[valid]
        time_valid = drag_fit.time_grid[valid]

        if traj_samples.t_est:
            print('{:d} of {:d} samples were skipped, as they are outside the modelled part of the ' \
                .format(n_skipped, len(time_array)) + 'trajectory:')
        else:
            print('Nothing produced - the sampling range is outside the modelled part of the trajectory:')

        print('    heights {:.3f} to {:.3f} km, times {:.3f} to {:.3f} s'.format(ht_valid[0]/1000, \
            np.min(ht_valid)/1000, time_valid[0], time_valid[-1]))

    if show_plots:
        print('The star * denotes samples extrapolated outside the observed part of the trajectory with ' \
            + 'the drag model.')
        print("The horizontal coordinates are apparent above a fixed ground, not topocentric in J2000!")
        print('Vel is the velocity in the ECI frame, Vgnd the velocity relative to the ground.')
        print()
        print('End coordinates (observed):')
        print('    Lat: {:.6f}'.format(np.degrees(traj.rend_lat)))
        print('    Lon: {:.6f}'.format(np.degrees(traj.rend_lon)))
        print('    Elev: {:.1f}'.format(traj.rend_ele))


    return traj_samples




if __name__ == "__main__":



    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description=""" Sample the positions on the given trajectory. 
        The beginning and ending heights should be given, as well as the height step. The function
        returns a list of sampled points on the trajectory and their geographical coordinates. """,
        formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('traj_pickle_file', type=str, help='Path to the trajectory .pickle file.')

    arg_parser.add_argument('beg_height', type=float, help='Sampling begin height (km). -1 to use real begin height.')
    arg_parser.add_argument('end_height', type=float, help='Sampling end height (km). -1 to use the real end height')
    arg_parser.add_argument('height_step', type=float, help='Sampling step (km).')

    arg_parser.add_argument('-a', '--dragabove', action='store_true', \
        help='Compute the drag above the beginning with the ballistic coefficient at the beginning, ' \
        + 'instead of moving without drag.')

    arg_parser.add_argument('-t', '--time', action='store_true', help='Sample by time instead of height. The begin, end and step are then given in seconds relative to the reference time of the trajectory, and -1 does not mean the observed begin and end.')



    # Parse the command line arguments
    cml_args = arg_parser.parse_args()


    # Unpack the file name and the directory path from the given arguments
    dir_path, file_name = os.path.split(cml_args.traj_pickle_file)

    # Convert units to meters, or keep the times in seconds
    if cml_args.time:
        units_conv = 1.0
    else:
        units_conv = 1000.0

    beg_ht = units_conv*cml_args.beg_height
    end_ht = units_conv*cml_args.end_height
    sample_step = units_conv*cml_args.height_step

    ############################


    # # Directory of the trajectory file
    # dir_path = "/home/dvida/Dropbox/UWO Master's/Projects/MILIG files/20180117_010828 Michigan fireball (2 stations) second"

    # # Trajectory pickle file
    # file_name = "20180117_010828_trajectory.pickle"


    # # Beginning height of sampling (m)
    # #   Use -1 for the beginning height of the fireball
    # beg_ht = 50000.0

    # # End height of sampling (m)
    # #   Use -1 for the final height
    # end_ht = 10000.0

    # # Sampling step (m)
    # sample_step = 100


    # Load the trajectory file
    traj = loadPickle(dir_path, file_name)


    # Run trajectory sampling
    sampleTrajectory(traj, beg_ht, end_ht, sample_step, show_plots=True, by_time=cml_args.time, \
        drag_above=cml_args.dragabove)


    # # Test the line and sphere intersection
    # centre = np.array([1.0, 0.0, 0.0])
    # radius = 1.0

    # origin = np.array([1.0, 1.0, 1.0])
    # direction = np.array([-1.0, -1.0, -1.0])


    # intersection = lineAndSphereIntersections(centre, radius, origin, direction)

    # print(intersection)

    # _plotSphereAndArrow(centre, radius, origin, direction, intersection)
