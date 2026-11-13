""" Given the trajectory, beginning and ending height (or time) of sampling and the step, this code will
    sample the trajectory and produce time, height, geo and ECI coordinates, velocity with its uncertainty,
    deceleration, dynamic pressure and ballistic coefficient for every sample.

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




# Names of the quantities stored for every sample
SAMPLE_ATTRIBUTES = ('t_est', 'ht', 'lat', 'lon', 'ele_geo', 'azim_norot', 'elev_norot', 'vel', 'vel_norot', \
    'length', 'eci', 'radiant_eci', 'vel_std', 'time_std', 'len_std', 'decel', 'dyn_pressure', 'k', \
    'log_k_std', 'extrapolated')



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

        # Position along the trajectory and the direction of the radiant
        self.length = []
        self.eci = []
        self.radiant_eci = []

        # Uncertainties of the motion from the drag fit
        self.vel_std = []
        self.time_std = []
        self.len_std = []

        # Dynamics of the drag model
        self.decel = []
        self.dyn_pressure = []
        self.k = []
        self.log_k_std = []

        # Flags of the samples outside the observed part
        self.extrapolated = []



    def addSample(self, t_est, ht, lat, lon, ele_geo, azim_norot, elev_norot, vel=None, vel_norot=None, \
        length=None, eci=None, radiant_eci=None, vel_std=None, time_std=None, len_std=None, decel=None, \
        dyn_pressure=None, k=None, log_k_std=None, extrapolated=None):
        """ Add a sample to the container.

        Arguments:
            t_est: [float] Time relative to the reference Julian date of the trajectory (s).
            ht: [float] Height above the mean sea level (m).
            lat: [float] Geodetic latitude (radians).
            lon: [float] Longitude (radians).
            ele_geo: [float] Height above the mean sea level computed from the ECI coordinates (m).
            azim_norot: [float] Ground-fixed azimuth of the radiant, +E of due N (radians).
            elev_norot: [float] Ground-fixed elevation of the radiant (radians).

        Keyword arguments:
            vel: [float] Velocity in the ECI frame (m/s). None by default.
            vel_norot: [float] Velocity relative to the ground (m/s). None by default.
            length: [float] Length along the trajectory from the state vector (m). None by default.
            eci: [ndarray] ECI coordinates, epoch of date (m). None by default.
            radiant_eci: [ndarray] Unit vector towards the radiant in the ECI frame, bent by gravity. None by
                default.
            vel_std: [float] Standard deviation of the velocity (m/s). None by default.
            time_std: [float] Standard deviation of the time (s), zero when sampling by time. None by default.
            len_std: [float] Standard deviation of the length along the trajectory (m), zero when sampling by
                height. None by default.
            decel: [float] Deceleration of the drag model (m/s^2), positive when the meteor slows down. None
                by default.
            dyn_pressure: [float] Dynamic pressure rho*v^2, with the velocity relative to the ground (Pa).
                None by default.
            k: [float] Ballistic coefficient of the drag model (m^2/kg). None by default.
            log_k_std: [float] Standard deviation of the natural logarithm of the ballistic coefficient. None
                by default.
            extrapolated: [bool] True if the sample is outside the observed part of the trajectory. None by
                default.
        """

        self.t_est.append(t_est)
        self.ht.append(ht)
        self.lat.append(lat)
        self.lon.append(lon)
        self.ele_geo.append(ele_geo)
        self.azim_norot.append(azim_norot)
        self.elev_norot.append(elev_norot)
        self.vel.append(vel)
        self.vel_norot.append(vel_norot)

        self.length.append(length)
        self.eci.append(eci)
        self.radiant_eci.append(radiant_eci)
        self.vel_std.append(vel_std)
        self.time_std.append(time_std)
        self.len_std.append(len_std)
        self.decel.append(decel)
        self.dyn_pressure.append(dyn_pressure)
        self.k.append(k)
        self.log_k_std.append(log_k_std)
        self.extrapolated.append(extrapolated)



    def toArrays(self):
        """ Convert the samples to numpy arrays.

        Return:
            [dict] Arrays of the sampled quantities, keyed by the names of the attributes. The ECI coordinates
                and the directions of the radiant have one row per sample, and the missing values are NaN.
        """

        arrays = {}
        for name in SAMPLE_ATTRIBUTES:

            values = getattr(self, name)

            # Keep the vectors as rows
            if name in ('eci', 'radiant_eci'):
                values = [np.full(3, np.nan) if val is None else val for val in values]
                arrays[name] = np.array(values, dtype=np.float64).reshape(-1, 3)

            elif name == 'extrapolated':
                arrays[name] = np.array(values, dtype=bool)

            else:
                arrays[name] = np.array(values, dtype=np.float64)


        return arrays



    def saveCSV(self, file_path):
        """ Save the samples to a CSV file, one row per sample.

        Arguments:
            file_path: [str] Path to the CSV file.
        """

        arrays = self.toArrays()

        # List the header, the values and the format of every column
        columns = [('Time (s)', arrays['t_est'], "{:.6f}"), \
            ('Time std (s)', arrays['time_std'], "{:.6f}"), \
            ('JD', self.traj.jdt_ref + arrays['t_est']/86400.0, "{:.10f}"), \
            ('Height (m)', arrays['ht'], "{:.2f}"), \
            ('Lat (deg)', np.degrees(arrays['lat']), "{:.8f}"), \
            ('Lon (deg)', np.degrees(arrays['lon']), "{:.8f}"), \
            ('Length (m)', arrays['length'], "{:.2f}"), \
            ('Length std (m)', arrays['len_std'], "{:.2f}"), \
            ('X ECI (m)', arrays['eci'][:, 0], "{:.2f}"), \
            ('Y ECI (m)', arrays['eci'][:, 1], "{:.2f}"), \
            ('Z ECI (m)', arrays['eci'][:, 2], "{:.2f}"), \
            ('Azim (deg)', np.degrees(arrays['azim_norot']), "{:.6f}"), \
            ('Elev (deg)', np.degrees(arrays['elev_norot']), "{:.6f}"), \
            ('Vel (m/s)', arrays['vel'], "{:.2f}"), \
            ('Vel std (m/s)', arrays['vel_std'], "{:.2f}"), \
            ('Vgnd (m/s)', arrays['vel_norot'], "{:.2f}"), \
            ('Decel (m/s^2)', arrays['decel'], "{:.4f}"), \
            ('Dyn pres (Pa)', arrays['dyn_pressure'], "{:.6e}"), \
            ('K (m^2/kg)', arrays['k'], "{:.6e}"), \
            ('ln K std', arrays['log_k_std'], "{:.4f}"), \
            ('Extrapolated', arrays['extrapolated'].astype(int), "{:d}")]

        with open(file_path, 'w') as f:

            # Write the header
            f.write("# Samples of the trajectory from the drag fit of wmpl.Utils.DragFit\n")
            f.write("# Reference Julian date: {:.10f}\n".format(self.traj.jdt_ref))
            f.write("# Azim and Elev are the radiant above a fixed ground, epoch of date\n")
            f.write("# Vel is in the ECI frame, Vgnd relative to the ground, Dyn pres is rho*Vgnd^2 " \
                + "(no wind)\n")
            f.write(", ".join([col[0] for col in columns]) + "\n")

            # Write one row per sample
            for i in range(len(arrays['t_est'])):
                f.write(", ".join([fmt.format(values[i]) for _, values, fmt in columns]) + "\n")



def sampleTrajectory(traj, beg_ht, end_ht, sample_step, show_plots=False, by_time=False, drag_above=False):
    """ Given the trajectory, beginning, end and step in meters, this function will sample the positions on
        the trajectory and compute the geographical coordinates, the ground-fixed azimuth and elevation of the
        radiant, the velocity with its uncertainty, the deceleration, the dynamic pressure and the ballistic
        coefficient for every sample. The times and velocities are computed from a drag model
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

        # Compute the uncertainties at the exact sampled times
        vel_std_array, len_std_array = drag_fit.motionStdAtTime(time_array)
        time_std_array = np.zeros_like(time_array)

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

        # Compute the uncertainties at the exact sampled heights
        vel_std_array, time_std_array = drag_fit.motionStdAtLength(len_array)
        len_std_array = np.zeros_like(len_array)


    vel_array = drag_fit.velocityAtLength(len_array)

    # Compute the velocity relative to the ground
    vel_norot_array = drag_fit.groundVelocityAtLength(len_array)

    # Compute the deceleration, the air density and the ballistic coefficient
    decel_array = drag_fit.decelerationAtLength(len_array)
    rho_array = drag_fit.airDensityAtLength(len_array)
    k_array, log_k_std_array = drag_fit.ballisticCoeffAtLength(len_array)

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
            + 'Vel (km/s), Vstd (m/s), Vgnd (km/s), Decel (m/s^2), Dyn pres (kPa)')


    # Open a trajectory sample container
    traj_samples = TrajectorySamples(traj, drag_fit)

    # Compute the geo coordinates, the ground-fixed azimuth and elevation and the velocity of every sample
    n_skipped = 0
    for i, (t_est, ht, len_val, vel) in enumerate(zip(time_array, height_array, len_array, vel_array)):

        # Skip the samples outside the modelled part of the trajectory
        if np.isnan(t_est) or np.isnan(len_val) or np.isnan(vel):
            n_skipped += 1
            continue

        # Compute geographical coordinates
        height_eci = drag_fit.eciAtLength(len_val)[0]
        jd = traj.jdt_ref + t_est/86400.0
        lat, lon, ele_geo = cartesian2Geo(jd, *height_eci)

        # Compute the ground-fixed alt/az of the radiant with the initial velocity, along the path bent by
        #   the gravity drop (ECI coordinates of date, no precession)
        radiant_eci = drag_fit.radiantAtLength(len_val)[0]
        azim_norot, elev_norot, _ = derotatedRadiantAltAz(traj.v_init*radiant_eci, height_eci, jd, lat, lon)

        # Compute the dynamic pressure with the velocity relative to the ground
        vel_norot = vel_norot_array[i]
        dyn_pressure = rho_array[i]*vel_norot**2

        # Mark the samples extrapolated outside the observed part
        extrapolated = (len_val < drag_fit.len_first) or (len_val > drag_fit.len_last)
        if extrapolated:
            time_marker = "*"
        else:
            time_marker = " "


        # Add point parameters
        traj_samples.addSample(t_est, ht, lat, lon, ele_geo, azim_norot, elev_norot, vel, vel_norot, \
            length=len_val, eci=height_eci, radiant_eci=radiant_eci, vel_std=vel_std_array[i], \
            time_std=time_std_array[i], len_std=len_std_array[i], decel=decel_array[i], \
            dyn_pressure=dyn_pressure, k=k_array[i], log_k_std=log_k_std_array[i], extrapolated=extrapolated)

        if show_plots:
            print(("{:s}{:7.3f}, {:13.1f}, {:10.6f}, {:11.6f}, {:10.1f}, {:10.6f}, {:10.6f}, {:10.3f}, " \
                + "{:10.1f}, {:11.3f}, {:13.2f}, {:14.3f}").format(time_marker, t_est, ht, np.degrees(lat), \
                np.degrees(lon), ele_geo, np.degrees(azim_norot), np.degrees(elev_norot), vel/1000, \
                vel_std_array[i], vel_norot/1000, decel_array[i], dyn_pressure/1000))


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
        print("Vstd is the 1 sigma uncertainty of Vel from the noise of the points, without the " \
            + "uncertainty of the trajectory solution or the bias of the drag model.")
        print("Decel is the deceleration of the drag model, Dyn pres the dynamic pressure rho*Vgnd^2, " \
            + "with the air at rest relative to the ground.")

        # Report the largest dynamic pressure among the samples
        if traj_samples.t_est:

            arrays = traj_samples.toArrays()
            i_peak = np.nanargmax(arrays['dyn_pressure'])
            print("Peak dynamic pressure of the samples: {:.3f} kPa at {:.3f} km, t = {:.3f} s".format( \
                arrays['dyn_pressure'][i_peak]/1000, arrays['ht'][i_peak]/1000, arrays['t_est'][i_peak]))

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

    arg_parser.add_argument('-c', '--csv', metavar='CSV_PATH', type=str, \
        help='Save the samples to the given CSV file.')



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
    traj_samples = sampleTrajectory(traj, beg_ht, end_ht, sample_step, show_plots=True, \
        by_time=cml_args.time, drag_above=cml_args.dragabove)

    # Save the samples
    if cml_args.csv is not None:
        traj_samples.saveCSV(cml_args.csv)
        print("Samples saved to: {:s}".format(cml_args.csv))


    # # Test the line and sphere intersection
    # centre = np.array([1.0, 0.0, 0.0])
    # radius = 1.0

    # origin = np.array([1.0, 1.0, 1.0])
    # direction = np.array([-1.0, -1.0, -1.0])


    # intersection = lineAndSphereIntersections(centre, radius, origin, direction)

    # print(intersection)

    # _plotSphereAndArrow(centre, radius, origin, direction, intersection)
