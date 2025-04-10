""" Given the trajectory, beginning and ending height of sampling and the step, this code will sample the 
    trajectory and produce time, height, geo and ECI coordinates for every sample.

    This is used as a step in dark flight modelling.
"""

from __future__ import print_function, division, absolute_import

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.signal

from wmpl.Utils.TrajConversions import cartesian2Geo, eci2RaDec, raDec2AltAz, altAz2RADec, \
    equatorialCoordPrecession, J2000_JD
from wmpl.Utils.Math import lineAndSphereIntersections, vectMag, vectNorm
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
    def __init__(self, traj):

        self.traj = traj

        self.t_est = []
        self.ht = []
        self.lat = []
        self.lon = []
        self.ele_geo = []
        self.azim_norot = []
        self.elev_norot = []

    def addSample(self, t_est, ht, lat, lon, ele_geo, azim_norot, elev_norot):

        self.t_est.append(t_est)
        self.ht.append(ht)
        self.lat.append(lat)
        self.lon.append(lon)
        self.ele_geo.append(ele_geo)
        self.azim_norot.append(azim_norot)
        self.elev_norot.append(elev_norot)


def sampleTrajectory(traj, beg_ht, end_ht, sample_step, show_plots=False):
    """ Given the trajectory, beginning, end and step in km, this function will interpolate the 
        fireball height vs. distance and return the coordinates of sampled positions and compute the azimuth
        and elevation for every point.
    
    Arguments:


    Return:
    """


    # Set begin and end heights, if not given
    if beg_ht < 0:
        beg_ht = traj.rbeg_ele

    if end_ht < 0:
        end_ht = traj.rend_ele


    # Generate heights for sampling
    height_array = np.flipud(np.arange(end_ht, beg_ht + sample_step, sample_step))

    ### Fit time vs. height

    time_data = []
    height_data = []

    for obs in traj.observations:

        time_data += obs.time_data.tolist()
        height_data += obs.model_ht.tolist()

        if show_plots:

            # Plot the station data
            plt.scatter(obs.time_data, obs.model_ht/1000, label=obs.station_id, marker='x', zorder=3)


    height_data = np.array(height_data)
    time_data = np.array(time_data)

    # Sort the arrays by decreasing time
    arr_sort_indices = np.argsort(time_data)[::-1]
    height_data = height_data[arr_sort_indices]
    time_data = time_data[arr_sort_indices]


    # Plot the non-smoothed time vs. height
    #plt.scatter(time_data, height_data/1000, label='Data')


    # Apply Savitzky-Golay to smooth out the height change
    height_data = scipy.signal.savgol_filter(height_data, 21, 5)

    if show_plots:
        plt.scatter(time_data, height_data/1000, label='Savitzky-Golay filtered', marker='+', zorder=3)


    # Sort the arrays by increasing heights (needed for interpolation)
    arr_sort_indices = np.argsort(height_data)
    height_data = height_data[arr_sort_indices]
    time_data = time_data[arr_sort_indices]


    # Interpolate height vs. time
    ht_vs_time_interp = scipy.interpolate.PchipInterpolator(height_data, time_data)


    # Plot the interpolation
    ht_arr = np.linspace(np.min(height_data), np.max(height_data), 1000)
    time_arr = ht_vs_time_interp(ht_arr)

    if show_plots:
        plt.plot(time_arr, ht_arr/1000, label='Interpolation', zorder=3)


        plt.legend()


        plt.xlabel('Time (s)')
        plt.ylabel('Height (km)')

        plt.grid()

        plt.show()

    ###



    # Take the ground above the state vector as the reference distance from the surface of the Earth
    ref_radius = vectMag(traj.state_vect_mini) - np.max(height_data)

    # Compute distance from the centre of the Earth to each height
    radius_array = ref_radius + height_array

    if show_plots:

        print('Beginning coordinates (observed):')
        print('    Lat: {:.6f}'.format(np.degrees(traj.rbeg_lat)))
        print('    Lon: {:.6f}'.format(np.degrees(traj.rbeg_lon)))
        print('    Elev: {:.1f}'.format(traj.rbeg_ele))
        print()
        print("Ground-fixed azimuth and altitude:")
        print(' Time(s), Sample ht (m),  Lat (deg),   Lon (deg), Height (m), Azim (deg), Elev (deg)')


    # Open a trajectory sample container
    traj_samples = TrajectorySamples(traj)

    # Go through every distance from the Earth centre and compute the geo coordinates at the given distance,
    #   as well as the point-to-point azimuth and elevation
    prev_eci = None
    good_data = False
    for ht, radius in zip(height_array, radius_array):

        # If the height is lower than the eng height, use a fixed velocity of 3 km/s

        if ht < traj.rend_ele:
            t_est = ht_vs_time_interp(traj.rend_ele) + abs(ht - traj.rend_ele)/3000
            time_marker = "*"

        else:
            
            # Estimate the fireball time at the given height using interpolated values
            t_est = ht_vs_time_interp(ht)
            time_marker = " "

        # Compute the intersection between the trajectory line and the sphere of radius at the given height
        intersections = lineAndSphereIntersections(np.array([0, 0, 0]), radius, traj.state_vect_mini, 
            traj.radiant_eci_mini)
        
        # if there are no intersections it means that the height was outside the observed start/end height of the meteor
        if len(intersections) == 0:
            continue
        # if we get here we have at least one valid row of data 
        good_data = True

        # Choose the intersection that is closer to the state vector
        inter_min_dist_indx = np.argmin([vectMag(inter - traj.state_vect_mini) for inter in intersections])
        height_eci = intersections[inter_min_dist_indx]


        # Compute the Julian date at the given height
        jd = traj.jdt_ref + t_est/86400.0

        # Compute geographical coordinates
        lat, lon, ele_geo = cartesian2Geo(jd, *height_eci)

        # Compute azimuth and elevation
        if prev_eci is not None:

            # Compute the vector pointing from the previous point to the current point
            direction_vect = vectNorm(prev_eci - height_eci)


            ### Compute the ground-fixed alt/az

            eci_x, eci_y, eci_z = height_eci

            # Calculate the geocentric latitude (latitude which considers the Earth as an elipsoid) of the reference 
            # trajectory point
            lat_geocentric = np.arctan2(eci_z, np.sqrt(eci_x**2 + eci_y**2))


            # Calculate the velocity of the Earth rotation at the position of the reference trajectory point (m/s)
            v_e = 2*np.pi*vectMag(height_eci)*np.cos(lat_geocentric)/86164.09053

            
            # Calculate the equatorial coordinates of east from the reference position on the trajectory
            azimuth_east = np.pi/2
            altitude_east = 0
            ra_east, dec_east = altAz2RADec(azimuth_east, altitude_east, jd, lat, lon)


            # The reference velocity vector has the average velocity and the given direction
            # Note that ideally this would be the instantaneous velocity
            v_ref_vect = traj.orbit.v_avg_norot*direction_vect

            v_ref_nocorr = np.zeros(3)

            # Calculate the derotated reference velocity vector/radiant
            v_ref_nocorr[0] = v_ref_vect[0] + v_e*np.cos(ra_east)
            v_ref_nocorr[1] = v_ref_vect[1] + v_e*np.sin(ra_east)
            v_ref_nocorr[2] = v_ref_vect[2]

            # Compute the radiant without Earth's rotation included
            ra_norot, dec_norot = eci2RaDec(vectNorm(v_ref_nocorr))

            # Precess to the epoch of date
            ra_norot, dec_norot = equatorialCoordPrecession(J2000_JD.days, jd, ra_norot, dec_norot)

            # Compute apparent alt/az
            azim_norot, elev_norot = raDec2AltAz(ra_norot, dec_norot, jd, lat, lon)


            ### 


        else:
            azim_norot = -np.inf
            elev_norot = -np.inf

        prev_eci = np.copy(height_eci)


        # Add point parameters
        traj_samples.addSample(t_est, ht, lat, lon, ele_geo, azim_norot, elev_norot)

        if show_plots:

            print("{:s}{:7.3f}, {:13.1f}, {:10.6f}, {:11.6f}, {:10.1f}, {:10.6f}, {:10.6f}".format(time_marker, t_est, ht, np.degrees(lat), np.degrees(lon), ele_geo, np.degrees(azim_norot), np.degrees(elev_norot)))


    if not good_data: 
        # if the height is outside the observed range
        print('nothing produced - probably the height range you chose was outside the observed meteor heights printed above and below')
        
    if show_plots:
        print('The star * denotes heights extrapolated after the end of the fireball, with the fixed velocity of 3 km/s.')
        print("The horizontal coordinates are apparent above a fixed ground, not topocentric in J2000!")


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



    # Parse the command line arguments
    cml_args = arg_parser.parse_args()


    # Unpack the file name and the directory path from the given arguments
    dir_path, file_name = os.path.split(cml_args.traj_pickle_file)

    # Convert units to meters
    beg_ht = 1000*cml_args.beg_height
    end_ht = 1000*cml_args.end_height
    sample_step = 1000*cml_args.height_step

    ############################


    # # Directory of the trajectory file
    # dir_path = "/home/dvida/Dropbox/UWO Master's/Projects/MILIG files/20180117_010828 Michigan fireball (2 stations) second"

    # # Trajectory pickle file
    # file_name = "20180117_010828_trajectory.pickle"


    # # Beginning height of sampling (m)
    # #   Use -1 for the beginning hieght of the fireball
    # beg_ht = 50000.0

    # # End height of sampling (m)
    # #   Use -1 for the final height
    # end_ht = 10000.0

    # # Sampling step (m)
    # sample_step = 100


    # Load the trajectory file
    traj = loadPickle(dir_path, file_name)


    # Run trajectory sampling
    sampleTrajectory(traj, beg_ht, end_ht, sample_step, show_plots=True)


    # # Test the line and sphere intersection
    # centre = np.array([1.0, 0.0, 0.0])
    # radius = 1.0

    # origin = np.array([1.0, 1.0, 1.0])
    # direction = np.array([-1.0, -1.0, -1.0])


    # intersection = lineAndSphereIntersections(centre, radius, origin, direction)

    # print(intersection)

    # _plotSphereAndArrow(centre, radius, origin, direction, intersection)
