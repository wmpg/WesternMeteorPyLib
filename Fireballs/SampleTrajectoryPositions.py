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

from Utils.TrajConversions import cartesian2Geo
from Utils.Math import lineAndSphereIntersections
from Utils.Pickling import loadPickle




def _plotSphereAndArrow(centre, radius, origin, direction, intersection_list):


    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    from itertools import product, combinations


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
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d


    class Arrow3D(FancyArrowPatch):

        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            FancyArrowPatch.draw(self, renderer)


    xa, ya, za = np.c_[origin, origin + direction]

    a = Arrow3D(xa, ya, za, mutation_scale=20,
                lw=1, arrowstyle="-|>", color="k")
    ax.add_artist(a)


    if intersection:

        for point in intersection_list:
            # draw the intersections
            ax.scatter(*point, color="r", s=100)

    plt.show()



from Utils.Math import vectMag, vectNorm




def sampleTrajectory(dir_path, file_name, beg_ht, end_ht, sample_step):
    """ Given the trajectory, beginning, end and step in km, this function will interpolate the 
        fireball height vs. distance and return the coordinates of sampled positions.
    
    Arguments:


    Return:
    """


    # Load the trajectory file
    traj = loadPickle(dir_path, file_name)


    # Convert heights to meters
    beg_ht *= 1000
    end_ht *= 1000
    sample_step *= 1000


    # Generate heights for sampling
    height_array = np.flipud(np.arange(end_ht, beg_ht + sample_step, sample_step))

    print(height_array)


    ### Fit time vs. height

    time_data = []
    height_data = []

    for obs in traj.observations:

        time_data += obs.time_data.tolist()
        height_data += obs.model_ht.tolist()


    height_data = np.array(height_data)
    time_data = np.array(time_data)

    # Sort the arrays by increasing height
    arr_sort_indices = np.argsort(height_data)
    height_data = height_data[arr_sort_indices]
    time_data = time_data[arr_sort_indices]



    # Plot the non-smoothed time vs. height
    plt.scatter(time_data, height_data/1000, label='Data')


    # Fit negative height as X axis must be rising
    ht_vs_time_interp = scipy.interpolate.PchipInterpolator(height_data, time_data)

    # Plot the interpolation
    ht_arr = np.linspace(np.min(height_data), np.max(height_data), 1000)
    time_arr = ht_vs_time_interp(ht_arr)

    plt.plot(time_arr, ht_arr/1000, label='Interpolation')


    plt.legend()
    


    plt.xlabel('Time (s)')
    plt.ylabel('Height (km)')

    plt.show()

    ###



    # Take the ground above the state vector as the reference distance from the surface of the Earth
    ref_radius = vectMag(traj.state_vect_mini) - traj.rbeg_ele

    # Compute distance from the centre of the Earth to each height
    radius_array = ref_radius + height_array


    print(' Time(s), Height (m),  Lat (deg),   Lon (deg), Elev (m)')
    # Go through every distance from the Earth centre and compute the geo coordinates at the given distance
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


        # Choose the intersection that is closer to the state vector
        inter_min_dist_indx = np.argmin([vectMag(inter) for inter in intersections])
        height_eci = intersections[inter_min_dist_indx]


        # Compute the Julian date at the given height
        jd = traj.jdt_ref + t_est/86400.0

        # Compute geographical coordinates
        lat, lon, ele = cartesian2Geo(jd, *height_eci)

        print("{:s}{:7.2f}, {:10.1f}, {:10.6f}, {:11.6f}, {:8.1f}".format(time_marker, t_est, ht, np.degrees(lat), np.degrees(lon), ele))


    print('The star * denotes heights extrapolated after the end of the fireball, with the fixed velocity of 3 km/s.')




if __name__ == "__main__":



    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description=""" Sample the positions on the given trajectory. 
        The beginning and ending heights should be given, as well as the height step. The function
        returns a list of sampled points on the trajectory and their geographical coordinates. """,
        formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('traj_pickle_file', type=str, help='Path to the trajectory .pickle file.')

    arg_parser.add_argument('beg_height', type=float, help='Sampling begin height (km).')
    arg_parser.add_argument('end_height', type=float, help='Sampling end height (km).')
    arg_parser.add_argument('height_step', type=float, help='Sampling step (km).')



    # Parse the command line arguments
    cml_args = arg_parser.parse_args()


    # Unpack the file name and the directory path from the given arguments
    dir_path, file_name = os.path.split(cml_args.traj_pickle_file)

    beg_ht = cml_args.beg_height
    end_ht = cml_args.end_height
    sample_step = cml_args.height_step

    ############################


    # # Directory of the trajectory file
    # dir_path = "/home/dvida/Dropbox/UWO Master's/Projects/MILIG files/20180117_010828 Michigan fireball (2 stations) second"

    # # Trajectory pickle file
    # file_name = "20180117_010828_trajectory.pickle"


    # # Beginning height of sampling (km)
    # #   Use -1 for the beginning hieght of the fireball
    # beg_ht = 50.0

    # # End height of sampling (km)
    # #   Use -1 for the final height
    # end_ht = 10.0

    # # Sampling step (km)
    # sample_step = 0.1



    samples = sampleTrajectory(dir_path, file_name, beg_ht, end_ht, sample_step)

    print(samples)




    # # Test the line and sphere intersection
    # centre = np.array([1.0, 0.0, 0.0])
    # radius = 1.0

    # origin = np.array([1.0, 1.0, 1.0])
    # direction = np.array([-1.0, -1.0, -1.0])


    # intersection = lineAndSphereIntersections(centre, radius, origin, direction)

    # print(intersection)

    # _plotSphereAndArrow(centre, radius, origin, direction, intersection)

    