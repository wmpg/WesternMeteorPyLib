""" Given the trajectory, beginning and ending height of sampling and the step, this code will sample the 
    trajectory and produce time, height, geo and ECI coordinates for every sample.

    This is used as a step in dark flight modelling.
"""

from __future__ import print_function, division, absolute_import


import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

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


    # Take the ground above the state vector as the reference distance from the surface of the Earth
    ref_radius = vectMag(traj.state_vect_mini) - traj.rbeg_ele

    # Compute distance from the centre of the Earth to each height
    radius_array = ref_radius + height_array


    print('Height (m),  Lat (deg),   Lon (deg), Elev (m)')
    # Go through every distance from the Earth centre and compute the geo coordinates at the given distance
    for ht, radius in zip(height_array, radius_array):

        # Compute the intersection between the trajectory line and the sphere of radius at the given height
        intersections = lineAndSphereIntersections(np.array([0, 0, 0]), radius, traj.state_vect_mini, 
            traj.radiant_eci_mini)


        # Choose the intersection that is closer to the state vector
        inter_min_dist_indx = np.argmin([vectMag(inter) for inter in intersections])
        height_eci = intersections[inter_min_dist_indx]


        # Compute geographical coordinates
        lat, lon, ele = cartesian2Geo(traj.jdt_ref, *height_eci)

        print("{:10.1f}, {:10.6f}, {:11.6f}, {:8.1f}".format(ht, np.degrees(lat), np.degrees(lon), ele))




if __name__ == "__main__":

    # Directory of the trajectory file
    dir_path = "/home/dvida/Dropbox/UWO Master's/Projects/MILIG files/20180117_010828 Michigan fireball (2 stations) second"

    # Trajectory pickle file
    file_name = "20180117_010828_trajectory.pickle"


    # Beginning height of sampling (km)
    #   Use -1 for the beginning hieght of the fireball
    beg_ht = 50.0

    # End height of sampling (km)
    #   Use -1 for the final height
    end_ht = 10.0

    # Sampling step (km)
    sample_step = 0.1


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

    