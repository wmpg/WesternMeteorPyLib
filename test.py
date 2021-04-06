""" Working script for testing chunks of code. Nothing really of value here. """

import numpy as np

from wmpl.Utils.Math import findClosestPoints, vectNorm, vectMag
from wmpl.Utils.TrajConversions import date2JD, ecef2ENU, enu2ECEF, cartesian2Geo, geo2Cartesian




def calcSpatialResidual(jd, state_vect, radiant_eci, stat, meas):
    """ Calculate horizontal and vertical residuals from the radiant line, for the given observed point.

    Arguments:
        jd: [float] Julian date
        state_vect: [3 element ndarray] ECI position of the state vector
        radiant_eci: [3 element ndarray] radiant direction vector in ECI
        stat: [3 element ndarray] position of the station in ECI
        meas: [3 element ndarray] line of sight from the station, in ECI

    Return:
        (hres, vres): [tuple of floats] residuals in horitontal and vertical direction from the radiant line

    """

    meas = vectNorm(meas)

    # Calculate closest points of approach (observed line of sight to radiant line) from the state vector
    obs_cpa, rad_cpa, d = findClosestPoints(stat, meas, state_vect, radiant_eci)


    # Vector pointing from the point on the trajectory to the point on the line of sight
    p = obs_cpa - rad_cpa

    # Calculate geographical coordinates of the state vector
    lat, lon, elev = cartesian2Geo(jd, *state_vect)

    # Calculate ENU (East, North, Up) vector at the position of the state vector, and direction of the radiant
    nn = np.array(ecef2ENU(lat, lon, *radiant_eci))

    # Convert the vector to polar coordinates
    theta = np.arctan2(nn[1], nn[0])
    phi = np.arccos(nn[2]/vectMag(nn))

    # Local reference frame unit vectors
    hx = np.array([            -np.cos(theta),              np.sin(theta),         0.0])
    vz = np.array([-np.cos(phi)*np.sin(theta), -np.cos(phi)*np.cos(theta), np.sin(phi)])
    hy = np.array([ np.sin(phi)*np.sin(theta),  np.sin(phi)*np.cos(theta), np.cos(phi)])
    
    # Calculate local reference frame unit vectors in ECEF coordinates
    ehorzx = enu2ECEF(lat, lon, *hx)
    ehorzy = enu2ECEF(lat, lon, *hy)
    evert  = enu2ECEF(lat, lon, *vz)

    ehx = np.dot(p, ehorzx)
    ehy = np.dot(p, ehorzy)

    # Calculate vertical residuals
    vres = np.sign(ehx)*np.hypot(ehx, ehy)

    # Calculate horizontal residuals
    hres = np.dot(p, evert)

    return hres, vres



if __name__ == "__main__":


    import sys

    import matplotlib.pyplot as plt

    import pyximport
    pyximport.install(setup_args={'include_dirs':[np.get_include()]})
    from wmpl.MetSim.MetSimErosionCyTools import luminousEfficiency



    ### Plot different lum effs ###

    # Range of velocities
    vel_range = np.linspace(11000, 72000, 100)

    # Range of masses
    masses = [1e-11, 1e-9, 1e-7, 1e-5, 0.001, 0.01, 0.1, 1, 10, 100, 1000]


    lum_eff_types = [1, 2, 3, 4, 5]
    lum_eff_labels = ["Revelle & Ceplecha (2001) - Type I", 
                      "Revelle & Ceplecha (2001) - Type II",
                      "Revelle & Ceplecha (2001) - Type III",
                      "Borovicka et al. (2013) - Kosice",
                      "CAMO faint meteors"]

    for i, lum_type in enumerate(lum_eff_types):

        for mass in masses:

            lum_list = []
            for vel in vel_range:
                lum = luminousEfficiency(lum_type, 0.0, vel, mass)

                lum_list.append(lum)

            plt.plot(vel_range/1000, 100*np.array(lum_list), label="{:s} kg".format(str(mass)), zorder=4)


        plt.title(lum_eff_labels[i])

        plt.xlabel("Velocity (km/s)")
        plt.ylabel("Tau (%)")

        plt.legend()

        plt.grid(color='0.9')

        plt.show()



    sys.exit()

    ### ###




    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    jd = date2JD(2018, 2, 14, 10, 30, 0)

    lat = np.radians(45.0)
    lon = np.radians(13.0)
    h = 100000

    state_vect = np.array(geo2Cartesian(lat, lon, h, jd))
    radiant_eci = np.array([0.0, 1.0, 0.0])

    stat = np.array(geo2Cartesian(lat, lon, h + 10, jd))
    meas = np.array([0.0, 0.0, 1.0])



    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    print(calcSpatialResidual(jd, state_vect, radiant_eci, stat, meas))


    # Plot the origin
    #ax.scatter(0, 0, 0)


    # Plot the first point
    ax.scatter(*state_vect)

    # Plot the line from the origin
    rad_x, rad_y, rad_z = -vectNorm(state_vect)
    rst_x, rst_y, rst_z = state_vect
    meteor_len = 1000000
    ax.quiver(rst_x, rst_y, rst_z, rad_x, rad_y, rad_z, length=meteor_len, normalize=True, color='b', 
            arrow_length_ratio=0.1)

    # Plot the radiant direction line
    rad_x, rad_y, rad_z = -radiant_eci
    rst_x, rst_y, rst_z = state_vect
    meteor_len = 1000000
    ax.quiver(rst_x, rst_y, rst_z, rad_x, rad_y, rad_z, length=meteor_len, normalize=True, color='r', 
            arrow_length_ratio=0.1)


    # Plot the second point
    ax.scatter(*stat)

    # Plot the direction of the second vector
    rad_x, rad_y, rad_z = -meas
    rst_x, rst_y, rst_z = stat
    meteor_len = 1000000
    ax.quiver(rst_x, rst_y, rst_z, rad_x, rad_y, rad_z, length=meteor_len, normalize=True, color='g', 
            arrow_length_ratio=0.1)





    # Calculate closest points of approach (observed line of sight to radiant line) from the state vector
    obs_cpa, rad_cpa, d = findClosestPoints(stat, meas, state_vect, radiant_eci)

    # Plot the closest points
    ax.scatter(*obs_cpa)
    ax.scatter(*rad_cpa)


    # Set a constant aspect ratio
    ax.set_aspect('equal', adjustable='box-forced')

    plt.show()
