""" Meteor trajectory simulator.

Features:
    - Simulating individual trajectories and projecting the observations to the observers.

"""

from __future__ import print_function, division, absolute_import


import numpy as np
import scipy.optimize

from wmpl.Trajectory.Orbit import calcOrbit
from wmpl.Utils.TrajConversions import date2JD, raDec2ECI
from wmpl.Utils.Math import angleBetweenSphericalCoords, vectMag




def geocentricRadiantToApparent(ra_g, dec_g, v_g, state_vector, jd_ref):
    """ Numerically converts the given geocentric radiant to the apparent radiant. 

    Arguments:
        ra_g: [float] Geocentric right ascension (radians).
        dec_g: [float] Geocentric declination (radians).
        v_g: [float] Geocentric velocity (m/s).
        state_vector: [ndarray of 3 elemens] (x, y, z) ECI coordinates of the initial state vector (meters).
        jd_ref: [float] reference Julian date of the event.

    Return:
        (ra_a, dec_a, v_init): [list]
            - ra_a: [float] Apparent R.A. (radians).
            - dec_a: [float] Apparent declination (radians).
            - v_init: [float] Initial velocity (m/s).
    """

    def _radiantDiff(radiant_eq, ra_g, dec_g, v_init, state_vector, jd_ref):

        ra_a, dec_a = radiant_eq

        # Convert the given RA and Dec to ECI coordinates
        radiant_eci = np.array(raDec2ECI(ra_a, dec_a))

        # Estimate the orbit with the given apparent radiant
        orbit = calcOrbit(radiant_eci, v_init, v_init, state_vector, jd_ref, stations_fixed=False, \
            reference_init=True)

        if orbit.ra_g is None:
            return None

        # Compare the difference between the calculated and the reference geocentric radiant
        return angleBetweenSphericalCoords(orbit.dec_g, orbit.ra_g, dec_g, ra_g)


    # Assume that the velocity at infinity corresponds to the initial velocity
    v_init = np.sqrt(v_g**2 + (2*6.67408*5.9722)*1e13/vectMag(state_vector))

    # Numerically find the apparent radiant
    res = scipy.optimize.minimize(_radiantDiff, x0=[ra_g, dec_g], args=(ra_g, dec_g, v_init, state_vector, jd_ref), \
        bounds=[(0, 2*np.pi), (-np.pi, np.pi)], tol=1e-13, method='SLSQP')

    ra_a, dec_a = res.x

    # Calculate all orbital parameters with the best estimation of apparent RA and Dec
    orb = calcOrbit(np.array(raDec2ECI(ra_a, dec_a)), v_init, v_init, state_vector, jd_ref, stations_fixed=False, \
            reference_init=True)

    return ra_a, dec_a, v_init, orb



if __name__ == "__main__":


    # Least squares solution
    # ----------------------
    # State vector (ECI):
    #  X =   4663824.46 +/- 1982.77 m
    #  Y =    417867.46 +/- 3446.91 m
    #  Z =   4458026.80 +/- 12423.32 m
    #  Vx =    -5237.36 +/- 62.35 m/s
    #  Vy =     8828.73 +/- 323.60 m/s
    #  Vz =    33076.10 +/- 630.18 m/s

    # Timing offsets:
    #          1: -0.004239 s
    #          2: 0.000000 s

    # reference point on the trajectory:
    #   Time: 2012-12-13 00:18:05.413143 UTC
    #   Lon   = -81.494888 +/- 0.0441 deg
    #   Lat   =  43.782497 +/- 0.0905 deg

    # Radiant (apparent):
    #   R.A.   = 120.67717 +/- 0.3462 deg
    #   Dec    = +72.75807 +/- 0.3992 deg
    #   Vavg   =  33.29536 +/- 0.2666 km/s
    #   Vinit  =  34.63242 +/- 0.5897 km/s
    # Radiant (geocentric):
    #   R.A.   = 124.39147 +/- 0.3695 deg
    #   Dec    = +71.73020 +/- 0.4479 deg
    #   Vg     =  32.80401 +/- 0.6236 km/s
    #   Vinf   =  34.63242 +/- 0.5897 km/s
    #   Zg     =  57.16852 +/- 0.3859 deg


    ra_g = np.radians(124.39147)
    dec_g = np.radians(+71.73020)

    # Geocentric velocity in m/s
    v_g = 32.80401*1000

    # ECI coordinates of the inital state vector
    state_vector = np.array([4663824.46, 417867.46, 4458026.80])

    # reference Julian date
    jd_ref = date2JD(2012, 12, 13, 0, 18, 5)



    ra_a, dec_a, v_init, orb = geocentricRadiantToApparent(ra_g, dec_g, v_g, state_vector, jd_ref)

    print('Apparent:')
    print('  R.A.:', np.degrees(ra_a))
    print('  Dec: ', np.degrees(dec_a))

    # Print the whole orbit
    print(orb)



    # Traj solver:
    # [-0.15122695  0.25492659  0.9550617 ] 34632.4213376 33295.3578823 [ 4663824.46446627   417867.46380932  4458026.80410226] 2456274.51256
    # [-0.1512269   0.25492656  0.95506171] 34632.4203999 34632.4203999 [ 4663824.46   417867.46  4458026.8 ] 2456274.51256
    # Inverse


    # ### TEST ORBIT CALCULATION
    # orb1 = calcOrbit(np.array([-0.15122695, 0.25492659, 0.9550617 ]), 34632.4213376, 33295.3578823, np.array([ 4663824.46446627, 417867.46380932, 4458026.80410226]), 2456274.51256)
    # print("ORB 1", np.degrees(orb1.ra_g), np.degrees(orb1.dec_g))

    # orb2 = calcOrbit(np.array([-0.12606042, 0.25102545, 0.95973694]), 34632.4203999, 34632.4203999, np.array([ 4663824.46, 417867.46, 4458026.8 ]), 2456274.51256)
    # print("ORB 2", np.degrees(orb2.ra_g), np.degrees(orb2.dec_g))