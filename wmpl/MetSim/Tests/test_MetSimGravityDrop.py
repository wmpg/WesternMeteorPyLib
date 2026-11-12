""" Tests of the gravity in MetSim against an independent integration of the equations of motion.

MetSim takes the height of every fragment from a straight line in space, from which gravity across the path
drops it, and adds gravity along the path to its speed. The drag slows both the meteoroid and the speed of the
drop. The tests integrate the motion of a single body in the plane of the path, with the central gravity of
the Earth, the drag and the mass loss of MetSim, and compare the heights and the speeds.

Run with pytest:
    python -m pytest wmpl/MetSim/Tests/test_MetSimGravityDrop.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np
import scipy.integrate

from wmpl.MetSim.MetSimErosion import Constants, runSimulation, atmDensityPoly, G0
from wmpl.Utils.AtmosphereDensity import fitAtmPoly
from wmpl.Utils.TrajConversions import date2JD



def bodyConstants(m_init, v_init, zenith_angle, h_kill):
    """ Make the constants of a single body without erosion and fragmentation.

    Arguments:
        m_init: [float] Initial mass (kg).
        v_init: [float] Initial speed (m/s).
        zenith_angle: [float] Zenith angle of the path at the beginning (deg).
        h_kill: [float] Height at which the simulation stops (m).

    Return:
        const: [Constants object] Constants of the body.
    """

    const = Constants()

    # Fit the atmosphere density over the whole flight
    const.h_init = 100000.0
    const.dens_co = fitAtmPoly(np.radians(45.3), np.radians(18.1), 10000, const.h_init, \
        date2JD(2020, 4, 20, 16, 15, 0))

    const.m_init = m_init
    const.v_init = v_init
    const.zenith_angle = np.radians(zenith_angle)
    const.sigma = 0.005/1e6
    const.rho = 3500.0
    const.rho_grain = 3500.0
    const.erosion_on = False
    const.disruption_on = False
    const.fragmentation_on = False
    const.h_kill = h_kill
    const.v_kill = 2500.0

    return const



def integrateMotion(const):
    """ Integrate the motion of a single body in the plane of its path, in an inertial frame centred on the
        Earth, with the central gravity, the drag and the mass loss of MetSim.

    Arguments:
        const: [Constants object] Constants of the body.

    Return:
        [OdeSolution object] Dense solution of the state (x, y, vx, vy, m, length) against time.
    """

    r_earth = const.r_earth
    gm = G0*r_earth**2
    k_shape = const.gamma*const.shape_factor*const.rho**(-2/3.0)

    def derivatives(t, state):

        x, y, vx, vy, m, _ = state

        r = np.hypot(x, y)
        v = np.hypot(vx, vy)
        rho_atm = atmDensityPoly(r - r_earth, const.dens_co)

        # Drag deceleration rate and mass loss of MetSim
        drag_rate = k_shape*m**(-1/3.0)*rho_atm*v
        dm_dt = -k_shape*const.sigma*m**(2/3.0)*rho_atm*v**3

        ax = -drag_rate*vx - gm*x/r**3
        ay = -drag_rate*vy - gm*y/r**3

        return [vx, vy, ax, ay, dm_dt, v]

    def belowKillHeight(t, state):
        return np.hypot(state[0], state[1]) - r_earth - const.h_kill

    belowKillHeight.terminal = True

    state0 = [0.0, r_earth + const.h_init, const.v_init*np.sin(const.zenith_angle), \
        -const.v_init*np.cos(const.zenith_angle), const.m_init, 0.0]

    solution = scipy.integrate.solve_ivp(derivatives, (0, 100), state0, method='DOP853', rtol=1e-11, \
        atol=1e-6, dense_output=True, events=belowKillHeight)

    return solution



def compareWithMotion(const, fractions=(0.5, 0.9, 1.0)):
    """ Run MetSim and compare its heights and speeds to the integrated motion.

    Arguments:
        const: [Constants object] Constants of the body.

    Keyword arguments:
        fractions: [tuple] Fractions of the flight at which the simulations are compared.

    Return:
        (dh_time, dh_length, dv_time): [tuple of ndarrays] Height differences at the same time and at the
            same length (m), and speed differences at the same time (m/s), MetSim minus the motion.
    """

    _, results_list, _ = runSimulation(const)
    results = np.array(results_list, dtype=np.float64)
    time_arr, ht_arr, len_arr, vel_arr = results[:, 0], results[:, 17], results[:, 18], results[:, 19]

    solution = integrateMotion(const)
    time_motion = np.linspace(0, min(time_arr[-1], solution.t[-1]), 1000)
    x, y, vx, vy, _, len_motion = solution.sol(time_motion)
    ht_motion = np.hypot(x, y) - const.r_earth
    vel_motion = np.hypot(vx, vy)

    dh_time, dh_length, dv_time = [], [], []
    for fraction in fractions:

        t = fraction*time_motion[-1]
        dh_time.append(np.interp(t, time_arr, ht_arr) - np.interp(t, time_motion, ht_motion))
        dv_time.append(np.interp(t, time_arr, vel_arr) - np.interp(t, time_motion, vel_motion))

        l = fraction*min(len_arr[-1], len_motion[-1])
        dh_length.append(np.interp(l, len_arr, ht_arr) - np.interp(l, len_motion, ht_motion))

    return np.array(dh_time), np.array(dh_length), np.array(dv_time)



def testFireballFollowsTheEquationsOfMotion():
    """ Check a 1000 kg body at 15 km/s and 45 deg, which drops by about 200 m in 6.7 s. """

    dh_time, dh_length, dv_time = compareWithMotion(bodyConstants(1000.0, 15000.0, 45.0, 30000.0))

    print("Height differences at the same time (m):", dh_time)
    print("Height differences at the same length (m):", dh_length)
    print("Speed differences (m/s):", dv_time)

    assert np.all(np.abs(dh_time) < 2.0)
    assert np.all(np.abs(dh_length) < 2.0)
    assert np.all(np.abs(dv_time) < 10.0)



def testDeceleratingBodyFollowsTheEquationsOfMotion():
    """ Check a 100 kg body at 13 km/s and 30 deg which slows down to 2.5 km/s, where the drag slows the drop.
    """

    dh_time, dh_length, dv_time = compareWithMotion(bodyConstants(100.0, 13000.0, 30.0, 15000.0))

    print("Height differences at the same time (m):", dh_time)
    print("Height differences at the same length (m):", dh_length)
    print("Speed differences (m/s):", dv_time)

    assert np.all(np.abs(dh_time) < 2.0)
    assert np.all(np.abs(dh_length) < 2.0)
    assert np.all(np.abs(dv_time) < 15.0)



def testGrazingBodyFollowsTheEquationsOfMotion():
    """ Check a 10 kg body at 15 km/s and 80 deg, which flies for 40 s and drops by about 3 km. """

    dh_time, dh_length, dv_time = compareWithMotion(bodyConstants(10.0, 15000.0, 80.0, 30000.0))

    print("Height differences at the same time (m):", dh_time)
    print("Height differences at the same length (m):", dh_length)
    print("Speed differences (m/s):", dv_time)

    assert np.all(np.abs(dh_time) < 15.0)
    assert np.all(np.abs(dh_length) < 15.0)
    assert np.all(np.abs(dv_time) < 15.0)



def testGravityDropIndependentOfTimeStep():
    """ Check that halving the time step changes the drop at the same time by less than 1%. """

    drops = []
    for dt in [0.005, 0.0025]:

        const = bodyConstants(1000.0, 15000.0, 45.0, 30000.0)
        const.dt = dt
        frag_main, results_list, _ = runSimulation(const)

        # Normalize the drop by the square of the flight time, which differs slightly between the steps
        t_total = results_list[-1][0]
        drops.append(frag_main.h_grav_drop_total/t_total**2)

    assert abs(drops[0] - drops[1])/drops[1] < 0.01



if __name__ == "__main__":

    testFireballFollowsTheEquationsOfMotion()
    testDeceleratingBodyFollowsTheEquationsOfMotion()
    testGrazingBodyFollowsTheEquationsOfMotion()
    testGravityDropIndependentOfTimeStep()

    print("All tests passed")
