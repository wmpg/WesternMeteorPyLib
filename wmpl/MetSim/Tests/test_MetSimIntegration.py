""" Tests of the time integration of MetSim against an independent integration of the equations of motion.

MetSim integrates the mass, the speed and the length of every fragment over a time step together, with the
air density at the height of every stage of the Runge-Kutta method. The tests check that a small fast meteor,
which slows down quickly in a steep density gradient, follows the equations of motion at the default time
step, and that a small grain deep in the atmosphere stops instead of speeding up.

Run with pytest:
    python -m pytest wmpl/MetSim/Tests/test_MetSimIntegration.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np

from wmpl.MetSim.MetSimErosion import Constants, runSimulation
from wmpl.MetSim.Tests.test_MetSimGravityDrop import bodyConstants, compareWithMotion



def testSmallFastMeteorFollowsTheEquationsOfMotion():
    """ Check a 1 g meteor at 30 km/s at the default time step, which was off by 39 m/s and 26 m when the mass
        and the speed were integrated separately at the density of the beginning of the step.
    """

    const = bodyConstants(1e-3, 30000.0, 45.0, 30000.0)
    const.sigma = 0.01/1e6

    dh_time, dh_length, dv_time = compareWithMotion(const, fractions=(0.25, 0.5, 0.75, 0.9, 1.0))

    print("Height differences at the same time (m):", dh_time)
    print("Speed differences (m/s):", dv_time)

    assert np.all(np.abs(dh_time) < 0.5)
    assert np.all(np.abs(dh_length) < 0.5)
    assert np.all(np.abs(dv_time) < 1.0)



def testIntegrationDoesNotDependOnTheTimeStep():
    """ Check that a 1 g meteor at 30 km/s has the same speed at the same time with a 5 and a 1 ms step. """

    speeds = []
    for dt in [0.005, 0.001]:

        const = bodyConstants(1e-3, 30000.0, 45.0, 30000.0)
        const.sigma = 0.01/1e6
        const.dt = dt

        _, results_list, _ = runSimulation(const)
        results = np.array(results_list, dtype=np.float64)
        speeds.append(np.interp(np.linspace(0.5, 3.5, 7), results[:, 0], results[:, 19]))

    assert np.max(np.abs(speeds[0] - speeds[1])) < 1.0



def testSmallGrainDeepInTheAtmosphereStops():
    """ Check that a grain whose drag would stop it within one step does not speed up or reverse. """

    const = bodyConstants(5e-10, 15000.0, 45.0, 10000.0)
    const.h_init = 40000.0

    _, results_list, _ = runSimulation(const)
    results = np.array(results_list, dtype=np.float64)

    assert np.all(results[:, 19] <= const.v_init)
    assert np.all(np.isfinite(results[:, 1]))
    assert np.all(results[:, 1] >= 0)



if __name__ == "__main__":

    testSmallFastMeteorFollowsTheEquationsOfMotion()
    testIntegrationDoesNotDependOnTheTimeStep()
    testSmallGrainDeepInTheAtmosphereStops()

    print("All tests passed")
