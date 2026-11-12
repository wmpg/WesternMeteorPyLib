""" Tests that the height drop due to gravity in MetSim grows with the square of the flight time.

The drop of a body falling from rest under a constant acceleration g is g*t**2/2. MetSim adds the drop to the
height of every fragment on top of the straight-line height, so after a flight of a few seconds the total drop
must be close to g*t**2/2, with g between its values at the begin and the end heights.

Run with pytest:
    python -m pytest wmpl/MetSim/Tests/test_MetSimGravityDrop.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np

from wmpl.MetSim.MetSimErosion import Constants, runSimulation, G0
from wmpl.Utils.AtmosphereDensity import fitAtmPoly
from wmpl.Utils.TrajConversions import date2JD



def _slowFireball():
    """ Make the constants of a slow, massive single-body fireball which flies for several seconds.

    Return:
        const: [Constants object] Constants of the fireball.
    """

    const = Constants()

    # Fit the atmosphere density over the whole flight
    const.h_init = 100000.0
    const.dens_co = fitAtmPoly(np.radians(45.3), np.radians(18.1), 20000, const.h_init, \
        date2JD(2020, 4, 20, 16, 15, 0))

    const.m_init = 1000.0
    const.v_init = 15000.0
    const.zenith_angle = np.radians(45.0)
    const.sigma = 0.001/1e6
    const.rho = 3500.0
    const.rho_grain = 3500.0
    const.erosion_on = False
    const.disruption_on = False
    const.fragmentation_on = False
    const.h_kill = 30000.0

    return const



def testGravityDropIsFreeFall():
    """ The total drop due to gravity is g*t**2/2 for the duration of the flight. """

    const = _slowFireball()
    frag_main, results_list, _ = runSimulation(const)

    # Duration of the flight and the height at the end
    t_total = results_list[-1][0]
    h_end = frag_main.h + frag_main.h_grav_drop_total

    # Gravity at the begin and the end heights bracket the mean gravity along the flight
    g_beg = G0/(1 + const.h_init/const.r_earth)**2
    g_end = G0/(1 + h_end/const.r_earth)**2

    drop_min = 0.5*g_beg*t_total**2
    drop_max = 0.5*g_end*t_total**2

    print("Flight time: {:.2f} s".format(t_total))
    print("Gravity drop: {:.1f} m, expected {:.1f} - {:.1f} m".format(frag_main.h_grav_drop_total, drop_min, \
        drop_max))

    assert t_total > 3.0
    assert 0.99*drop_min <= frag_main.h_grav_drop_total <= 1.01*drop_max



def testGravityDropIndependentOfTimeStep():
    """ Halving the time step does not change the total drop due to gravity at a given flight time. """

    drops = []
    for dt in [0.005, 0.0025]:

        const = _slowFireball()
        const.dt = dt
        frag_main, results_list, _ = runSimulation(const)

        # Normalize the drop by the square of the flight time, which differs slightly between the steps
        t_total = results_list[-1][0]
        drops.append(frag_main.h_grav_drop_total/t_total**2)

    assert abs(drops[0] - drops[1])/drops[1] < 0.01



if __name__ == "__main__":

    testGravityDropIsFreeFall()
    testGravityDropIndependentOfTimeStep()

    print("All tests passed")
