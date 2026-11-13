""" Tests of the time integration of MetSim against an independent integration of the equations of motion.

MetSim integrates the mass, the speed and the length of every fragment over a time step together, with the
air density at the height of every stage of the Runge-Kutta method. The tests check that a small fast meteor,
which slows down quickly in a steep density gradient, follows the equations of motion at the default time
step, that a small grain whose mass and speed change within a fraction of a step is integrated in
substeps, with its light added up over them, and that a small grain deep in the atmosphere stops instead of speeding up.

Run with pytest:
    python -m pytest wmpl/MetSim/Tests/test_MetSimIntegration.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np
import scipy.integrate

from wmpl.MetSim.MetSimErosion import Constants, runSimulation
from wmpl.MetSim.MetSimErosionCyTools import ablationStepRK4, AtmDensityTableCy, luminousEfficiency
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



def testSmallGrainIsIntegratedInSubsteps():
    """ Check one 5 ms step of a 1e-11 kg grain at 20 km/s in air of a constant density of 1e-4 kg/m^3,
        which loses 90 % of its mass and 30 % of its speed within the step, against an integration of the
        equations of motion. The step gave no mass left when an intermediate stage ran out of mass.
    """

    dt, m, v, rho_atm = 0.005, 1e-11, 20000.0, 1e-4
    K = 1.21*3000**(-2/3.0)
    sigma = 0.023e-6

    # A density table which gives the same density at any height
    atm_table = AtmDensityTableCy(np.array([-1e7, 1e7]), np.log10(np.array([rho_atm, rho_atm])))

    dm, dv, dl, _, _ = ablationStepRK4(dt, K, sigma, m, v, 0.0, 100000.0, np.radians(45.0), 6371000.0, 0.0, \
        atm_table, np.zeros(7), 1.0, 0, 0.7, 23*1.66*10**(-27), 3000.0)

    def motion(t, state):
        m_t, v_t = max(state[0], 1e-30), state[1]
        return [-K*sigma*m_t**(2/3.0)*rho_atm*v_t**3, -K*m_t**(-1/3.0)*rho_atm*v_t**2, v_t]

    sol = scipy.integrate.solve_ivp(motion, (0, dt), [m, v, 0.0], method='DOP853', rtol=1e-12, \
        atol=[1e-25, 1e-9, 1e-9])
    m_end, v_end, l_end = sol.y[:, -1]

    print("Mass, speed and length after the step:", m + dm, v + dv, dl)
    print("From the equations of motion:", m_end, v_end, l_end)

    assert abs(m + dm - m_end)/m_end < 1e-3
    assert abs(v + dv - v_end) < 1.0
    assert abs(dl - l_end) < 0.01



def testLightOfSubstepsIsAddedUp():
    """ Check that the energy radiated by a grain over a step which is split into substeps is the energy
        radiated along the equations of motion, with a luminous efficiency which depends on the speed and the
        mass. Taken once from the mass and the speed in the middle of the step, 11 % of it was lost.
    """

    dt, m, v, rho_atm = 0.005, 1e-11, 20000.0, 1e-4
    K = 1.21*3000**(-2/3.0)
    sigma = 0.023e-6
    lum_eff_type = 5

    atm_table = AtmDensityTableCy(np.array([-1e7, 1e7]), np.log10(np.array([rho_atm, rho_atm])))

    _, _, _, energy, _ = ablationStepRK4(dt, K, sigma, m, v, 0.0, 100000.0, np.radians(45.0), 6371000.0, \
        0.0, atm_table, np.zeros(7), 1.0, lum_eff_type, 0.7, 23*1.66*10**(-27), 3000.0)

    def motion(t, state):
        m_t, v_t = max(state[0], 1e-30), state[1]
        dm_dt = -K*sigma*m_t**(2/3.0)*rho_atm*v_t**3
        dv_dt = -K*m_t**(-1/3.0)*rho_atm*v_t**2
        tau = luminousEfficiency(lum_eff_type, 0.7, v_t, m_t)
        return [dm_dt, dv_dt, -tau*(dm_dt*v_t**2/2 + m_t*v_t*dv_dt)]

    sol = scipy.integrate.solve_ivp(motion, (0, dt), [m, v, 0.0], method='DOP853', rtol=1e-12, \
        atol=[1e-25, 1e-9, 1e-20])
    energy_exact = sol.y[2, -1]

    print("Radiated energy over the step (J):", energy, "from the equations of motion:", energy_exact)

    assert abs(energy - energy_exact)/energy_exact < 1e-3



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
    testSmallGrainIsIntegratedInSubsteps()
    testLightOfSubstepsIsAddedUp()
    testSmallGrainDeepInTheAtmosphereStops()

    print("All tests passed")
