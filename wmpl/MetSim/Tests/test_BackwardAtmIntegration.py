""" Tests for running MetSimErosion back up a solved trajectory (wmpl.MetSim.BackwardAtmIntegration).

Run under pytest, or directly:

    python -m wmpl.MetSim.Tests.test_BackwardAtmIntegration
"""

import os

import numpy as np

from wmpl.MetSim.BackwardAtmIntegration import backwardConstants, backwardState
from wmpl.MetSim.MetSimErosion import Fragment, runSimulation
from wmpl.Utils.Pickling import loadPickle
from wmpl.Utils.TrajConversions import cartesian2Geo


# A solved trajectory shipped with the repository (2019-10-23, four stations, 67 km/s, reference point at 116 km)
EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), \
    "Dynesty", "examples", "20191023_091225")
EXAMPLE_PICKLE = "20191023_091225_trajectory.pickle"


def _angleArcsec(a, b):
    return np.degrees(np.arccos(np.clip(np.dot(a, b)/np.linalg.norm(a)/np.linalg.norm(b), -1, 1)))*3600


def test_state_at_the_start_is_the_solver_state():
    """ Before any step, the ground-relative speed and direction MetSim starts with, taken back to ECI with the
        Earth's rotation, give back the solver's state vector exactly. A wrong sign of omega x r would be 681 m/s
        off here. """

    traj = loadPickle(EXAMPLE_DIR, EXAMPLE_PICKLE)
    const = backwardConstants(traj, 1e-3)
    frag = Fragment()
    frag.init(const, const.m_init, const.rho, const.v_init, const.sigma, const.gamma, const.zenith_angle, \
        const.erosion_mass_index, const.erosion_mass_min, const.erosion_mass_max)

    jd, pos, vel = backwardState(traj, frag, 0.0)

    assert jd == traj.jdt_ref
    assert np.linalg.norm(pos - traj.state_vect_mini) < 1e-6
    assert np.linalg.norm(vel + traj.v_init*traj.radiant_eci_mini) < 1e-6


def test_backward_run_ends_at_h_kill_on_the_radiant_line():
    """ Run back from 116 to 180 km, the end point is above h_kill by less than one step, at the same height in
        ECI as in MetSim to within the geoid undulation and the Earth's flattening (0.4 m here), and on the line
        back to the radiant in ECI to within the 7.5 arcsec gravity bends the path by over those 73 km. Turning
        the Earth the wrong way between the two times would put it 750 m, 2100 arcsec, off. """

    traj = loadPickle(EXAMPLE_DIR, EXAMPLE_PICKLE)
    const = backwardConstants(traj, 1e-3)
    frag, results, _ = runSimulation(const)
    t = results[-1][0]

    jd, pos, vel = backwardState(traj, frag, t)
    _, _, ht = cartesian2Geo(jd, *pos)

    assert t < 0 and jd == traj.jdt_ref + t/86400.0
    assert const.h_kill < frag.h < const.h_kill + const.v_init*abs(const.dt)
    assert abs(ht - frag.h) < 1.0
    assert _angleArcsec(pos - traj.state_vect_mini, traj.radiant_eci_mini) < 20
    assert _angleArcsec(vel, -traj.radiant_eci_mini) < 20


if __name__ == "__main__":
    test_state_at_the_start_is_the_solver_state()
    test_backward_run_ends_at_h_kill_on_the_radiant_line()
    print("All BackwardAtmIntegration checks passed.")
