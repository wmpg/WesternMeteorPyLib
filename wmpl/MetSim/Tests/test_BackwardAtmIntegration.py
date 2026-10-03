""" Tests for running MetSimErosion back up a solved trajectory (wmpl.MetSim.BackwardAtmIntegration).

Run under pytest, or directly:

    python -m wmpl.MetSim.Tests.test_BackwardAtmIntegration
"""

import os
from types import SimpleNamespace

import numpy as np

from wmpl.MetSim.BackwardAtmIntegration import backwardConstants, backwardState, backwardStates
from wmpl.MetSim.MetSimErosion import Fragment, runSimulation
from wmpl.Rebound.REBOUND import sampleStateVectors
from wmpl.Utils.Pickling import loadPickle
from wmpl.Utils.TrajConversions import cartesian2Geo


# A solved trajectory shipped with the repository (2019-10-23, four stations, 67 km/s, reference point at 116 km),
#   without uncertainties
EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), \
    "Dynesty", "examples", "20191023_091225")
EXAMPLE_PICKLE = "20191023_091225_trajectory.pickle"


def _exampleStart():
    traj = loadPickle(EXAMPLE_DIR, EXAMPLE_PICKLE)

    return traj, np.concatenate([traj.state_vect_mini, traj.v_init*traj.radiant_eci_mini])


def _angleArcsec(a, b):
    return np.degrees(np.arccos(np.clip(np.dot(a, b)/np.linalg.norm(a)/np.linalg.norm(b), -1, 1)))*3600


def test_state_at_the_start_is_the_solver_state():
    """ Before any step, the ground-relative speed and direction MetSim starts with, taken back to ECI with the
        Earth's rotation, give back the solver's state vector exactly. A wrong sign of omega x r would be 681 m/s
        off here. """

    traj, state_vect = _exampleStart()
    const = backwardConstants(traj.jdt_ref, state_vect, 1e-3)
    frag = Fragment()
    frag.init(const, const.m_init, const.rho, const.v_init, const.sigma, const.gamma, const.zenith_angle, \
        const.erosion_mass_index, const.erosion_mass_min, const.erosion_mass_max)

    jd, state_vect_back = backwardState(traj.jdt_ref, state_vect, frag, 0.0)

    assert jd == traj.jdt_ref
    assert np.linalg.norm(state_vect_back[:3] - state_vect[:3]) < 1e-6
    assert np.linalg.norm(state_vect_back[3:] - state_vect[3:]) < 1e-6


def test_backward_run_ends_at_h_kill_on_the_radiant_line():
    """ Run back from 116 to 180 km, the end point is above h_kill by less than one step, at the same height in
        ECI as in MetSim to within the geoid undulation and the Earth's flattening (0.4 m here), and on the line
        back to the radiant in ECI to within the 7.5 arcsec gravity bends the path by over those 73 km. Turning
        the Earth the wrong way between the two times would put it 750 m, 2100 arcsec, off. """

    traj, state_vect = _exampleStart()
    const = backwardConstants(traj.jdt_ref, state_vect, 1e-3)
    frag, results, _ = runSimulation(const)
    t = results[-1][0]

    jd, state_vect_back = backwardState(traj.jdt_ref, state_vect, frag, t)
    _, _, ht = cartesian2Geo(jd, *state_vect_back[:3])

    assert t < 0 and jd == traj.jdt_ref + t/86400.0
    assert const.h_kill < frag.h < const.h_kill + const.v_init*abs(const.dt)
    assert abs(ht - frag.h) < 1.0
    assert _angleArcsec(state_vect_back[:3] - state_vect[:3], traj.radiant_eci_mini) < 20
    assert _angleArcsec(state_vect_back[3:], traj.radiant_eci_mini) < 20


def test_realizations_end_at_the_nominal_epoch_carrying_their_offsets():
    """ Monte Carlo realizations (50 m and 50 m/s here) are run back for as long as the nominal solution takes to
        reach h_kill, so they all end at its epoch. A realization equal to the nominal one ends exactly where it
        does, and the others keep their offsets from it: over the 1.08 s, the velocity offset changes by less than
        0.03 m/s and the position offset moves by the velocity offset times that time to within 3 cm. Without
        uncertainties there are no realizations, and the nominal solution runs back on its own. """

    traj, state_vect = _exampleStart()
    assert sampleStateVectors(traj, 20, random_seed=1) == []

    traj.uncertainties = SimpleNamespace()
    traj.state_vect_cov = np.diag([50.0**2]*6)
    realizations = sampleStateVectors(traj, 20, random_seed=1)

    jd, states = backwardStates(traj.jdt_ref, [state_vect, state_vect.copy()] + realizations, 1e-3)
    t = (jd - traj.jdt_ref)*86400.0

    assert np.array_equal(states[0], states[1])
    for start, end in zip(realizations, states[2:]):
        offset_start, offset_end = start - state_vect, end - states[0]
        assert np.linalg.norm(offset_end[3:] - offset_start[3:]) < 0.03
        assert np.linalg.norm(offset_end[:3] - (offset_start[:3] - offset_start[3:]*t)) < 0.03

    jd_nominal, states_nominal = backwardStates(traj.jdt_ref, [state_vect], 1e-3)
    assert jd_nominal == jd and len(states_nominal) == 1 and np.array_equal(states_nominal[0], states[0])


if __name__ == "__main__":
    test_state_at_the_start_is_the_solver_state()
    test_backward_run_ends_at_h_kill_on_the_radiant_line()
    test_realizations_end_at_the_nominal_epoch_carrying_their_offsets()
    print("All BackwardAtmIntegration checks passed.")
